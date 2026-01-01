#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from prompts import PROMPT_TEMPLATE


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)

    return pd.read_table(p)


def _write_df(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df.to_csv(p, index=False)
        return
    if suffix in (".tsv", ".tab"):
        df.to_csv(p, index=False, sep="\t")
        return
    if suffix in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(p, index=False)
        return
    if suffix in (".jsonl", ".ndjson"):
        df.to_json(p, orient="records", lines=True)
        return
    if suffix == ".json":
        df.to_json(p, orient="records")
        return

    df.to_parquet(p, index=False)


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("response text is not a string")

    m = _CODE_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def _to_json_str(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


async def _infer_rows_async(
    texts: list[str],
    row_indices: list[int],
    model: str,
    base_url: str,
    api_key: str,
    batch_size: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    retry_backoff_secs: float,
    retry_jitter_secs: float,
    mock: bool,
    output_path: str,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrent)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def one(input_text: str, row_idx: int) -> dict[str, Any]:
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.replace("{text}", input_text)}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if mock:
            # Deterministic toy output for offline testing.
            lowered = input_text.lower()
            score = 0.0
            if any(w in lowered for w in ["love", "perfect", "great", "good", "amazing"]):
                score = 0.7
            if any(w in lowered for w in ["hate", "broke", "terrible", "awful"]):
                score = -0.8
            resp_text = _to_json_str({"sentiment_score": score, "sentiment_confidence": 0.75})
            parsed = _extract_json_object(resp_text)
            return {
                "input_row_index": row_idx,
                "input_text": input_text,
                "request_payload": _to_json_str(request_payload),
                "response_payload": _to_json_str({"mock": True}),
                "main_response": resp_text,
                "reasoning_response": "",
                "parsed_response": _to_json_str(parsed),
                "sentiment_score": float(parsed.get("sentiment_score")),
                "sentiment_confidence": float(parsed.get("sentiment_confidence")),
                "parse_error": "",
                "attempts_made": 1,
            }

        last_err: Exception | None = None
        last_response_payload: str = ""
        last_main_response: str = ""
        last_reasoning_response: str = ""
        last_attempt: int = 0

        for attempt in range(retries + 1):
            last_attempt = attempt + 1  # Track actual attempt number (1-indexed)
            async with semaphore:
                try:
                    resp = await client.chat.completions.create(**request_payload)
                    msg = resp.choices[0].message
                    main = (msg.content or "") if msg else ""
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                    resp_payload = _to_json_str(resp.model_dump() if hasattr(resp, "model_dump") else resp)

                    # Save response data BEFORE parsing, so we have it even if parsing fails
                    last_response_payload = resp_payload
                    last_main_response = main
                    last_reasoning_response = reasoning

                    # Check for truncation first
                    finish_reason = resp.choices[0].finish_reason if resp.choices else None
                    if finish_reason == "length":
                        raise ValueError(f"Response truncated (finish_reason=length)")

                    # Now try to parse - if this fails, we still have the response saved above
                    parsed = _extract_json_object(main)

                    score = parsed.get("sentiment_score")
                    conf = parsed.get("sentiment_confidence")

                    return {
                        "input_row_index": row_idx,
                        "input_text": input_text,
                        "request_payload": _to_json_str(request_payload),
                        "response_payload": resp_payload,
                        "main_response": main,
                        "reasoning_response": reasoning,
                        "parsed_response": _to_json_str(parsed),
                        "sentiment_score": float(score) if score is not None else float("nan"),
                        "sentiment_confidence": float(conf) if conf is not None else float("nan"),
                        "parse_error": "",
                        "attempts_made": last_attempt,
                    }
                except Exception as e:
                    last_err = e

            if attempt < retries:
                delay = (retry_backoff_secs * (2**attempt)) + (random.random() * retry_jitter_secs)
                await asyncio.sleep(delay)

        return {
            "input_row_index": row_idx,
            "input_text": input_text,
            "request_payload": _to_json_str(request_payload),
            "response_payload": last_response_payload,
            "main_response": last_main_response,
            "reasoning_response": last_reasoning_response,
            "parsed_response": "",
            "sentiment_score": float("nan"),
            "sentiment_confidence": float("nan"),
            "parse_error": str(last_err) if last_err else "unknown error",
            "attempts_made": last_attempt,
        }

    results: list[dict[str, Any]] = []
    with tqdm(total=len(texts), desc="Inferring", unit="row") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_indices = row_indices[i : i + batch_size]
            batch_results = await asyncio.gather(*[one(t, idx) for t, idx in zip(batch_texts, batch_indices)])
            results.extend(batch_results)

            # Flush to disk after each batch
            if output_path:
                existing_df = _load_df(output_path) if Path(output_path).exists() else pd.DataFrame()
                new_df = pd.DataFrame(batch_results)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
                _write_df(combined_df, output_path)

            pbar.update(len(batch_texts))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk LLM inference over a dataframe with a 'text' column.")
    parser.add_argument("--input", default=os.getenv("INPUT_FILE", "data/sentiment_demo.csv"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_FILE", "outputs/sentiment_scored.parquet"))
    parser.add_argument("--text-col", default=os.getenv("TEXT_COL", "text"))
    parser.add_argument("--model", default=os.getenv("MODEL", os.getenv("OPENAI_MODEL", "")))
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")))
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", 64))
    parser.add_argument("--max-concurrent", type=int, default=_env_int("MAX_CONCURRENT", 64))
    parser.add_argument("--max-tokens", type=int, default=_env_int("MAX_TOKENS", 64))
    parser.add_argument("--temperature", type=float, default=_env_float("TEMPERATURE", 0.0))
    parser.add_argument("--top-p", type=float, default=_env_float("TOP_P", 1.0))
    parser.add_argument("--retries", type=int, default=_env_int("RETRIES", 2))
    parser.add_argument("--retry-backoff-secs", type=float, default=_env_float("RETRY_BACKOFF_SECS", 0.5))
    parser.add_argument("--retry-jitter-secs", type=float, default=_env_float("RETRY_JITTER_SECS", 0.2))
    parser.add_argument("--mock", action="store_true", default=os.getenv("MOCK", "0").lower() in ("1", "true", "yes"))
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume (start from scratch)")
    args = parser.parse_args()

    if not args.model and not args.mock:
        raise SystemExit("Set --model (or MODEL/OPENAI_MODEL) or use --mock.")

    df = _load_df(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Input file must contain a '{args.text_col}' column.")

    # Auto-resume: check if output exists and load processed indices
    processed_indices: set[int] = set()
    if not args.no_resume and Path(args.output).exists():
        existing_df = _load_df(args.output)
        if "input_row_index" in existing_df.columns:
            processed_indices = set(existing_df["input_row_index"].dropna().astype(int).tolist())
            print(f"Resuming: found {len(processed_indices)} already-processed rows, skipping them.")

    # Build list of (index, text) pairs to process
    all_indices = list(range(len(df)))
    indices_to_process = [i for i in all_indices if i not in processed_indices]
    texts = [str(df[args.text_col].iloc[i]) for i in indices_to_process]

    if not texts:
        print(f"All rows already processed. Output: {args.output}")
        return

    results = asyncio.run(
        _infer_rows_async(
            texts=texts,
            row_indices=indices_to_process,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            retries=args.retries,
            retry_backoff_secs=args.retry_backoff_secs,
            retry_jitter_secs=args.retry_jitter_secs,
            mock=args.mock,
            output_path=args.output,
        )
    )

    print(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()

