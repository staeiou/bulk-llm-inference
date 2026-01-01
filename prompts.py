PROMPT_TEMPLATE = """You are a sentiment analysis system.

Task:
- Read the INPUT TEXT.
- Output ONLY a JSON object with exactly these keys:
  - sentiment_score: a number in [-1, 1] where -1 is very negative and +1 is very positive
  - sentiment_confidence: a number in [0, 1]
- Output format example: {"sentiment_score": number, "sentiment_confidence": number\}

INPUT TEXT:
{text}
"""

