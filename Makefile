.DEFAULT_GOAL := help
.PHONY: help demo demo-mock infer

PYTHON ?= python

# OpenAI-compatible config (vLLM/LiteLLM/OpenAI/OpenRouter/etc.)
MODEL ?= unsloth/llama-3.2-1b-instruct
BASE_URL ?= http://localhost:8000/v1
API_KEY ?= $(OPENAI_API_KEY)

INPUT_FILE ?= data/bbc_news_stories.tsv
OUTPUT_FILE ?= outputs/bbc_news_stories_sentiment_scored.tsv
TEXT_COL ?= text

BATCH_SIZE ?= 16
MAX_CONCURRENT ?= 16
MAX_TOKENS ?= 512
TEMPERATURE ?= 0.0
TOP_P ?= 1.0
RETRIES ?= 3

help:
	@echo "bulk-llm-inference"
	@echo ""
	@echo "Targets:"
	@echo "  make demo        - Run sentiment demo (real model)"
	@echo "  make demo-mock   - Run sentiment demo (no network; deterministic)"
	@echo "  make infer       - Run bulk inference (configurable)"
	@echo ""
	@echo "Common vars:"
	@echo "  MODEL=$(MODEL)"
	@echo "  BASE_URL=$(BASE_URL)"
	@echo "  API_KEY=$(API_KEY)"
	@echo "  INPUT_FILE=$(INPUT_FILE)"
	@echo "  OUTPUT_FILE=$(OUTPUT_FILE)"

demo:
	MODEL=$(MODEL) BASE_URL=$(BASE_URL) API_KEY=$(API_KEY) \
	INPUT_FILE=$(INPUT_FILE) OUTPUT_FILE=$(OUTPUT_FILE) TEXT_COL=$(TEXT_COL) \
	BATCH_SIZE=$(BATCH_SIZE) MAX_CONCURRENT=$(MAX_CONCURRENT) MAX_TOKENS=$(MAX_TOKENS) \
	TEMPERATURE=$(TEMPERATURE) TOP_P=$(TOP_P) RETRIES=$(RETRIES) \
	$(PYTHON) bulk_infer.py

demo-mock:
	MOCK=1 INPUT_FILE=$(INPUT_FILE) OUTPUT_FILE=$(OUTPUT_FILE) TEXT_COL=$(TEXT_COL) \
	BATCH_SIZE=$(BATCH_SIZE) MAX_CONCURRENT=$(MAX_CONCURRENT) \
	$(PYTHON) bulk_infer.py

infer: demo

