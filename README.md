# X/Twitter AI Agent Trainer

A Python starter project for building an AI agent from your own X (Twitter) data using the official X API.

> **Important:** this project does not scrape X. Use the official API and only train on content you own or are licensed to use. Follow X's Developer Agreement and privacy requirements.

## Features

- Collect recent posts from X API v2 search or user timelines.
- Normalize posts into JSONL training examples.
- Train a lightweight text-generation model with Hugging Face Transformers.
- Run an interactive local agent that responds using the trained model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your X API bearer token to `.env`, then collect and prepare data:

```bash
python -m x_agent.collect --query "from:your_username -is:retweet" --output data/raw_tweets.jsonl --max-results 100
python -m x_agent.prepare --input data/raw_tweets.jsonl --output data/train.jsonl
```

Train a local model:

```bash
python -m x_agent.train --dataset data/train.jsonl --output-dir models/x-agent --model-name distilgpt2 --epochs 1
```

Chat with the trained agent:

```bash
python -m x_agent.chat --model-dir models/x-agent
```

## Project layout

```text
x_agent/collect.py   # X API collection
x_agent/prepare.py   # JSONL preparation
x_agent/train.py     # model fine-tuning
x_agent/chat.py      # interactive generation
```

## Data format

Raw collection writes one JSON object per line with at least `id`, `text`, and `created_at`. The training preparation step writes JSONL records with a `text` field.
