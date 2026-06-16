# X/Twitter AI Agent Trainer

A Python starter project for building an AI agent from your own X (Twitter) data using the official X API.

> **Important:** this project does not scrape X. Use the official API and only train on content you own or are licensed to use. Follow X's Developer Agreement and privacy requirements.

## Features

- Collect recent posts from X API v2 search or user timelines.
- Normalize posts into JSONL training examples.
- Train a lightweight text-generation model with Hugging Face Transformers.
- Run an interactive local agent that responds using the trained model.
- Update retrieval memory from X on a one-second local polling loop, subject to X API rate limits.
- Persist long-term memory in SQLite, including a pinned first user memory that the agent keeps recalling.

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

Continuously update memory from X as fast as once per second:

```bash
python -m x_agent.realtime --query "from:your_username -is:retweet" --interval 1
```

Chat with the trained, memory-enabled agent. The first thing you tell the agent is pinned as long-term memory:

```bash
python -m x_agent.chat --model-dir models/x-agent --memory data/agent_memory.sqlite
```

## Project layout

```text
x_agent/collect.py   # X API collection
x_agent/prepare.py   # JSONL preparation
x_agent/train.py     # model fine-tuning
x_agent/chat.py      # interactive generation with memory
x_agent/memory.py    # SQLite long-term memory and search
x_agent/realtime.py  # one-second X-to-memory updater
```

## Data format

Raw collection writes one JSON object per line with at least `id`, `text`, and `created_at`. The training preparation step writes JSONL records with a `text` field.


## Real-time training vs. memory updates

Training a neural model on all of Twitter in one second is not realistic or compliant with platform limits. This project instead supports a production-style pattern: fine-tune periodically on licensed datasets, then update SQLite retrieval memory every second with fresh X API data so the agent can react quickly without retraining the entire model.
