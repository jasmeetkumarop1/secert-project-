"""Chat with a locally trained X-style text-generation agent with memory."""

from __future__ import annotations

import argparse

from x_agent.memory import AgentMemory


def build_prompt(user_prompt: str, memories: list[str]) -> str:
    """Build a generation prompt that includes remembered context."""
    memory_block = "\n".join(f"- {memory}" for memory in memories) or "- No remembered context yet."
    return f"""Relevant remembered context:
{memory_block}

User prompt: {user_prompt}
Agent post:"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive trained X agent with persistent memory")
    parser.add_argument("--model-dir", default="models/x-agent", help="Directory with trained model")
    parser.add_argument("--memory", default="data/agent_memory.sqlite", help="SQLite memory path")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Generated token limit")
    args = parser.parse_args()

    from transformers import pipeline

    memory = AgentMemory(args.memory)
    generator = pipeline("text-generation", model=args.model_dir, tokenizer=args.model_dir)
    print("X agent ready. Type 'exit' to quit. First message is pinned as long-term memory.")
    try:
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in {"exit", "quit"}:
                break
            memory.remember_first(prompt)
            memory.remember(prompt, source="conversation")
            remembered = [item.text for item in memory.search(prompt, limit=6)]
            formatted_prompt = build_prompt(prompt, remembered)
            result = generator(
                formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=generator.tokenizer.eos_token_id,
            )[0]["generated_text"]
            answer = result[len(formatted_prompt) :].strip()
            memory.remember(answer, source="agent_reply")
            print(answer)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
