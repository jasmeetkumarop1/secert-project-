"""Chat with a locally trained X-style text-generation agent."""

from __future__ import annotations

import argparse

from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive trained X agent")
    parser.add_argument("--model-dir", default="models/x-agent", help="Directory with trained model")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Generated token limit")
    args = parser.parse_args()

    generator = pipeline("text-generation", model=args.model_dir, tokenizer=args.model_dir)
    print("X agent ready. Type 'exit' to quit.")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        formatted_prompt = f"User prompt: {prompt}\nAgent post:"
        result = generator(
            formatted_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"]
        print(result[len(formatted_prompt) :].strip())


if __name__ == "__main__":
    main()
