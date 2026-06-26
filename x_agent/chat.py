"""Chat with a locally trained X-style text-generation agent."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive trained X agent")
    parser.add_argument("--model-dir", default="models/x-agent", help="Directory with trained model")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Generated token limit")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"Model directory not found: {model_dir}\n"
            "Train a model first with: python -m x_agent.train"
        )

    try:
        generator = pipeline("text-generation", model=args.model_dir, tokenizer=args.model_dir)
    except Exception as exc:
        raise SystemExit(f"Failed to load model from {args.model_dir}: {exc}") from exc

    print("X agent ready. Type 'exit' to quit.")
    try:
        while True:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                break

            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue

            formatted_prompt = f"User prompt: {prompt}\nAgent post:"
            try:
                results = generator(
                    formatted_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=generator.tokenizer.eos_token_id,
                )
            except Exception as exc:
                logger.error("Generation failed: %s", exc)
                print("Error: could not generate a response. Try a different prompt.")
                continue

            if not results or "generated_text" not in results[0]:
                print("Error: model returned an unexpected response format.")
                continue

            print(results[0]["generated_text"][len(formatted_prompt):].strip())
    except KeyboardInterrupt:
        pass
    finally:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
