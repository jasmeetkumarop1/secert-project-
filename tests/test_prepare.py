import json
import tempfile
import unittest
from pathlib import Path

from x_agent.prepare import clean_text, prepare_dataset


class CleanTextTests(unittest.TestCase):
    def test_clean_text_collapses_whitespace_and_newlines(self) -> None:
        self.assertEqual(
            clean_text("  Hello\n\n   X agent\t world  "),
            "Hello X agent world",
        )

    def test_clean_text_preserves_unicode_and_punctuation(self) -> None:
        self.assertEqual(
            clean_text("Building with AI 🚀 — safely!"),
            "Building with AI 🚀 — safely!",
        )


class PrepareDatasetTests(unittest.TestCase):
    def test_prepare_dataset_writes_wrapped_training_records_and_skips_short_posts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "raw.jsonl"
            output_path = Path(tmpdir) / "nested" / "train.jsonl"
            records = [
                {"id": "1", "text": "This is long enough for training."},
                {"id": "2", "text": "too short"},
                {"id": "3", "text": "Line one\nline two with extra   spaces"},
            ]
            input_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n\n",
                encoding="utf-8",
            )

            count = prepare_dataset(input_path, output_path, min_chars=20)

            self.assertEqual(count, 2)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [json.loads(line) for line in lines],
                [
                    {"text": "<x_post>This is long enough for training.</x_post>"},
                    {"text": "<x_post>Line one line two with extra spaces</x_post>"},
                ],
            )

    def test_prepare_dataset_handles_missing_text_as_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "raw.jsonl"
            output_path = Path(tmpdir) / "train.jsonl"
            input_path.write_text(
                json.dumps({"id": "1"}) + "\n" + json.dumps({"id": "2", "text": "A valid post body"}) + "\n",
                encoding="utf-8",
            )

            count = prepare_dataset(input_path, output_path, min_chars=1)

            self.assertEqual(count, 1)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {"text": "<x_post>A valid post body</x_post>"},
            )


if __name__ == "__main__":
    unittest.main()
