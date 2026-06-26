"""Unit tests for x_agent.prepare module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from x_agent.prepare import clean_text, prepare_dataset


class TestCleanText:
    """Tests for clean_text function."""

    def test_removes_newlines(self) -> None:
        """Should replace newlines with spaces."""
        assert clean_text("hello\nworld") == "hello world"

    def test_collapses_multiple_spaces(self) -> None:
        """Should collapse multiple spaces into one."""
        assert clean_text("hello    world") == "hello world"

    def test_handles_mixed_whitespace(self) -> None:
        """Should normalize mixed whitespace (newlines + spaces)."""
        assert clean_text("hello\n  world   foo\nbar") == "hello world foo bar"

    def test_strips_leading_trailing_spaces(self) -> None:
        """Should strip leading and trailing whitespace."""
        assert clean_text("  hello world  ") == "hello world"

    def test_empty_string(self) -> None:
        """Should return empty string for empty input."""
        assert clean_text("") == ""

    def test_single_word(self) -> None:
        """Should return single word unchanged."""
        assert clean_text("hello") == "hello"

    def test_preserves_special_characters(self) -> None:
        """Should preserve special characters like emoji and punctuation."""
        assert clean_text("Hello! 🌍 @user #tag") == "Hello! 🌍 @user #tag"

    def test_tabs_and_carriage_returns(self) -> None:
        """Should collapse tabs and other whitespace."""
        assert clean_text("hello\t\tworld") == "hello world"


class TestPrepareDataset:
    """Tests for prepare_dataset function."""

    def test_filters_short_posts(self, raw_jsonl_file: Path, tmp_dir: Path) -> None:
        """Should filter out posts shorter than min_chars."""
        output = tmp_dir / "train.jsonl"
        count = prepare_dataset(raw_jsonl_file, output, min_chars=20)

        # "Short" (5 chars) should be filtered out, 2 remain
        assert count == 2

    def test_output_format(self, raw_jsonl_file: Path, tmp_dir: Path) -> None:
        """Should wrap text in <x_post> tags."""
        output = tmp_dir / "train.jsonl"
        prepare_dataset(raw_jsonl_file, output, min_chars=20)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            record = json.loads(line)
            assert "text" in record
            assert record["text"].startswith("<x_post>")
            assert record["text"].endswith("</x_post>")

    def test_cleans_text_in_output(self, raw_jsonl_file: Path, tmp_dir: Path) -> None:
        """Should apply clean_text to the output."""
        output = tmp_dir / "train.jsonl"
        prepare_dataset(raw_jsonl_file, output, min_chars=20)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        # The third post has newlines and extra spaces
        last_record = json.loads(lines[1])
        text_content = last_record["text"].replace("<x_post>", "").replace("</x_post>", "")
        assert "\n" not in text_content
        assert "   " not in text_content

    def test_creates_output_directories(self, raw_jsonl_file: Path, tmp_dir: Path) -> None:
        """Should create parent directories for output path."""
        output = tmp_dir / "deep" / "nested" / "train.jsonl"
        prepare_dataset(raw_jsonl_file, output, min_chars=20)

        assert output.exists()

    def test_skips_blank_lines(self, tmp_dir: Path) -> None:
        """Should skip blank lines in the input file."""
        input_path = tmp_dir / "input.jsonl"
        input_path.write_text(
            '{"text": "This is a valid tweet for testing purposes"}\n'
            "\n"
            '{"text": "Another valid tweet with enough characters"}\n'
            "   \n",
            encoding="utf-8",
        )
        output = tmp_dir / "train.jsonl"
        count = prepare_dataset(input_path, output, min_chars=20)

        assert count == 2

    def test_min_chars_boundary(self, tmp_dir: Path) -> None:
        """Should include posts exactly at min_chars length."""
        input_path = tmp_dir / "input.jsonl"
        input_path.write_text(
            '{"text": "12345678901234567890"}\n'  # exactly 20 chars
            '{"text": "1234567890123456789"}\n',  # 19 chars, should be excluded
            encoding="utf-8",
        )
        output = tmp_dir / "train.jsonl"
        count = prepare_dataset(input_path, output, min_chars=20)

        assert count == 1

    def test_missing_text_field(self, tmp_dir: Path) -> None:
        """Should handle records without a text field (treated as empty)."""
        input_path = tmp_dir / "input.jsonl"
        input_path.write_text(
            '{"id": "1"}\n'
            '{"text": "This is a valid tweet for testing purposes"}\n',
            encoding="utf-8",
        )
        output = tmp_dir / "train.jsonl"
        count = prepare_dataset(input_path, output, min_chars=20)

        assert count == 1

    def test_custom_min_chars(self, tmp_dir: Path) -> None:
        """Should respect custom min_chars parameter."""
        input_path = tmp_dir / "input.jsonl"
        input_path.write_text(
            '{"text": "Short"}\n'
            '{"text": "Medium length text"}\n'
            '{"text": "This is a longer text that exceeds fifty characters easily"}\n',
            encoding="utf-8",
        )
        output = tmp_dir / "train.jsonl"
        count = prepare_dataset(input_path, output, min_chars=50)

        assert count == 1

    def test_returns_zero_when_all_filtered(self, tmp_dir: Path) -> None:
        """Should return 0 when all posts are below min_chars."""
        input_path = tmp_dir / "input.jsonl"
        input_path.write_text(
            '{"text": "tiny"}\n'
            '{"text": "also small"}\n',
            encoding="utf-8",
        )
        output = tmp_dir / "train.jsonl"
        count = prepare_dataset(input_path, output, min_chars=20)

        assert count == 0


class TestPrepareMain:
    """Tests for the prepare module's main() function."""

    def test_main_runs_successfully(
        self, raw_jsonl_file: Path, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should run main() and produce output."""
        output_path = str(tmp_dir / "train.jsonl")
        monkeypatch.setattr(
            "sys.argv",
            ["prepare", "--input", str(raw_jsonl_file), "--output", output_path],
        )

        from x_agent.prepare import main

        main()
        assert Path(output_path).exists()
