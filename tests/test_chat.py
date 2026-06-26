"""Unit tests for x_agent.chat module."""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock, call

import pytest


@pytest.fixture(autouse=True)
def mock_transformers():
    """Mock transformers module to avoid importing heavy ML dependencies."""
    mock_pipeline = MagicMock()
    mock_transformers_mod = MagicMock()
    mock_transformers_mod.pipeline = mock_pipeline

    with patch.dict(sys.modules, {"transformers": mock_transformers_mod}):
        # Remove cached x_agent.chat so it reimports with mocked transformers
        sys.modules.pop("x_agent.chat", None)
        yield mock_pipeline


class TestChatMain:
    """Tests for the chat module's main() function."""

    def test_main_initializes_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, mock_transformers: MagicMock
    ) -> None:
        """Should initialize text-generation pipeline with correct model dir."""
        monkeypatch.setattr(
            "sys.argv",
            ["chat", "--model-dir", "models/test-agent", "--max-new-tokens", "50"],
        )

        mock_generator = MagicMock()
        mock_generator.tokenizer.eos_token_id = 50256
        mock_transformers.return_value = mock_generator

        with patch("builtins.input", return_value="exit"):
            from x_agent.chat import main

            main()

        mock_transformers.assert_called_once_with(
            "text-generation", model="models/test-agent", tokenizer="models/test-agent"
        )

    def test_main_exits_on_quit(
        self, monkeypatch: pytest.MonkeyPatch, mock_transformers: MagicMock
    ) -> None:
        """Should exit loop when user types 'quit'."""
        monkeypatch.setattr(
            "sys.argv",
            ["chat", "--model-dir", "models/test-agent"],
        )

        mock_generator = MagicMock()
        mock_generator.tokenizer.eos_token_id = 50256
        mock_transformers.return_value = mock_generator

        with patch("builtins.input", return_value="quit"):
            from x_agent.chat import main

            main()

        # Generator should not be called since user exits immediately
        mock_generator.assert_not_called()

    def test_main_generates_response(
        self, monkeypatch: pytest.MonkeyPatch, mock_transformers: MagicMock
    ) -> None:
        """Should generate response for user input and exit on second input."""
        monkeypatch.setattr(
            "sys.argv",
            ["chat", "--model-dir", "models/test-agent", "--max-new-tokens", "80"],
        )

        mock_generator = MagicMock()
        mock_generator.tokenizer.eos_token_id = 50256

        prompt_text = "Hello agent"
        formatted_prompt = f"User prompt: {prompt_text}\nAgent post:"
        generated_text = f"{formatted_prompt} This is my response"

        mock_generator.return_value = [{"generated_text": generated_text}]
        mock_transformers.return_value = mock_generator

        inputs = iter(["Hello agent", "exit"])
        with patch("builtins.input", side_effect=inputs):
            from x_agent.chat import main

            main()

        mock_generator.assert_called_once_with(
            formatted_prompt,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=50256,
        )

    def test_main_handles_exit_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch, mock_transformers: MagicMock
    ) -> None:
        """Should handle EXIT, Exit, etc. case-insensitively."""
        monkeypatch.setattr(
            "sys.argv",
            ["chat", "--model-dir", "models/test-agent"],
        )

        mock_generator = MagicMock()
        mock_generator.tokenizer.eos_token_id = 50256
        mock_transformers.return_value = mock_generator

        with patch("builtins.input", return_value="EXIT"):
            from x_agent.chat import main

            main()

        mock_generator.assert_not_called()

    def test_main_strips_whitespace_from_input(
        self, monkeypatch: pytest.MonkeyPatch, mock_transformers: MagicMock
    ) -> None:
        """Should strip whitespace from user input."""
        monkeypatch.setattr(
            "sys.argv",
            ["chat", "--model-dir", "models/test-agent", "--max-new-tokens", "80"],
        )

        mock_generator = MagicMock()
        mock_generator.tokenizer.eos_token_id = 50256

        prompt_text = "Hello"
        formatted_prompt = f"User prompt: {prompt_text}\nAgent post:"
        generated_text = f"{formatted_prompt} Response"

        mock_generator.return_value = [{"generated_text": generated_text}]
        mock_transformers.return_value = mock_generator

        inputs = iter(["  Hello  ", "exit"])
        with patch("builtins.input", side_effect=inputs):
            from x_agent.chat import main

            main()

        # Should use stripped "Hello" not "  Hello  "
        mock_generator.assert_called_once()
        call_args = mock_generator.call_args[0]
        assert call_args[0] == formatted_prompt

    def test_default_max_new_tokens(
        self, monkeypatch: pytest.MonkeyPatch, mock_transformers: MagicMock
    ) -> None:
        """Should default to 80 max_new_tokens."""
        monkeypatch.setattr(
            "sys.argv",
            ["chat", "--model-dir", "models/test-agent"],
        )

        mock_generator = MagicMock()
        mock_generator.tokenizer.eos_token_id = 50256

        formatted_prompt = "User prompt: test\nAgent post:"
        generated_text = f"{formatted_prompt} output"
        mock_generator.return_value = [{"generated_text": generated_text}]
        mock_transformers.return_value = mock_generator

        inputs = iter(["test", "exit"])
        with patch("builtins.input", side_effect=inputs):
            from x_agent.chat import main

            main()

        call_kwargs = mock_generator.call_args[1]
        assert call_kwargs["max_new_tokens"] == 80
