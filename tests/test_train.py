"""Unit tests for x_agent.train module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_ml_modules():
    """Mock transformers and datasets modules to avoid heavy ML dependencies."""
    mock_datasets = MagicMock()
    mock_transformers = MagicMock()

    modules_to_mock = {
        "datasets": mock_datasets,
        "transformers": mock_transformers,
    }

    with patch.dict(sys.modules, modules_to_mock):
        # Remove cached x_agent.train so it reimports with mocked modules
        sys.modules.pop("x_agent.train", None)
        yield {
            "datasets": mock_datasets,
            "transformers": mock_transformers,
        }


class TestTrainFunction:
    """Tests for the train() function."""

    def test_train_calls_expected_components(
        self, tmp_dir: Path, mock_ml_modules: dict
    ) -> None:
        """Should load dataset, tokenizer, model, and run training."""
        dataset_path = tmp_dir / "train.jsonl"
        dataset_path.write_text(
            '{"text": "<x_post>Hello world this is a test</x_post>"}\n',
            encoding="utf-8",
        )
        output_dir = str(tmp_dir / "model_output")

        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"

        mock_model = MagicMock()
        mock_trainer_instance = MagicMock()

        mock_ml_modules["datasets"].load_dataset.return_value = mock_dataset
        mock_ml_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ml_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_ml_modules["transformers"].Trainer.return_value = mock_trainer_instance

        from x_agent.train import train

        train(
            dataset_path=str(dataset_path),
            output_dir=output_dir,
            model_name="distilgpt2",
            epochs=1.0,
            block_size=128,
        )

        mock_ml_modules["datasets"].load_dataset.assert_called_once_with(
            "json", data_files=str(dataset_path), split="train"
        )
        mock_dataset.map.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once_with(output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)

    def test_train_sets_pad_token_when_missing(
        self, tmp_dir: Path, mock_ml_modules: dict
    ) -> None:
        """Should set pad_token to eos_token when pad_token is None."""
        dataset_path = tmp_dir / "train.jsonl"
        dataset_path.write_text('{"text": "test"}\n', encoding="utf-8")
        output_dir = str(tmp_dir / "model_output")

        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"

        mock_model = MagicMock()
        mock_trainer_instance = MagicMock()

        mock_ml_modules["datasets"].load_dataset.return_value = mock_dataset
        mock_ml_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ml_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_ml_modules["transformers"].Trainer.return_value = mock_trainer_instance

        from x_agent.train import train

        train(str(dataset_path), output_dir, "distilgpt2", 1.0, 128)

        assert mock_tokenizer.pad_token == "<|endoftext|>"

    def test_train_preserves_existing_pad_token(
        self, tmp_dir: Path, mock_ml_modules: dict
    ) -> None:
        """Should not overwrite pad_token when it already exists."""
        dataset_path = tmp_dir / "train.jsonl"
        dataset_path.write_text('{"text": "test"}\n', encoding="utf-8")
        output_dir = str(tmp_dir / "model_output")

        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<|endoftext|>"

        mock_model = MagicMock()
        mock_trainer_instance = MagicMock()

        mock_ml_modules["datasets"].load_dataset.return_value = mock_dataset
        mock_ml_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ml_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_ml_modules["transformers"].Trainer.return_value = mock_trainer_instance

        from x_agent.train import train

        train(str(dataset_path), output_dir, "distilgpt2", 1.0, 128)

        assert mock_tokenizer.pad_token == "<pad>"

    def test_training_arguments_configuration(
        self, tmp_dir: Path, mock_ml_modules: dict
    ) -> None:
        """Should pass correct training arguments."""
        dataset_path = tmp_dir / "train.jsonl"
        dataset_path.write_text('{"text": "test"}\n', encoding="utf-8")
        output_dir = str(tmp_dir / "model_output")

        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        mock_model = MagicMock()
        mock_trainer_instance = MagicMock()

        mock_ml_modules["datasets"].load_dataset.return_value = mock_dataset
        mock_ml_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ml_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_ml_modules["transformers"].Trainer.return_value = mock_trainer_instance

        from x_agent.train import train

        train(str(dataset_path), output_dir, "distilgpt2", 3.0, 256)

        mock_ml_modules["transformers"].TrainingArguments.assert_called_once()
        call_kwargs = mock_ml_modules["transformers"].TrainingArguments.call_args[1]
        assert call_kwargs["output_dir"] == output_dir
        assert call_kwargs["num_train_epochs"] == 3.0
        assert call_kwargs["report_to"] == "none"

    def test_tokenize_function_called_with_batched(
        self, tmp_dir: Path, mock_ml_modules: dict
    ) -> None:
        """Should call dataset.map with batched=True and remove text column."""
        dataset_path = tmp_dir / "train.jsonl"
        dataset_path.write_text('{"text": "test"}\n', encoding="utf-8")
        output_dir = str(tmp_dir / "model_output")

        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"

        mock_model = MagicMock()
        mock_trainer_instance = MagicMock()

        mock_ml_modules["datasets"].load_dataset.return_value = mock_dataset
        mock_ml_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ml_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_ml_modules["transformers"].Trainer.return_value = mock_trainer_instance

        from x_agent.train import train

        train(str(dataset_path), output_dir, "distilgpt2", 1.0, 128)

        map_kwargs = mock_dataset.map.call_args[1]
        assert map_kwargs["batched"] is True
        assert map_kwargs["remove_columns"] == ["text"]


class TestTrainMain:
    """Tests for the train module's main() function."""

    def test_main_parses_arguments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_dir: Path, mock_ml_modules: dict
    ) -> None:
        """Should parse CLI arguments and call train."""
        dataset_path = str(tmp_dir / "train.jsonl")
        Path(dataset_path).write_text('{"text": "test"}\n', encoding="utf-8")
        output_dir = str(tmp_dir / "model")

        monkeypatch.setattr(
            "sys.argv",
            [
                "train",
                "--dataset", dataset_path,
                "--output-dir", output_dir,
                "--model-name", "distilgpt2",
                "--epochs", "2",
                "--block-size", "64",
            ],
        )

        # Set up all the mocks for a successful train call
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.map.return_value = mock_dataset
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_model = MagicMock()
        mock_trainer_instance = MagicMock()

        mock_ml_modules["datasets"].load_dataset.return_value = mock_dataset
        mock_ml_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ml_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_ml_modules["transformers"].Trainer.return_value = mock_trainer_instance

        from x_agent.train import main

        main()

        # Verify the train function was invoked (via the mocked dependencies)
        mock_ml_modules["datasets"].load_dataset.assert_called_once_with(
            "json", data_files=dataset_path, split="train"
        )
