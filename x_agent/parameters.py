"""Adaptive parameter scheduling for training and chat.

This module does not magically add neural-network weights to an existing model.
Instead, it grows practical agent parameters over time: training epochs, context
block size, gradient accumulation, retrieval depth, and generation token budget.
The scheduler persists state so every successful run can make the next run a bit
stronger while staying inside configurable safety caps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingParameters:
    """Training knobs that can be increased across repeated training runs."""

    epochs: float
    block_size: int
    gradient_accumulation_steps: int
    learning_rate: float


@dataclass(frozen=True)
class ChatParameters:
    """Inference knobs that can grow as the agent accumulates experience."""

    max_new_tokens: int
    memory_limit: int
    temperature: float
    top_p: float


class AdaptiveParameterScheduler:
    """Persisted scheduler for gradually increasing agent capacity over time."""

    def __init__(self, path: str | Path = "data/parameter_state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"training_runs": 0, "chat_turns": 0}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        """Persist scheduler state."""
        self.path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")

    def training_parameters(
        self,
        *,
        dataset_size: int,
        base_epochs: float,
        base_block_size: int,
        base_learning_rate: float = 5e-5,
        max_epochs: float = 5.0,
        max_block_size: int = 512,
        max_gradient_accumulation_steps: int = 16,
    ) -> TrainingParameters:
        """Return training parameters that scale with dataset size and run count."""
        runs = int(self.state.get("training_runs", 0))
        dataset_factor = min(4, max(0, dataset_size // 1_000))
        growth = runs + dataset_factor

        epochs = min(max_epochs, base_epochs + (0.25 * growth))
        block_size = min(max_block_size, base_block_size + (32 * growth))
        gradient_accumulation_steps = min(max_gradient_accumulation_steps, 4 + growth)
        learning_rate = max(1e-5, base_learning_rate * (0.98**runs))

        return TrainingParameters(
            epochs=round(epochs, 3),
            block_size=block_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
        )

    def chat_parameters(
        self,
        *,
        base_max_new_tokens: int,
        base_memory_limit: int = 6,
        max_new_tokens_cap: int = 256,
        max_memory_limit: int = 20,
    ) -> ChatParameters:
        """Return chat parameters that grow with conversation experience."""
        turns = int(self.state.get("chat_turns", 0))
        growth = turns // 10

        return ChatParameters(
            max_new_tokens=min(max_new_tokens_cap, base_max_new_tokens + (8 * growth)),
            memory_limit=min(max_memory_limit, base_memory_limit + growth),
            temperature=max(0.6, 0.8 - (0.01 * growth)),
            top_p=0.95,
        )

    def record_training_run(self, parameters: TrainingParameters) -> None:
        """Record that a training run completed with the given parameters."""
        self.state["training_runs"] = int(self.state.get("training_runs", 0)) + 1
        self.state["last_training_parameters"] = asdict(parameters)
        self.save()

    def record_chat_turn(self, parameters: ChatParameters) -> None:
        """Record that one chat turn completed with the given parameters."""
        self.state["chat_turns"] = int(self.state.get("chat_turns", 0)) + 1
        self.state["last_chat_parameters"] = asdict(parameters)
        self.save()
