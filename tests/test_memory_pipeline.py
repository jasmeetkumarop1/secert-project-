import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_agent.memory import AgentMemory
from x_agent.prepare import prepare_dataset
from x_agent.realtime import store_posts


class MemoryPipelineTests(unittest.TestCase):
    def test_prepare_filters_and_wraps_posts(self):
        with self.subTest("prepare jsonl"):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                raw_path = tmp_path / "raw.jsonl"
                train_path = tmp_path / "train.jsonl"
                raw_path.write_text(
                    "\n".join(
                        [
                            json.dumps({"id": "1", "text": "Hello from X social media. This is useful training text."}),
                            json.dumps({"id": "2", "text": "short"}),
                        ]
                    ),
                    encoding="utf-8",
                )

                count = prepare_dataset(raw_path, train_path, min_chars=20)

                self.assertEqual(count, 1)
                self.assertEqual(
                    json.loads(train_path.read_text(encoding="utf-8"))["text"],
                    "<x_post>Hello from X social media. This is useful training text.</x_post>",
                )

    def test_memory_pins_first_and_deduplicates_x_posts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            memory = AgentMemory(Path(directory) / "memory.sqlite")
            try:
                self.assertTrue(memory.remember_first("Remember my first instruction forever."))
                self.assertFalse(memory.remember_first("Do not replace the first instruction."))

                inserted = store_posts(
                    memory,
                    [
                        {
                            "id": "tweet-1",
                            "text": "Realtime X update about AI agents",
                            "created_at": "2026-06-16T12:00:00Z",
                        },
                        {"id": "tweet-1", "text": "Duplicate X update", "created_at": "2026-06-16T12:00:00Z"},
                    ],
                )

                results = memory.search("AI agents", limit=5)
                self.assertEqual(inserted, 1)
                self.assertEqual(results[0].text, "Remember my first instruction forever.")
                self.assertTrue(any(item.text == "Realtime X update about AI agents" for item in results))
            finally:
                memory.close()


class AdaptiveParameterSchedulerTests(unittest.TestCase):
    def test_parameters_grow_and_persist_over_time(self):
        import tempfile
        from x_agent.parameters import AdaptiveParameterScheduler

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "params.json"
            scheduler = AdaptiveParameterScheduler(state_path)

            first = scheduler.training_parameters(dataset_size=2_500, base_epochs=1.0, base_block_size=128)
            scheduler.record_training_run(first)
            second = scheduler.training_parameters(dataset_size=2_500, base_epochs=1.0, base_block_size=128)

            self.assertGreater(second.epochs, first.epochs)
            self.assertGreater(second.block_size, first.block_size)
            self.assertTrue(state_path.exists())

            chat = scheduler.chat_parameters(base_max_new_tokens=80)
            scheduler.record_chat_turn(chat)
            reloaded = AdaptiveParameterScheduler(state_path)
            self.assertEqual(reloaded.state["training_runs"], 1)
            self.assertEqual(reloaded.state["chat_turns"], 1)


if __name__ == "__main__":
    unittest.main()
