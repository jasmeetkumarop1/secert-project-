import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from x_agent.collect import SEARCH_URL, fetch_recent_posts, write_jsonl


class FetchRecentPostsTests(unittest.TestCase):
    def test_fetch_recent_posts_rejects_invalid_max_results_before_request(self) -> None:
        with patch("x_agent.collect.requests.get") as get:
            with self.assertRaisesRegex(ValueError, "max_results must be between 10 and 100"):
                fetch_recent_posts("from:example", "token", max_results=9)

        get.assert_not_called()

    def test_fetch_recent_posts_calls_x_api_and_returns_data(self) -> None:
        response = Mock()
        response.json.return_value = {"data": [{"id": "1", "text": "hello"}]}

        with patch("x_agent.collect.requests.get", return_value=response) as get:
            posts = fetch_recent_posts("from:example -is:retweet", "secret-token", max_results=25)

        self.assertEqual(posts, [{"id": "1", "text": "hello"}])
        response.raise_for_status.assert_called_once_with()
        get.assert_called_once_with(
            SEARCH_URL,
            headers={"Authorization": "Bearer secret-token"},
            params={
                "query": "from:example -is:retweet",
                "max_results": 25,
                "tweet.fields": "created_at,author_id,lang,public_metrics",
            },
            timeout=30,
        )

    def test_fetch_recent_posts_returns_empty_list_when_response_has_no_data(self) -> None:
        response = Mock()
        response.json.return_value = {"meta": {"result_count": 0}}

        with patch("x_agent.collect.requests.get", return_value=response):
            posts = fetch_recent_posts("from:example", "token", max_results=10)

        self.assertEqual(posts, [])


class WriteJsonlTests(unittest.TestCase):
    def test_write_jsonl_creates_parent_directory_and_preserves_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nested" / "posts.jsonl"
            posts = [
                {"id": "1", "text": "hello 🚀"},
                {"id": "2", "text": "second"},
            ]

            write_jsonl(posts, output)

            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [json.dumps(post, ensure_ascii=False) for post in posts],
            )


if __name__ == "__main__":
    unittest.main()
