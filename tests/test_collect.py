"""Unit tests for x_agent.collect module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from x_agent.collect import fetch_recent_posts, write_jsonl, SEARCH_URL


class TestFetchRecentPosts:
    """Tests for fetch_recent_posts function."""

    def test_successful_fetch(self) -> None:
        """Should return posts from a successful API response."""
        mock_data = [
            {"id": "1", "text": "Hello world", "created_at": "2024-01-01T00:00:00Z"}
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": mock_data}
        mock_response.raise_for_status.return_value = None

        with patch("x_agent.collect.requests.get", return_value=mock_response) as mock_get:
            result = fetch_recent_posts("from:testuser", "test_token", max_results=10)

        assert result == mock_data
        mock_get.assert_called_once_with(
            SEARCH_URL,
            headers={"Authorization": "Bearer test_token"},
            params={
                "query": "from:testuser",
                "max_results": 10,
                "tweet.fields": "created_at,author_id,lang,public_metrics",
            },
            timeout=30,
        )

    def test_empty_response(self) -> None:
        """Should return empty list when API returns no data key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None

        with patch("x_agent.collect.requests.get", return_value=mock_response):
            result = fetch_recent_posts("from:noone", "test_token", max_results=10)

        assert result == []

    def test_max_results_too_low(self) -> None:
        """Should raise ValueError when max_results < 10."""
        with pytest.raises(ValueError, match="max_results must be between 10 and 100"):
            fetch_recent_posts("query", "token", max_results=5)

    def test_max_results_too_high(self) -> None:
        """Should raise ValueError when max_results > 100."""
        with pytest.raises(ValueError, match="max_results must be between 10 and 100"):
            fetch_recent_posts("query", "token", max_results=200)

    def test_max_results_boundary_low(self) -> None:
        """Should accept max_results=10 (lower boundary)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None

        with patch("x_agent.collect.requests.get", return_value=mock_response):
            result = fetch_recent_posts("query", "token", max_results=10)
        assert result == []

    def test_max_results_boundary_high(self) -> None:
        """Should accept max_results=100 (upper boundary)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None

        with patch("x_agent.collect.requests.get", return_value=mock_response):
            result = fetch_recent_posts("query", "token", max_results=100)
        assert result == []

    def test_http_error_propagates(self) -> None:
        """Should propagate HTTP errors from the API."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

        with patch("x_agent.collect.requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                fetch_recent_posts("query", "bad_token", max_results=10)


class TestWriteJsonl:
    """Tests for write_jsonl function."""

    def test_writes_posts_as_jsonl(self, tmp_dir: Path) -> None:
        """Should write each post as a JSON line."""
        posts = [
            {"id": "1", "text": "First post"},
            {"id": "2", "text": "Second post"},
        ]
        output = tmp_dir / "output.jsonl"
        write_jsonl(posts, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == posts[0]
        assert json.loads(lines[1]) == posts[1]

    def test_creates_parent_directories(self, tmp_dir: Path) -> None:
        """Should create parent directories if they don't exist."""
        output = tmp_dir / "deep" / "nested" / "output.jsonl"
        write_jsonl([{"id": "1", "text": "test"}], output)

        assert output.exists()

    def test_empty_posts_list(self, tmp_dir: Path) -> None:
        """Should create an empty file for an empty posts list."""
        output = tmp_dir / "empty.jsonl"
        write_jsonl([], output)

        assert output.exists()
        assert output.read_text(encoding="utf-8") == ""

    def test_unicode_handling(self, tmp_dir: Path) -> None:
        """Should handle unicode characters without escaping."""
        posts = [{"id": "1", "text": "Hello 🌍 世界"}]
        output = tmp_dir / "unicode.jsonl"
        write_jsonl(posts, output)

        content = output.read_text(encoding="utf-8").strip()
        parsed = json.loads(content)
        assert parsed["text"] == "Hello 🌍 世界"


class TestMainFunction:
    """Tests for the collect module's main() function."""

    def test_missing_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should exit with error when X_BEARER_TOKEN is not set."""
        monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
        monkeypatch.setattr(
            "sys.argv",
            ["collect", "--query", "from:test"],
        )

        with pytest.raises(SystemExit):
            from x_agent.collect import main
            main()

    def test_main_success(self, monkeypatch: pytest.MonkeyPatch, tmp_dir: Path) -> None:
        """Should call fetch and write when token is present."""
        monkeypatch.setenv("X_BEARER_TOKEN", "fake_token")
        output_path = str(tmp_dir / "out.jsonl")
        monkeypatch.setattr(
            "sys.argv",
            ["collect", "--query", "from:test", "--output", output_path, "--max-results", "10"],
        )

        mock_posts = [{"id": "1", "text": "test post", "created_at": "2024-01-01T00:00:00Z"}]
        with patch("x_agent.collect.fetch_recent_posts", return_value=mock_posts):
            from x_agent.collect import main
            main()

        assert Path(output_path).exists()
