"""web_search_tool rejects a blank/whitespace-only query before any backend dispatch.

Regression for the Perplexity VALIDATION_ERROR ("query Invalid input") surfaced when a caller
passed an empty string query: the vendor burned a round trip only to reject it remotely. The fix
validates once in web_search_tool() itself, so every backend (not just Perplexity) is covered.
"""

import json
import os
from unittest.mock import MagicMock, patch

from tests.tools.conftest import register_all_web_providers


def test_empty_query_short_circuits_without_calling_backend():
    """An empty string query never reaches the provider's search()."""
    import tools.web_tools as wt

    register_all_web_providers()
    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}), \
         patch.object(wt, "_get_search_backend", return_value="perplexity"), \
         patch("plugins.web.perplexity.provider.httpx.post") as post:
        out = json.loads(wt.web_search_tool("", limit=5))

    post.assert_not_called()
    assert out["success"] is False
    assert "empty search query" in out["error"].lower()


def test_whitespace_only_query_short_circuits_without_calling_backend():
    """A whitespace-only query is treated the same as empty — stripped first, then rejected."""
    import tools.web_tools as wt

    register_all_web_providers()
    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}), \
         patch.object(wt, "_get_search_backend", return_value="perplexity"), \
         patch("plugins.web.perplexity.provider.httpx.post") as post:
        out = json.loads(wt.web_search_tool("   \n\t  ", limit=5))

    post.assert_not_called()
    assert out["success"] is False
    assert "empty search query" in out["error"].lower()


def test_query_with_surrounding_whitespace_is_stripped_before_dispatch():
    """A non-empty query with leading/trailing whitespace is trimmed before hitting the vendor."""
    import tools.web_tools as wt

    register_all_web_providers()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": []}
    resp.text = "{}"
    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}), \
         patch.object(wt, "_get_search_backend", return_value="perplexity"), \
         patch("plugins.web.perplexity.provider.httpx.post", return_value=resp) as post:
        out = json.loads(wt.web_search_tool("  bloom filter  ", limit=5))

    assert post.call_args.kwargs["json"]["query"] == "bloom filter"
    assert out["success"] is True
