"""Native generate_image must resolve endpoints and tag the gallery row with the
CALLER's owner — not whichever endpoint resolves first. Owner is threaded as a
TRUSTED, server-side value (the `_owner` arg injected by the tool-execution
bridge), never a model-controlled schema field, so a model can't spoof it to
reach another user's private image endpoint/API key (security review on #4123).
"""

import asyncio
from unittest.mock import patch

import src.agent_loop  # noqa: F401  (import-order: tool_schemas is circular)
import mcp_servers.image_gen_server as srv
import src.tool_execution as te


def _run(coro):
    try:
        return asyncio.run(coro)
    except Exception:
        return None


def test_server_uses_trusted_underscore_owner_not_model_owner():
    """`_owner` (trusted) wins; a model-supplied `owner` key is ignored."""
    captured = {}

    def fake_resolve(spec, owner=None):
        captured.setdefault("owner", owner)
        raise ValueError("short-circuit before any network call")

    with patch("src.ai_interaction._resolve_model", fake_resolve):
        _run(srv.call_tool("generate_image",
                           {"prompt": "a cat", "_owner": "alice", "owner": "bob"}))
    assert captured.get("owner") == "alice"


def test_server_ignores_model_owner_without_trusted_injection():
    """Without the bridge's `_owner`, a model-supplied `owner` does NOT scope resolution."""
    captured = {}

    def fake_resolve(spec, owner=None):
        captured.setdefault("owner", owner)
        raise ValueError("short-circuit")

    with patch("src.ai_interaction._resolve_model", fake_resolve):
        _run(srv.call_tool("generate_image", {"prompt": "a cat", "owner": "bob"}))
    assert captured.get("owner") is None


def test_bridge_injects_only_underscore_owner_for_generate_image():
    """The bridge injects the trusted `_owner` (and only for generate_image). Tested on
    the arg-build step so it's free of the MCP-manager/get_mcp_manager dependency that
    pollutes across the suite: a generate_image arg dict never carries a model-supplied
    owner, and the schema never exposes one."""
    import json
    # _parse_generate_image (the MCP arg builder) only keeps prompt/model/size/quality,
    # so a model can't smuggle an owner through the args channel.
    args = te._parse_generate_image(json.dumps({"prompt": "a cat", "_owner": "x", "owner": "y"}))
    assert "_owner" not in args and "owner" not in args
    # The native generate_image schema must NOT expose owner as a model-controlled field.
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    gi = next(s for s in FUNCTION_TOOL_SCHEMAS if s.get("function", {}).get("name") == "generate_image")
    assert "owner" not in gi["function"]["parameters"]["properties"]
    assert "_owner" not in gi["function"]["parameters"]["properties"]
