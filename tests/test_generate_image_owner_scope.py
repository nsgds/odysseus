"""Native generate_image must resolve endpoints and tag the gallery row with the
CALLER's owner — not whichever endpoint resolves first. Owner is threaded as a
TRUSTED, server-side value (the `_owner` arg injected by the tool-execution
bridge), never a model-controlled schema field, so a model can't spoof it to
reach another user's private image endpoint/API key (security review on #4123).
"""

import asyncio
import inspect
from unittest.mock import patch

import pytest

import src.agent_loop  # noqa: F401  (import-order: tool_schemas is circular)
import mcp_servers.image_gen_server as srv
import src.tool_execution as te


def _run(coro):
    try:
        return asyncio.run(coro)
    except Exception:
        return None


def _hermetic_settings():
    """Pin the settings reads call_tool makes so the test doesn't depend on the
    harness having a real settings store (get_setting must yield its default —
    notably image_gen_enabled=True — or call_tool aborts before resolution)."""
    return (patch("src.settings.get_setting", lambda key, default=None: default),
            patch("src.settings.load_settings", lambda: {}))


def test_server_uses_trusted_underscore_owner_not_model_owner():
    """`_owner` (trusted) wins; a model-supplied `owner` key is ignored."""
    captured = {}

    def fake_resolve(spec, owner=None):
        captured.setdefault("owner", owner)
        raise ValueError("short-circuit before any network call")

    s1, s2 = _hermetic_settings()
    with s1, s2, patch("src.ai_interaction._resolve_model", fake_resolve):
        _run(srv.call_tool("generate_image",
                           {"prompt": "a cat", "_owner": "alice", "owner": "bob"}))
    assert captured.get("owner") == "alice"


def test_server_ignores_model_owner_without_trusted_injection():
    """Without the bridge's `_owner`, a model-supplied `owner` does NOT scope resolution."""
    captured = {}

    def fake_resolve(spec, owner=None):
        captured.setdefault("owner", owner)
        raise ValueError("short-circuit")

    s1, s2 = _hermetic_settings()
    with s1, s2, patch("src.ai_interaction._resolve_model", fake_resolve):
        _run(srv.call_tool("generate_image", {"prompt": "a cat", "owner": "bob"}))
    # The resolver MUST have been reached (guards against a vacuous pass if
    # call_tool aborts earlier, e.g. on a settings read).
    assert "owner" in captured
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


# Source-inspection guards (matching the repo idiom in test_ai_interaction_owner_scope):
# that file pins do_generate_image (the UI path) as owner-scoped, but the AGENT path runs
# the image MCP server, which it does NOT cover. These pin the agent path so the #4123
# regression can't silently return.

def test_mcp_image_server_threads_trusted_owner():
    """The image MCP server's call_tool must read the trusted `_owner` and use it for both
    endpoint resolution and the gallery row."""
    src = inspect.getsource(srv.call_tool)
    assert 'arguments.get("_owner")' in src            # trusted owner, not a schema field
    assert "asyncio.to_thread(_resolve_model, cand, owner=owner)" in src  # owner-scoped resolution (off-loop)
    assert "owner=owner," in src                       # GalleryImage tagged with the owner


def test_bridge_injects_trusted_owner_for_generate_image_only():
    """_call_mcp_tool injects `_owner` only for generate_image, gated by tool name."""
    src = inspect.getsource(te._call_mcp_tool)
    assert 'tool == "generate_image"' in src
    assert '"_owner": owner' in src


def test_resolve_model_isolates_private_image_endpoint_by_owner():
    """Gold-standard runtime isolation (real owner_filter SQL on an in-memory DB).

    A private image endpoint owned by 'alice' must be invisible to 'bob', so the
    agent path — which now threads the caller's owner — never resolves bob's
    request to alice's endpoint or transmits alice's API key. Reproduces the
    #4123 P1 (the ownerless path leaks) and proves the fix closes it.
    """
    import httpx
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import src.ai_interaction as ai
    import src.database as dbmod
    # `src.database` is stubbed with MagicMocks in the test harness (see conftest);
    # use the real ORM class from core.database and point the stub at it so
    # _resolve_model's `from src.database import ...` resolves to real objects.
    from core.database import ModelEndpoint

    engine = create_engine("sqlite:///:memory:")
    ModelEndpoint.metadata.create_all(engine, tables=[ModelEndpoint.__table__])
    TestSession = sessionmaker(bind=engine)
    s = TestSession()
    s.add(ModelEndpoint(
        id="ep-alice", name="alice-private-img",
        base_url="http://alice-private:8080/v1", api_key="ALICE-SECRET-KEY",
        is_enabled=True, model_type="image", owner="alice",
    ))
    s.commit()
    s.close()

    probes = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "alice-image-model"}]}

    def _fake_get(url, headers=None, timeout=None, **kw):
        probes.append((headers or {}).get("Authorization"))
        return _Resp()

    with patch.object(dbmod, "SessionLocal", TestSession), \
         patch.object(dbmod, "ModelEndpoint", ModelEndpoint), \
         patch.object(ai, "resolve_endpoint_runtime", lambda ep, owner=None: (ep.base_url, ep.api_key)), \
         patch.object(httpx, "get", _fake_get):
        # alice resolves her OWN endpoint, with her key
        _url, mid, hdrs = ai._resolve_model("alice-image-model", owner="alice")
        assert mid == "alice-image-model"
        assert hdrs.get("Authorization") == "Bearer ALICE-SECRET-KEY"

        # bob is BLOCKED and never even probes alice's endpoint -> prompt + key never sent
        probes.clear()
        with pytest.raises(ValueError):
            ai._resolve_model("alice-image-model", owner="bob")
        assert probes == []

        # Contract: an OWNERLESS call (what the pre-fix MCP server did) is unscoped and
        # leaks alice's endpoint+key — which is exactly why the fix threads owner.
        _u2, mid2, hdrs2 = ai._resolve_model("alice-image-model", owner=None)
        assert mid2 == "alice-image-model"
        assert hdrs2.get("Authorization") == "Bearer ALICE-SECRET-KEY"
