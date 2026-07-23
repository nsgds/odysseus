"""#4123 image-capability gate: the image-generation resolution paths must not
bind a text/chat endpoint that merely lists a matching model id and then POST to
its /images/generations. _resolve_model grows an opt-in ``model_type`` filter;
the two image candidate loops use it two-pass (image-tagged first, then an
ungated fallback) so a text endpoint can't win the loop while a legitimately-image
endpoint left tagged 'llm' (the add-endpoint form default) still resolves.

Regression is expressed at the resolver level — the loops' behaviour is exactly
"walk candidates, skip a ValueError, accept the first hit", so proving the gate
turns a text-endpoint hit into a ValueError (and the ungated pass recovers a
mistagged image endpoint) pins the loop behaviour without mocking the async POST.
The loops' actual use of the gate is pinned by the source-inspection assertions
in test_generate_image_owner_scope.py / test_ai_interaction_owner_scope.py.
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.agent_loop  # noqa: F401  (import-order: tool_schemas is circular)
import mcp_servers.image_gen_server as srv
import src.ai_interaction as ai
import src.database as dbmod
from core.database import ModelEndpoint


def _resp(payload):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return _Resp()


def _seed(*rows):
    # StaticPool + one shared connection so the :memory: DB is visible from the
    # asyncio.to_thread worker _resolve_model runs in (call_tool offloads it).
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    ModelEndpoint.metadata.create_all(engine, tables=[ModelEndpoint.__table__])
    TestSession = sessionmaker(bind=engine)
    s = TestSession()
    for r in rows:
        s.add(r)
    s.commit()
    s.close()
    return TestSession


def _patches(TestSession, fake_get):
    return (
        patch.object(dbmod, "SessionLocal", TestSession),
        patch.object(dbmod, "ModelEndpoint", ModelEndpoint),
        patch.object(ai, "resolve_endpoint_runtime",
                     lambda ep, owner=None: (ep.base_url, ep.api_key)),
        patch.object(httpx, "get", fake_get),
    )


def _serves(mapping):
    """Return a fake httpx.get that serves a different /models body per host."""
    def _fake_get(url, headers=None, timeout=None, **kw):
        for host, payload in mapping.items():
            if host in url:
                return _resp(payload)
        return _resp({"data": []})
    return _fake_get


def _fake_async_client(posted):
    """httpx.AsyncClient stand-in for call_tool: captures the POST url + model into
    ``posted`` and returns a minimal 200 (url image) so call_tool completes."""
    class _PostResp:
        status_code = 200
        text = ""

        def json(self):
            return {"data": [{"url": "https://cdn.example/out.png"}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def post(self, url, json=None, headers=None):
            posted["url"] = url
            posted["model"] = (json or {}).get("model")
            return _PostResp()

    return _Client


def test_gate_skips_text_endpoint_and_selects_image_model():
    """Regression for #4123: a text endpoint listing an image-sounding id ('dall-e-3')
    resolves FIRST, so the ungated resolve binds it — but the image-gated resolve
    skips it, letting the loop fall through to the configured image model."""
    TestSession = _seed(
        # text endpoint inserted FIRST so it would win the ungated candidate loop
        ModelEndpoint(id="ep-text", name="text-proxy", base_url="http://text-proxy:8080/v1",
                      api_key="TEXT-KEY", is_enabled=True, model_type="llm", owner=None),
        ModelEndpoint(id="ep-img", name="img", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="image", owner=None),
    )
    fake_get = _serves({"text-proxy": {"data": [{"id": "dall-e-3"}]},
                        "img": {"data": [{"id": "gpt-image-1"}]}})
    p1, p2, p3, p4 = _patches(TestSession, fake_get)
    with p1, p2, p3, p4:
        # Without the gate, the text endpoint wins the 'dall-e-3' candidate (the bug).
        url, mid, _ = ai._resolve_model("dall-e-3", owner=None)
        assert mid == "dall-e-3"
        assert "text-proxy" in url

        # With the gate, that candidate is rejected (text endpoint filtered out)...
        with pytest.raises(ValueError):
            ai._resolve_model("dall-e-3", owner=None, model_type="image")

        # ...so the loop advances to the configured image model on the image endpoint.
        url2, mid2, _ = ai._resolve_model("gpt-image-1", owner=None, model_type="image")
        assert mid2 == "gpt-image-1"
        assert "img" in url2


def test_gate_does_not_over_exclude_a_correctly_tagged_image_endpoint():
    """Happy path: a properly image-tagged endpoint still resolves under the gate."""
    TestSession = _seed(
        ModelEndpoint(id="ep-img", name="img", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="image", owner=None),
    )
    p1, p2, p3, p4 = _patches(TestSession, _serves({"img": {"data": [{"id": "gpt-image-1"}]}}))
    with p1, p2, p3, p4:
        url, mid, hdrs = ai._resolve_model("gpt-image-1", owner=None, model_type="image")
        assert mid == "gpt-image-1"
        assert hdrs.get("Authorization") == "Bearer IMG-KEY"
        # url is a chat-completions URL so the '/chat/completions'->'/images/generations'
        # rewrite downstream still works.
        assert "img" in url


def test_default_model_type_none_is_unchanged_for_chat_endpoint():
    """Blast-radius guard: chat/vision/pipeline callers pass no model_type, so a
    plain 'llm' endpoint resolves exactly as before (the filter is opt-in)."""
    TestSession = _seed(
        ModelEndpoint(id="ep-chat", name="chat", base_url="http://chat:8080/v1",
                      api_key="CHAT-KEY", is_enabled=True, model_type="llm", owner=None),
    )
    p1, p2, p3, p4 = _patches(TestSession, _serves({"chat": {"data": [{"id": "gpt-4o-mini"}]}}))
    with p1, p2, p3, p4:
        _url, mid, _ = ai._resolve_model("gpt-4o-mini", owner=None)
        assert mid == "gpt-4o-mini"


def test_untagged_image_endpoint_recovered_by_ungated_fallback_pass():
    """No-regression proof for the two-pass design: an image endpoint left tagged
    'llm' (the add-endpoint form default) is filtered out by the image pass, but the
    ungated fallback pass (model_type=None) still resolves it — so a setup that works
    today is never hard-failed by the gate."""
    TestSession = _seed(
        ModelEndpoint(id="ep-img-llm", name="img-mistagged", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="llm", owner=None),
    )
    p1, p2, p3, p4 = _patches(TestSession, _serves({"img": {"data": [{"id": "gpt-image-1"}]}}))
    with p1, p2, p3, p4:
        # image pass: filtered out
        with pytest.raises(ValueError):
            ai._resolve_model("gpt-image-1", owner=None, model_type="image")
        # ungated fallback pass: recovered
        _url, mid, _ = ai._resolve_model("gpt-image-1", owner=None, model_type=None)
        assert mid == "gpt-image-1"


# --- _resolve_image_capable: the two-pass helper that gates do_generate_image's
#     terminal BINDING resolve (#4123 review follow-up). Pins loop order behaviorally,
#     not by source-string, and covers the ungated-terminal-resolve hole. -------------

def test_resolve_image_capable_prefers_image_endpoint_over_text():
    """The do_generate_image binding path: _resolve_image_capable must prefer an
    image-tagged endpoint even when a TEXT endpoint (inserted first, so it would win
    an ungated resolve) also serves the id — closing the ungated-terminal-resolve
    hole. Regression: without image-first two-pass, this bound the text proxy."""
    TestSession = _seed(
        # text endpoint FIRST so a plain ungated resolve would bind it
        ModelEndpoint(id="ep-text", name="text-proxy", base_url="http://text-proxy:8080/v1",
                      api_key="TEXT-KEY", is_enabled=True, model_type="llm", owner=None),
        ModelEndpoint(id="ep-img", name="img", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="image", owner=None),
    )
    fake_get = _serves({"text-proxy": {"data": [{"id": "dall-e-3"}]},
                        "img": {"data": [{"id": "dall-e-3"}]}})
    p1, p2, p3, p4 = _patches(TestSession, fake_get)
    with p1, p2, p3, p4:
        # sanity: a plain ungated resolve DOES bind the text proxy (the hole)
        u0, _m0, _ = ai._resolve_model("dall-e-3", owner=None)
        assert "text-proxy" in u0
        # the helper must bind the IMAGE endpoint instead
        url, mid, _ = ai._resolve_image_capable("dall-e-3", owner=None)
        assert "img" in url and "text-proxy" not in url
        assert mid == "dall-e-3"


def test_resolve_image_capable_falls_back_to_ungated_for_mistagged_image_endpoint():
    """Two-pass fallback: an image endpoint left tagged 'llm' still resolves via the
    ungated pass, so a working setup is not hard-failed by the image-first preference."""
    TestSession = _seed(
        ModelEndpoint(id="ep-img-llm", name="img-mistagged", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="llm", owner=None),
    )
    p1, p2, p3, p4 = _patches(TestSession, _serves({"img": {"data": [{"id": "gpt-image-1"}]}}))
    with p1, p2, p3, p4:
        url, mid, _ = ai._resolve_image_capable("gpt-image-1", owner=None)
        assert mid == "gpt-image-1" and "img" in url


# --- MCP server candidate-loop two-pass, END TO END: drive srv.call_tool against the
#     REAL _resolve_model over a seeded DB, stubbing only the async POST. This pins the
#     loop's image-first behaviour (a text endpoint listing the id never gets the POST),
#     which the isolated _resolve_model tests can't prove. --------------------------------

def test_call_tool_two_pass_binds_image_endpoint_not_text_proxy():
    """A text endpoint (inserted FIRST) advertising 'dall-e-3' must NOT receive the
    /images/generations POST: the loop's image-tagged pass binds the image endpoint
    that also serves it. Without the gate the ungated loop binds the text proxy."""
    TestSession = _seed(
        ModelEndpoint(id="ep-text", name="text-proxy", base_url="http://text-proxy:8080/v1",
                      api_key="TEXT-KEY", is_enabled=True, model_type="llm", owner=None),
        ModelEndpoint(id="ep-img", name="img", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="image", owner=None),
    )
    fake_get = _serves({"text-proxy": {"data": [{"id": "dall-e-3"}]},
                        "img": {"data": [{"id": "dall-e-3"}]}})
    posted = {}
    p1, p2, p3, p4 = _patches(TestSession, fake_get)
    with p1, p2, p3, p4, \
         patch.object(httpx, "AsyncClient", _fake_async_client(posted)), \
         patch("src.settings.get_setting", lambda key, default=None: default), \
         patch("src.settings.load_settings", lambda: {}):
        # sanity: an UNGATED resolve WOULD bind the text proxy — proves it's a live
        # candidate the gate is specifically excluding (so the assert below isn't vacuous).
        assert "text-proxy" in ai._resolve_model("dall-e-3", owner=None)[0]
        asyncio.run(srv.call_tool(
            "generate_image", {"prompt": "a cat", "model": "dall-e-3", "_owner": None}))

    assert posted.get("url") == "http://img:8080/v1/images/generations"   # image endpoint
    assert "text-proxy" not in (posted.get("url") or "")                  # never the text proxy
    assert posted.get("model") == "dall-e-3"                              # bound on the image endpoint


def test_gate_composes_with_owner_scoping():
    """The model_type='image' gate must AND-compose with owner_filter, not replace it —
    the production shape (image paths always thread a real caller owner). A private image
    endpoint owned by alice stays invisible to bob even under the gate."""
    TestSession = _seed(
        ModelEndpoint(id="ep-alice", name="alice-img", base_url="http://alice-img:8080/v1",
                      api_key="ALICE-KEY", is_enabled=True, model_type="image", owner="alice"),
        ModelEndpoint(id="ep-bob", name="bob-img", base_url="http://bob-img:8080/v1",
                      api_key="BOB-KEY", is_enabled=True, model_type="image", owner="bob"),
    )
    fake_get = _serves({"alice-img": {"data": [{"id": "gpt-image-1"}]},
                        "bob-img": {"data": [{"id": "bob-only"}]}})
    p1, p2, p3, p4 = _patches(TestSession, fake_get)
    with p1, p2, p3, p4:
        # alice resolves her own image endpoint under the gate, with her key
        _url, mid, hdrs = ai._resolve_model("gpt-image-1", owner="alice", model_type="image")
        assert mid == "gpt-image-1"
        assert hdrs.get("Authorization") == "Bearer ALICE-KEY"
        # bob is blocked from alice's image endpoint even under the gate: the model_type
        # and owner filters compose (AND), they don't replace each other.
        with pytest.raises(ValueError):
            ai._resolve_model("gpt-image-1", owner="bob", model_type="image")


def test_call_tool_ungated_fallback_recovers_mistagged_image_endpoint():
    """E2E: with NO image-tagged endpoint, the server loop's ungated second pass (_mt=None)
    must recover a legit image endpoint left tagged 'llm' and POST to it."""
    TestSession = _seed(
        ModelEndpoint(id="ep-img-llm", name="img-mistagged", base_url="http://img:8080/v1",
                      api_key="IMG-KEY", is_enabled=True, model_type="llm", owner=None),
    )
    posted = {}
    p1, p2, p3, p4 = _patches(TestSession, _serves({"img": {"data": [{"id": "gpt-image-1"}]}}))
    with p1, p2, p3, p4, \
         patch.object(httpx, "AsyncClient", _fake_async_client(posted)), \
         patch("src.settings.get_setting", lambda key, default=None: default), \
         patch("src.settings.load_settings", lambda: {}):
        asyncio.run(srv.call_tool(
            "generate_image", {"prompt": "a cat", "model": "gpt-image-1", "_owner": None}))

    assert posted.get("url") == "http://img:8080/v1/images/generations"   # recovered via ungated pass
    assert posted.get("model") == "gpt-image-1"
