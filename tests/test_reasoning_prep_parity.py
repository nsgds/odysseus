"""Payload-parity and cache regressions for the shared request-prep boundary.

The per-model reasoning preference must be honored by ALL builders — sync
llm_call, async llm_call_async, and the streaming path — not just streaming
(review P2). The mutation must run BEFORE the sync/async response-cache key is
computed, so two requests that differ only by reasoning mode never share a
cache entry. Models are made toggleable by seeding a control spec on the
endpoint (nothing is inferred from names).
"""
import asyncio
import json

import httpx

from src import llm_core
from src.model_capabilities import (
    REASONING_CONTROL_MESSAGE_DIRECTIVE as MD,
    REASONING_CONTROL_TEMPLATE_KWARG as TK,
)

MODEL = "nemotron-nano-12b-vl"
MD_CONTROL = {"mechanism": MD, "values": {"on": "/think"}}          # default-off
GLM = "glm-4.5-air"
TK_CONTROL = {"mechanism": TK, "kwarg_path": ["chat_template_kwargs", "enable_thinking"],
              "values": {"on": True, "off": False}}


def _seed_endpoint(url: str, modes, ident: str, *, model=MODEL, control=MD_CONTROL,
                   cached=None) -> None:
    """Seed an endpoint that makes `model` toggleable via `control`, with the
    given on/off `modes` preference."""
    from core.database import SessionLocal, ModelEndpoint, Base, engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == ident).delete(synchronize_session=False)
        db.add(ModelEndpoint(id=ident, name="RC prep", base_url=url, is_enabled=True,
                             cached_models=json.dumps(cached or [model]),
                             reasoning_controls=json.dumps({model: control}) if control else None,
                             reasoning_modes=json.dumps(modes) if modes else None))
        db.commit()
    finally:
        db.close()


def _cleanup_endpoint(ident: str) -> None:
    from core.database import SessionLocal, ModelEndpoint
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == ident).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _fake_sync_post(seen, content="OK"):
    def fake_post(url, headers=None, json=None, timeout=None):
        seen["json"] = json
        return httpx.Response(
            200, request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )
    return fake_post


def _fake_async_post(seen, content="OK"):
    async def fake_post(client, url, headers, **kwargs):
        seen["json"] = kwargs.get("json")
        return httpx.Response(
            200, request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )
    return fake_post


class _FakeStreamResponse:
    status_code = 200
    headers = {}

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"OK"}}]}'
        yield 'data: [DONE]'

    async def aread(self):
        return b""


class _FakeStreamClient:
    def __init__(self, seen):
        self._seen = seen

    def stream(self, method, url, json=None, headers=None, timeout=None):
        self._seen["json"] = json

        class _Ctx:
            async def __aenter__(self):
                return _FakeStreamResponse()

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _last_user_content(payload):
    users = [m for m in payload["messages"] if m.get("role") == "user"]
    return users[-1]["content"]


async def _collect(gen):
    return [chunk async for chunk in gen]


class TestPayloadParity:
    """Mode "on" must reach the outgoing payload on every path; off/auto must
    leave the payload untouched on every path."""

    def test_sync_call_injects_think(self, monkeypatch):
        url = "http://rc-prep-sync.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-prep-sync")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen))
        try:
            assert llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hello"}], max_tokens=5) == "OK"
            assert _last_user_content(seen["json"]) == "/think hello"
        finally:
            _cleanup_endpoint("rc-prep-sync")
            llm_core._response_cache.clear()

    def test_async_call_injects_think(self, monkeypatch):
        url = "http://rc-prep-async.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-prep-async")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", _fake_async_post(seen))
        try:
            result = asyncio.run(llm_core.llm_call_async(
                url, MODEL, [{"role": "user", "content": "hello"}], max_tokens=5))
            assert result == "OK"
            assert _last_user_content(seen["json"]) == "/think hello"
        finally:
            _cleanup_endpoint("rc-prep-async")
            llm_core._response_cache.clear()

    def test_stream_injects_think(self, monkeypatch):
        url = "http://rc-prep-stream.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-prep-stream")
        seen = {}
        monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeStreamClient(seen))
        try:
            chunks = asyncio.run(_collect(llm_core.stream_llm(
                url, MODEL, [{"role": "user", "content": "hello"}], max_tokens=5)))
            assert chunks, "stream should yield chunks"
            assert _last_user_content(seen["json"]) == "/think hello"
        finally:
            _cleanup_endpoint("rc-prep-stream")

    def test_no_think_off_direction_injects(self, monkeypatch):
        """A default-on model configured with {"off": "/no_think"}: mode off
        injects /no_think; on/auto inject nothing."""
        url = "http://rc-prep-nothink.invalid:9911/v1"
        control = {"mechanism": MD, "values": {"off": "/no_think"}}
        _seed_endpoint(url, {MODEL: "off"}, "rc-prep-nothink", control=control)
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen))
        try:
            llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert _last_user_content(seen["json"]) == "/no_think hi"
            _seed_endpoint(url, {MODEL: "on"}, "rc-prep-nothink", control=control)  # on undeclared
            seen2 = {}
            monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen2))
            llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hi2"}], max_tokens=5)
            assert _last_user_content(seen2["json"]) == "hi2"
        finally:
            _cleanup_endpoint("rc-prep-nothink")
            llm_core._response_cache.clear()

    def test_off_and_auto_leave_all_paths_untouched(self, monkeypatch):
        off_url = "http://rc-prep-off.invalid:9911/v1"
        auto_url = "http://rc-prep-auto.invalid:9911/v1"
        _seed_endpoint(off_url, {MODEL: "off"}, "rc-prep-off")   # off undeclared for {"on"} spec
        _seed_endpoint(auto_url, None, "rc-prep-auto")           # auto
        llm_core._response_cache.clear()
        sync_seen, auto_seen, async_seen, stream_seen = {}, {}, {}, {}
        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", _fake_async_post(async_seen))
        try:
            monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(sync_seen))
            llm_core.llm_call(off_url, MODEL, [{"role": "user", "content": "hello"}], max_tokens=5)
            monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(auto_seen))
            llm_core.llm_call(auto_url, MODEL, [{"role": "user", "content": "hello auto"}], max_tokens=5)
            asyncio.run(llm_core.llm_call_async(
                off_url, MODEL, [{"role": "user", "content": "hello again"}], max_tokens=5))
            monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeStreamClient(stream_seen))
            asyncio.run(_collect(llm_core.stream_llm(
                off_url, MODEL, [{"role": "user", "content": "hello stream"}], max_tokens=5)))
            assert _last_user_content(sync_seen["json"]) == "hello"
            assert _last_user_content(auto_seen["json"]) == "hello auto"
            assert _last_user_content(async_seen["json"]) == "hello again"
            assert _last_user_content(stream_seen["json"]) == "hello stream"
            auto_async_seen, auto_stream_seen = {}, {}
            monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", _fake_async_post(auto_async_seen))
            asyncio.run(llm_core.llm_call_async(
                auto_url, MODEL, [{"role": "user", "content": "auto async"}], max_tokens=5))
            monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeStreamClient(auto_stream_seen))
            asyncio.run(_collect(llm_core.stream_llm(
                auto_url, MODEL, [{"role": "user", "content": "auto stream"}], max_tokens=5)))
            assert _last_user_content(auto_async_seen["json"]) == "auto async"
            assert _last_user_content(auto_stream_seen["json"]) == "auto stream"
        finally:
            _cleanup_endpoint("rc-prep-off")
            _cleanup_endpoint("rc-prep-auto")
            llm_core._response_cache.clear()


class TestCacheTransition:
    def test_mode_flip_does_not_reuse_cached_response(self, monkeypatch):
        url = "http://rc-prep-cache.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "off"}, "rc-prep-cache")
        llm_core._response_cache.clear()
        calls = []

        def fake_post(u, headers=None, json=None, timeout=None):
            calls.append(json)
            body = "A" if len(calls) == 1 else "B"
            return httpx.Response(
                200, request=httpx.Request("POST", u),
                json={"choices": [{"message": {"content": body}}]},
            )

        monkeypatch.setattr(llm_core.httpx, "post", fake_post)
        msgs = [{"role": "user", "content": "explain X"}]
        try:
            assert llm_core.llm_call(url, MODEL, msgs, max_tokens=5) == "A"
            assert llm_core.llm_call(url, MODEL, msgs, max_tokens=5) == "A"
            assert len(calls) == 1
            _seed_endpoint(url, {MODEL: "on"}, "rc-prep-cache")
            assert llm_core.llm_call(url, MODEL, msgs, max_tokens=5) == "B"
            assert len(calls) == 2
            assert _last_user_content(calls[1]) == "/think explain X"
            _seed_endpoint(url, {MODEL: "off"}, "rc-prep-cache")
            assert llm_core.llm_call(url, MODEL, msgs, max_tokens=5) == "A"
            assert len(calls) == 2
        finally:
            _cleanup_endpoint("rc-prep-cache")
            llm_core._response_cache.clear()

    def test_mode_flip_does_not_reuse_cached_response_async(self, monkeypatch):
        url = "http://rc-prep-cache-async.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "off"}, "rc-prep-cache-async")
        llm_core._response_cache.clear()
        calls = []

        async def fake_post(client, u, headers, **kwargs):
            calls.append(kwargs.get("json"))
            body = "A" if len(calls) == 1 else "B"
            return httpx.Response(
                200, request=httpx.Request("POST", u),
                json={"choices": [{"message": {"content": body}}]},
            )

        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", fake_post)
        msgs = [{"role": "user", "content": "explain X"}]
        try:
            assert asyncio.run(llm_core.llm_call_async(url, MODEL, msgs, max_tokens=5)) == "A"
            assert asyncio.run(llm_core.llm_call_async(url, MODEL, msgs, max_tokens=5)) == "A"
            assert len(calls) == 1
            _seed_endpoint(url, {MODEL: "on"}, "rc-prep-cache-async")
            assert asyncio.run(llm_core.llm_call_async(url, MODEL, msgs, max_tokens=5)) == "B"
            assert len(calls) == 2
            assert _last_user_content(calls[1]) == "/think explain X"
            _seed_endpoint(url, {MODEL: "off"}, "rc-prep-cache-async")
            assert asyncio.run(llm_core.llm_call_async(url, MODEL, msgs, max_tokens=5)) == "A"
            assert len(calls) == 2
        finally:
            _cleanup_endpoint("rc-prep-cache-async")
            llm_core._response_cache.clear()


class TestResponseNormalization:
    """A message-directive model's orphan-`</think>` response is repaired at the
    cached builders (mechanism-based, not "did we inject"); out-of-band and
    non-message-directive models are never rewritten."""

    SHAPE_B = ('Okay, so the user asked "What is 2+2?" and wants a brief '
               'answer. So just state 4.\n</think>\n\n2 + 2 equals 4.\n')

    def test_sync_orphan_closer_is_repaired(self, monkeypatch):
        url = "http://rc-norm-sync.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-norm-sync")
        llm_core._response_cache.clear()
        calls, seen = [], {}
        body = _fake_sync_post(seen, content=self.SHAPE_B)

        def counting_post(*a, **kw):
            calls.append(1)
            return body(*a, **kw)

        monkeypatch.setattr(llm_core.httpx, "post", counting_post)
        try:
            out = llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert out == "<think>" + self.SHAPE_B
            from src.text_helpers import strip_think
            assert "user asked" not in strip_think(out)
            assert llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5) == out
            assert len(calls) == 1
        finally:
            _cleanup_endpoint("rc-norm-sync")
            llm_core._response_cache.clear()

    def test_async_orphan_closer_is_repaired(self, monkeypatch):
        url = "http://rc-norm-async.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-norm-async")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async",
                            _fake_async_post(seen, content=self.SHAPE_B))
        try:
            out = asyncio.run(llm_core.llm_call_async(
                url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5))
            assert out == "<think>" + self.SHAPE_B
        finally:
            _cleanup_endpoint("rc-norm-async")
            llm_core._response_cache.clear()

    def test_non_message_directive_model_never_repaired(self, monkeypatch):
        """A template-kwarg model's content (even with a literal closer) is
        answer text — the repair is mechanism-scoped and must skip it."""
        url = "http://rc-norm-tk.invalid:9911/v1"
        _seed_endpoint(url, {GLM: "on"}, "rc-norm-tk", model=GLM, control=TK_CONTROL)
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen, content=self.SHAPE_B))
        try:
            out = llm_core.llm_call(url, GLM, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert out == self.SHAPE_B  # verbatim — not a message-directive model
        finally:
            _cleanup_endpoint("rc-norm-tk")
            llm_core._response_cache.clear()

    def test_parserful_serve_literal_closer_not_repaired(self, monkeypatch):
        """A message-directive model on a serve WITH a reasoning parser returns
        reasoning out-of-band and a clean content field — a literal </think> in
        that answer is answer text and must NOT be repaired."""
        url = "http://rc-norm-oob.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-norm-oob")
        llm_core._response_cache.clear()
        answer = "The closing tag is written `</think>` in that template."

        def fake_post(u, headers=None, json=None, timeout=None):
            return httpx.Response(
                200, request=httpx.Request("POST", u),
                json={"choices": [{"message": {
                    "content": answer, "reasoning_content": "thought about tags"}}]},
            )

        monkeypatch.setattr(llm_core.httpx, "post", fake_post)
        try:
            out = llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert out == answer
        finally:
            _cleanup_endpoint("rc-norm-oob")
            llm_core._response_cache.clear()

    def test_parserful_serve_literal_closer_not_repaired_async(self, monkeypatch):
        url = "http://rc-norm-oob-async.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-norm-oob-async")
        llm_core._response_cache.clear()
        answer = "The closing tag is written `</think>` in that template."

        async def fake_post(client, u, headers, **kwargs):
            return httpx.Response(
                200, request=httpx.Request("POST", u),
                json={"choices": [{"message": {
                    "content": answer, "reasoning_content": "thought about tags"}}]},
            )

        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", fake_post)
        try:
            out = asyncio.run(llm_core.llm_call_async(
                url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5))
            assert out == answer
        finally:
            _cleanup_endpoint("rc-norm-oob-async")
            llm_core._response_cache.clear()

    def test_clean_response_untouched(self, monkeypatch):
        url = "http://rc-norm-clean.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-norm-clean")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen, content="2 + 2 = 4."))
        try:
            out = llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert out == "2 + 2 = 4."
        finally:
            _cleanup_endpoint("rc-norm-clean")
            llm_core._response_cache.clear()


class TestEndpointIdSeam:
    def test_sync_call_with_endpoint_id_resolves_exact_row(self, monkeypatch):
        url = "http://rc-prep-ident.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "off"}, "rc-prep-ident-a")
        _seed_endpoint(url, {MODEL: "on"}, "rc-prep-ident-b")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen))
        try:
            llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hello"}],
                              max_tokens=5, endpoint_id="rc-prep-ident-b")
            assert _last_user_content(seen["json"]) == "/think hello"
            llm_core.llm_call(url, MODEL, [{"role": "user", "content": "hello"}], max_tokens=5)
            assert _last_user_content(seen["json"]) == "hello"
        finally:
            _cleanup_endpoint("rc-prep-ident-a")
            _cleanup_endpoint("rc-prep-ident-b")
            llm_core._response_cache.clear()

    def test_async_call_with_endpoint_id_resolves_exact_row(self, monkeypatch):
        url = "http://rc-prep-ident-async.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "off"}, "rc-ida-a")
        _seed_endpoint(url, {MODEL: "on"}, "rc-ida-b")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", _fake_async_post(seen))
        try:
            asyncio.run(llm_core.llm_call_async(
                url, MODEL, [{"role": "user", "content": "hello"}],
                max_tokens=5, endpoint_id="rc-ida-b"))
            assert _last_user_content(seen["json"]) == "/think hello"
        finally:
            _cleanup_endpoint("rc-ida-a")
            _cleanup_endpoint("rc-ida-b")
            llm_core._response_cache.clear()


class TestRepairArm:
    def test_prep_reports_message_directive_mechanism(self):
        """prep returns is_message_directive so the response repair arms on
        mechanism (a default-on model reasons in auto with nothing injected)."""
        url = "http://rc-arm-test.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-arm-test")
        try:
            _, is_md, _ = llm_core._prepare_llm_messages(
                [{"role": "user", "content": "hello"}], MODEL, url)
            assert is_md is True
            # system-only: nothing injected, but still a message-directive model
            prepared, is_md2, _ = llm_core._prepare_llm_messages(
                [{"role": "system", "content": "sys only"}], MODEL, url)
            assert is_md2 is True
            assert all("/think" not in str(m.get("content")) for m in prepared)
        finally:
            _cleanup_endpoint("rc-arm-test")

    def test_template_kwarg_model_does_not_arm(self):
        url = "http://rc-arm-tk.invalid:9911/v1"
        _seed_endpoint(url, {GLM: "on"}, "rc-arm-tk", model=GLM, control=TK_CONTROL)
        try:
            _, is_md, frag = llm_core._prepare_llm_messages(
                [{"role": "user", "content": "hi"}], GLM, url)
            assert is_md is False
            assert frag == {"chat_template_kwargs": {"enable_thinking": True}}
        finally:
            _cleanup_endpoint("rc-arm-tk")

    def test_unconfigured_model_does_not_arm(self):
        url = "http://rc-arm-none.invalid:9911/v1"
        _seed_endpoint(url, None, "rc-arm-none")  # only MODEL configured
        try:
            _, is_md, frag = llm_core._prepare_llm_messages(
                [{"role": "user", "content": "hi"}], "gpt-4o", url)
            assert is_md is False and frag is None
        finally:
            _cleanup_endpoint("rc-arm-none")


class TestTemplateKwargControl:
    """Template-kwarg mechanism: the preference becomes a body fragment —
    bidirectional, keyed into the cache, absent for auto/unconfigured models."""

    def _seed_glm(self, url, modes, ident):
        _seed_endpoint(url, modes, ident, model=GLM, control=TK_CONTROL)

    def test_sync_off_sends_enable_thinking_false(self, monkeypatch):
        url = "http://rc-kwarg-sync.invalid:9911/v1"
        self._seed_glm(url, {GLM: "off"}, "rc-kwarg-sync")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(seen))
        try:
            llm_core.llm_call(url, GLM, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": False}
            assert _last_user_content(seen["json"]) == "hi"
        finally:
            _cleanup_endpoint("rc-kwarg-sync")
            llm_core._response_cache.clear()

    def test_async_on_sends_enable_thinking_true(self, monkeypatch):
        url = "http://rc-kwarg-async.invalid:9911/v1"
        self._seed_glm(url, {GLM: "on"}, "rc-kwarg-async")
        llm_core._response_cache.clear()
        seen = {}
        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", _fake_async_post(seen))
        try:
            asyncio.run(llm_core.llm_call_async(url, GLM, [{"role": "user", "content": "hi"}], max_tokens=5))
            assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": True}
            assert _last_user_content(seen["json"]) == "hi"
        finally:
            _cleanup_endpoint("rc-kwarg-async")
            llm_core._response_cache.clear()

    def test_stream_off_sends_enable_thinking_false(self, monkeypatch):
        url = "http://rc-kwarg-stream.invalid:9911/v1"
        self._seed_glm(url, {GLM: "off"}, "rc-kwarg-stream")
        seen = {}
        monkeypatch.setattr(llm_core, "_get_http_client", lambda: _FakeStreamClient(seen))
        try:
            asyncio.run(_collect(llm_core.stream_llm(url, GLM, [{"role": "user", "content": "hi"}], max_tokens=5)))
            assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": False}
        finally:
            _cleanup_endpoint("rc-kwarg-stream")

    def test_auto_and_unconfigured_send_no_kwarg(self, monkeypatch):
        url = "http://rc-kwarg-auto.invalid:9911/v1"
        self._seed_glm(url, None, "rc-kwarg-auto")  # auto
        llm_core._response_cache.clear()
        glm_seen, other_seen = {}, {}
        try:
            monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(glm_seen))
            llm_core.llm_call(url, GLM, [{"role": "user", "content": "hi"}], max_tokens=5)
            assert "chat_template_kwargs" not in glm_seen["json"]
            monkeypatch.setattr(llm_core.httpx, "post", _fake_sync_post(other_seen))
            llm_core.llm_call(url, "gpt-4o", [{"role": "user", "content": "hi"}], max_tokens=5)
            assert "chat_template_kwargs" not in other_seen["json"]
        finally:
            _cleanup_endpoint("rc-kwarg-auto")
            llm_core._response_cache.clear()

    def test_mode_flip_changes_cache_key(self, monkeypatch):
        url = "http://rc-kwarg-cache.invalid:9911/v1"
        self._seed_glm(url, {GLM: "off"}, "rc-kwarg-cache")
        llm_core._response_cache.clear()
        calls = []

        def fake_post(u, headers=None, json=None, timeout=None):
            calls.append(json)
            body = "A" if len(calls) == 1 else "B"
            return httpx.Response(200, request=httpx.Request("POST", u),
                                  json={"choices": [{"message": {"content": body}}]})

        monkeypatch.setattr(llm_core.httpx, "post", fake_post)
        msgs = [{"role": "user", "content": "explain X"}]
        try:
            assert llm_core.llm_call(url, GLM, msgs, max_tokens=5) == "A"
            assert llm_core.llm_call(url, GLM, msgs, max_tokens=5) == "A"
            assert len(calls) == 1
            self._seed_glm(url, {GLM: "on"}, "rc-kwarg-cache")
            assert llm_core.llm_call(url, GLM, msgs, max_tokens=5) == "B"
            assert len(calls) == 2
            assert calls[1]["chat_template_kwargs"] == {"enable_thinking": True}
            self._seed_glm(url, {GLM: "off"}, "rc-kwarg-cache")
            assert llm_core.llm_call(url, GLM, msgs, max_tokens=5) == "A"
            assert len(calls) == 2
        finally:
            _cleanup_endpoint("rc-kwarg-cache")
            llm_core._response_cache.clear()

    def test_mode_flip_changes_cache_key_async(self, monkeypatch):
        url = "http://rc-kwarg-cache-async.invalid:9911/v1"
        self._seed_glm(url, {GLM: "off"}, "rc-kwarg-cache-async")
        llm_core._response_cache.clear()
        calls = []

        async def fake_post(client, u, headers, **kwargs):
            calls.append(kwargs.get("json"))
            body = "A" if len(calls) == 1 else "B"
            return httpx.Response(200, request=httpx.Request("POST", u),
                                  json={"choices": [{"message": {"content": body}}]})

        monkeypatch.setattr(llm_core, "httpx_post_kimi_aware_async", fake_post)
        msgs = [{"role": "user", "content": "explain X"}]
        try:
            assert asyncio.run(llm_core.llm_call_async(url, GLM, msgs, max_tokens=5)) == "A"
            assert asyncio.run(llm_core.llm_call_async(url, GLM, msgs, max_tokens=5)) == "A"
            assert len(calls) == 1
            self._seed_glm(url, {GLM: "on"}, "rc-kwarg-cache-async")
            assert asyncio.run(llm_core.llm_call_async(url, GLM, msgs, max_tokens=5)) == "B"
            assert len(calls) == 2
            assert calls[1]["chat_template_kwargs"] == {"enable_thinking": True}
            self._seed_glm(url, {GLM: "off"}, "rc-kwarg-cache-async")
            assert asyncio.run(llm_core.llm_call_async(url, GLM, msgs, max_tokens=5)) == "A"
            assert len(calls) == 2
        finally:
            _cleanup_endpoint("rc-kwarg-cache-async")
            llm_core._response_cache.clear()

    def test_fragment_preserves_existing_outer_keys(self):
        payload = {"chat_template_kwargs": {"other": 1}, "model": "m"}
        llm_core._apply_reasoning_fragment(payload, {"chat_template_kwargs": {"enable_thinking": False}})
        assert payload["chat_template_kwargs"] == {"other": 1, "enable_thinking": False}


class TestFragmentMergeDefensiveBranches:
    def test_non_dict_existing_outer_value_replaced_not_crashed(self):
        payload = {"chat_template_kwargs": "upstream-set-a-string", "model": "m"}
        llm_core._apply_reasoning_fragment(payload, {"chat_template_kwargs": {"enable_thinking": True}})
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}

    def test_scalar_fragment_sets_top_level_key(self):
        payload = {"model": "m"}
        llm_core._apply_reasoning_fragment(payload, {"think": False})
        assert payload["think"] is False


class TestCallerImmutability:
    def test_prep_never_mutates_caller_messages(self):
        import copy
        url = "http://rc-prep-immut.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-prep-immut")
        try:
            msgs = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "look"}]},
                {"role": "assistant", "content": "x"},
                {"role": "user", "content": "hello"},
            ]
            snapshot = copy.deepcopy(msgs)
            prepared, _, _ = llm_core._prepare_llm_messages(msgs, MODEL, url)
            assert _last_user_content({"messages": prepared}) == "/think hello"
            assert msgs == snapshot

            multimodal_last = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
            snapshot2 = copy.deepcopy(multimodal_last)
            prepared2, _, _ = llm_core._prepare_llm_messages(multimodal_last, MODEL, url)
            assert prepared2[-1]["content"][0] == {"type": "text", "text": "/think"}
            assert multimodal_last == snapshot2
        finally:
            _cleanup_endpoint("rc-prep-immut")


class TestDoublePrep:
    def test_reprepare_is_idempotent(self):
        url = "http://rc-prep-idem.invalid:9911/v1"
        _seed_endpoint(url, {MODEL: "on"}, "rc-prep-idem")
        try:
            msgs = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"}]
            once, is_md1, _ = llm_core._prepare_llm_messages(msgs, MODEL, url)
            twice, is_md2, _ = llm_core._prepare_llm_messages(once, MODEL, url)
            assert twice == once
            assert is_md1 is is_md2 is True
            assert _last_user_content({"messages": twice}) == "/think hello"
            assert sum(1 for m in twice if m.get("role") == "system") == 1
        finally:
            _cleanup_endpoint("rc-prep-idem")
