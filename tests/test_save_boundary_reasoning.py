"""Save-boundary repair of the parser-less orphan-`</think>` reasoning shape.

The repair lives in _extract_thinking_meta and is scoped to models whose
declared control mechanism is the message directive (they produce the
orphan-closer shape). It is also OPT-IN per save path:
- save_assistant_response (streaming/agent saves) threads the model +
  endpoint — its content arrives raw from the stream and needs the repair.
- clean_thinking_for_save repairs only when the caller passes repair_model +
  repair_endpoint_url (stopped-mid-stream partials). Non-streaming callers
  pass nothing: their content was already adjudicated by
  llm_core._finalize_llm_response, and repairing it again would fold a
  parser-ful serve's literal "</think>" answer text as thinking.
"""
import json

from routes.chat_helpers import clean_thinking_for_save, _extract_thinking_meta, save_assistant_response
from src.model_capabilities import (
    REASONING_CONTROL_MESSAGE_DIRECTIVE as MD,
    REASONING_CONTROL_TEMPLATE_KWARG as TK,
)

NEMO = "nemotron-nano-12b-vl"
GLM = "glm-4.5-air"
URL = "http://rc-save.invalid:9911/v1"
SHAPE_B = ('Okay, so the user asked for a joke. Something short and groan-worthy '
           'works best here.\n</think>\n\nWhy did the scarecrow win an award?\n')


def _seed():
    from core.database import SessionLocal, ModelEndpoint, Base, engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == "rc-save").delete(synchronize_session=False)
        db.add(ModelEndpoint(
            id="rc-save", name="save", base_url=URL, is_enabled=True,
            cached_models=json.dumps([NEMO, GLM]),
            reasoning_controls=json.dumps({
                NEMO: {"mechanism": MD, "values": {"on": "/think"}},
                GLM: {"mechanism": TK, "kwarg_path": ["chat_template_kwargs", "enable_thinking"],
                      "values": {"on": True, "off": False}},
            })))
        db.commit()
    finally:
        db.close()


def _cleanup():
    from core.database import SessionLocal, ModelEndpoint
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == "rc-save").delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


URL_B = "http://rc-fallback.invalid:9911/v1"  # a SECOND endpoint (fallback / different backend)


def _seed_fb():
    """A separate endpoint at URL_B whose NEMO spec is the message directive —
    models the fallback-answered-on-a-different-endpoint case."""
    from core.database import SessionLocal, ModelEndpoint, Base, engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == "rc-fb").delete(synchronize_session=False)
        db.add(ModelEndpoint(id="rc-fb", name="fb", base_url=URL_B, is_enabled=True,
                             cached_models=json.dumps([NEMO]),
                             reasoning_controls=json.dumps({NEMO: {"mechanism": MD, "values": {"on": "/think"}}})))
        db.commit()
    finally:
        db.close()


def _cleanup_fb():
    from core.database import SessionLocal, ModelEndpoint
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == "rc-fb").delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


class _StubSess:
    def __init__(self, model=NEMO, endpoint_url=URL):
        self.model = model
        self.endpoint_url = endpoint_url
        self.history = []

    def add_message(self, msg):
        self.history.append(msg)


class TestStreamingSavePath:
    """Through the REAL streaming-save caller (save_assistant_response threads
    the endpoint; dropping either the model or endpoint arg must fail)."""

    def test_streaming_save_repairs_orphan_closer(self):
        _seed()
        try:
            sess = _StubSess()
            save_assistant_response(sess, None, "s1", SHAPE_B, {"model": NEMO}, incognito=True)
            msg = sess.history[-1]
            assert msg.content.startswith("Why did the scarecrow")
            assert "user asked" not in msg.content
            assert "user asked" in (msg.metadata or {}).get("thinking", "")
        finally:
            _cleanup()

    def test_streaming_save_skips_repair_when_reasoning_out_of_band(self):
        """A message-directive model on a PARSER-FUL serve streams reasoning
        out-of-band (md['thinking'] set), so its content is clean answer text.
        A clean answer legitimately containing a bare '</think>' must NOT be
        repaired/folded — and the real out-of-band reasoning must survive."""
        _seed()
        try:
            sess = _StubSess()
            clean = "To hide reasoning, models emit </think> to close the block. Put it after your thoughts."
            oob = "The user is asking how think blocks are written."
            save_assistant_response(sess, None, "s1", clean,
                                    {"model": NEMO, "thinking": oob}, incognito=True)
            msg = sess.history[-1]
            assert msg.content == clean          # not truncated at the literal </think>
            assert (msg.metadata or {}).get("thinking") == oob  # out-of-band reasoning preserved
        finally:
            _cleanup()

    def test_streaming_save_untouched_for_unconfigured_model(self):
        _seed()
        try:
            sess = _StubSess(model="gpt-4o")
            save_assistant_response(sess, None, "s1", SHAPE_B, {"model": "gpt-4o"}, incognito=True)
            msg = sess.history[-1]
            assert msg.content == SHAPE_B
            assert "thinking" not in (msg.metadata or {})
        finally:
            _cleanup()

    def test_template_kwarg_model_literal_closer_not_repaired(self):
        """A template-kwarg model (out-of-band reasoning) quoting the closer
        must not be repaired even on the streaming save path — the repair is
        scoped to the message-directive mechanism."""
        _seed()
        try:
            answer = "In that template the closing tag is written `</think>` verbatim."
            sess = _StubSess(model=GLM)
            save_assistant_response(sess, None, "s1", answer, {"model": GLM}, incognito=True)
            msg = sess.history[-1]
            assert msg.content == answer
            assert "thinking" not in (msg.metadata or {})
        finally:
            _cleanup()


class TestStoppedPartialSaves:
    """clean_thinking_for_save with explicit repair_model + repair_endpoint_url
    — the stopped-mid-stream call sites' contract."""

    def test_repair_opts_in(self):
        _seed()
        try:
            reply, md = clean_thinking_for_save(SHAPE_B, {"model": NEMO},
                                                repair_model=NEMO, repair_endpoint_url=URL)
            assert reply.startswith("Why did the scarecrow")
            assert "user asked" in md.get("thinking", "")
        finally:
            _cleanup()

    def test_stopped_partial_skips_repair_when_reasoning_streamed(self):
        """Stopped-partial save on a parser-ful serve: reasoning already streamed
        out-of-band, so reasoning_streamed=True suppresses the repair and a
        literal '</think>' in the clean partial is left verbatim."""
        _seed()
        try:
            clean = "Write </think> to end the block."
            reply, md = clean_thinking_for_save(clean, {"model": NEMO},
                                                repair_model=NEMO, repair_endpoint_url=URL,
                                                reasoning_streamed=True)
            assert reply == clean
            assert "thinking" not in md
        finally:
            _cleanup()

    def test_repair_unconfigured_untouched(self):
        _seed()
        try:
            reply, md = clean_thinking_for_save(SHAPE_B, {"model": "gpt-4o"},
                                                repair_model="gpt-4o", repair_endpoint_url=URL)
            assert reply == SHAPE_B
            assert "thinking" not in md
        finally:
            _cleanup()


class TestNonStreamingSavePath:
    """clean_thinking_for_save WITHOUT repair_model — the non-streaming routes'
    contract: metadata model alone must NOT trigger the repair (llm_core
    already adjudicated with request-context gates)."""

    def test_metadata_model_alone_does_not_repair(self):
        _seed()
        try:
            answer = "The closing tag is written `</think>` in that template."
            reply, md = clean_thinking_for_save(answer, {"model": NEMO})
            assert reply == answer
            assert "thinking" not in md
        finally:
            _cleanup()

    def test_well_formed_block_still_extracted_without_repair_model(self):
        text = "<think>brief thought</think>\n\nThe answer.\n"
        reply, md = clean_thinking_for_save(text, {"model": NEMO})
        assert reply == "The answer."
        assert md.get("thinking") == "brief thought"


class TestExtractorMechanism:
    def test_clean_content_untouched(self):
        _seed()
        try:
            reply, md = clean_thinking_for_save("The answer.", {"model": NEMO},
                                                repair_model=NEMO, repair_endpoint_url=URL)
            assert reply == "The answer."
            assert "thinking" not in md
        finally:
            _cleanup()

    def test_truncated_reasoning_only_keeps_raw_content(self):
        _seed()
        try:
            truncated = "Okay, the user wants a short title for a schedul"
            reply, md = clean_thinking_for_save(truncated, {"model": NEMO},
                                                repair_model=NEMO, repair_endpoint_url=URL)
            assert reply == truncated
            assert "thinking" not in md
        finally:
            _cleanup()

    def test_extractor_gate_needs_endpoint(self):
        _seed()
        try:
            # no endpoint_url -> can't resolve the mechanism -> no repair
            assert _extract_thinking_meta(SHAPE_B, model=NEMO) is None
            info = _extract_thinking_meta(SHAPE_B, model=NEMO, endpoint_url=URL)
            assert info and info["reply"].startswith("Why did the scarecrow")
        finally:
            _cleanup()


class TestFallbackAndRenameRepairIdentity:
    """The repair must resolve the spec by the identity that ACTUALLY answered
    (configured id + answering endpoint), not the served name or the primary
    session URL — else a parser-less message-directive model's reasoning leaks
    into saved content on the server-rename and cross-endpoint fallback paths."""

    def test_fallback_endpoint_url_unblocks_repair(self):
        # NEMO's message-directive spec lives on URL_B (a fallback endpoint).
        _seed(); _seed_fb()
        try:
            # keyed on the ANSWERING url -> repaired
            reply, md = clean_thinking_for_save(
                SHAPE_B, {"model": NEMO}, repair_model=NEMO,
                repair_endpoint_url=URL_B, repair_requested_model=NEMO)
            assert reply.startswith("Why did the scarecrow")
            assert "user asked" in md.get("thinking", "")
            # regression (the old bug): keyed on a primary url with NO NEMO spec -> NOT repaired
            reply2, md2 = clean_thinking_for_save(
                SHAPE_B, {"model": NEMO}, repair_model=NEMO,
                repair_endpoint_url="http://rc-primary-nospec.invalid/v1")
            assert reply2 == SHAPE_B
            assert "thinking" not in md2
        finally:
            _cleanup(); _cleanup_fb()

    def test_server_rename_repairs_via_requested_model(self):
        # Serve echoes a different id than the configured NEMO the spec is keyed by.
        _seed()
        try:
            renamed = "Nemotron-Nano-12B-v2-VL-SERVED"
            reply, md = clean_thinking_for_save(
                SHAPE_B, {"model": renamed}, repair_model=renamed,
                repair_endpoint_url=URL, repair_requested_model=NEMO)
            assert reply.startswith("Why did the scarecrow")
            assert "user asked" in md.get("thinking", "")
            # regression: without the configured id the served name misses -> no repair
            reply2, md2 = clean_thinking_for_save(
                SHAPE_B, {"model": renamed}, repair_model=renamed, repair_endpoint_url=URL)
            assert reply2 == SHAPE_B
            assert "thinking" not in md2
        finally:
            _cleanup()

    def test_save_assistant_response_endpoint_url_override(self):
        # sess primary URL has no NEMO spec; the answering endpoint (URL_B) does.
        _seed(); _seed_fb()
        try:
            sess = _StubSess(model="primary-model", endpoint_url="http://rc-primary-nospec.invalid/v1")
            save_assistant_response(sess, None, "s1", SHAPE_B,
                                    {"model": NEMO, "requested_model": NEMO}, incognito=True,
                                    endpoint_url=URL_B)
            msg = sess.history[-1]
            assert msg.content.startswith("Why did the scarecrow")
            assert "user asked" in (msg.metadata or {}).get("thinking", "")
        finally:
            _cleanup(); _cleanup_fb()
