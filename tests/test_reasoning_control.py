"""Unit tests for user-specified, model-agnostic reasoning control
(src/reasoning_control.py).

No model is toggleable by default; a per-model control spec (stored in
ModelEndpoint.reasoning_controls) declares the mechanism, and the preference
(reasoning_modes) selects on/off. Nothing is inferred from model names.
"""
import json

from src.reasoning_control import (
    inject_directive, reasoning_mode_for, validate_control_spec,
    control_evidence_for, resolve_reasoning_controls, is_message_directive_model,
    ON, OFF, AUTO,
)
from src.model_capabilities import (
    REASONING_CONTROL_MESSAGE_DIRECTIVE as MD,
    REASONING_CONTROL_TEMPLATE_KWARG as TK,
)

MD_ON = {"mechanism": MD, "values": {"on": "/think"}}          # default-off model
MD_OFF = {"mechanism": MD, "values": {"off": "/no_think"}}     # default-on model
MD_BOTH = {"mechanism": MD, "values": {"on": "/think", "off": "/no_think"}}
TK_BOTH = {"mechanism": TK, "kwarg_path": ["chat_template_kwargs", "enable_thinking"],
           "values": {"on": True, "off": False}}


def _seed(ident, url, *, controls=None, modes=None, cached=None, hidden=None, owner=None):
    from core.database import SessionLocal, ModelEndpoint, Base, engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == ident).delete(synchronize_session=False)
        db.add(ModelEndpoint(id=ident, name=ident, base_url=url, is_enabled=True,
                             owner=owner,
                             cached_models=json.dumps(cached or []),
                             hidden_models=json.dumps(hidden) if hidden else None,
                             reasoning_controls=json.dumps(controls) if controls else None,
                             reasoning_modes=json.dumps(modes) if modes else None))
        db.commit()
    finally:
        db.close()


def _cleanup(*idents):
    from core.database import SessionLocal, ModelEndpoint
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id.in_(idents)).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


class TestValidateControlSpec:
    """The API write path: only well-formed, implemented mechanisms are stored;
    everything else is rejected with a reason (not silently dropped)."""

    def test_valid_message_directive(self):
        norm, err = validate_control_spec(MD_ON)
        assert err is None and norm == {"mechanism": MD, "values": {"on": "/think"}}

    def test_valid_template_kwarg(self):
        norm, err = validate_control_spec(TK_BOTH)
        assert err is None
        assert norm["mechanism"] == TK
        assert norm["kwarg_path"] == ["chat_template_kwargs", "enable_thinking"]
        assert norm["values"] == {"on": True, "off": False}

    def test_reject_unknown_mechanism(self):
        norm, err = validate_control_spec({"mechanism": "reasoning_budget", "values": {"off": 0}})
        assert norm is None and "mechanism must be one of" in err

    def test_reject_non_dict(self):
        assert validate_control_spec("/think")[0] is None
        assert validate_control_spec(None)[0] is None

    def test_reject_empty_or_bad_values(self):
        assert validate_control_spec({"mechanism": MD, "values": {}})[0] is None
        assert validate_control_spec({"mechanism": MD, "values": {"maybe": "/think"}})[0] is None
        assert validate_control_spec({"mechanism": MD, "values": ["on"]})[0] is None

    def test_reject_unhashable_mechanism(self):
        # A JSON list/dict mechanism must not crash the `not in frozenset` check
        # (TypeError -> 500 on the write path); it returns a clean error instead.
        for bad in ([], {}, {"a": 1}, ["x"]):
            norm, err = validate_control_spec({"mechanism": bad, "values": {"on": "/think"}})
            assert norm is None and "mechanism must be one of" in err

    def test_reject_message_directive_non_string_value(self):
        norm, err = validate_control_spec({"mechanism": MD, "values": {"on": True}})
        assert norm is None and "whitespace-free token" in err

    def test_reject_message_directive_bad_tokens(self):
        # A message directive must be a single whitespace-free token: empty,
        # whitespace-only, multi-word, and edge-padded all break the
        # split()-based idempotency guard and are rejected.
        for bad in ("", "   ", "detailed thinking on", "/think ", " /think", "a\tb"):
            norm, err = validate_control_spec({"mechanism": MD, "values": {"on": bad}})
            assert norm is None and "whitespace-free token" in err, f"accepted {bad!r}"
        # a genuine single token still passes
        assert validate_control_spec({"mechanism": MD, "values": {"on": "/think"}})[0] is not None

    def test_reject_template_kwarg_missing_or_bad_path(self):
        assert validate_control_spec({"mechanism": TK, "values": {"on": True}})[0] is None
        assert validate_control_spec({"mechanism": TK, "kwarg_path": ["only-one"],
                                      "values": {"on": True}})[0] is None
        assert validate_control_spec({"mechanism": TK, "kwarg_path": ["a", ""],
                                      "values": {"on": True}})[0] is None

    def test_reject_template_kwarg_container_value(self):
        norm, err = validate_control_spec(
            {"mechanism": TK, "kwarg_path": ["a", "b"], "values": {"on": {"nested": 1}}})
        assert norm is None and "scalar" in err
        assert validate_control_spec(
            {"mechanism": TK, "kwarg_path": ["a", "b"], "values": {"on": [1, 2]}})[0] is None

    def test_reject_template_kwarg_null_value(self):
        # A null payload validates as a scalar-shaped value but would dispatch
        # identically to an absent direction (resolve treats None as "inject
        # nothing"), silently dropping the control — so reject it at write time.
        norm, err = validate_control_spec(
            {"mechanism": TK, "kwarg_path": ["chat_template_kwargs", "enable_thinking"],
             "values": {"off": None}})
        assert norm is None and "scalar" in err

    def test_reject_template_kwarg_reserved_outer_key(self):
        # An outer key that is a core request field would overwrite it in the
        # payload (messages/model/stream/...) and corrupt the call — reject it.
        for outer in ("messages", "model", "stream", "temperature", "tools"):
            norm, err = validate_control_spec(
                {"mechanism": TK, "kwarg_path": [outer, "x"], "values": {"on": True}})
            assert norm is None and "core request field" in err, f"accepted outer={outer!r}"
        # a normal nesting key is fine
        assert validate_control_spec(
            {"mechanism": TK, "kwarg_path": ["chat_template_kwargs", "enable_thinking"],
             "values": {"on": True}})[0] is not None

    def test_reject_template_kwarg_nonfinite_float(self):
        # NaN/Infinity are floats but not valid JSON — reject so they never
        # reach the wire.
        for bad in (float("nan"), float("inf"), float("-inf")):
            norm, err = validate_control_spec(
                {"mechanism": TK, "kwarg_path": ["chat_template_kwargs", "x"], "values": {"on": bad}})
            assert norm is None and "finite" in err, f"accepted {bad}"


class TestControlEvidenceFromStore:
    """control_evidence_for reads the per-model spec off the endpoint; an
    unconfigured model has no evidence (= not toggleable = the default)."""

    URL = "http://rc-ev.invalid:9911/v1"

    def test_configured_model_returns_normalized_spec(self):
        _seed("rc-ev", self.URL, controls={"m-md": MD_ON, "m-tk": TK_BOTH},
              cached=["m-md", "m-tk"])
        try:
            assert control_evidence_for("m-md", self.URL)["mechanism"] == MD
            assert control_evidence_for("m-tk", self.URL)["mechanism"] == TK
        finally:
            _cleanup("rc-ev")

    def test_unconfigured_model_has_no_evidence(self):
        _seed("rc-ev2", self.URL, controls={"m-md": MD_ON}, cached=["m-md", "other"])
        try:
            assert control_evidence_for("other", self.URL) is None
            assert control_evidence_for("m-md", "http://nope.invalid/v1") is None  # no endpoint
        finally:
            _cleanup("rc-ev2")

    def test_corrupt_stored_spec_degrades_to_none(self):
        # a hand-edited/garbage spec must not crash the read path
        _seed("rc-ev3", self.URL, controls={"m": {"mechanism": "bogus"}}, cached=["m"])
        try:
            assert control_evidence_for("m", self.URL) is None
        finally:
            _cleanup("rc-ev3")


class TestResolveWithSpecs:
    """Uniform dispatch: value-map lookup per mechanism; auto and undeclared
    directions inject nothing; is_message_directive tracks the mechanism."""

    URL = "http://rc-res.invalid:9911/v1"

    def test_message_directive_default_off_model(self):
        _seed("rc-res", self.URL, controls={"nemo": MD_ON}, cached=["nemo"],
              modes={"nemo": "on"})
        try:
            assert resolve_reasoning_controls("nemo", self.URL) == ("/think", None, True)
        finally:
            _cleanup("rc-res")

    def test_message_directive_default_on_model_no_think(self):
        _seed("rc-res", self.URL, controls={"qwen": MD_OFF}, cached=["qwen"],
              modes={"qwen": "off"})
        try:
            assert resolve_reasoning_controls("qwen", self.URL) == ("/no_think", None, True)
        finally:
            _cleanup("rc-res")

    def test_message_directive_undeclared_direction_injects_nothing_but_still_md(self):
        # default-on model in auto: nothing injected, but it IS a message-
        # directive model (reasons by default -> repair must still arm).
        _seed("rc-res", self.URL, controls={"qwen": MD_OFF}, cached=["qwen"])  # auto
        try:
            assert resolve_reasoning_controls("qwen", self.URL) == (None, None, True)
        finally:
            _cleanup("rc-res")

    def test_message_directive_bidirectional(self):
        _seed("rc-res", self.URL, controls={"m": MD_BOTH}, cached=["m"], modes={"m": "on"})
        try:
            assert resolve_reasoning_controls("m", self.URL) == ("/think", None, True)
        finally:
            _cleanup("rc-res")
        _seed("rc-res", self.URL, controls={"m": MD_BOTH}, cached=["m"], modes={"m": "off"})
        try:
            assert resolve_reasoning_controls("m", self.URL) == ("/no_think", None, True)
        finally:
            _cleanup("rc-res")

    def test_template_kwarg_both_directions(self):
        _seed("rc-res", self.URL, controls={"glm": TK_BOTH}, cached=["glm"], modes={"glm": "on"})
        try:
            assert resolve_reasoning_controls("glm", self.URL) == (
                None, {"chat_template_kwargs": {"enable_thinking": True}}, False)
        finally:
            _cleanup("rc-res")
        _seed("rc-res", self.URL, controls={"glm": TK_BOTH}, cached=["glm"], modes={"glm": "off"})
        try:
            assert resolve_reasoning_controls("glm", self.URL) == (
                None, {"chat_template_kwargs": {"enable_thinking": False}}, False)
        finally:
            _cleanup("rc-res")

    def test_template_kwarg_auto_sends_nothing(self):
        _seed("rc-res", self.URL, controls={"glm": TK_BOTH}, cached=["glm"])  # auto
        try:
            assert resolve_reasoning_controls("glm", self.URL) == (None, None, False)
        finally:
            _cleanup("rc-res")

    def test_kwarg_path_is_read_from_the_spec(self):
        spec = {"mechanism": TK, "kwarg_path": ["extra_body", "enable_reasoning"],
                "values": {"on": True}}
        _seed("rc-res", self.URL, controls={"m": spec}, cached=["m"], modes={"m": "on"})
        try:
            assert resolve_reasoning_controls("m", self.URL) == (
                None, {"extra_body": {"enable_reasoning": True}}, False)
        finally:
            _cleanup("rc-res")

    def test_unconfigured_model_is_not_toggleable_and_skips_mode_lookup(self, monkeypatch):
        # default: no control spec -> (None, None, False), and the cheap gate
        # means the preference lookup is never even reached.
        import src.reasoning_control as rc

        def bomb(*a, **k):
            raise AssertionError("reasoning_mode_for must not run for an unconfigured model")

        _seed("rc-res", self.URL, cached=["m"], modes={"m": "on"})  # modes but NO controls
        monkeypatch.setattr(rc, "reasoning_mode_for", bomb)
        try:
            assert rc.resolve_reasoning_controls("m", self.URL) == (None, None, False)
        finally:
            _cleanup("rc-res")


class TestIsMessageDirectiveModel:
    URL = "http://rc-md.invalid:9911/v1"

    def test_true_only_for_message_directive_specs(self):
        _seed("rc-md", self.URL, controls={"nemo": MD_ON, "glm": TK_BOTH},
              cached=["nemo", "glm"])
        try:
            assert is_message_directive_model("nemo", self.URL) is True
            assert is_message_directive_model("glm", self.URL) is False
            assert is_message_directive_model("unconfigured", self.URL) is False
        finally:
            _cleanup("rc-md")


class TestInjectDirective:
    def test_string_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"] == "/think hello"

    def test_multimodal_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"][0] == {"type": "text", "text": "/think"}

    def test_idempotent(self):
        msgs = [{"role": "user", "content": "/think hello"}]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"] == "/think hello"

    def test_targets_latest_user_turn(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "second"},
        ]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"] == "first"
        assert msgs[2]["content"] == "/think second"

    def test_no_user_message_returns_false(self):
        assert inject_directive([{"role": "system", "content": "sys"}], "/think") is False

    # exact-token idempotency: a lookalike substring must NOT suppress injection.
    def test_lookalike_substring_still_injects_string(self):
        msgs = [{"role": "user", "content": "/thinker is a great word"}]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"] == "/think /thinker is a great word"

    def test_lookalike_substring_still_injects_multimodal(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "my /thinker note"}]}]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"][0] == {"type": "text", "text": "/think"}
        assert msgs[0]["content"][1] == {"type": "text", "text": "my /thinker note"}

    def test_token_mid_message_is_idempotent_string(self):
        msgs = [{"role": "user", "content": "please /think about it"}]
        inject_directive(msgs, "/think")
        assert msgs[0]["content"] == "please /think about it"

    def test_no_think_directive_injects(self):
        msgs = [{"role": "user", "content": "hi"}]
        inject_directive(msgs, "/no_think")
        assert msgs[0]["content"] == "/no_think hi"


class TestNormalizeReasoningResponse:
    """Response-side repair of the parser-less orphan-`</think>` shape."""

    SHAPE_B = ('Okay, so the user asked "What is 2+2?" and wants a brief '
               'answer. So just state 4.\n</think>\n\n2 + 2 equals 4.\n')

    def test_orphan_closer_gets_opening_tag(self):
        from src.reasoning_control import normalize_reasoning_response
        assert normalize_reasoning_response(self.SHAPE_B) == "<think>" + self.SHAPE_B

    def test_well_formed_block_untouched(self):
        from src.reasoning_control import normalize_reasoning_response
        text = "<think>brief thought</think>\n\nanswer"
        assert normalize_reasoning_response(text) is text

    def test_reasoning_free_answer_untouched(self):
        from src.reasoning_control import normalize_reasoning_response
        for text in ("2 + 2 = 4.", "", "4"):
            assert normalize_reasoning_response(text) is text

    def test_non_string_untouched(self):
        from src.reasoning_control import normalize_reasoning_response
        assert normalize_reasoning_response(None) is None

    def test_repaired_block_is_consumable_downstream(self):
        from src.reasoning_control import normalize_reasoning_response
        from src.text_helpers import strip_think
        cleaned = strip_think(normalize_reasoning_response(self.SHAPE_B))
        assert "2 + 2 equals 4" in cleaned
        assert "user asked" not in cleaned

    def test_mentions_open_tag_before_orphan_closer_still_repaired(self):
        # Reasoning that merely MENTIONS "<think>" before its orphan closer must
        # still be repaired: it counts as well-formed only when the opening tag
        # is at the very start (what the downstream ^\s*<think> extractor needs).
        from src.reasoning_control import normalize_reasoning_response
        text = "Considering the <think> tag semantics, the reasoning follows</think>\n\nThe answer."
        assert normalize_reasoning_response(text) == "<think>" + text

    def test_mid_line_closer_is_answer_text_not_repaired(self):
        # A </think> sitting mid-line is answer text (not a reasoning terminator)
        # — repairing it would hide the first half of the answer as thinking.
        from src.reasoning_control import normalize_reasoning_response
        for text in ("To end the block write </think> after your thoughts.",
                     "The tag </think> closes it; place it last."):
            assert normalize_reasoning_response(text) == text

    def test_terminator_closer_repaired(self):
        # </think> followed by a newline OR end-of-string is a real orphan closer.
        from src.reasoning_control import normalize_reasoning_response
        assert normalize_reasoning_response("reasoning...</think>\n\nanswer").startswith("<think>")
        assert normalize_reasoning_response("reasoning only, truncated...</think>").startswith("<think>")

    def test_bom_prefixed_block_not_double_wrapped(self):
        # A well-formed block behind a BOM/zero-width mark is still well-formed.
        from src.reasoning_control import normalize_reasoning_response
        text = "\ufeff<think>t</think>\n\nans"
        assert normalize_reasoning_response(text) == text


class TestReasoningModeFor:
    def test_unknown_url_degrades_to_auto(self):
        assert reasoning_mode_for("some-model", "http://nonexistent.invalid:9/v1") == AUTO

    def test_error_path_never_logs_credentials(self, caplog, monkeypatch):
        """A lookup failure must not leak endpoint credentials into logs."""
        import logging
        import src.reasoning_control as rc

        url = "http://alice:s3cret-pw@host.invalid:9911/v1?api_key=qu3ry-key#frag-t0ken"

        def boom(model, endpoint_url, endpoint_id=None):
            raise RuntimeError(f"lookup failed [parameters: ('{endpoint_url}',)]")

        monkeypatch.setattr(rc, "_endpoint_for", boom)
        with caplog.at_level(logging.DEBUG, logger="src.reasoning_control"):
            assert rc.reasoning_mode_for("some-model", url) == AUTO

        rendered = " ".join(r.getMessage() for r in caplog.records)
        assert rendered, "the failure should still be logged"
        for secret in ("alice", "s3cret-pw", "api_key=qu3ry-key", "frag-t0ken"):
            assert secret not in rendered
        assert "host.invalid:9911" in rendered


class TestEndpointResolution:
    """When several endpoint rows share a base URL, the control/preference must
    be read from the row that actually serves the model (no-leak)."""

    def test_same_base_url_disambiguates_by_model(self):
        url = "http://shared-rc-test.invalid:9911/v1"
        _seed("rc-test-a", url, controls={"model-a": MD_ON}, cached=["model-a"], modes={"model-a": "on"})
        _seed("rc-test-b", url, controls={"model-b": MD_ON}, cached=["model-b"], modes={"model-b": "off"})
        try:
            assert reasoning_mode_for("model-a", url) == ON
            assert reasoning_mode_for("model-b", url) == OFF
            assert reasoning_mode_for("model-c", url) == AUTO
            # control evidence follows the same row disambiguation
            assert control_evidence_for("model-a", url)["mechanism"] == MD
            assert control_evidence_for("model-c", url) is None
        finally:
            _cleanup("rc-test-a", "rc-test-b")

    def test_lowercase_stored_key_fallback(self):
        url = "http://rc-case-test.invalid:9911/v1"
        mixed = "NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4"
        _seed("rc-case-test", url, controls={mixed.lower(): MD_ON}, cached=[mixed],
              modes={mixed.lower(): "on"})
        try:
            assert reasoning_mode_for(mixed, url) == ON
            assert control_evidence_for(mixed, url)["mechanism"] == MD
        finally:
            _cleanup("rc-case-test")

    def test_hidden_model_row_loses_disambiguation(self):
        url = "http://rc-hidden-test.invalid:9911/v1"
        _seed("rc-hid-a", url, controls={"model-h": MD_ON}, cached=["model-h"],
              hidden=["model-h"], modes={"model-h": "on"})
        _seed("rc-hid-b", url, controls={"model-h": MD_ON}, cached=["model-h"], modes={"model-h": "off"})
        try:
            assert reasoning_mode_for("model-h", url) == OFF
        finally:
            _cleanup("rc-hid-a", "rc-hid-b")

    def test_first_match_fallback_when_no_row_serves_model(self):
        url = "http://rc-fallback-test.invalid:9911/v1"
        _seed("rc-fb-a", url, cached=["other-x"], modes={"model-gone": "on"})
        _seed("rc-fb-b", url, cached=["other-y"])
        try:
            assert reasoning_mode_for("model-gone", url) == ON
        finally:
            _cleanup("rc-fb-a", "rc-fb-b")

    def test_endpoint_id_is_authoritative_over_url_resolution(self):
        url = "http://shared-rc-ident.invalid:9911/v1"
        _seed("rc-ident-a", url, cached=["model-x"], modes={"model-x": "off"})
        _seed("rc-ident-b", url, cached=["model-x"], modes={"model-x": "on"})
        try:
            assert reasoning_mode_for("model-x", url) == OFF
            assert reasoning_mode_for("model-x", url, endpoint_id="rc-ident-b") == ON
            assert reasoning_mode_for("model-x", url, endpoint_id="rc-ident-gone") == OFF
        finally:
            _cleanup("rc-ident-a", "rc-ident-b")

    def test_shared_row_wins_over_owner_row_deterministically(self):
        # Two owners hold rows at one base_url both serving the model. Reasoning
        # control is shared per (backend, model): resolution must be
        # deterministic — the shared (owner is None) row wins over an owner row,
        # regardless of seed/DB order — not per-owner and not DB-order roulette.
        url = "http://shared-rc-owner.invalid:9911/v1"
        # Seed the owner row FIRST (lower-in-DB) and the shared row second, so a
        # "first row wins" resolver would pick the owner row.
        _seed("rc-own-z", url, cached=["model-o"], modes={"model-o": "on"}, owner="alice")
        _seed("rc-own-a", url, cached=["model-o"], modes={"model-o": "off"}, owner=None)
        try:
            # shared row (rc-own-a, off) wins over the owner row (rc-own-z, on)
            assert reasoning_mode_for("model-o", url) == OFF
        finally:
            _cleanup("rc-own-z", "rc-own-a")
