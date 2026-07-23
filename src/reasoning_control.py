"""Per-model reasoning control — user-specified, model-agnostic.

Reasoning toggling is not a global setting: whether a model can be toggled,
and *how*, is a property of the specific served model, and it is **not
discoverable from any provider payload** (no /v1/models, /api/tags or /props
response advertises "honors /think" or "accepts enable_thinking"). So the
model→mechanism mapping has to be *stated* somewhere. This module stores it as
user-declared evidence, per model, on the endpoint — nothing is inferred from
model names, and no model is toggleable by default.

Two stores, both keyed by model id on ``ModelEndpoint``:
  • ``reasoning_controls`` — the *mechanism spec*: HOW this model toggles.
        {model_id: {"mechanism": ..., "values": {"on"|"off": <payload>}, ...}}
    Absent for a model  ⇒  not toggleable (the default). This is the "evidence"
    (source = endpoint_config / admin_override, the highest-confidence source
    in #2739's vocabulary).
  • ``reasoning_modes`` — the *preference*: on / off (absent = auto = leave the
    model's own default). Only meaningful when a control spec exists.

── The spec shape (uniform across mechanisms) ──
    {
      "mechanism": REASONING_CONTROL_MESSAGE_DIRECTIVE | ..._TEMPLATE_KWARG,
      "kwarg_path": ["outer", "inner"],          # template_kwarg only
      "values": {"on": <payload>, "off": <payload>}   # each direction OPTIONAL
    }
``values`` maps the user's intent (on/off) to the payload that achieves it;
each direction is optional — *present* means "explicit payload for this
direction", *absent* means "inject nothing / let the model do its default".
So a default-off model declares only its ``on`` control, a default-on model
only its ``off`` control, and a force-both model declares both.

  • message_directive payload = a directive string injected into the latest
    user turn (e.g. "/think" to turn on, "/no_think" to turn off).
  • template_kwarg payload = the value set at ``kwarg_path`` in the request
    body (e.g. true / false at chat_template_kwargs.enable_thinking).

Concrete examples:
    nemotron-VL (default-off):  {"mechanism": "…message_directive",
                                 "values": {"on": "/think"}}
    qwen3 hybrid (default-on):  {"mechanism": "…message_directive",
                                 "values": {"off": "/no_think"}}
    GLM (bidirectional kwarg):  {"mechanism": "…template_kwarg",
                                 "kwarg_path": ["chat_template_kwargs","enable_thinking"],
                                 "values": {"on": true, "off": false}}

── Dispatch (uniform) ──
``resolve_reasoning_controls`` reads the spec, resolves the mode, looks up
``values[mode]`` (None for auto or an undeclared direction ⇒ inject nothing),
and applies it per mechanism: a message directive is injected into the
messages; a template kwarg becomes a body fragment merged into the
OpenAI-compatible payload by ``llm_core._apply_reasoning_fragment`` (the
stored payload is the only thing that reaches the wire — the request path adds
no keys of its own). The template-kwarg dialect applies on OpenAI-compatible
serves (vLLM / llama.cpp / SGLang); an Ollama-native serve of the same weights
would need a native-``think`` mechanism, not yet implemented.

── Response/save repair ──
Parser-less serves whose chat template pre-fills the opening ``<think>`` into
the prompt return reasoning as bare prose ending in an orphan ``</think>``.
That shape only comes from message-directive models, so the orphan-closer
repair (``normalize_reasoning_response``) is armed whenever the model's
declared mechanism is the message directive and reasoning did not arrive
out-of-band — see llm_core._finalize_llm_response and
chat_helpers._extract_thinking_meta.

── Not implemented here (catalogued for later) ──
system-prompt directive ("detailed thinking on/off"); native top-level bool
(Ollama ``think``); structured object (hosted Anthropic/DeepSeek/…); and the
graded controls (thinkingBudget, reasoning_effort) which are dials, not binary
toggles. Each is a new mechanism value plus one dispatch branch; the storage
and API already accept "mechanism", so adding one does not reshape callers.
"""
from __future__ import annotations

import json
import logging
import math
from typing import List, Optional, Tuple

from core.log_safety import redact_url
from src.model_capabilities import (
    REASONING_CONTROL_MESSAGE_DIRECTIVE,
    REASONING_CONTROL_TEMPLATE_KWARG,
)

logger = logging.getLogger(__name__)

AUTO, ON, OFF = "auto", "on", "off"

# The mechanisms this module can actually apply today. The stored/API value is
# the canonical #2739 mechanism constant; anything outside this set is rejected
# by validate_control_spec (a user cannot declare a mechanism we don't yet
# implement, e.g. a budget/effort dial).
IMPLEMENTED_MECHANISMS = frozenset({
    REASONING_CONTROL_MESSAGE_DIRECTIVE,
    REASONING_CONTROL_TEMPLATE_KWARG,
})

# Top-level request-body fields a template-kwarg fragment must NOT target: its
# outer key becomes a payload key (llm_core._apply_reasoning_fragment), so an
# outer key like "messages"/"model"/"stream" would overwrite core request data
# and silently corrupt the call. The intended outer keys are nesting containers
# (chat_template_kwargs, extra_body); a reserved outer key is rejected at write.
_RESERVED_PAYLOAD_KEYS = frozenset({
    "model", "messages", "stream", "temperature", "max_tokens",
    "max_completion_tokens", "n", "stop", "tools", "tool_choice", "top_p",
    "frequency_penalty", "presence_penalty", "logit_bias", "user", "seed",
    "response_format", "reasoning_effort",
})


def validate_control_spec(spec) -> Tuple[Optional[dict], Optional[str]]:
    """Validate + normalize a single per-model control spec.

    Returns ``(normalized_spec, None)`` on success or ``(None, error)`` with a
    human-readable reason. Shared by the write path (the PATCH handler surfaces
    the error) and the read path (which treats any error as "no evidence").
    Normalization is minimal — canonical key order, a plain list kwarg_path —
    so what is stored round-trips unchanged.
    """
    if not isinstance(spec, dict):
        return None, "control spec must be an object"
    mechanism = spec.get("mechanism")
    # `not in frozenset` raises TypeError on an unhashable value (a JSON
    # list/dict mechanism), which on the write path would surface as a 500
    # instead of the intended 400 — the isinstance guard keeps it a clean error.
    if not isinstance(mechanism, str) or mechanism not in IMPLEMENTED_MECHANISMS:
        return None, ("mechanism must be one of "
                      f"{sorted(IMPLEMENTED_MECHANISMS)} (got {mechanism!r})")
    values = spec.get("values")
    if not isinstance(values, dict) or not values:
        return None, "values must be a non-empty object mapping 'on'/'off' to payloads"
    if not set(values).issubset({ON, OFF}):
        return None, f"values keys must be a subset of {{'on','off'}} (got {sorted(values)})"

    if mechanism == REASONING_CONTROL_MESSAGE_DIRECTIVE:
        # A message directive is a single "/think"-style soft-switch token. It
        # must carry no leading/trailing/internal whitespace: the idempotency
        # guard (_has_directive_token) matches against text.split() tokens, so a
        # multi-word or padded directive could never be detected as already
        # present and would double-inject on the re-prepare path. (Multi-word
        # phrases like "detailed thinking on" belong to the system-directive
        # mechanism, not this one.) v.split() == [v] holds iff v is exactly one
        # whitespace-free token.
        for k, v in values.items():
            if not isinstance(v, str) or v.split() != [v]:
                return None, (f"message-directive values must be a single "
                              f"whitespace-free token, e.g. '/think' (got {k}={v!r})")
        return {"mechanism": mechanism, "values": dict(values)}, None

    if mechanism == REASONING_CONTROL_TEMPLATE_KWARG:
        kwarg_path = spec.get("kwarg_path")
        if (not isinstance(kwarg_path, (list, tuple)) or len(kwarg_path) != 2
                or not all(isinstance(x, str) and x for x in kwarg_path)):
            return None, ("template-kwarg requires kwarg_path as a 2-element "
                          "[outer, inner] list of non-empty strings")
        if kwarg_path[0] in _RESERVED_PAYLOAD_KEYS:
            return None, (f"template-kwarg outer key {kwarg_path[0]!r} would overwrite a core "
                          f"request field; use a nesting key like 'chat_template_kwargs'")
        # A template-kwarg payload is set verbatim at kwarg_path in the request
        # body, so it must be a single FINITE JSON scalar (bool/string/number).
        # Reject containers AND null (a null payload validates but dispatches
        # identically to an absent direction — resolve treats None as "inject
        # nothing" — silently dropping the declared control), and reject
        # non-finite floats (NaN/Infinity are not valid JSON and would go on the
        # wire as invalid tokens).
        for k, v in values.items():
            if not isinstance(v, (bool, int, float, str)):
                return None, (f"template-kwarg values must be a JSON scalar "
                              f"(bool/string/number), not null or a container (got {k}={v!r})")
            if isinstance(v, float) and not math.isfinite(v):
                return None, f"template-kwarg values must be finite numbers (got {k}={v!r})"
        return {"mechanism": mechanism, "kwarg_path": [kwarg_path[0], kwarg_path[1]],
                "values": dict(values)}, None

    return None, f"unhandled mechanism {mechanism!r}"  # unreachable given the set check


def _normalize_control_spec(spec) -> Optional[dict]:
    """Read-path helper: validated spec or None (never raises for bad data)."""
    try:
        return validate_control_spec(spec)[0]
    except Exception:
        return None


def _controls_of(ep, model: str) -> Optional[dict]:
    """Read + normalize the control spec for `model` off an ALREADY-resolved
    endpoint row (or None). Never raises. Split out so resolve_reasoning_controls
    reads the spec AND the mode from ONE endpoint lookup instead of two."""
    try:
        raw = getattr(ep, "reasoning_controls", None) if ep is not None else None
        if not raw:
            return None
        controls = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(controls, dict):
            return None
        # Exact key first, then a lowercase fallback — `or` (not `is None`) so
        # this stays consistent with _mode_of; a falsy corrupt spec under the
        # exact key still yields to a valid lowercase-key spec.
        spec = controls.get(model) or controls.get((model or "").lower())
        return _normalize_control_spec(spec)
    except Exception:
        return None


def _mode_of(ep, model: str) -> str:
    """Read the on/off preference for `model` off an ALREADY-resolved endpoint
    row (or None), else auto. Never raises."""
    try:
        raw = getattr(ep, "reasoning_modes", None) if ep is not None else None
        if not raw:
            return AUTO
        modes = json.loads(raw) if isinstance(raw, str) else (raw or {})
        val = modes.get(model) or modes.get((model or "").lower())
        return val if val in (ON, OFF) else AUTO
    except Exception:
        return AUTO


def control_evidence_for(model: str, endpoint_url: str,
                         endpoint_id: Optional[str] = None) -> Optional[dict]:
    """The user-declared control spec for `model` on this endpoint, or None.

    The single consultation point for "does this model have a reasoning-control
    mechanism, and which one" — dispatch and the repair gates key on the
    returned ``mechanism``, never on a model name. None means "not configured"
    = not toggleable, which is also the no-leak guarantee (nothing is ever sent
    to a model the user has not explicitly described). Never raises.
    """
    try:
        return _controls_of(_endpoint_for(model, endpoint_url, endpoint_id=endpoint_id), model)
    except Exception as e:
        logger.debug("control_evidence_for failed for %s: %s",
                     redact_url(endpoint_url), type(e).__name__)
        return None


def resolve_reasoning_controls(model: str, url: str,
                               endpoint_id: Optional[str] = None) -> Tuple[Optional[str], Optional[dict], bool]:
    """Resolve the request-time controls for this (model, endpoint).

    Returns ``(message_directive, body_fragment, is_message_directive)``:
      • ``message_directive`` — a string to inject into the latest user turn,
        or None;
      • ``body_fragment`` — ``{outer: {inner: value}}`` to merge into the
        payload, or None;
      • ``is_message_directive`` — whether the model's declared mechanism is
        the message directive (used to arm the orphan-closer repair; True even
        in ``auto``/undeclared directions, because a default-on message-
        directive model reasons — and can emit the orphan-closer shape — with
        nothing injected).

    Both the control spec and the preference are read from a SINGLE endpoint
    lookup (one indexed query on model_endpoints.base_url, indexed for this
    per-call read), so every request — configured or not — pays one cheap
    indexed query, and the no-leak default (unconfigured model) costs exactly
    that. Runs on the hot paths (auto-title, memory extraction, compaction).
    """
    try:
        ep = _endpoint_for(model, url, endpoint_id=endpoint_id)  # ONE lookup, shared below
    except Exception:
        ep = None
    spec = _controls_of(ep, model)
    if spec is None:
        return None, None, False
    is_md = spec["mechanism"] == REASONING_CONTROL_MESSAGE_DIRECTIVE
    mode = _mode_of(ep, model)
    payload = spec["values"].get(mode)  # None for auto or an undeclared direction
    if payload is None:
        return None, None, is_md
    if is_md:
        return payload, None, True
    outer, inner = spec["kwarg_path"]
    return None, {outer: {inner: payload}}, False


def _has_directive_token(text: Optional[str], directive: str) -> bool:
    """Whether `directive` appears as a standalone whitespace-delimited token.

    A plain substring test would treat a lookalike such as "/thinker" as the
    directive being present and wrongly suppress the injection; exact-token
    matching only skips when the directive itself is already in the message.
    (A duplicate injection alongside e.g. "/think." is harmless — the erring
    side must be injecting, never silently dropping the user's preference.)
    Non-string values (odd multimodal blocks) simply don't contain the token.
    """
    return isinstance(text, str) and directive in text.split()


def inject_directive(messages: List[dict], directive: str) -> bool:
    """Prepend `directive` to the latest user message, in place (string or
    multimodal-list content). No-op if the directive is already present as an
    exact whitespace-delimited token. Returns whether a user message existed to
    carry it (False for system-only prompts)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if not _has_directive_token(content, directive):
                msg["content"] = f"{directive} {content}"
        elif isinstance(content, list):
            if not any(isinstance(b, dict) and _has_directive_token(b.get("text"), directive) for b in content):
                msg["content"] = [{"type": "text", "text": directive}] + content
        return True
    return False


def normalize_reasoning_response(text: str) -> str:
    """Repair the parser-less "orphan closer" reasoning shape, in the response
    direction — the mirror of ``inject_directive`` on the request side.

    Serving stacks without a reasoning parser, whose chat template pre-fills
    the opening ``<think>`` into the *prompt* (vLLM's Nemotron-VL template;
    DeepSeek-R1/QwQ official templates do the same), return reasoning as bare
    prose terminated by an orphan ``</think>``:

        ``Okay, the user asked …reasoning…</think>\\n\\nanswer``

    No opening tag means every downstream helper (strip_think,
    clean_thinking_for_save, the closed-block regexes) treats the reasoning as
    answer text. The streaming display path already folds this shape (a
    server-side auto-detect for the empty-reasoning variant plus the
    frontend's orphan-closer handling); this is the non-streaming equivalent:
    prepend ``<think>`` so the block is well-formed. Callers gate this on "the
    model's mechanism is the message directive" AND "reasoning did not arrive
    out-of-band" (see llm_core._finalize_llm_response), so out-of-band serves —
    where a literal closer in the answer would be answer text — are never
    rewritten. Well-formed blocks and reasoning-free answers pass through
    untouched.
    """
    if not isinstance(text, str) or "</think>" not in text:
        return text
    # Already well-formed if the OPENING tag is at the very start. Strip leading
    # whitespace AND invisible marks (BOM / zero-width / nbsp) the downstream
    # ^\\s*<think> extractor also skips, so a BOM-prefixed block isn't double-
    # wrapped. Any-occurrence of "<think>" (not just at start) would wrongly
    # treat reasoning prose that mentions the tag as well-formed.
    if text.lstrip("\ufeff\u200b\u200c\u200d\u2060\u00a0 \t\r\n\v\f").lower().startswith("<think"):
        return text
    # Only repair when the FIRST </think> TERMINATES the reasoning block — i.e.
    # it is followed by (optional horizontal whitespace then) a line break or
    # end-of-string: the real orphan-closer shape "reasoning…</think>\\n\\nanswer".
    # A </think> sitting mid-line is answer text (e.g. "write </think> to close
    # the block") and must NOT be folded, or half the answer is hidden as
    # thinking (visible data loss).
    # Non-regex terminator check (equivalent to ^[ \t]*(?:\r?\n|$) but linear, no
    # backtracking on adversarial many-tab model output): after any leading
    # horizontal whitespace the remainder must be empty, or begin a line break.
    after = text[text.find("</think>") + len("</think>"):]
    rest = after.lstrip(" \t")
    if rest and not (rest.startswith("\n") or rest.startswith("\r\n")):
        return text
    return "<think>" + text


def is_message_directive_model(model: str, endpoint_url: str,
                               endpoint_id: Optional[str] = None) -> bool:
    """Whether the model's declared control mechanism is the message directive
    (the only mechanism that produces the parser-less orphan-closer shape).
    Used by the save boundary to decide whether to run the repair."""
    spec = control_evidence_for(model, endpoint_url, endpoint_id=endpoint_id)
    return spec is not None and spec["mechanism"] == REASONING_CONTROL_MESSAGE_DIRECTIVE


def reasoning_mode_for(model: str, endpoint_url: str, endpoint_id: Optional[str] = None) -> str:
    """The stored *user preference* (`on`/`off`) for this model on the endpoint
    serving `endpoint_url`, else `auto`. Never raises.

    `auto`/`on`/`off` here is intent (what the user wants), kept distinct from
    the control spec (how the model toggles). `auto` means "no explicit choice
    — leave the model's default".
    """
    try:
        # Redact the URL and log only the exception class on failure: the
        # exception text can embed the raw URL (SQLAlchemy renders bound
        # parameters, and base_url is one), so it must not reach the record.
        return _mode_of(_endpoint_for(model, endpoint_url, endpoint_id=endpoint_id), model)
    except Exception as e:
        logger.debug("reasoning_mode_for failed for %s: %s", redact_url(endpoint_url), type(e).__name__)
        return AUTO


def _endpoint_serves(ep, model: str) -> bool:
    """Whether `model` is among an endpoint's enabled model ids (cached or
    pinned, minus hidden). Delegates to endpoint_resolver's shared visibility
    logic (single source of truth) with a local parse as a fallback if that
    module is unavailable at call time."""
    if not model:
        return False
    try:
        from src.endpoint_resolver import _endpoint_enabled_models
        return model in _endpoint_enabled_models(ep)
    except Exception:
        visible, hidden = set(), set()
        for attr, sink in (("cached_models", visible),
                           ("pinned_models", visible),
                           ("hidden_models", hidden)):
            raw = getattr(ep, attr, None)
            if not raw:
                continue
            try:
                ids = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if isinstance(ids, list):
                sink.update(ids)
        return model in (visible - hidden)


def _endpoint_for(model: str, endpoint_url: str, endpoint_id: Optional[str] = None):
    """Resolve the ModelEndpoint whose control/preference applies to this request.

    Stable identity first: when the caller supplies `endpoint_id` (it knows
    the exact row it is targeting), that row is authoritative — no URL
    inference at all. An id that matches no row falls through to the URL path
    rather than failing, so a stale id degrades to today's behaviour.

    URL fallback: one base URL can be shared by several endpoint rows
    (different api keys / owners / model sets), so a URL-only "first match"
    can read the wrong row. Among the URL matches we resolve DETERMINISTICALLY:
    prefer a row that serves `model`, then the shared (owner is None) row, then
    a stable id — never DB order. Reasoning control is treated as a property of
    the (backend, model), shared across owner-rows (how a served model toggles
    is decided by the serve, not by the caller), so per-owner divergence is not
    resolved here: with two owners' rows at one base_url declaring different
    controls, the shared/stable one wins (last-writer-wins on the preference).
    Per-owner resolution would need the caller's identity, which no production
    caller supplies today — sessions persist endpoint_url (not the row id) and
    fallback-chain candidates are bare (url, model, headers) tuples. Threading
    real ids/owner into those callers is the follow-up this seam exists for.

    Reuses agent_loop's candidate-key logic (lazy import to avoid an import cycle).
    """
    from core.database import SessionLocal, ModelEndpoint
    if endpoint_id:
        db = SessionLocal()
        try:
            ep = (db.query(ModelEndpoint)
                  .filter(ModelEndpoint.id == endpoint_id, ModelEndpoint.is_enabled == True)  # noqa: E712
                  .first())
            if ep is not None:
                return ep
        finally:
            db.close()
    try:
        from src.agent_loop import _endpoint_lookup_keys
        keys = _endpoint_lookup_keys(endpoint_url)
    except Exception:
        raw = (endpoint_url or "").strip()
        keys = [raw, raw.rstrip("/")]
    db = SessionLocal()
    try:
        matches, seen = [], set()
        for key in keys:
            for ep in (db.query(ModelEndpoint)
                       .filter(ModelEndpoint.base_url == key, ModelEndpoint.is_enabled == True)  # noqa: E712
                       .all()):
                if ep.id not in seen:
                    seen.add(ep.id)
                    matches.append(ep)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Several rows share this base_url (multi-tenancy: per-owner api keys /
        # model sets against the same backend). Reasoning control is a property
        # of the (backend, model) — how a served model toggles is decided by the
        # serve, not by who is asking — so it is shared across owner-rows.
        # Resolve DETERMINISTICALLY (not by DB order): prefer a row that serves
        # the model, then the shared (owner is None) row, then a stable id.
        serving = [ep for ep in matches if _endpoint_serves(ep, model)] or matches
        serving.sort(key=lambda ep: (getattr(ep, "owner", None) is not None, str(ep.id)))
        return serving[0]
    finally:
        db.close()
