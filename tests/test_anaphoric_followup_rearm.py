"""A vague follow-up right after a real tool turn ("do another", "add a couple
more to it", "try again") references the prior action only anaphorically — no
domain keyword and not a terse continuation — so `_classify_agent_request`
returns domains=[] and the just-used tool is stripped from selection. A small
model, asked to repeat an action it no longer has, confabulates: observed live
as a fabricated image link (images) and a wrong-tool cascade that built a
phantom document while claiming to update a note (notes).

Fix: when a turn matches NO domain (and isn't a continuation), re-arm the
domain of any tool a recent assistant turn actually used, reusing
_DOMAIN_TOOL_MAP as the tool->domain reverse map. Deliberately narrow — turns
that matched a domain of their own are untouched, casual/greeting turns exit
before the re-arm, and explicit continuations already inherit context.

The classifier is deterministic (no embeddings/DB), so it is exercised directly.
"""

from src.agent_loop import (
    _assistant_recently_used_tools,
    _classify_agent_request,
    _DOMAIN_TOOL_MAP,
)


def _tool_turn(tool, text="Done."):
    return {"role": "assistant", "content": text,
            "metadata": {"tool_events": [{"round": 1, "tool": tool, "exit_code": 0}]}}


def _plain(role, text):
    return {"role": role, "content": text}


def _classify(msgs):
    return _classify_agent_request(msgs, msgs[-1]["content"])


# --- _assistant_recently_used_tools -----------------------------------------

def test_collects_tools_from_tool_events():
    msgs = [_plain("user", "make a note to buy milk"), _tool_turn("manage_notes"),
            _plain("user", "add a couple more to it")]
    assert "manage_notes" in _assistant_recently_used_tools(msgs)


def test_generated_image_url_fallback_maps_to_image_tools():
    msgs = [_plain("user", "make an image"),
            {"role": "assistant", "content": "Direct link: https://x/api/generated-image/abc.png"},
            _plain("user", "do another one")]
    used = _assistant_recently_used_tools(msgs)
    assert {"generate_image", "edit_image"} <= used


def test_lookback_bound():
    """Tool turns older than the lookback window are not collected."""
    msgs = [_plain("user", "note please"), _tool_turn("manage_notes")]
    for i in range(5):  # 5 newer assistant turns push the tool turn out (lookback=4)
        msgs += [_plain("user", f"q{i}"), _plain("assistant", f"a{i}")]
    msgs += [_plain("user", "add more to it")]
    assert "manage_notes" not in _assistant_recently_used_tools(msgs)


# --- the classifier re-arm ---------------------------------------------------

def test_anaphoric_followup_rearms_used_tools_domain():
    """The live repro: notes turn, then a keyword-free follow-up."""
    msgs = [_plain("user", 'Make a note titled "Grocery run" to buy milk and eggs.'),
            _tool_turn("manage_notes"),
            _plain("user", "actually add a couple more to it — bread and coffee")]
    intent = _classify(msgs)
    assert "notes_calendar_tasks" in intent["domains"]
    assert intent["low_signal"] is False


def test_rearm_covers_every_mapped_domain():
    """The reverse map must work for each domain's tools, not just one."""
    for domain, tools in _DOMAIN_TOOL_MAP.items():
        tool = sorted(tools)[0]
        msgs = [_plain("user", "please handle this"), _tool_turn(tool),
                _plain("user", "hmm, now do the other one as well")]
        intent = _classify(msgs)
        assert domain in intent["domains"], f"{tool} should re-arm {domain}"


def test_turn_with_its_own_domain_is_untouched():
    """A follow-up that matched a domain keeps ONLY its own match — recent tool
    use must not piggyback extra domains onto keyword-matched turns."""
    msgs = [_plain("user", "make a note to buy milk"), _tool_turn("manage_notes"),
            _plain("user", "now reply to the latest email in my inbox")]
    intent = _classify(msgs)
    assert "email" in intent["domains"]
    assert "notes_calendar_tasks" not in intent["domains"]


def test_casual_turn_does_not_rearm():
    """Greetings/thanks exit on the casual path before the re-arm."""
    msgs = [_plain("user", "make a note to buy milk"), _tool_turn("manage_notes"),
            _plain("user", "thanks!")]
    intent = _classify(msgs)
    assert intent["domains"] == set()
    assert intent["low_signal"] is True


def test_unmapped_mcp_tool_is_inert():
    """A user-registered qualified MCP tool maps to no domain — no re-arm."""
    msgs = [_plain("user", "check the weather"), _tool_turn("mcp__weather__get_forecast"),
            _plain("user", "and tomorrow as well?")]
    intent = _classify(msgs)
    assert intent["domains"] == set()


def test_fresh_chat_unaffected():
    """No assistant history -> nothing to re-arm; fresh-turn behavior unchanged."""
    msgs = [_plain("user", "hello there, what can you do?")]
    intent = _classify(msgs)
    assert intent["domains"] == set()
