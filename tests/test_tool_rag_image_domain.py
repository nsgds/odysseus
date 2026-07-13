"""Regression: the agent tool-RAG domain classifier had no image/media domain,
so image requests classified to domains=[] -> low_signal=True. The damage is
two-fold:

1. A FRESH first turn (no prior conversation, no forced tools) takes the
   direct low-signal reply path in stream_agent_loop -- the model receives NO
   tools at all and can only refuse ("I'm a text-based AI...").
2. On contextual turns retrieval does run, but inclusion of generate_image
   then depends on embedding luck, and without the `images` _DOMAIN_TOOL_MAP
   entry no "Image rules" pack exists to steer the model.

This is the same failure shape the `integrations` domain fixed for api_call
(see the #3794 note in _classify_agent_request): matched no domain ->
low-signal -> the tool never deterministically reaches the model.

Root cause: `_classify_agent_request` sets
`low_signal = not continuation and not domains`; with no `images` domain,
prompts like "generate two images of X" matched nothing.

The classifier is deterministic string matching (no embeddings / no DB), so it
can be exercised directly.
"""

from src.agent_loop import (
    _classify_agent_request,
    _DOMAIN_TOOL_MAP,
    _DOMAIN_RULES,
    _domain_rules_for_tools,
)


def _classify(text):
    return _classify_agent_request([{"role": "user", "content": text}], text)


def test_image_generation_requests_get_image_domain():
    """Image-generation phrasings must match the `images` domain and NOT be
    treated as low-signal (which would skip tool retrieval)."""
    prompts = [
        "generate two images of this character: one action pose, one relaxed",
        "draw me a picture of a cat",
        "make an illustration of a spaceship",
        "create an image of a sunset over mountains",
        "design a logo for my coffee shop",
    ]
    for p in prompts:
        intent = _classify(p)
        assert "images" in intent["domains"], f"expected images domain for: {p!r}"
        assert intent["low_signal"] is False, f"must not be low_signal: {p!r}"


def test_fresh_first_turn_image_requests_are_not_low_signal():
    """The deterministic failure mode: a fresh chat's FIRST turn. When
    low_signal is True on a first turn, stream_agent_loop's direct low-signal
    path replies with NO tools, so the request cannot succeed regardless of
    retrieval or schemas. These phrasings (varied verb/noun, observed failing
    4/4 live) must classify low_signal=False so that path cannot fire."""
    first_turns = [
        "Draw a picture of a red fox sitting in a snowy forest.",
        "Create an image of a futuristic city skyline at night.",
        "Paint a watercolor of a mountain lake at dawn.",
        "I'd like a portrait of a steampunk owl wearing brass goggles.",
    ]
    for p in first_turns:
        intent = _classify(p)
        assert "images" in intent["domains"], f"expected images domain for: {p!r}"
        assert intent["low_signal"] is False, f"fresh-turn trap for: {p!r}"


def test_image_edit_requests_get_image_domain():
    """Edit/upscale/background phrasings also resolve to the image domain."""
    for p in (
        "upscale image 5",
        "remove the background from this photo",
        "remove background",
        "inpaint the selected area",
    ):
        intent = _classify(p)
        assert "images" in intent["domains"], f"expected images domain for: {p!r}"


def test_image_domain_seeds_generate_and_edit_image():
    """The domain must seed the actual image tools so they are offered even when
    semantic retrieval misses."""
    assert _DOMAIN_TOOL_MAP["images"] == {"generate_image", "edit_image"}


def test_image_domain_has_a_rule_pack():
    """Every domain in _DOMAIN_TOOL_MAP needs a matching _DOMAIN_RULES entry,
    otherwise _domain_rules_for_tools raises KeyError when the tools are selected."""
    assert "images" in _DOMAIN_RULES
    rules = _domain_rules_for_tools({"generate_image"})
    assert any("Image rules" in r for r in rules)


def test_non_image_requests_do_not_match_image_domain():
    """Guard against over-triggering: ordinary prompts must not be flagged image."""
    assert "images" not in _classify("what is the capital of France")["domains"]
    assert "images" not in _classify("reply to the latest email in my inbox")["domains"]


def test_software_image_artifacts_do_not_match_image_domain():
    """'image' as a software artifact is not visual media: docker/disk images
    must not trip the domain (they classify to files/settings paths instead)."""
    for p in (
        "pull the docker image and restart the container",
        "flash the disk image to the usb drive",
    ):
        assert "images" not in _classify(p)["domains"], f"false positive: {p!r}"
