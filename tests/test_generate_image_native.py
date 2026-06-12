"""Regression: `generate_image` was invokable only via fenced text, so models on
`supports_tools=True` endpoints (native function-calling — e.g. nemotron-vl) could
never call it. The tool-RAG classifier surfaced it into `relevant_tools` (see
test_tool_rag_image_domain.py / #3605), but the native tool list is built ONLY from
FUNCTION_TOOL_SCHEMAS, which had no `generate_image` entry — so it was dropped before
reaching the model, which then thrashed on ask_user/edit_image.

Fix has two halves, both covered here:
1. `generate_image` has a native schema in FUNCTION_TOOL_SCHEMAS (so native models are
   offered it), with `prompt` required + optional model/size/quality.
2. `_parse_generate_image` accepts the JSON args a native call delivers (via
   function_call_to_tool_block's default serialization) AND the legacy 4-line fenced
   form, so local/fenced models keep working.

The seam test drives a native call end-to-end through the converter into the executor's
arg parser — the exact path that silently broke.
"""

import json

import src.agent_loop  # noqa: F401  (ensures correct import order; tool_schemas is circular)
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block
from src.tool_execution import _parse_generate_image


def _schema(name):
    for s in FUNCTION_TOOL_SCHEMAS:
        if s.get("function", {}).get("name") == name:
            return s["function"]
    return None


def test_generate_image_has_native_schema():
    """Without this, native-tool models are never offered generate_image."""
    fn = _schema("generate_image")
    assert fn is not None, "generate_image missing from FUNCTION_TOOL_SCHEMAS"
    params = fn["parameters"]
    assert params["required"] == ["prompt"]
    for key in ("prompt", "model", "size", "quality"):
        assert key in params["properties"], f"missing param {key}"


def test_native_call_round_trips_to_executor_args():
    """A native function call -> ToolBlock -> _parse_generate_image must yield the
    structured args the image MCP tool expects (this is the seam that broke)."""
    raw_args = json.dumps({
        "prompt": "a red bicycle leaning on a wall",
        "model": "Qwen-Image-NVFP4",
        "size": "1024x1024",
        "quality": "high",
    })
    block = function_call_to_tool_block("generate_image", raw_args)
    assert block is not None and block.tool_type == "generate_image"
    args = _parse_generate_image(block.content)
    assert args == {
        "prompt": "a red bicycle leaning on a wall",
        "model": "Qwen-Image-NVFP4",
        "size": "1024x1024",
        "quality": "high",
    }


def test_native_call_prompt_only():
    """Optional fields omitted -> only prompt is passed through."""
    block = function_call_to_tool_block("generate_image", json.dumps({"prompt": "a fox"}))
    assert _parse_generate_image(block.content) == {"prompt": "a fox"}


def test_fenced_three_line_form_prompt_size_quality():
    """Fenced/local models deliver prompt / size / quality — NO model line (the
    server auto-selects the model). Positions map directly so a model can't end
    up in `model`."""
    content = "a sunset over mountains\n1024x1024\nhigh"
    assert _parse_generate_image(content) == {
        "prompt": "a sunset over mountains",
        "size": "1024x1024",
        "quality": "high",
    }


def test_fenced_prompt_and_size_only():
    """Quality omitted — prompt + size still align (size is line 2, not model)."""
    assert _parse_generate_image("a wide landscape\n1536x1024") == {
        "prompt": "a wide landscape",
        "size": "1536x1024",
    }


def test_fenced_size_never_lands_in_model_slot():
    """Regression for the old positional model slot: a size on line 2 must be
    parsed as size, never as a (then-unresolvable) model name."""
    args = _parse_generate_image("a castle at dusk\n1792x1024\nhigh")
    assert "model" not in args
    assert args["size"] == "1792x1024" and args["quality"] == "high"


def test_fenced_prompt_only_still_parses():
    assert _parse_generate_image("just a prompt") == {"prompt": "just a prompt"}


def test_prompt_text_starting_with_brace_is_not_swallowed_as_json():
    """A fenced prompt that happens to start with '{' but isn't a valid args object
    must fall back to line parsing, not vanish."""
    args = _parse_generate_image("{not really json} a quirky sign")
    assert args["prompt"] == "{not really json} a quirky sign"
