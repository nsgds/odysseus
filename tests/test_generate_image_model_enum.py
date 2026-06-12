"""generate_image's `model` param must be enum-constrained to the actually-installed
image models on the agent path, so a native-tool model can't invent a name (DALL-E 3,
stable-diffusion-*, ...) the backend lacks — which hard-fails at resolution.

Covers the pure schema-injection helper (with_image_model_enum). The dynamic model
list (list_image_model_ids) hits the DB/endpoints and is exercised live, not here.
"""

import src.agent_loop  # noqa: F401  (import-order: tool_schemas is circular)
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, with_image_model_enum


def _gen_image(schemas):
    for s in schemas:
        if s.get("function", {}).get("name") == "generate_image":
            return s
    return None


def test_enum_injected_for_generate_image():
    out = with_image_model_enum(FUNCTION_TOOL_SCHEMAS, ["Qwen-Image-NVFP4"])
    model = _gen_image(out)["function"]["parameters"]["properties"]["model"]
    assert model["enum"] == ["Qwen-Image-NVFP4"]


def test_multiple_models_all_listed():
    out = with_image_model_enum(FUNCTION_TOOL_SCHEMAS, ["Qwen-Image-NVFP4", "flux-1"])
    model = _gen_image(out)["function"]["parameters"]["properties"]["model"]
    assert model["enum"] == ["Qwen-Image-NVFP4", "flux-1"]


def test_empty_list_fails_open_no_enum():
    """No installed models discovered -> no enum, tool not blocked."""
    out = with_image_model_enum(FUNCTION_TOOL_SCHEMAS, [])
    model = _gen_image(out)["function"]["parameters"]["properties"]["model"]
    assert "enum" not in model
    assert out is FUNCTION_TOOL_SCHEMAS  # unchanged passthrough


def test_does_not_mutate_the_global_schema():
    """Injection must deep-copy generate_image, never mutate FUNCTION_TOOL_SCHEMAS."""
    with_image_model_enum(FUNCTION_TOOL_SCHEMAS, ["Qwen-Image-NVFP4"])
    original = _gen_image(FUNCTION_TOOL_SCHEMAS)["function"]["parameters"]["properties"]["model"]
    assert "enum" not in original, "global schema was mutated"


def test_other_tools_untouched():
    out = with_image_model_enum(FUNCTION_TOOL_SCHEMAS, ["Qwen-Image-NVFP4"])
    names_in = {s.get("function", {}).get("name") for s in FUNCTION_TOOL_SCHEMAS}
    names_out = {s.get("function", {}).get("name") for s in out}
    assert names_in == names_out
    # edit_image (sibling) must be byte-identical (passed by reference)
    ei_in = next(s for s in FUNCTION_TOOL_SCHEMAS if s["function"]["name"] == "edit_image")
    ei_out = next(s for s in out if s["function"]["name"] == "edit_image")
    assert ei_in is ei_out
