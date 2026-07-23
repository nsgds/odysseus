"""Direct contract tests for _extract_model_ids — the shared /models-shape
normalizer introduced for #4123 and used at three image-model-discovery sites.
It is exercised transitively via list_image_model_ids elsewhere; this locks the
shape-priority and tolerance rules independently so a future edit can't silently
invert the priority or drop a guard.
"""

from src.ai_interaction import _extract_model_ids


def test_openai_data_shape():
    assert _extract_model_ids({"data": [{"id": "a"}, {"id": "b"}]}) == ["a", "b"]


def test_bare_top_level_list():
    assert _extract_model_ids([{"id": "a"}, {"id": "b"}]) == ["a", "b"]


def test_ollama_models_name_shape():
    assert _extract_model_ids({"models": [{"name": "sdxl"}, {"name": "flux"}]}) == ["sdxl", "flux"]


def test_ollama_name_preferred_over_model():
    # DISTINCT name vs model so the preference is actually pinned (not vacuous).
    out = _extract_model_ids({"models": [{"name": "sdxl", "model": "registry/sdxl@sha256:abc"}]})
    assert out == ["sdxl"]


def test_ollama_model_used_when_name_absent():
    assert _extract_model_ids({"models": [{"model": "only-model"}]}) == ["only-model"]


def test_data_present_suppresses_models():
    # Both keys populated -> the OpenAI data ids win; models is ignored.
    body = {"data": [{"id": "from-data"}], "models": [{"name": "from-models"}]}
    assert _extract_model_ids(body) == ["from-data"]


def test_empty_data_falls_through_to_models():
    assert _extract_model_ids({"data": [], "models": [{"name": "m1"}]}) == ["m1"]


def test_non_dict_entries_skipped_in_list():
    assert _extract_model_ids([{"id": "a"}, "junk", 5, None, {"id": "b"}]) == ["a", "b"]


def test_non_dict_entries_skipped_in_models():
    assert _extract_model_ids({"models": ["junk", {"name": "ok"}, 3]}) == ["ok"]


def test_missing_ids_yield_empty():
    assert _extract_model_ids({"data": [{}, {"object": "model"}]}) == []


def test_none_and_scalar_bodies_are_safe():
    # r.json() can be None / str / int on a broken backend -> must not raise.
    assert _extract_model_ids(None) == []
    assert _extract_model_ids("nonsense") == []
    assert _extract_model_ids(123) == []


def test_empty_and_null_containers():
    assert _extract_model_ids({}) == []
    assert _extract_model_ids([]) == []
    assert _extract_model_ids({"data": None}) == []
    assert _extract_model_ids({"models": None}) == []


def test_non_string_id_values_skipped():
    # a malformed backend id (int) must be dropped, not returned -> downstream
    # .lower() matching in _resolve_model can't raise AttributeError.
    assert _extract_model_ids({"data": [{"id": 123}, {"id": "ok"}]}) == ["ok"]


def test_non_string_name_values_skipped():
    assert _extract_model_ids({"models": [{"name": 123}, {"name": "ok"}]}) == ["ok"]
    assert _extract_model_ids({"models": [{"name": 5, "model": "m-fallback"}]}) == ["m-fallback"]
