"""list_image_model_ids must normalize every /models response shape Odysseus
image endpoints return — OpenAI ``data[].id``, a bare top-level list, and native
Ollama ``/api/tags`` ``models[].name`` — so the generate_image ``model`` enum is
populated (not silently empty) against non-OpenAI image backends. An empty enum
lets a tool-calling model pick an unavailable model (#4123 review, 2026-07-18).

Uses the gold-standard in-memory-SQLite + faked httpx.get fixture (mirrors
tests/test_generate_image_owner_scope.py::test_resolve_model_isolates_...).
"""

from unittest.mock import patch

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.ai_interaction as ai
import src.database as dbmod
# core.database has the real ORM class; src.database is MagicMock-stubbed in the
# harness (see conftest), so point the stub at the real class for the call.
from core.database import ModelEndpoint


def _seed_image_endpoint():
    engine = create_engine("sqlite:///:memory:")
    ModelEndpoint.metadata.create_all(engine, tables=[ModelEndpoint.__table__])
    TestSession = sessionmaker(bind=engine)
    s = TestSession()
    s.add(ModelEndpoint(
        id="ep-img", name="img-endpoint",
        base_url="http://img:8080/v1", api_key="IMG-KEY",
        is_enabled=True, model_type="image", owner="alice",
    ))
    s.commit()
    s.close()
    return TestSession


def _resp(payload):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return _Resp()


def _run_with(payload, configured_image_model=""):
    TestSession = _seed_image_endpoint()
    calls = []

    def _fake_get(url, headers=None, timeout=None, **kw):
        calls.append(url)
        return _resp(payload)

    ai._IMAGE_MODELS_CACHE.clear()
    with patch.object(dbmod, "SessionLocal", TestSession), \
         patch.object(dbmod, "ModelEndpoint", ModelEndpoint), \
         patch.object(ai, "resolve_endpoint_runtime",
                      lambda ep, owner=None: (ep.base_url, ep.api_key)), \
         patch("src.settings.get_setting",
               lambda key, default=None: configured_image_model if key == "image_model" else default), \
         patch.object(httpx, "get", _fake_get):
        result = ai.list_image_model_ids(owner="alice")
    return result, calls


def test_ollama_api_tags_shape_populates_enum():
    # native Ollama /api/tags: {"models":[{"name":..,"model":..}]}, no "data"/"id"
    payload = {"models": [{"name": "sdxl:latest", "model": "sdxl:latest"},
                          {"name": "flux", "model": "flux"}]}
    result, calls = _run_with(payload)
    assert result == ["sdxl:latest", "flux"]   # name-over-model; pre-fix this was []
    assert calls                                # endpoint really probed (not a vacuous fail-open [])


def test_bare_top_level_list_shape_populates_enum():
    # a bare top-level list (e.g. some OpenAI-compat proxies)
    payload = [{"id": "together-img-1"}, {"id": "together-img-2"}]
    result, calls = _run_with(payload)
    assert result == ["together-img-1", "together-img-2"]  # pre-fix: [] (a list has no .get)
    assert calls


def test_openai_data_shape_still_works_and_dedups():
    # the shape that already worked, plus the existing case-insensitive dedup
    payload = {"data": [{"id": "gpt-image-1"}, {"id": "GPT-Image-1"}, {"id": "dall-e-3"}]}
    result, _ = _run_with(payload)
    assert result == ["gpt-image-1", "dall-e-3"]  # OpenAI path unchanged; near-dup collapsed, first-wins


def test_empty_probe_falls_back_to_configured_image_model():
    # endpoint reachable but serves nothing -> configured-model fallback still fires
    result, _ = _run_with({"data": []}, configured_image_model="z-image-turbo")
    assert result == ["z-image-turbo"]
