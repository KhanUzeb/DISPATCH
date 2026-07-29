from __future__ import annotations

from src.router.clients import ClientProfile
from src.router.index import InMemoryNumpyIndex
from src.router.passthrough import has_upstream_credentials, resolve_auth_headers
from src.router.prompting import build_routing_prompt
from src.router.schemas import RouteDefinition, Tier


def test_build_routing_prompt_handles_blank_and_nonpositive_limits():
    assert build_routing_prompt([]) == ""
    assert build_routing_prompt([("user", "  ")]) == ""
    assert build_routing_prompt([("assistant", "only assistant")]) == ""
    assert build_routing_prompt([("user", "hello")], max_chars=0) == ""
    assert build_routing_prompt([("USER", "hello")]) == "hello"


def test_empty_index_does_not_crash():
    index = InMemoryNumpyIndex()
    index.build_or_load([], [])
    assert index.query([0.1, 0.2, 0.3], top_k=3) == []


def test_index_rejects_mismatched_embedding_count():
    index = InMemoryNumpyIndex()
    routes = [RouteDefinition(name="r", tier=Tier.CHEAP, utterances=["a", "b"], threshold=0.5)]
    try:
        index.build_or_load(routes, [[0.1, 0.2]])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "must match" in str(exc)


def test_index_dimension_mismatch_returns_no_hits():
    index = InMemoryNumpyIndex()
    routes = [RouteDefinition(name="r", tier=Tier.CHEAP, utterances=["a"], threshold=0.5)]
    index.build_or_load(routes, [[0.1, 0.2, 0.3]])
    assert index.query([0.1, 0.2], top_k=3) == []


def test_has_upstream_credentials_openai_and_anthropic():
    assert has_upstream_credentials({}, protocol="openai") is False
    assert has_upstream_credentials({"Authorization": "Bearer"}, protocol="openai") is False
    assert has_upstream_credentials({"Authorization": "Bearer sk"}, protocol="openai") is True
    assert has_upstream_credentials({"x-api-key": "sk-ant"}, protocol="anthropic") is True
    assert has_upstream_credentials({"anthropic-version": "2023-06-01"}, protocol="anthropic") is False


def test_resolve_auth_headers_prefers_static_key():
    profile = ClientProfile(
        name="t",
        mode="passthrough",
        protocol="openai",
        upstream_base_url="https://example.com/v1",
        upstream_api_key="static-key",
        models={"cheap": "a", "mid": "b", "hard": "c"},
    )
    headers = resolve_auth_headers(
        profile,
        incoming_authorization="Bearer incoming",
        incoming_api_key=None,
        incoming_auth_token=None,
        protocol="openai",
    )
    assert headers["Authorization"] == "Bearer static-key"
