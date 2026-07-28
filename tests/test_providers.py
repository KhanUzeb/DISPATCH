from src.router.providers.factory import build_provider_registry
from src.router.providers.openai_compatible import OpenAICompatibleAdapter
from src.router.schemas import Provider, ProviderRequest


def test_provider_registry_includes_all_providers():
    registry = build_provider_registry()
    for provider in Provider:
        assert registry.get(provider) is not None


def test_openai_compatible_adapter_builds_chat_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "req-1",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

    def fake_post(url, timeout, headers, json):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("src.router.providers.openai_compatible.requests.post", fake_post)
    adapter = OpenAICompatibleAdapter(
        provider=Provider.OPENAI,
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )
    response = adapter.execute(
        "gpt-4o-mini",
        ProviderRequest(prompt="hi", messages=[{"role": "user", "content": "hi"}], max_output_tokens=16),
    )
    assert response.text == "hello"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["json"]["model"] == "gpt-4o-mini"
    assert response.provider == Provider.OPENAI
