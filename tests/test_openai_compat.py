from src.openai_compat import (
    ChatMessage,
    build_chat_completion_response,
    check_api_key,
    messages_to_prompt,
    normalize_messages,
)


def test_messages_to_prompt_includes_conversation_context():
    messages = [
        ChatMessage(role="system", content="You are helpful"),
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="summarize this please"),
    ]
    prompt = messages_to_prompt(messages)
    assert "system: You are helpful" in prompt
    assert "user: summarize this please" in prompt
    assert "assistant: ok" in prompt


def test_messages_to_prompt_truncates_from_end():
    messages = [ChatMessage(role="user", content="x" * 5000)]
    prompt = messages_to_prompt(messages, max_chars=100)
    assert len(prompt) == 100
    assert prompt.endswith("x" * 20)


def test_normalize_messages_handles_multimodal_text_parts():
    messages = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
        )
    ]
    assert normalize_messages(messages) == [{"role": "user", "content": "hello\nworld"}]


def test_build_chat_completion_response_shape():
    body = build_chat_completion_response(
        completion_id="chatcmpl-test",
        model="llama-3.1-8b-instant",
        text="hi",
        prompt_tokens=2,
        completion_tokens=1,
        created=123,
    )
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["total_tokens"] == 3


def test_check_api_key_optional(monkeypatch):
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    check_api_key(None)
    check_api_key("Bearer anything")
