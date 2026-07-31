from src.openai_compat import (
    ChatMessage,
    _response_format_requires_structured_output,
    build_chat_completion_response,
    check_api_key,
    messages_to_prompt,
    normalize_messages,
)


def test_messages_to_prompt_prefers_latest_user_intent():
    messages = [
        ChatMessage(role="system", content="You are helpful"),
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="summarize this please"),
    ]
    prompt = messages_to_prompt(messages)
    assert "system: You are helpful" in prompt
    assert "user: summarize this please" in prompt
    assert "assistant: ok" not in prompt
    assert "user: first" not in prompt


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


def test_response_format_requires_structured_output_only_for_json_types():
    assert _response_format_requires_structured_output(None) is False
    assert _response_format_requires_structured_output({"type": "text"}) is False
    assert _response_format_requires_structured_output("text") is False
    assert _response_format_requires_structured_output({"type": "json_object"}) is True
    assert _response_format_requires_structured_output({"type": "json_schema"}) is True
    assert _response_format_requires_structured_output("json_object") is True


