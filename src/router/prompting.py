from __future__ import annotations


def build_routing_prompt(
    messages: list[tuple[str, str]],
    *,
    max_chars: int = 4000,
) -> str:
    """Build a stable classification prompt from chat transcripts.

    We route primarily on the latest user intent so adapter-specific transcript
    formatting (OpenAI vs Anthropic) does not produce divergent tier picks.
    """
    if max_chars <= 0:
        return ""

    latest_user = ""
    latest_system = ""
    for role, text in messages:
        normalized = (text or "").strip()
        if not normalized:
            continue
        role_key = (role or "").strip().lower()
        if role_key == "system":
            latest_system = normalized
        elif role_key in {"user", "human"}:
            latest_user = normalized

    if not latest_user:
        return ""

    prompt = latest_user if not latest_system else f"system: {latest_system}\nuser: {latest_user}"
    if len(prompt) <= max_chars:
        return prompt
    return prompt[-max_chars:]
