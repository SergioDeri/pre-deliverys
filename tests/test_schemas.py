import pytest
from pydantic import ValidationError

from llm_client.schemas import ChatMessage, GenerationConfig, RetryPolicy


def test_chat_message_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


def test_chat_message_keeps_content_untouched() -> None:
    message = ChatMessage(role="user", content="\n  indented code\n")

    assert message.content == "\n  indented code\n"


def test_chat_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="hi")


def test_chat_message_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="hi", name="bob")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"top_p": 1.5},
        {"max_tokens": 0},
        {"timeout_s": 0},
        {"model": ""},
    ],
)
def test_generation_config_rejects_out_of_range_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate({"model": "some-model", **overrides})


def test_generation_config_accepts_temperature_bounds() -> None:
    assert GenerationConfig(model="m", temperature=0).temperature == 0
    assert GenerationConfig(model="m", temperature=2).temperature == 2


def test_retry_policy_doubles_the_delay_on_each_attempt() -> None:
    policy = RetryPolicy(base_delay_s=1, jitter=False)

    assert [policy.delay_for(attempt) for attempt in (1, 2, 3)] == [1, 2, 4]


def test_retry_policy_prefers_the_server_retry_after() -> None:
    policy = RetryPolicy(base_delay_s=1, jitter=False)

    assert policy.delay_for(1, retry_after_s=7) == 7


def test_retry_policy_never_waits_longer_than_the_cap() -> None:
    policy = RetryPolicy(base_delay_s=1, max_delay_s=3, jitter=False)

    assert policy.delay_for(5) == 3
    assert policy.delay_for(1, retry_after_s=60) == 3


def test_retry_policy_jitter_stays_within_a_tenth_of_the_delay() -> None:
    policy = RetryPolicy(base_delay_s=2)

    for _ in range(50):
        assert 2 <= policy.delay_for(1) <= 2.2
