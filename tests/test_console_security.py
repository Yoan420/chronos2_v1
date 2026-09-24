"""Redaction tests use synthetic credentials only, never real secret material."""
from __future__ import annotations

import pytest

from experiment_console.security import MASK, has_secrets, is_secret_key, redact, redact_text


@pytest.mark.parametrize("flag", ["--api-key", "--api_key", "--password", "--access-token", "--client-secret", "-token"])
def test_command_arguments_mask_the_value_after_a_secret_flag(flag):
    command = ["C:/configured python/python.exe", "script.py", flag, "synthetic-value", "--epochs", "3"]
    assert redact(command) == [*command[:3], MASK, "--epochs", "3"]
    assert has_secrets(command)
    assert command[3] == "synthetic-value", "redaction must not mutate the actual argv"


def test_multiple_adjacent_secret_flags_and_inline_assignments():
    command = ["python.exe", "--api-key", "synthetic-one", "--password", "synthetic-two", "--access-token=synthetic-three"]
    assert redact(command) == ["python.exe", "--api-key", MASK, "--password", MASK, f"--access-token={MASK}"]


def test_malformed_adjacent_flags_never_reveal_a_possible_secret_value():
    assert redact(["--password", "--api-key", "synthetic-value"]) == ["--password", MASK, MASK]


@pytest.mark.parametrize("key", [
    "minimum_training_tokens", "minimum_validation_tokens", "max_new_tokens", "tokenizer",
    "tokenizer_class", "token_count", "token_length", "num_token", "maxTokens",
])
def test_scientific_token_parameters_are_not_credentials(key):
    config = {key: 123, "model": "chronos2"}
    assert not is_secret_key(key)
    assert redact(config) == config
    assert not has_secrets(config)
    assert redact_text(f"{key}=123") == f"{key}=123"


def test_scientific_tokenizer_arguments_are_unchanged():
    command = ("python.exe", "--tokenizer", "scientific-tokenizer", "--max-new-tokens", "64")
    assert redact(command) == command
    assert not has_secrets(command)


@pytest.mark.parametrize("key", ["API_KEY", "apiKey", "accessToken", "HF_TOKEN", "db_password", "client-secret", "authorization", "private_key", "tokenizer_api_key"])
def test_real_secret_fields_remain_rejected_even_when_nested(key):
    config = {"model": "chronos2", "parameters": {key: "synthetic-value"}}
    assert is_secret_key(key)
    assert redact(config)["parameters"][key] == MASK
    assert has_secrets(config)


@pytest.mark.parametrize("kind", ["", "RSA ", "EC ", "OPENSSH ", "ENCRYPTED "])
def test_complete_multiline_private_keys_are_masked_in_text_and_nested_values(kind):
    block = f"-----BEGIN {kind}PRIVATE KEY-----\nSYNTHETIC_KEY_BODY\nSECOND_SYNTHETIC_LINE\n-----END {kind}PRIVATE KEY-----"
    assert redact_text(f"before\n{block}\nafter") == f"before\n{MASK}\nafter"
    assert redact({"description": block}) == {"description": MASK}
    assert has_secrets({"description": block})


def test_truncated_private_key_is_masked_through_end_of_available_text():
    text = "INFO\n-----BEGIN PRIVATE KEY-----\nSYNTHETIC_KEY_BODY\n"
    assert redact_text(text) == f"INFO\n{MASK}"


def test_escaped_quoted_secret_value_is_entirely_masked():
    text = r'{"password": "synthetic\"quote-tail", "metric": 42}'
    assert "synthetic" not in redact_text(text)
    assert "quote-tail" not in redact_text(text)
    assert '"metric": 42' in redact_text(text)


def test_raw_nested_json_log_does_not_hide_a_secret_assignment_from_detection():
    text = '{"parameters":{"password":"synthetic-password","tokenizer":"chronos2"}}'
    cleaned = redact_text(text)
    assert "synthetic-password" not in cleaned
    assert '"tokenizer":"chronos2"' in cleaned


def test_environment_secret_values_are_masked_without_masking_tokenizer_values(monkeypatch):
    monkeypatch.setenv("CONSOLE_TEST_API_KEY", "synthetic-environment-secret")
    monkeypatch.setenv("CONSOLE_TEST_TOKENIZER", "scientific-tokenizer-name")
    assert redact_text("synthetic-environment-secret scientific-tokenizer-name") == f"{MASK} scientific-tokenizer-name"


def test_url_passwords_and_bearer_tokens_are_masked():
    text = "https://user:synthetic-password@example.invalid/path Authorization: Bearer synthetic-bearer"
    cleaned = redact_text(text)
    assert "synthetic-password" not in cleaned
    assert "synthetic-bearer" not in cleaned
