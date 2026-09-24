"""Synthetic environment checks; no networking or process launches."""
from types import MappingProxyType, SimpleNamespace

import pytest

from experiment_console.runtime_environment import primary_environment_error


@pytest.mark.parametrize('name', ['HTTP_PROXY', 'https_proxy', 'AlL_PrOxY'])
@pytest.mark.parametrize('url', ['http://127.0.0.1:9', 'HTTP://LOCALHOST:9', 'http://[::1]:9', '127.0.0.1:9'])
def test_known_proxy_with_sandbox_marker_gives_actionable_error(name, url):
    message = primary_environment_error({'CODEX_SANDBOX_NETWORK_DISABLED': '1', name: url})
    assert message and 'Codex' in message and 'raccourci Windows' in message


@pytest.mark.parametrize('marker', ['1', 'true', 'YES', ' on '])
def test_marker_is_case_insensitive_and_accepts_truthy_values(marker):
    assert primary_environment_error({'codex_sandbox_network_disabled': marker, 'HTTP_PROXY': 'http://localhost:9'})


@pytest.mark.parametrize('marker', ['', '0', 'false', 'NO', ' off '])
def test_false_marker_does_not_reinterpret_a_user_proxy(marker):
    assert primary_environment_error({'CODEX_SANDBOX_NETWORK_DISABLED': marker, 'HTTPS_PROXY': 'http://127.0.0.1:9'}) is None


def test_marker_without_blocked_proxy_is_healthy_for_authorized_processes():
    assert primary_environment_error({'CODEX_SANDBOX_NETWORK_DISABLED': '1'}) is None
    assert primary_environment_error({'HTTPS_PROXY': 'http://127.0.0.1:9'}) is None


@pytest.mark.parametrize('url', [
    'http://corporate.example:8080', 'http://corporate.example:9',
    'http://127.0.0.1:8080', 'http://localhost', 'http://127.0.0.1:invalid',
    'http://[::1', '', 'http://127.0.0.1.example:9',
])
def test_other_or_malformed_proxy_settings_are_preserved(url):
    source = {'CODEX_SANDBOX_NETWORK_DISABLED': '1', 'HTTPS_PROXY': url, 'NO_PROXY': 'localhost'}
    assert primary_environment_error(MappingProxyType(source)) is None
    assert source['HTTPS_PROXY'] == url


def test_error_never_contains_proxy_credentials_and_input_is_not_mutated():
    source = {'CODEX_SANDBOX_NETWORK_DISABLED': '1',
              'HTTP_PROXY': 'http://synthetic-user:synthetic-password@localhost:9/private-path',
              'NO_PROXY': 'internal.example', 'SATURN_AUTHOR': 'synthetic-author'}
    before = source.copy()
    message = primary_environment_error(MappingProxyType(source))
    assert message
    for value in ['synthetic-user', 'synthetic-password', 'private-path', 'synthetic-author', 'internal.example']:
        assert value not in message
    assert source == before


def test_default_environment_is_read_only_and_explicit_empty_mapping_is_respected(monkeypatch):
    source = {'CODEX_SANDBOX_NETWORK_DISABLED': '1', 'ALL_PROXY': 'http://127.0.0.1:9'}
    monkeypatch.setattr('experiment_console.runtime_environment.os', SimpleNamespace(environ=MappingProxyType(source)))
    assert primary_environment_error()
    assert primary_environment_error({}) is None
    assert source == {'CODEX_SANDBOX_NETWORK_DISABLED': '1', 'ALL_PROXY': 'http://127.0.0.1:9'}
