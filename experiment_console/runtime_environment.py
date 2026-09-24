"""Read-only checks for a known inherited restriction on scientific requests."""
from __future__ import annotations

from collections.abc import Mapping
import os
from urllib.parse import urlsplit


_PROXY_NAMES = frozenset({'http_proxy', 'https_proxy', 'all_proxy'})
_LOOPBACK_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})
_FALSE_VALUES = frozenset({'', '0', 'false', 'no', 'off'})
_PRIMARY_ENVIRONMENT_ERROR = (
    'Le service NYX a hérité d’un environnement Codex dont l’accès réseau est bloqué. '
    'Les observations et Storm ne peuvent pas être actualisés. '
    'Redémarrez le service NYX depuis son raccourci Windows, hors de cet environnement, '
    'puis relancez le calcul.'
)


def primary_environment_error(environ: Mapping[str, str] | None = None) -> str | None:
    """Explain the inherited sandbox proxy without changing any environment value.

    A marker alone is insufficient: an authorized process may retain it with a
    usable network environment. Ordinary user or corporate proxies are left to
    the existing pipeline, including when no sandbox marker is present.
    """
    environment = os.environ if environ is None else environ
    if not any(name.casefold() == 'codex_sandbox_network_disabled'
               and value.strip().casefold() not in _FALSE_VALUES
               for name, value in environment.items()):
        return None
    for name, value in environment.items():
        if name.casefold() not in _PROXY_NAMES:
            continue
        try:
            value = value.strip()
            proxy = urlsplit(value if '://' in value else '//' + value)
            if proxy.hostname in _LOOPBACK_HOSTS and proxy.port == 9:
                return _PRIMARY_ENVIRONMENT_ERROR
        except ValueError:
            # An unrelated malformed URL must not masquerade as this diagnosis.
            continue
    return None
