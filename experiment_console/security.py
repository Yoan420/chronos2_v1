"""Redaction shared by persisted metadata, process logs and HTTP exports."""
from __future__ import annotations

import os
import re
from typing import Any

SECRET_KEY = re.compile(r"(?:^|_)(?:password|passwd|secret|secrets|authorization|credential|credentials|api_key|apikey|private_key|privatekey|access_key|accesskey)(?:_|$)", re.I)
TOKEN_MEASUREMENT = re.compile(r"(?:^|_)(?:token_(?:count|counts|length|lengths|limit|limits|budget|size|ids?)|(?:num|number_of|max|min|maximum|minimum|total)_token)(?:_|$)")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)", re.I | re.S)
ASSIGNMENT = re.compile(r'''(?x)(?P<prefix>(?<![\w-])["']?(?P<key>-{0,2}[A-Za-z_][A-Za-z0-9_.-]*)["']?\s*[:=]\s*)(?P<value>"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s,;{}\[\]]+)''')
MASK = "[MASQUÉ]"


def is_secret_key(value: str) -> bool:
    # Treat tokenizers and token counts as model parameters, while recognizing
    # snake_case, kebab-case, environment variables, and camelCase credentials.
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if SECRET_KEY.search(name):
        return True
    return "token" in name.split("_") and not TOKEN_MEASUREMENT.search(name)


def redact_text(value: str) -> str:
    text = str(value)
    text = PRIVATE_KEY.sub(MASK, text)
    text = re.sub(r"(?i)(bearer\s+)[\w.+=/\-]+", lambda m: m[1] + MASK, text)
    text = re.sub(r"(?i)(https?://)[^\s/@:]+:[^\s/@]+@", lambda m: m[1] + MASK + "@", text)
    text = ASSIGNMENT.sub(lambda match: match['prefix'] + MASK if is_secret_key(match['key']) else match[0], text)
    text = re.sub(r"\b(?:hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,})\b", MASK, text)
    for key, secret in os.environ.items():
        if is_secret_key(key) and len(secret) >= 6:
            text = text.replace(secret, MASK)
    return text


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): MASK if is_secret_key(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = []
        hide_next = False
        for item in value:
            items.append(MASK if hide_next else redact(item))
            hide_next = (isinstance(item, str) and
                         bool(re.fullmatch(r"--?[A-Za-z][A-Za-z0-9_-]*", item)) and is_secret_key(item))
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, str):
        return redact_text(value)
    return value


def has_secrets(value: Any) -> bool:
    return redact(value) != value
