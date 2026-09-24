"""Concise primary-run failures, extracted only from that run's own output."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from .security import redact_text


MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 65536
MAX_SUMMARY_CHARS = 600
_KEY_START = re.compile(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----', re.I)
_KEY_END = re.compile(r'-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----', re.I)
_NUCLEAR = re.compile(r'^\s*\[(Nuclear(?: Kalman)?)\]\s*(?:ECHEC|ÉCHEC)\b\s*([^:]*):\s*(.*)', re.I)


def _concise(value):
    # Redact before truncation, otherwise cutting an assignment/key block can
    # lose the context needed to identify its sensitive value.
    value = ' '.join(redact_text(value).split())
    return value if len(value) <= MAX_SUMMARY_CHARS else value[:MAX_SUMMARY_CHARS - 1] + '…'


class FailureSummary:
    def __init__(self):
        self.message = None
        self._priority = -1
        self._private = False

    def feed(self, line):
        if not isinstance(line, str):
            return
        if _KEY_START.search(line):
            self._private = True
        if self._private:
            if _KEY_END.search(line):
                self._private = False
            return
        if len(line) > MAX_LINE_BYTES:
            return
        clean = redact_text(line).strip()
        match = _NUCLEAR.match(clean)
        is_exception = bool(re.match(r'^(?:[A-Za-z_][\w.]*\.)?(?:\w*Error|\w*Exception):', clean))
        console_error = clean.startswith('[Console ERROR]')
        if not (match or is_exception or console_error):
            return
        lowered = clean.lower()
        if 'proxyerror' in lowered:
            service = 'Accès à Saturn impossible' if 'saturn' in lowered else 'Connexion au service impossible'
            unreachable = any(fragment in lowered for fragment in (
                'unable to connect to proxy', 'cannot connect to proxy',
                'failed to establish a new connection', 'connection refused', 'winerror 10061'))
            message = service + (': le proxy configuré est injoignable' if unreachable else ': erreur de proxy')
            # Only an explicit nested proxy connection identifies its address;
            # never mistake the outer HTTPS pool (the target service) for it.
            nested = clean[lowered.find('unable to connect to proxy'):] if 'unable to connect to proxy' in lowered else ''
            addresses = re.findall(r"(?:HTTPS?Connection|ProxyConnection)\(host=['\"]([a-zA-Z0-9._:-]{1,253})['\"],\s*port=(\d{1,5})", nested)
            if unreachable and addresses and 1 <= int(addresses[-1][1]) <= 65535:
                host, port = addresses[-1]
                message += f' ({host}:{port})'
            message += '.'
            priority = 100
        elif match:
            component, zone, explanation = match.groups()
            message = (zone.strip() + ' : ' if zone.strip() else '') + explanation
            # The outer batch exit-code line is less informative than the cause
            # emitted by the country process immediately before it.
            generic_batch = component.lower() == 'nuclear kalman' and re.search(
                r'(?:processus.*\bcode\s*[:=]?\s*-?\d+|\b(?:returncode|exit_code|exit code|code de retour)\s*[:=]?\s*-?\d+)', explanation, re.I)
            priority = 30 if generic_batch else (80 if component.lower() == 'nuclear' else 60)
        else:
            message = clean.removeprefix('[Console ERROR]').strip()
            priority = 70 if console_error else 50
        if message and priority > self._priority:
            self.message = _concise(message)
            self._priority = priority


def _safe_log(run_directory):
    directory = Path(os.path.abspath(run_directory))
    for component in [*reversed(directory.parents), directory]:
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024 or not stat.S_ISDIR(info.st_mode):
            raise ValueError('Journal de run non local.')
    target = directory / 'console.log'
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024 or not stat.S_ISREG(info.st_mode):
        raise ValueError('Journal de run non local.')
    return target, info


def read_failure_summary(run_directory, max_bytes=MAX_LOG_BYTES):
    """Bounded, read-only fallback for failures recorded before summaries existed."""
    try:
        target, before = _safe_log(run_directory)
        summary, remaining = FailureSummary(), max(0, min(int(max_bytes), MAX_LOG_BYTES))
        with target.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                return None
            discarding = False
            while remaining:
                raw = stream.readline(min(MAX_LINE_BYTES + 1, remaining))
                if not raw:
                    break
                remaining -= len(raw)
                if discarding:
                    discarding = not raw.endswith(b'\n')
                    continue
                if len(raw) > MAX_LINE_BYTES or not raw.endswith(b'\n') and remaining == 0:
                    # A partial line cannot safely preserve secret context.
                    if _KEY_START.search(raw.decode('utf-8', errors='replace')):
                        summary._private = True
                    discarding = not raw.endswith(b'\n')
                    continue
                summary.feed(raw.decode('utf-8', errors='replace'))
            _, after = _safe_log(run_directory)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                return None
        return summary.message
    except (OSError, ValueError, TypeError):
        return None
