"""Isolated desktop settings for the optional NYX CPU Preview shortcut."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


PREVIEW_PORT = 8766
PREVIEW_STATE_ROOT = Path('runs/.experiment_console_cpu_preview')


def preview_settings_path(project_root: str | Path | None = None) -> Path:
    """Copy the configured Python while isolating the preview service and queue."""
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    source = root / 'config' / 'experiment_console.json'
    settings = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(settings, dict) or not isinstance(settings.get('python_executable'), str):
        raise ValueError(f'Configuration Python NYX invalide : {source}')
    settings = {**settings, 'state_root': PREVIEW_STATE_ROOT.as_posix(), 'port': PREVIEW_PORT}
    state = root / PREVIEW_STATE_ROOT
    state.mkdir(parents=True, exist_ok=True)
    destination = state / 'desktop_settings.json'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=state,
                                         prefix='.desktop_settings.', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(settings, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
