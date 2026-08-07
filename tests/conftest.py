from __future__ import annotations

import sys
import types

try:
    import chronos2_modular.common  # noqa: F401
except ImportError:
    package = types.ModuleType("chronos2_modular")
    common = types.ModuleType("chronos2_modular.common")

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    class ZoneConfig:
        pass

    class ZoneData:
        pass

    def deep_get(mapping, path, default=None):
        current = mapping
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    common.LOGGER = DummyLogger()
    common.ZoneConfig = ZoneConfig
    common.ZoneData = ZoneData
    common.deep_get = deep_get
    package.common = common
    sys.modules["chronos2_modular"] = package
    sys.modules["chronos2_modular.common"] = common
