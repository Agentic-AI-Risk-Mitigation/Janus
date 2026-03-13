"""
Scenario auto-discovery registry.

Scans subdirectories of this package for modules containing a subclass of
BaseScenario and exposes them via get_scenario(name) and list_scenarios().
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from examples.shared.scenario_base import BaseScenario

_SCENARIOS: dict[str, type[BaseScenario]] = {}
_discovered = False


def _discover() -> None:
    """Import all scenario subpackages and register their scenario classes."""
    global _discovered
    if _discovered:
        return

    package_dir = Path(__file__).parent

    for info in pkgutil.iter_modules([str(package_dir)]):
        if not info.ispkg:
            continue
        try:
            mod = importlib.import_module(f"examples.scenarios.{info.name}.scenario")
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseScenario)
                    and attr is not BaseScenario
                    and hasattr(attr, "name")
                    and attr.name
                ):
                    _SCENARIOS[attr.name] = attr
        except (ImportError, AttributeError) as exc:
            print(f"[scenarios] Warning: could not load {info.name}: {exc}")

    _discovered = True


def get_scenario(name: str) -> BaseScenario:
    """Get an instantiated scenario by name."""
    _discover()
    if name not in _SCENARIOS:
        available = ", ".join(sorted(_SCENARIOS.keys()))
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return _SCENARIOS[name]()


def list_scenarios() -> list[dict]:
    """Return metadata for all discovered scenarios."""
    _discover()
    return [cls().to_metadata() for cls in _SCENARIOS.values()]
