from typing import Any

from chatree.sdk.client import Chatree, DialogueTree, NodeHandle

_EVAL_EXPORTS = {
    "BaselineMode",
    "BenchmarkReport",
    "BenchmarkRunner",
    "ModeBenchmarkResult",
    "MultiTangentScenario",
    "ScenarioStep",
    "default_multi_tangent_scenario",
}


def __getattr__(name: str) -> Any:
    if name in _EVAL_EXPORTS:
        from chatree.sdk import eval as _sdk_eval

        return getattr(_sdk_eval, name)
    raise AttributeError(f"module 'chatree.sdk' has no attribute {name!r}")


__all__ = [
    "BaselineMode",
    "BenchmarkReport",
    "BenchmarkRunner",
    "Chatree",
    "DialogueTree",
    "ModeBenchmarkResult",
    "MultiTangentScenario",
    "NodeHandle",
    "ScenarioStep",
    "default_multi_tangent_scenario",
]
