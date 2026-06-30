"""
registry.py

Generic registry system used for registering optimizers,
benchmarks, and future framework components.

Supports both method-call and decorator-style registration.
"""

from typing import Dict, Type


class Registry:
    """
    Generic registry class.

    Usage as decorator:
        @optimizer_registry.register
        class GWO(BaseOptimizer):
            ...

    Usage as method:
        optimizer_registry.register(GWO)
        gwo_cls = optimizer_registry.get("GWO")
    """

    def __init__(self, registry_name: str):
        self.registry_name = registry_name
        self._registry: Dict[str, Type] = {}

    def register(self, cls):
        """
        Register a class using its class name.
        Can be used as a decorator or called directly.
        """

        name = cls.__name__

        if name in self._registry:
            raise ValueError(
                f"{name} already exists in {self.registry_name} Registry."
            )

        self._registry[name] = cls

        return cls

    def get(self, name: str):
        """
        Retrieve a registered class by name.
        """

        if name not in self._registry:
            raise KeyError(
                f"{name} is not registered in {self.registry_name} Registry. "
                f"Available: {self.list()}"
            )

        return self._registry[name]

    def exists(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def list(self):
        """List all registered names."""
        return sorted(self._registry.keys())

    def __len__(self):
        return len(self._registry)

    def __contains__(self, name: str):
        return name in self._registry

    def __repr__(self):
        return (
            f"{self.registry_name}Registry"
            f"({list(self._registry.keys())})"
        )


# ============================================
# Global registry singletons
# ============================================

optimizer_registry = Registry("Optimizer")

benchmark_registry = Registry("Benchmark")
