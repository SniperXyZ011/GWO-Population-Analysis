"""
test_registry.py

Tests for core/registry.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.registry import Registry


class TestRegistry:

    def test_register_class(self):
        reg = Registry("Test")

        class Foo:
            pass

        reg.register(Foo)
        assert reg.exists("Foo")

    def test_register_as_decorator(self):
        reg = Registry("Test")

        @reg.register
        class Bar:
            pass

        assert reg.exists("Bar")
        assert Bar.__name__ == "Bar"

    def test_get_registered_class(self):
        reg = Registry("Test")

        class Baz:
            pass

        reg.register(Baz)
        retrieved = reg.get("Baz")
        assert retrieved is Baz

    def test_get_unregistered_raises(self):
        reg = Registry("Test")

        with pytest.raises(KeyError, match="NotRegistered"):
            reg.get("NotRegistered")

    def test_duplicate_registration_raises(self):
        reg = Registry("Test")

        class Dup:
            pass

        reg.register(Dup)

        with pytest.raises(ValueError, match="already exists"):
            reg.register(Dup)

    def test_list_returns_sorted(self):
        reg = Registry("Test")

        class Zebra:
            pass

        class Alpha:
            pass

        reg.register(Zebra)
        reg.register(Alpha)

        assert reg.list() == ["Alpha", "Zebra"]

    def test_len(self):
        reg = Registry("Test")

        class A:
            pass

        class B:
            pass

        reg.register(A)
        reg.register(B)

        assert len(reg) == 2

    def test_contains(self):
        reg = Registry("Test")

        class C:
            pass

        reg.register(C)

        assert "C" in reg
        assert "D" not in reg

    def test_repr(self):
        reg = Registry("Test")
        assert "TestRegistry" in repr(reg)

    def test_empty_registry(self):
        reg = Registry("Empty")
        assert len(reg) == 0
        assert reg.list() == []
