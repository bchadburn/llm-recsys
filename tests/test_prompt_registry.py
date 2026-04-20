"""Tests for PromptRegistry hash/versioning behavior."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evals.prompt_registry import PromptRegistry


def test_same_template_returns_same_version():
    """Registering the same text twice must return the same version."""
    registry = PromptRegistry({})
    v1 = registry.register("my_prompt", "Hello {name}")
    v2 = registry.register("my_prompt", "Hello {name}")
    assert v1 == v2 == "v1"


def test_changed_template_increments_version():
    """A different template for the same name must get a new version."""
    registry = PromptRegistry({})
    v1 = registry.register("my_prompt", "Hello {name}")
    v2 = registry.register("my_prompt", "Hi {name}")
    assert v1 == "v1"
    assert v2 == "v2"


def test_current_versions_reflects_latest():
    """current_versions() must return the most recent version for each name."""
    registry = PromptRegistry({})
    registry.register("prompt_a", "Version one text")
    registry.register("prompt_a", "Version two text")
    registry.register("prompt_b", "Only version")
    versions = registry.current_versions()
    assert versions["prompt_a"] == "v2"
    assert versions["prompt_b"] == "v1"
