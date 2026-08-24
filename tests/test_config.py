from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from kohonen import TrainingConfig


def test_defaults_match_the_brief() -> None:
    cfg = TrainingConfig()
    assert cfg.n_iterations == 100
    assert cfg.initial_learning_rate == 0.1
    assert cfg.initial_radius is None
    assert cfg.seed is None
    assert cfg.radius_cutoff is None


def test_config_is_frozen() -> None:
    cfg = TrainingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_iterations = 5  # type: ignore[misc]


def test_config_is_hashable_for_experiment_logging() -> None:
    assert hash(TrainingConfig()) == hash(TrainingConfig())


def test_resolved_radius_defaults_to_half_the_longest_side() -> None:
    assert TrainingConfig().resolved_radius(10, 4) == 5.0


def test_resolved_radius_honours_explicit_override() -> None:
    assert TrainingConfig(initial_radius=2.5).resolved_radius(10, 4) == 2.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iterations": -1}, "n_iterations"),
        ({"initial_learning_rate": 0.0}, "initial_learning_rate"),
        ({"initial_learning_rate": -0.5}, "initial_learning_rate"),
        ({"initial_radius": 0.0}, "initial_radius"),
        ({"radius_cutoff": 0.0}, "radius_cutoff"),
    ],
)
def test_invalid_values_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_zero_iterations_is_permitted() -> None:
    assert TrainingConfig(n_iterations=0).n_iterations == 0
