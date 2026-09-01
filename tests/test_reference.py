from __future__ import annotations

from pathlib import Path

import numpy as np

import kohonen
from kohonen._reference import reference_train


def test_reference_is_deterministic_given_initial_weights() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((5, 3))
    w0 = rng.random((4, 4, 3))

    a = reference_train(data, 10, 4, 4, w0.copy())
    b = reference_train(data, 10, 4, 4, w0.copy())

    np.testing.assert_array_equal(a, b)


def test_reference_does_not_mutate_caller_weights() -> None:
    rng = np.random.default_rng(1)
    data = rng.random((3, 3))
    w0 = rng.random((4, 4, 3))
    original = w0.copy()

    reference_train(data, 5, 4, 4, w0)

    np.testing.assert_array_equal(w0, original)


def test_reference_shape_matches_grid() -> None:
    rng = np.random.default_rng(2)
    data = rng.random((3, 5))
    w0 = rng.random((6, 4, 5))

    out = reference_train(data, 3, 6, 4, w0)

    assert out.shape == (6, 4, 5)


def test_reference_is_not_imported_by_package_modules() -> None:
    """The oracle must stay quarantined from shipped code."""
    package_dir = Path(kohonen.__file__).parent
    offenders = [
        path.name
        for path in package_dir.glob("*.py")
        if path.name != "_reference.py" and "_reference" in path.read_text()
    ]
    assert offenders == [], f"_reference is imported by: {offenders}"
