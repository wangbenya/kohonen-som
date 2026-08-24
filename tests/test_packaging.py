from __future__ import annotations

import kohonen


def test_package_exposes_version() -> None:
    assert isinstance(kohonen.__version__, str)
    assert kohonen.__version__.count(".") >= 2
