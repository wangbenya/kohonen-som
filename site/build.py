"""Inline the benchmark and snapshot data into the site template.

The published page must be self-contained -- its content-security policy blocks
every external request -- and every figure on it must trace to committed output.
Both fall out of generating the page from the JSON rather than typing numbers
into the HTML.

    uv run --extra dev python site/build.py
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "site" / "template.html"
OUTPUT = ROOT / "site" / "index.html"
IMAGE = ROOT / "site" / "image.png"
COMPARISON = ROOT / "benchmarks" / "results" / "comparison.json"
SNAPSHOTS = ROOT / "benchmarks" / "results" / "snapshots.json"

PYTHON_MATRIX = "3.10-3.14"
REPO = "wangbenya/kohonen-som"


def _sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _test_stats() -> tuple[str, str]:
    """Read the test count and coverage from a real run, never a literal."""
    result = subprocess.run(
        ["uv", "run", "--extra", "dev", "pytest", "--no-header", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Refusing to build: the test suite is not green.\n" + result.stdout[-2000:]
        )
    tests = coverage = "unknown"
    for line in result.stdout.splitlines():
        if " passed" in line:
            tests = line.split(" passed")[0].strip().split()[-1]
        if "Total coverage:" in line:
            coverage = line.split("Total coverage:")[1].strip()
    return tests, coverage


def main() -> None:
    for path in (COMPARISON, SNAPSHOTS):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the benchmark scripts first.")
    if not IMAGE.exists():
        raise SystemExit(f"Missing {IMAGE}.")

    # The page's CSP blocks every external request, so the figure is inlined as
    # a data URI rather than referenced.
    image_uri = "data:image/png;base64," + base64.b64encode(IMAGE.read_bytes()).decode(
        "ascii"
    )

    tests, coverage = _test_stats()
    meta = {
        "sha": _sha(),
        "repo": REPO,
        "tests": tests,
        "coverage": coverage,
        "python_matrix": PYTHON_MATRIX,
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    for token, payload in (
        ("__COMPARISON_JSON__", COMPARISON.read_text(encoding="utf-8")),
        ("__SNAPSHOTS_JSON__", SNAPSHOTS.read_text(encoding="utf-8")),
        ("__META_JSON__", json.dumps(meta)),
        ("__IMAGE_URI__", image_uri),
    ):
        if token not in html:
            raise SystemExit(f"Template is missing the {token} placeholder.")
        # str.replace, not re.sub: a JSON payload can contain a backslash, which
        # a regex replacement string would interpret as an escape.
        html = html.replace(token, payload)

    OUTPUT.write_text(html, encoding="utf-8")
    size = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({size:.0f} KB)")
    print(f"  commit {meta['sha']}, {tests} tests, {coverage} coverage")


if __name__ == "__main__":
    main()
