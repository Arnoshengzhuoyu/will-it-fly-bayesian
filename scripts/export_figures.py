"""Pull the figures out of the executed notebooks into figures/, as standalone PNGs.

The notebooks are where the figures are made and explained; this just makes them easy to drop
into a report or an email without exporting the whole notebook. Each file is named after the
notebook and the section heading it sits under, so the name says what the figure shows.

    python scripts/export_figures.py            # all notebooks
    python scripts/export_figures.py --clean    # delete figures/ first
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
FIGURES = ROOT / "figures"


def slug(text: str, limit: int = 48) -> str:
    """A filename-safe fragment of a heading."""
    text = re.sub(r"[`*_$\\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:limit].strip("-")


def heading_of(cells: list[dict], upto: int) -> str:
    """The nearest markdown heading above cell `upto`."""
    for cell in reversed(cells[:upto]):
        if cell["cell_type"] != "markdown":
            continue
        for line in "".join(cell["source"]).splitlines():
            if line.lstrip().startswith("#"):
                return line.lstrip("# ").strip()
    return ""


def export(path: Path) -> list[Path]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    stem = path.stem                       # e.g. 03_model1_logistic
    written: list[Path] = []
    seen: dict[str, int] = {}

    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if not png:
                continue
            name = slug(heading_of(cells, i)) or "figure"
            seen[name] = seen.get(name, 0) + 1
            suffix = f"-{seen[name]}" if seen[name] > 1 else ""
            dest = FIGURES / f"{stem}__{name}{suffix}.png"
            dest.write_bytes(base64.b64decode(png))
            written.append(dest)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true", help="remove figures/ before writing")
    args = ap.parse_args()

    if args.clean and FIGURES.exists():
        shutil.rmtree(FIGURES)
    FIGURES.mkdir(exist_ok=True)

    total = 0
    for nb in sorted(NOTEBOOKS.glob("*.ipynb")):
        files = export(nb)
        total += len(files)
        if files:
            print(f"{nb.name}")
            for f in files:
                print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")
        else:
            print(f"{nb.name}  — no figures (not executed?)")

    print(f"\n{total} figures in {FIGURES}")


if __name__ == "__main__":
    main()
