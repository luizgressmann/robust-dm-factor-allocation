"""Check notebooks for saved outputs and execution state."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = REPOSITORY_ROOT / "notebooks"


def notebook_issues(path: Path) -> list[str]:
    """List saved state and missing cell IDs."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    cell_ids_required = notebook.get("nbformat", 0) >= 4 and notebook.get("nbformat_minor", 0) >= 5

    for number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell_ids_required and not cell.get("id"):
            issues.append(f"cell {number} has no stable id")
        if cell.get("attachments"):
            issues.append(f"cell {number} contains an attachment")
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            issues.append(f"cell {number} has an execution count")
        if cell.get("outputs"):
            issues.append(f"cell {number} contains saved output")

    return issues


def main() -> int:
    failures: list[str] = []
    for path in sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb")):
        for issue in notebook_issues(path):
            failures.append(f"{path.relative_to(REPOSITORY_ROOT)}: {issue}")

    if failures:
        print("Notebooks must not contain saved outputs:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Notebook check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
