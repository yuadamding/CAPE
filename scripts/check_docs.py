"""Check local Markdown links without network access."""

from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for document in [root / "README.md", *sorted((root / "docs").rglob("*.md"))]:
        text = document.read_text()
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (document.parent / relative).resolve().exists():
                missing.append(f"{document.relative_to(root)} -> {target}")
    if missing:
        raise SystemExit("Missing documentation links:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
