#!/usr/bin/env python
"""Regenerate the reference table in ``docs/04-configuration.md`` from the models.

The table is the one place a config change has to be mirrored by hand, so it
drifted: thresholds moved in code and the doc kept quoting the old ones, and a
whole block of V3 settings was never added at all. Generating it means the doc
cannot disagree with ``config.py`` again.

    python scripts/gen_config_docs.py            # rewrite the table
    python scripts/gen_config_docs.py --check    # exit 1 if it is out of date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pydantic import BaseModel  # noqa: E402

from dialogue_locator.config import Settings  # noqa: E402

DOC = REPO / "docs" / "04-configuration.md"
HEADER = "| Env variable | Field | Default | Type | Description |\n|---|---|---|---|---|\n"


def type_name(annotation: Any) -> str:
    """Render a field's annotation the way the table shows it."""
    if get_origin(annotation) is not None:
        args = get_args(annotation)
        if args and type(None) in args:
            inner = " | ".join(type_name(a) for a in args if a is not type(None))
            return f"{inner} | None"
        return str(annotation).replace("typing.", "")
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def cell(value: str) -> str:
    """Escape a value for a markdown table cell.

    ``logging.fmt`` contains pipes; unescaped they split the row into extra
    columns and silently truncate what the reader sees.
    """
    return value.replace("|", "\\|").replace("\n", " ").strip()


def rows(model: type[BaseModel], prefix: str = "", env: str = "DL_") -> list[str]:
    out: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        path = f"{prefix}{name}"
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out += rows(annotation, f"{path}.", f"{env}{name.upper()}__")
            continue
        description = cell(field.description or "")
        default = cell(str(field.default))
        out.append(
            f"| `{env}{name.upper()}` | `{path}` | `{default}` | "
            f"`{type_name(annotation)}` | {description} |"
        )
    return out


def render() -> str:
    return HEADER + "\n".join(rows(Settings)) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of rewriting")
    args = parser.parse_args()

    text = DOC.read_text()
    pattern = re.compile(
        re.escape(HEADER) + r"(?:\|.*\n)+", re.M
    )
    if not pattern.search(text):
        print(f"error: reference table not found in {DOC}", file=sys.stderr)
        return 2

    updated = pattern.sub(lambda _: render(), text, count=1)
    if args.check:
        if updated != text:
            print(f"{DOC} is out of date; run: python scripts/gen_config_docs.py", file=sys.stderr)
            return 1
        print(f"{DOC} is up to date")
        return 0
    if updated != text:
        DOC.write_text(updated)
        print(f"rewrote the reference table in {DOC}")
    else:
        print(f"{DOC} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
