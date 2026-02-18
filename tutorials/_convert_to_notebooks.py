#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Convert tutorial .py files to Jupyter notebooks
# ===----------------------------------------------------------------------=== #
"""Convert tutorial Python files to Jupyter notebooks.

Parses the `# %% [markdown]` and `# %%` cell markers to create proper
notebook cells. Markdown cells extract content from comment blocks,
code cells contain the executable Python code.

Usage:
    python tutorials/_convert_to_notebooks.py
"""

import json
import re
import sys
from pathlib import Path

TUTORIALS_DIR = Path(__file__).parent


def py_to_notebook(py_path: Path) -> dict:
    """Convert a .py file with `# %%` markers to a Jupyter notebook dict."""
    content = py_path.read_text()
    lines = content.split("\n")

    cells = []
    current_cell_lines: list[str] = []
    current_cell_type: str | None = None

    def flush_cell():
        nonlocal current_cell_lines, current_cell_type
        if current_cell_type is None or not current_cell_lines:
            current_cell_lines = []
            return

        if current_cell_type == "markdown":
            # Strip leading `# ` from markdown lines
            md_lines = []
            for line in current_cell_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)

            # Remove leading/trailing blank lines
            while md_lines and md_lines[0].strip() == "":
                md_lines.pop(0)
            while md_lines and md_lines[-1].strip() == "":
                md_lines.pop()

            if md_lines:
                source = [line + "\n" for line in md_lines]
                source[-1] = source[-1].rstrip("\n")
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": source,
                    }
                )

        elif current_cell_type == "code":
            # Remove leading/trailing blank lines
            while current_cell_lines and current_cell_lines[0].strip() == "":
                current_cell_lines.pop(0)
            while current_cell_lines and current_cell_lines[-1].strip() == "":
                current_cell_lines.pop()

            if current_cell_lines:
                source = [line + "\n" for line in current_cell_lines]
                source[-1] = source[-1].rstrip("\n")
                cells.append(
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": source,
                    }
                )

        current_cell_lines = []
        current_cell_type = None

    # Skip module docstring and header lines (before first cell marker)
    in_preamble = True

    for line in lines:
        # Detect cell markers
        if line.strip() == "# %% [markdown]":
            if in_preamble:
                in_preamble = False
            flush_cell()
            current_cell_type = "markdown"
            continue
        elif line.strip() == "# %%":
            if in_preamble:
                in_preamble = False
            flush_cell()
            current_cell_type = "code"
            continue

        if in_preamble:
            continue

        if current_cell_type is not None:
            current_cell_lines.append(line)

    flush_cell()

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "cells": cells,
    }
    return notebook


def main():
    tutorial_files = sorted(TUTORIALS_DIR.glob("[0-9]*.py"))

    if not tutorial_files:
        print("No tutorial .py files found!")
        sys.exit(1)

    print(f"Converting {len(tutorial_files)} tutorial(s) to notebooks...\n")

    for py_path in tutorial_files:
        notebook = py_to_notebook(py_path)
        nb_path = py_path.with_suffix(".ipynb")
        nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        n_cells = len(notebook["cells"])
        print(f"  {py_path.name} → {nb_path.name} ({n_cells} cells)")

    print(f"\nDone! {len(tutorial_files)} notebooks created.")


if __name__ == "__main__":
    main()
