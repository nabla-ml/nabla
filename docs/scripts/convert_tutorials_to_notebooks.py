#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Convert example .py files to Jupyter notebooks
# ===----------------------------------------------------------------------=== #
"""Convert example Python files to Jupyter notebooks.

Parses the `# %% [markdown]` and `# %%` cell markers to create proper
notebook cells. Markdown cells extract content from comment blocks,
code cells contain the executable Python code.

Usage:
    venv/bin/python docs/scripts/convert_tutorials_to_notebooks.py
"""

import json
import re
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "examples"
DOCS_TUTORIALS_DIR = ROOT_DIR / "docs" / "examples"

ORDERED_NOTEBOOKS = [
    "01_tensors_and_ops",
    "02_autodiff",
    "03a_mlp_training_pytorch",
    "03b_mlp_training_jax",
    "04_transforms_and_compile",
    "05a_transformer_pytorch",
    "05b_transformer_jax",
    "06_mlp_pipeline_parallel",
    "07_mlp_pp_dp_training",
    "08_mlp_pipeline_inference",
    "09_jax_comparison_compiled",
    "10_lora_finetuning_mvp",
    "11_qlora_finetuning_mvp",
]


def _normalize_notebook_link(target: str) -> str:
    """Normalize internal markdown link targets to notebook-safe local links.

    Example:
    - "03a_mlp_training_pytorch.py" -> "03a_mlp_training_pytorch"
    - "./03a_mlp_training_pytorch.ipynb#sec" -> "03a_mlp_training_pytorch#sec"
    """
    if not target:
        return target

    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target) or target.startswith("mailto:"):
        return target
    if target.startswith("#"):
        return target

    path_part, hash_part = (target.split("#", 1) + [""])[:2]
    clean = Path(path_part).name

    if clean.endswith(".py") or clean.endswith(".ipynb"):
        clean = Path(clean).stem

    if hash_part:
        return f"{clean}#{hash_part}"
    return clean


def _normalize_and_validate_markdown_links(
    md_lines: list[str], valid_targets: set[str], source_name: str
) -> tuple[list[str], list[str]]:
    """Normalize markdown links and collect broken internal links."""

    broken: list[str] = []

    def repl(match: re.Match[str]) -> str:
        label = match.group(1)
        original_target = match.group(2).strip()
        normalized_target = _normalize_notebook_link(original_target)

        if (
            normalized_target
            and not normalized_target.startswith("#")
            and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", normalized_target)
            and not normalized_target.startswith("mailto:")
        ):
            local_path = normalized_target.split("#", 1)[0]
            if local_path and local_path not in valid_targets:
                broken.append(
                    f"{source_name}: unresolved local link '{original_target}' -> '{normalized_target}'"
                )

        return f"[{label}]({normalized_target})"

    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    normalized = [pattern.sub(repl, line) for line in md_lines]
    return normalized, broken


def _validate_python_sources(py_paths: list[Path]) -> None:
    """Validate syntax of source files before notebook generation."""
    errors: list[str] = []
    for path in py_paths:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.name}: line {exc.lineno}, col {exc.offset}: {exc.msg}")

    if errors:
        joined = "\n".join(errors)
        raise SyntaxError(f"Python syntax validation failed:\n{joined}")


def py_to_notebook(py_path: Path, valid_targets: set[str]) -> tuple[dict, list[str]]:
    """Convert a .py file with `# %%` markers to a Jupyter notebook dict."""
    content = py_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    cells = []
    current_cell_lines: list[str] = []
    current_cell_type: str | None = None
    broken_links: list[str] = []

    def flush_cell():
        nonlocal current_cell_lines, current_cell_type
        if current_cell_type is None or not current_cell_lines:
            current_cell_lines = []
            return

        if current_cell_type == "markdown":
            md_lines = []
            for line in current_cell_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)

            while md_lines and md_lines[0].strip() == "":
                md_lines.pop(0)
            while md_lines and md_lines[-1].strip() == "":
                md_lines.pop()

            md_lines, local_broken = _normalize_and_validate_markdown_links(
                md_lines, valid_targets=valid_targets, source_name=py_path.name
            )
            broken_links.extend(local_broken)

            if md_lines:
                source = [line + "\n" for line in md_lines]
                source[-1] = source[-1].rstrip("\n")
                cells.append(
                    {
                        "cell_type": "markdown",
                        "id": uuid.uuid4().hex[:8],
                        "metadata": {},
                        "source": source,
                    }
                )

        elif current_cell_type == "code":
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
                        "id": uuid.uuid4().hex[:8],
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": source,
                    }
                )

        current_cell_lines = []
        current_cell_type = None

    in_preamble = True

    for line in lines:
        if line.strip() == "# %% [markdown]":
            if in_preamble:
                in_preamble = False
            flush_cell()
            current_cell_type = "markdown"
            continue
        if line.strip() == "# %%":
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
    return notebook, broken_links


def main() -> None:
    example_files = [EXAMPLES_DIR / f"{name}.py" for name in ORDERED_NOTEBOOKS]
    example_files = [path for path in example_files if path.exists()]

    if not example_files:
        print("No example .py files found in examples/!")
        sys.exit(1)

    DOCS_TUTORIALS_DIR.mkdir(parents=True, exist_ok=True)

    _validate_python_sources(example_files)

    marker_warnings: list[str] = []
    for path in example_files:
        marker_count = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("# %%")
        )
        if marker_count < 2:
            marker_warnings.append(
                f"{path.name}: only {marker_count} cell marker(s); consider splitting into more notebook cells"
            )

    if marker_warnings:
        print("⚠️  Cell structure warnings:")
        for warning in marker_warnings:
            print(f"   - {warning}")
        print()

    valid_targets = {path.stem for path in example_files}

    print(f"Converting {len(example_files)} example(s) from examples/ to docs/examples/...\n")

    broken_links: list[str] = []
    for py_path in example_files:
        notebook, file_broken_links = py_to_notebook(py_path, valid_targets=valid_targets)
        broken_links.extend(file_broken_links)
        nb_path = DOCS_TUTORIALS_DIR / f"{py_path.stem}.ipynb"
        nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        n_cells = len(notebook["cells"])
        print(f"  {py_path.name} → {nb_path.name} ({n_cells} cells)")

    if broken_links:
        joined = "\n".join(f"  - {entry}" for entry in broken_links)
        raise ValueError(
            "Notebook conversion aborted due to broken internal links after normalization:\n"
            f"{joined}"
        )

    print(f"\nDone! {len(example_files)} notebooks created.")


if __name__ == "__main__":
    main()
