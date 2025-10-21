#!/usr/bin/env python3
"""Generate Sphinx RST files from docs/structure.json

This script reads the JSON file that defines the docs hierarchy and emits
reStructuredText files into the `docs/` tree, particularly under `docs/api/`.

Usage:
    cd docs
    ../venv/bin/python scripts/build_from_json.py

The script is intentionally small and avoids importing Sphinx directly.
"""

import json
import importlib
import inspect
from pathlib import Path
import textwrap

ROOT = Path(__file__).parent.parent
STRUCTURE_FILE = ROOT / "structure.json"
API_ROOT = ROOT / "api"


def slugify(name: str) -> str:
    return name.replace(" ", "_").lower()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_rst(path: Path, lines: list[str]):
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf8")
    print(f"✓ Wrote {path}")


def generate_api_index(api_section: dict):
    lines = []
    title = api_section.get("title", "API Reference")
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append(".. toctree::")
    lines.append("   :maxdepth: 2")
    lines.append("")

    for mod in api_section.get("modules", []):
        module_dir = f"api/{mod['id']}/index"
        lines.append(f"   {module_dir}")

    write_rst(API_ROOT / "index.rst", lines)


def generate_module_index(module_path: Path, module: dict):
    lines = []
    title = module.get("title", module['id'].capitalize())
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    if module.get('description'):
        lines.append(module['description'])
        lines.append("")

    lines.append(".. toctree::")
    lines.append("   :maxdepth: 1")
    lines.append("")

    for subsection in module.get('subsections', []):
        lines.append(f"   {subsection['id']}")

    ensure_dir(module_path)
    write_rst(module_path / "index.rst", lines)


def generate_subsection_rst(module_path: Path, subsection: dict):
    lines = []
    title = subsection.get('title', subsection['id'].capitalize())
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    if subsection.get('description'):
        lines.append(subsection['description'])
        lines.append("")

    lines.append(".. currentmodule:: nabla")
    lines.append("")

    for item in subsection.get('items', []):
        name = item['name']
        lines.append(name)
        lines.append("-" * len(name))
        lines.append("")

        # Try to inline the docstring so generated files are readable without
        # needing to run Sphinx. We still keep the autodoc directive below.
        doc_text = None
        obj_path = item.get('path')
        if obj_path:
            try:
                # Import the module portion and resolve attribute chain
                module_name, _, attr = obj_path.rpartition('.')
                module = importlib.import_module(module_name)
                obj = getattr(module, attr) if attr else module
                doc_text = inspect.getdoc(obj) or None
            except Exception:
                # If import fails (mocked modules, C extensions), skip inlining
                doc_text = None

        if doc_text:
            # Format docstring: convert doctest-style examples to code-blocks
            def format_docstring_to_rst(s: str) -> list[str]:
                out = []
                lines = s.split('\n')
                i = 0
                in_code = False
                code_buf = []
                while i < len(lines):
                    line = lines[i]
                    stripped = line.strip()
                    if stripped.startswith('>>>') or stripped.startswith('...'):
                        # start a code block
                        code_buf.append(line)
                        in_code = True
                    else:
                        if in_code:
                            # flush code buffer as a code-block
                            out.append('.. code-block:: python')
                            out.append('')
                            for cl in code_buf:
                                out.append(f'    {cl}')
                            out.append('')
                            code_buf = []
                            in_code = False
                        out.append(line)
                    i += 1

                if in_code and code_buf:
                    out.append('.. code-block:: python')
                    out.append('')
                    for cl in code_buf:
                        out.append(f'    {cl}')
                    out.append('')

                return out

            formatted = format_docstring_to_rst(doc_text)
            # Add a short heading to separate the prose from the autodoc
            lines.append('Description')
            lines.append('-----------')
            lines.append('')
            lines.extend(formatted)
            lines.append('')

        # Add autodoc directive for the canonical reference
        if item['type'] == 'class':
            lines.append(f".. autoclass:: {item['path']}")
            if item.get('show_methods'):
                lines.append('   :members:')
                lines.append('   :undoc-members:')
                lines.append('   :show-inheritance:')
        else:
            lines.append(f".. autofunction:: {item['path']}")

        lines.append("")

    write_rst(module_path / f"{subsection['id']}.rst", lines)


def generate_notebooks_index(section: dict):
    path = ROOT / section['path']
    ensure_dir(path)
    lines = []
    title = section.get('title', 'Tutorials')
    lines.append(title)
    lines.append('=' * len(title))
    lines.append('')
    lines.append('.. toctree::')
    lines.append('   :maxdepth: 1')
    lines.append('')
    for f in section.get('files', []):
        lines.append(f"   {f}")
    write_rst(path / 'index.rst', lines)


def run():
    with open(STRUCTURE_FILE, 'r', encoding='utf8') as fh:
        structure = json.load(fh)

    # Create API root
    ensure_dir(API_ROOT)

    for section in structure.get('sections', []):
        t = section.get('type')
        if t == 'api':
            generate_api_index(section)
            for module in section.get('modules', []):
                module_path = API_ROOT / module['id']
                ensure_dir(module_path)
                generate_module_index(module_path, module)
                for subsection in module.get('subsections', []):
                    generate_subsection_rst(module_path, subsection)
        elif t == 'notebooks':
            generate_notebooks_index(section)
        else:
            # manual sections are left untouched
            continue

    print('\nAll done. Review generated files under docs/api/')


if __name__ == '__main__':
    run()
