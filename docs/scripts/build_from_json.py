#!/usr/bin/env python3
"""Generate Sphinx RST files from docs/structure.json

This script reads the JSON file that defines the docs hierarchy and emits
reStructuredText files into the `docs/` tree, particularly under `docs/api/`.

It generates the appropriate 'autodoc' directives, and Sphinx handles the
actual docstring extraction during the build process. This is the standard
and most robust way to build API documentation with Sphinx.

Usage:
    # From the project root directory:
    python docs/scripts/build_from_json.py
"""

import json
from pathlib import Path

# This script assumes the following directory structure:
# your_project_root/
# ├── docs/
# │   ├── scripts/
# │   │   └── build_from_json.py  <-- THIS SCRIPT
# │   ├── structure.json
# │   └── conf.py
# └── nabla/
#     └── ... (your source code)

# Path to the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Path to the 'docs' directory
DOCS_ROOT = SCRIPT_DIR.parent
# Path to the project's root directory
PROJECT_ROOT = DOCS_ROOT.parent
# Path to the configuration and output directories
STRUCTURE_FILE = DOCS_ROOT / "structure.json"
API_ROOT = DOCS_ROOT / "api"


def ensure_dir(p: Path):
    """Create a directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
    print(f"✓ Ensured directory exists: {p}")


def write_rst(path: Path, lines: list[str]):
    """Write a list of strings to an RST file."""
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf8")
    print(f"✓ Wrote {path}")


def generate_api_index(api_section: dict):
    """Generates the main api/index.rst file."""
    lines = []
    title = api_section.get("title", "API Reference")
    lines.extend([title, "=" * len(title), ""])
    lines.extend([".. toctree::", "   :maxdepth: 2", ""])

    for mod in api_section.get("modules", []):
        # Path should be relative to the api/index.rst file
        module_path = f"{mod['id']}/index"
        lines.append(f"   {module_path}")

    write_rst(API_ROOT / "index.rst", lines)


def generate_module_index(module_path: Path, module: dict):
    """Generates the index.rst for a specific module (e.g., api/core/index.rst)."""
    lines = []
    title = module.get("title", module['id'].capitalize())
    lines.extend([title, "=" * len(title), ""])
    if module.get('description'):
        lines.extend([module['description'], ""])

    lines.extend([".. toctree::", "   :maxdepth: 1", ""])

    for subsection in module.get('subsections', []):
        lines.append(f"   {subsection['id']}")

    ensure_dir(module_path)
    write_rst(module_path / "index.rst", lines)


def generate_subsection_rst(module_path: Path, subsection: dict):
    """Generates the RST file for a subsection of a module (e.g., api/core/tensor.rst)."""
    lines = []
    title = subsection.get("title", subsection['id'].capitalize())
    lines.extend([title, "=" * len(title), ""])
    if subsection.get("description"):
        lines.extend([subsection["description"], ""])

    for item in subsection.get("items", []):
        name = item["name"]
        item_type = item.get("type", "function")  # Default to 'function' if not specified
        item_path = item["path"] # e.g., "nabla.core.tensor.Tensor"

        # Add the item title and the autodoc directive
        lines.extend([name, "-" * len(name), ""])
        lines.append(f".. auto{item_type}:: {item_path}")

        # Add common options for classes to document their methods
        if item_type == 'class' and item.get('show_methods', True): # Default to showing methods for classes
            lines.append("   :members:")
            lines.append("   :undoc-members:")
            lines.append("   :show-inheritance:")

        lines.append("")  # Add a blank line for proper spacing between entries

    write_rst(module_path / f"{subsection['id']}.rst", lines)


def generate_notebooks_index(section: dict):
    """Generates the index file for notebooks/tutorials."""
    path = DOCS_ROOT / section['path']
    ensure_dir(path)
    lines = []
    title = section.get('title', 'Tutorials')
    lines.extend([title, "=" * len(title), ""])
    lines.extend([".. toctree::", "   :maxdepth: 1", ""])
    for f in section.get('files', []):
        lines.append(f"   {f}")
    write_rst(path / 'index.rst', lines)


def run():
    """Main execution function to generate all documentation files."""
    print("Starting documentation generation from structure.json...")

    try:
        with open(STRUCTURE_FILE, 'r', encoding='utf8') as fh:
            structure = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: Could not find structure file at {STRUCTURE_FILE}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse {STRUCTURE_FILE}. Please check for syntax errors.")
        return

    # Create the main API output directory
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
        elif t == 'manual':
            # Manual sections are written by hand and are skipped by this script
            print(f"i Skipping manual section: {section['id']}")
            continue
        else:
            print(f"WARNING: Unknown section type '{t}' for section '{section['id']}'. Skipping.")

    print(f'\nAll done. Review generated files under {API_ROOT}')
    print('Next, run `sphinx-build` to build your HTML documentation.')


if __name__ == '__main__':
    run()