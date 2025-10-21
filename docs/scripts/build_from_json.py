#!/usr/bin/env python3
"""Generate Markdown files from docs/structure.json

This script reads the JSON file that defines the docs hierarchy and emits
Markdown files into the `docs/api/` tree.

It works by dynamically importing the library's modules using the exact paths
defined in `structure.json` and inspecting the live objects (classes, functions)
to extract their docstrings, signatures, and methods.

Usage:
    # From the project root directory:
    python docs/scripts/build_from_json.py
"""

import json
import textwrap
import sys
import importlib
import inspect
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = DOCS_ROOT.parent
STRUCTURE_FILE = DOCS_ROOT / "structure.json"
API_ROOT = DOCS_ROOT / "api"
# --- End Configuration ---


def extract_docstring_data(full_path: str) -> dict | None:
    """
    Introspects a live object using its full definition path to extract documentation.

    Args:
        full_path: The full dotted path to the object (e.g., "nabla.core.tensor.Tensor").

    Returns:
        A dictionary with docstring, signature, and methods, or None if not found.
    """
    try:
        if '.' not in full_path:
            raise ImportError(f"Path '{full_path}' is not a valid object path.")
        
        module_path, object_name = full_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, object_name)
        
    except (ImportError, AttributeError) as e:
        print(f"  └─ ❌ ERROR: Could not find '{object_name}' in module '{module_path}'.")
        print(f"     Please ensure the path '{full_path}' is correct in `structure.json`.")
        print(f"     Original error: {e}")
        return None

    result = {}
    docstring = inspect.getdoc(obj)
    result['docstring'] = textwrap.dedent(docstring).strip() if docstring else "*No docstring found.*"

    try:
        result['signature'] = str(inspect.signature(obj))
    except (ValueError, TypeError):
        result['signature'] = "()"

    if inspect.isclass(obj):
        result['methods'] = []
        original_module_name = obj.__module__
        for name, member in inspect.getmembers(obj):
            if not name.startswith('_') and inspect.isfunction(member) and member.__module__ == original_module_name:
                method_doc = inspect.getdoc(member)
                result['methods'].append({
                    'name': name,
                    'signature': str(inspect.signature(member)),
                    'docstring': textwrap.dedent(method_doc).strip() if method_doc else "*No docstring found.*"
                })
    return result


def ensure_dir(p: Path):
    """Create a directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def write_md(path: Path, lines: list[str]):
    """Write a list of strings to a Markdown file."""
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf8")
    print(f"✓ Wrote {path}")


def generate_api_index(api_section: dict):
    lines = [f"# {api_section.get('title', 'API Reference')}", ""]
    for mod in api_section.get("modules", []):
        lines.append(f"## [{mod['title']}]({mod['id']}/)")
    write_md(API_ROOT / "index.md", lines)


def generate_module_index(module_path: Path, module: dict):
    lines = [f"# {module.get('title', module['id'].capitalize())}", ""]
    if module.get('description'):
        lines.extend([module['description'], ""])
    for subsection in module.get('subsections', []):
        lines.append(f"- [{subsection['title']}]({subsection['id']}.md)")
    ensure_dir(module_path)
    write_md(module_path / "index.md", lines)


def generate_subsection_md(module_path: Path, subsection: dict):
    """Generates the final Markdown file with embedded docstrings."""
    lines = [f"# {subsection.get('title', subsection['id'].capitalize())}", ""]
    if subsection.get("description"):
        lines.extend([subsection["description"], ""])

    for item in subsection.get("items", []):
        item_path = item["path"]
        item_type = item.get("type", "function")
        print(f"  -> Processing '{item_path}'...")

        lines.extend([f"## `{item['name']}`", ""])
        data = extract_docstring_data(item_path)

        if data:
            signature = data.get('signature', '()')
            lines.append("```python")
            lines.append(f"{'class' if item_type == 'class' else 'def'} {item['name']}{signature}:")
            lines.append("```")
            lines.append(data.get('docstring', '*No docstring found.*'))
            if item_type == 'class' and item.get("show_methods") and data.get("methods"):
                lines.append("\n### Methods")
                for method in sorted(data["methods"], key=lambda m: m['name']):
                    lines.append(f"\n#### `{method['name']}`")
                    lines.append("```python")
                    lines.append(f"def {method['name']}{method['signature']}:")
                    lines.append("```")
                    lines.append(method.get('docstring', '*No docstring found.*'))
        else:
            lines.append("*Could not extract documentation. Please check the error messages above and correct `structure.json`.*")
        lines.append("\n---")

    write_md(module_path / f"{subsection['id']}.md", lines)


def run():
    # This is the critical step that allows `importlib` to find your 'nabla' module.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print("Starting Markdown documentation generation from structure.json...")

    with open(STRUCTURE_FILE, 'r', encoding='utf8') as fh:
        structure = json.load(fh)

    ensure_dir(API_ROOT)

    for section in structure.get('sections', []):
        if section.get('type') == 'api':
            print(f"\nProcessing API section: '{section['id']}'")
            generate_api_index(section)
            for module in section.get("modules", []):
                print(f"  Processing module: '{module['id']}'")
                module_path = API_ROOT / module['id']
                generate_module_index(module_path, module)
                for subsection in module.get("subsections", []):
                    generate_subsection_md(module_path, subsection)

    print('\nAll done. Review generated Markdown files under docs/api/')


if __name__ == '__main__':
    run()