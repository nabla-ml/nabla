#!/usr/bin/env python3
"""Generate Markdown files from docs/structure.json

This script reads the JSON file that defines the docs hierarchy and emits
Markdown files into the `docs/api/` tree.

It uses `docstring-parser` to parse NumPy-style docstrings into a structured
format, and then generates high-quality Markdown with proper sections and
syntax-highlighted code blocks. This version includes robust error handling
for malformed or incomplete docstring examples.

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
from docstring_parser import parse

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
    docstring_text = inspect.getdoc(obj)
    result['docstring_obj'] = parse(docstring_text) if docstring_text else parse("")

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
                    'docstring_obj': parse(method_doc) if method_doc else parse("")
                })
    return result


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_md(path: Path, lines: list[str]):
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf8")
    print(f"✓ Wrote {path}")


def generate_api_index(api_section: dict):
    title = api_section.get("title", "API Reference")
    lines = [f"# {title}", "", "```{toctree}", ":maxdepth: 2", ""]
    for mod in api_section.get("modules", []):
        lines.append(f"{mod['id']}/index")
    lines.append("```")
    write_md(API_ROOT / "index.md", lines)


def generate_module_index(module_path: Path, module: dict):
    title = module.get("title", module['id'].capitalize())
    lines = [f"# {title}", ""]
    if module.get('description'):
        lines.extend([module['description'], ""])
    lines.extend(["```{toctree}", ":maxdepth: 1", ""])
    for subsection in module.get('subsections', []):
        lines.append(f"{subsection['id']}")
    lines.append("```")
    ensure_dir(module_path)
    write_md(module_path / "index.md", lines)


def format_docstring_obj_to_md(docstring_obj) -> list[str]:
    """
    Takes a parsed docstring object and converts it to Markdown lines.
    This version includes robust error handling.
    """
    md_lines = []
    if docstring_obj.short_description:
        md_lines.extend([docstring_obj.short_description, ""])
    if docstring_obj.long_description:
        md_lines.extend([docstring_obj.long_description, ""])
    
    if docstring_obj.params:
        md_lines.extend(["**Parameters**", ""])
        for param in docstring_obj.params:
            type_info = f" : `{param.type_name}`" if param.type_name else ""
            default_info = f", default: `{param.default}`" if param.default else ""
            optional_info = ", optional" if param.is_optional else ""
            line = f"- **`{param.arg_name}`**{type_info}{optional_info}{default_info} – {param.description}"
            md_lines.append(line)
        md_lines.append("")

    if docstring_obj.returns:
        md_lines.extend(["**Returns**", ""])
        type_info = f"`{docstring_obj.returns.type_name}`" if docstring_obj.returns.type_name else ""
        line = f"{type_info} – {docstring_obj.returns.description}"
        md_lines.append(line)
        md_lines.append("")
        
    if docstring_obj.examples:
        md_lines.extend(["**Examples**", ""])
        for i, example in enumerate(docstring_obj.examples):
            if example.description:
                md_lines.extend(example.description.split('\n'))
            
            # --- THIS IS THE FIX ---
            # Check if the snippet exists before trying to access it.
            if example.snippet:
                md_lines.append("```python")
                md_lines.append(example.snippet.strip())
                md_lines.append("```")
            
            md_lines.append("")

    return md_lines


def generate_subsection_md(module_path: Path, subsection: dict):
    """
    Generates the final content Markdown file with properly formatted docstrings.
    """
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
            
            lines.extend(format_docstring_obj_to_md(data['docstring_obj']))
            
            if item_type == 'class' and item.get("show_methods") and data.get("methods"):
                lines.append("\n### Methods")
                for method in sorted(data["methods"], key=lambda m: m['name']):
                    lines.append(f"\n#### `{method['name']}`")
                    lines.append("```python")
                    lines.append(f"def {method['name']}{method['signature']}:")
                    lines.append("```")
                    lines.extend(format_docstring_obj_to_md(method['docstring_obj']))
        else:
            lines.append("*Could not extract documentation. Please check the error messages above and correct `structure.json`.*")
        lines.append("\n---")

    write_md(module_path / f"{subsection['id']}.md", lines)


def run():
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