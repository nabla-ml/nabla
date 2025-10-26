#!/usr/bin/env python3
"""Generate Markdown files from docs/structure.json

This script reads the JSON file that defines the docs hierarchy and emits
Markdown files into the `docs/api/` tree.

It can generate both module directories (with an index and subsections) and
top-level single pages for important objects like the Tensor.

Usage:
    # From the project root directory:
    python docs/scripts/build_from_json.py
"""

import json
import textwrap
import sys
import importlib
import inspect
import re
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
    result['raw_docstring'] = docstring_text
    docstring_obj = parse(textwrap.dedent(docstring_text)) if docstring_text else parse("")
    result['docstring_obj'] = docstring_obj

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
                parsed_method_doc = parse(textwrap.dedent(method_doc)) if method_doc else parse("")
                result['methods'].append({
                    'name': name,
                    'signature': str(inspect.signature(member)),
                    'docstring_obj': parsed_method_doc,
                    'raw_docstring': method_doc,
                })
    return result


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_md(path: Path, lines: list[str]):
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf8")
    print(f"✓ Wrote {path}")


def format_docstring_obj_to_md(docstring_obj, raw_docstring: str | None) -> list[str]:
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

    if raw_docstring and "Examples" in raw_docstring:
        try:
            _, examples_section = re.split(r'Examples\n\s*[-=]+', raw_docstring, maxsplit=1)
            example_lines = textwrap.dedent(examples_section).strip().split('\n')
            md_lines.extend(["**Examples**", ""])
            in_code_block = False
            for line in example_lines:
                is_code_start = line.strip().startswith('>>>')
                is_blank_line = not line.strip()
                if is_code_start and not in_code_block:
                    md_lines.append("```python")
                    md_lines.append(line)
                    in_code_block = True
                elif in_code_block:
                    if is_blank_line:
                        md_lines.append("```")
                        md_lines.append("")
                        in_code_block = False
                    else:
                        md_lines.append(line)
                else:
                    md_lines.append(line)
            if in_code_block:
                md_lines.append("```")
        except (ValueError, re.error):
            md_lines.append("**Examples**\n\n*Warning: Could not parse examples correctly.*")
    return md_lines


def generate_page_from_items(lines: list[str], items: list[dict]):
    """Helper to populate a page with documentation for a list of items."""
    for item in items:
        item_path = item["path"]
        item_type = item.get("type", "function")
        print(f"    -> Processing item '{item_path}'...")
        lines.extend([f"## `{item['name']}`", ""])
        data = extract_docstring_data(item_path)
        if data:
            signature = data.get('signature', '()')
            lines.append("```python")
            lines.append(f"{ 'class' if item_type == 'class' else 'def' } {item['name']}{signature}:")
            lines.append("```")
            lines.extend(format_docstring_obj_to_md(data['docstring_obj'], data.get('raw_docstring')))
            if item_type == 'class' and item.get("show_methods") and data.get("methods"):
                lines.append("\n### Methods")
                for method in sorted(data["methods"], key=lambda m: m['name']):
                    lines.append(f"\n#### `{method['name']}`")
                    lines.append("```python")
                    lines.append(f"def {method['name']}{method['signature']}:")
                    lines.append("```")
                    lines.extend(format_docstring_obj_to_md(method['docstring_obj'], method.get('raw_docstring')))
        else:
            lines.append("*Could not extract documentation. Please check the error messages above and correct `structure.json`.*")
        lines.append("\n---")

def process_directory(base_path: Path, config: dict):
    """Recursively processes a directory configuration from structure.json."""
    ensure_dir(base_path)

    # If there are items, this is a leaf node, generate the page.
    if "items" in config:
        title = config.get('title', config['id'].capitalize())
        lines = [f"# {title}", ""]
        if config.get("description"):
            lines.extend([config["description"], ""])
        generate_page_from_items(lines, config.get("items", []))
        write_md(base_path / "index.md", lines)
        return

    # If there are subsections, create an index and recurse.
    if "subsections" in config:
        title = config.get("title", config['id'].capitalize())
        lines = [f"# {title}", ""]
        if config.get('description'):
            lines.extend([config['description'], ""])
        lines.extend(["```{toctree}", ":maxdepth: 1", ""])
        for subsection in config.get('subsections', []):
            lines.append(f"{subsection['id']}/index")
        lines.append("```")
        write_md(base_path / "index.md", lines)

        for subsection in config.get("subsections", []):
            process_directory(base_path / subsection['id'], subsection)

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
            # Generate top-level API index (api/index.md)
            api_index_lines = [f"# {section.get('title', 'API Reference')}", "", "```{toctree}", ":maxdepth: 2", ""]
            for module in section.get("modules", []):
                api_index_lines.append(f"{module['id']}/index")
            api_index_lines.append("```")
            write_md(API_ROOT / "index.md", api_index_lines)

            # Process each top-level module
            for module in section.get("modules", []):
                print(f"  Processing module: '{module['id']}'")
                module_path = API_ROOT / module['id']
                process_directory(module_path, module)

    print('\nAll done. Review generated Markdown files under docs/api/')


if __name__ == '__main__':
    run()
