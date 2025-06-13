#!/usr/bin/env python3
"""
Automatically sync tutorial notebooks to documentation.

This script copies Jupyter notebooks from the tutorials/ directory to docs/tutorials/
and updates the tutorials index to include them.
"""

import json
import os
import shutil
from pathlib import Path


def copy_notebooks():
    """Copy notebooks from tutorials/ to docs/tutorials/"""
    print("📚 Syncing tutorial notebooks to documentation...")

    # Paths
    tutorials_dir = Path("tutorials")
    docs_tutorials_dir = Path("docs/tutorials")

    # Ensure docs/tutorials exists
    docs_tutorials_dir.mkdir(exist_ok=True)

    # Find all notebook files
    notebooks = list(tutorials_dir.glob("*.ipynb"))

    if not notebooks:
        print("❌ No notebook files found in tutorials/")
        return []

    copied_notebooks = []

    for notebook in notebooks:
        # Copy to docs/tutorials
        dest = docs_tutorials_dir / notebook.name
        shutil.copy2(notebook, dest)
        print(f"  📄 Copied {notebook.name}")
        copied_notebooks.append(notebook.name)

    return copied_notebooks


def get_notebook_info(notebook_path):
    """Extract title and description from notebook."""
    try:
        with open(notebook_path, encoding="utf-8") as f:
            nb = json.load(f)

        # Look for title in first markdown cell
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                if source:
                    # Join source lines and look for title
                    content = "".join(source) if isinstance(source, list) else source
                    lines = content.split("\\n")

                    for line in lines:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            # Look for description in subsequent lines
                            description = ""
                            for desc_line in lines[lines.index(line) + 1 :]:
                                if desc_line.strip() and not desc_line.startswith("#"):
                                    description = desc_line.strip()
                                    break
                            return title, description

        # Fallback to filename
        return notebook_path.stem.replace(
            "_", " "
        ).title(), "Interactive tutorial notebook"

    except Exception as e:
        print(f"⚠️  Could not read {notebook_path}: {e}")
        return notebook_path.stem.replace(
            "_", " "
        ).title(), "Interactive tutorial notebook"


def update_tutorials_index(notebooks):
    """Update the tutorials index.md to include notebooks."""
    docs_tutorials_dir = Path("docs/tutorials")
    index_path = docs_tutorials_dir / "index.md"

    # Create content
    content = """# Tutorials

Interactive tutorials to learn Nabla's features and capabilities.

"""

    if notebooks:
        content += """```{toctree}
:maxdepth: 1
:caption: Interactive Notebooks

"""

        # Add each notebook
        for notebook in sorted(notebooks):
            # Get notebook info
            notebook_path = docs_tutorials_dir / notebook
            title, description = get_notebook_info(notebook_path)

            # Add to toctree (without .ipynb extension)
            notebook_name = notebook.replace(".ipynb", "")
            content += f"{notebook_name}\\n"

        content += "```\\n\\n"

        # Add descriptions
        content += "## Available Tutorials\\n\\n"
        for notebook in sorted(notebooks):
            notebook_path = docs_tutorials_dir / notebook
            title, description = get_notebook_info(notebook_path)
            notebook_name = notebook.replace(".ipynb", "")

            content += f"### [{title}]({notebook_name})\\n\\n"
            content += f"{description}\\n\\n"

    else:
        content += "No tutorials available yet. Check back soon!\\n"

    # Write the index
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Updated {index_path}")


def install_nbsphinx():
    """Check if nbsphinx is installed, install if needed."""
    try:
        import nbsphinx

        print("✅ nbsphinx is already installed")
        return True
    except ImportError:
        print("📦 Installing nbsphinx...")
        import subprocess
        import sys

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nbsphinx"])
            print("✅ nbsphinx installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install nbsphinx")
            return False


def main():
    """Main function to sync tutorials."""
    print("🚀 Syncing tutorials to documentation...\\n")

    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    # Install nbsphinx if needed
    if not install_nbsphinx():
        return

    # Copy notebooks
    notebooks = copy_notebooks()

    # Update index
    update_tutorials_index(notebooks)

    if notebooks:
        print(f"\\n✨ Successfully synced {len(notebooks)} tutorial(s)")
        print("\\n📋 Next steps:")
        print("   1. Run 'cd docs && make html' to build documentation")
        print("   2. Your tutorials will be available in the documentation")
        print("   3. Users can download and run the notebooks locally")
    else:
        print("\\n⚠️  No tutorials found to sync")


if __name__ == "__main__":
    main()
