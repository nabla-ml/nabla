#!/usr/bin/env python3
"""
Nabla Documentation Update Script

This script regenerates the API documentation and builds the full documentation.
Use this script whenever you want to update the static API documentation files
that are committed to the repository.

Usage:
    python scripts/update_docs.py

The script will:
1. Generate static API documentation from the current Nabla source code
2. Build the complete documentation
3. Show the local documentation URL for preview

Note: The generated API documentation files should be committed to the repository
so that GitHub Actions can build the docs without installing Nabla.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"

    print("🔄 Updating Nabla Documentation")
    print("=" * 50)

    # Step 1: Generate API documentation
    print("\n📚 Step 1: Generating static API documentation...")
    try:
        subprocess.run(
            [sys.executable, str(script_dir / "generate_api_docs.py")],
            cwd=project_root,
            check=True,
        )
        print("✅ API documentation generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate API documentation: {e}")
        return 1

    # Step 2: Build documentation
    print("\n🏗️  Step 2: Building complete documentation...")
    try:
        subprocess.run(
            [
                "sphinx-build",
                "-b",
                "html",
                str(docs_dir),
                str(docs_dir / "_build" / "html"),
            ],
            cwd=docs_dir,
            check=True,
        )
        print("✅ Documentation built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build documentation: {e}")
        return 1

    # Step 3: Show results
    output_dir = docs_dir / "_build" / "html"
    index_file = output_dir / "index.html"

    print("\n🎉 Documentation update complete!")
    print("=" * 50)
    print(f"📁 Output directory: {output_dir}")
    print(f"🌐 Open in browser: file://{index_file.absolute()}")
    print("\n💡 Next steps:")
    print("1. Review the generated documentation")
    print("2. Commit the updated API documentation files in docs/api/")
    print("3. Push to trigger GitHub Pages deployment")

    return 0


if __name__ == "__main__":
    sys.exit(main())
