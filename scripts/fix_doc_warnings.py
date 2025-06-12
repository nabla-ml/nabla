#!/usr/bin/env python3
"""
Automatically fix documentation warnings in the Nabla project.

This script addresses:
1. Remaining value_and_grad references in __all__ lists
2. Docstring indentation issues
3. Missing toctree entries
4. Cross-reference warnings
"""

import os
import re
from pathlib import Path


def fix_value_and_grad_references():
    """Remove remaining value_and_grad references from nn module."""
    print("🔧 Fixing value_and_grad references...")
    
    nn_utils_init = Path("nabla/nn/utils/__init__.py")
    if nn_utils_init.exists():
        content = nn_utils_init.read_text()
        
        # Remove from __all__ list
        if '"value_and_grad",' in content:
            print("  🗑️  Removing value_and_grad from nn.utils.__all__")
            content = re.sub(r'\s*"value_and_grad",\n', '', content)
            nn_utils_init.write_text(content)


def fix_docstring_indentation():
    """Fix docstring indentation issues in transforms module."""
    print("🔧 Fixing docstring indentation...")
    
    files_to_fix = [
        "nabla/transforms/grad.py",
        "nabla/transforms/vjp.py"
    ]
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            continue
            
        print(f"  📝 Checking {file_path}")
        content = path.read_text()
        lines = content.split('\n')
        
        # Fix common docstring indentation issues
        fixed_lines = []
        in_docstring = False
        docstring_indent = 0
        
        for i, line in enumerate(lines):
            # Detect start of docstring
            if '"""' in line and not in_docstring:
                in_docstring = True
                docstring_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
                continue
            
            # Detect end of docstring
            if in_docstring and '"""' in line:
                in_docstring = False
                fixed_lines.append(line)
                continue
            
            # Fix indentation inside docstring
            if in_docstring:
                # Ensure consistent indentation
                if line.strip():  # Non-empty line
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent < docstring_indent + 4:
                        # Fix under-indented lines
                        line = ' ' * (docstring_indent + 4) + line.lstrip()
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Write back if changes were made
        new_content = '\n'.join(fixed_lines)
        if new_content != content:
            print(f"  ✅ Fixed indentation in {file_path}")
            path.write_text(new_content)


def fix_toctree_warnings():
    """Add missing documents to toctree."""
    print("🔧 Fixing toctree warnings...")
    
    index_path = Path("docs/index.md")
    if index_path.exists():
        content = index_path.read_text()
        
        # Check if SEO_COMPLETE is already in toctree
        if "SEO_COMPLETE" not in content:
            print("  📄 Adding SEO_COMPLETE to main toctree")
            # Add it to the toctree
            content = content.replace(
                "SETUP_SUMMARY\nEXCLUSION_GUIDE\n```",
                "SETUP_SUMMARY\nEXCLUSION_GUIDE\nSEO_COMPLETE\n```"
            )
            index_path.write_text(content)


def fix_cross_references():
    """Fix cross-reference warnings in SETUP_SUMMARY.md."""
    print("🔧 Fixing cross-reference warnings...")
    
    setup_summary = Path("docs/SETUP_SUMMARY.md")
    if setup_summary.exists():
        content = setup_summary.read_text()
        
        # Replace invalid cross-references with simple text
        replacements = {
            "[`nabla/utils/docs.py`](nabla/utils/docs.py)": "`nabla/utils/docs.py`",
            "[`docs/conf.py`](docs/conf.py)": "`docs/conf.py`",
            "[`scripts/generate_structured_docs.py`](scripts/generate_structured_docs.py)": "`scripts/generate_structured_docs.py`"
        }
        
        for old, new in replacements.items():
            if old in content:
                print(f"  🔗 Fixing cross-reference: {old}")
                content = content.replace(old, new)
        
        setup_summary.write_text(content)


def validate_fixes():
    """Validate that fixes worked by checking imports."""
    print("🧪 Validating fixes...")
    
    try:
        # Test basic imports
        import sys
        sys.path.insert(0, str(Path.cwd()))
        
        import nabla
        print("  ✅ nabla imports successfully")
        
        import nabla.nn
        print("  ✅ nabla.nn imports successfully")
        
        from nabla.transforms import value_and_grad
        print("  ✅ value_and_grad available from transforms")
        
        # Make sure value_and_grad is NOT in nn
        try:
            from nabla.nn import value_and_grad
            print("  ❌ ERROR: value_and_grad still in nabla.nn")
        except ImportError:
            print("  ✅ value_and_grad correctly excluded from nabla.nn")
            
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False
    
    return True


def main():
    """Run all fixes."""
    print("🚀 Automatically fixing documentation warnings...\n")
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    # Apply fixes
    fix_value_and_grad_references()
    fix_docstring_indentation()
    fix_toctree_warnings()
    fix_cross_references()
    
    # Validate
    if validate_fixes():
        print("\n✨ All fixes applied successfully!")
        print("\n🏗️  Now run 'cd docs && make clean && make html' to test the build")
    else:
        print("\n❌ Some fixes may not have worked correctly")


if __name__ == "__main__":
    main()
