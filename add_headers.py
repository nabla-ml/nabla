#!/usr/bin/env python3
"""
Script to add license headers to all Python files in nabla/ and tests/ directories.
"""

import os
import sys
from pathlib import Path

# The header to add to each file
HEADER = '''# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

'''

def has_header(content: str) -> bool:
    """Check if the file already has the header."""
    return "# ===----------------------------------------------------------------------=== #" in content

def add_header_to_file(file_path: Path) -> bool:
    """Add header to a single file. Returns True if file was modified."""
    try:
        # Read the current content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if header already exists
        if has_header(content):
            print(f"Skipping {file_path} (header already exists)")
            return False
        
        # Handle shebang lines - keep them at the top
        lines = content.split('\n')
        insert_index = 0
        
        # If first line is a shebang, keep it at the top
        if lines and lines[0].startswith('#!'):
            insert_index = 1
            new_content = lines[0] + '\n' + HEADER + '\n'.join(lines[1:])
        else:
            new_content = HEADER + content
        
        # Write the new content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Added header to {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory: Path) -> tuple[int, int]:
    """Process all Python files in a directory recursively."""
    modified_count = 0
    total_count = 0
    
    # Find all Python files
    for file_path in directory.rglob("*.py"):
        # Skip __pycache__ directories
        if "__pycache__" in str(file_path):
            continue
            
        total_count += 1
        if add_header_to_file(file_path):
            modified_count += 1
    
    return modified_count, total_count

def main():
    """Main function to process both nabla/ and tests/ directories."""
    script_dir = Path(__file__).parent
    
    # Directories to process
    directories = [
        script_dir / "nabla",
        script_dir / "tests"
    ]
    
    total_modified = 0
    total_files = 0
    
    print("Adding license headers to Python files...")
    print("=" * 50)
    
    for directory in directories:
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist")
            continue
        
        print(f"\nProcessing {directory}/")
        modified, total = process_directory(directory)
        total_modified += modified
        total_files += total
        
        print(f"Modified {modified} out of {total} files in {directory.name}/")
    
    print("=" * 50)
    print(f"Summary: Modified {total_modified} out of {total_files} total Python files")
    
    if total_modified > 0:
        print("\nRecommendation: Review the changes with 'git diff' before committing")

if __name__ == "__main__":
    main()
