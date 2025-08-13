import sys
import os
from bs4 import BeautifulSoup
from pathlib import Path

def fix_duplicate_viewport_tags(html_dir):
    """
    Removes duplicate viewport meta tags from HTML files.

    Args:
        html_dir (str): The directory containing HTML files to process.
    """
    print(f"🔍 Searching for HTML files in: {html_dir}")
    html_files = list(Path(html_dir).rglob("*.html"))
    
    if not html_files:
        print("⚠️ No HTML files found.")
        return

    fixed_count = 0
    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as e:
            print(f"⚠️  Could not read {html_file.name}: {e}")
            continue

        soup = BeautifulSoup(content, "html.parser")
        viewport_tags = soup.find_all("meta", attrs={"name": "viewport"})

        if len(viewport_tags) > 1:
            # Keep the first tag, remove the rest
            for tag in viewport_tags[1:]:
                tag.decompose()
            
            try:
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(str(soup))
                
                fixed_count += 1
                print(f"🔧 Fixed {html_file.name}")
            except IOError as e:
                print(f"❌ Could not write to {html_file.name}: {e}")


    if fixed_count > 0:
        print(f"✅ Processed {len(html_files)} files, fixed {fixed_count} files.")
    else:
        print("✅ No duplicate viewport tags found in any files.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_duplicate_viewport.py <html_directory>")
        sys.exit(1)
    
    html_directory = sys.argv[1]
    if not os.path.isdir(html_directory):
        print(f"❌ Error: Directory not found at '{html_directory}'")
        sys.exit(1)

    fix_duplicate_viewport_tags(html_directory)
