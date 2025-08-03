#!/usr/bin/env python3
"""
Generate a comprehensive sitemap.xml for Nabla documentation
Improved version that includes all HTML files, not just markdown sources.
"""

import shutil
from datetime import datetime
from pathlib import Path


def generate_sitemap():
    """Generate sitemap from both markdown sources and built HTML files"""
    base_url = "https://nablaml.com"
    today = datetime.now().strftime("%Y-%m-%d")

    # Priority mapping for different content types
    priorities = {
        "index.html": 1.0,
        "tutorials/index.html": 0.9,
        "api/index.html": 0.9,
        # Tutorials are high priority
        "understanding_nabla_part1.html": 0.9,
        "mlp_training_cpu.html": 0.9,
        "mlp_training_gpu.html": 0.9,
        "value_and_grads_cpu.html": 0.9,
        "value_and_grads_gpu.html": 0.9,
        "jax_vs_nabla_mlp_cpu.html": 0.9,
        "jax_vs_nabla_transformer_cpu.html": 0.9,
        # Core API modules
        "api/array.html": 0.8,
        "api/transforms.html": 0.8,
        "api/nn.html": 0.8,
    }

    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent  # scripts folder is inside docs folder
    build_dir = docs_dir / "_build" / "html"

    sitemap_entries = []

    # Method 1: If build exists, scan actual HTML files (most accurate)
    if build_dir.exists():
        print("📂 Scanning built HTML files...")
        html_files = list(build_dir.rglob("*.html"))

        for html_file in html_files:
            rel_path = html_file.relative_to(build_dir)

            # Skip unwanted files
            if any(
                skip in str(rel_path)
                for skip in [
                    "_static/",
                    "_sources/",
                    "genindex.html",
                    "search.html",
                    "py-modindex.html",
                    "searchindex.html",
                ]
            ):
                continue

            html_path = str(rel_path)

            # Generate URL
            if html_path == "index.html":
                url = f"{base_url}/"
            else:
                url = f"{base_url}/{html_path}"

            # Determine priority
            priority = priorities.get(html_path)
            if priority is None:
                # Default priorities by path pattern
                if html_path.startswith("tutorials/"):
                    priority = 0.8
                elif html_path.startswith("api/"):
                    priority = 0.6
                else:
                    priority = 0.5

            # Determine change frequency
            if html_path.startswith("api/") or html_path.startswith("tutorials/"):
                changefreq = "monthly"
            elif html_path == "index.html":
                changefreq = "weekly"
            else:
                changefreq = "monthly"

            sitemap_entries.append(
                {
                    "url": url,
                    "lastmod": today,
                    "changefreq": changefreq,
                    "priority": priority,
                }
            )

    # Method 2: Fallback to scanning markdown sources if no build
    else:
        print("📝 No build found, scanning markdown sources...")
        md_files = list(docs_dir.rglob("*.md"))

        for md_file in md_files:
            rel_path = md_file.relative_to(docs_dir)
            if any(
                skip in str(rel_path) for skip in ["_build", "__pycache__", "README"]
            ):
                continue

            html_path = str(rel_path).replace(".md", ".html")

            if html_path == "index.html":
                url = f"{base_url}/"
                priority = 1.0
            else:
                url = f"{base_url}/{html_path}"
                priority = priorities.get(html_path, 0.5)
                if priority == 0.5:  # Use pattern matching
                    if "tutorials/" in html_path:
                        priority = 0.8
                    elif "api/" in html_path:
                        priority = 0.6

            changefreq = (
                "monthly"
                if ("api/" in html_path or "tutorials/" in html_path)
                else "weekly"
            )

            sitemap_entries.append(
                {
                    "url": url,
                    "lastmod": today,
                    "changefreq": changefreq,
                    "priority": priority,
                }
            )

    # Remove duplicates and sort by priority
    seen_urls = set()
    unique_entries = []
    for entry in sitemap_entries:
        if entry["url"] not in seen_urls:
            seen_urls.add(entry["url"])
            unique_entries.append(entry)

    unique_entries.sort(key=lambda x: x["priority"], reverse=True)

    # Generate XML
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
"""

    for entry in unique_entries:
        xml_content += f"""    <url>
        <loc>{entry["url"]}</loc>
        <lastmod>{entry["lastmod"]}</lastmod>
        <changefreq>{entry["changefreq"]}</changefreq>
        <priority>{entry["priority"]:.1f}</priority>
    </url>
"""

    xml_content += "</urlset>\n"

    # Write sitemap to _static directory
    static_sitemap = docs_dir / "_static" / "sitemap.xml"
    static_sitemap.parent.mkdir(exist_ok=True)
    with open(static_sitemap, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # Also copy to build output if it exists
    if build_dir.exists():
        build_sitemap = build_dir / "sitemap.xml"
        build_robots = build_dir / "robots.txt"
        shutil.copy2(static_sitemap, build_sitemap)

        # Ensure robots.txt exists and is correct
        robots_path = docs_dir / "_static" / "robots.txt"
        if robots_path.exists():
            shutil.copy2(robots_path, build_robots)

        print(f"✅ Generated sitemap with {len(unique_entries)} URLs")
        print(f"📍 Saved to: {static_sitemap}")
        print(f"📍 Copied to: {build_sitemap}")
    else:
        print(f"✅ Generated sitemap with {len(unique_entries)} URLs")
        print(f"📍 Saved to: {static_sitemap}")

    # Print summary by priority
    priority_counts = {}
    for entry in unique_entries:
        p = entry["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1

    print("\n📊 Priority distribution:")
    for priority in sorted(priority_counts.keys(), reverse=True):
        count = priority_counts[priority]
        print(f"   Priority {priority}: {count} pages")


if __name__ == "__main__":
    generate_sitemap()
