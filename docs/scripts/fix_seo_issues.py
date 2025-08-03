#!/usr/bin/env python3
"""
SEO Issues Quick Fix - Removes duplicate viewport tags and improves meta descriptions
"""

import os
import re
from pathlib import Path


def fix_viewport_duplicates(html_dir):
    """Remove duplicate viewport meta tags from built HTML files"""
    html_files = Path(html_dir).glob("**/*.html")
    fixed_files = 0

    for html_file in html_files:
        with open(html_file, encoding="utf-8") as f:
            content = f.read()

        # Find all viewport tags
        viewport_pattern = r'<meta name="viewport"[^>]*>'
        viewports = re.findall(viewport_pattern, content)

        if len(viewports) > 1:
            # Replace all viewport tags with a single clean one
            clean_viewport = (
                '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            )

            # Remove all existing viewport tags
            content = re.sub(viewport_pattern, "", content)

            # Add single viewport tag after charset
            content = re.sub(
                r'(<meta charset="[^"]*"\s*/>)', r"\1\n    " + clean_viewport, content
            )

            with open(html_file, "w", encoding="utf-8") as f:
                f.write(content)

            fixed_files += 1
            print(f"✅ Fixed viewport in {html_file.name}")

    print(f"🔧 Fixed viewport tags in {fixed_files} files")


def create_page_meta_descriptions():
    """Create a mapping of page-specific meta descriptions"""

    meta_descriptions = {
        "index.html": "Nabla: GPU-accelerated Python library for array computation with NumPy-like API and JAX-style transformations. Fast, composable, and Mojo-integrated.",
        "api/index.html": "Complete API reference for Nabla - Python library for GPU-accelerated array operations, transformations, and neural networks.",
        "api/array.html": "Array class documentation - the fundamental array type in Nabla for GPU-accelerated computation.",
        "api/transforms.html": "Function transformations in Nabla: vmap, grad, jit, and other automatic differentiation tools.",
        "api/binary.html": "Binary operations in Nabla: add, multiply, divide, and other element-wise operations for GPU arrays.",
        "api/unary.html": "Unary operations in Nabla: sin, cos, exp, log, and other element-wise functions for GPU arrays.",
        "api/creation.html": "Array creation functions in Nabla: zeros, ones, random, and other array initialization methods.",
        "api/reduction.html": "Reduction operations in Nabla: sum, mean, max, and other aggregation functions for GPU arrays.",
        "api/manipulation.html": "Array manipulation in Nabla: reshape, transpose, concatenate, and other shape operations.",
        "api/linalg.html": "Linear algebra operations in Nabla: matrix multiplication, decompositions, and other linear algebra functions.",
        "tutorials/index.html": "Learn Nabla with hands-on tutorials covering GPU computing, automatic differentiation, and machine learning examples.",
        "tutorials/understanding_nabla_part1.html": "Introduction to Nabla fundamentals: arrays, operations, and GPU acceleration basics.",
        "tutorials/jax_vs_nabla_mlp_cpu.html": "Performance comparison between JAX and Nabla for MLP training on CPU with code examples.",
        "tutorials/jax_vs_nabla_transformer_cpu.html": "JAX vs Nabla transformer training comparison - performance benchmarks and implementation details.",
        "tutorials/value_and_grads_cpu.html": "Learn automatic differentiation in Nabla with practical examples of gradient computation.",
        "tutorials/mlp_training_cpu.html": "Step-by-step guide to training neural networks with Nabla on CPU - complete tutorial with code.",
    }

    return meta_descriptions


def update_meta_descriptions(html_dir, meta_descriptions):
    """Update meta descriptions in HTML files"""
    updated_files = 0

    for rel_path, description in meta_descriptions.items():
        html_file = Path(html_dir) / rel_path

        if html_file.exists():
            with open(html_file, encoding="utf-8") as f:
                content = f.read()

            # Update generic description with page-specific one
            generic_desc = "Python library for GPU-accelerated array computation with NumPy-like API, JAX-style transformations (vmap, grad, jit), and Mojo integration."

            if generic_desc in content:
                content = content.replace(generic_desc, description)

                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(content)

                updated_files += 1
                print(f"✅ Updated description in {rel_path}")

    print(f"📝 Updated meta descriptions in {updated_files} files")


def validate_seo_fixes(html_dir):
    """Validate that SEO fixes are working"""
    issues = []
    successes = []

    # Check homepage
    index_file = Path(html_dir) / "index.html"
    if index_file.exists():
        with open(index_file) as f:
            content = f.read()

        # Check viewport
        viewport_count = content.count('name="viewport"')
        if viewport_count == 1:
            successes.append("✅ Single viewport tag on homepage")
        else:
            issues.append(f"❌ {viewport_count} viewport tags on homepage")

        # Check Twitter card
        if "twitter:image" in content:
            successes.append("✅ Twitter card image present")
        else:
            issues.append("❌ Twitter card image missing")

        # Check structured data
        if "application/ld+json" in content:
            successes.append("✅ Structured data present")
        else:
            issues.append("❌ Structured data missing")

        # Check canonical URL
        if 'rel="canonical"' in content:
            successes.append("✅ Canonical URL present")
        else:
            issues.append("❌ Canonical URL missing")

    print("\n🔍 SEO VALIDATION RESULTS:")
    for success in successes:
        print(success)
    for issue in issues:
        print(issue)

    return len(issues) == 0


def main():
    print("🚀 FIXING ALL SEO ISSUES!")
    print("=" * 50)

    html_dir = "/Users/tillife/Documents/CodingProjects/nabla/docs/_build/html"

    if not os.path.exists(html_dir):
        print("❌ Build directory not found. Please build documentation first.")
        return

    # Fix 1: Remove duplicate viewport tags
    print("\n1. Fixing duplicate viewport tags...")
    fix_viewport_duplicates(html_dir)

    # Fix 2: Update meta descriptions
    print("\n2. Updating page-specific meta descriptions...")
    meta_descriptions = create_page_meta_descriptions()
    update_meta_descriptions(html_dir, meta_descriptions)

    # Fix 3: Validate all fixes
    print("\n3. Validating SEO fixes...")
    all_good = validate_seo_fixes(html_dir)

    if all_good:
        print("\n🎉 ALL SEO ISSUES FIXED!")
        print("Your documentation now has:")
        print("• Single viewport tags (mobile-friendly)")
        print("• Page-specific meta descriptions")
        print("• Proper social media cards")
        print("• Complete structured data")
        print("\n🚀 Ready to deploy!")
    else:
        print("\n⚠️ Some issues remain. Check validation results above.")


if __name__ == "__main__":
    main()
