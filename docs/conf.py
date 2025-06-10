# Configuration file for the Sphinx documentation builder.


# -- Project information -----------------------------------------------------
project = "Nabla"
project_copyright = "2025, Nabla Team"
author = "Nabla Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
]

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "404.md",
    "**/gen_modules/**",
    "gallery_examples/**",
    "auto_examples/**",
    "api/generated/**",
    "sg_execution_times.rst",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Nabla"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_favicon = "_static/nabla-logo.png"

# GitHub Pages optimization
html_baseurl = "/"
html_use_index = True
html_domain_indices = True
html_use_modindex = True

html_theme_options = {
    # Repository integration
    "repository_url": "https://github.com/nabla-ml/nabla",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "use_fullscreen_button": False,
    "use_download_button": False,
    # Path to docs in the repository
    "path_to_docs": "docs",
    # Navigation and sidebar
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    # Secondary sidebar (TOC) - static, no collapse
    "secondary_sidebar_items": ["page-toc"],
    "show_prev_next": True,
    # Logo and branding - Use image instead of text
    "logo": {
        "image_dark": "_static/nabla-logo.png",
        "image_light": "_static/nabla-logo.png",
        "text": "NABLA",  # Fallback text
    },
    # Extra footer content
    "extra_footer": """
    <div style="text-align: center; font-size: 0.875rem; color: #888888;">
      Nabla 2025
    </div>
    """,
}
