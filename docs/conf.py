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
html_favicon = "_static/nabla-logo.svg"

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
    "use_issues_button": True,
    "use_edit_page_button": True,
    # Path to docs in the repository
    "path_to_docs": "docs",
    # Navigation and sidebar
    "show_navbar_depth": 2,
    "use_sidenotes": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    # Logo and branding - Simple text instead of image
    "logo": {
        "text": "NABLA",
    },
    # Extra footer content
    "extra_footer": """
    <div>
      <a href="https://github.com/nabla-ml/nabla">Nabla</a> - Built with
      <a href="https://www.sphinx-doc.org/">Sphinx</a> using a
      <a href="https://github.com/executablebooks/sphinx-book-theme">Sphinx Book Theme</a>
    </div>
    """,
}
