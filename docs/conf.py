# =============================================================================
# SPHINX CONFIGURATION FILE
# =============================================================================
#
# This file is the central configuration for the Sphinx documentation builder.
# It tells Sphinx how to build the documentation website for the Nabla project.
#
# =============================================================================

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# =============================================================================
# SECTION 1: MOCKING EXTERNAL DEPENDENCIES
# =============================================================================
#
# This section prevents the documentation build from crashing if it encounters
# code that imports heavy libraries (like numpy, jax, etc.). By "mocking"
# these libraries, we can build the docs in a lightweight environment without
# needing to install the full Nabla library and its dependencies.
#
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    "max", "max.dtype", "max.graph", "max.tensor",
    "mojo", "numpy", "jax", "torch",
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# =============================================================================
# SECTION 2: PROJECT INFORMATION
# =============================================================================
project = "Nabla"
project_copyright = "2025, Nabla ML"
author = "Nabla ML"
release = "0.1.0"

# =============================================================================
# SECTION 3: GENERAL SPHINX CONFIGURATION
# =============================================================================
#
# Core settings for Sphinx, including extensions and source file parsers.
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# List of Sphinx extensions (plugins) to use.
extensions = [
    "sphinx.ext.autodoc",       # Core library for autodoc generation
    "sphinx.ext.autosummary",   # Create summary tables
    "sphinx.ext.napoleon",      # Support for Google-style docstrings
    "sphinx.ext.viewcode",      # Add links to highlighted source code
    "sphinx.ext.intersphinx",   # Link to other projects' documentation
    "sphinx.ext.mathjax",       # Render math via JavaScript
    "myst_parser",              # Parse Markdown files
    "sphinx_design",            # For UI components like cards and grids
    "nbsphinx",                 # For rendering Jupyter Notebooks
    "IPython.sphinxext.ipython_console_highlighting", # Better highlighting
]

# Paths to directories containing custom templates or static files.
templates_path = ["_templates"]
html_static_path = ["_static"]

# Patterns to exclude from the build process.
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "**/gen_modules/**", "README.md", "scripts/README.md"
]

# =============================================================================
# SECTION 4: HTML OUTPUT AND THEME CONFIGURATION
# =============================================================================
#
# Settings that control the look and feel of the final HTML website.
#
html_theme = "sphinx_book_theme"
html_title = "Nabla - High-Performance ML Computing"
html_short_title = "Nabla"
html_favicon = "_static/nabla-logo.svg"

# Files to be copied directly to the output directory (e.g., for SEO).
html_extra_path = ["_static/robots.txt", "_static/sitemap.xml"]

# Custom CSS and JavaScript files.
html_css_files = ["custom_minimal.css", "seo-advanced.css"]
html_js_files = ["performance.js", "custom.js"]

# Theme-specific options for the Sphinx Book Theme.
html_theme_options = {
    "repository_url": "https://github.com/nabla-ml/nabla",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/nabla-logo.png",
        "image_dark": "_static/nabla-logo.png",
    },
    
}

# Global context variables available to all templates.
html_context = {
    "default_mode": "dark",
}

# =============================================================================
# SECTION 5: EXTENSION-SPECIFIC CONFIGURATION
# =============================================================================
#
# Configuration for the various Sphinx extensions enabled in Section 3.
#
# -- MyST Parser (Markdown) --
myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "html_admonition",
    "html_image", "replacements", "smartquotes", "substitution", "tasklist",
]

# -- Napoleon (Docstrings) --
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Autodoc (Code Documentation) --
autodoc_mock_imports = MOCK_MODULES
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Autosummary --
autosummary_generate = not os.environ.get('CI')

# -- Intersphinx (Cross-project Linking) --
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- nbsphinx (Jupyter Notebooks) --
nbsphinx_execute = "never"  # Don't execute notebooks during the build
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"
nbsphinx_timeout = 60
nbsphinx_assume_equations = True

# Suppress common warnings
suppress_warnings = [
    'nbsphinx',
    'misc.highlighting_failure',
    'app.add_directive',
    'ref.citation',
    'ref.footnote',
    'toc.excluded',
    'toc.not_readable',
]

# Nitpicky mode - don't fail on missing references
nitpicky = False
nitpick_ignore = []

# =============================================================================
# SECTION 6: SEO CONFIGURATION
# =============================================================================
#
# This setting is used by the sitemap generator to create absolute URLs.
# It reads from an environment variable, which is set in the GitHub workflow.
#
html_baseurl = os.environ.get("DOCS_BASE_URL", "https://nablaml.com/")