# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock


# Mock imports for modules that aren't available during CI/CD
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# Mock external dependencies that may not be available during docs build
MOCK_MODULES = [
    "max",
    "max.dtype",
    "max.graph",
    "max.tensor",
    "mojo",
    "numpy",
    "jax",
    "torch",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# Add the project to Python path only if we're building locally

if Path("../nabla").exists():
    sys.path.insert(0, str(Path("../").resolve()))

# -- Project information -----------------------------------------------------
project = "Nabla"
project_copyright = "2025, Nabla Team"
author = "Nabla Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "nbsphinx",  # Jupyter notebook support
    "IPython.sphinxext.ipython_console_highlighting",  # Better code highlighting
]

# MathJax configuration
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# AutoSummary settings - disable generation during CI
autosummary_generate = False  # We'll pre-generate these
autosummary_imported_members = False
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Autodoc settings for better handling of missing modules
autodoc_mock_imports = MOCK_MODULES
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

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

# -- Notebook configuration ---------------------------------------------------
# nbsphinx configuration for Jupyter notebooks
nbsphinx_execute = "never"  # Don't execute notebooks during build (pre-executed)
nbsphinx_allow_errors = True  # Allow notebooks with errors to be included
nbsphinx_timeout = 120  # Execution timeout in seconds

# Basic nbsphinx settings
nbsphinx_codecell_lexer = "ipython3"

# Prolog/epilog for notebooks with Google Colab links
nbsphinx_prolog = """
.. raw:: html

    <div class="notebook-buttons">
        <a href="https://colab.research.google.com/github/nabla-ml/nabla/blob/main/tutorials/{{ env.docname.split('/')[-1] }}.ipynb" class="colab-button" target="_blank">Open in Google Colab</a>
        <a href="{{ env.docname }}.ipynb" class="notebook-download" download>Download Notebook</a>
    </div>
"""

# Epilog to add download link
nbsphinx_epilog = """
----

.. raw:: html

    <div class="notebook-buttons">
        <a href="https://colab.research.google.com/github/nabla-ml/nabla/blob/main/tutorials/{{ env.docname.split('/')[-1] }}.ipynb" class="colab-button" target="_blank">Open in Google Colab</a>
        <a href="{{ env.docname }}.ipynb" class="notebook-download" download>Download Notebook</a>
    </div>
"""

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/gen_modules/**",
    "gallery_examples/**",
    "auto_examples/**",
    "sg_execution_times.rst",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Nabla - High-Performance ML Computing"
html_short_title = "Nabla"
html_static_path = ["_static"]
html_extra_path = ["_static/robots.txt", "_static/sitemap.xml"]
html_css_files = ["custom_minimal.css", "seo-advanced.css"]
html_js_files = ["seo.js", "performance.js"]
html_favicon = "_static/nabla-logo.svg"

# SEO Configuration using standard Sphinx options
# Use environment variable if available (for CI/CD), otherwise use default
html_baseurl = os.environ.get("DOCS_BASE_URL", "https://nablaml.com/")

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
    "show_toc_level": 2,
    "navigation_with_keys": True,
    # Logo and branding - Use image logo
    "logo": {
        "image_light": "_static/nabla-logo.png",
        "image_dark": "_static/nabla-logo.png",
    },
    # Force dark mode only
    "use_theme_switcher": False,
    "theme_switcher_json_url": "",  # Disable theme switcher completely
}


# Add post-build hook to fix viewport tags automatically
def setup(app):
    app.connect("build-finished", fix_viewport_tags)


def fix_viewport_tags(app, exception):
    if exception is None and app.builder.name == "html":
        import os
        import subprocess

        build_dir = os.path.join(app.outdir)
        try:
            subprocess.run(
                [
                    "python",
                    os.path.join(app.srcdir, "scripts/fix_duplicate_viewport.py"),
                    build_dir,
                ],
                check=True,
                cwd=app.srcdir,
            )
            print("✅ Automatically fixed duplicate viewport tags")
        except Exception as e:
            print(f"⚠️  Could not auto-fix viewport tags: {e}")


html_context = {
    "default_mode": "dark",
    "theme_mode": "dark",  # Force dark theme
}

# Remove the html_logo since we use it in theme options
# html_logo = "_static/nabla-logo.png"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}
