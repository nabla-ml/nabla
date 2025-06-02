# Configuration file for the Sphinx documentation builder.

import os
import sys
from unittest.mock import MagicMock

# Mock imports for modules that aren't available during CI/CD
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock external dependencies that may not be available during docs build
MOCK_MODULES = [
    'max',
    'max.dtype',
    'max.graph',
    'max.tensor',
    'mojo',
    'numpy',
    'jax',
    'torch',
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# Add the project to Python path only if we're building locally
if os.path.exists('../nabla'):
    sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
project = 'Nabla'
copyright = '2025, Nabla Team'
author = 'Nabla Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_design',
]

# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
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
autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Autodoc settings for better handling of missing modules
autodoc_mock_imports = MOCK_MODULES
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

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

templates_path = ['_templates']
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    '404.md', 
    '**/gen_modules/**', 
    'gallery_examples/**', 
    'auto_examples/**',
    'sg_execution_times.rst'
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'  # Using Sphinx Book Theme like JAX docs
html_title = "Nabla"
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/nabla-logo.svg'

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

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}
