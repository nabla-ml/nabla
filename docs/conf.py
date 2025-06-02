# Configuration file for the Sphinx documentation builder.

import os
import sys
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
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosectionlabel',  # For auto-generating section labels
    'sphinx_design',                # For enhanced design components
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

# AutoSummary settings
autosummary_generate = True
autosummary_imported_members = True
autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Turn on autosummary with recursive generation
autosummary_generate_overwrite = True

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': 'gallery_examples',  # Path to clean gallery examples directory
    'gallery_dirs': 'auto_examples',  # Output directory for gallery
    'filename_pattern': r'.*\.py$',  # Process all Python files in gallery_examples
    'ignore_pattern': r'__init__\.py',  # Only ignore __init__.py files
    'download_all_examples': True,
    'show_memory': False,  # Disable memory profiling to avoid warnings
    'min_reported_time': 1,
    'plot_gallery': 'True',  # Enable plotting but quotes ensure it's not executed
    'line_numbers': True,
    'remove_config_comments': True,
    'default_thumb_file': '_static/nabla-logo.svg',
    'expected_failing_examples': ['gallery_examples/plot_basic_operations.py'],  # List examples expected to fail
    'matplotlib_animations': True,  # Enable matplotlib animations
    'doc_module': ('nabla',),  # Add modules to be documented with autodoc
    'backreferences_dir': 'gen_modules/backreferences',  # Directory for storing backreferences
    'reference_url': {'nabla': None},
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '404.md', '**/gen_modules/**']

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
