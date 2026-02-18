# Configuration file for Sphinx documentation builder
import sys
from pathlib import Path

# Add parent directory to path so we can import interlace
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project information
project = "Interlace"
copyright = "2025, Interlace Contributors"  # noqa: A001
author = "Interlace Contributors"

# The short X.Y version
version = "0.0"
# The full version
release = "0.0.1"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory
templates_path = ["_templates"]

# List of patterns to ignore when building documentation
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and used by html_theme_path
html_theme_options = {
    "canonical_url": "https://lucaswiman.github.io/interlace",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
}

# Add any paths that contain custom static files here
html_static_path = ["_static"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Read the Docs configuration
on_rtd = True
html_context = {
    "display_github": True,
    "github_user": "lucaswiman",
    "github_repo": "interlace",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
