# Configuration file for Sphinx documentation builder
import os
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path so we can import frontrun
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project information
project = "Frontrun"
copyright = "2026, Lucas Wiman"  # noqa: A001
author = "Frontrun Contributors"

from frontrun import __version__ as release

version = ".".join(release.split(".")[:2])

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

# Add any paths that contain templates here, relative to this directory
templates_path = ["_templates"]

# List of patterns to ignore when building documentation
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use — furo supports automatic light/dark mode via prefers-color-scheme
html_theme = "furo"

# Add any paths that contain custom static files here
html_static_path = ["_static"]

# Include custom CSS for Atkinson Hyperlegible fonts
html_css_files = ["custom.css"]

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

# --- SEO -------------------------------------------------------------------
# Canonical base URL: furo emits <link rel="canonical"> from this, and
# sphinx-sitemap uses it to build sitemap.xml (referenced from
# https://lucaswiman.github.io/robots.txt, which governs this project site).
html_baseurl = "https://lucaswiman.github.io/frontrun/"
sitemap_url_scheme = "{link}"  # no lang/version prefix; docs are unversioned
sitemap_excludes = ["search.html", "genindex.html", "py-modindex.html"]

# Open Graph / Twitter card tags for link previews.
ogp_site_url = html_baseurl
ogp_site_name = "Frontrun"
ogp_description_length = 200
ogp_type = "website"
ogp_custom_meta_tags = ['<meta name="twitter:card" content="summary">']
ogp_social_cards = {"enable": False}  # avoid an implicit matplotlib dependency

# Source links
html_theme_options = {
    "source_repository": "https://github.com/lucaswiman/frontrun",
    "source_branch": "main",
    "source_directory": "docs/",
}


def _run_example(example_file: str, report_path: str, extra_args: list[str] | None = None) -> None:
    """Run *example_file* via ``frontrun python`` and write a report to *report_path*.

    Uses the ``frontrun`` binary that lives alongside ``sys.executable`` so the
    LD_PRELOAD environment is set up correctly.  Fails silently so the rest of
    the documentation still builds when the Rust extension is unavailable.
    """
    import subprocess

    frontrun_bin = os.path.join(os.path.dirname(sys.executable), "frontrun")
    cmd = [frontrun_bin, "python", example_file, report_path] + (extra_args or [])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        print(f"Generated DPOR example report: {report_path}")
    except Exception as exc:
        print(f"Warning: skipping {os.path.basename(example_file)} ({exc})")


def _generate_dpor_example_reports(app: Any) -> None:
    """Generate all interactive HTML reports for the documentation examples."""
    static_dir = os.path.join(os.path.dirname(__file__), "_static")
    examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))

    _run_example(
        os.path.join(examples_dir, "dpor_bank_transfer.py"),
        os.path.join(static_dir, "dpor_bank_transfer.html"),
    )
    _run_example(
        os.path.join(examples_dir, "dpor_bank_transfer_locked.py"),
        os.path.join(static_dir, "dpor_bank_transfer_locked.html"),
    )
    _run_example(
        os.path.join(examples_dir, "dpor_sqlite_counter.py"),
        os.path.join(static_dir, "dpor_sqlite_counter.html"),
    )
    _run_example(
        os.path.join(examples_dir, "dpor_sqlite_counter.py"),
        os.path.join(static_dir, "dpor_sqlite_counter_fixed.html"),
        extra_args=["fixed"],
    )
    _run_example(
        os.path.join(examples_dir, "dpor_dining_philosophers.py"),
        os.path.join(static_dir, "dpor_dining_philosophers.html"),
    )


def setup(app: Any) -> None:
    app.connect("builder-inited", _generate_dpor_example_reports)
