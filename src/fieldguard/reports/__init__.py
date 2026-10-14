"""Report writers: HTML (Jinja2 with inline SVG) and the redacted outputs used by the CLI."""

from fieldguard.reports.html import build_context, render_html, write_html

__all__ = ["build_context", "render_html", "write_html"]
