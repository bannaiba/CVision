"""
modules/__init__.py
===================
Package marker for the CVision modules package.

Exposes convenience top-level imports so callers can write:
    from modules import ingestion, parser, embedding
instead of:
    from modules.ingestion import ...
"""

from modules import ingestion, parser, embedding

__all__ = ["ingestion", "parser", "embedding"]
