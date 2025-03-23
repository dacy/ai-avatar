"""
Confluence module for extracting text from Confluence pages.
"""

# Conditional imports to avoid dependency issues
try:
    from ...recylce_bin.crawler import ConfluenceCrawler
    from ...recylce_bin.simple_crawler import SimpleConfluenceCrawler
    __all__ = ['ConfluenceCrawler', 'SimpleConfluenceCrawler']
except ImportError:
    __all__ = []

# Try to import embeddings if available
try:
    from ...recylce_bin.embeddings import ConfluenceEmbedder
    __all__ = __all__ + ['ConfluenceEmbedder']
except ImportError:
    pass 