"""Scrapers for explicit/written and adult sites. All required for raw training (user has permission)."""
from .literotica import scrape_literotica
from .ao3 import scrape_ao3
from .asstr import scrape_asstr
from .storiesonline import scrape_storiesonline
from .pornhub import scrape_pornhub
from .aznude import scrape_aznude
from .xhamster import scrape_xhamster

__all__ = [
    "scrape_literotica", "scrape_ao3", "scrape_asstr", "scrape_storiesonline",
    "scrape_pornhub", "scrape_aznude", "scrape_xhamster",
]
