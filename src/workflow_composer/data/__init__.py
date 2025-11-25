"""
Data Management Module
======================

Tools for downloading and managing reference data.
"""

from .downloader import (
    DataDownloader,
    DownloadedFile,
    Reference,
    REFERENCE_SOURCES,
    INDEX_SOURCES,
    SAMPLE_DATASETS
)

__all__ = [
    "DataDownloader",
    "DownloadedFile", 
    "Reference",
    "REFERENCE_SOURCES",
    "INDEX_SOURCES",
    "SAMPLE_DATASETS"
]
