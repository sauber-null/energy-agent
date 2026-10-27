"""PDF to Markdown, one string per page.

pymupdf4llm (bundled with PyMuPDF 2.x) renders tables as Markdown tables, which keeps the
"Nr. | Aktion | Frist | Hinweis" columns of the GPKE process tables together.
"""

from pathlib import Path

from pymupdf import pymupdf4llm


def pdf_to_markdown_pages(path: Path) -> list[str]:
    return [page["text"] for page in pymupdf4llm.to_markdown(str(path), page_chunks=True)]
