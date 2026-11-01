# documents.py: loads supported file formats and extracts clean text with section metadata

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePath
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError


# Keep extracted text associated with the source heading and optional PDF page.
@dataclass(frozen=True)
class DocumentSection:
    heading: str | None
    text: str
    page_number: int | None = None


# Normalize Markdown headings and preserve section boundaries for chunking.
def _markdown_sections(text: str) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            if "\n".join(lines).strip():
                sections.append(DocumentSection(heading, "\n".join(lines).strip()))
            heading = match.group(1).strip()
            lines = []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append(DocumentSection(heading, "\n".join(lines).strip()))
    return sections or [DocumentSection(None, text.strip())]


# Extract text from HTML while retaining headings as section labels.
def _html_sections(text: str) -> list[DocumentSection]:
    soup = BeautifulSoup(text, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    sections: list[DocumentSection] = []
    heading: str | None = None
    blocks: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        value = " ".join(element.get_text(" ", strip=True).split())
        if not value:
            continue
        if element.name and element.name.startswith("h"):
            if blocks:
                sections.append(DocumentSection(heading, "\n".join(blocks)))
            heading = value
            blocks = []
        else:
            blocks.append(value)
    if blocks:
        sections.append(DocumentSection(heading, "\n".join(blocks)))
    if not sections:
        plain = " ".join(soup.get_text(" ", strip=True).split())
        sections.append(DocumentSection(None, plain))
    return sections


# Read PDF pages independently so retrieved chunks can cite their page numbers.
def _pdf_sections(content: bytes) -> list[DocumentSection]:
    import io

    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
    except PdfReadError as error:
        raise ValueError("The uploaded PDF is malformed or encrypted.") from error
    sections = [
        DocumentSection(None, page.extract_text() or "", page_number=index + 1)
        for index, page in enumerate(reader.pages)
    ]
    return [section for section in sections if section.text.strip()]


# Convert a supported upload into normalized sections without executing document content.
def load_document(filename: str, content: bytes) -> list[DocumentSection]:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}:
        raise ValueError(f"Unsupported file type {suffix or '(no extension)'}; use Markdown, text, HTML, or PDF.")
    if suffix == ".pdf":
        return _pdf_sections(content)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{filename} is not valid UTF-8 text.") from error
    if suffix in {".md", ".markdown"}:
        sections = _markdown_sections(text)
    elif suffix in {".html", ".htm"}:
        sections = _html_sections(text)
    else:
        sections = [DocumentSection(None, text.strip())]
    return [
        DocumentSection(section.heading, section.text, section.page_number)
        for section in sections
        if section.text.strip()
    ]
