import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".html": "html", ".htm": "html", ".md": "md"}


def _meta(path, section, page=0):
    p = Path(path)
    # date_added is stored as an integer like 20261002 so Chroma can filter on ranges.
    added = int(datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d"))
    return {
        "source_file": p.name,
        "file_type": SUPPORTED[p.suffix.lower()],
        "date_added": added,
        "section": section,
        "page": page,
    }


def _records_from_markdown(text, path):
    """Split markdown-style text into one record per heading section."""
    records, title, buf = [], "Introduction", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            records.append({"text": body, "metadata": _meta(path, title)})

    for line in text.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            flush()
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    flush()
    return records


def load_pdf(path):
    reader = PdfReader(path)
    records = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append({"text": text, "metadata": _meta(path, f"page {page_no}", page_no)})
    return records


def load_docx(path):
    doc = DocxDocument(path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Turn Word headings into markdown headings so we can split on them.
        lines.append(f"## {text}" if para.style.name.startswith("Heading") else text)
    return _records_from_markdown("\n\n".join(lines), path)


def load_html(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for h in soup.find_all(["h1", "h2", "h3"]):
        h.replace_with(f"\n\n## {h.get_text(strip=True)}\n\n")
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    return _records_from_markdown(text, path)


def load_markdown(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return _records_from_markdown(text, path)


LOADERS = {"pdf": load_pdf, "docx": load_docx, "html": load_html, "md": load_markdown}


def load_file(path):
    kind = SUPPORTED[Path(path).suffix.lower()]
    return LOADERS[kind](path)


def load_directory(data_dir):
    records = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                records.extend(load_file(path))
            except Exception as e:  # one bad file shouldn't stop everything
                print(f"[skip] {path.name}: {e}")
    return records