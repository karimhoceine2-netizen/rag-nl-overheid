"""Fetch source pages, keep the raw HTML, and extract clean text.

Two separate steps, so the HTTP part (slow, and load on someone else's server) runs once:
  1. fetch: data/sources.txt -> data/raw/<doc_id>.html  (+ data/raw/manifest.jsonl)
  2. parse: data/raw/*.html  -> data/docs.jsonl

Run: uv run python -m rag.ingest            (fetch missing pages, then parse)
     uv run python -m rag.ingest --refresh  (re-download everything)
"""

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from rag.models import Document

SOURCES_PATH = Path("data/sources.txt")
RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"
DOCS_PATH = Path("data/docs.jsonl")

USER_AGENT = "rag-nl-overheid/0.1 (student portfolio project; polite crawler)"
DELAY_SECONDS = 1.0  # be polite: at most one request per second
# Only text explicitly released under these terms may enter the corpus (see DATA.md).
ALLOWED_LICENSES = {"CC0 1.0 Universal"}


def doc_id_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def read_sources(path: Path = SOURCES_PATH) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["doc_id"]: r for r in records}


def save_manifest(manifest: dict[str, dict], path: Path = MANIFEST_PATH) -> None:
    lines = [json.dumps(manifest[k], ensure_ascii=False) for k in sorted(manifest)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_all(urls: list[str], refresh: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
        for url in urls:
            doc_id = doc_id_from_url(url)
            target = RAW_DIR / f"{doc_id}.html"
            if target.exists() and doc_id in manifest and not refresh:
                continue
            response = client.get(url)
            response.raise_for_status()
            target.write_text(response.text, encoding="utf-8")
            manifest[doc_id] = {
                "doc_id": doc_id,
                "url": url,
                "retrieved_at": date.today().isoformat(),
            }
            save_manifest(manifest)  # save after every page so an interrupted run keeps progress
            print(f"fetched {url}")
            time.sleep(DELAY_SECONDS)


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
    return tag.get("content") if tag else None


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # get_text(" ") puts a space before punctuation that followed a link: "toeslagen.nl ."
    return re.sub(r"\s+([.,;:!?)])", r"\1", text)


def parse_html(html: str, url: str, retrieved_at: str) -> Document:
    """Extract the article text from a rijksoverheid.nl page.

    Only the `intro` and `rich-text` blocks inside <main> are kept; video blocks (with a
    duplicate transcript), link lists and the feedback bar are page furniture, not content.
    """
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main")
    if main is None:
        raise ValueError(f"no <main> element in {url}")

    title = _meta(soup, "og:title") or (soup.h1.get_text(strip=True) if soup.h1 else "")
    title = title.removesuffix(" | Rijksoverheid.nl").strip()

    lines: list[str] = []
    for block in main.find_all("div", class_=["intro", "rich-text"], recursive=False):
        for el in block.find_all(["h2", "h3", "h4", "p", "li"]):
            # A <p> or nested <li> inside an <li> is already part of that list item's text.
            if el.find_parent("li") is not None:
                continue
            text = _clean(el.get_text(" ", strip=True))
            if not text:
                continue
            if el.name.startswith("h"):
                lines.append(f"## {text}")
            elif el.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

    modified = _meta(soup, "DCTERMS.modified")
    return Document(
        doc_id=doc_id_from_url(url),
        url=url,
        title=title,
        text="\n".join(lines),
        retrieved_at=retrieved_at,
        modified=modified[:10] if modified else None,
        license=_meta(soup, "DCTERMS.rights") or "unknown",
    )


def parse_all(urls: list[str]) -> list[Document]:
    manifest = load_manifest()
    docs: list[Document] = []
    for url in urls:
        doc_id = doc_id_from_url(url)
        record = manifest[doc_id]
        html = (RAW_DIR / f"{doc_id}.html").read_text(encoding="utf-8")
        doc = parse_html(html, record["url"], record["retrieved_at"])
        if doc.license not in ALLOWED_LICENSES:
            print(f"SKIP (license {doc.license!r}): {url}")
            continue
        if not doc.text:
            print(f"SKIP (no text extracted): {url}")
            continue
        docs.append(doc)
    return docs


def write_docs(docs: list[Document], path: Path = DOCS_PATH) -> None:
    lines = [doc.model_dump_json() for doc in docs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download all pages")
    parser.add_argument("--parse-only", action="store_true", help="skip the fetch step")
    args = parser.parse_args()

    urls = read_sources()
    if not args.parse_only:
        fetch_all(urls, refresh=args.refresh)
    docs = parse_all(urls)
    write_docs(docs)
    print(f"Wrote {len(docs)} documents to {DOCS_PATH}")


if __name__ == "__main__":
    main()
