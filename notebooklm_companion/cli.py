"""NotebookLM Companion generator.

CLI entry point: `notebooklm-companion`
"""

# --- BEGIN embedded implementation (from rag_chunk_enricher_revised.py) ---
#!/usr/bin/env python3
"""
RAG Chunk Enricher — Context-Enriched Chunking for Any RAG System
=================================================================

Takes a PDF (or plain text / Markdown) document and produces context-enriched
chunks that improve retrieval quality when uploaded to *any* RAG system,
including those that do not implement hierarchical (parent-child) retrieval.

In addition to JSON/JSONL chunk exports, this script can generate a
NotebookLM-friendly *companion* Markdown document ("restructured" format).
That companion is designed for platforms where you cannot control internal
chunking/retrieval: it injects stable anchors, page spans, hierarchy cues,
and (optionally) indices for definitions and assumptions.

Output formats:
  - JSON   (one file, array of enriched chunk objects)
  - JSONL  (one chunk per line — easy to ingest into vector DBs)
  - Markdown (one .md file with clearly delimited chunks)
  - Plain text (one .txt per chunk, for systems that ingest files)
  - Restructured Markdown (one .md companion document for NotebookLM-style RAG)

Usage:
  python rag_chunk_enricher_revised.py input.pdf --output-dir ./chunks --format json
  python rag_chunk_enricher_revised.py input.pdf --format jsonl --chunk-size 1500
  python rag_chunk_enricher_revised.py input.pdf --format restructured --companion-words 320

Author: Generated for Tony Myers, Birmingham Newman University
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

log = logging.getLogger(__name__)

# ── Optional dependencies ──
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None  # type: ignore[assignment]
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None  # type: ignore[misc, assignment]
    PYPDF_AVAILABLE = False


# ============================================================
# Data classes
# ============================================================

@dataclass
class DocumentMeta:
    """Metadata about the source document."""
    title: str = ""
    authors: str = ""
    filename: str = ""
    total_pages: int = 0
    sha256: str = ""


@dataclass
class Section:
    """A detected section in the document hierarchy."""
    level: int            # 1 = top-level heading, 2 = subheading, etc.
    title: str            # The heading text
    page_start: int       # 0-indexed page where this section starts
    page_end: int = -1    # 0-indexed page where this section ends (-1 = unknown)
    children: List["Section"] = field(default_factory=list)
    text: str = ""        # The raw text content under this heading


@dataclass
class EnrichedChunk:
    """A single context-enriched chunk ready for RAG ingestion."""
    chunk_id: str                   # Unique identifier
    document_title: str             # Document title
    document_authors: str           # Authors (if available)
    section_breadcrumb: str         # e.g., "Results → Interaction Effects"
    contextual_summary: str         # Brief summary of the section this chunk belongs to
    page_range: str                 # e.g., "pp. 5-6"
    text: str                       # The actual chunk text
    granularity: str = "paragraph"  # "paragraph", "section", or "document"
    char_count: int = 0
    source_filename: str = ""

    def to_enriched_text(self) -> str:
        """Produce the full enriched text for embedding."""
        parts: List[str] = []
        if self.document_title:
            parts.append(f"Document: {self.document_title}")
        if self.document_authors:
            parts.append(f"Authors: {self.document_authors}")
        if self.section_breadcrumb:
            parts.append(f"Section: {self.section_breadcrumb}")
        if self.contextual_summary:
            parts.append(f"Context: {self.contextual_summary}")
        if self.page_range:
            parts.append(f"Pages: {self.page_range}")
        parts.append("")
        parts.append(self.text)
        return "\n".join(parts)


# ============================================================
# PDF extraction
# ============================================================

def _extract_text_by_page_pdfplumber(path: Path) -> List[Tuple[int, str]]:
    """Extract text per page using pdfplumber (better layout preservation)."""
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def _extract_text_by_page_pypdf(path: Path) -> List[Tuple[int, str]]:
    """Extract text per page using pypdf (fallback)."""
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def extract_pdf(path: Path) -> Tuple[DocumentMeta, List[Tuple[int, str]]]:
    """Extract document metadata and per-page text from a PDF."""
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    if PDFPLUMBER_AVAILABLE:
        pages = _extract_text_by_page_pdfplumber(path)
    elif PYPDF_AVAILABLE:
        pages = _extract_text_by_page_pypdf(path)
    else:
        raise RuntimeError(
            "Neither pdfplumber nor pypdf is installed. "
            "Install one with: pip install pdfplumber  OR  pip install pypdf"
        )

    title = ""
    authors = ""
    if PYPDF_AVAILABLE:
        try:
            reader = PdfReader(str(path))
            meta = reader.metadata
            if meta:
                title = (meta.title or "").strip()
                authors = (meta.author or "").strip()
        except Exception:
            pass

    if not title and pages:
        first_lines = pages[0][1].strip().split("\n")
        if first_lines:
            candidate = first_lines[0].strip()
            if 5 < len(candidate) < 200 and not candidate.isdigit():
                title = candidate

    if not title:
        title = path.stem.replace("_", " ").replace("-", " ")

    doc_meta = DocumentMeta(
        title=title,
        authors=authors,
        filename=path.name,
        total_pages=len(pages),
        sha256=sha256,
    )
    return doc_meta, pages


def extract_text_file(path: Path) -> Tuple[DocumentMeta, List[Tuple[int, str]]]:
    """Extract from a plain text or Markdown file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    title = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break
        elif line and not title:
            title = line[:150]

    if not title:
        title = path.stem.replace("_", " ").replace("-", " ")

    doc_meta = DocumentMeta(
        title=title,
        authors="",
        filename=path.name,
        total_pages=1,
        sha256=sha256,
    )
    return doc_meta, [(0, text)]


# ============================================================
# Section / heading detection
# ============================================================

_STANDARD_SECTIONS = {
    "abstract", "introduction", "background", "literature review",
    "methods", "method", "methodology", "materials and methods",
    "participants", "procedure", "design", "measures",
    "results", "findings", "analysis", "data analysis", "statistical analysis",
    "discussion", "implications", "limitations",
    "conclusion", "conclusions", "summary",
    "references", "bibliography", "appendix", "acknowledgements",
    "supplementary", "supplementary materials",
}


def _is_likely_heading(line: str) -> Tuple[bool, int, str]:
    """Return (is_heading, level, clean_title)."""
    line = line.strip()
    if not line or len(line) > 200:
        return False, 0, ""

    md_match = re.match(r"^(#{1,4})\s+(.+)$", line)
    if md_match:
        level = len(md_match.group(1))
        return True, level, md_match.group(2).strip()

    if line.isupper() and len(line) > 4 and " " in line:
        normalised = line.title()
        if normalised.lower().rstrip(":").strip() in _STANDARD_SECTIONS:
            return True, 1, normalised
        return True, 1, normalised

    numbered = re.match(
        r"^(?:([IVXLC]+)\.?\s+|(\d+)\.?\s+|(\d+\.\d+)\.?\s+|(\d+\.\d+\.\d+)\.?\s+)"
        r"(.+)$",
        line,
    )
    if numbered:
        if numbered.group(1):
            return True, 1, numbered.group(5).strip()
        if numbered.group(2):
            return True, 1, numbered.group(5).strip()
        if numbered.group(3):
            return True, 2, numbered.group(5).strip()
        if numbered.group(4):
            return True, 3, numbered.group(5).strip()

    if line.lower().rstrip(":").strip() in _STANDARD_SECTIONS:
        return True, 1, line.strip()

    return False, 0, ""


def detect_sections(pages: List[Tuple[int, str]]) -> List[Section]:
    """Detect document sections from page text."""
    sections: List[Section] = []
    current_section: Optional[Section] = None

    for page_idx, page_text in pages:
        lines = page_text.split("\n")
        for line in lines:
            is_heading, level, clean_title = _is_likely_heading(line)
            if is_heading:
                if current_section is not None:
                    current_section.page_end = page_idx
                    current_section.text = current_section.text.strip()

                current_section = Section(level=level, title=clean_title, page_start=page_idx)
                sections.append(current_section)
            else:
                if current_section is not None:
                    current_section.text += line + "\n"
                elif not sections and line.strip():
                    current_section = Section(level=0, title="Preamble", page_start=page_idx)
                    current_section.text = line + "\n"
                    sections.append(current_section)

    if current_section is not None and pages:
        current_section.page_end = pages[-1][0]
        current_section.text = current_section.text.strip()

    return sections


def build_breadcrumb(sections: List[Section], target_idx: int) -> str:
    """Build a breadcrumb path like "Methods → Statistical Analysis"."""
    target = sections[target_idx]
    ancestors: List[str] = []

    for i in range(target_idx - 1, -1, -1):
        if sections[i].level < target.level:
            ancestors.insert(0, sections[i].title)
            target_level = sections[i].level
            for j in range(i - 1, -1, -1):
                if sections[j].level < target_level:
                    ancestors.insert(0, sections[j].title)
                    target_level = sections[j].level
            break

    ancestors.append(target.title)
    return " → ".join(ancestors)


# ============================================================
# Chunking
# ============================================================

def _clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph breaks."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    # Conservative: good enough for English academic prose without heavy deps
    return re.split(r"(?<=[.!?])\s+(?=[A-Z(“\"'])", text.strip())


def _generate_contextual_summary(section: Section) -> str:
    """Rule-based contextual summary, prioritising definitional/assumption cues."""
    text = _clean_text(section.text or "")
    if not text:
        return ""

    sentences = [s.strip() for s in _split_sentences(text) if s.strip()]
    if not sentences:
        return ""

    cue_patterns = [
        re.compile(r"\bwe\s+define\b", re.I),
        re.compile(r"\bis\s+defined\s+as\b", re.I),
        re.compile(r"\blet\s+.+\s+denote\b", re.I),
        re.compile(r"\bassum(e|es|ing)\b", re.I),
        re.compile(r"\bprovided\s+that\b", re.I),
        re.compile(r"\bunder\s+the\s+assumption\b", re.I),
        re.compile(r"\bsuppose\b", re.I),
        re.compile(r"\bwe\s+(argue|show|prove|demonstrate|propose)\b", re.I),
    ]

    preferred: List[str] = []
    for s in sentences:
        if any(p.search(s) for p in cue_patterns):
            preferred.append(s)
        if len(preferred) >= 2:
            break

    picked = preferred if preferred else sentences[:2]

    out: List[str] = []
    n = 0
    for s in picked:
        if n + len(s) > 320:
            break
        out.append(s)
        n += len(s)

    return " ".join(out)


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[Tuple[int, int, str]]:
    """Character-based chunking (kept for backward compatibility)."""
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = re.split(r"\n\n+", text)

    chunks: List[Tuple[int, int, str]] = []
    current_chunk = ""
    current_start = 0
    running_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            running_pos += 2
            continue

        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append((current_start, current_start + len(current_chunk), current_chunk))

            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = overlap_text + "\n\n" + para
                current_start = running_pos - len(overlap_text)
            else:
                current_chunk = para
                current_start = running_pos
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                current_start = running_pos

        running_pos += len(para) + 2

    if current_chunk.strip():
        chunks.append((current_start, current_start + len(current_chunk), current_chunk))

    return chunks


def chunk_text_words(
    text: str,
    target_words: int = 300,
    overlap_words: int = 60,
) -> List[str]:
    """Paragraph-respecting word-budget chunking with sentence-aligned overlap."""
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal current, cur_len
        if current:
            chunks.append("\n\n".join(current).strip())
        current, cur_len = [], 0

    for para in paragraphs:
        plen = len(para.split())

        if current and (cur_len + plen) > target_words:
            flush()

            if overlap_words > 0 and chunks:
                tail = chunks[-1]
                sents = [s for s in _split_sentences(tail) if s.strip()]
                carry: List[str] = []
                carry_len = 0
                for sent in reversed(sents):
                    slen = len(sent.split())
                    if carry_len + slen > overlap_words:
                        break
                    carry.insert(0, sent.strip())
                    carry_len += slen
                if carry:
                    current = [" ".join(carry).strip()]
                    cur_len = carry_len

        current.append(para)
        cur_len += plen

    flush()
    return chunks


# ============================================================
# Enrichment pipeline
# ============================================================

def enrich_document(
    path: Path,
    chunk_size: int = 1500,
    overlap: int = 200,
    multi_granularity: bool = False,
) -> Tuple[DocumentMeta, List[EnrichedChunk]]:
    """Extract → detect sections → chunk → enrich."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc_meta, pages = extract_pdf(path)
    elif suffix in (".txt", ".md", ".markdown"):
        doc_meta, pages = extract_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    log.info("Extracted %d pages from '%s' (title: '%s')", len(pages), path.name, doc_meta.title)

    sections = detect_sections(pages)
    log.info("Detected %d sections", len(sections))

    if not sections:
        full_text = "\n\n".join(text for _, text in pages)
        sections = [Section(
            level=1,
            title=doc_meta.title or "Document",
            page_start=0,
            page_end=len(pages) - 1,
            text=full_text,
        )]

    all_chunks: List[EnrichedChunk] = []
    chunk_counter = 0

    for sec_idx, section in enumerate(sections):
        if not (section.text or "").strip():
            continue

        breadcrumb = build_breadcrumb(sections, sec_idx)
        context_summary = _generate_contextual_summary(section)

        raw_chunks = chunk_text(section.text, chunk_size=chunk_size, overlap=overlap)

        for _, __, chunk_text_str in raw_chunks:
            chunk_counter += 1
            page_start = section.page_start + 1
            page_end = section.page_end + 1 if section.page_end >= 0 else page_start

            chunk = EnrichedChunk(
                chunk_id=f"{doc_meta.sha256[:12]}:{chunk_counter:05d}",
                document_title=doc_meta.title,
                document_authors=doc_meta.authors,
                section_breadcrumb=breadcrumb,
                contextual_summary=context_summary,
                page_range=f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}",
                text=chunk_text_str,
                granularity="paragraph",
                char_count=len(chunk_text_str),
                source_filename=doc_meta.filename,
            )
            all_chunks.append(chunk)

        if multi_granularity and len(raw_chunks) > 1:
            section_text = _clean_text(section.text)
            if len(section_text) > 5000:
                section_text = section_text[:5000] + " [...]"

            chunk_counter += 1
            all_chunks.append(EnrichedChunk(
                chunk_id=f"{doc_meta.sha256[:12]}:S{sec_idx:03d}",
                document_title=doc_meta.title,
                document_authors=doc_meta.authors,
                section_breadcrumb=breadcrumb,
                contextual_summary=context_summary,
                page_range=f"pp. {section.page_start + 1}-{section.page_end + 1}",
                text=section_text,
                granularity="section",
                char_count=len(section_text),
                source_filename=doc_meta.filename,
            ))

    if multi_granularity:
        doc_summary_parts: List[str] = []
        for section in sections:
            if (section.text or "").strip():
                summary = _generate_contextual_summary(section)
                if summary:
                    doc_summary_parts.append(f"{section.title}: {summary}")
        doc_summary = "\n".join(doc_summary_parts)
        if doc_summary:
            all_chunks.append(EnrichedChunk(
                chunk_id=f"{doc_meta.sha256[:12]}:DOC",
                document_title=doc_meta.title,
                document_authors=doc_meta.authors,
                section_breadcrumb="Document Summary",
                contextual_summary="Overview of all sections in the document.",
                page_range=f"pp. 1-{doc_meta.total_pages}",
                text=doc_summary,
                granularity="document",
                char_count=len(doc_summary),
                source_filename=doc_meta.filename,
            ))

    log.info(
        "Generated %d enriched chunks (%d paragraph, %d section, %d document)",
        len(all_chunks),
        sum(1 for c in all_chunks if c.granularity == "paragraph"),
        sum(1 for c in all_chunks if c.granularity == "section"),
        sum(1 for c in all_chunks if c.granularity == "document"),
    )

    return doc_meta, all_chunks


# ============================================================
# Output formatters
# ============================================================

def write_json(chunks: List[EnrichedChunk], output_path: Path) -> None:
    data = []
    for chunk in chunks:
        d = asdict(chunk)
        d["enriched_text"] = chunk.to_enriched_text()
        data.append(d)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d chunks to %s", len(chunks), output_path)


def write_jsonl(chunks: List[EnrichedChunk], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            d = asdict(chunk)
            d["enriched_text"] = chunk.to_enriched_text()
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log.info("Wrote %d chunks to %s", len(chunks), output_path)


def write_markdown(chunks: List[EnrichedChunk], output_path: Path) -> None:
    lines: List[str] = ["# RAG-Optimised Chunks\n"]
    for chunk in chunks:
        lines.append(f"## Chunk {chunk.chunk_id} [{chunk.granularity}]\n")
        lines.append(f"**Document:** {chunk.document_title}  ")
        if chunk.document_authors:
            lines.append(f"**Authors:** {chunk.document_authors}  ")
        lines.append(f"**Section:** {chunk.section_breadcrumb}  ")
        lines.append(f"**Pages:** {chunk.page_range}  ")
        if chunk.contextual_summary:
            lines.append(f"**Context:** {chunk.contextual_summary}  ")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")
        lines.append("---\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %d chunks to %s", len(chunks), output_path)


def write_text_files(chunks: List[EnrichedChunk], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        filename = f"{chunk.chunk_id.replace(':', '_')}.txt"
        (output_dir / filename).write_text(chunk.to_enriched_text(), encoding="utf-8")
    log.info("Wrote %d chunk files to %s", len(chunks), output_dir)


# ============================================================
# NotebookLM companion ("restructured")
# ============================================================

def _extract_definitions_and_assumptions(
    blocks: List[Tuple[str, str, str]],
) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str]]]:
    """
    Extract (very) lightweight indices from (anchor, page_range, text) blocks.

    Returns:
      definitions: term -> list of (anchor, page_range)
      assumptions: list of (anchor, page_range)

    This is intentionally conservative and deterministic.
    """
    definitions: Dict[str, List[Tuple[str, str]]] = {}
    assumptions: List[Tuple[str, str]] = []

    # Definition patterns
    #  - "We define X as ..."
    #  - "X is defined as ..."
    #  - "Let X denote ..."
    p_we_define = re.compile(r"\bwe\s+define\s+([^\.\n]{1,80}?)\s+as\b", re.I)
    p_is_defined = re.compile(r"\b([^\.\n]{1,80}?)\s+is\s+defined\s+as\b", re.I)
    p_let_denote = re.compile(r"\blet\s+([^\.\n]{1,60}?)\s+denote\b", re.I)

    p_assume = re.compile(r"\b(assume|assuming|assumption|suppose|provided\s+that|under\s+the\s+assumption)\b", re.I)

    def _normalise_term(raw: str) -> str:
        t = raw.strip().strip(":;,.()[]{}\"'“”")
        t = re.sub(r"\s+", " ", t)
        # Prevent pathological long captures
        if len(t) > 80:
            t = t[:80].rstrip() + "…"
        return t

    for anchor, page_range, text in blocks:
        # Scan sentence-wise for precision
        for sent in _split_sentences(_clean_text(text)):
            s = sent.strip()
            if not s:
                continue

            m1 = p_we_define.search(s)
            if m1:
                term = _normalise_term(m1.group(1))
                if term:
                    definitions.setdefault(term, []).append((anchor, page_range))

            m2 = p_is_defined.search(s)
            if m2:
                term = _normalise_term(m2.group(1))
                # Avoid capturing pronouns or empty junk
                if term and term.lower() not in {"it", "this", "that", "they", "we"}:
                    definitions.setdefault(term, []).append((anchor, page_range))

            m3 = p_let_denote.search(s)
            if m3:
                term = _normalise_term(m3.group(1))
                if term:
                    definitions.setdefault(term, []).append((anchor, page_range))

            if p_assume.search(s):
                assumptions.append((anchor, page_range))
                break

    # De-duplicate and stabilise ordering
    for term, refs in list(definitions.items()):
        seen = set()
        uniq: List[Tuple[str, str]] = []
        for a, p in refs:
            key = (a, p)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((a, p))
        definitions[term] = uniq

    # Dedupe assumptions
    seen_a = set()
    uniq_a: List[Tuple[str, str]] = []
    for a, p in assumptions:
        if (a, p) in seen_a:
            continue
        seen_a.add((a, p))
        uniq_a.append((a, p))

    return definitions, uniq_a


def write_restructured_document(
    doc_meta: DocumentMeta,
    sections: List[Section],
    output_path: Path,
    companion_words: int = 300,
    companion_overlap_words: int = 60,
    include_indices: bool = True,
) -> None:
    """
    Write a NotebookLM-style companion Markdown document.

    Key features:
      - Stable per-block anchors (DOCID:Cxxxxx)
      - Page spans per section and per block
      - Breadcrumb/path hints
      - Optional indices for definitions and assumptions
      - Word-based chunking to avoid mid-sentence slicing

    Upload this alongside the original PDF to improve retrieval.
    """

    doc_id = doc_meta.sha256[:12]

    # Build section order as they appear (already in order from detect_sections)
    ordered_sections = [s for s in sections if (s.text or "").strip()]

    # Pre-build blocks to enable indices
    block_records: List[Tuple[str, str, str, str]] = []  # (breadcrumb, anchor, pages, text)

    # Chunk counters
    counter = 0

    for sec_idx, section in enumerate(ordered_sections):
        breadcrumb = build_breadcrumb(ordered_sections, sec_idx) if ordered_sections else section.title
        pages = f"p. {section.page_start + 1}" if section.page_start == section.page_end else f"pp. {section.page_start + 1}-{section.page_end + 1}"

        blocks = chunk_text_words(section.text, target_words=companion_words, overlap_words=companion_overlap_words)
        for b in blocks:
            counter += 1
            anchor = f"{doc_id}:C{counter:05d}"
            block_records.append((breadcrumb, anchor, pages, b))

    # Indices are built over blocks only (not section summaries)
    defs_index: Dict[str, List[Tuple[str, str]]] = {}
    assumptions_index: List[Tuple[str, str]] = []
    if include_indices:
        defs_index, assumptions_index = _extract_definitions_and_assumptions(
            [(a, p, t) for _, a, p, t in block_records]
        )

    # Start writing
    lines: List[str] = [f"# {doc_meta.title}\n"]
    if doc_meta.authors:
        lines.append(f"*{doc_meta.authors}*\n")
    lines.append(f"*Source filename: {doc_meta.filename}*\n")
    lines.append(f"*Document ID: {doc_id}*\n")
    lines.append("")

    lines.append("## How to use this companion\n")
    lines.append("Upload this Markdown alongside the original PDF. Use the anchors to request surrounding context (Prev/Next) or to cite evidence deterministically.\n")

    if include_indices:
        lines.append("## Definitions index\n")
        if defs_index:
            for term in sorted(defs_index.keys(), key=lambda s: s.lower()):
                refs = ", ".join([f"{a} ({p})" for a, p in defs_index[term][:5]])
                lines.append(f"- **{term}** → {refs}")
        else:
            lines.append("- (No definition cues detected by the rule-based extractor.)")
        lines.append("")

        lines.append("## Assumptions index\n")
        if assumptions_index:
            for a, p in assumptions_index[:30]:
                lines.append(f"- {a} ({p})")
            if len(assumptions_index) > 30:
                lines.append(f"- … ({len(assumptions_index) - 30} more)")
        else:
            lines.append("- (No assumption cues detected by the rule-based extractor.)")
        lines.append("")

    # Now emit sections with headers and blocks
    # We need fast access from breadcrumb to blocks while preserving original order.
    blocks_by_breadcrumb: Dict[str, List[Tuple[str, str]]] = {}  # breadcrumb -> [(anchor, text)]
    pages_by_breadcrumb: Dict[str, str] = {}

    for breadcrumb, anchor, pages, text in block_records:
        blocks_by_breadcrumb.setdefault(breadcrumb, []).append((anchor, text))
        pages_by_breadcrumb[breadcrumb] = pages

    prev_anchor: Optional[str] = None
    all_anchors = [a for _, a, _, _ in block_records]
    anchor_to_next: Dict[str, Optional[str]] = {}
    for i, a in enumerate(all_anchors):
        anchor_to_next[a] = all_anchors[i + 1] if i + 1 < len(all_anchors) else None

    # Emit in the order sections were detected
    for sec_idx, section in enumerate(ordered_sections):
        breadcrumb = build_breadcrumb(ordered_sections, sec_idx) if ordered_sections else section.title
        parts = [p.strip() for p in breadcrumb.split(" → ") if p.strip()]
        depth = len(parts)
        heading_prefix = "#" * min(depth + 1, 4)  # ##, ###, ####

        heading_text = parts[-1] if parts else breadcrumb
        if depth > 2:
            heading_text = f"{parts[-2]}: {parts[-1]}"

        lines.append(f"{heading_prefix} {heading_text}\n")
        if depth > 1:
            lines.append(f"*Section path: {breadcrumb}*\n")

        sec_pages = pages_by_breadcrumb.get(breadcrumb, "")
        if sec_pages:
            lines.append(f"*Pages: {sec_pages}*\n")

        ctx = _generate_contextual_summary(section)
        if ctx:
            lines.append(f"*Context cue: {ctx}*\n")

        section_blocks = blocks_by_breadcrumb.get(breadcrumb, [])
        for anchor, text in section_blocks:
            next_anchor = anchor_to_next.get(anchor)
            lines.append(f"[Anchor: {anchor}]" + (f" [Prev: {prev_anchor}]" if prev_anchor else "") + (f" [Next: {next_anchor}]" if next_anchor else ""))
            if sec_pages:
                lines.append(f"[Pages: {sec_pages}]")
            lines.append("")
            lines.append(text)
            lines.append("\n---\n")
            prev_anchor = anchor

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote NotebookLM companion document to %s", output_path)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Chunk Enricher — produce context-enriched chunks for any RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s paper.pdf --format json
              %(prog)s paper.pdf --format jsonl --chunk-size 1000 --overlap 150
              %(prog)s paper.pdf --format markdown --multi-granularity
              %(prog)s paper.pdf --format text --output-dir ./chunks/
              %(prog)s paper.pdf --format restructured  # NotebookLM companion
        """),
    )
    parser.add_argument("input", type=Path, help="Input file (PDF, .txt, or .md)")
    parser.add_argument(
        "--format", "-f",
        choices=["json", "jsonl", "markdown", "text", "restructured", "all"],
        default="json",
        help="Output format (default: json). 'all' produces every format.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input file)",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=1500,
        help="Target chunk size in characters for JSON/JSONL exports (default: 1500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap in characters for JSON/JSONL exports (default: 200)",
    )
    parser.add_argument(
        "--multi-granularity", "-m",
        action="store_true",
        help="Also generate section-level and document-level chunks",
    )

    # Companion (restructured) controls
    parser.add_argument(
        "--companion-words",
        type=int,
        default=300,
        help="Target block size in words for the NotebookLM companion (default: 300)",
    )
    parser.add_argument(
        "--companion-overlap-words",
        type=int,
        default=60,
        help="Overlap in words (sentence-aligned) for the NotebookLM companion (default: 60)",
    )
    parser.add_argument(
        "--no-indices",
        action="store_true",
        help="Do not include definitions/assumptions indices in the companion",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    doc_meta, chunks = enrich_document(
        args.input,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        multi_granularity=args.multi_granularity or args.format in ("all", "restructured"),
    )

    # Extract sections for companion (restructured)
    suffix = args.input.suffix.lower()
    if suffix == ".pdf":
        _, pages = extract_pdf(args.input)
    else:
        _, pages = extract_text_file(args.input)
    sections = detect_sections(pages)

    formats_to_write = [args.format] if args.format != "all" else [
        "json", "jsonl", "markdown", "text", "restructured",
    ]

    for fmt in formats_to_write:
        if fmt == "json":
            write_json(chunks, output_dir / f"{stem}_enriched_chunks.json")
        elif fmt == "jsonl":
            write_jsonl(chunks, output_dir / f"{stem}_enriched_chunks.jsonl")
        elif fmt == "markdown":
            write_markdown(chunks, output_dir / f"{stem}_enriched_chunks.md")
        elif fmt == "text":
            write_text_files(chunks, output_dir / f"{stem}_chunks")
        elif fmt == "restructured":
            write_restructured_document(
                doc_meta=doc_meta,
                sections=sections,
                output_path=output_dir / f"{stem}_notebooklm_companion.md",
                companion_words=args.companion_words,
                companion_overlap_words=args.companion_overlap_words,
                include_indices=not args.no_indices,
            )

    # Print summary
    print(f"\n{'=' * 60}")
    print("  RAG Chunk Enricher — Summary")
    print(f"{'=' * 60}")
    print(f"  Document:     {doc_meta.title}")
    if doc_meta.authors:
        print(f"  Authors:      {doc_meta.authors}")
    print(f"  Pages:        {doc_meta.total_pages}")
    print(f"  Sections:     {len(sections)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"    Paragraph:  {sum(1 for c in chunks if c.granularity == 'paragraph')}")
    print(f"    Section:    {sum(1 for c in chunks if c.granularity == 'section')}")
    print(f"    Document:   {sum(1 for c in chunks if c.granularity == 'document')}")
    print(f"  Chunk size:   {args.chunk_size} chars (overlap: {args.overlap})")
    if args.format in ("restructured", "all"):
        print(f"  Companion:    {args.companion_words} words (overlap: {args.companion_overlap_words} words)")
    print(f"  Output:       {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

# --- END embedded implementation ---

def run(argv=None):
    """Programmatic entry point (used by console_scripts)."""
    return main(argv)

if __name__ == "__main__":
    raise SystemExit(run())
