# NotebookLM Companion Generator

This tool generates a *companion Markdown* file from an academic PDF to improve retrieval quality in NotebookLM (and similar RAG systems).

## What it produces

For an input `paper.pdf`, it creates:

- `paper_notebooklm_companion.md`

The companion contains stable anchors, section breadcrumbs, page spans, and (optionally) indices for definitions/assumptions.

## Recommended NotebookLM workflow

Upload **both**:
1) The original PDF (authoritative ground truth for quotes/pages)
2) The generated companion Markdown (retrieval accelerator)

A reliable prompting pattern is:
- “Use the PDF as the authoritative source. Use the companion to locate anchors/pages. Quote from the PDF.”

---

## Option A: Install as a CLI (pip)

```bash
pip install -e .
notebooklm-companion /path/to/paper.pdf -f restructured
```

---

## Desktop app (Windows/macOS) via GitHub Releases

Download a release build from GitHub Releases:
- Windows: `NotebookLM-Companion-windows.zip` → contains `NotebookLM-Companion.exe`
- macOS: `NotebookLM-Companion-macos.zip` → contains `NotebookLM-Companion.app`

Choose a PDF and click **Generate companion**. The `.md` file is created next to the PDF.

---

## Developers: build desktop apps locally

```bash
python -m pip install pyinstaller
pyinstaller --noconfirm --clean --windowed --onefile --name "NotebookLM-Companion" notebooklm_companion/gui_app.py
```

Outputs appear in `dist/`.
