import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from notebooklm_companion.core import main as core_main


APP_TITLE = "NotebookLM Companion Generator"


def _run_generation(pdf_path: Path, done_callback):
    try:
        args = [str(pdf_path), "-f", "restructured"]
        try:
            core_main(args)
        except SystemExit as e:
            if getattr(e, "code", 0) not in (0, None):
                raise RuntimeError(f"Generator exited with code {e.code}") from None

        stem = pdf_path.stem
        companion = pdf_path.with_name(f"{stem}_notebooklm_companion.md")
        if not companion.exists():
            matches = list(pdf_path.parent.glob("*_notebooklm_companion.md"))
            companion = matches[0] if matches else None

        if not companion or not companion.exists():
            raise FileNotFoundError("Could not locate generated companion Markdown file.")

        done_callback(success=True, companion_path=companion, error=None)

    except Exception as e:
        done_callback(success=False, companion_path=None, error=e)


def main():
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("520x240")
    root.resizable(False, False)

    selected_pdf = tk.StringVar(value="")
    status_var = tk.StringVar(value="Select a PDF to begin.")

    def choose_pdf():
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path:
            selected_pdf.set(path)
            status_var.set("Ready. Click Generate.")

    def on_done(success: bool, companion_path: Path | None, error: Exception | None):
        def _update():
            generate_btn.config(state=tk.NORMAL)
            choose_btn.config(state=tk.NORMAL)
            if success and companion_path:
                status_var.set(f"Done. Created: {companion_path.name}")
                messagebox.showinfo(
                    APP_TITLE,
                    "Companion generated successfully.\n\n"
                    f"Created file:\n{companion_path}\n\n"
                    "Upload BOTH files to NotebookLM (or your RAG):\n"
                    "1) The original PDF (ground truth)\n"
                    "2) The companion Markdown (retrieval accelerator)"
                )
            else:
                status_var.set("Failed. See details.")
                details = "".join(traceback.format_exception(error)) if error else "Unknown error."
                messagebox.showerror(APP_TITLE, f"Generation failed.\n\n{details}")
        root.after(0, _update)

    def generate():
        path = selected_pdf.get().strip()
        if not path:
            messagebox.showwarning(APP_TITLE, "Please select a PDF first.")
            return
        pdf_path = Path(path)
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            messagebox.showwarning(APP_TITLE, "Selected file is not a valid PDF.")
            return

        status_var.set("Generating companion…")
        generate_btn.config(state=tk.DISABLED)
        choose_btn.config(state=tk.DISABLED)

        t = threading.Thread(
            target=_run_generation,
            args=(pdf_path, on_done),
            daemon=True,
        )
        t.start()

    pad = {"padx": 12, "pady": 8}

    header = tk.Label(root, text="Generate a NotebookLM-optimised companion Markdown from a PDF", font=("Helvetica", 12, "bold"))
    header.pack(anchor="w", **pad)

    frame = tk.Frame(root)
    frame.pack(fill="x", **pad)

    choose_btn = tk.Button(frame, text="Choose PDF…", command=choose_pdf, width=14)
    choose_btn.pack(side="left")

    pdf_label = tk.Label(frame, textvariable=selected_pdf, wraplength=350, justify="left")
    pdf_label.pack(side="left", padx=10)

    generate_btn = tk.Button(root, text="Generate companion", command=generate, width=22, height=2)
    generate_btn.pack(**pad)

    status = tk.Label(root, textvariable=status_var, anchor="w", fg="#333333")
    status.pack(fill="x", **pad)

    hint = tk.Label(
        root,
        text="Tip: Upload BOTH the original PDF and the generated companion .md into NotebookLM.",
        anchor="w",
        fg="#555555",
        wraplength=500,
        justify="left",
    )
    hint.pack(fill="x", padx=12, pady=(0, 10))

    root.mainloop()


if __name__ == "__main__":
    main()
