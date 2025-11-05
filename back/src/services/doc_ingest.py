# back/src/services/doc_ingest.py
from __future__ import annotations
import re

def _strip_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def extract_pdf_text(path: str, max_chars: int = 6000) -> str:
    """
    Extrae texto de un PDF. Intenta PyMuPDF (fitz) y cae a pypdf.
    Devuelve texto limpio truncado a max_chars.
    """
    text = ""
    # 1) PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        parts = []
        for page in doc:
            parts.append(page.get_text() or "")
            if sum(len(p) for p in parts) >= max_chars * 2:  # corte temprano
                break
        text = " ".join(parts)
    except Exception:
        # 2) pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            parts = []
            for i, page in enumerate(reader.pages):
                parts.append(page.extract_text() or "")
                if sum(len(p) for p in parts) >= max_chars * 2:
                    break
            text = " ".join(parts)
        except Exception:
            text = ""

    return _strip_ws(text)[:max_chars]
