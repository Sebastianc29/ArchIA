# back/src/services/doc_ingest.py
from __future__ import annotations
import re
from typing import Optional, Dict, List

def _strip_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def _extract_text_pymupdf(path: str, max_chars: int) -> str:
    """Extrae texto usando PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        parts = []
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        
        for page_num, page in enumerate(doc):
            text = page.get_text() or ""
            if text.strip():
                parts.append(f"[Page {page_num + 1}]\n{text}")
            if sum(len(p) for p in parts) >= max_chars * 2:
                break
        
        doc.close()
        return " ".join(parts), metadata
    except Exception as e:
        return "", {}

def _extract_text_pypdf(path: str, max_chars: int) -> str:
    """Extrae texto usando pypdf (fallback)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        metadata = reader.metadata or {}
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"[Page {page_num + 1}]\n{text}")
            if sum(len(p) for p in parts) >= max_chars * 2:
                break
        
        return " ".join(parts), metadata
    except Exception:
        return "", {}

def extract_pdf_text(path: str, max_chars: int = 8000) -> str:
    """
    Extrae texto de un PDF. Intenta PyMuPDF (fitz) primero, luego pypdf.
    Devuelve texto limpio truncado a max_chars.
    
    Args:
        path: Ruta al archivo PDF
        max_chars: Máximo de caracteres a extraer (default 8000)
    
    Returns:
        Texto extraído del PDF
    """
    text = ""
    
    # 1) PyMuPDF (más robusto)
    text, _ = _extract_text_pymupdf(path, max_chars)
    
    # 2) Fallback a pypdf si PyMuPDF falló
    if not text.strip():
        text, _ = _extract_text_pypdf(path, max_chars)
    
    # Normalizar espacios en blanco
    cleaned = _strip_ws(text)
    return cleaned[:max_chars]

def extract_pdf_with_metadata(path: str, max_chars: int = 8000) -> Dict[str, any]:
    """
    Extrae texto y metadatos de un PDF.
    
    Returns:
        Dict con 'text' y 'metadata'
    """
    text = ""
    metadata = {}
    
    # PyMuPDF primero
    text, metadata = _extract_text_pymupdf(path, max_chars)
    
    # Fallback
    if not text.strip():
        text, metadata = _extract_text_pypdf(path, max_chars)
    
    return {
        "text": _strip_ws(text)[:max_chars],
        "metadata": metadata or {},
        "char_count": len(text)
    }

def extract_images_from_pdf(path: str, max_images: int = 5) -> List[str]:
    """
    Extrae imágenes de un PDF y las guarda como archivos temporales.
    
    Args:
        path: Ruta al PDF
        max_images: Máximo número de imágenes a extraer
    
    Returns:
        Lista de rutas de archivos de imagen
    """
    import os
    from pathlib import Path
    
    images = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        temp_dir = Path(path).parent / ".temp_images"
        temp_dir.mkdir(exist_ok=True)
        
        img_count = 0
        for page_num, page in enumerate(doc):
            if img_count >= max_images:
                break
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                if img_count >= max_images:
                    break
                
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_path = temp_dir / f"page_{page_num + 1}_img_{img_index}.png"
                pix.save(str(img_path))
                images.append(str(img_path))
                img_count += 1
        
        doc.close()
    except Exception as e:
        pass
    
    return images
