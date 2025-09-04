# back/build_vectorstore.py
from __future__ import annotations
import os, shutil
from pathlib import Path
import argparse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.development"), override=False)
load_dotenv(find_dotenv(".env"), override=False)

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent          # .../back
DEFAULT_CHROMA_DIR = str((BASE_DIR / "chroma_db").resolve())
DOCS_DIR = BASE_DIR / "docs"

def build_vectorstore(persist_directory: str, force_rebuild: bool=False):
    if force_rebuild and os.path.isdir(persist_directory):
        print(f"[build] Borrando BD previa: {persist_directory}")
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)

    print(f"[build] docs_dir          = {DOCS_DIR}")
    print(f"[build] persist_directory = {persist_directory}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pdf_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(str(DOCS_DIR), glob="**/*.txt",
                                 loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8", "errors": "ignore"})

    raw_docs = []
    for loader in (pdf_loader, txt_loader):
        try:
            loaded = loader.load()
            raw_docs += loaded
            print(f"[build] Cargados {len(loaded):4d} docs de {loader.__class__.__name__}")
        except Exception as e:
            print(f"[WARN] loader error ({loader.__class__.__name__}): {e}")

    for d in raw_docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or ""
        d.metadata["source"] = os.path.basename(src)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    print(f"[build] Total chunks: {len(chunks)}")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="arquia",
        persist_directory=persist_directory,
    )
    # desde Chroma 0.4.x persiste automático
    print("[build] ¡Vector store construido y persistido!")

    # Resumen
    vs = Chroma(collection_name="arquia", embedding_function=embeddings,
                persist_directory=persist_directory)
    metas = (vs._collection.get(include=['metadatas']).get('metadatas')) or []
    print(f"[build] Resumen -> chunks en BD: {len(metas)}")
    print("[build] Fuentes (primeras 20):",
          *sorted({(m or {}).get("source","unknown") for m in metas if m})[:20],
          sep="\n   - ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Borra y reconstruye")
    args = parser.parse_args()

    persist_directory = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    build_vectorstore(persist_directory=persist_directory, force_rebuild=args.rebuild)
    print(f"[build] Listo. BD en: {persist_directory}")

if __name__ == "__main__":
    main()
