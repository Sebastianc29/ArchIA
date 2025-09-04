from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import shutil

# .env (usa .env.development si lo tienes)
#load_dotenv(dotenv_path=find_dotenv('.env.development'))

# Rutas robustas ancladas a back/
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"   

# Loader (lee TODO lo que haya en docs/)
loader = DirectoryLoader(str(DOCS_DIR))

COLLECTION_NAME = "rag-chroma"

def _add_metadata(docs):
    """
    Normaliza y enriquece metadatos para cada documento:
    - source_path: ruta original del archivo
    - title: nombre base del archivo (sirve para filtrar/citar)
    """
    for d in docs:
        d.metadata = d.metadata or {}
        # Algunos loaders guardan 'source', otros 'file_path'
        source = d.metadata.get("source") or d.metadata.get("file_path") or ""
        d.metadata.setdefault("source_path", str(source))
        # Título por defecto: nombre del archivo sin extensión
        if source:
            d.metadata.setdefault("title", Path(source).stem)
        else:
            d.metadata.setdefault("title", "docs")
    return docs

def create_or_load_vectorstore():
    """
    Crea la BD de Chroma si no existe; si existe, la carga.
    """
    embedding = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        print(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embedding,
            collection_name=COLLECTION_NAME
        )
    else:
        print(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        # 1) Cargar documentos
        docs = loader.load()

        # 2) Añadir metadatos útiles (source_path, title)
        docs = _add_metadata(docs)

        # 3) Split (chunking)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        doc_splits = text_splitter.split_documents(docs)

        # 4) Construir y persistir Chroma
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=COLLECTION_NAME,
            embedding=embedding,
            persist_directory=str(CHROMA_DB_DIR)
        )
        vectorstore.persist()

    return vectorstore

def get_retriever(title: str = None, k: int = 4):
    """
    Devuelve un retriever del vector store.
    Si pasas 'title', filtra los chunks al libro/archivo con ese título.
    """
    vectorstore = create_or_load_vectorstore()
    if title:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": {"title": {"$eq": title}}}
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def rebuild_vectorstore():
    """
    Borra la BD existente y la reconstruye (úsalo si cambiaste docs/ o parámetros).
    """
    if CHROMA_DB_DIR.exists():
        print(f"Removing existing Chroma DB at {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)
    return create_or_load_vectorstore()

# if __name__ == "__main__":
#     vs = create_or_load_vectorstore()
#     print("✅ Vectorstore listo")
# src/rag_agent.py
import os
from pathlib import Path

# usa el paquete nuevo (quita el warning)
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
# ... (otros imports)

# === ruta por defecto: back/chroma_db ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../back
DEFAULT_CHROMA_DIR = str((PROJECT_ROOT / "chroma_db").resolve())

def create_or_load_vectorstore():
    persist_directory = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    print(f"[RAG] persist_directory = {persist_directory}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # si ya existe algo, solo cargar
    if os.path.isdir(persist_directory) and any(Path(persist_directory).iterdir()):
        print(f"Loading existing Chroma DB from {persist_directory}")
        return Chroma(
            collection_name="arquia",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    # ... aquí tu código para crear el índice si no existe (o delega al build)
    # (puedes mantener el resto como lo tienes)
