from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv

# Asegura que 'back/' esté en sys.path para poder importar 'src'
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Carga variables de entorno (si usas .env.development)
load_dotenv(find_dotenv('.env.development'))

from src.rag_agent import create_or_load_vectorstore 

if __name__ == "__main__":
    create_or_load_vectorstore()
    print("Vectorstore listo en back/chroma_db")
