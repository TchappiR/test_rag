import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

try :
	GROQ_API_KEY=os.environ["GROQ_API_KEY"]
except:
	raise(Exception("The groq api does not exist"))

EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

MODERATION_MODEL = "meta-llama/llama-guard-4-12b"

BASE_DIR = Path(__file__).parent

# Dossier où ChromaDB écrira ses fichiers persistés
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# Nom logique de la collection à l'intérieur de ChromaDB.
COLLECTION_NAME = "corpus_villebrume"

# Fichier CSV contenant le corpus
CORPUS_CSV = str(BASE_DIR / "05_corpus_rag.csv")
