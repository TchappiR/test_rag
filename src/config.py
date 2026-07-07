import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try :
	GROQ_API_KEY=os.environ["GROQ_API_KEY"]
except:
	raise(Exception("The groq api does not exist"))

EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

LLM_MODEL = "openai/gpt-oss-120b"

MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"

# Dossier où ChromaDB écrira ses fichiers persistés
CHROMA_PATH = str(DATA_DIR / "chroma_db")

# Nom logique de la collection à l'intérieur de ChromaDB.
COLLECTION_NAME = "corpus_villebrume"

# Fichier CSV contenant le corpus
CORPUS_CSV = str(DATA_DIR / "05_corpus_rag.csv")