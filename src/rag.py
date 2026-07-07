import os
from dotenv import load_dotenv
from groq import Groq

from .config import (
    LLM_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    PROMPTS_DIR,
)
from .vector_db import VectorDB
from .moderateur import Moderateur


class RAG:
    def __init__(self):
        # 1) Charger les variables d'environnement depuis .env.
        #    load_dotenv() lit le fichier .env à la racine et expose
        #    GROQ_API_KEY via os.environ. La clé reste HORS du code.
        load_dotenv()
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY introuvable. Crée un fichier .env à la racine "
                "avec la ligne : GROQ_API_KEY=ta_cle"
            )

        # 2) Client Groq : il lit GROQ_API_KEY automatiquement.
        self.client = Groq()

        # 3) L'agent modérateur (Brique 2), à qui on passe LE client.
        self.moderateur = Moderateur(self.client)

        # 4) La base vectorielle (Brique 1). On ne passe PAS de chunks :
        #    la base doit déjà exister sur disque (créée via index_corpus).
        #    Si elle n'existe pas, VectorDB lèvera une erreur explicite.
        self.db = VectorDB(CHROMA_PATH, COLLECTION_NAME)

        # 5) Le prompt "à trous" du RAG, chargé depuis son fichier.
        chemin_prompt = PROMPTS_DIR / "rag_prompt.txt"
        with open(chemin_prompt, encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.model = LLM_MODEL

    # ==================================================================
    def answer_question(self, question):
        """Pipeline complet : modération -> retrieval -> LLM."""

        # --- ÉTAPE SÉCURITÉ (avant tout) ---------------------------------
        verdict = self.moderateur.moderate(question)
        if verdict["is_prompt_injection"]:
            # On s'arrête ici : le LLM principal n'est jamais appelé.
            return ("Question bloquée : une tentative de détournement "
                    "(injection de prompt) a été détectée.")

        # --- ÉTAPE RETRIEVAL --------------------------------------------
        # On récupère les 3 chunks les plus proches (Brique 1).
        chunks = self.db.retrieve(question, n=3)

        # --- ÉTAPE PROMPT ------------------------------------------------
        # On remplace le marqueur {{Chunks}} du template par les chunks
        # mis en forme. Le prompt système est ainsi "rempli" à la volée.
        contexte = self._formater_chunks(chunks)
        system_prompt = self.prompt_template.replace("{{Chunks}}", contexte)

        # --- ÉTAPE GÉNÉRATION -------------------------------------------
        # message system = prompt rempli ; message user = la question.
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return completion.choices[0].message.content

    # ------------------------------------------------------------------
    @staticmethod
    def _formater_chunks(chunks):
        """Met les chunks en une liste numérotée lisible pour le LLM."""
        lignes = []
        for i, c in enumerate(chunks, start=1):
            # On montre le texte et sa source ; l'ordre (1, 2, 3) reflète
            # la pertinence décroissante, ce que le prompt explique au LLM.
            lignes.append(f"[{i}] (source: {c['source']}) {c['text']}")
        return "\n".join(lignes)
