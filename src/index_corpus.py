"""
src/index_corpus.py — Construire puis tester la Brique 1
========================================================

À lancer depuis la RACINE du projet, en tant que module :

    poetry run python -m src.index_corpus

Le 1er lancement CRÉE la base (téléchargement du modèle + encodage) ;
les suivants la RECHARGENT instantanément (preuve de la persistance).

Test-clé du TP : « Quelle est la couleur du chat de Bob ? » doit faire
remonter EN TÊTE la phrase sur Henri le chat bleu.
"""

import csv

from .config import CORPUS_CSV, CHROMA_PATH, COLLECTION_NAME
from .vector_db import VectorDB


def charger_corpus(chemin_csv):
    """Lit le CSV et renvoie une liste de dicts (un par chunk)."""
    with open(chemin_csv, newline="", encoding="utf-8") as f:
        # DictReader : la 1re ligne (en-tête) donne les noms de colonnes,
        # donc chaque ligne devient un dict {"id","text","source","categorie"}.
        return list(csv.DictReader(f))


def afficher(question, resultats):
    print(f"\n QUESTION : {question}")
    for rang, chunk in enumerate(resultats, start=1):
        print(f"  {rang}. (dist={chunk['distance']:.3f}) {chunk['text']}")


def main():
    # 1) charger le corpus
    chunks = charger_corpus(CORPUS_CSV)
    print(f"Corpus chargé : {len(chunks)} chunks.")

    # 2) construire OU recharger la base (l'aiguillage décide tout seul)
    db = VectorDB(
        persist_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        chunks=chunks,
    )
    print(f"Base prête. Modèle d'embedding : {db.model_name}")

    # 3) les 5 questions de test
    questions = [
        "Quelle est la couleur du chat de Bob ?",         # -> Henri, chat BLEU
        "Comment s'appelle le chien d'Alice ?",           # -> Gaston, chien vert
        "Que collectionne la tortue de Carla ?",          # -> bouchons de liège
        "Combien Villebrume compte-t-elle d'habitants ?", # -> 212
        "Quel métier consiste à ranger le brouillard ?",  # -> plieur de brume
    ]
    for q in questions:
        afficher(q, db.retrieve(q, n=3))


if __name__ == "__main__":
    main()
