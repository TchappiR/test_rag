import chromadb
from sentence_transformers import SentenceTransformer

import config


class VectorDB:
    # ------------------------------------------------------------------
    # CONSTRUCTEUR — c'est lui qui contient l'aiguillage crée/recharge
    # ------------------------------------------------------------------
    def __init__(self, persist_path, collection_name, chunks=None):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_path)
        collections_existantes = [
            c.name if hasattr(c, "name") else c
            for c in self.client.list_collections()
        ]
        base_existe_deja = collection_name in collections_existantes

        if base_existe_deja:
            self._recharger()
        elif chunks is not None:
            self._creer(chunks)
        else:
            raise ValueError(
                f"Aucune collection '{collection_name}' trouvée dans "
                f"'{persist_path}', et aucun chunk fourni pour la créer. "
                f"Passez l'argument 'chunks=' pour construire la base."
            )

    # ==================================================================
    #  ÉTAPE 3.1 — CRÉER la base
    # ==================================================================
    def _creer(self, chunks):
        """Encode tous les chunks et les insère dans une collection neuve."""
        self.model_name = config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.model_name,
            },
        )

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"source": c["source"], "categorie": c["categorie"]}
            for c in chunks
        ]

        embeddings = self._encode(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    