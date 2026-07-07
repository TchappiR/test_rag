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

    # ==================================================================
    #  ÉTAPE 3.2 — RECHARGER la base (aucun encodage de corpus ici !)
    # ==================================================================
    def _recharger(self):
        """Rouvre une collection existante et recharge le BON modèle."""
        self.collection = self.client.get_collection(self.collection_name)
        self.model_name = self.collection.metadata["embedding_model"]
        self.model = SentenceTransformer(self.model_name)

    # ==================================================================
    #  Méthode d'encodage — utilisée à la création ET à la recherche
    # ==================================================================
    def _encode(self, textes):

        vecteurs = self.model.encode(
            textes,
            batch_size=32,              
            normalize_embeddings=True,  
                                        
            show_progress_bar=True
        )
        # ChromaDB veut des listes Python, pas des tableaux numpy -> .tolist()
        return vecteurs.tolist()

    # ==================================================================
    #  ÉTAPE 3.3 — RECHERCHER les n chunks les plus proches
    # ==================================================================
    def retrieve(self, question, n=3):
        vecteur_question = self._encode([question])
        resultats = self.collection.query(
            query_embeddings=vecteur_question,
            n_results=n,
        )
        ids        = resultats["ids"][0]
        documents  = resultats["documents"][0]
        metadatas  = resultats["metadatas"][0]
        distances  = resultats["distances"][0]
       
        chunks_trouves = []
        for i in range(len(ids)):
            chunks_trouves.append({
                "id": ids[i],
                "text": documents[i],
                "source": metadatas[i]["source"],
                "categorie": metadatas[i]["categorie"],
                "distance": distances[i],
            })
        return chunks_trouves
    