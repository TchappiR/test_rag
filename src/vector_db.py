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

    