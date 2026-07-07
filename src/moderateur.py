import json
from .config import MODERATION_MODEL, PROMPTS_DIR


class Moderateur:
    def __init__(self, client):
        # On REÇOIT le client Groq déjà créé par le RAG, au lieu d'en
        # ouvrir un deuxième : un seul client pour tout le programme.
        self.client = client
        self.model = MODERATION_MODEL

        # Le prompt système (la "politique") vit dans un fichier texte,
        # pas dans le code : on peut le retravailler sans toucher au .py.
        chemin_prompt = PROMPTS_DIR / "moderateur_prompt.txt"
        with open(chemin_prompt, encoding="utf-8") as f:
            self.system_prompt = f.read()

    # ------------------------------------------------------------------
    def moderate(self, question):
        """Renvoie {"is_prompt_injection": bool} pour la question donnée."""

        # Appel classique "chat completions" :
        #   - message system = notre politique de détection
        #   - message user   = la question brute à classer
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
        )
        contenu = completion.choices[0].message.content or ""

        # Le modèle est censé répondre du JSON. On le transforme en dict
        # Python de façon ROBUSTE (voir _extraire_json).
        decision = self._extraire_json(contenu)

        # On ne garde que le champ qui nous intéresse, et on le force en
        # booléen propre. .get(...) évite un plantage si la clé manque.
        return {
            "is_prompt_injection": bool(decision.get("is_prompt_injection", False))
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extraire_json(texte):
        """
        Transforme la réponse du modèle en dictionnaire Python.

        On ne fait pas un json.loads(texte) direct, car un modèle peut
        entourer le JSON de texte ("Voici ma réponse : {...}"). On isole
        donc la portion entre la 1re accolade ouvrante et la dernière
        fermante, puis on la parse. En cas d'échec, on renvoie {} : le
        RAG traitera alors la question comme NON-injection (choix de ne
        pas bloquer par erreur une question légitime — un compromis à
        discuter selon qu'on préfère la sécurité ou l'expérience).
        """
        debut = texte.find("{")
        fin = texte.rfind("}")
        if debut == -1 or fin == -1:
            return {}
        try:
            return json.loads(texte[debut:fin + 1])
        except json.JSONDecodeError:
            return {}
