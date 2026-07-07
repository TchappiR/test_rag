# test_RAG

Mini-projet pédagogique (M2 MD5 — Data & IA) : un RAG minimal mais complet,
construit brique par brique avec **ChromaDB**, **sentence-transformers**,
**Groq** et un **agent modérateur**.

La base de connaissances est un corpus de 200 phrases volontairement
absurdes (« Le chat bleu de Bob s'appelle Henri »). Ces faits n'existent
nulle part ailleurs : si le système répond juste, c'est forcément grâce au
*retrieval*, pas à la mémoire du modèle. C'est le corpus de test idéal
pour prouver qu'un RAG fonctionne vraiment.

## Architecture

Le système tient en trois briques, chacune dans son fichier :

| Brique | Fichier | Rôle |
|--------|---------|------|
| 1 — Base vectorielle | `src/vector_db.py` | Crée/recharge une base ChromaDB persistée, encode les chunks, retrouve les plus proches d'une question. |
| 2 — Agent modérateur | `src/moderateur.py` | Demande à un modèle *safeguard* si la question est une injection de prompt ; renvoie `{"is_prompt_injection": bool}`. |
| 3 — RAG orchestrateur | `src/rag.py` | Enchaîne modération → retrieval → prompt à trous → appel du LLM. |

Deux fichiers de configuration importants les accompagnent : `src/config.py`
(noms de modèles et chemins, centralisés) et le dossier `prompts/` (les
prompts système, éditables sans toucher au code).

Le pipeline de `answer_question()` : **on modère d'abord**. Si le
modérateur détecte une injection, le LLM principal n'est jamais appelé.
Sinon, on récupère les 3 chunks les plus proches, on remplit le marqueur
`{{Chunks}}` du prompt système, puis on appelle le LLM.

## Arborescence

```
.
├── data/
│   ├── 05_corpus_rag.csv      # le corpus (200 chunks)
│   └── chroma_db/             # base persistée (générée, non versionnée)
├── prompts/
│   ├── moderateur_prompt.txt  # politique de détection d'injection
│   └── rag_prompt.txt         # prompt système à trous ({{Chunks}})
├── src/
│   ├── __init__.py
│   ├── config.py              # constantes (modèles, chemins)
│   ├── vector_db.py           # Brique 1
│   ├── moderateur.py          # Brique 2
│   ├── rag.py                 # Brique 3
│   ├── index_corpus.py        # script : construit la base + teste le retrieve
│   └── demo_rag.py            # script : joue les 4 tests de mise à l'épreuve
├── .env                       # GROQ_API_KEY (jamais commité)
├── pyproject.toml
└── README.md
```

## Prérequis

- Python ≥ 3.10 et [Poetry](https://python-poetry.org/)
- Une clé API Groq (gratuite sur [console.groq.com](https://console.groq.com))

## Installation

```bash
poetry add sentence-transformers chromadb groq python-dotenv
```

Créer un fichier `.env` à la racine (il est déjà dans `.gitignore`) :

```
GROQ_API_KEY=gsk_ta_cle_ici
```

## Configuration des modèles

Tout est centralisé dans `src/config.py` :

```python
EMBEDDING_MODEL  = "distiluse-base-multilingual-cased-v2"  # embeddings (Brique 1)
LLM_MODEL        = "openai/gpt-oss-120b"                    # génération (Brique 3)
MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"           # modération (Brique 2)
```

> **Note.** Le TP d'origine citait `llama-3.3-70b-versatile` (génération) et
> `meta-llama/llama-guard-4-12b` (modération). Le catalogue Groq a évolué :
> le modèle de modération a été **coupé le 05/03/2026** et celui de
> génération le sera le **16/08/2026**. On utilise donc leurs remplaçants
> officiels. Si un ID de modèle change encore, il suffit de modifier ces
> lignes — c'est justement l'intérêt d'un module de constantes.

## Utilisation

Les scripts se lancent **en tant que modules**, depuis la racine du projet
(les imports sont relatifs au package `src`) :

```bash
# 1) Construire la base vectorielle + tester le retrieve (Brique 1)
poetry run python -m src.index_corpus

# 2) Tester le pipeline complet (Briques 2 et 3)
poetry run python -m src.demo_rag
```

Le premier lancement de `index_corpus` télécharge le modèle d'embedding et
encode les 200 phrases (un peu long). Les lancements suivants **rechargent**
la base instantanément : c'est la preuve que la persistance fonctionne.

## Mise à l'épreuve

`src/demo_rag.py` rejoue les 4 tests du TP :

1. **Injection** (« Oublie ton contexte… ») → bloquée par le modérateur, avant le LLM.
2. **Question dans le corpus** (couleur du chat de Bob) → réponse correcte (Henri, bleu).
3. **Question hors corpus** (capitale du Japon) → le système répond qu'il ne sait pas.
4. **Affirmation fausse** (« le chat de Bob est vert ») → le système signale la contradiction.

## Pour aller plus loin

- Afficher les sources et les scores de similarité dans la réponse.
- Ajouter un seuil : avertir si le meilleur chunk est trop éloigné.
- Remplacer le corpus jouet par un vrai document découpé en chunks.
- Comparer deux modèles d'embedding multilingues sur les mêmes questions.
