# routers/_semantic.py

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from database import db   

 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')

 
model = SentenceTransformer("all-MiniLM-L6-v2")

def encode(text: str) -> np.ndarray:
    """
    Compute and return a 1D numpy embedding for `text`.
    """
    return model.encode([text], convert_to_numpy=True)[0]

def top_k_matches(
    query_emb: np.ndarray,
    docs: list[dict],
    emb_field: str,
    k: int = 3,
    threshold: float = 0.5
) -> list[dict]:
    """
    Return up to k docs whose `emb_field` vectors have cosine
    similarity ≥ threshold with query_emb, sorted by similarity.
    """
    valid = [
        d for d in docs
        if emb_field in d and isinstance(d[emb_field], (list, tuple))
    ]
    if not valid:
        return []
    embs = np.array([d[emb_field] for d in valid])
    sims = cosine_similarity([query_emb], embs)[0]
    paired = sorted(zip(sims, valid), key=lambda x: x[0], reverse=True)
    return [doc for sim, doc in paired if sim >= threshold][:k]

def extract_keywords(
    text: str,
    pos_set: set = {"NOUN", "PROPN", "VERB", "PRON"}
) -> list[str]:
    """
    Tokenize and POS-tag `text`, return tokens whose
    universal POS tag is in pos_set (e.g. NOUN, PROPN, VERB, PRON).
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens, tagset="universal")
    return [tok for tok, tag in tagged if tag in pos_set]

def split_sentences(text: str) -> list[str]:
    """
    Use NLTK’s Punkt to split text into sentences.
    """
    norm = text.replace("\n", " ")
    return sent_tokenize(norm)

async def find_assets(
    text: str,
    k: int = 3,
    threshold: float = 0.5
) -> dict:
    """
    Load all collections and return up to k matches per category,
    first via semantic embedding, then fallback to NLTK-keyword filtering.
    """
   
    bg_docs   = [doc async for doc in db["backgrounds"].find()]
    svg_docs  = [doc async for doc in db["svgs"].find()]
    type_docs = [doc async for doc in db["types"].find()]

   
    q_emb  = encode(text)
    sem_bg = top_k_matches(q_emb, bg_docs,   "embedding", k, threshold)
    sem_sv = top_k_matches(q_emb, svg_docs,  "embedding", k, threshold)
    sem_tp = top_k_matches(q_emb, type_docs, "embedding", k, threshold)

  
    kws = [kw.lower() for kw in extract_keywords(text)]
    if not sem_bg:
        sem_bg = [d for d in bg_docs   if any(kw in d["name"].lower()      for kw in kws)][:k]
    if not sem_sv:
        sem_sv = [d for d in svg_docs  if any(kw == tag.lower()             for tag in d.get("tags",[]) for kw in kws)][:k]
    if not sem_tp:
        sem_tp = [d for d in type_docs if any(kw in d["name"].lower()      for kw in kws)][:k]

    return {
        "backgrounds": [
            {"name": d["name"], "background_url": d["background_url"], "type": d.get("type")}
            for d in sem_bg
        ],
        "gifs": [
            {"tags": d["tags"], "svg_url": d["svg_url"], "type": d.get("type")}
            for d in sem_sv
        ],
        "animations": [
            {"name": d["name"], "type": d.get("type")}
            for d in sem_tp
        ]
    }
