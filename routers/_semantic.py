# routers/_semantic.py

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Set

from database import db

# one-time downloads; remove or move to startup if you prefer
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode(text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True)[0]

def top_k_matches(
    query_emb: np.ndarray,
    docs: List[dict],
    emb_field: str,
    k: int = 3,
    threshold: float = 0.5
) -> List[dict]:
    valid = [d for d in docs if emb_field in d and isinstance(d[emb_field], (list, tuple))]
    if not valid:
        return []
    embs = np.array([d[emb_field] for d in valid])
    sims = cosine_similarity([query_emb], embs)[0]
    paired = sorted(zip(sims, valid), key=lambda x: x[0], reverse=True)
    return [doc for sim, doc in paired if sim >= threshold][:k]

def extract_keywords(text: str, pos_set: Set[str] = {"NOUN", "PROPN", "VERB"}) -> List[str]:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens, tagset="universal")
    return [tok for tok, tag in tagged if tag in pos_set]

def merge_and_dedupe(sem: List[dict], kw: List[dict], key: str, k: int) -> List[dict]:
    merged = []
    seen = set()
    for d in sem + kw:
        val = d.get(key)
        if val and val not in seen:
            seen.add(val)
            merged.append(d)
        if len(merged) >= k:
            break
    return merged

async def find_assets(
    text: str,
    k: int = 3,
    threshold: float = 0.5
) -> dict:
    # load all docs once per call
    bg_docs   = [doc async for doc in db["backgrounds"].find()]
    svg_docs  = [doc async for doc in db["svgs"].find()]
    type_docs = [doc async for doc in db["types"].find()]

    # semantic search
    q_emb   = encode(text)
    sem_bg  = top_k_matches(q_emb, bg_docs,   "embedding", k, threshold)
    sem_sv  = top_k_matches(q_emb, svg_docs,  "embedding", k, threshold)
    sem_tp  = top_k_matches(q_emb, type_docs, "embedding", k, threshold)

    # keyword fallback (always computed)
    kws = [w.lower() for w in extract_keywords(text)]
    kw_bg = [d for d in bg_docs   if any(kw in d["name"].lower()      for kw in kws)]
    kw_sv = [d for d in svg_docs  if any(kw == tag.lower()             for tag in d.get("tags", []) for kw in kws)]
    kw_tp = [d for d in type_docs if any(kw in d["name"].lower()      for kw in kws)]

    # merge + dedupe per category
    merged_bg = merge_and_dedupe(sem_bg, kw_bg, "name", k)
    merged_sv = merge_and_dedupe(sem_sv, kw_sv, "svg_url", k)
    merged_tp = merge_and_dedupe(sem_tp, kw_tp, "name", k)

    return {
        "backgrounds": [
            {"name": d["name"], "background_url": d["background_url"]}
            for d in merged_bg
        ],
        "gifs": [
            {"tags": d.get("tags", []), "svg_url": d["svg_url"]}
            for d in merged_sv
        ],
        "animations": [
            {"name": d["name"]}
            for d in merged_tp
        ],
    }
