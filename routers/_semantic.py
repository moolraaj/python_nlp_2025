import logging
import numpy as np
from typing import List, Set, Dict, Any

from nltk import download, map_tag
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from database import db

logger = logging.getLogger(__name__)

# —————————————————————————————————————————————————————————————————————————————
# 1) Prep functions for Startup
# —————————————————————————————————————————————————————————————————————————————

def prep_nltk():
    """Download all NLTK corpora once at startup."""
    download('punkt')
    download('averaged_perceptron_tagger')
    download('wordnet')
    download('omw-1.4')
    # you had 'punkt_tab' and 'averaged_perceptron_tagger_eng' but those
    # aren't valid package names, so we drop them
    logger.info("✅ NLTK data ready")

# —————————————————————————————————————————————————————————————————————————————
# 2) Lazy model + lemmatizer
# —————————————————————————————————————————————————————————————————————————————

_lemmatizer: WordNetLemmatizer = None
_model: SentenceTransformer = None

def get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# —————————————————————————————————————————————————————————————————————————————
# 3) Your existing helper functions, updated only to call get_model()/get_lemmatizer()
# —————————————————————————————————————————————————————————————————————————————

def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def encode(text: str) -> np.ndarray:
    """Encode text into embedding vector (lazy-loads the model on first call)."""
    model = get_model()
    return model.encode([text], convert_to_numpy=True)[0]

def top_k_matches(
    query_emb: np.ndarray,
    docs: List[Dict[str, Any]],
    emb_field: str,
    k: int = 3,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    valid = [d for d in docs if emb_field in d and isinstance(d[emb_field], (list, np.ndarray))]
    if not valid:
        return []
    embs = np.array([d[emb_field] for d in valid])
    sims = cosine_similarity([query_emb], embs)[0]
    paired = sorted(zip(sims, valid), key=lambda x: x[0], reverse=True)
    logger.debug(f"Top matches before threshold (k={k}, threshold={threshold}):")
    for sim, doc in paired[:5]:
        logger.debug(f"  Similarity: {sim:.3f} - Doc: {doc.get('name', doc.get('svg_url', 'Unknown'))}")
    return [doc for sim, doc in paired if sim >= threshold][:k]

def extract_keywords(text: str, pos_set: Set[str] = {"NOUN", "PROPN", "VERB", "ADJ"}) -> List[str]:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    keywords = []
    for tok, tag in tagged:
        pos = get_wordnet_pos(tag)
        lemma = get_lemmatizer().lemmatize(tok.lower(), pos=pos)
        universal = map_tag('en-ptb', 'universal', tag)
        if universal in pos_set:
            keywords.append(lemma)
    logger.debug(f"Extracted keywords: {keywords}")
    return keywords

def merge_and_dedupe(
    sem: List[Dict[str, Any]],
    kw: List[Dict[str, Any]],
    key: str,
    k: int,
    priority: str = 'semantic'
) -> List[Dict[str, Any]]:
    merged, seen = [], set()
    first, second = (sem, kw) if priority=='semantic' else (kw, sem)
    for d in first+second:
        val = d.get(key)
        if val and val not in seen:
            seen.add(val)
            merged.append(d)
        if len(merged)>=k:
            break
    return merged

# —————————————————————————————————————————————————————————————————————————————
# 4) The actual search function
# —————————————————————————————————————————————————————————————————————————————

async def find_assets(
    text: str,
    k: int = 3,
    threshold: float = 0.5,
    keyword_threshold: int = 2
) -> Dict[str, List[Dict[str, Any]]]:
    if not text.strip():
        return {"backgrounds": [], "gifs": [], "animations": []}
    logger.info(f"Searching for: '{text}'")
    bg_docs = [doc async for doc in db["backgrounds"].find()]
    svg_docs = [doc async for doc in db["svgs"].find()]
    type_docs= [doc async for doc in db["types"].find()]

    q_emb = encode(text)
    sem_bg= top_k_matches(q_emb, bg_docs, "embedding", k, threshold)
    sem_sv= top_k_matches(q_emb, svg_docs, "embedding", k, threshold)
    sem_tp= top_k_matches(q_emb, type_docs,"embedding", k, threshold)

    kws = extract_keywords(text)
    logger.debug(f"Final keywords used for matching: {kws}")

    if len(kws)>=keyword_threshold:
        kw_bg = [d for d in bg_docs  if any(kw in d["name"].lower() for kw in kws)]
        kw_sv = [d for d in svg_docs if any(kw in tag.lower() for tag in d.get("tags",[]) for kw in kws)]
        kw_tp = [d for d in type_docs if any(kw in d["name"].lower() for kw in kws)]
    else:
        kw_bg, kw_sv, kw_tp = [], [], []
        logger.debug("Skipping keyword matching - not enough keywords")

    merged_bg = merge_and_dedupe(sem_bg, kw_bg,  "name", k)
    merged_sv = merge_and_dedupe(sem_sv, kw_sv,  "svg_url", k)
    merged_tp = merge_and_dedupe(sem_tp, kw_tp,  "name", k)

    return {
        "backgrounds":[{"name":d["name"],"background_url":d["background_url"]} for d in merged_bg],
        "gifs":[{"tags":d.get("tags",[]),"svg_url":d["svg_url"]}               for d in merged_sv],
        "animations":[{"name":d["name"]}                                       for d in merged_tp],
    }
