# routers/_semantic.py
 
import nltk
 
from nltk.tokenize import word_tokenize
 
from nltk import pos_tag
 
from nltk.stem import WordNetLemmatizer
 
from nltk.corpus import wordnet
 
from sentence_transformers import SentenceTransformer
 
from sklearn.metrics.pairwise import cosine_similarity
 
import numpy as np
 
from typing import List, Set, Dict, Any
 
import logging
 
import asyncio
 
import concurrent.futures
 
from database import db
 
logging.basicConfig(level=logging.INFO)
 
logger = logging.getLogger(__name__)
 
# Create thread pool for CPU tasks
 
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
 
# Download NLTK data
 
try:
 
    nltk.data.find('tokenizers/punkt')
 
except LookupError:
 
    nltk.download('punkt', quiet=True)
 
try:
 
    nltk.data.find('taggers/averaged_perceptron_tagger')
 
except LookupError:
 
    nltk.download('averaged_perceptron_tagger', quiet=True)
 
try:
 
    nltk.data.find('corpora/wordnet')
 
except LookupError:
 
    nltk.download('wordnet', quiet=True)
 
lemmatizer = WordNetLemmatizer()
 
# Initialize model
 
try:
 
    import torch
 
    if torch.backends.mps.is_available():
 
        device = "mps"
 
        logger.info("🚀 Using MPS (Apple Silicon GPU)")
 
    else:
 
        device = "cpu"
 
        logger.info("⚡ Using CPU")
 
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
 
except Exception as e:
 
    logger.warning(f"Failed to initialize model with GPU: {e}")
 
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
 
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
 
def encode_sync(text: str) -> np.ndarray:
 
    try:
 
        return model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
 
    except Exception as e:
 
        logger.error(f"Encoding failed: {e}")
 
        return np.zeros(384)
 
async def encode_async(text: str) -> np.ndarray:
 
    loop = asyncio.get_event_loop()
 
    try:
 
        return await asyncio.wait_for(
 
            loop.run_in_executor(thread_pool, encode_sync, text),
 
            timeout=10.0
 
        )
 
    except asyncio.TimeoutError:
 
        logger.error("Encoding timeout")
 
        return np.zeros(384)
 
    except Exception as e:
 
        logger.error(f"Encoding async failed: {e}")
 
        return np.zeros(384)
 
def extract_keywords_sync(text: str) -> List[str]:
 
    if not text or not text.strip():
 
        return []
 
    try:
 
        tokens = word_tokenize(text)
 
        tagged = pos_tag(tokens)
 
        keywords = []
 
        for tok, tag in tagged:
 
            pos = get_wordnet_pos(tag)
 
            lemma = lemmatizer.lemmatize(tok.lower(), pos=pos)
 
            universal_tag = nltk.map_tag('en-ptb', 'universal', tag)
 
            if universal_tag in {"NOUN", "PROPN", "VERB", "ADJ"} and len(lemma) > 2:
 
                keywords.append(lemma)
 
        seen = set()
 
        unique_keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]
 
        return unique_keywords
 
    except Exception as e:
 
        logger.error(f"Keyword extraction sync failed: {e}")
 
        return []
 
async def extract_keywords_async(text: str) -> List[str]:
 
    loop = asyncio.get_event_loop()
 
    try:
 
        return await asyncio.wait_for(
 
            loop.run_in_executor(thread_pool, extract_keywords_sync, text),
 
            timeout=5.0
 
        )
 
    except asyncio.TimeoutError:
 
        logger.error("Keyword extraction timeout")
 
        return []
 
    except Exception as e:
 
        logger.error(f"Keyword extraction async failed: {e}")
 
        return []
 
def top_k_matches_sync(
 
    query_emb: np.ndarray,
 
    docs: List[Dict[str, Any]],
 
    emb_field: str,
 
    k: int = 3,
 
    threshold: float = 0.5
 
) -> List[Dict[str, Any]]:
 
    try:
 
        valid = [d for d in docs if emb_field in d and d[emb_field] is not None and len(d[emb_field]) > 0]
 
        if not valid:
 
            return []
 
        embs = np.array([d[emb_field] for d in valid])
 
        sims = cosine_similarity([query_emb], embs)[0]
 
        paired = sorted(zip(sims, valid), key=lambda x: x[0], reverse=True)
 
        return [doc for sim, doc in paired if sim >= threshold][:k]
 
    except Exception as e:
 
        logger.error(f"Similarity matching sync failed: {e}")
 
        return []
 
async def top_k_matches_async(
 
    query_emb: np.ndarray,
 
    docs: List[Dict[str, Any]],
 
    emb_field: str,
 
    k: int = 3,
 
    threshold: float = 0.5
 
) -> List[Dict[str, Any]]:
 
    loop = asyncio.get_event_loop()
 
    try:
 
        return await asyncio.wait_for(
 
            loop.run_in_executor(thread_pool, top_k_matches_sync, query_emb, docs, emb_field, k, threshold),
 
            timeout=5.0
 
        )
 
    except asyncio.TimeoutError:
 
        logger.error("Similarity matching timeout")
 
        return []
 
    except Exception as e:
 
        logger.error(f"Similarity matching async failed: {e}")
 
        return []
 
def merge_and_dedupe(
 
    sem: List[Dict[str, Any]],
 
    kw: List[Dict[str, Any]],
 
    key: str,
 
    k: int,
 
    priority: str = 'semantic'
 
) -> List[Dict[str, Any]]:
 
    merged = []
 
    seen = set()
 
    first, second = (sem, kw) if priority == 'semantic' else (kw, sem)
 
    for d in first + second:
 
        if len(merged) >= k:
 
            break
 
        val = d.get(key)
 
        if val and val not in seen:
 
            seen.add(val)
 
            merged.append(d)
 
    return merged
 
async def find_assets(
 
    text: str,
 
    k: int = 3,
 
    threshold: float = 0.5,
 
    keyword_threshold: int = 2
 
) -> Dict[str, List[Dict[str, Any]]]:
 
    DEFAULT_BG = {
 
        "_id": "6826f327e51bbad156d7f295",
 
        "name": "default",
 
        "background_url": "https://res.cloudinary.com/do6qy56kf/image/upload/v1747383965/o_h_app/ivv1nkea3bkykpzqnxle.jpg"
 
    }
 
    response = {
 
        "backgrounds": [],
 
        "gifs": [],
 
        "animations": []
 
    }
 
    if not text.strip():
 
        return response
 
    try:
 
        logger.info(f"🔍 Starting search for: '{text}'")
 
        async def fetch_with_timeout(collection, timeout=10):
 
            try:
 
                return await asyncio.wait_for(collection.find().to_list(length=None), timeout=timeout)
 
            except asyncio.TimeoutError:
 
                logger.error(f"Timeout fetching from {collection.name}")
 
                return []
 
            except Exception as e:
 
                logger.error(f"Error fetching from {collection.name}: {e}")
 
                return []
 
        bg_docs, svg_docs, type_docs = await asyncio.gather(
 
            fetch_with_timeout(db["backgrounds"]),
 
            fetch_with_timeout(db["svgs"]),
 
            fetch_with_timeout(db["types"])
 
        )
 
        logger.info(f"📊 Fetched {len(bg_docs)} backgrounds, {len(svg_docs)} SVGs, {len(type_docs)} types")
 
        encode_task = encode_async(text)
 
        keywords_task = extract_keywords_async(text)
 
        q_emb, kws = await asyncio.gather(encode_task, keywords_task)
 
        logger.info(f"🔑 Extracted {len(kws)} keywords: {kws}")
 
        sem_bg_task = top_k_matches_async(q_emb, bg_docs, "embedding", k, threshold)
 
        sem_sv_task = top_k_matches_async(q_emb, svg_docs, "embedding", k, threshold)
 
        sem_tp_task = top_k_matches_async(q_emb, type_docs, "embedding", k, threshold)
 
        sem_bg, sem_sv, sem_tp = await asyncio.gather(sem_bg_task, sem_sv_task, sem_tp_task)
 
        if len(kws) >= keyword_threshold:
 
            kw_bg = [d for d in bg_docs if any(kw in d.get("name", "").lower() for kw in kws)]
 
            kw_sv = [d for d in svg_docs if any(
 
                kw in tag.lower() for tag in d.get("tags", []) for kw in kws
 
            )]
 
            kw_tp = [d for d in type_docs if any(kw in d.get("name", "").lower() for kw in kws)]
 
        else:
 
            kw_bg, kw_sv, kw_tp = [], [], []
 
        merged_bg = merge_and_dedupe(sem_bg, kw_bg, "name", k, 'semantic')
 
        merged_sv = merge_and_dedupe(sem_sv, kw_sv, "svg_url", k, 'semantic')
 
        merged_tp = merge_and_dedupe(sem_tp, kw_tp, "name", k, 'semantic')
 
        if not merged_bg:
 
            logger.info("Using default background")
 
            merged_bg = [DEFAULT_BG]
 
        response["backgrounds"] = [
 
            {"name": d["name"], "background_url": d["background_url"]}
 
            for d in merged_bg
 
        ]
 
        response["gifs"] = [
 
            {"tags": d.get("tags", []), "svg_url": d["svg_url"]}
 
            for d in merged_sv
 
        ]
 
        response["animations"] = [
 
            {"name": d["name"]}
 
            for d in merged_tp
 
        ]
 
        logger.info(f"✅ Search completed - BG: {len(merged_bg)}, SVG: {len(merged_sv)}, Type: {len(merged_tp)}")
 
        return response
 
    except Exception as fatal_error:
 
        logger.critical(f"💥 Unexpected error in find_assets: {str(fatal_error)}", exc_info=True)
 
        return {
 
            "backgrounds": [{"name": DEFAULT_BG["name"], "background_url": DEFAULT_BG["background_url"]}],
 
            "gifs": [],
 
            "animations": []
 
        }
 