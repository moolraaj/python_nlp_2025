# # routers/_semantic.py

# import nltk
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from typing import List, Set, Dict, Any
# import logging

# from database import db

 
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

 
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')

# lemmatizer = WordNetLemmatizer()
# model = SentenceTransformer("all-MiniLM-L6-v2" ,device="cpu")

# def get_wordnet_pos(treebank_tag: str) -> str:
#     """Map treebank POS tags to WordNet POS tags"""
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# def encode(text: str) -> np.ndarray:
#     """Encode text into embedding vector"""
#     return model.encode([text], convert_to_numpy=True)[0]

# def top_k_matches(
#     query_emb: np.ndarray,
#     docs: List[Dict[str, Any]],
#     emb_field: str,
#     k: int = 3,
#     threshold: float = 0.5
# ) -> List[Dict[str, Any]]:
#     """Find top k matching documents based on cosine similarity"""
#     valid = [d for d in docs if emb_field in d and isinstance(d[emb_field], (list, np.ndarray))]
#     if not valid:
#         return []
    
#     embs = np.array([d[emb_field] for d in valid])
#     sims = cosine_similarity([query_emb], embs)[0]
#     paired = sorted(zip(sims, valid), key=lambda x: x[0], reverse=True)
    
  
#     logger.debug(f"Top matches before threshold (k={k}, threshold={threshold}):")
#     for sim, doc in paired[:5]:
#         logger.debug(f"  Similarity: {sim:.3f} - Doc: {doc.get('name', doc.get('svg_url', 'Unknown'))}")
    
#     return [doc for sim, doc in paired if sim >= threshold][:k]

# def extract_keywords(text: str, pos_set: Set[str] = {"NOUN", "PROPN", "VERB", "ADJ"}) -> List[str]:
#     """Extract and lemmatize keywords from text"""
#     tokens = word_tokenize(text)
#     tagged = pos_tag(tokens)
    
#     keywords = []
#     for tok, tag in tagged:
#         pos = get_wordnet_pos(tag)
#         lemma = lemmatizer.lemmatize(tok.lower(), pos=pos)
#         universal_tag = nltk.map_tag('en-ptb', 'universal', tag)
        
#         if universal_tag in pos_set:
#             keywords.append(lemma)
    
 
#     logger.debug(f"Extracted keywords: {keywords}")
#     return keywords

# def merge_and_dedupe(
#     sem: List[Dict[str, Any]],
#     kw: List[Dict[str, Any]],
#     key: str,
#     k: int,
#     priority: str = 'semantic'
# ) -> List[Dict[str, Any]]:
#     """Merge semantic and keyword results with deduplication"""
#     merged = []
#     seen = set()
    
    
#     first, second = (sem, kw) if priority == 'semantic' else (kw, sem)
    
#     for d in first + second:
#         val = d.get(key)
#         if val and val not in seen:
#             seen.add(val)
#             merged.append(d)
#         if len(merged) >= k:
#             break
    
#     return merged

# async def find_assets(
#     text: str,
#     k: int = 3,
#     threshold: float = 0.5,
#     keyword_threshold: int = 2
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """Find matching assets with default background fallback ONLY when no backgrounds found"""
    
 
#     DEFAULT_BG = {
#         "_id": "6826f327e51bbad156d7f295",
#         "name": "default",
#         "background_url": "https://res.cloudinary.com/do6qy56kf/image/upload/v1747383965/o_h_app/ivv1nkea3bkykpzqnxle.jpg"
#     }
    
   
#     response = {
#         "backgrounds": [],
#         "gifs": [],
#         "animations": []
#     }

#     if not text.strip():
#         return response

#     try:
#         logger.info(f"Searching for: '{text}'")
        
 
#         bg_docs = [doc async for doc in db["backgrounds"].find()]
#         svg_docs = [doc async for doc in db["svgs"].find()]
#         type_docs = [doc async for doc in db["types"].find()]

     
#         q_emb = encode(text)
#         sem_bg = top_k_matches(q_emb, bg_docs, "embedding", k, threshold)
#         kws = extract_keywords(text)
        
#         if len(kws) >= keyword_threshold:
#             kw_bg = [d for d in bg_docs if any(kw in d["name"].lower() for kw in kws)]
#         else:
#             kw_bg = []
        
#         merged_bg = merge_and_dedupe(sem_bg, kw_bg, "name", k, 'semantic')
        
     
#         if not merged_bg:
#             merged_bg = [DEFAULT_BG]
        
#         response["backgrounds"] = [
#             {"name": d["name"], "background_url": d["background_url"]}
#             for d in merged_bg
#         ]

     
#         sem_sv = top_k_matches(q_emb, svg_docs, "embedding", k, threshold)
#         sem_tp = top_k_matches(q_emb, type_docs, "embedding", k, threshold)
        
#         if len(kws) >= keyword_threshold:
#             kw_sv = [d for d in svg_docs if any(kw in tag.lower() for tag in d.get("tags", []) for kw in kws)]
#             kw_tp = [d for d in type_docs if any(kw in d["name"].lower() for kw in kws)]
#         else:
#             kw_sv, kw_tp = [], []
        
#         response["gifs"] = [
#             {"tags": d.get("tags", []), "svg_url": d["svg_url"]}
#             for d in merge_and_dedupe(sem_sv, kw_sv, "svg_url", k, 'semantic')
#         ]
        
#         response["animations"] = [
#             {"name": d["name"]}
#             for d in merge_and_dedupe(sem_tp, kw_tp, "name", k, 'semantic')
#         ]

#         return response

#     except Exception as fatal_error:
#         logger.critical(f"Unexpected error in find_assets: {str(fatal_error)}")
#         return {
#             "backgrounds": [DEFAULT_BG],  
#             "gifs": [],
#             "animations": []
#         }




import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Set, Dict, Any
import logging

from database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        start = time.time()
        model.encode(["test sentence"])
        print(f"CPU Time Used: {time.time() - start:.2f}s") 
    return model

def encode(text: str) -> np.ndarray:
    """Encode text into embedding vector"""
    return get_model().encode([text], convert_to_numpy=True)[0]

def top_k_matches(
    query_emb: np.ndarray,
    docs: List[Dict[str, Any]],
    emb_field: str,
    k: int = 3,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Find top k matching documents based on cosine similarity"""
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

def merge_and_dedupe(
    sem: List[Dict[str, Any]],
    kw: List[Dict[str, Any]],
    key: str,
    k: int,
    priority: str = 'semantic'
) -> List[Dict[str, Any]]:
    """Merge semantic and keyword results with deduplication"""
    merged = []
    seen = set()
    
    first, second = (sem, kw) if priority == 'semantic' else (kw, sem)
    
    for d in first + second:
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
    threshold: float = 0.5,
    keyword_threshold: int = 2
) -> Dict[str, List[Dict[str, Any]]]:
    """Find matching assets with robust error handling."""
    DEFAULT_BG = {
        "_id": "6826f327e51bbad156d7f295",
        "name": "default",
        "background_url": "https://res.cloudinary.com/do6qy56kf/image/upload/v1747383965/o_h_app/ivv1nkea3bkykpzqnxle.jpg"
    }
    
    empty_response = {"backgrounds": [], "gifs": [], "animations": []}
    
    if not text.strip():
        return empty_response

    try:
        logger.info(f"Searching for: '{text}'")
        
        try:
            bg_docs = [doc async for doc in db["backgrounds"].find()]
            svg_docs = [doc async for doc in db["svgs"].find()]
            type_docs = [doc async for doc in db["types"].find()]
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            return empty_response
     
        try:
            q_emb = encode(text)  
            sem_bg = top_k_matches(q_emb, bg_docs, "embedding", k, threshold)
            sem_sv = top_k_matches(q_emb, svg_docs, "embedding", k, threshold)
            sem_tp = top_k_matches(q_emb, type_docs, "embedding", k, threshold)
        except Exception as model_error:
            logger.error(f"Model encoding error: {str(model_error)}")
            sem_bg, sem_sv, sem_tp = [], [], []

        # Simple keyword matching without NLTK
        try:
            kws = [word.lower() for word in text.split() if len(word) > 2]  # Basic keyword extraction
            logger.debug(f"Keywords: {kws}")

            if len(kws) >= keyword_threshold:
                kw_bg = [d for d in bg_docs if any(kw in d["name"].lower() for kw in kws)]
                kw_sv = [d for d in svg_docs if any(kw in tag.lower() for tag in d.get("tags", []) for kw in kws)]
                kw_tp = [d for d in type_docs if any(kw in d["name"].lower() for kw in kws)]
            else:
                kw_bg, kw_sv, kw_tp = [], [], []
                logger.debug("Skipping keyword matching - not enough keywords")
        except Exception as kw_error:
            logger.error(f"Keyword matching error: {str(kw_error)}")
            kw_bg, kw_sv, kw_tp = [], [], []

        try:
            merged_bg = merge_and_dedupe(sem_bg, kw_bg, "name", k, 'semantic')
            merged_sv = merge_and_dedupe(sem_sv, kw_sv, "svg_url", k, 'semantic')
            merged_tp = merge_and_dedupe(sem_tp, kw_tp, "name", k, 'semantic')

            # Add default background if no backgrounds found
            if not merged_bg:
                merged_bg = [DEFAULT_BG]

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
        except Exception as merge_error:
            logger.error(f"Result merging error: {str(merge_error)}")
            # Return default background even in case of merge error
            return {
                "backgrounds": [
                    {"name": DEFAULT_BG["name"], "background_url": DEFAULT_BG["background_url"]}
                ],
                "gifs": [],
                "animations": []
            }

    except Exception as fatal_error:
        logger.critical(f"Unexpected error in find_assets: {str(fatal_error)}")
        # Return default background even in case of fatal error
        return {
            "backgrounds": [
                {"name": DEFAULT_BG["name"], "background_url": DEFAULT_BG["background_url"]}
            ],
            "gifs": [],
            "animations": []
        }