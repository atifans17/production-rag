from src.index import embed_texts, get_client
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src import config

def _chroma_where(filters):
    """Turn {'file_type': 'pdf', 'date_from': 20260101} into a Chroma filter."""
    if not filters or not isinstance(filters, dict):
        return None
        
    conds = []
    for key, val in filters.items():
        if val is None or val == "":
            continue
        if key == "date_from":
            conds.append({"date_added": {"$gte": val}})
        elif key == "date_to":
            conds.append({"date_added": {"$lte": val}})
        else:
            conds.append({key: val})

    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def dense_search(collection, query, k=5, filters=None):
    col = get_client().get_collection(collection)
    
    # 1. Clean the where clause
    where_clause = _chroma_where(filters)
    
    # 2. Build kwargs dynamically
    query_args = {
        "query_embeddings": embed_texts([query]),
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    
    # CRITICAL: ONLY add the 'where' key if it has actual conditions.
    # Never pass where=None or where={} into col.query()
    if where_clause and isinstance(where_clause, dict) and len(where_clause) > 0:
        query_args["where"] = where_clause

    res = col.query(**query_args)
    ...


_bm25_cache = {}

def tokenize(text):
    return re.findall(r"\w+", text.lower())


def _matches(meta, filters):
    """Same filter logic as _chroma_where, but applied in Python for BM25."""
    if not filters:
        return True
    for key, val in filters.items():
        if key == "date_from":
            if meta.get("date_added", 0) < val:
                return False
        elif key == "date_to":
            if meta.get("date_added", 0) > val:
                return False
        elif meta.get(key) != val:
            return False
    return True


def refresh_bm25(collection):
    """(Re)build the BM25 index from everything in the collection."""
    data = get_client().get_collection(collection).get(include=["documents", "metadatas"])
    _bm25_cache[collection] = {
        "bm25": BM25Okapi([tokenize(d) for d in data["documents"]]),
        "ids": data["ids"],
        "docs": data["documents"],
        "metas": data["metadatas"],
    }


def bm25_search(collection, query, k=15, filters=None):
    if collection not in _bm25_cache:
        refresh_bm25(collection)
    idx = _bm25_cache[collection]
    scores = idx["bm25"].get_scores(tokenize(query))
    results = []
    for i in np.argsort(scores)[::-1]:
        if scores[i] <= 0:
            break  # no more keyword matches
        if not _matches(idx["metas"][i], filters):
            continue
        results.append({
            "id": idx["ids"][i],
            "text": idx["docs"][i],
            "metadata": idx["metas"][i],
            "bm25_score": float(scores[i]),
        })
        if len(results) == k:
            break
    return results


def rrf_fuse(*ranked_lists, k=60):
    """Reciprocal Rank Fusion: merge several ranked lists into one."""
    fused = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            entry = fused.setdefault(item["id"], {**item, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (k + rank + 1)
            entry.update({key: v for key, v in item.items() if key.endswith("_score") and key != "rrf_score"})
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def rerank(query, candidates, top_n=5):
    if not candidates:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def retrieve(query, collection, mode="hybrid_rerank", top_k=5, filters=None, candidates=15):
    if mode == "dense":
        return dense_search(collection, query, top_k, filters)
    if mode == "bm25":
        return bm25_search(collection, query, top_k, filters)

    dense = dense_search(collection, query, candidates, filters)
    sparse = bm25_search(collection, query, candidates, filters)
    fused = rrf_fuse(dense, sparse)

    if mode == "hybrid":
        return fused[:top_k]
    if mode == "hybrid_rerank":
        return rerank(query, fused, top_n=top_k)
    raise ValueError(f"Unknown mode: {mode}")