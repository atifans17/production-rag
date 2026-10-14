import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src import config
from src.generate import REFUSAL, generate_answer
from src.retrieve import refresh_bm25, retrieve


class RAGPipeline:
    def __init__(self, collection=None, mode="hybrid_rerank", top_k=5):
        self.collection = collection or config.DEFAULT_COLLECTION
        self.mode = mode
        self.top_k = top_k
        self.cache = {}
        Path(config.LOG_DIR).mkdir(exist_ok=True)

    # ---------- helpers ----------
    @staticmethod
    def _ms(start):
        return int((time.perf_counter() - start) * 1000)

    def _key(self, question, filters):
        normalized = " ".join(question.lower().split())
        raw = json.dumps(
            {"q": normalized, "f": filters, "c": self.collection, "m": self.mode},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _describe(i, chunk):
        m = chunk["metadata"]
        return {
            "doc": f"Doc {i}",
            "source_file": m["source_file"],
            "section": m["section"],
            "text": chunk["text"],
            "score": chunk.get("rerank_score"),
        }

    @staticmethod
    def _citations(answer, retrieved):
        numbers = sorted({int(n) for n in re.findall(r"Doc\s*(\d+)", answer)})
        return [retrieved[n - 1] for n in numbers if 1 <= n <= len(retrieved)]

    def _log(self, question, result):
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "cached": result["cached"],
            "refused": result["refused"],
            "latency_ms": result["latency_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "cost_usd": result["cost_usd"],
        }
        with open(Path(config.LOG_DIR) / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def refresh(self):
        """Call after new documents are indexed."""
        self.cache.clear()
        refresh_bm25(self.collection)

    # ---------- main entry point ----------
    def ask(self, question, filters=None, use_cache=True):
        start = time.perf_counter()
        key = self._key(question, filters)

        # 1) Cache hit: skip embedding, retrieval and the LLM completely.
        if use_cache and key in self.cache:
            result = dict(self.cache[key])
            result.update(cached=True, cost_usd=0.0, prompt_tokens=0,
                          completion_tokens=0, latency_ms=self._ms(start))
            self._log(question, result)
            return result

        # 2) Retrieve (and re-rank).
        chunks = retrieve(question, self.collection, mode=self.mode,
                          top_k=self.top_k, filters=filters)
        retrieved = [self._describe(i, c) for i, c in enumerate(chunks, start=1)]

        # 3) Retrieval gate: best chunk not relevant enough, so refuse without calling the LLM.
        too_weak = (not chunks) or (
            self.mode == "hybrid_rerank"
            and chunks[0]["rerank_score"] < config.RELEVANCE_THRESHOLD
        )
        if too_weak:
            result = {"answer": REFUSAL, "refused": True, "citations": [],
                      "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        else:
            # 4) Generate with the strict prompt.
            gen = generate_answer(question, chunks)
            refused = REFUSAL.lower() in gen["answer"].lower()
            result = {
                "answer": gen["answer"],
                "refused": refused,
                "citations": [] if refused else self._citations(gen["answer"], retrieved),
                "prompt_tokens": gen["prompt_tokens"],
                "completion_tokens": gen["completion_tokens"],
                "cost_usd": gen["cost_usd"],
            }

        result.update(retrieved=retrieved, cached=False)
        self.cache[key] = dict(result)
        result["latency_ms"] = self._ms(start)
        self._log(question, result)
        return result