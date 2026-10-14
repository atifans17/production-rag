import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src import config
from src.index import add_file_to_index
from src.loaders import SUPPORTED
from src.pipeline import RAGPipeline

app = FastAPI(title="Production RAG API")
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok", "collection": pipeline.collection}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    r = pipeline.ask(req.question, req.filters)
    return {
        "answer": r["answer"],
        "refused": r["refused"],
        "citations": r["citations"],
        "latency_ms": r["latency_ms"],
        "cost_usd": r["cost_usd"],
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "cached": r["cached"],
    }


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    safe_name = Path(file.filename).name  # strips any folder parts (path-traversal protection)
    if Path(safe_name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported type. Use: {sorted(SUPPORTED)}")
    dest = Path(config.DATA_DIR) / safe_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks_added = add_file_to_index(str(dest), pipeline.collection)
    pipeline.refresh()  # rebuild BM25 + clear the cache so new content is searchable
    return {"filename": safe_name, "chunks_added": chunks_added}