import sys

import chromadb
from sentence_transformers import SentenceTransformer

from src import config
from src.chunking import chunk_records
from src.loaders import load_directory, load_file

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def embed_texts(texts):
    vectors = get_embedder().encode(texts, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()


def get_client():
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def _add_chunks(collection, chunks, batch_size=128):
    for i in range(0, len(chunks), batch_size):
        part = chunks[i:i + batch_size]
        texts = [c["text"] for c in part]
        collection.upsert(
            ids=[c["id"] for c in part],
            documents=texts,
            embeddings=embed_texts(texts),
            metadatas=[c["metadata"] for c in part],
        )
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")


def build_index(config_name):
    """Build a fresh collection from everything in data/."""
    print(f"Building collection '{config_name}' ...")
    chunks = chunk_records(load_directory(config.DATA_DIR), config_name)
    client = get_client()
    try:
        client.delete_collection(config_name)
    except Exception:
        pass  # didn't exist yet
    collection = client.create_collection(config_name, metadata={"hnsw:space": "cosine"})
    _add_chunks(collection, chunks)
    print(f"Done: {len(chunks)} chunks in '{config_name}'.")


def add_file_to_index(path, collection_name):
    """Add (or replace) a single file. Used by the /upload endpoint in Step 12."""
    chunks = chunk_records(load_file(path), collection_name)
    collection = get_client().get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    if chunks:
        collection.delete(where={"source_file": chunks[0]["metadata"]["source_file"]})
        _add_chunks(collection, chunks)
    return len(chunks)


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["fixed_500_50"]):
        build_index(name)