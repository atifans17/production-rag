from langchain_text_splitters import RecursiveCharacterTextSplitter

# Each config becomes its own Chroma collection, so you can compare them fairly.
CHUNK_CONFIGS = {
    "fixed_200_20":  {"size": 200, "overlap": 20, "structure": False},
    "fixed_500_0":   {"size": 500, "overlap": 0,  "structure": False},
    "fixed_500_50":  {"size": 500, "overlap": 50, "structure": False},
    "struct_500_50": {"size": 500, "overlap": 50, "structure": True},
}


def make_splitter(cfg):
    kwargs = {}
    if cfg["structure"]:
        # Prefer breaking at paragraph, then line, then sentence boundaries.
        kwargs["separators"] = ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=cfg["size"],
        chunk_overlap=cfg["overlap"],
        **kwargs,
    )


def chunk_records(records, config_name):
    cfg = CHUNK_CONFIGS[config_name]
    splitter = make_splitter(cfg)
    counters = {}  # per-file counter keeps chunk IDs stable and unique
    chunks = []
    for rec in records:
        meta = rec["metadata"]
        for piece in splitter.split_text(rec["text"]):
            text = piece
            if cfg["structure"]:
                # "Contextual header": tells the embedding which doc/section this came from.
                text = f"{meta['source_file']} | {meta['section']}\n{piece}"
            n = counters.get(meta["source_file"], 0)
            counters[meta["source_file"]] = n + 1
            chunks.append({
                "id": f"{config_name}::{meta['source_file']}::{n}",
                "text": text,
                "metadata": dict(meta),
            })
    return chunks