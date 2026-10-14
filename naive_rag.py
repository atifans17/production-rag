import sys

from src.generate import generate_answer
from src.retrieve import retrieve

question = " ".join(sys.argv[1:]) or "What is this collection about?"
chunks = retrieve(question, "fixed_500_50", mode="dense", top_k=5)

print("\nRetrieved chunks:")
for i, c in enumerate(chunks, 1):
    print(f"  [{i}] {c['metadata']['source_file']} ({c['metadata']['section']})  score={c['dense_score']:.3f}")

out = generate_answer(question, chunks)
print("\nANSWER:\n", out["answer"])
print(f"\ntokens in/out: {out['prompt_tokens']}/{out['completion_tokens']}  cost: ${out['cost_usd']}")