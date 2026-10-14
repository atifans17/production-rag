import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from src.retrieve import retrieve


def norm(s):
    return re.sub(r"\s+", " ", s).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="fixed_500_50")
    ap.add_argument("--mode", default="dense", help="dense | bm25 | hybrid | hybrid_rerank")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    questions = [q for q in json.load(open("eval/questions.json", encoding="utf-8")) if q["answerable"]]
    hit1 = hit3 = hit5 = 0
    rr_total = 0.0
    misses = []

    for q in questions:
        chunks = retrieve(q["question"], args.collection, mode=args.mode, top_k=5)
        target = norm(q["expected_substring"])
        rank = next((i for i, c in enumerate(chunks, 1) if target in norm(c["text"])), None)
        if rank:
            rr_total += 1 / rank
            hit1 += rank <= 1
            hit3 += rank <= 3
            hit5 += rank <= 5
        else:
            misses.append(q)

    n = len(questions)
    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "collection": args.collection,
        "mode": args.mode,
        "note": args.note,
        "n": n,
        "hit@1": round(hit1 / n, 3),
        "hit@3": round(hit3 / n, 3),
        "hit@5": round(hit5 / n, 3),
        "mrr": round(rr_total / n, 3),
    }
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    with open("eval/results/retrieval_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n{args.collection} | {args.mode}  ({n} questions)")
    print(f"Hit@1={result['hit@1']:.0%}  Hit@3={result['hit@3']:.0%}  Hit@5={result['hit@5']:.0%}  MRR={result['mrr']}")
    print("\nMarkdown row for your README:")
    print(f"| {args.collection} | {args.mode} | {result['hit@1']:.0%} | {result['hit@3']:.0%} | {result['hit@5']:.0%} | {result['mrr']} |")
    if misses:
        print("\nMissed questions:")
        for q in misses:
            print(f"  id={q['id']}: {q['question']}")


if __name__ == "__main__":
    main()