import argparse
import json
import re
import statistics
from datetime import datetime
import time
from pathlib import Path
import openai
from src import config
from src.generate import get_llm
from src.pipeline import RAGPipeline


def judge(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = get_llm().chat.completions.create(
                model=config.LLM_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a strict grader. Reply with ONLY a JSON object."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}
        except openai.OpenAIError:
            wait_time = (attempt + 1) * 6
            print(f"[Judge 503/429] Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=config.DEFAULT_COLLECTION)
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    pipe = RAGPipeline(collection=args.collection, mode=args.mode)
    questions = json.load(open("eval/questions.json", encoding="utf-8"))

    unans_total = unans_ok = 0
    ans_total = false_refusals = faithful = correct = judged = 0
    costs, latencies = [], []

    for q in questions:
        r = pipe.ask(q["question"], use_cache=False)
        costs.append(r["cost_usd"])
        latencies.append(r["latency_ms"])

        if not q["answerable"]:
            unans_total += 1
            if r["refused"]:
                unans_ok += 1
            else:
                print(f"[LEAK] id={q['id']} answered an unanswerable question: {r['answer'][:120]}")
            time.sleep(2)
            continue

        ans_total += 1
        if r["refused"]:
            false_refusals += 1
            time.sleep(2)
            continue

        context = "\n\n".join(f"[{d['doc']}] {d['text']}" for d in r["retrieved"])
        time.sleep(2)
        f = judge(
            f"Context:\n{context}\n\nAnswer:\n{r['answer']}\n\n"
            'Is EVERY factual claim in the Answer supported by the Context? '
            'Reply as JSON: {"faithful": true or false}'
        )
        time.sleep(2)
        c = judge(
            f"Question: {q['question']}\nReference answer: {q['answer']}\n"
            f"Candidate answer: {r['answer']}\n\n"
            'Does the candidate answer convey the same key facts as the reference? '
            'Reply as JSON: {"correct": true or false}'
        )
        judged += 1
        faithful += bool(f.get("faithful"))
        correct += bool(c.get("correct"))
        time.sleep(2)

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "collection": args.collection,
        "mode": args.mode,
        "note": args.note,
        "refusal_accuracy": round(unans_ok / max(unans_total, 1), 3),
        "false_refusal_rate": round(false_refusals / max(ans_total, 1), 3),
        "faithfulness": round(faithful / max(judged, 1), 3),
        "correctness": round(correct / max(judged, 1), 3),
        "avg_cost_usd": round(statistics.mean(costs), 6),
        "avg_latency_ms": int(statistics.mean(latencies)),
    }
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    with open("eval/results/answer_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    print("\n=== Answer quality ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()