import json
import re

from src.loaders import load_directory


def norm(s):
    return re.sub(r"\s+", " ", s).lower().strip()


questions = json.load(open("eval/questions.json", encoding="utf-8"))
corpus = [norm(r["text"]) for r in load_directory("data")]

bad = 0
for q in questions:
    if q["answerable"] and not any(norm(q["expected_substring"]) in t for t in corpus):
        print(f"[NOT FOUND] id={q['id']}: {q['expected_substring']!r}")
        bad += 1
answerable = sum(q["answerable"] for q in questions)
print(f"{answerable} answerable, {len(questions) - answerable} unanswerable, {bad} bad substrings")