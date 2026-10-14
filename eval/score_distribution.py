import json

import numpy as np

from src import config
from src.retrieve import retrieve

questions = json.load(open("eval/questions.json", encoding="utf-8"))
answerable, unanswerable = [], []

for q in questions:
    top = retrieve(q["question"], config.DEFAULT_COLLECTION, mode="hybrid_rerank", top_k=1)
    score = top[0]["rerank_score"] if top else -99.0
    (answerable if q["answerable"] else unanswerable).append(score)

a, u = np.array(answerable), np.array(unanswerable)
print(f"Answerable   : min={a.min():.2f}  10th pct={np.percentile(a, 10):.2f}  median={np.median(a):.2f}")
print(f"Unanswerable : max={u.max():.2f}  90th pct={np.percentile(u, 90):.2f}  median={np.median(u):.2f}")
print("\nPick a threshold between the two groups. If they overlap, favour refusing")
print("slightly too often over answering wrongly, then rely on the prompt as the second layer.")