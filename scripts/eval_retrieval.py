"""Retrieval eval: is the right article ranked 1st (hit@1) or in the top 3 (hit@3)?"""

import json
from pathlib import Path

from support_triage_agent.rag import search

cases = [
    json.loads(line)
    for line in Path("evals/retrieval_cases.jsonl").read_text().splitlines()
    if line.strip()
]

hit1 = hit3 = 0
for case in cases:
    top = [article for article, _, _ in search(case["query"], k=3)]
    ok1 = top[0] in case["expected"]
    ok3 = any(a in case["expected"] for a in top)
    hit1 += ok1
    hit3 += ok3
    mark = "PASS" if ok1 else ("TOP3" if ok3 else "FAIL")
    print(f"{mark:5} {case['query'][:48]:48} -> {top[0]}")

n = len(cases)
print(f"\nhit@1: {hit1}/{n} = {hit1 / n:.0%}    hit@3: {hit3}/{n} = {hit3 / n:.0%}")
