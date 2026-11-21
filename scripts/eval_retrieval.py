"""Retrieval eval: hit@1 and hit@3. Use --rewrite to run Claude query rewriting first."""

import json
import sys
from pathlib import Path

from support_triage_agent.rag import search

REWRITE = "--rewrite" in sys.argv
if REWRITE:
    from support_triage_agent.graph import understand

cases = [
    json.loads(line)
    for line in Path("evals/retrieval_cases.jsonl").read_text().splitlines()
    if line.strip()
]

hit1 = hit3 = 0
for case in cases:
    query = understand({"email": case["query"]})["search_query"] if REWRITE else case["query"]
    top = [article for article, _, _ in search(query, k=3)]
    ok1 = top[0] in case["expected"]
    ok3 = any(a in case["expected"] for a in top)
    hit1 += ok1
    hit3 += ok3
    mark = "PASS" if ok1 else ("TOP3" if ok3 else "FAIL")
    shown = f"{case['query'][:30]} => {query[:30]}" if REWRITE else case["query"][:48]
    print(f"{mark:5} {shown:64} -> {top[0]}")

n = len(cases)
label = "with rewrite" if REWRITE else "baseline"
print(f"\n[{label}] hit@1: {hit1}/{n} = {hit1 / n:.0%}    hit@3: {hit3}/{n} = {hit3 / n:.0%}")
