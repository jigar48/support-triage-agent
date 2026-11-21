import sys

from support_triage_agent.rag import search

query = " ".join(sys.argv[1:]) or "how long does a refund take"
print(f"Query: {query}\n")
for article, _content, similarity in search(query):
    print(f"{similarity:.3f}  {article}")
