import sys

from support_triage_agent.graph import graph

email = " ".join(sys.argv[1:]) or "paisa kab aayega? I returned the shoes last week"
result = graph.invoke({"email": email})

print("Category:    ", result["category"])
print("Language:    ", result["language"])
print("Search query:", result["search_query"])
print("Articles:    ", [f'{d["article"]} ({d["similarity"]:.2f})' for d in result["docs"]])
print("\n--- Draft reply ---\n" + result["draft"])
