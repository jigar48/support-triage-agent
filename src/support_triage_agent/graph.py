"""Step 2: understand -> retrieve -> draft."""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from support_triage_agent.rag import search

load_dotenv()

FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-5-5")
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "claude-sonnet-5-5")

Category = Literal[
    "order_status", "shipping_delivery", "returns_refunds", "billing_payment", "product_question"
]


class Understanding(BaseModel):
    category: Category = Field(description="Main topic of the email")
    search_query: str = Field(
        description="Short English search query for the help center (max 10 words). "
        "Translate Hinglish or other languages into English and use help-center wording."
    )
    language: str = Field(description="Language of the email, e.g. English, Hinglish")


class TriageState(TypedDict, total=False):
    email: str
    category: str
    search_query: str
    language: str
    docs: list[dict]
    draft: str


fast_llm = ChatAnthropic(model=FAST_MODEL, max_tokens=300)
draft_llm = ChatAnthropic(model=DRAFT_MODEL, max_tokens=600)
understander = fast_llm.with_structured_output(Understanding)

UNDERSTAND_PROMPT = """You triage customer support emails for ShopLane, an online store.
Return the email's category, a short English search query for our help center, and its language.

Categories:
- order_status: where is my order, tracking
- shipping_delivery: shipping options and costs, address changes, delayed or lost packages
- returns_refunds: returns, refunds, exchanges, cancellations, damaged items
- billing_payment: payment methods, failed payments, double charges, bank disputes
- product_question: sizing, warranty, product details"""

DRAFT_PROMPT = """You are a ShopLane support agent writing a reply email.
Rules:
- Use ONLY the help-center articles below. Never invent policies, numbers, or timelines.
- After each fact, cite the article title in square brackets, e.g. [Refund timelines].
- If the articles do not answer the question, say a teammate will follow up. Do not guess.
- Write in English, friendly and concise (under 120 words). Sign off as "ShopLane Support".

Articles:
{articles}"""


def understand(state: TriageState) -> TriageState:
    result = understander.invoke([("system", UNDERSTAND_PROMPT), ("human", state["email"])])
    return {
        "category": result.category,
        "search_query": result.search_query,
        "language": result.language,
    }


def retrieve(state: TriageState) -> TriageState:
    results = search(state["search_query"], k=3)
    docs = [
        {"article": article, "content": content, "similarity": similarity}
        for article, content, similarity in results
    ]
    return {"docs": docs}


def text_of(content) -> str:
    """Claude may return a plain string or a list of content blocks; keep only the text."""
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )


def draft(state: TriageState) -> TriageState:
    articles = "\n\n---\n\n".join(d["content"] for d in state["docs"])
    reply = draft_llm.invoke(
        [("system", DRAFT_PROMPT.format(articles=articles)), ("human", state["email"])]
    )
    return {"draft": text_of(reply.content)}


def build_graph():
    g = StateGraph(TriageState)
    g.add_node("understand", understand)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_edge(START, "understand")
    g.add_edge("understand", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", END)
    return g.compile()


graph = build_graph()
