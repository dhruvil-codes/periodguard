from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Tuple

from periodguard.corpus import Corpus
from periodguard.models import Document, RetrievalMode


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer in lowercase."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def score_document(query: str, doc: Document) -> float:
    """
    Deterministic BM25-like token overlap scoring between query and document text.
    Gives additional weight to key financial metrics and period terms.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc.text) + _tokenize(doc.reporting_period) + _tokenize(doc.doc_type) + _tokenize(doc.company)

    if not query_tokens or not doc_tokens:
        return 0.0

    doc_token_counts = {}
    for token in doc_tokens:
        doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

    financial_keywords = {
        "revenue", "sales", "ebitda", "margin", "income", "profit", "cash", "flow",
        "guidance", "capex", "debt", "growth", "cost", "operations", "q1", "q2",
        "q3", "q4", "fy24", "fy25", "fy26", "fy27", "annual", "quarterly", "report",
        "commentary", "freight", "automation", "tariff", "business", "segment"
    }

    score = 0.0
    for q_token in set(query_tokens):
        # Ignore stop words
        if q_token in {"the", "and", "or", "did", "in", "of", "to", "what", "as", "a", "is", "for", "by", "with", "at"}:
            continue
        count = doc_token_counts.get(q_token, 0)
        if count > 0:
            weight = 3.0 if q_token in financial_keywords else 1.0
            score += weight * (1.0 + (count / len(doc_tokens)))

    return score


class DeterministicRetriever:
    """Retrieves relevant evidence documents deterministically using metadata filters and token scoring."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    def retrieve(
        self,
        query: str,
        company: str,
        as_of_date: Optional[date] = None,
        mode: RetrievalMode = RetrievalMode.CORRECT,
        top_k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents.
        
        - In CORRECT mode: as_of_date is enforced, filtering out documents published after as_of_date.
        - In BROKEN mode: as_of_date is ignored, allowing future documents to enter the candidate pool.
        """
        effective_as_of_date = as_of_date if mode == RetrievalMode.CORRECT else None
        
        # If company filter matches documents in corpus, filter by it; otherwise consider all documents in corpus
        candidate_docs = self.corpus.filter_documents(
            company=company if company else None,
            as_of_date=effective_as_of_date,
        )
        if not candidate_docs and company:
            # Fallback to date-only filter if specific company string didn't match
            candidate_docs = self.corpus.filter_documents(
                company=None,
                as_of_date=effective_as_of_date,
            )

        scored_docs = []
        for doc in candidate_docs:
            score = score_document(query, doc)
            scored_docs.append((doc, score))

        # Sort deterministically: descending by score, then ascending by doc id
        scored_docs.sort(key=lambda x: (-x[1], x[0].id))
        return scored_docs[:top_k]
