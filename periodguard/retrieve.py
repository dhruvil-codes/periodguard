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
    Deterministic token overlap scoring between query and document text.
    Gives baseline score for company matches and additional weight to exact financial period/metric tokens.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc.text) + _tokenize(doc.reporting_period) + _tokenize(doc.doc_type)

    # Baseline score so any document in candidate pool has positive ranking
    score = 1.0

    if not query_tokens or not doc_tokens:
        return score

    doc_token_counts = {}
    for token in doc_tokens:
        doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

    for q_token in set(query_tokens):
        # Ignore common stop words
        if q_token in {"the", "and", "or", "did", "in", "of", "to", "what", "as", "a", "is", "about", "for"}:
            continue
        count = doc_token_counts.get(q_token, 0)
        if count > 0:
            # Boost high-signal financial tokens
            weight = 3.0 if q_token in {"ebitda", "margin", "q4", "fy25", "q3", "reason", "management", "revenue", "document", "report", "call"} else 1.5
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
        candidate_docs = self.corpus.filter_documents(
            company=company,
            as_of_date=effective_as_of_date,
        )

        scored_docs = []
        for doc in candidate_docs:
            score = score_document(query, doc)
            scored_docs.append((doc, score))

        # Sort deterministically: descending by score, then ascending by doc id
        scored_docs.sort(key=lambda x: (-x[1], x[0].id))
        return scored_docs[:top_k]
