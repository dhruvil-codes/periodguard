from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from periodguard.models import Document


class Corpus:
    """Manages financial documents with metadata filtering capabilities."""

    def __init__(self, documents: Optional[List[Document]] = None):
        self._documents: Dict[str, Document] = {}
        if documents:
            for doc in documents:
                self._documents[doc.id] = doc

    @classmethod
    def from_json_file(cls, filepath: str | Path) -> Corpus:
        """Load corpus from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        docs = [Document(**item) for item in raw_data]
        return cls(docs)

    def all_documents(self) -> List[Document]:
        """Return all documents in the corpus."""
        return list(self._documents.values())

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a single document by its unique ID."""
        return self._documents.get(doc_id)

    def filter_documents(
        self,
        company: Optional[str] = None,
        as_of_date: Optional[date] = None,
    ) -> List[Document]:
        """
        Filter documents by company and publication date boundary.
        
        - If company is provided, only matches for that company are returned.
        - If as_of_date is provided, documents with publication_date > as_of_date
          are strictly excluded (temporal gate).
        """
        filtered = []
        for doc in self._documents.values():
            if company and doc.company.strip().lower() != company.strip().lower():
                continue
            if as_of_date and doc.publication_date > as_of_date:
                continue
            filtered.append(doc)
        return filtered
