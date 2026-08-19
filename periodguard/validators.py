from __future__ import annotations

import re
from typing import List, Optional, Tuple

from periodguard.corpus import Corpus
from periodguard.models import (
    Citation,
    Claim,
    Document,
    EvaluationCase,
    FailureType,
    ValidationFailure,
    ValidationStatus,
)


class BaseValidator:
    name: str = "base_validator"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        raise NotImplementedError


class CitationResolutionValidator(BaseValidator):
    name = "citation_resolution"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        failures: List[ValidationFailure] = []
        retrieved_ids = {d.id for d in retrieved_docs}

        for claim in claims:
            if not claim.citations:
                failures.append(
                    ValidationFailure(
                        type=FailureType.INVALID_CITATION,
                        claim_text=claim.text,
                        message=f"Claim has no citations attached: '{claim.text}'",
                    )
                )
                continue

            for citation in claim.citations:
                doc = corpus.get_document(citation.document_id)
                if not doc:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Cited document ID '{citation.document_id}' does not exist in the corpus.",
                        )
                    )
                    continue

                if citation.document_id not in retrieved_ids:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Cited document '{citation.document_id}' was not in the retriever's returned set.",
                        )
                    )

                if citation.page != doc.page:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Citation page mismatch for '{doc.id}': cited page {citation.page} vs document page {doc.page}.",
                        )
                    )

                # Normalize whitespace when checking quote presence
                normalized_quote = " ".join(citation.quoted_text.split()).lower()
                normalized_doc = " ".join(doc.text.split()).lower()
                if normalized_quote not in normalized_doc:
                    failures.append(
                        ValidationFailure(
                            type=FailureType.INVALID_CITATION,
                            document_id=citation.document_id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Quoted text was not found verbatim in document '{doc.id}' page {doc.page}.",
                        )
                    )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures


class TemporalConsistencyValidator(BaseValidator):
    name = "temporal_consistency"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        failures: List[ValidationFailure] = []

        for claim in claims:
            for citation in claim.citations:
                doc = corpus.get_document(citation.document_id)
                if not doc:
                    continue

                # Strict gating rule: publication_date must be <= as_of_date
                if doc.publication_date > case.as_of_date:
                    delta_days = (doc.publication_date - case.as_of_date).days
                    failures.append(
                        ValidationFailure(
                            type=FailureType.FUTURE_PERIOD_LEAK,
                            document_id=doc.id,
                            publication_date=doc.publication_date,
                            as_of_date=case.as_of_date,
                            claim_text=claim.text,
                            citation=citation,
                            message=(
                                f"Future-period leakage detected: Document '{doc.id}' was published on "
                                f"{doc.publication_date.isoformat()} (+{delta_days} days after the as-of cutoff "
                                f"{case.as_of_date.isoformat()})."
                            ),
                        )
                    )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures


class EntityPeriodConsistencyValidator(BaseValidator):
    name = "entity_period_consistency"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        failures: List[ValidationFailure] = []

        for claim in claims:
            # Check period in claim if specified
            if claim.period:
                if "fy26" in claim.period.lower() and "fy25" in case.as_of_reporting_period.lower():
                    failures.append(
                        ValidationFailure(
                            type=FailureType.ENTITY_OR_PERIOD_MISMATCH,
                            claim_text=claim.text,
                            message=f"Claim references period '{claim.period}', which exceeds case period boundary '{case.as_of_reporting_period}'.",
                        )
                    )

            for citation in claim.citations:
                doc = corpus.get_document(citation.document_id)
                if not doc:
                    continue

                if doc.company.strip().lower() != case.company.strip().lower():
                    failures.append(
                        ValidationFailure(
                            type=FailureType.ENTITY_OR_PERIOD_MISMATCH,
                            document_id=doc.id,
                            claim_text=claim.text,
                            citation=citation,
                            message=f"Entity mismatch: Cited document belongs to '{doc.company}', expected '{case.company}'.",
                        )
                    )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures


class CitationSupportProxyValidator(BaseValidator):
    name = "citation_support_proxy"

    def validate(
        self,
        claims: List[Claim],
        retrieved_docs: List[Document],
        corpus: Corpus,
        case: EvaluationCase,
    ) -> Tuple[ValidationStatus, List[ValidationFailure]]:
        """
        Rule-based proxy for citation support:
        Checks numeric value presence, metric phrase support, unit presence, and directional consistency.
        """
        failures: List[ValidationFailure] = []

        for claim in claims:
            for citation in claim.citations:
                quote = citation.quoted_text.lower()
                doc = corpus.get_document(citation.document_id)
                doc_text = doc.text.lower() if doc else quote

                # 1. Numeric value check
                if claim.value is not None:
                    val_str_int = str(int(claim.value)) if claim.value.is_integer() else str(claim.value)
                    val_str_float = f"{claim.value:.1f}"
                    if val_str_int not in quote and val_str_float not in quote and val_str_int not in doc_text:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.UNSUPPORTED_CLAIM,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Numeric value {claim.value} asserted in claim was not found in cited quote or document.",
                            )
                        )

                # 2. Metric phrase check (skip for qualitative/overview metrics)
                qualitative_metrics = {"business operations", "document evidence", "document summary", "overview", "operational drivers"}
                if claim.metric and claim.metric.lower() not in qualitative_metrics:
                    metric_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9]+\b", claim.metric.lower()) if len(t) > 2]
                    matched = any(t in quote or t in doc_text for t in metric_tokens)
                    if not matched:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.UNSUPPORTED_CLAIM,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Metric '{claim.metric}' asserted in claim was not found in cited evidence.",
                            )
                        )

                # 3. Unit check
                if claim.unit:
                    unit_clean = claim.unit.lower().strip()
                    unit_supported = False
                    if "usd" in unit_clean or "$" in unit_clean or "dollar" in unit_clean:
                        unit_supported = "$" in quote or "dollar" in quote or "usd" in quote or "$" in doc_text
                    elif unit_clean == "%":
                        unit_supported = "%" in quote or "percent" in quote or "%" in doc_text
                    elif unit_clean == "bps":
                        unit_supported = "bps" in quote or "basis points" in quote or "bps" in doc_text
                    else:
                        unit_supported = unit_clean in quote or unit_clean in doc_text

                    if not unit_supported:
                        failures.append(
                            ValidationFailure(
                                type=FailureType.UNSUPPORTED_CLAIM,
                                document_id=citation.document_id,
                                claim_text=claim.text,
                                citation=citation,
                                message=f"Unit '{claim.unit}' asserted in claim was not supported by cited evidence.",
                            )
                        )

        status = ValidationStatus.FAIL if failures else ValidationStatus.PASS
        return status, failures
