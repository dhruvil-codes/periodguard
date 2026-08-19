from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from periodguard.answer import AnswerSynthesizer, LLMAnswerAdapter
from periodguard.corpus import Corpus
from periodguard.models import (
    EvaluationCase,
    EvaluationReport,
    RetrievalMode,
    StructuredAnswer,
    ValidationFailure,
    ValidationStatus,
)
from periodguard.retrieve import DeterministicRetriever
from periodguard.validators import (
    BaseValidator,
    CitationResolutionValidator,
    CitationSupportProxyValidator,
    EntityPeriodConsistencyValidator,
    TemporalConsistencyValidator,
)


def get_default_case() -> EvaluationCase:
    return EvaluationCase(
        id="case_acme_q4_fy25_ebitda",
        company="Acme Industries",
        question="As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.",
        as_of_date=date(2025, 5, 15),
        as_of_reporting_period="Q4 FY25",
        expected_metric="EBITDA margin",
        expected_unit="bps",
        negative_control_documents=["acme_fy26_annual_report", "globex_q4_fy25_results"],
    )


class Evaluator:
    """Orchestrates retrieval, answer generation (deterministic or LLM), validation, and reporting."""

    def __init__(
        self,
        corpus: Optional[Corpus] = None,
        validators: Optional[List[BaseValidator]] = None,
        use_llm: bool = False,
    ):
        if corpus is None:
            default_path = Path(__file__).parent.parent / "data" / "corpus.json"
            self.corpus = Corpus.from_json_file(default_path)
        else:
            self.corpus = corpus

        self.retriever = DeterministicRetriever(self.corpus)
        self.validators: List[BaseValidator] = validators or [
            CitationResolutionValidator(),
            TemporalConsistencyValidator(),
            EntityPeriodConsistencyValidator(),
            CitationSupportProxyValidator(),
        ]
        self.use_llm = use_llm
        self.llm_adapter = LLMAnswerAdapter()

    def evaluate(
        self,
        case: Optional[EvaluationCase] = None,
        mode: RetrievalMode = RetrievalMode.CORRECT,
        custom_answer: Optional[StructuredAnswer] = None,
    ) -> EvaluationReport:
        if case is None:
            case = get_default_case()

        # Step 1: Retrieve evidence based on mode
        scored_docs = self.retriever.retrieve(
            query=case.question,
            company=case.company,
            as_of_date=case.as_of_date,
            mode=mode,
            top_k=3,
        )
        retrieved_docs = [doc for doc, _ in scored_docs]

        # Step 2: Answer synthesis (custom, live LLM, or deterministic fixture)
        if custom_answer is not None:
            answer = custom_answer
        elif self.use_llm:
            answer = self.llm_adapter.generate_answer(
                query=case.question,
                retrieved_docs=retrieved_docs,
                company=case.company,
                mode=mode,
            )
        else:
            answer = AnswerSynthesizer.generate_answer(retrieved_docs, mode, query=case.question)

        # Step 3: Run validators
        checks: Dict[str, ValidationStatus] = {}
        all_failures: List[ValidationFailure] = []

        for validator in self.validators:
            status, failures = validator.validate(
                claims=answer.claims,
                retrieved_docs=retrieved_docs,
                corpus=self.corpus,
                case=case,
            )
            checks[validator.name] = status
            all_failures.extend(failures)

        overall_status = ValidationStatus.FAIL if all_failures else ValidationStatus.PASS

        return EvaluationReport(
            status=overall_status,
            mode=mode,
            question=case.question,
            as_of_date=case.as_of_date,
            checks=checks,
            failures=all_failures,
            claims=answer.claims,
            retrieved_documents=retrieved_docs,
        )

    def run_both_modes(self, case: Optional[EvaluationCase] = None) -> Dict[str, EvaluationReport]:
        if case is None:
            case = get_default_case()

        correct_report = self.evaluate(case=case, mode=RetrievalMode.CORRECT)
        broken_report = self.evaluate(case=case, mode=RetrievalMode.BROKEN)

        return {
            "correct_mode": correct_report,
            "broken_mode": broken_report,
        }


def run_cli():
    parser = argparse.ArgumentParser(description="PeriodGuard Evaluation Harness")
    parser.add_argument("--llm", action="store_true", help="Use live LLM adapter (OpenAI / Gemini)")
    args = parser.parse_args()

    evaluator = Evaluator(use_llm=args.llm)
    reports = evaluator.run_both_modes()
    correct = reports["correct_mode"]
    broken = reports["broken_mode"]

    print("=" * 60)
    print(f"PERIODGUARD EVALUATION RUN {'(LIVE LLM)' if args.llm else '(DETERMINISTIC)'}")
    print("=" * 60)
    print(f"CORRECT MODE: {correct.status.value}")
    if correct.failures:
        for f in correct.failures:
            print(f"  - [{f.type.value}] {f.message}")

    broken_failure_types = ", ".join(sorted({f.type.value for f in broken.failures}))
    print(f"BROKEN MODE: {broken.status.value} -- {broken_failure_types}")
    if broken.failures:
        for f in broken.failures:
            print(f"  - [{f.type.value}] Doc: {f.document_id} (Pub: {f.publication_date}) vs As-Of: {f.as_of_date}")
            print(f"    Message: {f.message}")
    print("=" * 60)


if __name__ == "__main__":
    run_cli()
