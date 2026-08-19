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
        api_key: Optional[str] = None,
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
        self.llm_adapter = LLMAnswerAdapter(api_key=api_key)

    def evaluate(
        self,
        case: Optional[EvaluationCase] = None,
        mode: RetrievalMode = RetrievalMode.CORRECT,
        custom_answer: Optional[StructuredAnswer] = None,
        api_key: Optional[str] = None,
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
        elif self.use_llm or bool(api_key):
            adapter = LLMAnswerAdapter(api_key=api_key) if api_key else self.llm_adapter
            answer = adapter.generate_answer(
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
            status, fails = validator.validate(
                claims=answer.claims,
                retrieved_docs=retrieved_docs,
                corpus=self.corpus,
                case=case,
            )
            checks[validator.name] = status
            if fails:
                all_failures.extend(fails)

        overall_status = ValidationStatus.PASS if len(all_failures) == 0 else ValidationStatus.FAIL

        return EvaluationReport(
            status=overall_status,
            mode=mode,
            question=case.question,
            as_of_date=case.as_of_date,
            checks=checks,
            failures=all_failures,
            claims=answer.claims,
            retrieved_documents=retrieved_docs,
            answer_text=answer.text,
        )

    def run_both_modes(
        self,
        case: Optional[EvaluationCase] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, EvaluationReport]:
        if case is None:
            case = get_default_case()

        report_correct = self.evaluate(case=case, mode=RetrievalMode.CORRECT, api_key=api_key)
        report_broken = self.evaluate(case=case, mode=RetrievalMode.BROKEN, api_key=api_key)

        return {
            "correct_mode": report_correct,
            "broken_mode": report_broken,
        }


def main():
    parser = argparse.ArgumentParser(description="PeriodGuard Financial Evaluation Harness")
    parser.add_argument(
        "--mode",
        choices=["correct", "broken", "both"],
        default="both",
        help="Evaluation mode to run (correct, broken, or both)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use live LLM answer adapter instead of deterministic synthesizer",
    )
    args = parser.parse_args()

    evaluator = Evaluator(use_llm=args.llm)

    if args.mode in {"correct", "both"}:
        rep_c = evaluator.evaluate(mode=RetrievalMode.CORRECT)
        print("=== CORRECT MODE REPORT ===")
        print(f"Status: {rep_c.status.value}")
        print(f"Answer: {rep_c.answer_text}")
        print(f"Checks: {[k + '=' + v.value for k, v in rep_c.checks.items()]}")
        print(f"Failures: {len(rep_c.failures)}")
        for f in rep_c.failures:
            print(f"  - {f.type.value}: {f.message}")

    if args.mode in {"broken", "both"}:
        rep_b = evaluator.evaluate(mode=RetrievalMode.BROKEN)
        print("\n=== BROKEN MODE REPORT (Naive RAG / Future Leak) ===")
        print(f"Status: {rep_b.status.value}")
        print(f"Answer: {rep_b.answer_text}")
        print(f"Checks: {[k + '=' + v.value for k, v in rep_b.checks.items()]}")
        print(f"Failures: {len(rep_b.failures)}")
        for f in rep_b.failures:
            print(f"  - {f.type.value}: {f.message}")


if __name__ == "__main__":
    main()
