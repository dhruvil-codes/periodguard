from __future__ import annotations

from typing import List
from periodguard.models import Citation, Claim, Document, RetrievalMode, StructuredAnswer


class AnswerSynthesizer:
    """
    Synthesizes structured claims and citations based on retrieved evidence.
    In production this could wrap an LLM, but for deterministic evaluation it
    extracts claims from the top retrieved document set.
    """

    @staticmethod
    def generate_answer(retrieved_docs: List[Document], mode: RetrievalMode) -> StructuredAnswer:
        doc_ids = [d.id for d in retrieved_docs]

        # If broken mode retrieved the future FY26 report, simulate an answer citing that report
        if "acme_fy26_annual_report" in doc_ids:
            return StructuredAnswer(
                text="Acme Industries' EBITDA margin improved by 40 bps sequentially in Q4 FY25 versus Q3 FY25. Management noted in subsequent review that freight savings drove this improvement before early FY26 tariff pressures.",
                claims=[
                    Claim(
                        text="Acme Industries' EBITDA margin improved by 40 bps sequentially in Q4 FY25 versus Q3 FY25.",
                        metric="EBITDA margin",
                        value=40.0,
                        unit="bps",
                        period="Q4 FY25 vs Q3 FY25",
                        citations=[
                            Citation(
                                document_id="acme_fy26_annual_report",
                                page=45,
                                quoted_text="EBITDA margin improved by 40 bps in Q4 FY25 sequentially before contracting in H1 FY26 due to subsequent global tariff hikes.",
                            )
                        ],
                    ),
                    Claim(
                        text="Full-year FY26 consolidated revenue reached $1.85 billion.",
                        metric="Revenue",
                        value=1.85,
                        unit="billion USD",
                        period="FY26",
                        citations=[
                            Citation(
                                document_id="acme_fy26_annual_report",
                                page=45,
                                quoted_text="Total FY26 consolidated revenue reached $1.85 billion.",
                            )
                        ],
                    ),
                ],
            )

        # Standard correct answer citing Q4 FY25 Results and Earnings Call
        claims: List[Claim] = []
        if "acme_q4_fy25_results" in doc_ids:
            claims.append(
                Claim(
                    text="Acme Industries' EBITDA margin increased by 40 bps sequentially to 18.4% in Q4 FY25 compared to 18.0% in Q3 FY25.",
                    metric="EBITDA margin",
                    value=40.0,
                    unit="bps",
                    period="Q4 FY25 vs Q3 FY25",
                    citations=[
                        Citation(
                            document_id="acme_q4_fy25_results",
                            page=12,
                            quoted_text="EBITDA margin increased by 40 bps sequentially to 18.4% compared to 18.0% in Q3 FY25.",
                        )
                    ],
                )
            )

        if "acme_q4_fy25_earnings_call" in doc_ids:
            claims.append(
                Claim(
                    text="Management stated the EBITDA margin expansion was primarily driven by lower ocean freight rates and automation efficiencies.",
                    metric="EBITDA margin",
                    value=40.0,
                    unit="bps",
                    period="Q4 FY25 vs Q3 FY25",
                    citations=[
                        Citation(
                            document_id="acme_q4_fy25_earnings_call",
                            page=4,
                            quoted_text="Our EBITDA margin expansion of 40 bps in Q4 FY25 versus Q3 FY25 was primarily driven by lower ocean freight rates, plant automation efficiencies, and favorable product mix in our specialty materials segment.",
                        )
                    ],
                )
            )

        return StructuredAnswer(
            text="Acme Industries' EBITDA margin improved by 40 bps sequentially in Q4 FY25 versus Q3 FY25 to 18.4%, driven by lower ocean freight rates and plant automation efficiencies.",
            claims=claims,
        )
