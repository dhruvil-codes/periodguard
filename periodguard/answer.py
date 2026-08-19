from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import List, Optional
from periodguard.models import Citation, Claim, Document, RetrievalMode, StructuredAnswer


class AnswerSynthesizer:
    """
    Synthesizes structured claims and citations based on retrieved evidence.
    Handles standard fixtures deterministically, and dynamically synthesizes
    claims from any custom uploaded PDF/document.
    """

    @staticmethod
    def generate_answer(
        retrieved_docs: List[Document],
        mode: RetrievalMode = RetrievalMode.CORRECT,
        query: str = "",
    ) -> StructuredAnswer:
        if not retrieved_docs:
            return StructuredAnswer(
                text="No eligible evidence documents were found matching your criteria and as-of cutoff date.",
                claims=[],
            )

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

        # Standard fixture matches
        if any(d.id in {"acme_q4_fy25_results", "acme_q4_fy25_earnings_call"} for d in retrieved_docs):
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

            if claims:
                return StructuredAnswer(
                    text="Acme Industries' EBITDA margin improved by 40 bps sequentially in Q4 FY25 versus Q3 FY25 to 18.4%, driven by lower ocean freight rates and plant automation efficiencies.",
                    claims=claims,
                )

        # Dynamic fallback for custom uploaded files / queries
        claims = []
        synthesized_sentences = []
        for doc in retrieved_docs:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", doc.text) if len(s.strip()) > 20]
            if sentences:
                best_sentence = sentences[0]
                synthesized_sentences.append(best_sentence)

                # Extract first number if any
                num_match = re.search(r"\b\d+(\.\d+)?\b", best_sentence)
                val = float(num_match.group(0)) if num_match else None

                unit = None
                if "%" in best_sentence:
                    unit = "%"
                elif "bps" in best_sentence.lower():
                    unit = "bps"
                elif "$" in best_sentence or "usd" in best_sentence.lower():
                    unit = "USD"

                claims.append(
                    Claim(
                        text=best_sentence,
                        metric=doc.doc_type or "Financial Metric",
                        value=val,
                        unit=unit,
                        period=doc.reporting_period,
                        citations=[
                            Citation(
                                document_id=doc.id,
                                page=doc.page or 1,
                                quoted_text=best_sentence,
                            )
                        ],
                    )
                )

        full_text = " ".join(synthesized_sentences) if synthesized_sentences else f"Evidence retrieved from {len(retrieved_docs)} filings."
        return StructuredAnswer(text=full_text, claims=claims)


class LLMAnswerAdapter:
    """
    Live LLM Adapter supporting OpenAI-compatible endpoints (OpenAI, Gemini, Ollama, Groq).
    Prompts the LLM with retrieved evidence and requests structured JSON output with claims and citations.
    Falls back gracefully to AnswerSynthesizer if no API key is set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Document],
        company: str,
        mode: RetrievalMode = RetrievalMode.CORRECT,
    ) -> StructuredAnswer:
        if not self.is_available:
            return AnswerSynthesizer.generate_answer(retrieved_docs, mode, query=query)

        doc_context = "\n\n".join([
            f"Document ID: {d.id}\nCompany: {d.company}\nDoc Type: {d.doc_type}\nPeriod: {d.reporting_period}\nPublished Date: {d.publication_date}\nPage: {d.page}\nContent: {d.text}"
            for d in retrieved_docs
        ])

        system_prompt = (
            "You are a financial research analyst. Answer the user's question strictly based ONLY on the provided evidence documents. "
            "You must return structured claims with citations citing the document_id, page number, and exact quoted_text. "
            "Respond ONLY in valid JSON matching this schema:\n"
            "{\n"
            '  "text": "Synthesized full answer",\n'
            '  "claims": [\n'
            "    {\n"
            '      "text": "Specific claim sentence",\n'
            '      "metric": "e.g. EBITDA margin",\n'
            '      "value": 40.0,\n'
            '      "unit": "bps",\n'
            '      "period": "Q4 FY25 vs Q3 FY25",\n'
            '      "citations": [\n'
            '        {"document_id": "doc_id", "page": 12, "quoted_text": "exact quote"}\n'
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = f"Target Company: {company}\nQuestion: {query}\n\nEvidence Documents:\n{doc_context}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                parsed_json = json.loads(content)
                return StructuredAnswer(**parsed_json)
        except Exception:
            return AnswerSynthesizer.generate_answer(retrieved_docs, mode, query=query)
