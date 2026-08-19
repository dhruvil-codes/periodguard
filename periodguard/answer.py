from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import List, Optional, Tuple
from periodguard.models import Citation, Claim, Document, RetrievalMode, StructuredAnswer


def _extract_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly."""
    lines = text.split("\n")
    sentences = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line_s)
        for p in parts:
            p_clean = p.strip()
            if len(p_clean) > 12:
                sentences.append(p_clean)
    return sentences


def _score_sentence(sentence: str, query_tokens: List[str]) -> float:
    s_lower = sentence.lower()
    score = 0.0
    for tok in query_tokens:
        if tok in s_lower:
            score += 3.0
    # Boost numeric and financial terms
    if re.search(r"(\$\s*\d+|\b\d+(\.\d+)?\s*(bps|%|million|billion|thousand|usd)\b)", s_lower):
        score += 2.5
    if any(k in s_lower for k in ["revenue", "ebitda", "margin", "growth", "income", "segment", "guidance", "profit", "cash", "sales", "operating", "debt", "capex", "cost"]):
        score += 2.0
    return score


def _extract_numeric_metric(sentence: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Intelligently extracts metric name, numeric value, and unit from a financial sentence.
    Avoids mistaking calendar dates or years (like 2025 or March 31) for financial values.
    """
    # 1. Currency (e.g. $420 million, $1.85 billion, $50,000)
    curr_match = re.search(r"\$\s*(\d+(\.\d+)?)\s*(million|billion|thousand|m|b|k)?", sentence, re.IGNORECASE)
    if curr_match:
        val = float(curr_match.group(1))
        scale = (curr_match.group(3) or "").lower()
        unit = "million USD" if scale in {"million", "m"} else ("billion USD" if scale in {"billion", "b"} else "USD")
        
        # Metric name heuristics
        s_lower = sentence.lower()
        if "revenue" in s_lower or "sales" in s_lower:
            metric = "Revenue"
        elif "net income" in s_lower or "profit" in s_lower:
            metric = "Net Income"
        elif "operating cash" in s_lower or "free cash flow" in s_lower or "cash" in s_lower:
            metric = "Cash Flow"
        elif "capex" in s_lower or "capital expenditure" in s_lower:
            metric = "CapEx"
        elif "debt" in s_lower or "liabilities" in s_lower:
            metric = "Total Debt"
        else:
            metric = "Financial Metric"
        return metric, val, unit

    # 2. Basis points (e.g. 40 bps, 50 basis points)
    bps_match = re.search(r"\b(\d+(\.\d+)?)\s*(bps|basis points)\b", sentence, re.IGNORECASE)
    if bps_match:
        val = float(bps_match.group(1))
        metric = "EBITDA margin" if "margin" in sentence.lower() or "ebitda" in sentence.lower() else "Basis Point Change"
        return metric, val, "bps"

    # 3. Percentages (e.g. 18.4%, 25 percent)
    pct_match = re.search(r"\b(\d+(\.\d+)?)\s*(%|percent|percentage)\b", sentence, re.IGNORECASE)
    if pct_match:
        val = float(pct_match.group(1))
        metric = "EBITDA margin" if "margin" in sentence.lower() else ("Growth Rate" if "growth" in sentence.lower() else "Percentage Metric")
        return metric, val, "%"

    # Qualitative explanation
    s_lower = sentence.lower()
    if any(k in s_lower for k in ["management", "commentary", "ceo", "cfo", "driven by", "because", "due to", "strategy", "outlook"]):
        return "Management Commentary", None, None
    if any(k in s_lower for k in ["product", "service", "manufactur", "operat", "business", "segment", "market", "solutions"]):
        return "Business Operations", None, None

    return "Document Evidence", None, None


class AnswerSynthesizer:
    """
    Generalized Financial RAG Synthesizer:
    Works dynamically with ANY ingested PDF, 10-Q, 10-K, earnings call transcript, or custom document.
    Extracts relevant facts, figures, and qualitative context to construct a clean natural-language answer.
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

        q_lower = (query or "").lower().strip()
        stop_words = {"what", "is", "the", "about", "did", "in", "and", "to", "of", "a", "an", "for", "tell", "me", "how", "does", "do", "were", "are", "was"}
        q_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower) if t not in stop_words]

        claims: List[Claim] = []
        answer_paragraphs: List[str] = []

        # Iterate across retrieved documents to select the highest-scoring sentences
        for doc in retrieved_docs:
            sentences = _extract_sentences(doc.text)
            if not sentences:
                continue

            if q_tokens:
                scored = [(s, _score_sentence(s, q_tokens)) for s in sentences]
                scored.sort(key=lambda x: -x[1])
                top_sentences = [s for s, sc in scored[:2] if sc > 0]
                if not top_sentences:
                    top_sentences = [sentences[0]]
            else:
                top_sentences = [sentences[0]]

            doc_claims_text = []
            for s in top_sentences:
                metric_name, num_val, unit_val = _extract_numeric_metric(s)
                
                claim = Claim(
                    text=s,
                    metric=metric_name,
                    value=num_val,
                    unit=unit_val,
                    period=doc.reporting_period,
                    citations=[
                        Citation(
                            document_id=doc.id,
                            page=doc.page or 1,
                            quoted_text=s,
                        )
                    ],
                )
                claims.append(claim)
                doc_claims_text.append(s)

            if doc_claims_text:
                answer_paragraphs.append(" ".join(doc_claims_text))

        # Compose cohesive overall answer
        if answer_paragraphs:
            composed_text = " ".join(answer_paragraphs)
        else:
            composed_text = f"Retrieved {len(retrieved_docs)} relevant document(s) matching your query."

        return StructuredAnswer(text=composed_text, claims=claims)


class LLMAnswerAdapter:
    """
    Live LLM Adapter supporting Groq, OpenAI, Gemini, and local LLM endpoints.
    Prompts the model with retrieved evidence and returns structured claims + citations.
    Falls back seamlessly to AnswerSynthesizer if no API key is provided.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        groq_key = os.environ.get("GROQ_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        self.api_key = api_key or groq_key or openai_key or gemini_key

        if (self.api_key and self.api_key.startswith("gsk_")) or (groq_key and not base_url):
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = model or "llama-3.3-70b-versatile"
        else:
            self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = model or "gpt-4o-mini"

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
            "You are an expert financial research AI assistant. Answer the user's question directly, conversationally, and accurately based strictly on the provided evidence documents. "
            "Cite all facts with document_id, page, and exact quoted_text. "
            "Respond ONLY in valid JSON matching this schema:\n"
            "{\n"
            '  "text": "Natural conversational synthesized answer directly answering the question",\n'
            '  "claims": [\n'
            "    {\n"
            '      "text": "Specific claim sentence",\n'
            '      "metric": "e.g. Revenue, EBITDA margin, Business Operations, Net Income, or CapEx",\n'
            '      "value": 420.0,\n'
            '      "unit": "million USD",\n'
            '      "period": "Q4 FY25",\n'
            '      "citations": [\n'
            '        {"document_id": "doc_id", "page": 1, "quoted_text": "exact quote from document"}\n'
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
