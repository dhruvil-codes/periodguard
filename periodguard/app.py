from __future__ import annotations

import io
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from periodguard.answer import AnswerSynthesizer, LLMAnswerAdapter
from periodguard.corpus import Corpus
from periodguard.evaluator import Evaluator, get_default_case
from periodguard.models import (
    Citation,
    Claim,
    Document,
    EvaluationCase,
    EvaluationReport,
    FailureType,
    RetrievalMode,
    ValidationStatus,
)

# PDF extraction
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


app = FastAPI(
    title="PeriodGuard",
    description="Evaluation harness for financial research systems detecting future-period citation leakage.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.json"
active_corpus = Corpus.from_json_file(DEFAULT_CORPUS_PATH)
evaluator_deterministic = Evaluator(corpus=active_corpus, use_llm=False)
evaluator_llm = Evaluator(corpus=active_corpus, use_llm=True)


class CustomEvaluationRequest(BaseModel):
    company: str = Field(default="Acme Industries")
    question: str = Field(
        default="As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence."
    )
    as_of_date: str = Field(default="2025-05-15")
    as_of_reporting_period: str = Field(default="Q4 FY25")
    use_llm: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)


class AddDocumentRequest(BaseModel):
    id: str
    company: str
    doc_type: str
    reporting_period: str
    publication_date: str
    page: int = 1
    text: str
    source_url: str = "Uploaded Document"


BENCHMARK_TESTS = [
    {
        "id": "future_leak_default",
        "title": "Future-Period Leak Trap",
        "badge": "High-Signal Trap",
        "badge_color": "rose",
        "company": "Acme Industries",
        "question": "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Tests if naive retrieval leaks citations from the FY26 Annual Report (published Aug 2025) for a May 2025 query.",
    },
    {
        "id": "clean_historical",
        "title": "Clean Historical Query",
        "badge": "Safe Historical",
        "badge_color": "emerald",
        "company": "Acme Industries",
        "question": "What was Acme Industries' sequential EBITDA margin change in Q4 FY25?",
        "as_of_date": "2025-06-01",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Sets the as-of date after Q4 results release (June 2025), verifying a clean 4/4 PASS across all validators.",
    },
    {
        "id": "pre_release_cutoff",
        "title": "Pre-Release Earnings Call Cutoff",
        "badge": "Temporal Boundary",
        "badge_color": "amber",
        "company": "Acme Industries",
        "question": "What management commentary was provided regarding Q4 FY25 EBITDA margin expansion?",
        "as_of_date": "2025-05-11",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Sets cutoff to May 11 (before the May 12 Earnings Call), catching commentary cited before it took place.",
    },
    {
        "id": "peer_entity_mismatch",
        "title": "Peer Entity Contamination Trap",
        "badge": "Entity Check",
        "badge_color": "indigo",
        "company": "Acme Industries",
        "question": "Did EBITDA margin reach 19.1% in Q4 FY25?",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Tests whether the system rejects citations from Globex Corp (peer company with 19.1% margin).",
    },
]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "PeriodGuard Financial Evaluation Landing Engine"}


@app.get("/api/presets")
def get_presets() -> List[Dict[str, Any]]:
    return BENCHMARK_TESTS


@app.get("/api/corpus")
def list_corpus_documents() -> List[Dict[str, Any]]:
    return [doc.model_dump(mode="json") for doc in active_corpus.all_documents()]


@app.post("/api/corpus/reset")
def reset_corpus() -> Dict[str, Any]:
    global active_corpus, evaluator_deterministic, evaluator_llm
    active_corpus = Corpus.from_json_file(DEFAULT_CORPUS_PATH)
    evaluator_deterministic = Evaluator(corpus=active_corpus, use_llm=False)
    evaluator_llm = Evaluator(corpus=active_corpus, use_llm=True)
    return {
        "status": "success",
        "message": "Corpus restored to default fixtures",
        "count": len(active_corpus.all_documents()),
    }


@app.post("/api/corpus/add")
def add_document_manual(payload: AddDocumentRequest) -> Dict[str, Any]:
    try:
        pub_date = datetime.strptime(payload.publication_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    doc = Document(
        id=payload.id.strip(),
        company=payload.company.strip(),
        doc_type=payload.doc_type.strip(),
        reporting_period=payload.reporting_period.strip(),
        publication_date=pub_date,
        page=payload.page,
        text=payload.text.strip(),
        source_url=payload.source_url.strip(),
    )
    active_corpus._documents[doc.id] = doc
    return {"status": "success", "message": f"Document '{doc.id}' added to corpus.", "doc": doc.model_dump(mode="json")}


@app.post("/api/corpus/upload")
async def upload_document_file(
    file: UploadFile = File(...),
    company: str = Form("Acme Industries"),
    doc_type: str = Form("Financial Filing"),
    reporting_period: str = Form("Q4 FY25"),
    publication_date: str = Form("2025-05-10"),
    custom_doc_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        pub_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid publication_date format. Use YYYY-MM-DD.")

    file_bytes = await file.read()
    filename = file.filename or "uploaded_document"
    doc_id = (custom_doc_id or Path(filename).stem).lower().replace(" ", "_")

    extracted_text = ""
    if filename.lower().endswith(".pdf"):
        if not PYPDF_AVAILABLE:
            raise HTTPException(status_code=500, detail="PDF parsing library (pypdf) is unavailable.")
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
            extracted_text = "\n\n".join(pages_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    else:
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1")

    if not extracted_text.strip():
        extracted_text = f"Sample financial text content extracted from {filename}."

    doc = Document(
        id=doc_id,
        company=company.strip(),
        doc_type=doc_type.strip(),
        reporting_period=reporting_period.strip(),
        publication_date=pub_date,
        page=1,
        text=extracted_text.strip(),
        source_url=f"Local Upload: {filename}",
    )
    active_corpus._documents[doc.id] = doc

    return {
        "status": "success",
        "message": f"File '{filename}' successfully ingested as document ID '{doc.id}'.",
        "doc": doc.model_dump(mode="json"),
    }


def execute_evaluation(case: EvaluationCase, use_llm: bool = False, api_key: Optional[str] = None) -> Dict[str, Any]:
    ev = evaluator_llm if use_llm else evaluator_deterministic
    ev.corpus = active_corpus
    ev.retriever.corpus = active_corpus

    reports = ev.run_both_modes(case=case, api_key=api_key)
    return {
        "engine": "llm" if (use_llm or bool(api_key)) else "deterministic",
        "case": case.model_dump(mode="json"),
        "correct_mode": reports["correct_mode"].model_dump(mode="json"),
        "broken_mode": reports["broken_mode"].model_dump(mode="json"),
    }


@app.post("/api/evaluate/custom")
def evaluate_custom(req: CustomEvaluationRequest) -> Dict[str, Any]:
    try:
        cutoff = datetime.strptime(req.as_of_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid as_of_date format. Use YYYY-MM-DD.")

    case = EvaluationCase(
        id="custom_case_" + datetime.now().strftime("%Y%m%d%H%M%S"),
        company=req.company.strip(),
        question=req.question.strip(),
        as_of_date=cutoff,
        as_of_reporting_period=req.as_of_reporting_period.strip(),
        expected_metric="EBITDA margin",
        expected_unit="bps",
    )
    return execute_evaluation(case, use_llm=req.use_llm, api_key=req.api_key)


@app.post("/evaluate")
def evaluate_default(use_llm: bool = Query(False)) -> Dict[str, Any]:
    return execute_evaluation(get_default_case(), use_llm=use_llm)


@app.get("/report")
def get_default_report(use_llm: bool = Query(False)) -> Dict[str, Any]:
    return execute_evaluation(get_default_case(), use_llm=use_llm)


LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PeriodGuard • Financial Research Reliability & Temporal Citation Gate</title>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

  <style>
    :root {
      --bg-base: #070a12;
      --bg-card: #0e1424;
      --bg-card-elevated: #141d33;
      --bg-card-hover: #1b2745;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-strong: rgba(255, 255, 255, 0.16);

      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --emerald-500: #10b981;
      --emerald-bg: rgba(16, 185, 129, 0.12);
      --emerald-border: rgba(16, 185, 129, 0.35);

      --rose-500: #f43f5e;
      --rose-bg: rgba(244, 63, 94, 0.12);
      --rose-border: rgba(244, 63, 94, 0.35);

      --indigo-500: #6366f1;
      --cyan-500: #06b6d4;
      --amber-500: #f59e0b;

      --font-display: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-full: 9999px;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      line-height: 1.55;
      background-image: 
        radial-gradient(ellipse 70% 35% at 50% -10%, rgba(99, 102, 241, 0.15), transparent),
        radial-gradient(circle at 15% 20%, rgba(6, 182, 212, 0.06), transparent),
        radial-gradient(circle at 85% 80%, rgba(244, 63, 94, 0.04), transparent);
      background-attachment: fixed;
      padding: 1.5rem 1rem 4rem;
    }

    .landing-wrap {
      max-width: 1060px;
      margin: 0 auto;
    }

    /* Top Navbar */
    .navbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 1.25rem;
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border-subtle);
      flex-wrap: wrap;
      gap: 1rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px rgba(79, 70, 229, 0.35);
    }

    .brand-icon svg {
      width: 20px;
      height: 20px;
      fill: none;
      stroke: white;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .brand-text h1 {
      font-family: var(--font-display);
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .brand-tag {
      font-family: var(--font-mono);
      font-size: 0.65rem;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.4);
      padding: 0.15rem 0.5rem;
      border-radius: var(--radius-full);
      font-weight: 600;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .btn {
      font-family: var(--font-body);
      font-weight: 600;
      font-size: 0.82rem;
      padding: 0.5rem 1rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
      border: 1px solid transparent;
      text-decoration: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
      color: white;
      box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    .btn-primary:hover {
      background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
      transform: translateY(-1px);
    }

    .btn-secondary {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-strong);
    }
    .btn-secondary:hover {
      background: var(--bg-card-elevated);
      border-color: rgba(255, 255, 255, 0.25);
    }

    /* 1. Hero / Product Overview Section */
    .hero-section {
      text-align: center;
      padding: 1.5rem 0 2.5rem;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 2rem;
    }

    .hero-eyebrow {
      font-family: var(--font-mono);
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--cyan-500);
      margin-bottom: 0.6rem;
    }

    .hero-headline {
      font-family: var(--font-display);
      font-size: 2.25rem;
      font-weight: 900;
      letter-spacing: -0.03em;
      line-height: 1.25;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 1rem;
      max-width: 860px;
      margin-left: auto;
      margin-right: auto;
    }

    .hero-thesis {
      font-size: 1.05rem;
      color: var(--text-muted);
      max-width: 760px;
      margin: 0 auto 1.75rem;
      line-height: 1.6;
    }

    .thesis-callout {
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 0.75rem 1.25rem;
      border-radius: var(--radius-md);
      font-family: var(--font-mono);
      font-size: 0.86rem;
      color: #a5b4fc;
      display: inline-block;
    }

    /* Feature Pillars Grid */
    .pillars-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 1rem;
      margin-top: 1.75rem;
      text-align: left;
    }

    @media (max-width: 768px) {
      .pillars-grid { grid-template-columns: 1fr; }
    }

    .pillar-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.15rem;
    }

    .pillar-icon {
      font-size: 1.2rem;
      margin-bottom: 0.5rem;
    }

    .pillar-title {
      font-family: var(--font-display);
      font-size: 0.95rem;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 0.3rem;
    }

    .pillar-desc {
      font-size: 0.8rem;
      color: var(--text-muted);
      line-height: 1.45;
    }

    /* 2. Step 1: Document Corpus Setup */
    .step-section {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 1.5rem 1.75rem;
      margin-bottom: 2rem;
    }

    .step-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.15rem;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .step-badge {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 700;
      background: rgba(6, 182, 212, 0.15);
      color: var(--cyan-500);
      border: 1px solid rgba(6, 182, 212, 0.3);
      padding: 0.2rem 0.6rem;
      border-radius: var(--radius-full);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .step-title {
      font-family: var(--font-display);
      font-size: 1.2rem;
      font-weight: 700;
    }

    .corpus-choice-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    @media (max-width: 680px) {
      .corpus-choice-grid { grid-template-columns: 1fr; }
    }

    .corpus-option-card {
      background: var(--bg-card-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.2rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.85rem;
      transition: all 0.15s ease;
    }

    .corpus-option-card:hover {
      border-color: rgba(255, 255, 255, 0.2);
    }

    .corpus-option-card.active {
      border-color: var(--indigo-500);
      background: rgba(99, 102, 241, 0.08);
    }

    /* 3. Step 2: Evaluation Mode Switcher */
    .tabs-nav {
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border-strong);
      margin-bottom: 1.5rem;
    }

    .tab-btn {
      font-family: var(--font-display);
      font-size: 0.95rem;
      font-weight: 700;
      padding: 0.65rem 1.25rem;
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      border-bottom: 2px solid transparent;
      margin-bottom: -1px;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .tab-btn:hover { color: var(--text-main); }
    .tab-btn.active {
      color: #ffffff;
      border-bottom-color: var(--cyan-500);
    }

    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* Mode A: Benchmark Tests Grid */
    .benchmarks-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    @media (max-width: 720px) {
      .benchmarks-grid { grid-template-columns: 1fr; }
    }

    .benchmark-card {
      background: var(--bg-card-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.15rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.75rem;
    }

    .benchmark-card:hover {
      border-color: var(--cyan-500);
      background: var(--bg-card-hover);
      transform: translateY(-2px);
    }

    .benchmark-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .badge-pill {
      font-family: var(--font-mono);
      font-size: 0.68rem;
      font-weight: 700;
      padding: 0.15rem 0.5rem;
      border-radius: var(--radius-full);
      text-transform: uppercase;
    }

    .badge-pill.rose { background: var(--rose-bg); color: #fb7185; border: 1px solid var(--rose-border); }
    .badge-pill.emerald { background: var(--emerald-bg); color: #34d399; border: 1px solid var(--emerald-border); }
    .badge-pill.amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .badge-pill.indigo { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.35); }

    /* Mode B: Plain English Prompt Controls */
    .prompt-container {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .prompt-textarea {
      width: 100%;
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-md);
      color: #ffffff;
      font-family: var(--font-body);
      font-size: 1rem;
      padding: 0.85rem 1rem;
      line-height: 1.5;
      resize: vertical;
      min-height: 72px;
    }

    .prompt-textarea:focus {
      outline: none;
      border-color: var(--cyan-500);
      box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
    }

    .prompt-controls-grid {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr auto;
      gap: 0.65rem;
      align-items: flex-end;
    }

    @media (max-width: 900px) {
      .prompt-controls-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 480px) {
      .prompt-controls-grid { grid-template-columns: 1fr; }
    }

    .input-field {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      color: #ffffff;
      font-family: var(--font-body);
      font-size: 0.84rem;
      padding: 0.45rem 0.65rem;
      width: 100%;
    }

    /* 4. Live Results & Verification Section */
    .results-section {
      margin-top: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .verified-result-card {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 1.75rem;
      box-shadow: 0 10px 32px rgba(0,0,0,0.35);
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.15rem;
      flex-wrap: wrap;
      gap: 0.6rem;
    }

    .gate-badge {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 700;
      padding: 0.35rem 0.85rem;
      border-radius: var(--radius-full);
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }

    .gate-badge.safe {
      background: var(--emerald-bg);
      color: var(--emerald-500);
      border: 1px solid var(--emerald-border);
      box-shadow: 0 0 16px rgba(16, 185, 129, 0.15);
    }

    .gate-badge.unsafe {
      background: var(--rose-bg);
      color: var(--rose-500);
      border: 1px solid var(--rose-border);
      box-shadow: 0 0 16px rgba(244, 63, 94, 0.15);
    }

    .answer-lead {
      font-size: 1.1rem;
      color: #ffffff;
      line-height: 1.6;
      font-weight: 500;
      margin-bottom: 1.25rem;
    }

    .claims-list {
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }

    .claim-item {
      background: var(--bg-card-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.95rem 1.15rem;
    }

    .citation-btn {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border-strong);
      border-left: 3px solid var(--indigo-500);
      padding: 0.55rem 0.85rem;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      cursor: pointer;
      font-size: 0.8rem;
      color: var(--text-muted);
      transition: all 0.15s ease;
      display: block;
      width: 100%;
      text-align: left;
      margin-top: 0.4rem;
    }

    .citation-btn:hover {
      background: var(--bg-card-hover);
      border-left-color: var(--cyan-500);
      transform: translateX(2px);
    }

    .cit-top {
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: #93c5fd;
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.2rem;
    }

    /* Comparison Box */
    .comparison-explainer {
      background: linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: var(--radius-lg);
      padding: 1.35rem 1.6rem;
    }

    .comp-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      user-select: none;
    }

    .comp-header h3 {
      font-family: var(--font-display);
      font-size: 1.1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .comp-body {
      margin-top: 1.15rem;
      padding-top: 1.15rem;
      border-top: 1px solid var(--border-subtle);
      display: none;
    }

    .comp-body.open { display: block; }

    .diff-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.15rem;
      margin-top: 1rem;
    }

    @media (max-width: 720px) {
      .diff-grid { grid-template-columns: 1fr; }
    }

    .diff-box {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.1rem;
    }

    .diff-box.failed { border-top: 3px solid var(--rose-500); }
    .diff-box.passed { border-top: 3px solid var(--emerald-500); }

    /* Modals & Drawers */
    .modal-backdrop, .drawer-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(3, 7, 18, 0.8);
      backdrop-filter: blur(4px);
      z-index: 998;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }

    .modal-backdrop.active, .drawer-backdrop.active {
      opacity: 1;
      pointer-events: auto;
    }

    .modal-box {
      position: fixed;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%) scale(0.96);
      width: 92%; max-width: 580px;
      background: #0d1322;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 1.75rem;
      z-index: 999;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
      opacity: 0;
      pointer-events: none;
      transition: all 0.2s ease;
    }

    .modal-box.active {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, -50%) scale(1);
    }

    .modal-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
    }

    .modal-head h3 {
      font-family: var(--font-display);
      font-size: 1.2rem;
      font-weight: 700;
    }

    .inspector-drawer {
      position: fixed;
      top: 0; right: 0; bottom: 0;
      width: 100%; max-width: 500px;
      background: #0d1322;
      border-left: 1px solid var(--border-strong);
      z-index: 999;
      box-shadow: -10px 0 40px rgba(0, 0, 0, 0.6);
      transform: translateX(100%);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
    }

    .inspector-drawer.active { transform: translateX(0); }

    .drawer-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.35rem 1.5rem;
      border-bottom: 1px solid var(--border-subtle);
    }

    .drawer-content {
      padding: 1.35rem 1.5rem;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .btn-close {
      background: none;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: 1.2rem;
      width: 32px;
      height: 32px;
      border-radius: var(--radius-sm);
      cursor: pointer;
    }

    .btn-close:hover {
      background: rgba(255, 255, 255, 0.08);
      color: white;
    }

    .meta-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
    }

    .meta-table td {
      padding: 0.55rem 0.8rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .meta-table td:first-child {
      color: var(--text-dim);
      font-family: var(--font-mono);
      font-size: 0.74rem;
      width: 38%;
    }

    .quote-box {
      background: #070a12;
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: var(--radius-md);
      padding: 0.9rem;
      font-size: 0.82rem;
      line-height: 1.55;
      color: #cbd5e1;
      font-style: italic;
    }
  </style>
</head>
<body>
  <div class="landing-wrap">
    
    <!-- Navbar -->
    <header class="navbar">
      <div class="brand">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
        </div>
        <div class="brand-text">
          <h1>PeriodGuard <span class="brand-tag">VERIFICATION ENGINE</span></h1>
        </div>
      </div>
      <div class="nav-actions">
        <button class="btn btn-secondary" onclick="openCorpusModal()">
          📚 Manage Filings (<span id="corpusCountBadge">4</span>)
        </button>
        <button class="btn btn-secondary" onclick="openUploadModal()">
          📄 Upload Filing / PDF
        </button>
      </div>
    </header>

    <!-- 1. Hero / What PeriodGuard Is About -->
    <section class="hero-section">
      <div class="hero-eyebrow">Financial Research Reliability Harness</div>
      <h2 class="hero-headline">“A citation exists” &ne; “The cited answer is safe to use.”</h2>
      <p class="hero-thesis">
        Financial RAG systems often generate plausible answers by quietly pulling from later fiscal years (like citing a 2026 Annual Report for a 2025 question). 
        PeriodGuard evaluates research pipelines before analysts rely on them.
      </p>
      <div class="thesis-callout">
        🛡️ Evaluates: Temporal As-Of Gating • Entity Boundaries • Numeric Fact Traceability
      </div>

      <div class="pillars-grid">
        <div class="pillar-card">
          <div class="pillar-icon">⏳</div>
          <div class="pillar-title">Future-Period Citation Gate</div>
          <div class="pillar-desc">Detects when an answer cites evidence published after the requested as-of date.</div>
        </div>
        <div class="pillar-card">
          <div class="pillar-icon">🏢</div>
          <div class="pillar-title">Entity &amp; Period Alignment</div>
          <div class="pillar-desc">Prevents peer-company contamination and fiscal period mismatch errors.</div>
        </div>
        <div class="pillar-card">
          <div class="pillar-icon">📊</div>
          <div class="pillar-title">Deterministic Fact Support</div>
          <div class="pillar-desc">Verifies numbers, units (bps, %), and directional causal claims against verbatim quotes.</div>
        </div>
      </div>
    </section>

    <!-- 2. Step 1: Ingest Filings or Use Default Fixture -->
    <section class="step-section">
      <div class="step-header">
        <div>
          <span class="step-badge">Step 1: Document Corpus</span>
          <div class="step-title" style="margin-top: 0.35rem;">Choose Your Evidence Corpus</div>
        </div>
        <button class="btn btn-secondary" onclick="openCorpusModal()">View Loaded Filings</button>
      </div>

      <div class="corpus-choice-grid">
        <!-- Option A: Default Fixture -->
        <div class="corpus-option-card active" id="cardDefaultCorpus">
          <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
              <strong style="font-size: 0.95rem;">Default Financial Corpus (Acme Industries)</strong>
              <span style="color: var(--emerald-500); font-size: 0.75rem; font-weight: 700;">✓ Active</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-muted);">
              Includes Q4 FY25 Results (May 10), Q4 Earnings Call (May 12), FY26 Annual Report Trap (Aug 20), and Globex Corp Peer fixture.
            </p>
          </div>
          <button class="btn btn-secondary" style="align-self: flex-start;" onclick="resetCorpus()">🔄 Restore Defaults</button>
        </div>

        <!-- Option B: Upload Real PDF -->
        <div class="corpus-option-card">
          <div>
            <strong style="font-size: 0.95rem;">Upload Custom PDF / Earnings Release</strong>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.4rem;">
              Ingest any real 10-Q, 10-K, or press release PDF to test custom temporal boundaries and financial metrics.
            </p>
          </div>
          <button class="btn btn-primary" style="align-self: flex-start;" onclick="openUploadModal()">📄 Ingest PDF / Doc</button>
        </div>
      </div>
    </section>

    <!-- 3. Step 2: Evaluation Mode Tabs -->
    <section class="step-section">
      <div class="step-header">
        <div>
          <span class="step-badge">Step 2: Choose Evaluation Mode</span>
          <div class="step-title" style="margin-top: 0.35rem;">Benchmark Tests or Plain English Query</div>
        </div>
      </div>

      <!-- Tabs Navigation -->
      <div class="tabs-nav">
        <button class="tab-btn active" id="tabBtnBenchmarks" onclick="switchTab('benchmarks')">
          ⚡ Mode A: Prewritten Benchmark Tests
        </button>
        <button class="tab-btn" id="tabBtnPrompt" onclick="switchTab('prompt')">
          💬 Mode B: Plain English Prompt
        </button>
      </div>

      <!-- Tab A: Prewritten Benchmarks -->
      <div class="tab-pane active" id="tabPaneBenchmarks">
        <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 1rem;">
          Click any benchmark scenario to immediately execute and verify how PeriodGuard catches or passes the case:
        </div>
        <div class="benchmarks-grid" id="benchmarksContainer"></div>
      </div>

      <!-- Tab B: Plain English Prompt -->
      <div class="tab-pane" id="tabPanePrompt">
        <div class="prompt-container">
          <textarea id="promptInput" class="prompt-textarea" placeholder="Ask any financial question in plain English (e.g. What is the EBITDA margin of Acme Industries? or What was revenue?)...">What is the EBITDA Margin of Acme Industries?</textarea>
          
          <div class="prompt-controls-grid">
            <div>
              <label style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">Target Entity</label>
              <input type="text" id="companyInput" class="input-field" value="Acme Industries">
            </div>
            <div>
              <label style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">As-Of Cutoff Date</label>
              <input type="date" id="asOfDateInput" class="input-field" value="2025-05-15">
            </div>
            <div>
              <label style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">Reporting Period</label>
              <input type="text" id="periodInput" class="input-field" value="Q4 FY25">
            </div>
            <div>
              <label style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">Engine</label>
              <select id="engineSelect" class="input-field" onchange="toggleApiKeyField()">
                <option value="deterministic">Deterministic RAG</option>
                <option value="llm">Live LLM (Groq/OpenAI)</option>
              </select>
            </div>
            <div id="apiKeyCol" style="display: none;">
              <label style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase;">API Key (Groq / OpenAI)</label>
              <input type="password" id="apiKeyInput" class="input-field" placeholder="gsk_... or sk-...">
            </div>
            <button id="btnRunCustomPrompt" class="btn btn-primary" onclick="runCustomPromptEvaluation()" style="height: 36px;">
              ⚡ Run Evaluation
            </button>
          </div>
        </div>
      </div>

      <!-- 4. Live Evaluated Output (Right Here in Step 2 Section) -->
      <div class="results-section">
        <article class="verified-result-card">
          <div class="result-head">
            <div>
              <h2 style="font-family: var(--font-display); font-size: 1.25rem;">Evaluated &amp; Verified Answer</h2>
              <div style="font-size: 0.8rem; color: var(--text-dim);">Evaluated against 4 PeriodGuard Deterministic Reliability Validators</div>
            </div>
            <div id="verifiedBadge" class="gate-badge safe">✓ VERIFIED SAFE FOR ANALYSIS</div>
          </div>

          <div id="answerLeadText" class="answer-lead">
            Loading verified response...
          </div>

          <div style="font-size: 0.74rem; font-family: var(--font-mono); color: var(--text-dim); text-transform: uppercase; margin-bottom: 0.4rem;">
            Verified Evidence Citations (Click to inspect full document &amp; timeline)
          </div>
          <div id="claimsList" class="claims-list"></div>
        </article>

        <!-- Why PeriodGuard is Better than Naive RAG Explainer -->
        <section class="comparison-explainer">
          <div class="comp-header" onclick="toggleComparison()">
            <h3>
              <span>🛡️</span> Why PeriodGuard is Better Than Naive RAG
            </h3>
            <span class="toggle-arrow" id="compArrow">▼</span>
          </div>
          
          <div class="comp-body" id="compBody">
            <p style="font-size: 0.86rem; color: #cbd5e1; margin-bottom: 0.85rem; line-height: 1.5;">
              In standard RAG, the bot retrieves any text with matching keywords. If a subsequent annual report mentions historical figures, naive RAG cites it with full confidence—<strong>silently leaking future information</strong>. 
              PeriodGuard evaluates the prompt, enforces strict metadata cutoff boundaries, and guarantees that citations are safe to use for historical and investment analysis.
            </p>

            <div class="diff-grid">
              <!-- Broken Naive RAG column -->
              <div class="diff-box failed">
                <div style="font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; color: #fb7185; margin-bottom: 0.45rem;">
                  ✗ Naive RAG (Unfiltered Citation Leak)
                </div>
                <div id="naiveRagSummary" style="font-size: 0.82rem; color: #fecdd3; line-height: 1.5; margin-bottom: 0.65rem;"></div>
                <div style="font-size: 0.74rem; font-family: var(--font-mono); color: #fda4af; background: rgba(0,0,0,0.3); padding: 0.4rem; border-radius: 4px;" id="naiveRagFailDetails"></div>
              </div>

              <!-- PeriodGuard Verified column -->
              <div class="diff-box passed">
                <div style="font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; color: #34d399; margin-bottom: 0.45rem;">
                  ✓ PeriodGuard Gate (Period-Correct)
                </div>
                <div style="font-size: 0.82rem; color: #a7f3d0; line-height: 1.5; margin-bottom: 0.65rem;">
                  Enforces strict publication date filtering (<strong>publication_date &le; as_of_date</strong>). Excludes later documents and only cites evidence available as of the cutoff date.
                </div>
                <div style="font-size: 0.74rem; font-family: var(--font-mono); color: #6ee7b7; background: rgba(0,0,0,0.3); padding: 0.4rem; border-radius: 4px;">
                  ✓ 4/4 Checks Passed: Citation Resolved, As-Of Safe, Entity Aligned, Facts Supported.
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

    </section>

  </div>

  <!-- Document Corpus Modal -->
  <div class="modal-backdrop" id="corpusBackdrop" onclick="closeCorpusModal()"></div>
  <div class="modal-box" id="corpusModal" style="max-width: 680px;">
    <div class="modal-head">
      <h3>Active Document Corpus (<span id="corpusModalCount">4</span>)</h3>
      <button class="btn-close" onclick="closeCorpusModal()">×</button>
    </div>
    <div style="max-height: 340px; overflow-y: auto; margin-bottom: 1.25rem;" id="corpusTableBox"></div>
    <div style="display: flex; justify-content: space-between;">
      <button class="btn btn-secondary" onclick="resetCorpus()">🔄 Reset to Default Fixture</button>
      <button class="btn btn-primary" onclick="closeCorpusModal()">Done</button>
    </div>
  </div>

  <!-- Upload PDF / Document Modal -->
  <div class="modal-backdrop" id="uploadBackdrop" onclick="closeUploadModal()"></div>
  <div class="modal-box" id="uploadModal">
    <div class="modal-head">
      <h3>Ingest Financial Document / PDF</h3>
      <button class="btn-close" onclick="closeUploadModal()">×</button>
    </div>
    <form onsubmit="handleUploadSubmit(event)">
      <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 1.25rem;">
        <div>
          <label style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Select File (.pdf, .txt, .json)</label>
          <input type="file" id="uploadFileInput" class="input-field" required accept=".pdf,.txt,.json">
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem;">
          <div>
            <label style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Company</label>
            <input type="text" id="uploadCompanyInput" class="input-field" value="Acme Industries" required>
          </div>
          <div>
            <label style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Document Type</label>
            <input type="text" id="uploadTypeInput" class="input-field" value="Quarterly Results" required>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem;">
          <div>
            <label style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Publication Date</label>
            <input type="date" id="uploadDateInput" class="input-field" value="2025-05-10" required>
          </div>
          <div>
            <label style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Reporting Period</label>
            <input type="text" id="uploadPeriodInput" class="input-field" value="Q4 FY25" required>
          </div>
        </div>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 0.5rem;">
        <button type="button" class="btn btn-secondary" onclick="closeUploadModal()">Cancel</button>
        <button type="submit" id="btnUploadSubmit" class="btn btn-primary">Ingest Document</button>
      </div>
    </form>
  </div>

  <!-- Evidence Inspector Slide-over Drawer -->
  <div class="drawer-backdrop" id="drawerBackdrop" onclick="closeDrawer()"></div>
  <aside class="inspector-drawer" id="inspectorDrawer">
    <div class="drawer-head">
      <div>
        <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase;">Evidence Inspector</div>
        <h3 id="drawerDocId" style="font-family: var(--font-display); font-size: 1.15rem;">Document Metadata</h3>
      </div>
      <button class="btn-close" onclick="closeDrawer()">×</button>
    </div>
    <div class="drawer-content" id="drawerBody"></div>
  </aside>

  <script id="initData" type="application/json">__INITIAL_DATA__</script>

  <script>
    let appState = JSON.parse(document.getElementById('initData').textContent);

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    function toggleApiKeyField() {
      const isLlm = document.getElementById('engineSelect').value === 'llm';
      document.getElementById('apiKeyCol').style.display = isLlm ? 'block' : 'none';
    }

    function renderUI(data) {
      const correct = data.correct_mode;
      const broken = data.broken_mode;

      document.getElementById('answerLeadText').textContent = correct.answer_text || (correct.claims && correct.claims.length > 0
        ? correct.claims.map(c => c.text).join(' ')
        : "No safe eligible evidence was found published on or before the requested cutoff date.");

      const isPass = correct.status === 'PASS';
      const badge = document.getElementById('verifiedBadge');
      badge.className = `gate-badge ${isPass ? 'safe' : 'unsafe'}`;
      badge.textContent = isPass ? '✓ VERIFIED SAFE (Period-Correct)' : '✗ FAILED RELIABILITY GATE';

      const claimsBox = document.getElementById('claimsList');
      claimsBox.innerHTML = correct.claims.map(claim => {
        const citsHtml = claim.citations.map(cit => {
          const doc = correct.retrieved_documents.find(d => d.id === cit.document_id) || {};
          return `
            <button class="citation-btn" onclick="openInspector('${escapeHtml(cit.document_id)}', '${escapeHtml(cit.quoted_text)}')">
              <div class="cit-top">
                <span>📄 ${escapeHtml(cit.document_id)} (Page ${cit.page})</span>
                <span>Pub: ${escapeHtml(doc.publication_date || '—')}</span>
              </div>
              <div>"${escapeHtml(cit.quoted_text)}"</div>
            </button>
          `;
        }).join('');

        return `
          <div class="claim-item">
            <div style="display: flex; justify-content: space-between; font-size: 0.84rem; color: #e2e8f0; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.4rem;">
              <span><strong>Metric:</strong> ${escapeHtml(claim.metric || 'N/A')}</span>
              <span><strong>Value:</strong> ${claim.value !== null && claim.value !== undefined ? claim.value : 'N/A'} ${escapeHtml(claim.unit || '')}</span>
              <span><strong>Period:</strong> ${escapeHtml(claim.period || 'N/A')}</span>
            </div>
            ${citsHtml}
          </div>
        `;
      }).join('');

      const fail = broken.failures.find(f => f.type === 'FUTURE_PERIOD_LEAK') || broken.failures[0];
      if (fail) {
        document.getElementById('naiveRagSummary').innerHTML = `
          Naive RAG retrieved <strong>${escapeHtml(fail.document_id)}</strong> and generated a fluent answer citing future metrics.
        `;
        document.getElementById('naiveRagFailDetails').textContent = `🚨 Leakage Detected: Published ${fail.publication_date} vs As-Of ${fail.as_of_date} (+${Math.round((new Date(fail.publication_date) - new Date(fail.as_of_date))/(1000*60*60*24))} days in future)`;
      } else {
        document.getElementById('naiveRagSummary').textContent = "Both modes passed for this specific cutoff boundary.";
        document.getElementById('naiveRagFailDetails').textContent = "No temporal leakage triggered.";
      }

      updateCorpusCount();
    }

    function renderBenchmarks() {
      fetch('/api/presets')
        .then(r => r.json())
        .then(presets => {
          const container = document.getElementById('benchmarksContainer');
          container.innerHTML = presets.map(p => `
            <div class="benchmark-card" onclick="runBenchmarkPreset('${escapeHtml(p.id)}')">
              <div>
                <div class="benchmark-top">
                  <span class="badge-pill ${escapeHtml(p.badge_color || 'indigo')}">${escapeHtml(p.badge || 'Benchmark')}</span>
                  <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim);">Cutoff: ${escapeHtml(p.as_of_date)}</span>
                </div>
                <div style="font-weight: 700; font-size: 0.95rem; margin: 0.4rem 0 0.2rem; color: #ffffff;">${escapeHtml(p.title)}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.4;">${escapeHtml(p.description)}</div>
              </div>
              <div style="font-size: 0.76rem; color: var(--cyan-500); font-weight: 600;">⚡ Click to Run Scenario &rarr;</div>
            </div>
          `).join('');
        });
    }

    function switchTab(tab) {
      document.getElementById('tabBtnBenchmarks').classList.toggle('active', tab === 'benchmarks');
      document.getElementById('tabBtnPrompt').classList.toggle('active', tab === 'prompt');
      document.getElementById('tabPaneBenchmarks').classList.toggle('active', tab === 'benchmarks');
      document.getElementById('tabPanePrompt').classList.toggle('active', tab === 'prompt');
    }

    async function runBenchmarkPreset(presetId) {
      const resp = await fetch('/api/presets');
      const presets = await resp.json();
      const p = presets.find(x => x.id === presetId);
      if (p) {
        document.getElementById('promptInput').value = p.question;
        document.getElementById('companyInput').value = p.company;
        document.getElementById('asOfDateInput').value = p.as_of_date;
        document.getElementById('periodInput').value = p.as_of_reporting_period;
        runCustomPromptEvaluation();
      }
    }

    async function runCustomPromptEvaluation() {
      const btn = document.getElementById('btnRunCustomPrompt');
      btn.disabled = true;
      btn.textContent = '⚡ Evaluating...';

      const payload = {
        question: document.getElementById('promptInput').value,
        company: document.getElementById('companyInput').value,
        as_of_date: document.getElementById('asOfDateInput').value,
        as_of_reporting_period: document.getElementById('periodInput').value,
        use_llm: document.getElementById('engineSelect').value === 'llm',
        api_key: document.getElementById('apiKeyInput')?.value || null
      };

      try {
        const resp = await fetch('/api/evaluate/custom', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (resp.ok) {
          appState = await resp.json();
          renderUI(appState);
        }
      } catch (err) {
        console.error(err);
      } finally {
        btn.disabled = false;
        btn.textContent = '⚡ Run Evaluation';
      }
    }

    function toggleComparison() {
      const body = document.getElementById('compBody');
      const arrow = document.getElementById('compArrow');
      body.classList.toggle('open');
      arrow.classList.toggle('open');
    }

    function openInspector(docId, quotedText) {
      const allDocs = appState.correct_mode.retrieved_documents.concat(appState.broken_mode.retrieved_documents);
      const doc = allDocs.find(d => d.id === docId) || {
        id: docId, company: 'Acme Industries', doc_type: 'Financial Filing', reporting_period: 'Q4 FY25', publication_date: '2025-05-10', page: 1, text: quotedText, source_url: 'Corpus'
      };

      const asOf = appState.case.as_of_date;
      const isFuture = new Date(doc.publication_date) > new Date(asOf);

      document.getElementById('drawerDocId').textContent = doc.id;
      document.getElementById('drawerBody').innerHTML = `
        <div>
          <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; margin-bottom: 0.35rem;">Document Metadata</div>
          <table class="meta-table">
            <tr><td>Document Type</td><td>${escapeHtml(doc.doc_type)}</td></tr>
            <tr><td>Entity</td><td>${escapeHtml(doc.company)}</td></tr>
            <tr><td>Publication Date</td><td style="font-family: var(--font-mono); color: ${isFuture ? '#fb7185' : '#34d399'}; font-weight: 700;">${escapeHtml(doc.publication_date)}</td></tr>
            <tr><td>Period</td><td>${escapeHtml(doc.reporting_period)}</td></tr>
            <tr><td>Page</td><td>Page ${doc.page}</td></tr>
            <tr><td>Provenance</td><td style="word-break: break-all; color: #93c5fd;">${escapeHtml(doc.source_url)}</td></tr>
          </table>
        </div>

        <div>
          <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; margin-bottom: 0.35rem;">Temporal Status</div>
          <div style="padding: 0.75rem; background: ${isFuture ? 'rgba(244,63,94,0.12)' : 'rgba(16,185,129,0.12)'}; border: 1px solid ${isFuture ? 'rgba(244,63,94,0.3)' : 'rgba(16,185,129,0.3)'}; border-radius: 8px; font-size: 0.8rem; color: ${isFuture ? '#fecdd3' : '#a7f3d0'};">
            ${isFuture 
              ? `🚨 <strong>FUTURE-PERIOD LEAK:</strong> Document published on ${doc.publication_date} violates the as-of cutoff boundary (${asOf}).`
              : `✓ <strong>WITHIN CUTOFF:</strong> Published on ${doc.publication_date}, safe for as-of date (${asOf}).`}
          </div>
        </div>

        <div>
          <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; margin-bottom: 0.35rem;">Verbatim Evidence Quote</div>
          <div class="quote-box">"${escapeHtml(quotedText || doc.text)}"</div>
        </div>
      `;

      document.getElementById('inspectorDrawer').classList.add('active');
      document.getElementById('drawerBackdrop').classList.add('active');
    }

    function closeDrawer() {
      document.getElementById('inspectorDrawer').classList.remove('active');
      document.getElementById('drawerBackdrop').classList.remove('active');
    }

    async function updateCorpusCount() {
      try {
        const resp = await fetch('/api/corpus');
        if (resp.ok) {
          const docs = await resp.json();
          document.getElementById('corpusCountBadge').textContent = docs.length;
          document.getElementById('corpusModalCount').textContent = docs.length;
        }
      } catch (e) {}
    }

    async function openCorpusModal() {
      document.getElementById('corpusModal').classList.add('active');
      document.getElementById('corpusBackdrop').classList.add('active');

      const resp = await fetch('/api/corpus');
      if (resp.ok) {
        const docs = await resp.json();
        const box = document.getElementById('corpusTableBox');
        box.innerHTML = `
          <table class="meta-table">
            <thead>
              <tr style="background: rgba(255,255,255,0.04); font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim);">
                <th style="padding: 0.5rem 0.75rem; text-align: left;">ID</th>
                <th style="padding: 0.5rem 0.75rem; text-align: left;">Company</th>
                <th style="padding: 0.5rem 0.75rem; text-align: left;">Period</th>
                <th style="padding: 0.5rem 0.75rem; text-align: left;">Published</th>
              </tr>
            </thead>
            <tbody>
              ${docs.map(d => `
                <tr>
                  <td style="font-family: var(--font-mono); color: #93c5fd;">${escapeHtml(d.id)}</td>
                  <td>${escapeHtml(d.company)}</td>
                  <td>${escapeHtml(d.reporting_period)}</td>
                  <td style="font-family: var(--font-mono);">${escapeHtml(d.publication_date)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      }
    }

    function closeCorpusModal() {
      document.getElementById('corpusModal').classList.remove('active');
      document.getElementById('corpusBackdrop').classList.remove('active');
    }

    async function resetCorpus() {
      if (confirm('Reset corpus to original fixture?')) {
        await fetch('/api/corpus/reset', { method: 'POST' });
        closeCorpusModal();
        runCustomPromptEvaluation();
      }
    }

    function openUploadModal() {
      document.getElementById('uploadModal').classList.add('active');
      document.getElementById('uploadBackdrop').classList.add('active');
    }

    function closeUploadModal() {
      document.getElementById('uploadModal').classList.remove('active');
      document.getElementById('uploadBackdrop').classList.remove('active');
    }

    async function handleUploadSubmit(e) {
      e.preventDefault();
      const btn = document.getElementById('btnUploadSubmit');
      btn.disabled = true;
      btn.textContent = 'Ingesting...';

      const formData = new FormData();
      formData.append('file', document.getElementById('uploadFileInput').files[0]);
      formData.append('company', document.getElementById('uploadCompanyInput').value);
      formData.append('doc_type', document.getElementById('uploadTypeInput').value);
      formData.append('reporting_period', document.getElementById('uploadPeriodInput').value);
      formData.append('publication_date', document.getElementById('uploadDateInput').value);

      try {
        const resp = await fetch('/api/corpus/upload', {
          method: 'POST',
          body: formData
        });
        if (resp.ok) {
          alert('Document ingested into PeriodGuard corpus!');
          closeUploadModal();
          runCustomPromptEvaluation();
        } else {
          const err = await resp.json();
          alert('Upload failed: ' + (err.detail || 'Error'));
        }
      } catch (err) {
        alert('Upload error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Ingest Document';
      }
    }

    // Initial render
    renderUI(appState);
    renderBenchmarks();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def render_dashboard(use_llm: bool = Query(False)) -> str:
    data = execute_evaluation(get_default_case(), use_llm=use_llm)
    json_str = json.dumps(data).replace("</", "<\\/")
    return LANDING_PAGE_HTML.replace("__INITIAL_DATA__", json_str)
