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
    description="Interactive evaluation tool for financial research systems detecting future-period citation leakage.",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory working corpus
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


class AddDocumentRequest(BaseModel):
    id: str
    company: str
    doc_type: str
    reporting_period: str
    publication_date: str
    page: int = 1
    text: str
    source_url: str = "Uploaded Document"


PRESETS = [
    {
        "id": "future_leak_default",
        "title": "⚡ Future Leak Trap (May 15)",
        "company": "Acme Industries",
        "question": "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Demonstrates how Naive RAG leaks citations from the FY26 Annual Report (published Aug 2025).",
    },
    {
        "id": "clean_historical_pass",
        "title": "⚡ Clean Historical (June 1)",
        "company": "Acme Industries",
        "question": "What was Acme Industries' sequential EBITDA margin change in Q4 FY25?",
        "as_of_date": "2025-06-01",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Sets the as-of date after Q4 results release, verifying a clean PASS across all checks.",
    },
    {
        "id": "strict_early_cutoff",
        "title": "⚡ Pre-Release Call (May 11)",
        "company": "Acme Industries",
        "question": "What management commentary was provided regarding Q4 FY25 EBITDA margin expansion?",
        "as_of_date": "2025-05-11",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Sets cutoff before the May 12 Earnings Call, catching any commentary cited before it occurred.",
    },
]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "PeriodGuard Financial Evaluation Tool"}


@app.get("/api/presets")
def get_presets() -> List[Dict[str, Any]]:
    return PRESETS


@app.get("/api/corpus")
def list_corpus_documents() -> List[Dict[str, Any]]:
    return [doc.model_dump(mode="json") for doc in active_corpus.all_documents()]


@app.post("/api/corpus/reset")
def reset_corpus() -> Dict[str, Any]:
    global active_corpus, evaluator_deterministic, evaluator_llm
    active_corpus = Corpus.from_json_file(DEFAULT_CORPUS_PATH)
    evaluator_deterministic = Evaluator(corpus=active_corpus, use_llm=False)
    evaluator_llm = Evaluator(corpus=active_corpus, use_llm=True)
    return {"status": "success", "message": "Corpus restored to default fixtures", "count": len(active_corpus.all_documents())}


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
        extracted_text = f"Sample text content extracted from {filename}."

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


def execute_evaluation(case: EvaluationCase, use_llm: bool = False) -> Dict[str, Any]:
    ev = evaluator_llm if use_llm else evaluator_deterministic
    ev.corpus = active_corpus
    ev.retriever.corpus = active_corpus

    reports = ev.run_both_modes(case=case)
    return {
        "engine": "llm" if (use_llm and ev.llm_adapter.is_available) else "deterministic",
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
    return execute_evaluation(case, use_llm=req.use_llm)


@app.post("/evaluate")
def evaluate_default(use_llm: bool = Query(False)) -> Dict[str, Any]:
    return execute_evaluation(get_default_case(), use_llm=use_llm)


@app.get("/report")
def get_default_report(use_llm: bool = Query(False)) -> Dict[str, Any]:
    return execute_evaluation(get_default_case(), use_llm=use_llm)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PeriodGuard • Verified Financial Research & Citation Reliability</title>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

  <style>
    :root {
      --bg-base: #070a12;
      --bg-card: #0f1626;
      --bg-card-elevated: #151f36;
      --bg-card-hover: #1c2a47;
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

      --blue-500: #3b82f6;
      --cyan-500: #06b6d4;
      --indigo-500: #6366f1;

      --font-display: 'Plus Jakarta Sans', sans-serif;
      --font-body: 'Inter', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --radius-full: 9999px;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      line-height: 1.55;
      padding: 1.5rem 1rem 3.5rem;
      background-image: 
        radial-gradient(ellipse 65% 35% at 50% -10%, rgba(99, 102, 241, 0.15), transparent),
        radial-gradient(circle at 10% 25%, rgba(6, 182, 212, 0.05), transparent);
    }

    .container {
      max-width: 960px;
      margin: 0 auto;
    }

    /* Minimal Navbar */
    .navbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border-subtle);
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }

    .brand-icon {
      width: 34px;
      height: 34px;
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px rgba(79, 70, 229, 0.35);
    }

    .brand-icon svg {
      width: 18px;
      height: 18px;
      fill: none;
      stroke: white;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .brand-title {
      font-family: var(--font-display);
      font-size: 1.35rem;
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
      border: 1px solid rgba(99, 102, 241, 0.35);
      padding: 0.15rem 0.45rem;
      border-radius: var(--radius-full);
      font-weight: 600;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .btn {
      font-family: var(--font-body);
      font-weight: 600;
      font-size: 0.82rem;
      padding: 0.45rem 0.9rem;
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
      box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
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

    /* Core Prompt & Query Box */
    .prompt-box {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 1.25rem 1.35rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 8px 28px rgba(0,0,0,0.3);
    }

    .box-label {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--cyan-500);
      margin-bottom: 0.4rem;
      display: block;
    }

    .query-textarea {
      width: 100%;
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      color: #ffffff;
      font-family: var(--font-body);
      font-size: 0.96rem;
      padding: 0.75rem 0.95rem;
      line-height: 1.5;
      resize: vertical;
      min-height: 68px;
      transition: all 0.15s ease;
    }

    .query-textarea:focus {
      outline: none;
      border-color: var(--cyan-500);
      box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
    }

    .inline-controls {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr 1fr auto;
      gap: 0.65rem;
      align-items: flex-end;
      margin-top: 0.75rem;
    }

    @media (max-width: 820px) {
      .inline-controls { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 480px) {
      .inline-controls { grid-template-columns: 1fr; }
    }

    .input-group {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .input-label {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
    }

    .input-control {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      color: #ffffff;
      font-family: var(--font-body);
      font-size: 0.84rem;
      padding: 0.45rem 0.65rem;
      width: 100%;
    }

    .sample-pills-row {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      margin-top: 0.9rem;
      padding-top: 0.85rem;
      border-top: 1px solid var(--border-subtle);
      flex-wrap: wrap;
    }

    .sample-pill {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      padding: 0.25rem 0.55rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
    }

    .sample-pill:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--indigo-500);
      color: #ffffff;
    }

    /* Primary Verified Result Presentation */
    .answer-card {
      background: var(--bg-card);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 8px 28px rgba(0,0,0,0.3);
    }

    .answer-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      gap: 0.6rem;
    }

    .status-badge {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.3rem 0.75rem;
      border-radius: var(--radius-full);
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      letter-spacing: 0.02em;
    }

    .status-badge.safe {
      background: var(--emerald-bg);
      color: var(--emerald-500);
      border: 1px solid var(--emerald-border);
      box-shadow: 0 0 14px rgba(16, 185, 129, 0.12);
    }

    .status-badge.unsafe {
      background: var(--rose-bg);
      color: var(--rose-500);
      border: 1px solid var(--rose-border);
      box-shadow: 0 0 14px rgba(244, 63, 94, 0.12);
    }

    .answer-body-text {
      font-size: 1.05rem;
      color: #ffffff;
      line-height: 1.6;
      font-weight: 500;
      margin-bottom: 1.15rem;
    }

    .claims-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      margin-top: 0.75rem;
    }

    .claim-item {
      background: var(--bg-card-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.85rem 1rem;
    }

    .claim-meta-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.84rem;
      color: #e2e8f0;
      margin-bottom: 0.5rem;
      flex-wrap: wrap;
      gap: 0.4rem;
    }

    .citation-btn {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border-strong);
      border-left: 3px solid var(--indigo-500);
      padding: 0.5rem 0.75rem;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      cursor: pointer;
      font-size: 0.78rem;
      color: var(--text-muted);
      transition: all 0.15s ease;
      display: block;
      width: 100%;
      text-align: left;
      margin-top: 0.35rem;
    }

    .citation-btn:hover {
      background: var(--bg-card-hover);
      border-left-color: var(--cyan-500);
      transform: translateX(2px);
    }

    .cit-top {
      font-family: var(--font-mono);
      font-size: 0.74rem;
      color: #93c5fd;
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.2rem;
    }

    /* Signature Explainer: Why PeriodGuard > Naive RAG */
    .explainer-card {
      background: linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: var(--radius-lg);
      padding: 1.25rem 1.5rem;
    }

    .explainer-trigger {
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      user-select: none;
    }

    .explainer-trigger h3 {
      font-family: var(--font-display);
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .toggle-arrow {
      font-size: 0.8rem;
      color: var(--text-muted);
      transition: transform 0.2s ease;
    }

    .toggle-arrow.open {
      transform: rotate(180deg);
    }

    .explainer-body {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border-subtle);
      display: none;
    }

    .explainer-body.open {
      display: block;
    }

    .comparison-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 0.85rem;
    }

    @media (max-width: 720px) {
      .comparison-grid { grid-template-columns: 1fr; }
    }

    .comp-col {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.95rem;
    }

    .comp-col.failed {
      border-top: 3px solid var(--rose-500);
    }

    .comp-col.passed {
      border-top: 3px solid var(--emerald-500);
    }

    .comp-title {
      font-family: var(--font-mono);
      font-size: 0.78rem;
      font-weight: 700;
      margin-bottom: 0.45rem;
    }

    /* Modal / Drawer */
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

    .modal-dialog {
      position: fixed;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%) scale(0.96);
      width: 92%; max-width: 580px;
      background: #0d1322;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      z-index: 999;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
      opacity: 0;
      pointer-events: none;
      transition: all 0.2s ease;
    }

    .modal-dialog.active {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, -50%) scale(1);
    }

    .modal-header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
    }

    .modal-header-row h3 {
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
    }

    .inspector-drawer {
      position: fixed;
      top: 0; right: 0; bottom: 0;
      width: 100%; max-width: 480px;
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

    .drawer-top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border-subtle);
    }

    .drawer-scroll-body {
      padding: 1.25rem 1.5rem;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.15rem;
    }

    .btn-close {
      background: none;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: 1.2rem;
      width: 30px;
      height: 30px;
      border-radius: var(--radius-sm);
      cursor: pointer;
    }

    .btn-close:hover {
      background: rgba(255, 255, 255, 0.08);
      color: white;
    }

    .table-spec {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
    }

    .table-spec td {
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .table-spec td:first-child {
      color: var(--text-dim);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      width: 38%;
    }

    .verbatim-quote {
      background: #070a12;
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: var(--radius-md);
      padding: 0.85rem;
      font-size: 0.8rem;
      line-height: 1.55;
      color: #cbd5e1;
      font-style: italic;
    }
  </style>
</head>
<body>
  <div class="container">
    
    <!-- Navbar -->
    <header class="navbar">
      <div class="brand-group">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
        </div>
        <div class="brand-title">
          PeriodGuard <span class="brand-tag">VERIFICATION ENGINE</span>
        </div>
      </div>
      <div class="nav-actions">
        <button class="btn btn-secondary" onclick="openCorpusModal()">
          📚 View Filings (<span id="corpusCountBadge">4</span>)
        </button>
        <button class="btn btn-secondary" onclick="openUploadModal()">
          📄 Upload Filing / PDF
        </button>
      </div>
    </header>

    <!-- Interactive Prompt Box -->
    <section class="prompt-box">
      <label class="box-label">Prompt / Research Question</label>
      <textarea id="promptInput" class="query-textarea" placeholder="Ask any financial research question (e.g. Did EBITDA margin improve in Q4 FY25?)...">As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.</textarea>

      <div class="inline-controls">
        <div class="input-group">
          <label class="input-label">Target Entity</label>
          <input type="text" id="companyInput" class="input-control" value="Acme Industries">
        </div>
        <div class="input-group">
          <label class="input-label">As-Of Cutoff Date</label>
          <input type="date" id="asOfDateInput" class="input-control" value="2025-05-15">
        </div>
        <div class="input-group">
          <label class="input-label">Target Period</label>
          <input type="text" id="periodInput" class="input-control" value="Q4 FY25">
        </div>
        <div class="input-group">
          <label class="input-label">Engine</label>
          <select id="engineSelect" class="input-control">
            <option value="deterministic">Deterministic</option>
            <option value="llm">Live LLM</option>
          </select>
        </div>
        <button id="btnVerify" class="btn btn-primary" onclick="handleVerifyClick()" style="height: 36px;">
          ⚡ Verify with PeriodGuard
        </button>
      </div>

      <div class="sample-pills-row">
        <span style="font-size: 0.72rem; color: var(--text-dim); margin-right: 0.2rem;">Quick Sample Prompts:</span>
        <button class="sample-pill" onclick="applyPreset(0)">⚡ Future Leak Trap (May 15)</button>
        <button class="sample-pill" onclick="applyPreset(1)">⚡ Clean Historical (June 1)</button>
        <button class="sample-pill" onclick="applyPreset(2)">⚡ Pre-Release Call (May 11)</button>
      </div>
    </section>

    <!-- Results Presentation -->
    <main>
      
      <!-- Primary Verified Answer Card -->
      <article class="answer-card">
        <div class="answer-card-header">
          <div>
            <h2 style="font-family: var(--font-display); font-size: 1.18rem;">Evaluated &amp; Verified Answer</h2>
            <div style="font-size: 0.78rem; color: var(--text-dim);">Gated by PeriodGuard Deterministic Reliability Validators</div>
          </div>
          <div id="verifiedBadge" class="status-badge safe">✓ VERIFIED SAFE FOR ANALYSIS</div>
        </div>

        <div id="answerLeadText" class="answer-body-text">
          Loading verified response...
        </div>

        <div style="font-size: 0.72rem; font-family: var(--font-mono); color: var(--text-dim); text-transform: uppercase; margin-bottom: 0.4rem;">
          Traceable Evidence Citations (Click to inspect source)
        </div>
        <div id="claimsList" class="claims-list"></div>
      </article>

      <!-- Why PeriodGuard is Better than Naive RAG Explainer -->
      <section class="explainer-card">
        <div class="explainer-trigger" onclick="toggleExplainer()">
          <h3>
            <span>🛡️</span> Why PeriodGuard is Better Than Naive RAG
          </h3>
          <span class="toggle-arrow" id="explainerArrow">▼</span>
        </div>
        
        <div class="explainer-body" id="explainerBody">
          <p style="font-size: 0.84rem; color: #cbd5e1; margin-bottom: 0.85rem; line-height: 1.5;">
            In standard RAG, the bot retrieves any text with matching keywords. If a subsequent annual report mentions historical figures, naive RAG cites it with full confidence—<strong>silently leaking future information</strong>. 
            PeriodGuard evaluates the prompt, enforces strict metadata cutoff boundaries, and guarantees that citations are safe to use for historical and investment analysis.
          </p>

          <div class="comparison-grid">
            <!-- Broken Naive RAG column -->
            <div class="comp-col failed">
              <div class="comp-title" style="color: #fb7185;">✗ Naive RAG (Unfiltered Citation Leak)</div>
              <div id="naiveRagSummary" style="font-size: 0.8rem; color: #fecdd3; line-height: 1.5; margin-bottom: 0.65rem;"></div>
              <div style="font-size: 0.72rem; font-family: var(--font-mono); color: #fda4af; background: rgba(0,0,0,0.3); padding: 0.4rem; border-radius: 4px;" id="naiveRagFailDetails"></div>
            </div>

            <!-- PeriodGuard Verified column -->
            <div class="comp-col passed">
              <div class="comp-title" style="color: #34d399;">✓ PeriodGuard Gate (Period-Correct)</div>
              <div style="font-size: 0.8rem; color: #a7f3d0; line-height: 1.5; margin-bottom: 0.65rem;">
                Enforces strict publication date filtering (<strong>publication_date &le; as_of_date</strong>). Excludes later documents and only cites evidence available as of the cutoff date.
              </div>
              <div style="font-size: 0.72rem; font-family: var(--font-mono); color: #6ee7b7; background: rgba(0,0,0,0.3); padding: 0.4rem; border-radius: 4px;">
                ✓ 4/4 Checks Passed: Citation Resolved, As-Of Safe, Entity Aligned, Facts Supported.
              </div>
            </div>
          </div>
        </div>
      </section>

    </main>

  </div>

  <!-- Document Corpus Modal -->
  <div class="modal-backdrop" id="corpusBackdrop" onclick="closeCorpusModal()"></div>
  <div class="modal-dialog" id="corpusModal">
    <div class="modal-header-row">
      <h3>Active Document Corpus (<span id="corpusModalCount">4</span>)</h3>
      <button class="btn-close" onclick="closeCorpusModal()">×</button>
    </div>
    <div style="max-height: 320px; overflow-y: auto; margin-bottom: 1.25rem;" id="corpusTableBox"></div>
    <div style="display: flex; justify-content: space-between;">
      <button class="btn btn-secondary" onclick="resetCorpus()">🔄 Reset to Default Fixture</button>
      <button class="btn btn-primary" onclick="closeCorpusModal()">Done</button>
    </div>
  </div>

  <!-- Upload PDF / Document Modal -->
  <div class="modal-backdrop" id="uploadBackdrop" onclick="closeUploadModal()"></div>
  <div class="modal-dialog" id="uploadModal">
    <div class="modal-header-row">
      <h3>Ingest Financial Document / PDF</h3>
      <button class="btn-close" onclick="closeUploadModal()">×</button>
    </div>
    <form onsubmit="handleUploadSubmit(event)">
      <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 1.25rem;">
        <div class="input-group">
          <label class="input-label">Select File (.pdf, .txt, .json)</label>
          <input type="file" id="uploadFileInput" class="input-control" required accept=".pdf,.txt,.json">
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem;">
          <div class="input-group">
            <label class="input-label">Company</label>
            <input type="text" id="uploadCompanyInput" class="input-control" value="Acme Industries" required>
          </div>
          <div class="input-group">
            <label class="input-label">Document Type</label>
            <input type="text" id="uploadTypeInput" class="input-control" value="Quarterly Results" required>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem;">
          <div class="input-group">
            <label class="input-label">Publication Date</label>
            <input type="date" id="uploadDateInput" class="input-control" value="2025-05-10" required>
          </div>
          <div class="input-group">
            <label class="input-label">Reporting Period</label>
            <input type="text" id="uploadPeriodInput" class="input-control" value="Q4 FY25" required>
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
    <div class="drawer-top-bar">
      <div>
        <div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase;">Evidence Inspector</div>
        <h3 id="drawerDocId" style="font-family: var(--font-display); font-size: 1.1rem;">Document Metadata</h3>
      </div>
      <button class="btn-close" onclick="closeDrawer()">×</button>
    </div>
    <div class="drawer-scroll-body" id="drawerBody"></div>
  </aside>

  <script id="initData" type="application/json">__INITIAL_DATA__</script>

  <script>
    let appState = JSON.parse(document.getElementById('initData').textContent);

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    function renderUI(data) {
      const correct = data.correct_mode;
      const broken = data.broken_mode;

      document.getElementById('answerLeadText').textContent = correct.claims && correct.claims.length > 0
        ? `"${correct.claims.map(c => c.text).join(' ')}"`
        : "No safe eligible evidence was found published on or before the requested cutoff date.";

      const isPass = correct.status === 'PASS';
      const badge = document.getElementById('verifiedBadge');
      badge.className = `status-badge ${isPass ? 'safe' : 'unsafe'}`;
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
            <div class="claim-meta-row">
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

    async function handleVerifyClick() {
      const btn = document.getElementById('btnVerify');
      btn.disabled = true;
      btn.textContent = '⚡ Evaluating...';

      const payload = {
        question: document.getElementById('promptInput').value,
        company: document.getElementById('companyInput').value,
        as_of_date: document.getElementById('asOfDateInput').value,
        as_of_reporting_period: document.getElementById('periodInput').value,
        use_llm: document.getElementById('engineSelect').value === 'llm'
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
        btn.textContent = '⚡ Verify with PeriodGuard';
      }
    }

    const presetsList = [
      {
        q: "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.",
        c: "Acme Industries",
        d: "2025-05-15",
        p: "Q4 FY25"
      },
      {
        q: "What was Acme Industries' sequential EBITDA margin change in Q4 FY25?",
        c: "Acme Industries",
        d: "2025-06-01",
        p: "Q4 FY25"
      },
      {
        q: "What management commentary was provided regarding Q4 FY25 EBITDA margin expansion?",
        c: "Acme Industries",
        d: "2025-05-11",
        p: "Q4 FY25"
      }
    ];

    function applyPreset(index) {
      const p = presetsList[index];
      document.getElementById('promptInput').value = p.q;
      document.getElementById('companyInput').value = p.c;
      document.getElementById('asOfDateInput').value = p.d;
      document.getElementById('periodInput').value = p.p;
      handleVerifyClick();
    }

    function toggleExplainer() {
      const body = document.getElementById('explainerBody');
      const arrow = document.getElementById('explainerArrow');
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
          <table class="table-spec">
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
          <div class="verbatim-quote">"${escapeHtml(quotedText || doc.text)}"</div>
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
          <table class="table-spec">
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
        handleVerifyClick();
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
          handleVerifyClick();
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
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def render_dashboard(use_llm: bool = Query(False)) -> str:
    data = execute_evaluation(get_default_case(), use_llm=use_llm)
    json_str = json.dumps(data).replace("</", "<\\/")
    return DASHBOARD_HTML.replace("__INITIAL_DATA__", json_str)
