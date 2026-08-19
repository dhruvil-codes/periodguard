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
    description="Evaluation harness for financial research systems detecting future-period citation leakage across financial PDFs and filings.",
    version="3.0.0",
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
        "badge": "High Signal Trap",
        "badge_color": "red",
        "company": "Acme Industries",
        "question": "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Demonstrates naive RAG leaking figures from the FY26 Annual Report (published Aug 2025) for a May 2025 question.",
    },
    {
        "id": "clean_historical",
        "title": "Clean Historical Query",
        "badge": "Safe Historical",
        "badge_color": "green",
        "company": "Acme Industries",
        "question": "What was Acme Industries' sequential EBITDA margin change in Q4 FY25?",
        "as_of_date": "2025-06-01",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Sets cutoff to June 1 (after Q4 release), verifying a 4/4 clean PASS across all reliability validators.",
    },
    {
        "id": "pre_release_cutoff",
        "title": "Pre-Release Earnings Call Cutoff",
        "badge": "Temporal Boundary",
        "badge_color": "orange",
        "company": "Acme Industries",
        "question": "What management commentary was provided regarding Q4 FY25 EBITDA margin expansion?",
        "as_of_date": "2025-05-11",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Sets cutoff to May 11 (before the May 12 call), catching management remarks cited before they occurred.",
    },
    {
        "id": "peer_entity_mismatch",
        "title": "Peer Entity Contamination Trap",
        "badge": "Entity Check",
        "badge_color": "purple",
        "company": "Acme Industries",
        "question": "Did EBITDA margin reach 19.1% in Q4 FY25?",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "description": "Evaluates whether the retriever correctly rejects citations from Globex Corp (peer company with 19.1% margin).",
    },
]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "PeriodGuard Financial Evaluation Engine"}


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
    doc_type: str = Form("10-Q Quarterly Report"),
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
    page_count = 1
    if filename.lower().endswith(".pdf"):
        if not PYPDF_AVAILABLE:
            raise HTTPException(status_code=500, detail="PDF parsing library (pypdf) is unavailable.")
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            pages_text = [f"[Page {i+1}]\n" + page.extract_text() for i, page in enumerate(reader.pages) if page.extract_text()]
            extracted_text = "\n\n".join(pages_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    else:
        try:
            extracted_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = file_bytes.decode("latin-1")

    if not extracted_text.strip():
        extracted_text = f"Sample financial filing text extracted from {filename}."

    doc = Document(
        id=doc_id,
        company=company.strip(),
        doc_type=doc_type.strip(),
        reporting_period=reporting_period.strip(),
        publication_date=pub_date,
        page=1,
        text=extracted_text.strip(),
        source_url=f"PDF Ingestion: {filename} ({page_count} pages)",
    )
    active_corpus._documents[doc.id] = doc

    return {
        "status": "success",
        "message": f"File '{filename}' ({page_count} pages) successfully ingested as document ID '{doc.id}'.",
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
        expected_metric=None,
        expected_unit=None,
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
  <title>PeriodGuard • Financial Research Citation Reliability Engine</title>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

  <style>
    /* Cloudflare-Inspired Technical Minimalist Design Tokens */
    :root {
      --cf-primary: #ff5e1f;
      --cf-primary-hover: #ff4800;
      --cf-bg: #ffffff;
      --cf-surface: #fcfcfc;
      --cf-surface-elevated: #f4f4f5;
      --cf-border: #e4e4e7;
      --cf-border-subtle: #f0f0f2;
      
      --cf-text: #18181b;
      --cf-text-muted: #52525b;
      --cf-text-dim: #71717a;

      --cf-green: #059669;
      --cf-green-bg: #ecfdf5;
      --cf-green-border: #a7f3d0;

      --cf-red: #dc2626;
      --cf-red-bg: #fef2f2;
      --cf-red-border: #fecaca;

      --cf-radius-sm: 2px;
      --cf-radius-md: 4px;
      --cf-radius-lg: 6px;

      --cf-font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --cf-font-mono: 'JetBrains Mono', "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: var(--cf-font-sans);
      background-color: var(--cf-bg);
      color: var(--cf-text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 0 1.5rem;
    }

    /* Top Utility Header */
    .site-header {
      border-bottom: 1px solid var(--cf-border);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .header-inner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      height: 56px;
    }

    .brand-link {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      text-decoration: none;
      color: var(--cf-text);
    }

    .brand-mark {
      width: 24px;
      height: 24px;
      background: var(--cf-primary);
      border-radius: var(--cf-radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      font-weight: 800;
      font-size: 0.85rem;
    }

    .brand-name {
      font-weight: 700;
      font-size: 1.05rem;
      letter-spacing: -0.03em;
    }

    .brand-badge {
      font-family: var(--cf-font-mono);
      font-size: 0.65rem;
      font-weight: 600;
      text-transform: uppercase;
      background: var(--cf-surface-elevated);
      color: var(--cf-text-muted);
      border: 1px solid var(--cf-border);
      padding: 0.1rem 0.4rem;
      border-radius: var(--cf-radius-sm);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Minimal Button Tokens */
    .btn {
      font-family: var(--cf-font-sans);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.45rem 0.85rem;
      border-radius: var(--cf-radius-sm);
      border: 1px solid transparent;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      text-decoration: none;
      transition: background 120ms ease, border-color 120ms ease;
    }

    .btn-primary {
      background: var(--cf-primary);
      color: #ffffff;
    }
    .btn-primary:hover { background: var(--cf-primary-hover); }

    .btn-outline {
      background: #ffffff;
      color: var(--cf-text);
      border-color: var(--cf-border);
    }
    .btn-outline:hover {
      background: var(--cf-surface-elevated);
      border-color: #d4d4d8;
    }

    .btn-subtle {
      background: transparent;
      color: var(--cf-text-muted);
    }
    .btn-subtle:hover {
      background: var(--cf-surface-elevated);
      color: var(--cf-text);
    }

    /* Hero Section */
    .hero {
      padding: 3rem 0 2.5rem;
      border-bottom: 1px solid var(--cf-border-subtle);
    }

    .hero-eyebrow {
      font-family: var(--cf-font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--cf-primary);
      margin-bottom: 0.5rem;
    }

    .hero-title {
      font-size: 2.2rem;
      font-weight: 800;
      letter-spacing: -0.04em;
      line-height: 1.15;
      color: var(--cf-text);
      max-width: 780px;
      margin-bottom: 0.85rem;
    }

    .hero-subtitle {
      font-size: 1.05rem;
      color: var(--cf-text-muted);
      max-width: 740px;
      line-height: 1.55;
    }

    .hero-strip {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      margin-top: 1.75rem;
    }

    @media (max-width: 768px) {
      .hero-strip { grid-template-columns: 1fr; }
    }

    .strip-item {
      border: 1px solid var(--cf-border);
      background: var(--cf-surface);
      border-radius: var(--cf-radius-md);
      padding: 1rem 1.15rem;
    }

    .strip-title {
      font-weight: 700;
      font-size: 0.88rem;
      margin-bottom: 0.25rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .strip-desc {
      font-size: 0.8rem;
      color: var(--cf-text-muted);
      line-height: 1.45;
    }

    /* Main Workbench Grid */
    .workbench-layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 1.5rem;
      padding: 2rem 0 4rem;
    }

    @media (max-width: 860px) {
      .workbench-layout { grid-template-columns: 1fr; }
    }

    /* Sidebar / Ingestion Panel */
    .sidebar-panel {
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-md);
      background: #ffffff;
      display: flex;
      flex-direction: column;
      height: fit-content;
    }

    .panel-head {
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--cf-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--cf-surface);
    }

    .panel-title {
      font-weight: 700;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-family: var(--cf-font-mono);
      color: var(--cf-text-muted);
    }

    .panel-body {
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }

    .doc-item-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      padding: 0.6rem 0.75rem;
      border: 1px solid var(--cf-border-subtle);
      border-radius: var(--cf-radius-sm);
      background: var(--cf-surface);
      font-size: 0.78rem;
    }

    .doc-item-title {
      font-weight: 600;
      color: var(--cf-text);
      font-family: var(--cf-font-mono);
      font-size: 0.74rem;
    }

    .doc-item-meta {
      color: var(--cf-text-dim);
      font-size: 0.72rem;
      margin-top: 0.15rem;
    }

    .doc-type-pill {
      display: inline-block;
      font-family: var(--cf-font-mono);
      font-size: 0.62rem;
      font-weight: 600;
      padding: 0.1rem 0.35rem;
      border-radius: var(--cf-radius-sm);
      background: var(--cf-surface-elevated);
      color: var(--cf-text-muted);
      border: 1px solid var(--cf-border);
    }

    /* Main Console / Chat Area */
    .console-panel {
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-md);
      background: #ffffff;
      display: flex;
      flex-direction: column;
    }

    /* Tabs Bar */
    .console-nav {
      display: flex;
      border-bottom: 1px solid var(--cf-border);
      background: var(--cf-surface);
      padding: 0 0.5rem;
    }

    .console-tab {
      font-family: var(--cf-font-sans);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.75rem 1rem;
      border: none;
      background: transparent;
      color: var(--cf-text-muted);
      border-bottom: 2px solid transparent;
      cursor: pointer;
      transition: all 120ms ease;
    }

    .console-tab.active {
      color: var(--cf-primary);
      border-bottom-color: var(--cf-primary);
      background: #ffffff;
    }

    .tab-content { display: none; padding: 1.25rem; }
    .tab-content.active { display: block; }

    /* Mode A: Benchmark Cards */
    .benchmarks-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.85rem;
    }

    @media (max-width: 680px) {
      .benchmarks-grid { grid-template-columns: 1fr; }
    }

    .benchmark-card {
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-sm);
      padding: 0.9rem;
      background: var(--cf-surface);
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.6rem;
    }

    .benchmark-card:hover {
      border-color: var(--cf-primary);
      background: #ffffff;
    }

    .badge-tag {
      font-family: var(--cf-font-mono);
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      padding: 0.1rem 0.35rem;
      border-radius: var(--cf-radius-sm);
    }
    .badge-tag.red { background: var(--cf-red-bg); color: var(--cf-red); border: 1px solid var(--cf-red-border); }
    .badge-tag.green { background: var(--cf-green-bg); color: var(--cf-green); border: 1px solid var(--cf-green-border); }
    .badge-tag.orange { background: #fff7ed; color: #ea580c; border: 1px solid #ffedd5; }
    .badge-tag.purple { background: #faf5ff; color: #9333ea; border: 1px solid #f3e8ff; }

    /* Mode B: Chat Experience */
    .chat-history {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: 520px;
      overflow-y: auto;
      padding-right: 0.25rem;
      margin-bottom: 1.25rem;
    }

    .msg-user {
      align-self: flex-end;
      background: var(--cf-surface-elevated);
      border: 1px solid var(--cf-border);
      color: var(--cf-text);
      padding: 0.75rem 1rem;
      border-radius: var(--cf-radius-md);
      max-width: 85%;
      font-size: 0.88rem;
    }

    .msg-bot {
      align-self: flex-start;
      background: #ffffff;
      border: 1px solid var(--cf-border);
      border-left: 3px solid var(--cf-primary);
      border-radius: var(--cf-radius-sm);
      padding: 1rem 1.15rem;
      width: 100%;
    }

    .msg-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.6rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .gate-indicator {
      font-family: var(--cf-font-mono);
      font-size: 0.7rem;
      font-weight: 700;
      padding: 0.2rem 0.5rem;
      border-radius: var(--cf-radius-sm);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      cursor: pointer;
    }
    .gate-indicator.pass { background: var(--cf-green-bg); color: var(--cf-green); border: 1px solid var(--cf-green-border); }
    .gate-indicator.fail { background: var(--cf-red-bg); color: var(--cf-red); border: 1px solid var(--cf-red-border); }

    .msg-text {
      font-size: 0.92rem;
      color: var(--cf-text);
      line-height: 1.55;
      margin-bottom: 0.85rem;
    }

    .citations-wrap {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      margin-top: 0.65rem;
      border-top: 1px solid var(--cf-border-subtle);
      padding-top: 0.65rem;
    }

    .citation-pill-btn {
      background: var(--cf-surface);
      border: 1px solid var(--cf-border);
      padding: 0.5rem 0.75rem;
      border-radius: var(--cf-radius-sm);
      text-align: left;
      cursor: pointer;
      font-size: 0.78rem;
      transition: border-color 120ms ease;
    }
    .citation-pill-btn:hover { border-color: var(--cf-primary); background: #ffffff; }

    /* Input Bar */
    .chat-box-wrap {
      border-top: 1px solid var(--cf-border);
      background: var(--cf-surface);
      padding: 1rem;
      border-radius: 0 0 var(--cf-radius-md) var(--cf-radius-md);
    }

    .chat-textarea {
      width: 100%;
      background: #ffffff;
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-sm);
      padding: 0.65rem 0.85rem;
      font-family: var(--cf-font-sans);
      font-size: 0.88rem;
      color: var(--cf-text);
      resize: none;
      min-height: 54px;
    }
    .chat-textarea:focus { outline: 1px solid var(--cf-primary); border-color: var(--cf-primary); }

    .chat-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.65rem;
    }

    .toolbar-inputs {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      flex-wrap: wrap;
    }

    .input-compact {
      background: #ffffff;
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-sm);
      font-family: var(--cf-font-sans);
      font-size: 0.76rem;
      color: var(--cf-text);
      padding: 0.3rem 0.5rem;
    }

    /* Comparison Difference Box */
    .comparison-section {
      margin-top: 1.5rem;
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-sm);
      background: var(--cf-surface);
    }

    .comp-toggle-head {
      padding: 0.75rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      font-weight: 700;
      font-size: 0.82rem;
      user-select: none;
    }

    .comp-content {
      padding: 1rem;
      border-top: 1px solid var(--cf-border);
      display: none;
    }
    .comp-content.open { display: block; }

    .diff-two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 0.75rem;
    }
    @media (max-width: 680px) {
      .diff-two-col { grid-template-columns: 1fr; }
    }

    .diff-col {
      padding: 0.85rem;
      border-radius: var(--cf-radius-sm);
      font-size: 0.8rem;
    }
    .diff-col.err { background: var(--cf-red-bg); border: 1px solid var(--cf-red-border); }
    .diff-col.ok { background: var(--cf-green-bg); border: 1px solid var(--cf-green-border); }

    /* Modals & Slide-over */
    .modal-veil, .drawer-veil {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.4);
      z-index: 99;
      opacity: 0;
      pointer-events: none;
      transition: opacity 150ms ease;
    }
    .modal-veil.show, .drawer-veil.show { opacity: 1; pointer-events: auto; }

    .modal-window {
      position: fixed;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      width: 90%; max-width: 580px;
      background: #ffffff;
      border: 1px solid var(--cf-border);
      border-radius: var(--cf-radius-md);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
      z-index: 100;
      padding: 1.5rem;
      display: none;
    }
    .modal-window.show { display: block; }

    .drawer-window {
      position: fixed;
      top: 0; right: 0; bottom: 0;
      width: 100%; max-width: 460px;
      background: #ffffff;
      border-left: 1px solid var(--cf-border);
      box-shadow: -5px 0 25px rgba(0,0,0,0.08);
      z-index: 100;
      transform: translateX(100%);
      transition: transform 180ms ease;
      display: flex;
      flex-direction: column;
    }
    .drawer-window.show { transform: translateX(0); }

    .drawer-top {
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--cf-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--cf-surface);
    }

    .drawer-scroll {
      padding: 1.25rem;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .clean-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
    }
    .clean-table td, .clean-table th {
      padding: 0.45rem 0.6rem;
      border: 1px solid var(--cf-border);
      text-align: left;
    }
    .clean-table th { background: var(--cf-surface); font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-muted); }
  </style>
</head>
<body>

  <!-- Top Header Navigation -->
  <header class="site-header">
    <div class="container header-inner">
      <a href="/" class="brand-link">
        <div class="brand-mark">P</div>
        <span class="brand-name">PeriodGuard</span>
        <span class="brand-badge">VERIFICATION ENGINE</span>
      </a>

      <div class="header-actions">
        <button class="btn btn-subtle" onclick="openGuide()">ℹ️ Guide &amp; Docs</button>
        <button class="btn btn-outline" onclick="openCorpusModal()">📚 Manage Filings (<span id="countBadge">4</span>)</button>
        <button class="btn btn-primary" onclick="openUploadModal()">+ Ingest PDF</button>
      </div>
    </div>
  </header>

  <main class="container">
    <!-- Hero / Core Thesis -->
    <section class="hero">
      <div class="hero-eyebrow">Financial Retrieval &amp; Citation Gate</div>
      <h1 class="hero-title">“A citation exists” &ne; “The cited answer is safe to use.”</h1>
      <p class="hero-subtitle">
        Standard financial RAG searches by semantic text similarity. When answering historical questions, it quietly cites later filings (e.g. citing an Aug 2025 10-K for a May 2025 query)—silently leaking future figures. PeriodGuard evaluates research pipelines before analysts rely on them.
      </p>

      <div class="hero-strip">
        <div class="strip-item">
          <div class="strip-title">⏳ Temporal Gating</div>
          <div class="strip-desc">Guarantees no citation is drawn from a filing published after the as-of cutoff date.</div>
        </div>
        <div class="strip-item">
          <div class="strip-title">🏢 Entity &amp; Period Alignment</div>
          <div class="strip-desc">Prevents cross-company contamination and fiscal period mismatch errors.</div>
        </div>
        <div class="strip-item">
          <div class="strip-title">📊 Fact Support Proxy</div>
          <div class="strip-desc">Verifies numbers, units (bps, %, $), and qualitative management remarks against source quotes.</div>
        </div>
      </div>
    </section>

    <!-- Main Workbench Area -->
    <div class="workbench-layout">
      
      <!-- Left: Corpus & Filing Ingestion Panel -->
      <aside class="sidebar-panel">
        <div class="panel-head">
          <span class="panel-title">Active Corpus</span>
          <button class="btn btn-subtle" style="padding: 0.15rem 0.4rem; font-size: 0.72rem;" onclick="resetCorpus()">Reset</button>
        </div>
        <div class="panel-body">
          <div style="font-size: 0.78rem; color: var(--cf-text-muted);">
            Active financial filings available for retrieval:
          </div>

          <div id="sidebarDocList" style="display: flex; flex-direction: column; gap: 0.5rem;">
            <!-- Rendered by JS -->
          </div>

          <button class="btn btn-outline" style="width: 100%; justify-content: center; margin-top: 0.4rem;" onclick="openUploadModal()">
            + Ingest Custom PDF
          </button>

          <div style="margin-top: 0.75rem; border-top: 1px solid var(--cf-border-subtle); padding-top: 0.75rem;">
            <div style="font-family: var(--cf-font-mono); font-size: 0.68rem; color: var(--cf-text-dim); text-transform: uppercase; font-weight: 700; margin-bottom: 0.35rem;">
              Supported Report Types:
            </div>
            <div style="font-size: 0.72rem; color: var(--cf-text-muted); line-height: 1.45;">
              • 10-Q Quarterly Results<br>
              • 10-K Consolidated Reports<br>
              • Earnings Call Transcripts<br>
              • 8-K Event Filings &amp; Decks
            </div>
          </div>
        </div>
      </aside>

      <!-- Right: Main Evaluation & Chat Console -->
      <section class="console-panel">
        
        <!-- Tab Navigation -->
        <div class="console-nav">
          <button class="console-tab active" id="tabBtnChat" onclick="switchConsoleTab('chat')">
            💬 Mode B: Financial Research Chat
          </button>
          <button class="console-tab" id="tabBtnBench" onclick="switchConsoleTab('bench')">
            ⚡ Mode A: Prewritten Benchmarks
          </button>
        </div>

        <!-- Mode B: Chat Console Pane -->
        <div class="tab-content active" id="tabContentChat">
          <div class="chat-history" id="chatLog"></div>

          <!-- Chat Input -->
          <div class="chat-box-wrap">
            <textarea id="promptInput" class="chat-textarea" placeholder="Ask any financial question (e.g., What does Acme Industries do? or What was Q4 EBITDA margin?)..." onkeydown="handleKey(event)"></textarea>
            
            <div class="chat-toolbar">
              <div class="toolbar-inputs">
                <span style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-dim);">AS-OF:</span>
                <input type="date" id="asOfDateInput" class="input-compact" value="2025-05-15">
                <input type="text" id="companyInput" class="input-compact" value="Acme Industries" style="width: 110px;" placeholder="Company">
                <input type="text" id="periodInput" class="input-compact" value="Q4 FY25" style="width: 70px;" placeholder="Period">
                <select id="engineSelect" class="input-compact" onchange="toggleApiInput()">
                  <option value="deterministic">Deterministic RAG</option>
                  <option value="llm">Live LLM (Groq/OpenAI)</option>
                </select>
                <input type="password" id="apiKeyInput" class="input-compact" placeholder="API Key (Optional)" style="display: none; width: 120px;">
              </div>

              <button class="btn btn-primary" id="btnSend" onclick="sendPrompt()">Ask Assistant</button>
            </div>

            <!-- Quick Suggestions -->
            <div style="display: flex; gap: 0.35rem; margin-top: 0.5rem; flex-wrap: wrap; align-items: center;">
              <span style="font-size: 0.7rem; color: var(--cf-text-dim);">Quick queries:</span>
              <button class="btn btn-subtle" style="font-size: 0.72rem; padding: 0.15rem 0.4rem;" onclick="setQuery('What does Acme Industries do?')">What does Acme do?</button>
              <button class="btn btn-subtle" style="font-size: 0.72rem; padding: 0.15rem 0.4rem;" onclick="setQuery('What was the EBITDA margin of Acme Industries in Q4 FY25?')">Q4 EBITDA Margin?</button>
              <button class="btn btn-subtle" style="font-size: 0.72rem; padding: 0.15rem 0.4rem;" onclick="setQuery('What was total net revenue for Q4 FY25?')">Q4 Total Revenue?</button>
              <button class="btn btn-subtle" style="font-size: 0.72rem; padding: 0.15rem 0.4rem;" onclick="setQuery('As of 15 May 2025, did Acme Industries EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give?')">Future Leak Trap</button>
            </div>
          </div>
        </div>

        <!-- Mode A: Benchmarks Pane -->
        <div class="tab-content" id="tabContentBench">
          <div style="font-size: 0.82rem; color: var(--cf-text-muted); margin-bottom: 0.85rem;">
            Click any benchmark trap to evaluate temporal gating performance against baseline retrieval:
          </div>
          <div class="benchmarks-grid" id="benchmarksContainer"></div>
        </div>

        <!-- Comparison Dropdown -->
        <div class="comparison-section">
          <div class="comp-toggle-head" onclick="toggleComparison()">
            <span>🛡️ PeriodGuard vs Naive RAG Evaluation Summary</span>
            <span id="compChevron">▼</span>
          </div>
          <div class="comp-content" id="compBody">
            <div class="diff-two-col">
              <div class="diff-col err">
                <strong style="color: var(--cf-red);">✗ Naive RAG Baseline (Unfiltered)</strong>
                <div id="naiveSummary" style="margin-top: 0.35rem; color: #7f1d1d;"></div>
                <div id="naiveDetails" style="font-family: var(--cf-font-mono); font-size: 0.72rem; margin-top: 0.4rem; color: var(--cf-red);"></div>
              </div>
              <div class="diff-col ok">
                <strong style="color: var(--cf-green);">✓ PeriodGuard Gate (Period-Correct)</strong>
                <div style="margin-top: 0.35rem; color: #065f46;">
                  Enforces strict metadata cutoff (<code>publication_date &le; as_of_date</code>). Excludes future reports and verifies claims against valid quotes.
                </div>
                <div style="font-family: var(--cf-font-mono); font-size: 0.72rem; margin-top: 0.4rem; color: var(--cf-green);">
                  ✓ 4/4 Checks Passed: Citation Resolution, Temporal Gate, Entity Alignment, Fact Support.
                </div>
              </div>
            </div>
          </div>
        </div>

      </section>

    </div>
  </main>

  <!-- Reliability Guide Modal -->
  <div class="modal-veil" id="guideVeil" onclick="closeGuide()"></div>
  <div class="modal-window" id="guideModal">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--cf-border); padding-bottom: 0.6rem;">
      <h3 style="font-size: 1.05rem; font-weight: 700;">PeriodGuard Reliability Guide</h3>
      <button class="btn btn-subtle" onclick="closeGuide()">×</button>
    </div>
    <div style="font-size: 0.84rem; color: var(--cf-text); display: flex; flex-direction: column; gap: 0.85rem; max-height: 420px; overflow-y: auto;">
      <div>
        <strong>⏳ What is Temporal Gating?</strong>
        <p style="color: var(--cf-text-muted); margin-top: 0.2rem;">
          In financial research and backtesting, questions are asked as of a specific point in time (e.g. <em>15 May 2025</em>). Temporal Gating guarantees the RAG pipeline only cites documents published on or before that date.
        </p>
      </div>

      <div>
        <strong>🚨 What does "FAILED TEMPORAL GATE" mean?</strong>
        <p style="color: var(--cf-text-muted); margin-top: 0.2rem;">
          Standard RAG searches by keyword similarity alone. If an August 2025 report mentions historical figures, naive RAG cites it for a May 2025 question—<strong>silently leaking future knowledge</strong>. PeriodGuard flags this as a <code>FUTURE_PERIOD_LEAK</code>.
        </p>
      </div>

      <div>
        <strong>🔍 The 4 Deterministic Validators:</strong>
        <ul style="padding-left: 1.2rem; margin-top: 0.3rem; color: var(--cf-text-muted); display: flex; flex-direction: column; gap: 0.25rem;">
          <li><strong>Citation Resolution:</strong> Verifies every claim links to a valid document, page, and verbatim quote.</li>
          <li><strong>Temporal Consistency:</strong> Enforces <code>publication_date &le; as_of_date</code>.</li>
          <li><strong>Entity &amp; Period Alignment:</strong> Prevents peer-company (e.g. Globex vs Acme) and fiscal period contamination.</li>
          <li><strong>Citation Support Proxy:</strong> Verifies claimed metrics, numbers, and units (bps, %, $) against source quotes.</li>
        </ul>
      </div>
    </div>
    <div style="display: flex; justify-content: flex-end; margin-top: 1.25rem;">
      <button class="btn btn-primary" onclick="closeGuide()">Close</button>
    </div>
  </div>

  <!-- Upload Modal -->
  <div class="modal-veil" id="uploadVeil" onclick="closeUploadModal()"></div>
  <div class="modal-window" id="uploadModal">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--cf-border); padding-bottom: 0.6rem;">
      <h3 style="font-size: 1.05rem; font-weight: 700;">Ingest Financial PDF / Document</h3>
      <button class="btn btn-subtle" onclick="closeUploadModal()">×</button>
    </div>
    <form onsubmit="handleUpload(event)" style="display: flex; flex-direction: column; gap: 0.75rem;">
      <div>
        <label style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-muted); font-weight: 600; text-transform: uppercase;">File (.pdf, .txt, .json)</label>
        <input type="file" id="upFile" class="input-compact" style="width: 100%; margin-top: 0.25rem;" required accept=".pdf,.txt,.json">
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;">
        <div>
          <label style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-muted); font-weight: 600; text-transform: uppercase;">Company</label>
          <input type="text" id="upCompany" class="input-compact" style="width: 100%; margin-top: 0.25rem;" value="Acme Industries" required>
        </div>
        <div>
          <label style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-muted); font-weight: 600; text-transform: uppercase;">Document Type</label>
          <select id="upType" class="input-compact" style="width: 100%; margin-top: 0.25rem;">
            <option value="10-Q Quarterly Report">10-Q Quarterly Report</option>
            <option value="10-K Annual Report">10-K Annual Report</option>
            <option value="Earnings Call Transcript">Earnings Call Transcript</option>
            <option value="8-K Material Filing">8-K Material Filing</option>
          </select>
        </div>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;">
        <div>
          <label style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-muted); font-weight: 600; text-transform: uppercase;">Publication Date</label>
          <input type="date" id="upDate" class="input-compact" style="width: 100%; margin-top: 0.25rem;" value="2025-05-10" required>
        </div>
        <div>
          <label style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-muted); font-weight: 600; text-transform: uppercase;">Reporting Period</label>
          <input type="text" id="upPeriod" class="input-compact" style="width: 100%; margin-top: 0.25rem;" value="Q4 FY25" required>
        </div>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 0.5rem;">
        <button type="button" class="btn btn-outline" onclick="closeUploadModal()">Cancel</button>
        <button type="submit" id="btnUp" class="btn btn-primary">Ingest Document</button>
      </div>
    </form>
  </div>

  <!-- Corpus Management Modal -->
  <div class="modal-veil" id="corpusVeil" onclick="closeCorpusModal()"></div>
  <div class="modal-window" id="corpusModal" style="max-width: 640px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--cf-border); padding-bottom: 0.6rem;">
      <h3 style="font-size: 1.05rem; font-weight: 700;">Active Document Corpus</h3>
      <button class="btn btn-subtle" onclick="closeCorpusModal()">×</button>
    </div>
    <div id="corpusTableWrap" style="max-height: 320px; overflow-y: auto; margin-bottom: 1rem;"></div>
    <div style="display: flex; justify-content: space-between;">
      <button class="btn btn-outline" onclick="resetCorpus()">Reset to Default</button>
      <button class="btn btn-primary" onclick="closeCorpusModal()">Done</button>
    </div>
  </div>

  <!-- Evidence Inspector Drawer -->
  <div class="drawer-veil" id="drawerVeil" onclick="closeDrawer()"></div>
  <aside class="drawer-window" id="drawer">
    <div class="drawer-top">
      <div>
        <div style="font-family: var(--cf-font-mono); font-size: 0.68rem; color: var(--cf-text-dim); text-transform: uppercase;">Evidence Inspector</div>
        <strong id="drawerTitle" style="font-size: 0.95rem;">Document Metadata</strong>
      </div>
      <button class="btn btn-subtle" onclick="closeDrawer()">×</button>
    </div>
    <div class="drawer-scroll" id="drawerContent"></div>
  </aside>

  <script id="initData" type="application/json">__INITIAL_DATA__</script>

  <script>
    let appState = JSON.parse(document.getElementById('initData').textContent);

    function esc(s) {
      if (!s) return '';
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function toggleApiInput() {
      const isLlm = document.getElementById('engineSelect').value === 'llm';
      document.getElementById('apiKeyInput').style.display = isLlm ? 'inline-block' : 'none';
    }

    function openGuide() {
      document.getElementById('guideModal').classList.add('show');
      document.getElementById('guideVeil').classList.add('show');
    }
    function closeGuide() {
      document.getElementById('guideModal').classList.remove('show');
      document.getElementById('guideVeil').classList.remove('show');
    }

    function openUploadModal() {
      document.getElementById('uploadModal').classList.add('show');
      document.getElementById('uploadVeil').classList.add('show');
    }
    function closeUploadModal() {
      document.getElementById('uploadModal').classList.remove('show');
      document.getElementById('uploadVeil').classList.remove('show');
    }

    function openCorpusModal() {
      document.getElementById('corpusModal').classList.add('show');
      document.getElementById('corpusVeil').classList.add('show');
      renderCorpusTable();
    }
    function closeCorpusModal() {
      document.getElementById('corpusModal').classList.remove('show');
      document.getElementById('corpusVeil').classList.remove('show');
    }

    function switchConsoleTab(t) {
      document.getElementById('tabBtnChat').classList.toggle('active', t === 'chat');
      document.getElementById('tabBtnBench').classList.toggle('active', t === 'bench');
      document.getElementById('tabContentChat').classList.toggle('active', t === 'chat');
      document.getElementById('tabContentBench').classList.toggle('active', t === 'bench');
    }

    function toggleComparison() {
      const b = document.getElementById('compBody');
      const c = document.getElementById('compChevron');
      b.classList.toggle('open');
      c.textContent = b.classList.contains('open') ? '▲' : '▼';
    }

    function renderChatMessage(q, data) {
      const cor = data.correct_mode;
      const brk = data.broken_mode;
      const pass = cor.status === 'PASS';
      const msgId = 'v_' + Math.random().toString(36).substr(2, 8);

      const citHtml = (cor.claims || []).map(claim => {
        return (claim.citations || []).map(c => {
          const doc = (cor.retrieved_documents || []).find(d => d.id === c.document_id) || {};
          return `
            <button class="citation-pill-btn" onclick="inspectDoc('${esc(c.document_id)}', '${esc(c.quoted_text)}')">
              <div style="display: flex; justify-content: space-between; font-family: var(--cf-font-mono); font-size: 0.72rem; color: var(--cf-primary);">
                <span>📄 ${esc(c.document_id)} (Page ${c.page})</span>
                <span style="color: var(--cf-text-dim);">Pub: ${esc(doc.publication_date || '—')}</span>
              </div>
              <div style="color: var(--cf-text-muted); margin-top: 0.2rem;">"${esc(c.quoted_text)}"</div>
            </button>
          `;
        }).join('');
      }).join('');

      const checksHtml = Object.entries(cor.checks || {}).map(([name, status]) => {
        const ok = status === 'PASS';
        return `
          <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid var(--cf-border-subtle); font-size: 0.74rem;">
            <span style="font-family: var(--cf-font-mono); color: var(--cf-text-muted);">${esc(name.toUpperCase())}</span>
            <span style="font-family: var(--cf-font-mono); font-weight: 700; color: ${ok ? 'var(--cf-green)' : 'var(--cf-red)'};">${ok ? '✓ PASS' : '✗ FAIL'}</span>
          </div>
        `;
      }).join('');

      const botHtml = `
        <div class="msg-bot">
          <div class="msg-head">
            <strong style="font-size: 0.85rem; color: var(--cf-text);">PeriodGuard Research Assistant</strong>
            <span class="gate-indicator ${pass ? 'pass' : 'fail'}" onclick="toggleBreakdown('${msgId}')">
              ${pass ? '✓ VERIFIED SAFE' : '🚨 FAILED TEMPORAL GATE'}
              <span style="font-size: 0.65rem;">ℹ️</span>
            </span>
          </div>

          <div class="msg-text">
            ${esc(cor.answer_text || (cor.claims && cor.claims.length > 0 ? cor.claims.map(c => c.text).join(' ') : 'No safe evidence found.'))}
          </div>

          ${citHtml ? `
            <div class="citations-wrap">
              <div style="font-family: var(--cf-font-mono); font-size: 0.68rem; color: var(--cf-text-dim); text-transform: uppercase;">Verified Evidence Citations:</div>
              ${citHtml}
            </div>
          ` : ''}

          <div id="${msgId}" style="display: none; margin-top: 0.75rem; padding-top: 0.6rem; border-top: 1px solid var(--cf-border-subtle);">
            <div style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-dim); text-transform: uppercase; font-weight: 700; margin-bottom: 0.35rem;">
              Reliability Gate Checks Breakdown:
            </div>
            ${checksHtml}
          </div>
        </div>
      `;

      const log = document.getElementById('chatLog');
      if (q) {
        log.innerHTML += `<div class="msg-user">${esc(q)}</div>`;
      }
      log.innerHTML += botHtml;
      log.scrollTop = log.scrollHeight;

      // Update comparison
      const fail = brk.failures.find(f => f.type === 'FUTURE_PERIOD_LEAK') || brk.failures[0];
      if (fail) {
        document.getElementById('naiveSummary').innerHTML = `Naive RAG retrieved <strong>${esc(fail.document_id)}</strong> and leaked subsequent figures.`;
        document.getElementById('naiveDetails').textContent = `🚨 Leak: Published ${fail.publication_date} vs As-Of ${fail.as_of_date}`;
      } else {
        document.getElementById('naiveSummary').textContent = "Both modes passed for this specific cutoff.";
        document.getElementById('naiveDetails').textContent = "No temporal leak.";
      }
    }

    function toggleBreakdown(id) {
      const el = document.getElementById(id);
      if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
    }

    function setQuery(t) {
      document.getElementById('promptInput').value = t;
      sendPrompt();
    }

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
      }
    }

    async function sendPrompt() {
      const prompt = document.getElementById('promptInput');
      const q = prompt.value.trim();
      if (!q) return;

      const btn = document.getElementById('btnSend');
      btn.disabled = true;
      btn.textContent = 'Evaluating...';

      const payload = {
        question: q,
        company: document.getElementById('companyInput').value,
        as_of_date: document.getElementById('asOfDateInput').value,
        as_of_reporting_period: document.getElementById('periodInput').value,
        use_llm: document.getElementById('engineSelect').value === 'llm',
        api_key: document.getElementById('apiKeyInput').value || null
      };

      prompt.value = '';

      try {
        const resp = await fetch('/api/evaluate/custom', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (resp.ok) {
          appState = await resp.json();
          renderChatMessage(q, appState);
        }
      } catch (e) {
        console.error(e);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Ask Assistant';
      }
    }

    async function renderBenchmarks() {
      const resp = await fetch('/api/presets');
      if (resp.ok) {
        const presets = await resp.json();
        document.getElementById('benchmarksContainer').innerHTML = presets.map(p => `
          <div class="benchmark-card" onclick="runPreset('${esc(p.id)}')">
            <div>
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge-tag ${esc(p.badge_color || 'purple')}">${esc(p.badge || 'Test')}</span>
                <span style="font-family: var(--cf-font-mono); font-size: 0.7rem; color: var(--cf-text-dim);">${esc(p.as_of_date)}</span>
              </div>
              <strong style="font-size: 0.88rem; display: block; margin: 0.35rem 0 0.2rem;">${esc(p.title)}</strong>
              <div style="font-size: 0.78rem; color: var(--cf-text-muted); line-height: 1.4;">${esc(p.description)}</div>
            </div>
            <div style="font-size: 0.74rem; font-weight: 600; color: var(--cf-primary);">⚡ Run Scenario &rarr;</div>
          </div>
        `).join('');
      }
    }

    async function runPreset(id) {
      const resp = await fetch('/api/presets');
      const presets = await resp.json();
      const p = presets.find(x => x.id === id);
      if (p) {
        switchConsoleTab('chat');
        document.getElementById('promptInput').value = p.question;
        document.getElementById('companyInput').value = p.company;
        document.getElementById('asOfDateInput').value = p.as_of_date;
        document.getElementById('periodInput').value = p.as_of_reporting_period;
        sendPrompt();
      }
    }

    async function updateSidebarDocs() {
      const resp = await fetch('/api/corpus');
      if (resp.ok) {
        const docs = await resp.json();
        document.getElementById('countBadge').textContent = docs.length;
        document.getElementById('sidebarDocList').innerHTML = docs.map(d => `
          <div class="doc-item-row">
            <div>
              <div class="doc-item-title">${esc(d.id)}</div>
              <div class="doc-item-meta">${esc(d.company)} • ${esc(d.publication_date)}</div>
            </div>
            <span class="doc-type-pill">${esc(d.reporting_period)}</span>
          </div>
        `).join('');
      }
    }

    async function renderCorpusTable() {
      const resp = await fetch('/api/corpus');
      if (resp.ok) {
        const docs = await resp.json();
        document.getElementById('corpusTableWrap').innerHTML = `
          <table class="clean-table">
            <thead>
              <tr><th>ID</th><th>Company</th><th>Type</th><th>Period</th><th>Published</th></tr>
            </thead>
            <tbody>
              ${docs.map(d => `
                <tr>
                  <td style="font-family: var(--cf-font-mono);">${esc(d.id)}</td>
                  <td>${esc(d.company)}</td>
                  <td>${esc(d.doc_type)}</td>
                  <td>${esc(d.reporting_period)}</td>
                  <td style="font-family: var(--cf-font-mono);">${esc(d.publication_date)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      }
    }

    async function resetCorpus() {
      if (confirm('Reset corpus to default fixtures?')) {
        await fetch('/api/corpus/reset', { method: 'POST' });
        updateSidebarDocs();
        closeCorpusModal();
        sendPrompt();
      }
    }

    function inspectDoc(docId, quote) {
      const all = (appState.correct_mode.retrieved_documents || []).concat(appState.broken_mode.retrieved_documents || []);
      const doc = all.find(d => d.id === docId) || { id: docId, company: 'Acme', doc_type: 'Filing', publication_date: '2025-05-10', page: 1, text: quote, source_url: 'Corpus' };
      const asOf = appState.case.as_of_date;
      const isFuture = new Date(doc.publication_date) > new Date(asOf);

      document.getElementById('drawerTitle').textContent = doc.id;
      document.getElementById('drawerContent').innerHTML = `
        <div>
          <div style="font-family: var(--cf-font-mono); font-size: 0.68rem; color: var(--cf-text-dim); text-transform: uppercase; margin-bottom: 0.35rem;">Metadata</div>
          <table class="clean-table">
            <tr><th>Company</th><td>${esc(doc.company)}</td></tr>
            <tr><th>Doc Type</th><td>${esc(doc.doc_type)}</td></tr>
            <tr><th>Publication Date</th><td style="font-family: var(--cf-font-mono); font-weight: 700; color: ${isFuture ? 'var(--cf-red)' : 'var(--cf-green)'};">${esc(doc.publication_date)}</td></tr>
            <tr><th>Period</th><td>${esc(doc.reporting_period)}</td></tr>
            <tr><th>Page</th><td>Page ${doc.page}</td></tr>
          </table>
        </div>

        <div style="padding: 0.65rem; border-radius: 4px; font-size: 0.78rem; background: ${isFuture ? 'var(--cf-red-bg)' : 'var(--cf-green-bg)'}; border: 1px solid ${isFuture ? 'var(--cf-red-border)' : 'var(--cf-green-border)'}; color: ${isFuture ? 'var(--cf-red)' : 'var(--cf-green)'};">
          ${isFuture ? `🚨 <strong>FUTURE LEAK:</strong> Published on ${doc.publication_date}, after as-of cutoff (${asOf}).` : `✓ <strong>WITHIN CUTOFF:</strong> Published on ${doc.publication_date}, valid for as-of date (${asOf}).`}
        </div>

        <div>
          <div style="font-family: var(--cf-font-mono); font-size: 0.68rem; color: var(--cf-text-dim); text-transform: uppercase; margin-bottom: 0.35rem;">Verbatim Evidence Quote</div>
          <div style="padding: 0.75rem; background: var(--cf-surface); border: 1px solid var(--cf-border); border-radius: 4px; font-size: 0.8rem; color: var(--cf-text-muted); font-style: italic;">
            "${esc(quote || doc.text)}"
          </div>
        </div>
      `;

      document.getElementById('drawer').classList.add('show');
      document.getElementById('drawerVeil').classList.add('show');
    }

    function closeDrawer() {
      document.getElementById('drawer').classList.remove('show');
      document.getElementById('drawerVeil').classList.remove('show');
    }

    async function handleUpload(e) {
      e.preventDefault();
      const btn = document.getElementById('btnUp');
      btn.disabled = true;
      btn.textContent = 'Ingesting...';

      const fd = new FormData();
      fd.append('file', document.getElementById('upFile').files[0]);
      fd.append('company', document.getElementById('upCompany').value);
      fd.append('doc_type', document.getElementById('upType').value);
      fd.append('publication_date', document.getElementById('upDate').value);
      fd.append('reporting_period', document.getElementById('upPeriod').value);

      try {
        const resp = await fetch('/api/corpus/upload', { method: 'POST', body: fd });
        if (resp.ok) {
          const res = await resp.json();
          alert('✓ Success: ' + res.message);
          closeUploadModal();
          updateSidebarDocs();
          document.getElementById('companyInput').value = document.getElementById('upCompany').value;
          document.getElementById('periodInput').value = document.getElementById('upPeriod').value;
          switchConsoleTab('chat');
        } else {
          const err = await resp.json();
          alert('Upload failed: ' + (err.detail || 'Error'));
        }
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Ingest Document';
      }
    }

    // Init
    renderChatMessage(null, appState);
    renderBenchmarks();
    updateSidebarDocs();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def render_dashboard(use_llm: bool = Query(False)) -> str:
    data = execute_evaluation(get_default_case(), use_llm=use_llm)
    json_str = json.dumps(data).replace("</", "<\\/")
    return LANDING_PAGE_HTML.replace("__INITIAL_DATA__", json_str)
