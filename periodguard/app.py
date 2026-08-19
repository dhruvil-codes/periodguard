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

# PDF Support
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


app = FastAPI(
    title="PeriodGuard",
    description="Interactive evaluation workbench for financial research systems detecting future-period citation leakage.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory corpus and evaluators
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
    expected_metric: Optional[str] = Field(default="EBITDA margin")
    expected_unit: Optional[str] = Field(default="bps")
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
        "title": "Default Case: Future-Period Leak",
        "badge": "Temporal Gate Trap",
        "company": "Acme Industries",
        "question": "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.",
        "as_of_date": "2025-05-15",
        "as_of_reporting_period": "Q4 FY25",
        "expected_metric": "EBITDA margin",
        "expected_unit": "bps",
        "description": "Tests if retrieval leaks the FY26 Annual Report (published Aug 2025) to answer a May 2025 query.",
    },
    {
        "id": "clean_historical_pass",
        "title": "Clean Historical Query (Safe Window)",
        "badge": "Safe Query",
        "company": "Acme Industries",
        "question": "What was Acme Industries' sequential EBITDA margin change in Q4 FY25?",
        "as_of_date": "2025-06-01",
        "as_of_reporting_period": "Q4 FY25",
        "expected_metric": "EBITDA margin",
        "expected_unit": "bps",
        "description": "Sets the as-of date after Q4 results release (June 2025), verifying a clean PASS across both modes.",
    },
    {
        "id": "strict_early_cutoff",
        "title": "Strict Early Cutoff (Before Call)",
        "badge": "Pre-Release Cutoff",
        "company": "Acme Industries",
        "question": "What management commentary was provided regarding Q4 FY25 EBITDA margin expansion?",
        "as_of_date": "2025-05-11",
        "as_of_reporting_period": "Q4 FY25",
        "expected_metric": "EBITDA margin",
        "expected_unit": "bps",
        "description": "Sets cutoff to May 11, 2025 (before the May 12 Earnings Call), catching calls cited before occurrence.",
    },
]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "PeriodGuard Interactive Evaluation Workbench"}


@app.get("/api/presets")
def get_presets() -> List[Dict[str, Any]]:
    return PRESETS


@app.get("/api/corpus")
def list_corpus_documents() -> List[Dict[str, Any]]:
    docs = active_corpus.all_documents()
    return [doc.model_dump(mode="json") for doc in docs]


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
        # Text or JSON
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
        expected_metric=req.expected_metric,
        expected_unit=req.expected_unit,
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
  <title>PeriodGuard • Financial Research Reliability & Citation Leakage Workbench</title>
  
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">

  <style>
    :root {
      --bg-base: #06090f;
      --bg-surface: #0c121e;
      --bg-surface-elevated: #121a2c;
      --bg-surface-hover: #19243d;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-strong: rgba(255, 255, 255, 0.16);
      --border-accent: rgba(99, 102, 241, 0.35);

      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;

      --emerald-accent: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.18);
      --emerald-border: rgba(16, 185, 129, 0.35);
      --emerald-badge-bg: rgba(6, 78, 59, 0.5);
      --emerald-badge-text: #34d399;

      --rose-accent: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.18);
      --rose-border: rgba(244, 63, 94, 0.38);
      --rose-badge-bg: rgba(136, 19, 55, 0.5);
      --rose-badge-text: #fb7185;

      --indigo-accent: #6366f1;
      --cyan-accent: #06b6d4;
      --amber-accent: #f59e0b;

      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 18px;

      --font-display: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.55;
      background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.14), transparent),
        radial-gradient(circle at 15% 25%, rgba(6, 182, 212, 0.06), transparent),
        radial-gradient(circle at 85% 75%, rgba(244, 63, 94, 0.05), transparent);
      background-attachment: fixed;
      padding: 2rem 1.5rem 4rem;
    }

    .app-container {
      max-width: 1340px;
      margin: 0 auto;
    }

    /* Top Navigation Header */
    .top-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.75rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border-subtle);
      flex-wrap: wrap;
      gap: 1.25rem;
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .brand-icon {
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 24px rgba(79, 70, 229, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .brand-icon svg {
      width: 24px;
      height: 24px;
      fill: none;
      stroke: white;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .brand-text h1 {
      font-family: var(--font-display);
      font-size: 1.75rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .brand-text p {
      font-size: 0.88rem;
      color: var(--text-secondary);
      margin-top: 0.15rem;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .btn {
      font-family: var(--font-body);
      font-weight: 600;
      font-size: 0.86rem;
      padding: 0.55rem 1.15rem;
      border-radius: var(--radius-md);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      border: 1px solid transparent;
      text-decoration: none;
    }

    .btn-primary {
      background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
      color: white;
      box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35);
    }

    .btn-primary:hover {
      background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
    }

    .btn-secondary {
      background: var(--bg-surface);
      color: var(--text-primary);
      border: 1px solid var(--border-strong);
    }

    .btn-secondary:hover {
      background: var(--bg-surface-elevated);
      border-color: rgba(255, 255, 255, 0.25);
    }

    /* Workbench Controls Banner */
    .workbench-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.5rem 1.75rem;
      margin-bottom: 2rem;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .panel-title {
      font-family: var(--font-display);
      font-size: 1.1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .preset-pill-group {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .preset-btn {
      font-family: var(--font-mono);
      font-size: 0.75rem;
      padding: 0.3rem 0.65rem;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .preset-btn:hover, .preset-btn.active {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--indigo-accent);
      color: #ffffff;
    }

    .form-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr auto;
      gap: 1rem;
      align-items: flex-end;
    }

    @media (max-width: 1080px) {
      .form-grid {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 640px) {
      .form-grid {
        grid-template-columns: 1fr;
      }
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .form-label {
      font-family: var(--font-mono);
      font-size: 0.74rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-secondary);
    }

    .form-input, .form-select {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      color: #ffffff;
      font-family: var(--font-body);
      font-size: 0.88rem;
      padding: 0.55rem 0.8rem;
      transition: border 0.15s ease;
      width: 100%;
    }

    .form-input:focus, .form-select:focus {
      outline: none;
      border-color: var(--cyan-accent);
      box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
    }

    /* Side-by-Side Dual Comparison Grid */
    .comparison-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.75rem;
      margin-bottom: 3rem;
    }

    @media (max-width: 960px) {
      .comparison-grid {
        grid-template-columns: 1fr;
      }
    }

    .mode-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.75rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      position: relative;
      transition: all 0.3s ease;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
    }

    .mode-card.correct-card {
      border-top: 4px solid var(--emerald-accent);
    }

    .mode-card.broken-card {
      border-top: 4px solid var(--rose-accent);
    }

    .mode-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }

    .mode-card-title h3 {
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
      color: #ffffff;
    }

    .mode-card-title p {
      font-size: 0.84rem;
      color: var(--text-secondary);
      margin-top: 0.2rem;
    }

    .status-badge {
      font-family: var(--font-mono);
      font-size: 0.88rem;
      font-weight: 700;
      padding: 0.35rem 0.85rem;
      border-radius: var(--radius-md);
      letter-spacing: 0.05em;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }

    .status-badge.pass {
      background: var(--emerald-badge-bg);
      color: var(--emerald-badge-text);
      border: 1px solid var(--emerald-border);
      box-shadow: 0 0 16px var(--emerald-glow);
    }

    .status-badge.fail {
      background: var(--rose-badge-bg);
      color: var(--rose-badge-text);
      border: 1px solid var(--rose-border);
      box-shadow: 0 0 16px var(--rose-glow);
    }

    .section-label {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-muted);
      margin-bottom: 0.6rem;
    }

    /* Validator Checks Table */
    .checks-list {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      background: rgba(0, 0, 0, 0.22);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.75rem;
    }

    .check-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.4rem 0.5rem;
      border-radius: var(--radius-sm);
      font-size: 0.84rem;
    }

    .check-title {
      font-weight: 500;
      color: var(--text-secondary);
    }

    .mini-status {
      font-family: var(--font-mono);
      font-size: 0.74rem;
      font-weight: 700;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
    }

    .mini-status.pass {
      color: #34d399;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .mini-status.fail {
      color: #fb7185;
      background: rgba(244, 63, 94, 0.12);
      border: 1px solid rgba(244, 63, 94, 0.3);
    }

    /* Diagnosis / Failure Alerts */
    .pass-diagnosis-box {
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid var(--emerald-border);
      border-radius: var(--radius-md);
      padding: 0.9rem 1.1rem;
      font-size: 0.86rem;
      color: #6ee7b7;
      display: flex;
      align-items: flex-start;
      gap: 0.6rem;
    }

    .failure-alert-box {
      background: rgba(244, 63, 94, 0.08);
      border: 1px solid var(--rose-border);
      border-radius: var(--radius-md);
      padding: 1.1rem;
      box-shadow: 0 0 20px rgba(244, 63, 94, 0.08);
    }

    .fail-badge-title {
      color: #f43f5e;
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 0.84rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      margin-bottom: 0.4rem;
    }

    .fail-description {
      font-size: 0.86rem;
      color: #fecdd3;
      margin-bottom: 0.75rem;
      line-height: 1.45;
    }

    .fail-timeline-callout {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(244, 63, 94, 0.25);
      padding: 0.55rem 0.75rem;
      border-radius: var(--radius-sm);
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: #fda4af;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.4rem;
    }

    /* Claims & Citations */
    .claims-stack {
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }

    .claim-item-card {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1rem 1.15rem;
      transition: all 0.2s ease;
    }

    .claim-item-card:hover {
      border-color: rgba(255, 255, 255, 0.15);
      background: var(--bg-surface-hover);
    }

    .claim-assertion-text {
      font-size: 0.92rem;
      font-weight: 500;
      color: #e2e8f0;
      line-height: 1.45;
      margin-bottom: 0.65rem;
    }

    .claim-attributes {
      display: flex;
      gap: 0.6rem;
      flex-wrap: wrap;
      margin-bottom: 0.85rem;
    }

    .attr-tag {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      background: rgba(0, 0, 0, 0.3);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      color: var(--text-secondary);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .attr-tag b { color: #ffffff; }

    .citation-btn {
      width: 100%;
      text-align: left;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border-subtle);
      border-left: 3px solid var(--indigo-accent);
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      padding: 0.65rem 0.85rem;
      cursor: pointer;
      transition: all 0.2s ease;
      margin-top: 0.4rem;
      display: block;
    }

    .citation-btn:hover {
      background: rgba(30, 41, 59, 0.9);
      border-left-color: var(--cyan-accent);
      transform: translateX(2px);
    }

    .cit-btn-header {
      display: flex;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 0.76rem;
      color: #93c5fd;
      margin-bottom: 0.25rem;
    }

    .cit-btn-quote {
      font-size: 0.8rem;
      color: #cbd5e1;
      font-style: italic;
      line-height: 1.4;
    }

    /* Modal / Drawer */
    .modal-backdrop, .drawer-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(3, 7, 18, 0.75);
      backdrop-filter: blur(5px);
      z-index: 998;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease;
    }

    .modal-backdrop.active, .drawer-backdrop.active {
      opacity: 1;
      pointer-events: auto;
    }

    .modal-dialog {
      position: fixed;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%) scale(0.95);
      width: 90%; max-width: 620px;
      background: #0d121f;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-lg);
      padding: 2rem;
      z-index: 999;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
      opacity: 0;
      pointer-events: none;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .modal-dialog.active {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, -50%) scale(1);
    }

    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }

    .modal-header h3 {
      font-family: var(--font-display);
      font-size: 1.3rem;
      font-weight: 700;
    }

    .inspector-drawer {
      position: fixed;
      top: 0; right: 0; bottom: 0;
      width: 100%; max-width: 520px;
      background: #0d121f;
      border-left: 1px solid var(--border-strong);
      z-index: 999;
      box-shadow: -12px 0 40px rgba(0, 0, 0, 0.6);
      transform: translateX(100%);
      transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
    }

    .inspector-drawer.active { transform: translateX(0); }

    .drawer-top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.5rem 1.75rem;
      border-bottom: 1px solid var(--border-subtle);
    }

    .drawer-top-bar h3 {
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
    }

    .btn-close {
      background: none;
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      font-size: 1.25rem;
      width: 32px;
      height: 32px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
    }

    .btn-close:hover {
      background: rgba(255, 255, 255, 0.08);
      color: white;
    }

    .drawer-body {
      padding: 1.75rem;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .doc-meta-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
    }

    .doc-meta-table td {
      padding: 0.65rem 0.9rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .doc-meta-table td:first-child {
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 0.76rem;
      width: 38%;
    }

    .doc-meta-table td:last-child {
      color: #ffffff;
      font-weight: 600;
    }

    .timeline-card {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1.1rem;
    }

    .timeline-nodes {
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      position: relative;
      padding-left: 1.2rem;
    }

    .timeline-nodes::before {
      content: '';
      position: absolute;
      left: 4px; top: 6px; bottom: 6px;
      width: 2px;
      background: var(--border-strong);
    }

    .t-node { position: relative; font-size: 0.82rem; }
    .t-node::after {
      content: '';
      position: absolute;
      left: -1.2rem; top: 4px;
      width: 10px; height: 10px;
      border-radius: 50%;
      background: var(--cyan-accent);
      box-shadow: 0 0 8px var(--cyan-accent);
    }
    .t-node.future::after {
      background: var(--rose-accent);
      box-shadow: 0 0 10px var(--rose-accent);
    }

    .t-node-title { font-weight: 600; color: #ffffff; }
    .t-node-desc { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); }

    .verbatim-box {
      background: #070a12;
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: var(--radius-md);
      padding: 1rem;
      font-size: 0.84rem;
      line-height: 1.6;
      color: #cbd5e1;
      font-style: italic;
    }

    footer {
      border-top: 1px solid var(--border-subtle);
      padding-top: 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--text-muted);
      font-size: 0.82rem;
      flex-wrap: wrap;
      gap: 1rem;
    }
  </style>
</head>
<body>
  <div class="app-container">
    
    <!-- Top Header -->
    <header class="top-header">
      <div class="brand-group">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
        </div>
        <div class="brand-text">
          <h1>PeriodGuard <span style="font-size: 0.7rem; font-family: var(--font-mono); background: rgba(99,102,241,0.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4); padding: 0.2rem 0.5rem; border-radius: 9999px; -webkit-text-fill-color: #a5b4fc;">WORKBENCH</span></h1>
          <p>Financial Research Reliability & Future-Period Citation Leakage Testing Harness</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary" onclick="openCorpusModal()">
          📚 Manage Corpus (<span id="corpusCountBadge">4</span> docs)
        </button>
        <button class="btn btn-secondary" onclick="openUploadModal()">
          📄 Ingest PDF / Doc
        </button>
        <button id="btnRunEval" class="btn btn-primary" onclick="runWorkbenchEvaluation()">
          ⚡ Run Evaluation
        </button>
      </div>
    </header>

    <!-- Interactive Workbench Parameters -->
    <section class="workbench-panel">
      <div class="panel-header">
        <div class="panel-title">
          <span>🛠️</span> Evaluation Case Parameters
        </div>
        <div class="preset-pill-group">
          <span style="font-size: 0.75rem; color: var(--text-muted); align-self: center; margin-right: 0.2rem;">Quick Presets:</span>
          <button class="preset-btn active" onclick="loadPreset('future_leak_default')">Future Leak Trap</button>
          <button class="preset-btn" onclick="loadPreset('clean_historical_pass')">Safe Historical</button>
          <button class="preset-btn" onclick="loadPreset('strict_early_cutoff')">Strict Pre-Release</button>
        </div>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label class="form-label">Research Question</label>
          <input type="text" id="inputQuestion" class="form-input" value="As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.">
        </div>
        <div class="form-group">
          <label class="form-label">Target Entity</label>
          <input type="text" id="inputCompany" class="form-input" value="Acme Industries">
        </div>
        <div class="form-group">
          <label class="form-label">As-Of Cutoff Date</label>
          <input type="date" id="inputAsOfDate" class="form-input" value="2025-05-15">
        </div>
        <div class="form-group">
          <label class="form-label">Reporting Period</label>
          <input type="text" id="inputPeriod" class="form-input" value="Q4 FY25">
        </div>
        <div class="form-group">
          <label class="form-label">Engine</label>
          <select id="selectEngine" class="form-select">
            <option value="deterministic">Deterministic</option>
            <option value="llm">Live LLM</option>
          </select>
        </div>
      </div>
    </section>

    <!-- Side-by-Side Dual Comparison Grid -->
    <main class="comparison-grid" id="comparisonGrid">
      <!-- Dynamic rendering via JS -->
    </main>

    <!-- Footer -->
    <footer>
      <span><strong>PeriodGuard</strong> • Interactive reliability test bench for financial research pipelines.</span>
      <span>Reference Implementation • Evaluates temporal safety & citation traceability</span>
    </footer>

  </div>

  <!-- Document Corpus Modal -->
  <div class="modal-backdrop" id="corpusModalBackdrop" onclick="closeCorpusModal()"></div>
  <div class="modal-dialog" id="corpusModal" style="max-width: 780px;">
    <div class="modal-header">
      <h3>Active Document Corpus (<span id="corpusListCount">4</span>)</h3>
      <button class="btn-close" onclick="closeCorpusModal()">×</button>
    </div>
    <div style="max-height: 400px; overflow-y: auto; margin-bottom: 1.5rem;" id="corpusTableContainer">
      <!-- Populated via JS -->
    </div>
    <div style="display: flex; justify-content: space-between;">
      <button class="btn btn-secondary" onclick="resetCorpusToDefault()">🔄 Reset to Default Fixture</button>
      <button class="btn btn-primary" onclick="closeCorpusModal()">Done</button>
    </div>
  </div>

  <!-- Ingest PDF / Document Modal -->
  <div class="modal-backdrop" id="uploadModalBackdrop" onclick="closeUploadModal()"></div>
  <div class="modal-dialog" id="uploadModal">
    <div class="modal-header">
      <h3>Ingest Financial Document / PDF</h3>
      <button class="btn-close" onclick="closeUploadModal()">×</button>
    </div>
    <form id="uploadForm" onsubmit="handleDocumentUpload(event)">
      <div style="display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1.5rem;">
        <div class="form-group">
          <label class="form-label">Select File (.pdf, .txt, .json)</label>
          <input type="file" id="uploadFile" class="form-input" required accept=".pdf,.txt,.json">
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
          <div class="form-group">
            <label class="form-label">Company Name</label>
            <input type="text" id="uploadCompany" class="form-input" value="Acme Industries" required>
          </div>
          <div class="form-group">
            <label class="form-label">Document Type</label>
            <input type="text" id="uploadDocType" class="form-input" value="Quarterly Results" required>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
          <div class="form-group">
            <label class="form-label">Publication Date</label>
            <input type="date" id="uploadPubDate" class="form-input" value="2025-05-10" required>
          </div>
          <div class="form-group">
            <label class="form-label">Reporting Period</label>
            <input type="text" id="uploadPeriod" class="form-input" value="Q4 FY25" required>
          </div>
        </div>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 0.75rem;">
        <button type="button" class="btn btn-secondary" onclick="closeUploadModal()">Cancel</button>
        <button type="submit" id="btnUploadSubmit" class="btn btn-primary">Ingest Document</button>
      </div>
    </form>
  </div>

  <!-- Slide-over Citation Inspector Drawer -->
  <div class="drawer-backdrop" id="drawerBackdrop" onclick="closeDrawer()"></div>
  <aside class="inspector-drawer" id="inspectorDrawer">
    <div class="drawer-top-bar">
      <div>
        <div class="section-label" style="margin-bottom: 0.2rem;">Evidence Inspector</div>
        <h3 id="drawerDocId">Document Metadata</h3>
      </div>
      <button class="btn-close" onclick="closeDrawer()">×</button>
    </div>
    <div class="drawer-body" id="drawerBody"></div>
  </aside>

  <script id="initData" type="application/json">__INITIAL_DATA__</script>

  <script>
    let appState = JSON.parse(document.getElementById('initData').textContent);

    const checkLabels = {
      citation_resolution: 'Citation Resolution',
      temporal_consistency: 'Temporal Consistency',
      entity_period_consistency: 'Entity / Period Alignment',
      citation_support_proxy: 'Citation Support Proxy'
    };

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    function renderDashboard(data) {
      const grid = document.getElementById('comparisonGrid');
      grid.innerHTML = `
        ${renderModeCard(data.correct_mode, false, data)}
        ${renderModeCard(data.broken_mode, true, data)}
      `;
      updateCorpusCount();
    }

    function renderModeCard(report, isBroken, data) {
      const isPass = report.status === 'PASS';
      const cardClass = isBroken ? 'broken-card' : 'correct-card';
      const title = isBroken ? 'Unfiltered Mode (Broken)' : 'Date-Filtered Mode';
      const subtitle = isBroken ? 'Disables temporal filter · admits future evidence' : `Enforces as-of date (${data.case.as_of_date})`;
      const badgeClass = isPass ? 'pass' : 'fail';
      const badgeText = isPass ? '✓ PASS' : '✗ FAIL';

      const checksHtml = Object.entries(report.checks).map(([key, val]) => `
        <div class="check-item">
          <span class="check-title">${checkLabels[key] || key}</span>
          <span class="mini-status ${val.toLowerCase()}">${val}</span>
        </div>
      `).join('');

      let alertHtml = '';
      if (!isBroken || report.failures.length === 0) {
        alertHtml = `
          <div class="pass-diagnosis-box">
            <span>✓</span>
            <div>No temporal leakage detected. Future documents strictly excluded by publication date gate.</div>
          </div>
        `;
      } else {
        const primaryFail = report.failures.find(f => f.type === 'FUTURE_PERIOD_LEAK') || report.failures[0];
        const daysDiff = Math.round((new Date(primaryFail.publication_date) - new Date(primaryFail.as_of_date)) / (1000 * 60 * 60 * 24));
        alertHtml = `
          <div class="failure-alert-box">
            <div class="fail-badge-title">🚨 ${escapeHtml(primaryFail.type)}</div>
            <div class="fail-description">${escapeHtml(primaryFail.message)}</div>
            <div class="fail-timeline-callout">
              <span><strong>Offending Doc:</strong> ${escapeHtml(primaryFail.document_id)}</span>
              <span><strong>Published:</strong> ${escapeHtml(primaryFail.publication_date)} (Delta: +${daysDiff} days)</span>
            </div>
          </div>
        `;
      }

      const claimsHtml = report.claims.map(claim => {
        const citationsHtml = claim.citations.map(cit => {
          const doc = report.retrieved_documents.find(d => d.id === cit.document_id) || {};
          return `
            <button class="citation-btn" onclick="openInspector('${escapeHtml(cit.document_id)}', '${escapeHtml(cit.quoted_text)}')">
              <div class="cit-btn-header">
                <span>📄 ${escapeHtml(cit.document_id)}</span>
                <span>Page ${cit.page} · Pub: ${escapeHtml(doc.publication_date || '—')}</span>
              </div>
              <div class="cit-btn-quote">"${escapeHtml(cit.quoted_text)}"</div>
            </button>
          `;
        }).join('');

        return `
          <div class="claim-item-card">
            <div class="claim-assertion-text">"${escapeHtml(claim.text)}"</div>
            <div class="claim-attributes">
              <span class="attr-tag">Metric: <b>${escapeHtml(claim.metric || 'N/A')}</b></span>
              <span class="attr-tag">Value: <b>${claim.value !== null && claim.value !== undefined ? claim.value : 'N/A'} ${escapeHtml(claim.unit || '')}</b></span>
              <span class="attr-tag">Period: <b>${escapeHtml(claim.period || 'N/A')}</b></span>
            </div>
            <div>
              <div class="section-label" style="font-size: 0.68rem; margin-bottom: 0.3rem;">Cited Evidence (Click to inspect)</div>
              ${citationsHtml}
            </div>
          </div>
        `;
      }).join('');

      return `
        <div class="mode-card ${cardClass}">
          <div class="mode-card-header">
            <div class="mode-card-title">
              <h3>${title}</h3>
              <p>${subtitle}</p>
            </div>
            <span class="status-badge ${badgeClass}">${badgeText}</span>
          </div>

          <div>
            <div class="section-label">Deterministic Validator Checks</div>
            <div class="checks-list">${checksHtml}</div>
          </div>

          <div>
            <div class="section-label">${isBroken ? 'Leakage Diagnosis' : 'Temporal Verification'}</div>
            ${alertHtml}
          </div>

          <div>
            <div class="section-label">Synthesized Claims & Citations</div>
            <div class="claims-stack">${claimsHtml}</div>
          </div>
        </div>
      `;
    }

    async function runWorkbenchEvaluation() {
      const btn = document.getElementById('btnRunEval');
      btn.disabled = true;
      btn.innerHTML = '⚡ Running...';

      const payload = {
        question: document.getElementById('inputQuestion').value,
        company: document.getElementById('inputCompany').value,
        as_of_date: document.getElementById('inputAsOfDate').value,
        as_of_reporting_period: document.getElementById('inputPeriod').value,
        use_llm: document.getElementById('selectEngine').value === 'llm'
      };

      try {
        const resp = await fetch('/api/evaluate/custom', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (resp.ok) {
          appState = await resp.json();
          renderDashboard(appState);
        }
      } catch (err) {
        console.error('Evaluation failed', err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Run Evaluation';
      }
    }

    function loadPreset(presetId) {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');

      if (presetId === 'future_leak_default') {
        document.getElementById('inputQuestion').value = "As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence.";
        document.getElementById('inputCompany').value = "Acme Industries";
        document.getElementById('inputAsOfDate').value = "2025-05-15";
        document.getElementById('inputPeriod').value = "Q4 FY25";
      } else if (presetId === 'clean_historical_pass') {
        document.getElementById('inputQuestion').value = "What was Acme Industries' sequential EBITDA margin change in Q4 FY25?";
        document.getElementById('inputCompany').value = "Acme Industries";
        document.getElementById('inputAsOfDate').value = "2025-06-01";
        document.getElementById('inputPeriod').value = "Q4 FY25";
      } else if (presetId === 'strict_early_cutoff') {
        document.getElementById('inputQuestion').value = "What management commentary was provided regarding Q4 FY25 EBITDA margin expansion?";
        document.getElementById('inputCompany').value = "Acme Industries";
        document.getElementById('inputAsOfDate').value = "2025-05-11";
        document.getElementById('inputPeriod').value = "Q4 FY25";
      }
      runWorkbenchEvaluation();
    }

    function openInspector(docId, quotedText) {
      const allDocs = appState.correct_mode.retrieved_documents.concat(appState.broken_mode.retrieved_documents);
      const doc = allDocs.find(d => d.id === docId) || {
        id: docId, company: 'Acme Industries', doc_type: 'Financial Document', reporting_period: 'Q4 FY25', publication_date: '2025-05-10', page: 1, text: quotedText, source_url: 'Corpus'
      };

      const asOf = appState.case.as_of_date;
      const isFuture = new Date(doc.publication_date) > new Date(asOf);

      document.getElementById('drawerDocId').textContent = doc.id;
      document.getElementById('drawerBody').innerHTML = `
        <div>
          <div class="section-label">Document Metadata</div>
          <table class="doc-meta-table">
            <tr><td>Document Type</td><td>${escapeHtml(doc.doc_type)}</td></tr>
            <tr><td>Target Entity</td><td>${escapeHtml(doc.company)}</td></tr>
            <tr><td>Publication Date</td><td style="font-family: var(--font-mono); color: ${isFuture ? '#fb7185' : '#34d399'};">${escapeHtml(doc.publication_date)}</td></tr>
            <tr><td>Reporting Period</td><td>${escapeHtml(doc.reporting_period)}</td></tr>
            <tr><td>Section Page</td><td>Page ${doc.page}</td></tr>
            <tr><td>Provenance</td><td style="font-size: 0.75rem; word-break: break-all;">${escapeHtml(doc.source_url)}</td></tr>
          </table>
        </div>

        <div class="timeline-card">
          <div class="section-label">Temporal Boundary Timeline</div>
          <div class="timeline-nodes">
            <div class="t-node">
              <div class="t-node-title">Case As-Of Boundary</div>
              <div class="t-node-desc">${asOf} • Eligible cutoff date</div>
            </div>
            <div class="t-node ${isFuture ? 'future' : ''}">
              <div class="t-node-title">Document Publication</div>
              <div class="t-node-desc">${escapeHtml(doc.publication_date)} • ${isFuture ? '🚨 FUTURE LEAK (Violates Boundary)' : '✓ Within Safe Window'}</div>
            </div>
          </div>
        </div>

        <div>
          <div class="section-label">Verbatim Evidence Quote</div>
          <div class="verbatim-box">"${escapeHtml(quotedText || doc.text)}"</div>
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
          document.getElementById('corpusListCount').textContent = docs.length;
        }
      } catch (e) {}
    }

    async function openCorpusModal() {
      document.getElementById('corpusModal').classList.add('active');
      document.getElementById('corpusModalBackdrop').classList.add('active');

      const resp = await fetch('/api/corpus');
      if (resp.ok) {
        const docs = await resp.json();
        const container = document.getElementById('corpusTableContainer');
        container.innerHTML = `
          <table class="doc-meta-table">
            <thead>
              <tr style="background: rgba(255,255,255,0.04); text-align: left; font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);">
                <th style="padding: 0.6rem 0.9rem;">ID</th>
                <th style="padding: 0.6rem 0.9rem;">Company</th>
                <th style="padding: 0.6rem 0.9rem;">Period</th>
                <th style="padding: 0.6rem 0.9rem;">Published</th>
                <th style="padding: 0.6rem 0.9rem;">Type</th>
              </tr>
            </thead>
            <tbody>
              ${docs.map(d => `
                <tr>
                  <td style="font-family: var(--font-mono); color: #93c5fd;">${escapeHtml(d.id)}</td>
                  <td>${escapeHtml(d.company)}</td>
                  <td>${escapeHtml(d.reporting_period)}</td>
                  <td style="font-family: var(--font-mono);">${escapeHtml(d.publication_date)}</td>
                  <td style="font-size: 0.76rem; color: var(--text-secondary);">${escapeHtml(d.doc_type)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      }
    }

    function closeCorpusModal() {
      document.getElementById('corpusModal').classList.remove('active');
      document.getElementById('corpusModalBackdrop').classList.remove('active');
    }

    async function resetCorpusToDefault() {
      if (confirm('Reset corpus to original 4 financial records?')) {
        await fetch('/api/corpus/reset', { method: 'POST' });
        closeCorpusModal();
        runWorkbenchEvaluation();
      }
    }

    function openUploadModal() {
      document.getElementById('uploadModal').classList.add('active');
      document.getElementById('uploadModalBackdrop').classList.add('active');
    }

    function closeUploadModal() {
      document.getElementById('uploadModal').classList.remove('active');
      document.getElementById('uploadModalBackdrop').classList.remove('active');
    }

    async function handleDocumentUpload(e) {
      e.preventDefault();
      const btn = document.getElementById('btnUploadSubmit');
      btn.disabled = true;
      btn.textContent = 'Ingesting...';

      const formData = new FormData();
      formData.append('file', document.getElementById('uploadFile').files[0]);
      formData.append('company', document.getElementById('uploadCompany').value);
      formData.append('doc_type', document.getElementById('uploadDocType').value);
      formData.append('reporting_period', document.getElementById('uploadPeriod').value);
      formData.append('publication_date', document.getElementById('uploadPubDate').value);

      try {
        const resp = await fetch('/api/corpus/upload', {
          method: 'POST',
          body: formData
        });
        if (resp.ok) {
          alert('Document successfully ingested into PeriodGuard corpus!');
          closeUploadModal();
          runWorkbenchEvaluation();
        } else {
          const err = await resp.json();
          alert('Upload error: ' + (err.detail || 'Failed to upload'));
        }
      } catch (err) {
        alert('Upload failed: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Ingest Document';
      }
    }

    renderDashboard(appState);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def render_dashboard(use_llm: bool = Query(False)) -> str:
    data = execute_evaluation(get_default_case(), use_llm=use_llm)
    json_str = json.dumps(data).replace("</", "<\\/")
    return DASHBOARD_HTML.replace("__INITIAL_DATA__", json_str)
