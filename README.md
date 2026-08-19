# PeriodGuard • Evaluation & Reliability Workbench

> **An interactive evaluation workbench and testing tool for financial research systems. Tests whether financial answers are period-correct, entity-correct, numerically supported, and traceable to the exact evidence they cite.**

---

## 🎯 Problem: Why Citation Presence Is Insufficient

In financial research workflows, retrieval pipelines and LLMs generate answers from filings across multiple fiscal years. When temporal boundaries are not strictly enforced, systems frequently cite subsequent earnings reports to answer historical questions.

This creates a high-signal, deceptive failure mode:
1. The answer looks fluent and numerically plausible.
2. The citation resolves to a real document and page.
3. **A basic citation-presence check passes it.**
4. **However, the source was published months after the requested as-of date (Future-Period Citation Leak).**

> **Core Axiom:** *"A citation exists"* $\neq$ *"The cited answer is safe to use."*

PeriodGuard makes this failure visible, reproducible, interactive, and machine-detectable.

---

## 🛠️ Interactive Capabilities

- **⚡ Live Evaluation Workbench:** Tune research questions, target company, as-of cutoff date, and target fiscal periods in real-time.
- **📄 Ingest Custom PDFs & Filings:** Upload any earnings release or 10-Q PDF (`pypdf` extraction) into the live testing corpus.
- **📚 Dynamic Corpus Manager:** Inspect active documents in memory, add manual records, or reset to standard fixtures.
- **🔍 Deep Evidence Inspector:** Click any citation chip to view full document metadata, provenance URL, verbatim context, and visual timeline chart comparing publication date against as-of boundaries.
- **🤖 Dual-Engine Support:** Run offline deterministic evaluation (0 token cost, <0.5s execution) or connect to live LLMs (OpenAI, Gemini, Ollama) via `--llm`.

---

## 🏗️ Architecture

```
                 Corpus Documents / Uploaded PDFs
                                │
                                ▼
                     Metadata Filtering Engine
                   (Company & As-Of Date Gate)
                                │
                                ▼
                     Deterministic Retrieval
                    (Token-Overlap Relevance)
                                │
                                ▼
                  Structured Claim Synthesizer
                   (Deterministic / Live LLM)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
    Citation                 Temporal            Entity / Period
   Resolution              Consistency              Alignment
(INVALID_CITATION)    (FUTURE_PERIOD_LEAK)   (ENTITY_OR_PERIOD_MISMATCH)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                     Citation Support Proxy
                      (UNSUPPORTED_CLAIM)
                                │
                                ▼
               Interactive Workbench & JSON Report
```

---

## ⚡ Quickstart

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Launch Interactive Web Workbench
```powershell
uvicorn periodguard.app:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to:
- Test custom financial questions and as-of dates.
- Ingest local PDF filings and observe immediate evaluation changes.
- Inspect side-by-side mode comparisons with timeline diagnostics.

### 3. Run Single-Command Evaluation (CLI)
```powershell
python -m periodguard.evaluator
```

**Terminal Output:**
```
============================================================
PERIODGUARD EVALUATION RUN (DETERMINISTIC)
============================================================
CORRECT MODE: PASS
BROKEN MODE: FAIL -- ENTITY_OR_PERIOD_MISMATCH, FUTURE_PERIOD_LEAK, UNSUPPORTED_CLAIM
  - [FUTURE_PERIOD_LEAK] Doc: acme_fy26_annual_report (Pub: 2025-08-20) vs As-Of: 2025-05-15
    Message: Future-period citation leak: Document 'acme_fy26_annual_report' was published on 2025-08-20, which violates the as-of boundary (2025-05-15).
============================================================
```

### 4. Run Automated Test Suite
```powershell
pytest -v
```
*(Runs 23 unit & integration tests covering validators T1–T8, corpus ingestion, and API routes).*

---

## 🔍 The 4 Core Validators

| Validator | Target Failure | Failure Code |
|---|---|---|
| **Citation Resolution** | Document ID missing, page mismatch, or quote not found in retrieved set. | `INVALID_CITATION` |
| **Temporal Consistency** | Document published after requested as-of date (`publication_date > as_of_date`). | `FUTURE_PERIOD_LEAK` |
| **Entity / Period Alignment** | Document belongs to peer company or refers to future fiscal period. | `ENTITY_OR_PERIOD_MISMATCH` |
| **Citation Support Proxy** | Metric name, numeric value, unit, or directional terms unsupported by quote. | `UNSUPPORTED_CLAIM` |

---

## 🌐 API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the interactive evaluation workbench UI |
| `GET` | `/api/presets` | Returns pre-configured evaluation scenarios |
| `GET` | `/api/corpus` | Lists all documents currently in the testing corpus |
| `POST` | `/api/corpus/upload` | Ingests uploaded PDF or text file with financial metadata |
| `POST` | `/api/corpus/reset` | Resets in-memory corpus to default fixture |
| `POST` | `/api/evaluate/custom` | Runs custom evaluation with user-defined question & as-of date |
| `POST` | `/evaluate` | Executes default evaluation across both modes |
| `GET` | `/report` | Returns latest evaluation report JSON |

---

## 🛡️ Disclaimer
*PeriodGuard is an independent reference harness for evaluating financial research reliability. It is not affiliated with, nor does it access or test the private infrastructure of CalQuity.*
