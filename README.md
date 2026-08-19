# PeriodGuard

> **An evaluation harness for financial research systems that tests whether financial claims are period-correct, entity-correct, numerically supported, and traceable to the evidence they cite.**

---

## 🎯 Problem Statement: Why Citation Presence Is Insufficient

In financial research workflows, LLMs and retrieval pipelines generate responses from corpora spanning multiple reporting years and publication dates. If retrieval fails to strictly enforce temporal boundaries, a system can easily retrieve later information to answer a historical question.

This creates a high-signal, deceptive failure mode:
1. The answer looks numerically plausible and fluent.
2. The answer is accompanied by a valid, resolvable citation.
3. **A naive "citation-presence" check passes it.**
4. **However, the citation leaks future data published after the user's requested as-of date.**

> **Core Axiom:** *"A citation exists"* $\neq$ *"The cited answer is safe to use."*

PeriodGuard makes this failure visible, reproducible, and machine-detectable.

---

## 🏗️ Architecture

```
                      data/corpus.json
                            │
                            ▼
                Corpus Metadata Loader
                            │
                            ▼
                 Company & As-Of Filter
                 (Enabled vs Disabled)
                            │
                            ▼
                 Deterministic Retrieval
                 (Token-Overlap Scoring)
                            │
                            ▼
                 Structured Claim Synthesizer
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
    Citation             Temporal        Entity / Period
   Resolution          Consistency          Alignment
(INVALID_CITATION) (FUTURE_PERIOD_LEAK) (ENTITY_OR_PERIOD_MISMATCH)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                 Citation Support Proxy
                  (UNSUPPORTED_CLAIM)
                            │
                            ▼
             Evaluation Report (CLI & Web UI)
```

---

## ⚡ Quickstart

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run Single-Command Evaluation (CLI)
```powershell
python -m periodguard.evaluator
```

**Terminal Output:**
```
============================================================
PERIODGUARD EVALUATION RUN
============================================================
CORRECT MODE: PASS
BROKEN MODE: FAIL -- ENTITY_OR_PERIOD_MISMATCH, FUTURE_PERIOD_LEAK, UNSUPPORTED_CLAIM
  - [FUTURE_PERIOD_LEAK] Doc: acme_fy26_annual_report (Pub: 2025-08-20) vs As-Of: 2025-05-15
    Message: Future-period citation leak: Document 'acme_fy26_annual_report' was published on 2025-08-20, which violates the as-of boundary (2025-05-15).
============================================================
```

### 3. Run Automated Tests
```powershell
pytest -v
```
*(Runs 17 unit and integration tests including T1–T8 validator test cases).*

### 4. Launch Browser Dashboard (FastAPI)
```powershell
uvicorn periodguard.app:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) to inspect the side-by-side evaluation dashboard.

---

## 🔍 The 4 Core Validators

| Validator | Purpose | Failure Code |
|---|---|---|
| **Citation Resolution** | Verifies document ID exists in corpus, cited page matches, and quote is present in retrieved evidence. | `INVALID_CITATION` |
| **Temporal Consistency** | Verifies that cited document publication date does not violate the requested as-of cut-off date (`publication_date <= as_of_date`). | `FUTURE_PERIOD_LEAK` |
| **Entity / Period Alignment** | Verifies company matches target entity and claim reporting period does not exceed case boundary. | `ENTITY_OR_PERIOD_MISMATCH` |
| **Citation Support Proxy** | Rule-based proxy verifying numeric values, metric phrases, units, and directional terms in cited quotes. | `UNSUPPORTED_CLAIM` |

---

## 📊 Default Evaluation Case

- **Question:** *"As of 15 May 2025, did Acme Industries' EBITDA margin improve in Q4 FY25 versus Q3 FY25, and what reason did management give? Cite the evidence."*
- **Target Company:** Acme Industries
- **As-Of Date:** `2025-05-15`
- **Target Period:** `Q4 FY25`

### Comparative Results:

| Mode | Behavior | Result | Failure Code | Offending Document |
|---|---|---|---|---|
| **Correct Mode** | Enforces `publication_date <= 2025-05-15` | **`PASS`** | None | None (Excludes FY26 Annual Report) |
| **Broken Mode** | Disables as-of date filter | **`FAIL`** | `FUTURE_PERIOD_LEAK` | `acme_fy26_annual_report` (Published: `2025-08-20`) |

---

## 🌐 API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the HTML evaluation dashboard |
| `GET` | `/health` | Returns service health status |
| `POST` | `/evaluate` | Executes evaluation across both modes and returns structured JSON |
| `GET` | `/report` | Returns the latest evaluation report JSON |

---

## ⚠️ Limitations & Future Enhancements

### Current MVP Scope:
- Uses local structured JSON corpus fixture (`data/corpus.json`) instead of live PDF parsing.
- Uses transparent rule-based token overlap and proxy citation verification rather than an external LLM judge.
- Offline deterministic execution with zero required API keys.

### Future Roadmap:
1. Real PDF / 10-Q / Earnings Call transcript ingestion pipeline.
2. Hybrid BM25 + dense vector embeddings retrieval.
3. LLM-based semantic citation entailment judge.
4. MCP response adapter for CalQuity-compatible financial workflows.

---

## 🛡️ Disclaimer
*PeriodGuard is an independent reference harness for evaluating financial research reliability. It is not affiliated with, nor does it access or test the private infrastructure of CalQuity.*
