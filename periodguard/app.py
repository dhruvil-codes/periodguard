from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from periodguard.evaluator import Evaluator, get_default_case
from periodguard.models import EvaluationReport, RetrievalMode

app = FastAPI(
    title="PeriodGuard",
    description="Evaluation harness for financial research systems detecting future-period citation leaks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator = Evaluator()
_latest_reports_cache: Dict[str, EvaluationReport] = {}


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "PeriodGuard Evaluation Harness"}


@app.post("/evaluate")
def run_evaluation() -> Dict[str, Any]:
    global _latest_reports_cache
    reports = evaluator.run_both_modes()
    _latest_reports_cache = reports
    return {
        "correct_mode": reports["correct_mode"].model_dump(mode="json"),
        "broken_mode": reports["broken_mode"].model_dump(mode="json"),
    }


@app.get("/report")
def get_report() -> Dict[str, Any]:
    global _latest_reports_cache
    if not _latest_reports_cache:
        _latest_reports_cache = evaluator.run_both_modes()
    return {
        "correct_mode": _latest_reports_cache["correct_mode"].model_dump(mode="json"),
        "broken_mode": _latest_reports_cache["broken_mode"].model_dump(mode="json"),
    }


@app.get("/", response_class=HTMLResponse)
def render_dashboard() -> str:
    global _latest_reports_cache
    if not _latest_reports_cache:
        _latest_reports_cache = evaluator.run_both_modes()

    correct = _latest_reports_cache["correct_mode"]
    broken = _latest_reports_cache["broken_mode"]
    case = get_default_case()

    def format_status_badge(status: str) -> str:
        if status == "PASS":
            return '<span class="badge badge-pass">✓ PASS</span>'
        return '<span class="badge badge-fail">✗ FAIL</span>'

    def format_checks_table(checks: Dict[str, Any]) -> str:
        rows = []
        labels = {
            "citation_resolution": "Citation Resolution",
            "temporal_consistency": "Temporal Consistency",
            "entity_period_consistency": "Entity / Period Alignment",
            "citation_support_proxy": "Citation Support Proxy",
        }
        for k, v in checks.items():
            status_val = v.value if hasattr(v, "value") else str(v)
            badge = format_status_badge(status_val)
            rows.append(f"""
                <tr>
                    <td class="check-name">{labels.get(k, k)}</td>
                    <td class="check-status">{badge}</td>
                </tr>
            """)
        return "".join(rows)

    def format_claims_html(claims: list) -> str:
        items = []
        for c in claims:
            c_dict = c.model_dump(mode="json") if hasattr(c, "model_dump") else c
            citations_html = []
            for cit in c_dict.get("citations", []):
                citations_html.append(f"""
                    <div class="citation-box">
                        <div class="cit-header">
                            <span class="cit-doc">📄 {cit['document_id']}</span>
                            <span class="cit-page">Page {cit['page']}</span>
                        </div>
                        <div class="cit-quote">"{cit['quoted_text']}"</div>
                    </div>
                """)
            items.append(f"""
                <div class="claim-card">
                    <div class="claim-text">"{c_dict['text']}"</div>
                    <div class="claim-meta">
                        <span><strong>Metric:</strong> {c_dict.get('metric') or 'N/A'}</span>
                        <span><strong>Value:</strong> {c_dict.get('value') or 'N/A'} {c_dict.get('unit') or ''}</span>
                        <span><strong>Period:</strong> {c_dict.get('period') or 'N/A'}</span>
                    </div>
                    <div class="citations-container">
                        {''.join(citations_html)}
                    </div>
                </div>
            """)
        return "".join(items)

    def format_failures_html(failures: list) -> str:
        if not failures:
            return '<div class="no-failures">No validation failures detected. Temporal boundary respected.</div>'
        items = []
        for f in failures:
            f_dict = f.model_dump(mode="json") if hasattr(f, "model_dump") else f
            items.append(f"""
                <div class="failure-card">
                    <div class="fail-type">🚨 {f_dict['type']}</div>
                    <div class="fail-msg">{f_dict['message']}</div>
                    {f"<div class='fail-detail'><strong>Offending Document:</strong> {f_dict['document_id']} (Published: {f_dict['publication_date']}) vs <strong>Case As-Of Date:</strong> {f_dict['as_of_date']}</div>" if f_dict.get('document_id') else ""}
                </div>
            """)
        return "".join(items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PeriodGuard | Financial Research Reliability Harness</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-card: #111827;
            --bg-card-hover: #172033;
            --border-color: #1f293d;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-dim: #6b7280;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --emerald-500: #10b981;
            --emerald-900: rgba(16, 185, 129, 0.15);
            --rose-500: #f43f5e;
            --rose-900: rgba(244, 63, 94, 0.15);
            --amber-500: #f59e0b;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            line-height: 1.5;
            padding: 2.5rem 1.5rem;
        }}
        .container {{
            max-width: 1240px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }}
        .logo-group h1 {{
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .logo-group p {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }}
        .btn-rerun {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            padding: 0.65rem 1.25rem;
            font-weight: 600;
            font-size: 0.9rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        }}
        .btn-rerun:hover {{
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            transform: translateY(-1px);
        }}
        .case-banner {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .case-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-cyan);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .case-question {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 1rem;
        }}
        .case-tags {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.88rem;
            color: var(--text-muted);
            flex-wrap: wrap;
        }}
        .case-tag strong {{
            color: var(--text-main);
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}
        @media (max-width: 900px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .mode-column {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            position: relative;
        }}
        .mode-column.correct-mode {{
            border-top: 4px solid var(--emerald-500);
        }}
        .mode-column.broken-mode {{
            border-top: 4px solid var(--rose-500);
        }}
        .mode-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .mode-title {{
            font-size: 1.15rem;
            font-weight: 700;
        }}
        .badge {{
            display: inline-block;
            padding: 0.3rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 700;
            border-radius: 6px;
            letter-spacing: 0.04em;
        }}
        .badge-pass {{
            background-color: var(--emerald-900);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }}
        .badge-fail {{
            background-color: var(--rose-900);
            color: #fb7185;
            border: 1px solid rgba(251, 113, 133, 0.3);
        }}
        .section-subhead {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-dim);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .checks-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .checks-table td {{
            padding: 0.6rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.88rem;
        }}
        .check-name {{
            color: var(--text-main);
        }}
        .check-status {{
            text-align: right;
        }}
        .claim-card {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        .claim-text {{
            font-weight: 500;
            font-size: 0.92rem;
            margin-bottom: 0.6rem;
            color: #e2e8f0;
        }}
        .claim-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            background: rgba(0, 0, 0, 0.2);
            padding: 0.4rem 0.6rem;
            border-radius: 4px;
        }}
        .citation-box {{
            background-color: #0c1322;
            border-left: 3px solid var(--accent-blue);
            padding: 0.6rem 0.8rem;
            border-radius: 0 6px 6px 0;
            font-size: 0.82rem;
            margin-top: 0.5rem;
        }}
        .cit-header {{
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #93c5fd;
            margin-bottom: 0.3rem;
        }}
        .cit-quote {{
            color: #cbd5e1;
            font-style: italic;
            font-size: 0.8rem;
        }}
        .failure-card {{
            background-color: var(--rose-900);
            border: 1px solid rgba(244, 63, 94, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        .fail-type {{
            color: #f43f5e;
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }}
        .fail-msg {{
            font-size: 0.86rem;
            color: #fecdd3;
            margin-bottom: 0.4rem;
        }}
        .fail-detail {{
            font-size: 0.78rem;
            color: #fda4af;
            font-family: 'JetBrains Mono', monospace;
        }}
        .no-failures {{
            font-size: 0.86rem;
            color: #34d399;
            background: var(--emerald-900);
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid rgba(52, 211, 153, 0.2);
        }}
        footer {{
            text-align: center;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-dim);
            font-size: 0.82rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <h1>PeriodGuard</h1>
                <p>Financial Research Citation Leakage & Temporal Reliability Evaluation Harness</p>
            </div>
            <button class="btn-rerun" onclick="location.reload()">⚡ Re-run Evaluation</button>
        </header>

        <div class="case-banner">
            <div class="case-title">Target Evaluation Case (Default Fixture)</div>
            <div class="case-question">"{case.question}"</div>
            <div class="case-tags">
                <div class="case-tag"><strong>Target Entity:</strong> {case.company}</div>
                <div class="case-tag"><strong>As-Of Cutoff Date:</strong> {case.as_of_date.isoformat()}</div>
                <div class="case-tag"><strong>Target Reporting Period:</strong> {case.as_of_reporting_period}</div>
                <div class="case-tag"><strong>Expected Metric:</strong> {case.expected_metric} ({case.expected_unit})</div>
            </div>
        </div>

        <div class="comparison-grid">
            <!-- Correct Mode Card -->
            <div class="mode-column correct-mode">
                <div class="mode-header">
                    <div>
                        <div class="mode-title">Date-Filtered Mode</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Enforces as-of date (2025-05-15)</div>
                    </div>
                    {format_status_badge(correct.status.value)}
                </div>

                <div>
                    <div class="section-subhead">Deterministic Checks</div>
                    <table class="checks-table">
                        {format_checks_table(correct.checks)}
                    </table>
                </div>

                <div>
                    <div class="section-subhead">Failures & Leakage Diagnosis</div>
                    {format_failures_html(correct.failures)}
                </div>

                <div>
                    <div class="section-subhead">Synthesized Claims & Verified Citations</div>
                    {format_claims_html(correct.claims)}
                </div>
            </div>

            <!-- Broken Mode Card -->
            <div class="mode-column broken-mode">
                <div class="mode-header">
                    <div>
                        <div class="mode-title">Unfiltered Mode (Broken)</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Disables temporal boundary filter</div>
                    </div>
                    {format_status_badge(broken.status.value)}
                </div>

                <div>
                    <div class="section-subhead">Deterministic Checks</div>
                    <table class="checks-table">
                        {format_checks_table(broken.checks)}
                    </table>
                </div>

                <div>
                    <div class="section-subhead">Failures & Leakage Diagnosis</div>
                    {format_failures_html(broken.failures)}
                </div>

                <div>
                    <div class="section-subhead">Synthesized Claims & Leaked Citations</div>
                    {format_claims_html(broken.claims)}
                </div>
            </div>
        </div>

        <footer>
            PeriodGuard MVP • Reference reliability evaluation harness • Evaluates temporal safety, entity boundaries, and citation resolution.
        </footer>
    </div>
</body>
</html>
"""
