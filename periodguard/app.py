from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query
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

evaluator_deterministic = Evaluator(use_llm=False)
evaluator_llm = Evaluator(use_llm=True)
_latest_reports_cache: Dict[str, EvaluationReport] = {}


def get_evaluation_data(use_llm: bool = False) -> Dict[str, Any]:
    global _latest_reports_cache
    ev = evaluator_llm if use_llm else evaluator_deterministic
    reports = ev.run_both_modes()
    _latest_reports_cache = reports
    return {
        "engine": "llm" if (use_llm and ev.llm_adapter.is_available) else "deterministic",
        "case": get_default_case().model_dump(mode="json"),
        "correct_mode": reports["correct_mode"].model_dump(mode="json"),
        "broken_mode": reports["broken_mode"].model_dump(mode="json"),
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "PeriodGuard Evaluation Harness"}


@app.post("/evaluate")
def run_evaluate_api(use_llm: bool = Query(False)) -> Dict[str, Any]:
    return get_evaluation_data(use_llm=use_llm)


@app.get("/report")
def get_report_api(use_llm: bool = Query(False)) -> Dict[str, Any]:
    return get_evaluation_data(use_llm=use_llm)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PeriodGuard • Financial Research Reliability & Citation Leakage Harness</title>
  
  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">

  <style>
    :root {
      --bg-base: #07090e;
      --bg-surface: #0e131f;
      --bg-surface-elevated: #141c2e;
      --bg-surface-hover: #1b263d;
      --border-subtle: rgba(255, 255, 255, 0.07);
      --border-strong: rgba(255, 255, 255, 0.14);
      --border-accent: rgba(99, 102, 241, 0.35);

      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --text-dim: #475569;

      --emerald-accent: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.16);
      --emerald-border: rgba(16, 185, 129, 0.35);
      --emerald-badge-bg: rgba(6, 78, 59, 0.45);
      --emerald-badge-text: #34d399;

      --rose-accent: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.16);
      --rose-border: rgba(244, 63, 94, 0.38);
      --rose-badge-bg: rgba(136, 19, 55, 0.45);
      --rose-badge-text: #fb7185;

      --indigo-accent: #6366f1;
      --cyan-accent: #06b6d4;
      --amber-accent: #f59e0b;

      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 18px;
      --radius-full: 9999px;

      --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: var(--font-body);
      background-color: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.55;
      background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.12), transparent),
        radial-gradient(circle at 10% 20%, rgba(6, 182, 212, 0.05), transparent),
        radial-gradient(circle at 90% 80%, rgba(244, 63, 94, 0.04), transparent);
      background-attachment: fixed;
      padding: 2.5rem 1.5rem 4rem;
    }

    .app-container {
      max-width: 1320px;
      margin: 0 auto;
    }

    /* Top Navigation Header */
    .top-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1.75rem;
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

    .version-tag {
      font-size: 0.7rem;
      font-family: var(--font-mono);
      font-weight: 600;
      background: rgba(99, 102, 241, 0.18);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.35);
      padding: 0.2rem 0.5rem;
      border-radius: var(--radius-full);
      -webkit-text-fill-color: #a5b4fc;
    }

    .brand-text p {
      font-size: 0.88rem;
      color: var(--text-secondary);
      margin-top: 0.15rem;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.9rem;
    }

    .engine-toggle-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: var(--bg-surface);
      border: 1px solid var(--border-strong);
      padding: 0.45rem 0.85rem;
      border-radius: var(--radius-full);
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-secondary);
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.4);
    }

    .engine-indicator-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--cyan-accent);
      box-shadow: 0 0 10px var(--cyan-accent);
      animation: pulse-dot 2s infinite ease-in-out;
    }

    @keyframes pulse-dot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
    }

    .btn {
      font-family: var(--font-body);
      font-weight: 600;
      font-size: 0.88rem;
      padding: 0.6rem 1.2rem;
      border-radius: var(--radius-md);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
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

    /* Target Case Hero Card */
    .case-hero {
      background: linear-gradient(145deg, rgba(20, 28, 46, 0.8) 0%, rgba(14, 19, 31, 0.95) 100%);
      border: 1px solid var(--border-accent);
      border-radius: var(--radius-lg);
      padding: 1.75rem 2rem;
      margin-bottom: 2.25rem;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
      position: relative;
      overflow: hidden;
    }

    .case-hero::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
    }

    .case-header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }

    .case-eyebrow {
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--cyan-accent);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .case-question-text {
      font-family: var(--font-display);
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #ffffff;
      line-height: 1.45;
      margin-bottom: 1.35rem;
    }

    .case-chips-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .meta-chip {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border-subtle);
      padding: 0.45rem 0.85rem;
      border-radius: var(--radius-md);
      font-size: 0.82rem;
      color: var(--text-secondary);
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }

    .meta-chip strong {
      color: var(--text-primary);
      font-weight: 600;
    }

    .meta-chip.highlight-date {
      border-color: rgba(6, 182, 212, 0.35);
      background: rgba(6, 182, 212, 0.08);
    }
    .meta-chip.highlight-date strong {
      color: #38bdf8;
      font-family: var(--font-mono);
    }

    /* Section Banner Axiom */
    .axiom-banner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
      padding: 0 0.5rem;
    }

    .axiom-banner h2 {
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .axiom-quote {
      font-size: 0.84rem;
      font-style: italic;
      color: var(--text-muted);
      font-family: var(--font-mono);
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

    /* Badges */
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
      color: var(--text-dim);
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
      transition: background 0.15s ease;
    }

    .check-item:hover {
      background: rgba(255, 255, 255, 0.03);
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

    /* Synthesized Claims & Citations */
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

    .attr-tag b {
      color: #ffffff;
    }

    /* Interactive Citation Box Button */
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

    /* Slide-over Citation Inspector Drawer */
    .drawer-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(3, 7, 18, 0.7);
      backdrop-filter: blur(4px);
      z-index: 998;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease;
    }

    .drawer-backdrop.active {
      opacity: 1;
      pointer-events: auto;
    }

    .inspector-drawer {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: 100%;
      max-width: 520px;
      background: #0d121f;
      border-left: 1px solid var(--border-strong);
      z-index: 999;
      box-shadow: -12px 0 40px rgba(0, 0, 0, 0.6);
      transform: translateX(100%);
      transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
    }

    .inspector-drawer.active {
      transform: translateX(0);
    }

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

    .btn-close-drawer {
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

    .btn-close-drawer:hover {
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
      left: 4px;
      top: 6px;
      bottom: 6px;
      width: 2px;
      background: var(--border-strong);
    }

    .t-node {
      position: relative;
      font-size: 0.82rem;
    }

    .t-node::after {
      content: '';
      position: absolute;
      left: -1.2rem;
      top: 4px;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--cyan-accent);
      box-shadow: 0 0 8px var(--cyan-accent);
    }

    .t-node.future::after {
      background: var(--rose-accent);
      box-shadow: 0 0 10px var(--rose-accent);
    }

    .t-node-title {
      font-weight: 600;
      color: #ffffff;
    }

    .t-node-desc {
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-secondary);
    }

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

    /* Footer */
    footer {
      border-top: 1px solid var(--border-subtle);
      padding-top: 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--text-dim);
      font-size: 0.82rem;
      flex-wrap: wrap;
      gap: 1rem;
    }

    footer span strong {
      color: var(--text-muted);
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
          <h1>PeriodGuard <span class="version-tag">MVP 1.0</span></h1>
          <p>Financial Research Reliability & Future-Period Citation Leakage Harness</p>
        </div>
      </div>
      <div class="header-actions">
        <div class="engine-toggle-badge">
          <span class="engine-indicator-dot"></span>
          <span id="engineLabel">Engine: Deterministic Fixture</span>
        </div>
        <button id="btnRerun" class="btn btn-primary" onclick="triggerEvaluation()">
          ⚡ Re-run Evaluation
        </button>
      </div>
    </header>

    <!-- Case Hero Banner -->
    <section class="case-hero">
      <div class="case-header-row">
        <div class="case-eyebrow">
          <span>🎯</span> Target Evaluation Case (Default Fixture)
        </div>
      </div>
      <div class="case-question-text" id="caseQuestion">
        Loading target evaluation case...
      </div>
      <div class="case-chips-grid">
        <div class="meta-chip"><span>🏢 Target Entity:</span> <strong id="caseCompany">—</strong></div>
        <div class="meta-chip highlight-date"><span>📅 As-Of Boundary:</span> <strong id="caseAsOfDate">—</strong></div>
        <div class="meta-chip"><span>📊 Target Period:</span> <strong id="casePeriod">—</strong></div>
        <div class="meta-chip"><span>📈 Expected Metric:</span> <strong id="caseMetric">—</strong></div>
      </div>
    </section>

    <!-- Axiom Banner -->
    <div class="axiom-banner">
      <h2>Retrieval Mode Comparison</h2>
      <span class="axiom-quote">"A citation exists" ≠ "The cited answer is safe to use"</span>
    </div>

    <!-- Side-by-Side Dual Comparison Grid -->
    <main class="comparison-grid" id="comparisonGrid">
      <!-- Correct Mode Card (Injected via JS) -->
      <!-- Broken Mode Card (Injected via JS) -->
    </main>

    <!-- Footer -->
    <footer>
      <span><strong>PeriodGuard MVP</strong> • Evaluates temporal safety, entity boundaries, and citation traceability.</span>
      <span>Reference reliability test harness</span>
    </footer>

  </div>

  <!-- Slide-over Citation Inspector Drawer -->
  <div class="drawer-backdrop" id="drawerBackdrop" onclick="closeDrawer()"></div>
  <aside class="inspector-drawer" id="inspectorDrawer">
    <div class="drawer-top-bar">
      <div>
        <div class="section-label" style="margin-bottom: 0.2rem;">Provenance Inspector</div>
        <h3 id="drawerDocId">Document Metadata</h3>
      </div>
      <button class="btn-close-drawer" onclick="closeDrawer()">×</button>
    </div>
    <div class="drawer-body" id="drawerBody">
      <!-- Content populated dynamically -->
    </div>
  </aside>

  <!-- Initial State Injection -->
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
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    function renderDashboard(data) {
      // 1. Populate Case Hero
      const c = data.case;
      document.getElementById('caseQuestion').textContent = `"${c.question}"`;
      document.getElementById('caseCompany').textContent = c.company;
      document.getElementById('caseAsOfDate').textContent = c.as_of_date;
      document.getElementById('casePeriod').textContent = c.as_of_reporting_period;
      document.getElementById('caseMetric').textContent = `${c.expected_metric} (${c.expected_unit})`;
      document.getElementById('engineLabel').textContent = `Engine: ${data.engine === 'llm' ? 'Live LLM Adapter' : 'Deterministic Fixture'}`;

      // 2. Render Cards
      const grid = document.getElementById('comparisonGrid');
      grid.innerHTML = `
        ${renderModeCard(data.correct_mode, false, data)}
        ${renderModeCard(data.broken_mode, true, data)}
      `;
    }

    function renderModeCard(report, isBroken, data) {
      const isPass = report.status === 'PASS';
      const cardClass = isBroken ? 'broken-card' : 'correct-card';
      const title = isBroken ? 'Unfiltered Mode (Broken)' : 'Date-Filtered Mode';
      const subtitle = isBroken ? 'Disables temporal filter · allows future leaks' : 'Enforces as-of date (2025-05-15)';
      const badgeClass = isPass ? 'pass' : 'fail';
      const badgeText = isPass ? '✓ PASS' : '✗ FAIL';

      // Checks list
      const checksHtml = Object.entries(report.checks).map(([key, val]) => `
        <div class="check-item">
          <span class="check-title">${checkLabels[key] || key}</span>
          <span class="mini-status ${val.toLowerCase()}">${val}</span>
        </div>
      `).join('');

      // Diagnosis / Failure Alert
      let alertHtml = '';
      if (!isBroken || report.failures.length === 0) {
        alertHtml = `
          <div class="pass-diagnosis-box">
            <span>✓</span>
            <div>No temporal leakage detected. Future documents strictly gated by publication date.</div>
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

      // Claims & Citations
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
            <div class="section-label">Synthesized Claims & Traceable Citations</div>
            <div class="claims-stack">${claimsHtml}</div>
          </div>
        </div>
      `;
    }

    function openInspector(docId, quotedText) {
      const allDocs = appState.correct_mode.retrieved_documents.concat(appState.broken_mode.retrieved_documents);
      const doc = allDocs.find(d => d.id === docId) || {
        id: docId,
        company: 'Acme Industries',
        doc_type: 'Financial Document',
        reporting_period: 'Q4 FY25',
        publication_date: '2025-05-10',
        page: 1,
        text: quotedText,
        source_url: 'https://example.com'
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
            <tr><td>Provenance</td><td style="font-size: 0.75rem; word-break: break-all;"><a href="${escapeHtml(doc.source_url)}" target="_blank" style="color: #93c5fd;">${escapeHtml(doc.source_url)}</a></td></tr>
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

    async function triggerEvaluation() {
      const btn = document.getElementById('btnRerun');
      btn.disabled = true;
      btn.innerHTML = '⚡ Running...';
      try {
        const resp = await fetch('/evaluate', { method: 'POST' });
        if (resp.ok) {
          appState = await resp.json();
          renderDashboard(appState);
        }
      } catch (err) {
        console.error('Failed to rerun evaluation', err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Re-run Evaluation';
      }
    }

    // Initial render
    renderDashboard(appState);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def render_dashboard(use_llm: bool = Query(False)) -> str:
    data = get_evaluation_data(use_llm=use_llm)
    json_str = json.dumps(data).replace("</", "<\\/")
    return DASHBOARD_HTML.replace("__INITIAL_DATA__", json_str)
