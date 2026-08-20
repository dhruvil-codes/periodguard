# Sample Real-World Financial PDFs for Testing & Ingestion

Here are direct links to real financial filings and earnings releases that you can download as PDFs:

### 1. Apple Inc. (AAPL)
- **Q1 FY25 Financial Statements PDF (Published: 2025-01-30, Period: Q1 FY25):**  
  👉 [https://www.apple.com/newsroom/pdfs/FY25_Q1_Consolidated_Financial_Statements.pdf](https://www.apple.com/newsroom/pdfs/FY25_Q1_Consolidated_Financial_Statements.pdf)
- **Q4 FY24 Financial Statements PDF (Published: 2024-10-31, Period: Q4 FY24):**  
  👉 [https://www.apple.com/newsroom/pdfs/FY24_Q4_Consolidated_Financial_Statements.pdf](https://www.apple.com/newsroom/pdfs/FY24_Q4_Consolidated_Financial_Statements.pdf)

### 2. Tesla Inc. (TSLA)
- **Q4 2024 Shareholder Update PDF (Published: 2025-01-29, Period: Q4 2024):**  
  👉 [https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2024-Update.pdf](https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2024-Update.pdf)

### 3. Infosys Limited (Indian Capital Markets)
- **Q3 FY25 Financial Results Press Release PDF (Published: 2025-01-16, Period: Q3 FY25):**  
  👉 [https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q3/documents/ifrs-usd-press-release.pdf](https://www.infosys.com/investors/reports-filings/quarterly-results/2024-2025/q3/documents/ifrs-usd-press-release.pdf)

---

### 💡 The Perfect "Future Leak" Test Pairing:
To demonstrate **Future-Period Leakage** live with real PDFs:
1. Download **Apple Q4 FY24** (Published: *2024-10-31*) and **Apple Q1 FY25** (Published: *2025-01-30*).
2. Ingest both files into PeriodGuard.
3. Set your chat **As-Of Date** to `2024-11-15` (between the two reports).
4. Ask: *"What was Apple's total revenue and gross margin?"*
5. Show that PeriodGuard strictly cites Q4 FY24 (valid as of Nov 2024) and rejects Q1 FY25 as a temporal leak!
