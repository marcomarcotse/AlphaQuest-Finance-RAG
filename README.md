# AlphaQuest: Finance RAG

AlphaQuest is a local financial filings assistant for the Magnificent Seven. It combines a deterministic financial fact engine with a LlamaIndex RAG fallback, so numeric questions are answered from audited statement rows instead of free-form LLM guesses.

## What Changed

- **Deterministic fact store**: extracts company, year, metric, value, source file, page, and statement row into `storage/financial_facts.json`.
- **Cloud-ready snapshot**: ships a small public demo fact store at `app_data/financial_facts.json`, so the app can run without bundled PDFs, vector storage, or Ollama.
- **Better metric support**: handles revenue, net income, operating income, and R&D / technology expense.
- **Correct total-row extraction**: avoids treating category rows like `Revenue:` or `Revenues` as totals when a `Total revenue(s)` row exists.
- **Company-specific filing logic**: maps Nvidia's January fiscal year to the project year used by the local PDF filenames.
- **Source-first answers**: Streamlit answers include source PDF, page, and row label for every number.
- **RAG fallback**: narrative filing questions still use the LlamaIndex vector index when no supported metric is detected.
- **Tests**: includes unit and PDF integration tests for the extraction bugs that used to hurt answer quality.

## Covered Companies

| Company | Ticker | Local 10-K Years |
|---|---:|---:|
| Apple | AAPL | 2023, 2024 |
| Microsoft | MSFT | 2023, 2024 |
| Alphabet | GOOG | 2023, 2024 |
| Amazon | AMZN | 2023, 2024 |
| Nvidia | NVDA | 2023, 2024 |
| Meta | META | 2023, 2024 |
| Tesla | TSLA | 2023, 2024 |

## Tech Stack

- **App**: Streamlit
- **Public demo mode**: Streamlit + bundled financial facts
- **Local RAG mode**: LlamaIndex, Llama 3 via Ollama, Ollama embeddings
- **PDF extraction**: PyMuPDF and PyMuPDF4LLM
- **Tests**: Python `unittest`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The public app only needs `streamlit` because it uses the committed fact snapshot in `app_data/`.

For local PDF ingestion and optional RAG fallback, install the full pipeline dependencies:

```bash
pip install -r requirements-ingest.txt
ollama pull llama3
```

Place 10-K PDFs in `data/` using this naming pattern:

```text
2024-AAPL-10K.pdf
2024-MSFT-10K.pdf
```

## Build Index and Facts

```bash
python ingest.py
```

This creates:

- `storage/default__vector_store.json` and related LlamaIndex files
- `storage/financial_facts.json`

To refresh the public demo snapshot after rebuilding facts:

```bash
copy storage\financial_facts.json app_data\financial_facts.json
```

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Deploy

This repository is ready for Streamlit Community Cloud:

1. Push the repo to GitHub.
2. Go to `https://share.streamlit.io`.
3. Click **Create app**.
4. Select repository `marcomarcotse/AlphaQuest-Finance-RAG`.
5. Use branch `main` and entrypoint `app.py`.
6. Pick a public app URL and deploy.

The deployed app supports deterministic financial metric questions from the bundled facts. Narrative RAG questions need local vector storage and Ollama, so they are intentionally optional in public demo mode.

## Example Questions

- Which company had the highest revenue in 2024?
- Which company had the highest net income in 2023?
- Compare Apple and Microsoft revenue in 2024.
- Who spent the most on R&D in 2024?
- What does Alphabet say about its main business?

## Verify

```bash
python -m unittest discover -s tests -v
```

The integration tests use local PDFs when `data/` is present. They verify known tricky cases, including Microsoft and Tesla total revenue, Amazon `Net income (loss)`, Nvidia fiscal-year mapping, and deterministic ranking answers.
