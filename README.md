# AlphaQuest: Finance RAG

AlphaQuest is a local financial filings assistant for the Magnificent Seven. It combines a deterministic financial fact engine with a LlamaIndex RAG fallback, so numeric questions are answered from audited statement rows instead of free-form LLM guesses.

## What Changed

- **Deterministic fact store**: extracts company, year, metric, value, source file, page, and statement row into `storage/financial_facts.json`.
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
- **RAG**: LlamaIndex
- **LLM**: Llama 3 via Ollama
- **Embeddings**: Ollama embeddings
- **PDF extraction**: PyMuPDF and PyMuPDF4LLM
- **Tests**: Python `unittest`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
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

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

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
