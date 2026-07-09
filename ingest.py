from __future__ import annotations

import logging
import sys
from pathlib import Path

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from finance_core import FACT_STORE_PATH, build_fact_store, get_file_metadata


DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("alphaquest.ingest")


def _page_is_financially_interesting(text: str) -> bool:
    lower = text.lower()
    return any(
        term in lower
        for term in (
            "net income",
            "net earnings",
            "total net sales",
            "total revenue",
            "revenue",
            "consolidated statements",
            "income statements",
            "operations",
            "research and development",
            "operating income",
            "income from operations",
        )
    )


def load_pdf_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    try:
        import fitz
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError("PyMuPDF and pymupdf4llm are required for ingestion.") from exc

    documents: list[Document] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        logger.info("Converting %s to Markdown", pdf_path.name)
        try:
            pdf_doc = fitz.open(pdf_path)
            pages: list[str] = []

            for page_index, page in enumerate(pdf_doc):
                page_text = page.get_text()
                if _page_is_financially_interesting(page_text):
                    try:
                        page_markdown = pymupdf4llm.to_markdown(
                            str(pdf_path),
                            pages=[page_index],
                            show_progress=False,
                        )
                    except Exception:
                        logger.exception(
                            "Falling back to plain text for %s page %s",
                            pdf_path.name,
                            page_index + 1,
                        )
                        page_markdown = page_text
                else:
                    page_markdown = page_text

                pages.append(f"<!-- Page {page_index + 1} -->\n{page_markdown}")

            page_count = len(pdf_doc)
            pdf_doc.close()
            metadata = get_file_metadata(pdf_path)
            metadata["file_name"] = pdf_path.name
            documents.append(Document(text="\n\n".join(pages), metadata=metadata))
            logger.info(
                "Loaded %s as %s (%s), %s pages",
                pdf_path.name,
                metadata["company"],
                metadata["year"],
                page_count,
            )
        except Exception:
            logger.exception("Error processing %s", pdf_path)

    return documents


def build_vector_index(documents: list[Document]) -> VectorStoreIndex:
    logger.info("Creating vector index")
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=150)
    nodes = node_parser.get_nodes_from_documents(documents)
    return VectorStoreIndex(nodes)


def main() -> None:
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.pdf")):
        logger.error("No PDF files found in %s", DATA_DIR)
        return

    Settings.llm = Ollama(model="llama3", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="llama3")

    documents = load_pdf_documents(DATA_DIR)
    if not documents:
        logger.error("No documents were loaded.")
        return

    index = build_vector_index(documents)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    logger.info("Index persisted to %s", STORAGE_DIR)

    facts = build_fact_store(DATA_DIR, FACT_STORE_PATH)
    logger.info("Financial fact store written to %s (%s facts)", FACT_STORE_PATH, len(facts))

    try:
        query_engine = index.as_query_engine(similarity_top_k=4)
        response = query_engine.query("What is the main business of Alphabet Inc?")
        logger.info("Smoke test response: %s", response)
    except Exception:
        logger.exception("Smoke test failed. The index and fact store were still created.")


if __name__ == "__main__":
    main()
