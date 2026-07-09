from __future__ import annotations

from pathlib import Path

import streamlit as st

from finance_core import (
    FACT_STORE_PATH,
    METRICS,
    FinancialFact,
    answer_financial_question,
    available_companies,
    available_years,
    build_fact_store,
    load_or_build_facts,
)


DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")


st.set_page_config(page_title="AlphaQuest: Finance RAG", layout="wide")
st.title("AlphaQuest")
st.caption("Financial fact engine and RAG assistant for Magnificent Seven 10-K filings.")


@st.cache_resource(show_spinner=False)
def get_fact_store() -> list[FinancialFact]:
    return load_or_build_facts(DATA_DIR, FACT_STORE_PATH)


@st.cache_resource(show_spinner=False)
def load_rag_index():
    if not STORAGE_DIR.exists():
        return None

    try:
        from llama_index.core import Settings, StorageContext, load_index_from_storage
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
    except ImportError:
        return None

    Settings.llm = Ollama(model="llama3", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="llama3")
    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    return load_index_from_storage(storage_context)


def answer_with_rag(question: str) -> str:
    try:
        from llama_index.core import PromptTemplate
    except ImportError:
        return (
            "I could not detect a supported financial metric in that question, "
            "and the optional LlamaIndex RAG dependencies are not installed in this deployment."
        )

    index = load_rag_index()
    if index is None:
        return (
            "I could not detect a supported financial metric in that question, "
            "and the LlamaIndex storage folder is missing. Run `python ingest.py` "
            "after placing PDFs in `data/`."
        )

    qa_template = PromptTemplate(
        "You are AlphaQuest, a careful financial filings assistant.\n"
        "Use only the context below. If the answer is not in the context, say so.\n"
        "When you cite numbers, include the company, period, and wording from the filing.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )
    query_engine = index.as_query_engine(similarity_top_k=6, text_qa_template=qa_template)
    return str(query_engine.query(question))


def facts_as_rows(facts: list[FinancialFact]) -> list[dict[str, object]]:
    return [
        {
            "Company": fact.company,
            "Year": fact.year,
            "Metric": fact.metric_label,
            "Value ($M)": fact.value,
            "Source": f"{fact.source_file}, p. {fact.page}",
            "Row": fact.row_label,
        }
        for fact in facts
    ]


facts = get_fact_store()

with st.sidebar:
    st.header("Fact Store")
    if facts:
        st.metric("Facts", len(facts))
        st.write("**Companies:** " + ", ".join(available_companies(facts)))
        st.write("**Years:** " + ", ".join(available_years(facts)))
        st.write("**Metrics:** " + ", ".join(spec.label for spec in METRICS.values()))
    else:
        st.warning("No facts found. Add PDFs to `data/` and rebuild.")

    if st.button("Rebuild facts from PDFs", use_container_width=True):
        with st.spinner("Extracting audited statement facts from PDFs..."):
            build_fact_store(DATA_DIR, FACT_STORE_PATH)
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if facts:
        with st.expander("Preview extracted facts"):
            st.dataframe(facts_as_rows(facts), hide_index=True, use_container_width=True)


if not facts:
    st.error(
        "No financial facts were loaded. Run `python ingest.py`, or use the sidebar "
        "button after adding 10-K PDFs to `data/`."
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask about revenue, net income, operating income, R&D, or filing context..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing filings..."):
        fact_answer = answer_financial_question(prompt, facts)
        if fact_answer.used_facts:
            response = fact_answer.markdown
        else:
            response = answer_with_rag(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)
        if fact_answer.used_facts and fact_answer.facts:
            with st.expander("Source rows"):
                st.dataframe(
                    facts_as_rows(list(fact_answer.facts)),
                    hide_index=True,
                    use_container_width=True,
                )

    st.session_state.messages.append({"role": "assistant", "content": response})
