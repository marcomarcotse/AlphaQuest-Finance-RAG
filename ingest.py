import logging
import sys
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_file_metadata(file_path):
    """
    Extracts metadata from the filename.
    Expected format: YYYY-Company-Type.pdf or similar.
    Example: 2023-Alphabet-10K.pdf
    """
    filename = os.path.basename(file_path).lower()
    metadata = {}
    
    # Keyword matching for companies (including tickers)
    companies_map = {
        "alphabet": "Alphabet", "google": "Alphabet", "goog": "Alphabet",
        "amazon": "Amazon", "amzn": "Amazon",
        "apple": "Apple", "aapl": "Apple",
        "meta": "Meta", "facebook": "Meta",
        "microsoft": "Microsoft", "msft": "Microsoft",
        "nvidia": "Nvidia", "nvda": "Nvidia",
        "tesla": "Tesla", "tsla": "Tesla"
    }
    
    for key, name in companies_map.items():
        if key in filename:
            metadata["company"] = name
            break
    else:
        metadata["company"] = "Unknown"
        
    # extract year (4 digits)
    import re
    year_match = re.search(r"20\d{2}", filename)
    if year_match:
        metadata["year"] = year_match.group(0)
    else:
        metadata["year"] = "Unknown"
        
    return metadata

def main():
    # 1. Setup local models via Ollama
    Settings.llm = Ollama(model="llama3", request_timeout=120.0)
    Settings.embed_model = OllamaEmbedding(model_name="llama3")

    # 2. Load data with metadata
    print("Loading documents from ./data with metadata...")
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("Error: No files found in ./data directory.")
        return

    # 2. Load data with metadata using PyMuPDF4LLM for Markdown conversion
    print("Loading documents from ./data with metadata and PyMuPDF4LLM...")
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("Error: No files found in ./data directory.")
        return

    import pymupdf4llm
    import pymupdf
    from llama_index.core import Document

    documents = []
    data_dir = "./data"
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            try:
                print(f"Converting {filename} to Markdown with enhanced table extraction...")
                
                # Open PDF and process page by page for better table capture
                pdf_doc = pymupdf.open(file_path)
                all_pages_md = []
                
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    page_text = page.get_text()
                    
                    # Check if page likely contains income statement
                    lower_text = page_text.lower()
                    is_financial_page = any(term in lower_text for term in [
                        'net income', 'net earnings', 'consolidated statements',
                        'operations', 'income statement', 'comprehensive income'
                    ])
                    
                    if is_financial_page:
                        # Use pymupdf4llm for this page with table extraction
                        try:
                            page_md = pymupdf4llm.to_markdown(
                                file_path, 
                                pages=[page_num],
                                show_progress=False
                            )
                            all_pages_md.append(f"<!-- Page {page_num + 1} -->\n{page_md}")
                        except:
                            all_pages_md.append(f"<!-- Page {page_num + 1} -->\n{page_text}")
                    else:
                        # Regular text extraction for non-financial pages
                        all_pages_md.append(f"<!-- Page {page_num + 1} -->\n{page_text}")
                
                num_pages = len(pdf_doc)  # Save before closing
                pdf_doc.close()
                md_text = "\n\n".join(all_pages_md)
                
                # Extract metadata
                meta = get_file_metadata(file_path)
                meta["file_name"] = filename
                
                # Create Document
                doc = Document(text=md_text, metadata=meta)
                documents.append(doc)
                print(f"-> Loaded {filename} as {meta['company']} ({meta['year']}) - {num_pages} pages")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()

    print(f"Loaded {len(documents)} documents (converted to Markdown).")

    # 3. Create Index with better chunking for tables
    print("Creating index with larger chunks (2048 tokens) for table preservation...")
    from llama_index.core.node_parser import SentenceSplitter
    
    node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes)

    # 4. Save Index for later use
    index.storage_context.persist(persist_dir="./storage")
    print("Index created and persisted to ./storage")

    # 5. Quick Test
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the main business of Alphabet Inc?")
    print("\nTest Query Response:")
    print(response)

if __name__ == "__main__":
    main()
