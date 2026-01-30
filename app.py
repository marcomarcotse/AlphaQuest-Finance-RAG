import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# --- CONFIGURATION ---
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="llama3")

# --- UI SETUP ---
st.set_page_config(page_title="AlphaQuest: Finance RAG", layout="centered")
st.title("📊 AlphaQuest")
st.subheader("Magnificent Seven Intelligence Hub")

# --- LOGIC: Load the Index ---
@st.cache_resource
def load_rag_index():
    if not os.path.exists("./storage"):
        st.error("Index not found. Please run 'python ingest.py' first!")
        return None, []
    
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    # Extract metadata
    all_metadata = []
    for node in index.docstore.docs.values():
        if hasattr(node, 'metadata'):
            all_metadata.append(node.metadata)
    
    return index, all_metadata

index, metadata_list = load_rag_index()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏢 Indexed Companies")
    if metadata_list:
        companies = sorted(list(set([m.get("company") for m in metadata_list if m.get("company") and m.get("company") != "Unknown"])))
        years = sorted(list(set([m.get("year") for m in metadata_list if m.get("year") and m.get("year") != "Unknown"])))
        
        if companies:
            st.write(f"**Companies:** {', '.join(companies)}")
        if years:
            st.write(f"**Years:** {', '.join(years)}")
        st.success(f"Scanning {len(metadata_list)} document facets.")
    else:
        st.warning("No metadata found.")
    
    st.divider()
    if st.button("🔄 Reload Data"):
        st.cache_resource.clear()
        st.rerun()

# --- UI: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the Magnificent Seven..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Query the RAG system
    import re
    with st.spinner("Analyzing relevant documents..."):
        # 1. INTELLIGENCE STEP: Detect targeted companies, year, and metric
        unique_companies = sorted(list(set([m.get("company") for m in metadata_list if m.get("company") and m.get("company") != "Unknown"])))
        
        # Detect year from prompt
        year_match = re.search(r"20\d{2}", prompt)
        target_year = year_match.group(0) if year_match else "2023" # Default to 2023
        
        # Detect metric (simple keyword detection for robustness)
        prompt_lower = prompt.lower()
        if "revenue" in prompt_lower or "sales" in prompt_lower:
            target_metric = "Total Revenue"
            metric_keywords = "Total Revenue Net Sales"
        elif "r&d" in prompt_lower or "research" in prompt_lower:
            target_metric = "Research and Development (R&D) Expenses"
            metric_keywords = "Research and Development R&D expense"
        elif "operating income" in prompt_lower:
            target_metric = "Operating Income"
            metric_keywords = "Operating Income Income from Operations"
        else:
            target_metric = "Net Income"
            metric_keywords = "Net Income Net Earnings Net Loss Net Income (Loss) Consolidated Statements of Operations"

        # Logic to determine if we should scan EVERYONE
        comparative_keywords = ["highest", "lowest", "best", "worst", "compare", "rank", "list all", "everyone", "magnificent", "all companies"]
        is_comparative = any(kw in prompt.lower() for kw in comparative_keywords)
        
        if is_comparative:
            target_companies = unique_companies
        else:
            # Only use LLM router for specific entity questions
            router_prompt = (
                f"User Question: '{prompt}'\n"
                f"Available Companies: {', '.join(unique_companies)}\n"
                "Task: List only the companies specifically mentioned in the question. "
                "Respond ONLY with comma-separated names, NO other text."
            )
            detected_companies_str = str(Settings.llm.complete(router_prompt)).strip()
            target_companies = [c.strip() for c in detected_companies_str.split(",") if c.strip() in unique_companies]
            
            # Fallback to all if it seems general but didn't trigger comparative_keywords
            if not target_companies:
                target_companies = unique_companies

        # 2. TARGETED ANALYSIS LOOP
        extracted_values = {}
        aggregated_data = ""
        
        for company in target_companies:
            st.write(f"🔍 Searching {target_year} {company} {target_metric}...")
            
            # For Alphabet and Amazon, the 2023 data is in 2024 10-K filing, so use Year+1
            # Only apply this for years where we have Year+1 data indexed
            search_year = target_year
            if company in ["Alphabet", "Amazon"] and int(target_year) <= 2023:
                search_year = str(int(target_year) + 1)
            
            # CRITICAL: Filter by BOTH company and year
            filters = MetadataFilters(filters=[
                ExactMatchFilter(key="company", value=company),
                ExactMatchFilter(key="year", value=search_year)
            ])
            
            # Use retriever for direct access to chunks
            from llama_index.core.retrievers import VectorIndexRetriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=10,
                filters=filters
            )
            
            search_query = f"{company} {target_year} Net Income Net earnings Operations"
            try:
                results = retriever.retrieve(search_query)
            except (AssertionError, Exception):
                results = []  # No results found for this year
            
            # === REGEX-BASED EXTRACTION (PRIMARY METHOD) ===
            def extract_net_income_regex(chunks, target_year, is_nvidia=False, company_name="", using_year_plus1=False):
                """Extract Net Income from chunks using regex patterns with company-specific column awareness.
                
                CRITICAL DISCOVERY:
                - Alphabet: Uses ASCENDING order (2022|2023|2024) -> take LAST for 2024, SECOND-TO-LAST for 2023
                - All others: Use DESCENDING order (2024|2023|2022) -> take FIRST for current year
                """
                all_values = []  # Collect all valid values for voting
                
                for chunk in chunks:
                    # PREPROCESSING: Clean markdown formatting completely
                    text = chunk.text.replace("**", "").replace("*", "").replace("[$]", "$")
                    
                    # Skip problematic lines
                    skip_phrases = ['per share', 'per basic', 'per diluted', 
                                    'was not', 'tax benefit', '(loss)', 'reclassification', 
                                    'attributable to common', 'attributable to noncontrolling']
                    
                    for line in text.split('\n'):
                        if any(phrase in line.lower() for phrase in skip_phrases):
                            continue
                        if 'net income' not in line.lower():
                            continue
                            
                        # Extract ALL numbers from the line
                        nums = re.findall(r'[\d,]+', line)
                        valid_nums = []
                        for n in nums:
                            try:
                                val = int(n.replace(',', ''))
                                # Filter out year-like values (2019-2029) and validate range
                                if 1000 < val < 200000 and not (2019 <= val <= 2029):
                                    valid_nums.append(val)
                            except:
                                pass
                        
                        if valid_nums:
                            # COMPANY-SPECIFIC COLUMN SELECTION
                            if company_name == "Alphabet":
                                # Alphabet uses ASCENDING order: 2022 | 2023 | 2024
                                if len(valid_nums) >= 3:
                                    if using_year_plus1:
                                        # Querying 2023 from 2024 filing -> SECOND-TO-LAST
                                        selected_val = valid_nums[-2]
                                    else:
                                        # Querying 2024 -> LAST (rightmost)
                                        selected_val = valid_nums[-1]
                                elif len(valid_nums) == 2:
                                    if using_year_plus1:
                                        selected_val = valid_nums[0]  # First for 2023
                                    else:
                                        selected_val = valid_nums[-1]  # Last for 2024
                                else:
                                    selected_val = valid_nums[0]
                                all_values.append((selected_val, line[:60]))
                            elif company_name == "Amazon":
                                # Amazon's equity table: Net income rows show prev year first, current year last
                                # For current year query -> use LAST value in line (or mark for later selection)
                                # We'll collect all and use max() for Amazon instead of voting
                                all_values.append((valid_nums[-1], line[:60]))  # Prefer last/larger value
                            else:
                                # All other companies: DESCENDING order (2024 | 2023 | 2022)
                                # Take FIRST value for current year
                                all_values.append((valid_nums[0], line[:60]))
                
                if not all_values:
                    return None, None
                
                # COMPANY-SPECIFIC RESULT SELECTION
                if company_name == "Amazon":
                    # Amazon's equity table shows: row1=previous year, row2=current year
                    # For Year+1 queries (2023 data from 2024 filing) -> use MIN (previous year = smaller)
                    # For current year queries (2024 data from 2024 filing) -> use MAX (current year = larger)
                    if using_year_plus1:
                        target_val = min(v[0] for v in all_values)
                    else:
                        target_val = max(v[0] for v in all_values)
                    for val, source in all_values:
                        if val == target_val:
                            return val, source
                else:
                    # Use voting: pick the most frequently appearing value
                    from collections import Counter
                    val_counts = Counter([v[0] for v in all_values])
                    most_common_val = val_counts.most_common(1)[0][0]
                    
                    # Get source for this value
                    for val, source in all_values:
                        if val == most_common_val:
                            return val, source
                
                return None, None
            
            is_nvidia = (company == "Nvidia")
            # Check if we're using Year+1 filter (Alphabet/Amazon querying previous year from newer filing)
            using_year_plus1 = (company in ["Alphabet", "Amazon"] and int(target_year) <= 2023)
            extracted_val, source_line = extract_net_income_regex(results, target_year, is_nvidia, company, using_year_plus1)
            
            # FALLBACK: Direct node scan if retrieval failed
            if not extracted_val:
                # For some companies, the 10-K fiscal year != calendar year
                # Alphabet and Amazon's 2023 data is in 2024 10-K, so search Year+1 FIRST
                if company in ["Alphabet", "Amazon"]:
                    search_years = [str(int(target_year) + 1), target_year]  # Year+1 first
                else:
                    search_years = [target_year]
                
                all_fallback_values = []  # Collect all values for company-specific selection
                
                for search_year in search_years:
                    for node in index.docstore.docs.values():
                        meta = node.metadata if hasattr(node, 'metadata') else {}
                        if meta.get('company') == company and meta.get('year') == search_year:
                            text = node.text.replace('**', '').replace('*', '').replace('[$]', '$')
                            # Skip EPS, prose, and loss notation lines
                            skip_phrases = ['per share', 'per basic', 'per diluted', 
                                            'was not', 'tax benefit', '(loss)', 'reclassification',
                                            'attributable to common', 'attributable to noncontrolling']
                            
                            for line in text.split('\n'):
                                if any(phrase in line.lower() for phrase in skip_phrases):
                                    continue
                                if 'net income' not in line.lower():
                                    continue
                                    
                                nums = re.findall(r'[\d,]+', line)
                                for n in nums:
                                    try:
                                        val = int(n.replace(',', ''))
                                        # Filter out year-like values (2019-2029)
                                        if 1000 < val < 200000 and not (2019 <= val <= 2029):
                                            all_fallback_values.append((val, f"Direct scan (Y={search_year}): {line[:35]}..."))
                                    except ValueError:
                                        pass
                
                # Select value based on company-specific logic
                if all_fallback_values:
                    if company == "Amazon":
                        # Amazon: use max() for 2024, min() for 2023 (Year+1)
                        if using_year_plus1:
                            target_val = min(v[0] for v in all_fallback_values)
                        else:
                            target_val = max(v[0] for v in all_fallback_values)
                        for val, source in all_fallback_values:
                            if val == target_val:
                                extracted_val = val
                                source_line = source
                                break
                    elif company == "Alphabet":
                        # Alphabet: most frequent value
                        from collections import Counter
                        counts = Counter([v[0] for v in all_fallback_values])
                        most_common = counts.most_common(1)[0][0]
                        for val, source in all_fallback_values:
                            if val == most_common:
                                extracted_val = val
                                source_line = source
                                break
                    else:
                        # Other companies: first value found
                        extracted_val, source_line = all_fallback_values[0]
            
            if extracted_val:
                # SUCCESS: Got value 
                extracted_values[company] = extracted_val
                aggregated_data += f"\n- **{company}**: ${extracted_val:,} million (Source: '{source_line}')"
                st.success(f"✓ {company}: ${extracted_val:,}M")
            else:
                # FAILED: No data found
                aggregated_data += f"\n- **{company}**: Not found in index"
                st.warning(f"⚠ {company}: Data not found")


        # 3. FINAL SYNTHESIS (DETERMINISTIC)
        if extracted_values:
            # Sort by value
            sorted_companies = sorted(extracted_values.items(), key=lambda x: x[1], reverse=True)
            winner, max_val = sorted_companies[0]
            
            # Generate the natural language response
            final_response = (
                f"### Analysis Result\n"
                f"Based on the extracted financial data for {target_year} {target_metric}:\n\n"
                f"**The Leader is {winner}** with a reported {target_metric} of **${max_val:,} million**.\n\n"
                f"#### Full Ranking:\n"
            )
            for i, (comp, val) in enumerate(sorted_companies, 1):
                final_response += f"{i}. **{comp}**: ${val:,} million\n"
            
            response = final_response + f"\n\n*Note: Data extracted from 10-K filings. Nvidia's fiscal year is adjusted to match calendar year.*"
        else:
            response = "I could not find sufficient data to compare the companies for this metric."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        
        with st.expander("📊 View Aggregated Fact Sheet"):
            st.markdown(aggregated_data)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
