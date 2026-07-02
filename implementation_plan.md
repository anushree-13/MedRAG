# Agentic Hybrid RAG — Implementation Plan

A multi-phase, multi-agent RAG system for clinical/research PDFs featuring hybrid search (FAISS + BM25), LLaMA 3 generation via Groq, knowledge graph extraction, gap analysis, and a glassmorphism Streamlit dashboard.

---

## Project Structure

```
d:\Major_project\
├── .env                          # API keys (GROQ_API_KEY)
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
├── requirements.txt              # All Python dependencies
├── app.py                        # Streamlit entry point (Phase 5)
│
├── agents/                       # All agent modules
│   ├── __init__.py
│   │
│   ├── ingestion/                # Phase 1 — Knowledge Infrastructure
│   │   ├── __init__.py
│   │   ├── parsing_agent.py      # PDF text extraction
│   │   ├── chunking_agent.py     # 512/64 token splitting
│   │   └── indexing_agent.py     # FAISS + BM25 index builder
│   │
│   ├── retrieval/                # Phase 2 — Intent & Retrieval Engine
│   │   ├── __init__.py
│   │   ├── query_classifier.py   # Intent detection (answer/compare/gap)
│   │   └── retrieval_agent.py    # Hybrid search + RRF fusion
│   │
│   ├── analysis/                 # Phase 3 — Multi-Agent Analysis
│   │   ├── __init__.py
│   │   ├── extraction_agent.py   # Dataset/metric/hardware extraction
│   │   ├── llm_generator.py      # LLaMA 3 via Groq synthesis
│   │   └── gap_detection.py      # K-Means clustering for research gaps
│   │
│   └── synthesis/                # Phase 4 — Structured Output
│       ├── __init__.py
│       ├── knowledge_graph.py    # NetworkX triple extraction & graph
│       ├── validation_agent.py   # Faithfulness & self-reflection scoring
│       └── visualization.py      # Comparison tables & citation charts
│
├── ui/                           # Phase 5 — Streamlit UI components
│   ├── __init__.py
│   ├── styles.py                 # Glassmorphism CSS injection
│   ├── sidebar.py                # File upload + agent status sidebar
│   ├── chat_view.py              # Chat answer panel
│   ├── graph_view.py             # Knowledge graph panel
│   └── analytics_view.py         # Analytics dashboard panel
│
├── core/                         # Shared utilities
│   ├── __init__.py
│   ├── config.py                 # Central configuration & env loading
│   ├── models.py                 # Pydantic data models
│   └── orchestrator.py           # Agent orchestration pipeline
│
└── data/                         # Runtime data (git-ignored)
    ├── uploads/                  # Raw uploaded PDFs
    ├── parsed/                   # Extracted text JSONs
    ├── chunks/                   # Chunked text with metadata
    ├── indexes/                  # FAISS index + BM25 pickle
    └── graphs/                   # Serialized knowledge graphs
```

---

## Tech Stack

| Component | Technology | Version / Notes |
|---|---|---|
| **Language** | Python | 3.10+ |
| **UI Framework** | Streamlit | Latest (with custom CSS) |
| **PDF Extraction** | PyMuPDF (`fitz`) | Handles digital + scanned PDFs |
| **Embeddings** | `sentence-transformers` | Model: `all-MiniLM-L6-v2` (384-dim) |
| **Semantic Search** | FAISS (`faiss-cpu`) | `IndexFlatIP` (inner product on L2-normed vectors) |
| **Keyword Search** | `rank-bm25` | BM25Okapi |
| **LLM Generation** | Groq API | Model: `llama-3.1-8b-instant` |
| **Knowledge Graph** | NetworkX | Directed graph with labeled edges |
| **Graph Viz (UI)** | `streamlit-agraph` | Interactive node-link rendering |
| **Clustering** | scikit-learn | KMeans + Silhouette scoring |
| **Charts** | Plotly | Interactive comparison / citation charts |
| **Data Models** | Pydantic | Typed config & data transfer objects |

---

## User Review Required

> [!IMPORTANT]
> **Groq API Key**: You will need a free API key from [console.groq.com](https://console.groq.com/). This will be stored in a `.env` file and **never** committed to version control. Please confirm you have or can obtain this key.

> [!IMPORTANT]
> **Embedding Model Choice**: The plan uses `all-MiniLM-L6-v2` (general-purpose, fast, 384-dim). For clinical/biomedical text, a domain-specific model like `pritamdeka/S-PubMedBert-MS-MARCO` would yield better retrieval quality but is larger and slower. Which do you prefer?

> [!WARNING]
> **LLaMA Model Selection**: Groq offers several LLaMA variants. The plan uses `llama-3.1-8b-instant` for maximum speed. If you need higher quality for complex medical reasoning, `llama-3.3-70b-versatile` is available but has lower rate limits. Please confirm your preference.

---

## Open Questions

> [!IMPORTANT]
> **PDF Corpus**: Do you already have a set of clinical/research PDFs ready, or should the system be designed purely for on-demand upload via the UI?

> [!NOTE]
> **Voice Input**: Your Phase 2 mentions "voice/text" input. Should we integrate speech-to-text (e.g., `whisper` via Groq's audio API), or is text-only sufficient for the initial build?

> [!NOTE]
> **Deployment Target**: Is this intended to run locally only, or do you plan to deploy it (e.g., Streamlit Cloud, Docker)? This affects how we handle FAISS index persistence and file storage.

---

## Proposed Changes

### Phase 1: Knowledge Infrastructure (Ingestion Agent)

**Objective**: Turn uploaded PDFs into a searchable hybrid database with dual FAISS + BM25 indexes.

---

#### [NEW] [config.py](file:///d:/Major_project/core/config.py)
- Central configuration using Pydantic `BaseSettings`
- Loads `.env` for `GROQ_API_KEY`
- Defines all paths (`DATA_DIR`, `UPLOAD_DIR`, `INDEX_DIR`, etc.)
- Configurable constants: `CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`, `EMBEDDING_MODEL`, `LLM_MODEL`, `RRF_K=60`, `TOP_K=10`

#### [NEW] [models.py](file:///d:/Major_project/core/models.py)
- `Document`: `id`, `filename`, `full_text`, `metadata` (page count, title, authors)
- `Chunk`: `id`, `doc_id`, `text`, `page_num`, `start_char`, `end_char`, `embedding` (optional)
- `SearchResult`: `chunk`, `score`, `source` (semantic/keyword/hybrid)
- `AgentStatus`: `name`, `state` (idle/running/done/error), `message`, `progress`

#### [NEW] [parsing_agent.py](file:///d:/Major_project/agents/ingestion/parsing_agent.py)
- Uses **PyMuPDF** (`fitz`) for text extraction
- Page-by-page extraction preserving structure
- Header/footer/page-number stripping via regex
- Table detection heuristics (column-aligned text patterns)
- Returns `Document` objects with metadata (title, authors parsed from first page)
- Error handling for corrupted/password-protected PDFs

#### [NEW] [chunking_agent.py](file:///d:/Major_project/agents/ingestion/chunking_agent.py)
- **Token-aware chunking** using the `sentence-transformers` tokenizer
- Primary split: **512 tokens** per chunk
- Overlap: **64 tokens** sliding window
- Sentence-boundary-aware splitting (never breaks mid-sentence)
  - Uses `nltk.sent_tokenize` for sentence detection
  - Accumulates sentences until 512 token budget, then slides back by 64 tokens
- Each `Chunk` carries: `doc_id`, `page_num`, positional metadata

#### [NEW] [indexing_agent.py](file:///d:/Major_project/agents/ingestion/indexing_agent.py)
- **Semantic Index (FAISS)**:
  - Encodes all chunks with `sentence-transformers` model
  - L2-normalizes embeddings → uses `IndexFlatIP` (cosine similarity via inner product)
  - Saves index to `data/indexes/faiss.index`
  - Saves chunk-ID-to-index mapping as JSON
- **Keyword Index (BM25)**:
  - Tokenizes chunks (lowercase, removes stopwords)
  - Builds `BM25Okapi` from `rank-bm25`
  - Pickles the BM25 object + token corpus to `data/indexes/bm25.pkl`
- **Incremental support**: Can add new documents without full rebuild (appends to both indexes)

---

### Phase 2: Intent & Retrieval Engine (Reasoning Agent)

**Objective**: Classify user intent and execute high-precision hybrid search with RRF fusion.

---

#### [NEW] [query_classifier.py](file:///d:/Major_project/agents/retrieval/query_classifier.py)
- **Three intent categories**:
  1. `ANSWER` — Direct factual question ("What is the accuracy of model X?")
  2. `COMPARE` — Comparison request ("Compare method A vs method B")
  3. `GAP_ANALYSIS` — Research gap exploration ("What areas are under-explored?")
- **Implementation**: Lightweight LLM call to Groq with a structured prompt
  - Input: user query string
  - Output: `{"intent": "ANSWER|COMPARE|GAP_ANALYSIS", "entities": [...], "refined_query": "..."}`
- Falls back to `ANSWER` if classification confidence is low

#### [NEW] [retrieval_agent.py](file:///d:/Major_project/agents/retrieval/retrieval_agent.py)
- **Semantic Search**: Encodes query → searches FAISS index → returns top-K ranked results
- **Keyword Search**: Tokenizes query → scores all chunks via BM25 → returns top-K ranked results
- **Reciprocal Rank Fusion (RRF)**:
  ```
  Score(doc) = Σ 1/(k + rank_i)   where k=60 (tunable)
  ```
  - Merges both ranked lists into a single fused ranking
  - Returns top-N chunks (default N=10) with RRF scores
- **Intent-aware weighting**:
  - `ANSWER`: 50/50 semantic/keyword
  - `COMPARE`: 60/40 semantic/keyword (entities matter more)
  - `GAP_ANALYSIS`: 70/30 semantic/keyword (conceptual similarity matters)

---

### Phase 3: Multi-Agent Analysis Layer (Analytical Agent)

**Objective**: Deep research analysis beyond simple QA — extraction, generation, and gap detection.

---

#### [NEW] [extraction_agent.py](file:///d:/Major_project/agents/analysis/extraction_agent.py)
- Regex + LLM pattern matching to extract structured data from chunks:
  - **Datasets**: Names, sizes, domains (e.g., "MIMIC-III, 40K patients")
  - **Metrics**: Accuracy, F1, AUC, p-values with their values
  - **Hardware**: GPU models, training time, memory usage
  - **Methods**: Algorithm names, architectures, frameworks
- Output: Structured JSON per chunk with extracted entities
- Uses Groq LLaMA for ambiguous cases (regex first, LLM fallback)

#### [NEW] [llm_generator.py](file:///d:/Major_project/agents/analysis/llm_generator.py)
- **Groq API client** wrapper with retry logic and rate limiting
- **Prompt templates** per intent type:
  - `ANSWER`: "Given the following context, answer the question. Cite sources."
  - `COMPARE`: "Compare the following approaches across these dimensions: [metrics, datasets, results]."
  - `GAP_ANALYSIS`: "Identify research gaps and unexplored directions based on this corpus."
- **Reasoning Trace**: Each response includes a structured trace:
  ```json
  {
    "reasoning_steps": ["Step 1: Identified 3 relevant papers...", "Step 2: ..."],
    "sources_used": ["chunk_id_1", "chunk_id_2"],
    "confidence": 0.85,
    "answer": "..."
  }
  ```
- Streaming support for real-time UI updates

#### [NEW] [gap_detection.py](file:///d:/Major_project/agents/analysis/gap_detection.py)
- **Embedding-based clustering**:
  1. Takes all chunk embeddings from FAISS index
  2. Applies **K-Means** clustering (auto-selects k via Silhouette score, range 3–15)
  3. Identifies cluster centroids and computes intra-cluster density
- **Gap identification**:
  - Sparse clusters (low density) → "under-explored areas"
  - Gaps between clusters (high inter-cluster distance) → "research frontiers"
  - Generates natural language descriptions of each cluster's theme using LLM
- Output: `GapReport` with cluster labels, themes, density scores, and gap descriptions

---

### Phase 4: Structured Output & Future Enhancements (Synthesis Agent)

**Objective**: Knowledge graphs, validation, and rich visualizations.

---

#### [NEW] [knowledge_graph.py](file:///d:/Major_project/agents/synthesis/knowledge_graph.py)
- **Triple extraction** via LLM (Groq):
  - Input: Retrieved chunks
  - Prompt: "Extract (Subject, Predicate, Object) triples. Subjects/Objects are: Paper, Method, Dataset, Metric, Result."
  - Output: List of `(entity1, relationship, entity2)` tuples
- **Graph construction** with NetworkX:
  - `DiGraph` with typed nodes (Paper, Method, Dataset, Metric, Result)
  - Edges carry relationship labels and source chunk IDs
  - Entity normalization (fuzzy matching to merge duplicates like "Attention" / "Attention Mechanism")
- **Persistence**: Serialize to `data/graphs/` as GraphML for reload
- **Export**: Converts to `streamlit-agraph` compatible format (nodes/edges lists)

#### [NEW] [validation_agent.py](file:///d:/Major_project/agents/synthesis/validation_agent.py)
- **Faithfulness Check**:
  - Decomposes the generated answer into atomic claims
  - For each claim, checks if it can be inferred from the source chunks
  - Score: `faithful_claims / total_claims` (0.0 to 1.0)
- **Self-Reflection**:
  - Asks the LLM: "Review your answer. Are there any unsupported claims or logical errors?"
  - Produces a `reflection_notes` list and an adjusted confidence score
- **Output**: `ValidationReport` with `faithfulness_score`, `reflection_notes`, `final_confidence`
- All validation uses a separate LLM call (judge model) to avoid self-bias

#### [NEW] [visualization.py](file:///d:/Major_project/agents/synthesis/visualization.py)
- **Comparison Tables**: Plotly `Table` traces from extracted metrics
  - Rows: Papers/Methods, Columns: Metrics (Accuracy, F1, etc.)
  - Color-coded cells (green=best, red=worst per column)
- **Citation Trend Charts**: Plotly bar/line charts
  - X-axis: Publication year, Y-axis: Count or cumulative
  - Grouped by method/topic cluster
- **Cluster Visualization**: 2D scatter plot (UMAP-reduced embeddings)
  - Points colored by K-Means cluster
  - Hover shows chunk text snippet
- All charts return Plotly `Figure` objects for Streamlit rendering

---

### Phase 5: Aesthetic UI Integration (Streamlit Dashboard)

**Objective**: Build an interactive glassmorphism dashboard with multi-view panels and real-time agent traces.

---

#### [NEW] [.streamlit/config.toml](file:///d:/Major_project/.streamlit/config.toml)
- Dark theme configuration:
  ```toml
  [theme]
  primaryColor = "#7C3AED"
  backgroundColor = "#0F0F1A"
  secondaryBackgroundColor = "#1A1A2E"
  textColor = "#E2E8F0"
  font = "sans serif"
  ```

#### [NEW] [styles.py](file:///d:/Major_project/ui/styles.py)
- Injects glassmorphism CSS via `st.markdown`:
  - `.glass-panel`: Semi-transparent cards with `backdrop-filter: blur(12px)`, rounded corners, subtle borders
  - `.glass-sidebar`: Sidebar styling with gradient background
  - Animated gradient background (slow color shift via CSS `@keyframes`)
  - Custom scrollbar, button hover effects, status indicator animations
  - Google Fonts import (Inter)
- Color palette: Deep navy (`#0F0F1A`), Purple accent (`#7C3AED`), Cyan highlight (`#06B6D4`), Emerald success (`#10B981`)

#### [NEW] [sidebar.py](file:///d:/Major_project/ui/sidebar.py)
- **File Upload Widget**: `st.file_uploader` for multiple PDFs (drag-and-drop)
- **Ingestion Trigger**: "Process Documents" button → triggers Phase 1 pipeline
- **Agent Status Monitor**: Live status cards for each agent
  - Animated pulse indicator (🟢 idle / 🔵 running / ✅ done / 🔴 error)
  - Progress bars for long-running operations
- **Document List**: Shows ingested documents with page counts and chunk counts
- **Settings Panel**: Expandable section for tuning `CHUNK_SIZE`, `TOP_K`, `RRF_K`

#### [NEW] [chat_view.py](file:///d:/Major_project/ui/chat_view.py)
- **Chat Interface**: `st.chat_message` based conversation
- **Query Input**: Text input with submit button
- **Response Display**:
  - Streamed answer text (token-by-token from Groq)
  - Expandable "Reasoning Trace" section showing step-by-step agent logic
  - Source citations with expandable chunk previews
  - Validation badge: Faithfulness score + confidence indicator (color-coded)
- **History**: Session-state based conversation history

#### [NEW] [graph_view.py](file:///d:/Major_project/ui/graph_view.py)
- **Interactive Knowledge Graph**: `streamlit-agraph` component
  - Color-coded nodes by type (Paper=purple, Method=cyan, Dataset=green, Metric=orange, Result=pink)
  - Edge labels showing relationships
  - Physics-based layout (force-directed)
  - Click-to-focus on a node shows connected entities
- **Graph Stats Panel**: Node/edge counts, most connected entities, relationship type distribution
- **Filter Controls**: Checkboxes to show/hide node types

#### [NEW] [analytics_view.py](file:///d:/Major_project/ui/analytics_view.py)
- **Comparison Dashboard**: Rendered Plotly tables and charts from `visualization.py`
- **Gap Analysis Panel**: Cluster visualization (UMAP scatter) + gap descriptions
- **Metrics Overview**: Summary statistics (total papers, unique methods, datasets covered)
- **Export**: Download buttons for CSV tables and PNG charts

#### [NEW] [app.py](file:///d:/Major_project/app.py)
- Main Streamlit entry point
- Page config: Wide layout, custom page title/icon
- Tab-based navigation: "💬 Chat" | "🕸️ Knowledge Graph" | "📊 Analytics"
- Imports and renders sidebar + selected view
- Session state management for all agent states and conversation history

#### [NEW] [orchestrator.py](file:///d:/Major_project/core/orchestrator.py)
- **Pipeline controller** that chains all agents:
  1. `ingest(files)` → parsing → chunking → indexing
  2. `query(text)` → classify → retrieve → extract → generate → validate
  3. `analyze()` → gap detection → knowledge graph → visualization
- Status callbacks for real-time UI updates
- Error propagation with graceful degradation (if one agent fails, others continue where possible)

---

### Supporting Files

#### [NEW] [requirements.txt](file:///d:/Major_project/requirements.txt)
```
streamlit>=1.40.0
streamlit-agraph>=0.0.45
PyMuPDF>=1.24.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
rank-bm25>=0.2.2
groq>=0.11.0
networkx>=3.3
plotly>=5.24.0
scikit-learn>=1.5.0
umap-learn>=0.5.6
nltk>=3.9.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
python-dotenv>=1.0.1
numpy>=1.26.0
pandas>=2.2.0
```

#### [NEW] [.env](file:///d:/Major_project/.env)
```
GROQ_API_KEY=your_api_key_here
```

#### [NEW] [.gitignore](file:///d:/Major_project/.gitignore)
```
.env
data/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

---

## Verification Plan

### Automated Tests

1. **Phase 1 — Ingestion Pipeline**:
   - Unit test: Parse a sample PDF → verify text extraction is non-empty and metadata is populated
   - Unit test: Chunk a known text → verify chunk count, token sizes (≤512), and overlap (64 tokens)
   - Unit test: Build FAISS + BM25 indexes → verify index sizes match chunk count
   - Command: `python -m pytest tests/ -v`

2. **Phase 2 — Retrieval**:
   - Integration test: Ingest a sample PDF → query → verify returned chunks contain relevant text
   - Unit test: RRF fusion with mock rank lists → verify correct score computation
   - Unit test: Query classifier returns valid intent enum for test queries

3. **Phase 3 — Analysis**:
   - Unit test: Extraction agent on known text → verify regex captures datasets/metrics
   - Integration test: Full query pipeline → verify LLM response contains reasoning trace fields
   - Unit test: K-Means on synthetic embeddings → verify cluster count and gap identification

4. **Phase 4 — Synthesis**:
   - Unit test: Triple extraction on sample text → verify valid (S, P, O) tuples
   - Unit test: Knowledge graph construction → verify node/edge counts
   - Unit test: Faithfulness scoring on a known faithful answer → score ≈ 1.0

5. **Phase 5 — UI**:
   - Manual browser test: Upload PDF → verify sidebar updates → ask question → verify response renders
   - Manual browser test: Switch between Chat / Graph / Analytics tabs → verify each renders correctly
   - Visual check: Glassmorphism styling applied correctly (blur, transparency, animations)

### Manual Verification

- **End-to-end demo**: Upload 2-3 clinical/research PDFs → ask varied questions → verify quality of answers, citations, and knowledge graph
- **Performance**: Verify ingestion of a 50-page PDF completes in < 60 seconds
- **Groq API**: Verify streaming responses render in real-time in the chat view

---

## Execution Order

| Step | Phase | Estimated Effort | Dependencies |
|------|-------|-----------------|--------------|
| 1 | Shared core (`config.py`, `models.py`) | Small | None |
| 2 | Phase 1: Ingestion agents | Medium | Step 1 |
| 3 | Phase 2: Retrieval agents | Medium | Step 2 |
| 4 | Phase 3: Analysis agents | Large | Steps 2, 3 |
| 5 | Phase 4: Synthesis agents | Large | Steps 3, 4 |
| 6 | Phase 5: UI + Orchestrator | Large | Steps 1–5 |
| 7 | Integration testing & polish | Medium | Step 6 |
