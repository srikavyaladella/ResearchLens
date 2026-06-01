"""
app.py  –  Streamlit UI for the Research Gap Finder
----------------------------------------------------
Run:  streamlit run app.py
"""

import os
import numpy as np
import streamlit as st

from data_ingestion  import extract_text_from_pdf
from chunking        import chunk_text
from embeddings      import EmbeddingModel
from index_storage   import VectorStore
from ai_core         import generate_summary
from gap_analyzer    import analyze_research_gaps
from retrieval       import retrieve_relevant_chunks
from knowledge_graph import CitationGraph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Gap Finder",
    page_icon="📚",
    layout="wide",
)

st.title("AI Research Paper Summariser & Gap Finder")
st.write("Upload multiple research papers (PDF) to get AI summaries, gap analysis, and Q&A.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. Upload **2 or more** PDF papers
    2. Click **Process Papers**
    3. Read the **summaries** per paper
    4. Click **Run Gap Analysis**
    5. Use **Q&A** to query paper contents
    6. Check the **Similarity Graph**
    """)
    st.divider()
    st.subheader(" Models Used")
    st.caption("• Embeddings: all-MiniLM-L6-v2 (local, free)")
    st.caption("• Summarisation: facebook/bart-large-cnn")
    st.caption("• Gap analysis: flan-t5-large or Mistral-7B")
    st.divider()
    # st.subheader(" Optional: HF API Token")
    # st.caption("Add HF_API_TOKEN to .env for faster, better AI results.")
    # st.caption("Get free token: huggingface.co/settings/tokens")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "vector_store":    None,
    "embedding_model": None,
    "summaries":       [],
    "paper_names":     [],
    "mean_embeddings": [],
    "processed":       False,
    "gap_report":      "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button(" Process Papers", type="primary"):

    # Reset state
    st.session_state.summaries       = []
    st.session_state.paper_names     = []
    st.session_state.mean_embeddings = []
    st.session_state.processed       = False
    st.session_state.gap_report      = ""

    embedding_model = EmbeddingModel()
    vector_store    = VectorStore(dimension=384)

    progress_bar = st.progress(0)
    total        = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f" {uploaded_file.name}")
        temp_path = f"temp_{uploaded_file.name}"

        try:
          # Save buffer to temp file
         with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ── STEP 1: Extract ──────────────────────────────────────────────
        text = extract_text_from_pdf(temp_path)
        if not text.strip():
           st.warning(
              f"No text found in **{uploaded_file.name}** "
              "(possibly a scanned/image PDF). Skipping."
        )
        continue
    st.success(f"Text extracted ({len(text):,} characters)")

    # ── STEP 2: Chunk ────────────────────────────────────────────────
    chunks = chunk_text(text)
    st.write(f"Chunks created: **{len(chunks)}**")

    # ── STEP 3: Embed ────────────────────────────────────────────────
    with st.spinner("Creating embeddings…"):
        embeddings = embedding_model.create_embeddings(chunks)
    st.success("Embeddings created")

    # ── STEP 4: Store ────────────────────────────────────────────────
    vector_store.add_embeddings(embeddings, chunks)
    st.session_state.mean_embeddings.append(np.mean(embeddings, axis=0))

    # ── STEP 5: Summarise ────────────────────────────────────────────
    with st.spinner("Generating summary… (may take 30–60s on first run)"):
        summary = generate_summary(text[:6000])

    if not summary or not summary.strip():
        summary = text[:500] + "…"   # safe fallback

    st.session_state.summaries.append(summary)
    st.session_state.paper_names.append(uploaded_file.name)

    with st.expander("Summary", expanded=True):
        st.write(summary)

except Exception as e:
    st.error(f"Error processing **{uploaded_file.name}**: {e}")
    import traceback
    with st.expander("🔍 Error details"):
        st.code(traceback.format_exc())

finally:
    # ✅ ALWAYS clean up temp file, even if error occurs
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as cleanup_error:
        st.warning(f"Could not delete temp file {temp_path}: {cleanup_error}")

progress_bar.progress((i + 1) / total)

    # Persist to session
    st.session_state.vector_store    = vector_store
    st.session_state.embedding_model = embedding_model
    st.session_state.processed       = True

    n = len(st.session_state.summaries)
    st.success(f"Processing complete! {n} paper(s) ready.")

# ── Post-processing sections (shown after papers are loaded) ──────────────────
if st.session_state.processed and st.session_state.summaries:

    # ── Gap Analysis ──────────────────────────────────────────────────────────
    st.divider()
    st.header("Research Gap Analysis")

    if len(st.session_state.summaries) < 2:
        st.warning("Upload at least **2 papers** to run gap analysis.")
    else:
        if st.button("Run Gap Analysis"):
            with st.spinner("Analysing research gaps… (may take up to 2 min on CPU)"):
                try:
                    gap_report = analyze_research_gaps(st.session_state.summaries)
                    st.session_state.gap_report = gap_report
                except Exception as e:
                    st.error(f"Gap analysis failed: {e}")
                    st.session_state.gap_report = ""

        if st.session_state.gap_report:
            st.markdown(st.session_state.gap_report)

            # Download button
            st.download_button(
                label=" Download Gap Report",
                data=st.session_state.gap_report,
                file_name="gap_report.md",
                mime="text/markdown",
            )

    # ── Q&A Section ───────────────────────────────────────────────────────────
    st.divider()
    st.header(" Ask Questions About Your Papers")
    st.caption("Searches the full text of all uploaded papers using semantic similarity.")

    query = st.text_input("Ask anything about the uploaded papers…", key="qa_input")

    if query and st.session_state.vector_store:
        with st.spinner("Searching…"):
            try:
                results = retrieve_relevant_chunks(
                    st.session_state.vector_store,
                    query,
                    top_k=3,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                results = []

        if results:
            st.subheader(" Most Relevant Excerpts")
            for j, chunk in enumerate(results, 1):
                with st.expander(f"Excerpt {j}", expanded=(j == 1)):
                    st.write(chunk)
        else:
            st.warning("No relevant chunks found. Try rephrasing your question.")

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    st.divider()
    st.header(" Paper Similarity Graph")
    st.caption("Papers are linked when their content similarity (cosine) ≥ 0.70.")

    names    = st.session_state.paper_names
    mean_emb = st.session_state.mean_embeddings

    if len(names) >= 2 and len(mean_emb) >= 2:
        try:
            citation_graph = CitationGraph()
            citation_graph.find_similar_papers(names, mean_emb, threshold=0.70)
            graph_data = citation_graph.to_dict()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Nodes (Papers)")
                for node in graph_data["nodes"]:
                    st.write(f"• {node}")

            with col2:
                st.subheader("Connections")
                if graph_data["edges"]:
                    for edge in graph_data["edges"]:
                        st.write(
                            f"• **{edge['source']}**  ↔  **{edge['target']}**  "
                            f"_(similarity: {edge['weight']:.2f})_"
                        )
                    top = citation_graph.most_connected_paper()
                    if top:
                        st.success(f" Most connected paper: **{top}**")
                else:
                    st.info(
                        "No strong connections found (threshold 0.70). "
                        "Papers may cover different topics."
                    )

        except Exception as e:
            st.error(f"Knowledge graph error: {e}")
    else:
        st.info("Upload 2 or more papers to see similarity connections.")

elif not st.session_state.processed:
    st.info("Upload PDFs and click **Process Papers** to get started.")
