"""
main.py  –  CLI pipeline for the Research Gap Finder
------------------------------------------------------
Run:  python main.py
Place your PDFs in  data/raw_papers/  before running.
"""

import os
import numpy as np

from config          import PAPERS_FOLDER, VECTOR_STORE_PATH
from data_ingestion  import extract_text_from_pdf
from chunking        import chunk_text
from embeddings      import EmbeddingModel
from index_storage   import VectorStore
from ai_core         import generate_summary
from gap_analyzer    import analyze_research_gaps
from knowledge_graph import CitationGraph
from utils           import save_output, ensure_directories


def main():
    ensure_directories()

    # ── collect PDF files ────────────────────────────────────────────────────
    if not os.path.exists(PAPERS_FOLDER):
        os.makedirs(PAPERS_FOLDER, exist_ok=True)

    pdf_files = [f for f in os.listdir(PAPERS_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in '{PAPERS_FOLDER}'. Add papers and re-run.")
        return

    print(f"Found {len(pdf_files)} paper(s): {pdf_files}\n")

    # ── shared objects – created ONCE before the loop ────────────────────────
    embedding_model       = EmbeddingModel()
    vector_store          = VectorStore(dimension=384)
    citation_graph        = CitationGraph()
    all_summaries         = []
    paper_mean_embeddings = []
    processed_names       = []

    # ── per-paper pipeline ───────────────────────────────────────────────────
    for file_name in pdf_files:
        pdf_path = os.path.join(PAPERS_FOLDER, file_name)
        print(f"\n{'='*60}")
        print(f"Processing: {file_name}")

        # STEP 1 – Extract text
        try:
            text = extract_text_from_pdf(pdf_path)
            print("  ✅ Text extracted")
        except Exception as e:
            print(f"  ❌ Extraction failed: {e} — skipping.")
            continue

        if not text.strip():
            print("  ⚠️  No text found (scanned PDF?) — skipping.")
            continue

        # STEP 2 – Chunk
        chunks = chunk_text(text)
        print(f"  ✅ {len(chunks)} chunks created")

        # STEP 3 – Embed
        embeddings = embedding_model.create_embeddings(chunks)
        print("  ✅ Embeddings created")

        # STEP 4 – Store in shared vector DB
        vector_store.add_embeddings(embeddings, chunks)
        paper_mean_embeddings.append(np.mean(embeddings, axis=0))
        print("  ✅ Added to vector store")

        # STEP 5 – Summarise
        print("  ⏳ Generating summary…")
        summary = generate_summary(text[:6000])
        all_summaries.append(summary)
        processed_names.append(file_name)
        citation_graph.add_paper(file_name)

        output_path = f"outputs/summaries/{file_name.replace('.pdf', '.txt')}"
        save_output(summary, output_path)
        print(f"  ✅ Summary saved → {output_path}")

    if not all_summaries:
        print("\nNo papers were processed successfully. Exiting.")
        return

    # ── save vector store once (after all papers) ────────────────────────────
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save(VECTOR_STORE_PATH)
    print(f"\n✅ Vector store saved → {VECTOR_STORE_PATH}")

    # ── knowledge graph ───────────────────────────────────────────────────────
    if len(processed_names) >= 2:
        citation_graph.find_similar_papers(
            processed_names,
            paper_mean_embeddings,
            threshold=0.70,
        )
        connections = citation_graph.show_connections()
        print(f"\n✅ Knowledge graph: {len(connections)} connection(s) found")
        for c in connections:
            print(f"   {c[0]}  ↔  {c[1]}")

    # ── gap analysis ──────────────────────────────────────────────────────────
    if len(all_summaries) >= 2:
        print("\n🔍 Analysing research gaps…")
        gap_report = analyze_research_gaps(all_summaries)
        save_output(gap_report, "outputs/gap_reports/gap_report.txt")
        print("✅ Gap report saved → outputs/gap_reports/gap_report.txt")
    else:
        print("\n⚠️  Only 1 paper processed. Add at least 2 for gap analysis.")

    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()