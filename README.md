# Research Gap Finder 

An AI-powered system to analyze research papers, extract key insights, identify research gaps, and build knowledge graphs from academic literature.

## Overview

This project processes PDF research papers to:
- **Extract & Chunk**: Parse PDFs and intelligently segment text into meaningful chunks
- **Embed**: Generate semantic embeddings using transformer models
- **Summarize**: Create concise summaries using state-of-the-art NLP models
- **Analyze Gaps**: Identify missing areas and research opportunities across papers
- **Map Knowledge**: Build a knowledge graph showing relationships between papers
- **Visualize**: Interactive Streamlit UI for exploring results

## Features

 **Local-First**: No external APIs required (with optional HuggingFace Inference API support)  
 **Efficient**: Uses lightweight, optimized models (all-MiniLM-L6-v2 for embeddings)  
 **Scalable**: Processes multiple PDFs in batch with persistent vector storage  
 **Connected**: Builds citation networks and similarity graphs  
 **AI-Powered**: Leverages transformers for summarization and gap analysis  

## Project Structure

```
research-gap-ai/
├── main.py                    # CLI pipeline entry point
├── app.py                     # Streamlit web UI
├── config.py                  # Configuration & environment variables
├── requirements.txt           # Python dependencies
│
├── data_ingestion.py         # PDF extraction & text parsing
├── chunking.py               # Text segmentation strategies
├── embeddings.py             # Embedding model management
├── index_storage.py          # FAISS vector store operations
├── ai_core.py                # Summarization & LLM integration
├── gap_analyzer.py           # Research gap identification
├── knowledge_graph.py        # Citation graph & paper similarity
├── retrieval.py              # Vector similarity search
├── utils.py                  # Helper utilities
│
├── data/
│   ├── raw_papers/           # Place your PDF files here
│   └── vector_store/         # Persistent embeddings index
├── outputs/
│   ├── summaries/            # Generated paper summaries
│   └── gap_reports/          # Gap analysis reports
│
└── venv/                      # Python virtual environment
```

## Installation

### 1. Clone & Setup
```bash
cd research-gap-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.template .env
```

Edit `.env` and (optionally) add your HuggingFace API token:
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
HF_API_TOKEN=hf_your_token_here
PAPERS_FOLDER=data/raw_papers
VECTOR_STORE_PATH=data/vector_store
```

**Note**: HF_API_TOKEN is optional. Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope only).

## Quick Start

### Run CLI Pipeline
```bash
python main.py
```

**Steps:**
1. Place PDF files in `data/raw_papers/`
2. Run the command above
3. Check outputs in `outputs/summaries/` and `outputs/gap_reports/`

### Launch Web UI
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Workflow

```
PDFs
  ↓
[Extract Text] → [Chunk Text] → [Create Embeddings]
  ↓
[Vector Store] → [Semantic Search] → [Similarity Analysis]
  ↓
[Summarization] ← [Gap Analysis] ← [Knowledge Graph]
  ↓
Outputs (Summaries, Gap Reports, Citation Networks)
```

## Dependencies

| Category | Libraries |
|----------|-----------|
| **PDF Processing** | pypdf |
| **Text Chunking** | langchain-text-splitters |
| **Embeddings** | sentence-transformers, faiss-cpu, numpy |
| **Models** | transformers, torch, accelerate |
| **Graph Analysis** | networkx, scikit-learn |
| **API** | requests |
| **Web UI** | streamlit |
| **Config** | python-dotenv |
| **Data** | pandas |

## Key Models

- **Embeddings**: `all-MiniLM-L6-v2` (384-dim, ~22MB)
- **Summarization**: `facebook/bart-large-cnn` (HuggingFace)
- **Gap Analysis**: `mistralai/Mistral-7B-Instruct-v0.1` (HuggingFace)

All models are free and don't require approval or paid APIs.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `HF_API_TOKEN` | HuggingFace API token (optional) | `""` |
| `PAPERS_FOLDER` | Path to input PDFs | `data/raw_papers` |
| `VECTOR_STORE_PATH` | Path to save embeddings | `data/vector_store` |

### Fallback Behavior

- **With HF_API_TOKEN**: Uses HuggingFace Inference API (faster, cloud-based)
- **Without token**: Falls back to local pipeline (slower but works offline)

## Output Files

After running `python main.py`:

```
outputs/
├── summaries/
│   ├── paper1.txt
│   ├── paper2.txt
│   └── ...
└── gap_reports/
    └── gap_report.txt
```

The gap report includes:
- Missing research areas
- Overlapping topics
- Potential research directions
- Recommendations based on analyzed papers

## Example Usage

```bash
# 1. Add PDFs to data/raw_papers/
cp ~/Downloads/arxiv_papers/*.pdf data/raw_papers/

# 2. Set HF token (optional, for faster inference)
export HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# 3. Run pipeline
python main.py

# 4. View results
ls outputs/summaries/
cat outputs/gap_reports/gap_report.txt

# 5. Launch UI for interactive exploration
streamlit run app.py
```

## Troubleshooting

### No PDFs found
- Ensure PDFs are in `data/raw_papers/`
- Check file permissions

### "No text found" warning
- PDF might be scanned/image-based (not text-extractable)
- Use OCR preprocessing if needed

### Out of Memory
- Process fewer PDFs at once
- Reduce chunk size in `chunking.py`
- Use smaller embedding model in `.env`

### Slow inference
- Add HuggingFace API token for cloud acceleration
- Or reduce number of papers processed

## Performance Notes

- **Small dataset** (1-10 papers): ~2-5 min (local)
- **Medium dataset** (10-50 papers): ~10-20 min (local)
- **With HF API**: 2-3x faster
- **Vector search**: O(1) similarity queries after indexing

## Future Enhancements

- [ ] Multi-language support
- [ ] Citation extraction and cross-linking
- [ ] Interactive gap recommendation engine
- [ ] Export to BibTeX/JSON
- [ ] Advanced NLP for entity recognition
- [ ] Collaborative filtering for paper recommendations

## License

This project is provided as-is for research and educational purposes.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

---

**Happy researching!** 
