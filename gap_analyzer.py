"""
gap_analyzer.py  –  Real AI-powered research gap analysis
----------------------------------------------------------
Strategy (free, no paid API needed):

   Priority 1 → HuggingFace FREE Inference API (Mistral-7B-Instruct)
                Requires HF_API_TOKEN in .env

   Priority 2 → Local flan-t5-large via "text2text-generation" task
                (NOT "summarization" – removed in transformers 5.x)

   Priority 3 → Rule-based structured analysis as final safety net
                (always returns a useful report even if AI fails)
"""

import requests
from config import HF_API_TOKEN, HF_GAP_MODEL


# ────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────

def _build_prompt(summaries: list) -> str:
    numbered = "\n\n".join(
        f"Paper {i+1}:\n{s}" for i, s in enumerate(summaries)
    )
    return (
        "You are a senior research analyst. "
        "Given the following research paper summaries, identify:\n"
        "1. Common themes and overlapping findings\n"
        "2. Specific research gaps and unsolved problems\n"
        "3. Contradictions or disagreements between the papers\n"
        "4. Promising future research directions\n\n"
        f"{numbered}\n\n"
        "Provide a structured, detailed analysis:"
    )


# ────────────────────────────────────────────────────────────────
# HF Inference API path
# ────────────────────────────────────────────────────────────────

def _gap_via_hf_api(prompt: str) -> str:
    """
    Call HuggingFace Inference API for gap analysis.
    
    Parameters
    ----------
    prompt : str
        Formatted prompt for the model
        
    Returns
    -------
    str
        Generated gap analysis text
        
    Raises
    ------
    RuntimeError
        If API returns an error
    ValueError
        If response format is unexpected
    """
    url     = f"https://api-inference.huggingface.co/models/{HF_GAP_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and data:
        item = data[0]
        # Try multiple possible response keys for compatibility
        text = item.get("generated_text") or item.get("text") or str(item).strip()
        if text:
            return text
    
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF API: {data['error']}")
    
    raise ValueError(f"Unexpected HF API response: {data}")


# ────────────────────────────────────────────────────────────────
# Local model path  (flan-t5-large)
# ────────────────────────────────────────────────────────────────

def _gap_locally(summaries: list) -> str:
    """
    Uses 'text2text-generation' — works in transformers 4.x AND 5.x.
    'summarization' task was removed in transformers 5.x.
    """
    try:
        from transformers import pipeline

        pipe = pipeline(
            "text2text-generation",   # ✅ compatible with transformers 4.x & 5.x
            model="google/flan-t5-large",
        )

        combined = " | ".join(s[:300] for s in summaries)
        prompt = (
            f"Identify research gaps from these paper summaries: {combined}. "
            "List: 1) Common themes 2) Gaps 3) Future directions"
        )
        result = pipe(prompt, max_new_tokens=400, truncation=True)
        return result[0]["generated_text"].strip()

    except Exception as e:
        print(f"[gap_analyzer] Local model failed ({e}). Using rule-based fallback.")
        return _rule_based_analysis(summaries)


# ────────────────────────────────────────────────────────────────
# Rule-based fallback (always works, no dependencies)
# ────────────────────────────────────────────────────────────────

def _rule_based_analysis(summaries: list) -> str:
    """
    Structured analysis based on the summaries text.
    Used when both API and local model are unavailable.
    Extracts keywords and patterns from the actual summaries.
    """
    all_text = " ".join(summaries).lower()

    # Common research keywords to scan for
    themes_keywords   = ["deep learning", "neural", "transformer", "bert", "attention",
                         "classification", "detection", "generation", "nlp", "vision",
                         "reinforcement", "graph", "language model", "embedding"]
    gap_keywords      = ["limited", "challenge", "however", "lack", "future", "improve",
                         "not yet", "remain", "open problem", "insufficient"]
    method_keywords   = ["proposed", "method", "approach", "model", "framework",
                         "algorithm", "technique", "architecture"]

    found_themes  = [k for k in themes_keywords  if k in all_text]
    found_gaps    = [k for k in gap_keywords      if k in all_text]
    found_methods = [k for k in method_keywords   if k in all_text]

    themes_text  = ", ".join(found_themes[:5])  if found_themes  else "General AI/ML research"
    methods_text = ", ".join(found_methods[:4]) if found_methods else "Various methodologies"
    gaps_note    = (
        "The papers mention limitations/challenges suggesting open research areas."
        if found_gaps else
        "Further investigation needed to identify explicit gaps."
    )

    return (
        f"## 1. Common Themes\n"
        f"Across the analysed papers, recurring topics include: **{themes_text}**.\n"
        f"Methods mentioned: {methods_text}.\n\n"
        f"## 2. Research Gaps\n"
        f"{gaps_note}\n"
        f"- Cross-domain generalisation is rarely fully addressed.\n"
        f"- Scalability and real-world deployment remain underexplored.\n"
        f"- Reproducibility and benchmark diversity need improvement.\n\n"
        f"## 3. Contradictions\n"
        f"Different papers may use different evaluation metrics or datasets, "
        f"making direct comparison difficult. Methodological disagreements "
        f"may exist but require deeper manual review.\n\n"
        f"## 4. Future Directions\n"
        f"- Larger-scale empirical validation across diverse datasets.\n"
        f"- Energy-efficient and lightweight model variants.\n"
        f"- Interdisciplinary applications beyond current scope.\n"
        f"- Standardised benchmarks for fair comparison.\n"
    )


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────

def analyze_research_gaps(summaries: list) -> str:
    """
    Perform AI gap analysis across multiple paper summaries.
    Returns a structured markdown report. ALWAYS returns something.
    """
    if not summaries:
        return "# Research Gap Analysis\n\nNo summaries provided."

    raw = ""

    # ── Try HF API first ─────────────────────────────────────────────────────
    if HF_API_TOKEN:
        try:
            raw = _gap_via_hf_api(_build_prompt(summaries))
        except Exception as e:
            print(f"[gap_analyzer] HF API failed ({e}). Trying local model…")

    # ── Try local model ──────────────────────────────────────────────────────
    if not raw:
        raw = _gap_locally(summaries)

    return (
        f"# Research Gap Analysis Report\n\n"
        f"**Papers Analysed:** {len(summaries)}\n\n"
        f"---\n\n"
        f"{raw}\n"
    )
