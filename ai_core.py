"""
ai_core.py  –  Research paper summarisation
--------------------------------------------
Strategy (free, no paid API needed):

  Priority 1 → HuggingFace FREE Inference API  (fast, no local GPU needed)
               Requires HF_API_TOKEN in .env
               Get free token: https://huggingface.co/settings/tokens

  Priority 2 → Local transformers pipeline (CPU fallback, ~1.6 GB download)
               Works with transformers 4.x AND 5.x
               Uses "text2text-generation" task (NOT "summarization" which was
               removed in transformers 5.x and causes the error you saw)
"""

import requests
from config import HF_API_TOKEN, HF_SUMMARY_MODEL


def _trim(text: str, max_chars: int = 3000) -> str:
    """Keep only the first max_chars characters (enough context for a summary)."""
    return text[:max_chars]


def _summarise_via_hf_api(text: str) -> str:
    """
    Call the HuggingFace FREE Inference API.
    facebook/bart-large-cnn is a public model – no approval needed.
    """
    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 200,
            "min_length": 60,
            "do_sample": False,
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # API returns: [{"summary_text": "..."}]
    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"].strip()

    # Sometimes returns an error dict while model is loading
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF API model loading: {data['error']}")

    raise ValueError(f"Unexpected HF API response format: {data}")


def _summarise_locally(text: str) -> str:
    """
    Local CPU fallback using facebook/bart-large-cnn.

    KEY FIX: transformers 5.x removed the 'summarization' pipeline task.
    We use 'text2text-generation' which works on BOTH 4.x and 5.x.
    BART is a seq2seq model so text2text-generation is semantically correct.
    """
    try:
        # ── Try transformers pipeline ────────────────────────────────────────
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(HF_SUMMARY_MODEL)
        model     = AutoModelForSeq2SeqLM.from_pretrained(HF_SUMMARY_MODEL)

        # Use the task name that exists in BOTH transformers 4.x and 5.x
        summariser = pipeline(
            "text2text-generation",      
            model=model,
            tokenizer=tokenizer,
        )

        result = summariser(
            text,
            max_new_tokens=200,
            truncation=True,
        )
        return result[0]["generated_text"].strip()

    except Exception as inner_e:
        # ── Final fallback: simple extractive summary ────────────────────────
        # If transformers itself is broken / no torch, return first 3 sentences
        print(f"[ai_core] Local model failed ({inner_e}). Using extractive fallback.")
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 40]
        return ". ".join(sentences[:5]) + "." if sentences else text[:500]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(text: str) -> str:
    """
    Summarise a research paper text.
    Tries HF Inference API first (if token present), then local model,
    then extractive fallback – so it ALWAYS returns something.
    """
    snippet = _trim(text)

    # ── Path 1: HF free API ──────────────────────────────────────────────────
    if HF_API_TOKEN:
        try:
            summary = _summarise_via_hf_api(snippet)
            if summary:
                return summary
        except Exception as e:
            print(f"[ai_core] HF API failed ({e}). Falling back to local model…")

    # ── Path 2 + 3: local model or extractive ───────────────────────────────
    return _summarise_locally(snippet)