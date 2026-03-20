# RLM vs LLM (Long-Context Demo)

A Streamlit app that compares:

- **Normal LLM path**: one-shot prompt over a truncated long context.
- **RLM-style path**: retrieve relevant chunks first, then reason over them.

This is designed to make long-context failure modes visible and show why retrieval-augmented reasoning is usually stronger for large inputs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional API Key

If you want real model outputs, set:

```bash
export OPENAI_API_KEY="your_key_here"
```

Without an API key, the app uses a deterministic heuristic fallback so the comparison still works.
