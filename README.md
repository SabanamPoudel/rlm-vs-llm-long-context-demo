# RLM vs LLM (Long-Context Demo)

A Streamlit app that compares:

- **Normal LLM path**: one-shot prompt over a truncated long context.
- **RLM-style path**: retrieve relevant chunks first, then reason over them.

This is designed to make long-context failure modes visible and show why retrieval-augmented reasoning is usually stronger for large inputs.

## Live Demo

Open the deployed app here:

[RLM vs LLM Streamlit App](https://rlm-vs-llm-long-context-demo.streamlit.app/)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

