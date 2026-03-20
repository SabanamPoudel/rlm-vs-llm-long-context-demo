import os
import re

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def add_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        .stApp {
            font-family: "Space Grotesk", sans-serif;
            background:
                radial-gradient(1100px 500px at 0% 0%, rgba(0, 128, 255, 0.14), transparent 65%),
                radial-gradient(900px 450px at 100% 0%, rgba(255, 125, 66, 0.14), transparent 60%),
                linear-gradient(140deg, #eef6ff, #f6fbff 48%, #fff8f2);
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none;
        }

        .hero {
            border: 1px solid rgba(20, 33, 61, 0.14);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(8px);
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(1.35rem, 2.3vw, 2.1rem);
            line-height: 1.15;
            color: #13233f;
        }

        .hero p {
            margin: 0.48rem 0 0;
            color: #335074;
        }

        .section-title {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #0c7697;
            font-size: 0.84rem;
            font-weight: 700;
            margin: 0.9rem 0 0.55rem;
        }

        .stButton > button {
            border-radius: 999px;
            font-weight: 650;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(110deg, #0c7697, #2f9ab8 50%, #ff7d42);
            color: white;
            border: none;
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.86) !important;
            border: 1px solid rgba(20, 33, 61, 0.16) !important;
        }

        /* Tablet: tighten spacing and make controls easier to tap */
        @media (max-width: 1024px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 0.9rem 0.95rem;
            }

            .stButton > button {
                min-height: 42px;
            }
        }

        /* Mobile: force one-column flow for all content rows */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }

            .hero h1 {
                font-size: 1.4rem;
            }

            .hero p {
                font-size: 0.92rem;
            }

            .main [data-testid="stHorizontalBlock"] {
                flex-direction: column;
                gap: 0.65rem;
            }

            .main [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            [data-testid="stTextArea"] textarea {
                min-height: 170px !important;
            }

            .stButton > button {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_tokens(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def make_chunks(text, chunk_words=220, overlap_words=40):
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    idx = 1
    jump = chunk_words - overlap_words
    if jump <= 0:
        jump = 1

    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunk_text = " ".join(words[i:end])
        chunks.append({"idx": idx, "text": chunk_text})
        i += jump
        idx += 1

    return chunks


def score_chunk(chunk_text, question):
    q = to_tokens(question)
    c = to_tokens(chunk_text)
    if not q or not c:
        return 0.0

    counts = {}
    for x in c:
        counts[x] = counts.get(x, 0) + 1

    # simple lexical match score, nothing fancy
    s = 0.0
    for x in q:
        s += counts.get(x, 0)

    return s / (len(c) ** 0.5)


def get_top_chunks(chunks, question, top_k=4):
    scored = []
    for c in chunks:
        sc = score_chunk(c["text"], question)
        scored.append((c, sc))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, min(top_k, len(scored)))]


def heuristic_answer(context, question):
    q = set(to_tokens(question))
    parts = re.split(r"(?<=[.!?])\s+", context)

    best_sent = ""
    best_score = -1
    for p in parts:
        p_set = set(to_tokens(p))
        hit = len(q.intersection(p_set))
        if hit > best_score:
            best_score = hit
            best_sent = p.strip()

    if best_sent and best_score > 0:
        return "Likely answer from context: " + best_sent
    return "I could not find a grounded answer in the provided context."


def ask_openai(model, system_prompt, user_prompt):
    if OpenAI is None:
        raise RuntimeError("openai package not available")

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=key)
    res = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return res.choices[0].message.content or ""


def run_normal_path(long_text, question, model, context_char_limit):
    # This cut is done on purpose so long-context weakness is visible.
    visible = long_text[:context_char_limit]

    sys = "You answer from context only. If unsure, say unsure."
    usr = (
        "Context:\n"
        + visible
        + "\n\nQuestion: "
        + question
        + "\nGive a concise answer and mention if context might be incomplete."
    )

    try:
        ans = ask_openai(model, sys, usr)
        mode = "OpenAI"
    except Exception:
        ans = heuristic_answer(visible, question)
        mode = "Heuristic fallback"

    return ans, mode


def run_rlm_path(long_text, question, model, chunk_words, overlap_words, top_k):
    chunks = make_chunks(long_text, chunk_words, overlap_words)
    top = get_top_chunks(chunks, question, top_k)

    lines = []
    for item, sc in top:
        lines.append(f"[Chunk {item['idx']} | score={sc:.2f}]\n{item['text']}")
    retrieved_text = "\n\n".join(lines)

    sys = (
        "You are an RLM-style assistant. Use only retrieved chunks. "
        "Cite chunk IDs like [Chunk 3]. If missing evidence, say insufficient evidence."
    )
    usr = (
        "Retrieved chunks:\n"
        + retrieved_text
        + "\n\nQuestion: "
        + question
        + "\nAnswer with short reasoning and chunk citations."
    )

    try:
        ans = ask_openai(model, sys, usr)
        mode = "OpenAI"
    except Exception:
        local_context = "\n".join([x[0]["text"] for x in top])
        ans = heuristic_answer(local_context, question)
        mode = "Heuristic fallback"

    return ans, top, chunks, mode


def build_demo_doc(repeats=140):
    filler = (
        "Quarterly operations summary: inventory moved through multiple warehouses, "
        "regional teams adjusted staffing, and product lines were rebalanced. "
        "Customer service metrics changed with seasonality while shipment timing stayed stable."
    )

    block = []
    for i in range(repeats):
        block.append(f"Section {i + 1}: {filler}")

    final_fact = (
        "Critical audit note: The emergency rollback code for Project Atlas is RLM-9274-ZETA, "
        "approved on 14 February 2026 by the resilience board."
    )
    block.append("\nFINAL APPENDIX:\n" + final_fact)

    doc = "\n\n".join(block)
    q = "What is the emergency rollback code for Project Atlas?"
    expected = "RLM-9274-ZETA"
    return doc, q, expected


def load_demo():
    doc, q, expected = build_demo_doc()
    st.session_state["long_text"] = doc
    st.session_state["question"] = q
    st.session_state["expected"] = expected


def has_expected(answer, expected):
    return expected.lower() in answer.lower()


def main():
    st.set_page_config(page_title="RLM vs LLM", layout="wide")
    add_css()

    st.markdown(
        """
        <div class="hero">
            <h1>RLM vs LLM for Long-Context QA</h1>
            <p>Compare single-pass prompting with retrieval-first reasoning and inspect what each method sees.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "long_text" not in st.session_state:
        st.session_state["long_text"] = ""
    if "question" not in st.session_state:
        st.session_state["question"] = ""
    if "expected" not in st.session_state:
        st.session_state["expected"] = ""

    # quick stats near top
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Doc chars", f"{len(st.session_state['long_text']):,}")
    with c2:
        st.metric("Doc words (approx)", f"{len(st.session_state['long_text'].split()):,}")

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Model", value="gpt-4o-mini")
        context_char_limit = st.slider("Normal LLM context cap (chars)", 800, 12000, 3000, 200)
        chunk_words = st.slider("RLM chunk size (words)", 80, 400, 220, 20)
        overlap_words = st.slider("RLM overlap (words)", 0, 120, 40, 10)
        top_k = st.slider("RLM retrieved chunks", 1, 8, 4, 1)

    st.markdown('<div class="section-title">Input Workspace</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        long_text = st.text_area(
            "Long document",
            key="long_text",
            height=320,
            placeholder="Paste a long context here...",
        )

    with right:
        question = st.text_area(
            "Question",
            key="question",
            height=140,
            placeholder="Ask a question about the long document...",
        )
        expected = st.text_input("Expected answer (optional, for scoring)", key="expected")
        st.button("Load built-in long-context demo", on_click=load_demo)

    st.caption("Tip: keep normal LLM context cap low versus document size to show truncation issues.")

    run = st.button("Run Comparison", type="primary")
    if not run:
        return

    if not long_text.strip() or not question.strip():
        st.warning("Please provide both long text and a question.")
        return

    with st.spinner("Running normal LLM path..."):
        normal_answer, normal_mode = run_normal_path(long_text, question, model, context_char_limit)

    with st.spinner("Running RLM-style path..."):
        rlm_answer, top, all_chunks, rlm_mode = run_rlm_path(
            long_text,
            question,
            model,
            chunk_words,
            overlap_words,
            top_k,
        )

    st.caption(f"Document split into {len(all_chunks)} chunks for retrieval.")

    st.markdown('<div class="section-title">Comparison Results</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("### Normal LLM")
        st.write(normal_answer)
        vis = min(len(long_text), context_char_limit)
        st.caption(f"Mode: {normal_mode} | Visible context: {vis} chars")

    with r2:
        st.markdown("### RLM-style Pipeline")
        st.write(rlm_answer)
        st.caption(f"Mode: {rlm_mode} | Retrieved chunks: {len(top)}")

    st.markdown("### Retrieved Evidence")
    for item, sc in top:
        with st.expander(f"Chunk {item['idx']} (score={sc:.2f})"):
            st.write(item["text"])

    if expected.strip():
        st.markdown("### Simple Score")
        normal_hit = has_expected(normal_answer, expected)
        rlm_hit = has_expected(rlm_answer, expected)
        a, b, c = st.columns(3)
        a.metric("Expected token", expected)
        b.metric("Normal LLM hit", "Yes" if normal_hit else "No")
        c.metric("RLM hit", "Yes" if rlm_hit else "No")

        if rlm_hit and not normal_hit:
            st.success("RLM-style pipeline found the expected answer while normal LLM missed it.")
        elif normal_hit and not rlm_hit:
            st.info("Normal LLM found it first for this case. You can tune retrieval settings.")
        elif normal_hit and rlm_hit:
            st.info("Both approaches found the expected answer on this input.")
        else:
            st.warning("Neither approach found the expected answer. Try changing settings.")


if __name__ == "__main__":
    main()
