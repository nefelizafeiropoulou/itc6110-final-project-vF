"""Streamlit interface for the Colosseum review RAG system.

The application uses local Hugging Face models and requires no API key. Its
retriever, prompt and generator mirror the supervised-learning notebook.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "rome_colosseum_visitor_reviews_final.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-small"

st.set_page_config(
    page_title="Colosseum Review Assistant",
    page_icon="🏛️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_rag_components():
    """Load the dataset, embeddings and local instruction-tuned generator."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. Place it beside app.py."
        )

    review_frame = pd.read_csv(
        DATA_PATH,
        usecols=["text", "rating", "travel_month", "tripType"],
    )
    review_frame = review_frame.dropna(subset=["text"]).reset_index(drop=True)
    review_frame["text"] = review_frame["text"].astype(str).str.strip()
    review_frame = review_frame[review_frame["text"].ne("")].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(APP_DIR / "HF-CACHE"),
        device=device,
    )
    review_embeddings = encoder.encode(
        review_frame["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_MODEL,
        cache_dir=str(APP_DIR / "HF-CACHE"),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_MODEL,
        cache_dir=str(APP_DIR / "HF-CACHE"),
    ).to(device)
    model.eval()
    return review_frame, encoder, review_embeddings, tokenizer, model, device


def retrieve_reviews(question, review_frame, encoder, review_embeddings, k=4):
    """Return the top-k review documents and their cosine similarities."""
    question_embedding = encoder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    scores = review_embeddings @ question_embedding
    best_ids = np.argsort(scores)[::-1][: min(k, len(scores))]

    results = []
    for review_id in best_ids:
        row = review_frame.iloc[int(review_id)]
        results.append(
            {
                "document_id": int(review_id),
                "similarity": float(scores[review_id]),
                "text": row["text"],
                "rating": row.get("rating"),
                "travel_month": row.get("travel_month"),
                "trip_type": row.get("tripType"),
            }
        )
    return results


def build_grounded_prompt(question, retrieved_reviews):
    """Create the same evidence-constrained prompt used in the notebook."""
    context_blocks = []
    for rank, review in enumerate(retrieved_reviews, start=1):
        excerpt = review["text"].replace("\n", " ")[:650]
        context_blocks.append(f"[Review {rank}] {excerpt}")
    context = "\n".join(context_blocks)
    return (
        "You are a Colosseum visitor-review assistant. Answer only from the "
        "review evidence below. Summarize rather than copy. Do not invent facts. "
        "If the evidence does not answer the question, say: 'The retrieved "
        "reviews do not provide enough information.' Keep the answer under "
        "80 words.\n\n"
        f"Question: {question}\n\nReview evidence:\n{context}\n\nAnswer:"
    )


def generate_answer(question, retrieved_reviews, tokenizer, model, device):
    """Generate a deterministic answer grounded in the retrieved reviews."""
    prompt = build_grounded_prompt(question, retrieved_reviews)
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def rag_answer(question, k=4, minimum_similarity=0.20):
    """Run retrieval and generation and return answer, evidence and latency."""
    review_frame, encoder, review_embeddings, tokenizer, model, device = (
        load_rag_components()
    )
    started = time.perf_counter()
    retrieved = retrieve_reviews(
        question,
        review_frame,
        encoder,
        review_embeddings,
        k=k,
    )

    if not retrieved or retrieved[0]["similarity"] < minimum_similarity:
        answer = "The retrieved reviews do not provide enough information."
    else:
        answer = generate_answer(
            question, retrieved, tokenizer, model, device
        )
    return answer, retrieved, time.perf_counter() - started


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []


with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider("Number of reviews", 2, 8, 4)
    minimum_similarity = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="Below this score, the assistant reports insufficient evidence.",
    )
    st.caption(f"Embedding model: `{EMBEDDING_MODEL.split('/')[-1]}`")
    st.caption(f"Generator: `{GENERATOR_MODEL}`")
    st.caption("Models run locally; no API key is required.")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("Limitations and responsible use"):
        st.write(
            "Answers summarize visitor reviews, not official Colosseum policy. "
            "Models can make mistakes; verify tickets, opening hours, prices and "
            "accessibility information using official sources."
        )


st.title("🏛️ Colosseum Review Assistant")
st.write(
    "Ask about queues, tickets, tours, crowds, accessibility or other experiences "
    "reported by Colosseum visitors. Every answer includes its retrieved evidence."
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(message["metadata"])
            with st.expander("Retrieved visitor reviews"):
                for position, review in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Review {position}** · similarity "
                        f"{review['similarity']:.3f}"
                    )
                    st.write(review["text"])


question = st.chat_input(
    "For example: How can visitors avoid long ticket queues?"
)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Loading models and searching visitor reviews..."):
                answer, sources, latency = rag_answer(
                    question,
                    k=top_k,
                    minimum_similarity=minimum_similarity,
                )
            st.markdown(answer)
            metadata = f"Generated in {latency:.2f} seconds from {len(sources)} reviews"
            st.caption(metadata)

            with st.expander("Retrieved visitor reviews"):
                for position, review in enumerate(sources, start=1):
                    details = [f"similarity {review['similarity']:.3f}"]
                    if pd.notna(review["rating"]):
                        details.append(f"rating {review['rating']}/5")
                    if pd.notna(review["travel_month"]):
                        details.append(f"travel month {review['travel_month']}")
                    if pd.notna(review["trip_type"]):
                        details.append(f"trip type {review['trip_type']}")
                    st.markdown(f"**Review {position}** · " + " · ".join(details))
                    st.write(review["text"])
                    if position < len(sources):
                        st.divider()

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                    "sources": sources,
                }
            )
        except Exception as error:
            st.error("The review assistant could not complete the request.")
            st.exception(error)

if st.session_state.chat_history:
    st.divider()
    st.write("Was the latest answer useful?")
    feedback = st.feedback("thumbs", key=f"feedback_{len(st.session_state.chat_history)}")
    if feedback is not None:
        st.session_state.feedback.append(int(feedback))
        st.caption("Thank you—this session-only rating supports UX evaluation.")
