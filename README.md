---
title: Colosseum Review Assistant
emoji: 🏛️
colorFrom: amber
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Colosseum Review Assistant

This Streamlit application is the deployment component of the ITC6110 NLP
project. It uses retrieval-augmented generation (RAG) to answer questions from
8,000+ Colosseum visitor reviews.

The app runs entirely with local Hugging Face models and requires no API key:

- `sentence-transformers/all-MiniLM-L6-v2` creates normalized document and
  question embeddings.
- Cosine similarity retrieves the most relevant reviews.
- `google/flan-t5-small` generates a short answer using only the retrieved
  evidence.

## Run locally

Use Python 3.12 where possible:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-app.txt
streamlit run app.py
```

The first run downloads both models and creates a local `HF-CACHE` directory.
Later runs reuse that cache. The app automatically uses CUDA when it is
available and otherwise runs on CPU.

## Deploy on Hugging Face Spaces

1. Create a new Hugging Face Space and select **Docker** as the SDK.
2. Upload or push `app.py`, `Dockerfile`, `requirements-app.txt`, this README,
   and `rome_colosseum_visitor_reviews_final.csv`.
3. Wait for the Docker build and model downloads to finish.
4. Test several questions, the retrieved-review expanders, the similarity
   threshold, error handling and the feedback control.
5. Add the final public Space URL to the project report and notebook before
   submission.

The Docker configuration exposes Streamlit on the port expected by Hugging
Face Spaces. No secret or access token is required for the public models used
by this project.

## Suggested acceptance tests

| Question | Expected evidence |
|---|---|
| How can visitors avoid queues? | Online or advance tickets, tours, queue advice |
| What do visitors say about guided tours? | Tour and guide experiences |
| What else can be visited nearby? | Roman Forum and/or Palatine Hill |
| Is wheelchair access discussed? | Accessibility evidence or an insufficient-evidence response |

For each test, confirm that the answer is supported by the displayed reviews,
that unsupported claims are not introduced, and that the interface remains
understandable on both desktop and narrow screens.

## Limitations

The source material consists of visitor opinions rather than official policy.
Ticket prices, opening hours, safety rules and accessibility arrangements can
change. Users should verify time-sensitive information through official
Colosseum sources.
