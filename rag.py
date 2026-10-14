#!/usr/bin/env python3
"""
RAG starter — portfolio project (not production, not work history).

Retrieval-Augmented Generation in one file, zero dependencies beyond numpy:
    1. LOAD    read the sample docs from ./docs
    2. CHUNK   split each doc into overlapping windows
    3. INDEX   build a TF-IDF vector for every chunk
    4. RETRIEVE rank chunks by cosine similarity to the question
    5. ANSWER  present the top chunks as a cited, extractive answer

Usage:
    python3 rag.py                  run the demo (3 questions + mini eval)
    python3 rag.py "your question"  answer one question
    python3 rag.py --eval           retrieval hit-rate on hand-written Q/A pairs

Upgrade path (see README): swap TF-IDF for dense embeddings
(sentence-transformers), add BM25 + reranker, plug an LLM into step 5.
"""

import math
import os
import re
import sys
from collections import Counter

import numpy as np

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHUNK_SIZE = 500   # characters per chunk
CHUNK_OVERLAP = 100
TOP_K = 3

STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been by as at "
    "it its this that these those from which who whom what when where how "
    "all any each every no not only also than then there their they them "
    "he she we you your his her our will would can could should may might "
    "do does did done has have had such into over under more most other some "
    "if but about between through during".split()
)


# ---------------------------------------------------------------- LOAD ----
def load_docs(docs_dir=DOCS_DIR):
    """Read every .txt file in docs/ -> list of {title, text}."""
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(docs_dir, fname), encoding="utf-8") as f:
            text = f.read().strip()
        title = text.splitlines()[0].strip()
        docs.append({"title": title, "file": fname, "text": text})
    return docs


# --------------------------------------------------------------- CHUNK ----
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fixed-size character windows with overlap so ideas aren't cut mid-sentence."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def build_chunks(docs):
    """Flatten docs -> list of {doc_title, file, idx, text}."""
    chunks = []
    for doc in docs:
        for i, ctext in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {"doc_title": doc["title"], "file": doc["file"],
                 "idx": i, "text": ctext}
            )
    return chunks


# --------------------------------------------------------------- INDEX ----
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS and len(t) > 1]


def build_index(chunks):
    """
    TF-IDF index. Each chunk becomes a vector where rare, distinctive
    words (like 'MC/DC') weigh more than common ones (like 'software').
    Vectors are L2-normalized so dot product == cosine similarity.
    """
    tokenized = [tokenize(c["text"]) for c in chunks]
    vocab = sorted({t for toks in tokenized for t in toks})
    word_to_id = {w: i for i, w in enumerate(vocab)}

    n = len(chunks)
    df = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1
    idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in vocab])

    mat = np.zeros((n, len(vocab)))
    for i, toks in enumerate(tokenized):
        counts = Counter(toks)
        for t, c in counts.items():
            mat[i, word_to_id[t]] = c * idf[word_to_id[t]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return {"vocab": word_to_id, "idf": idf, "matrix": mat / norms}


def embed_query(query, index):
    """Turn the question into a vector in the same space as the chunks."""
    vec = np.zeros(len(index["vocab"]))
    for t, c in Counter(tokenize(query)).items():
        if t in index["vocab"]:
            vec[index["vocab"][t]] = c * index["idf"][index["vocab"][t]]
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ------------------------------------------------------------ RETRIEVE ----
def retrieve(query, chunks, index, k=TOP_K):
    """Cosine similarity between question vector and every chunk vector."""
    q = embed_query(query, index)
    scores = index["matrix"] @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top if scores[i] > 0]


# --------------------------------------------------------------- ANSWER ----
def answer(query, chunks, index, k=TOP_K):
    """Extractive answer: the retrieved chunks, cited. A generator LLM
    would plug in here, writing from these chunks instead of quoting them."""
    hits = retrieve(query, chunks, index, k)
    if not hits:
        return "No relevant passages found. Try different wording."
    lines = [f"Q: {query}\n"]
    for rank, (chunk, score) in enumerate(hits, 1):
        lines.append(
            f"[{rank}] {chunk['doc_title']} (chunk {chunk['idx']}, "
            f"score {score:.3f})\n    {chunk['text'][:320]}..."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------- EVAL ----
# Hand-written question -> file that MUST be retrieved. Hit = expected doc
# in the top-k. This is the seed of the eval-harness habit: every RAG
# change gets measured, not eyeballed.
EVAL_SET = [
    ("What coverage is required for Level A software?", "do-178c-verification.txt"),
    ("How many objectives does Level A have?", "do-178c-software-levels.txt"),
    ("What does AC 20-115D recognize?", "faa-ac-20-115d.txt"),
    ("What is a Development Assurance Level?", "arp4754a-dal-allocation.txt"),
    ("What is TQL-1?", "do-330-tool-qualification.txt"),
    ("Why does dead code matter for traceability?", "traceability.txt"),
]


def run_eval(chunks, index, k=TOP_K):
    hits = 0
    for question, expected_file in EVAL_SET:
        retrieved_files = [c["file"] for c, _ in retrieve(question, chunks, index, k)]
        ok = expected_file in retrieved_files
        hits += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {question}")
    print(f"\nHit@{k}: {hits}/{len(EVAL_SET)} = {hits/len(EVAL_SET):.0%}")


# ------------------------------------------------------------------ CLI ----
DEMO_QUESTIONS = [
    "What test coverage does Level A require?",
    "How do Development Assurance Levels relate to DO-178C software levels?",
    "What is tool qualification and when is TQL-1 needed?",
]


def main():
    docs = load_docs()
    chunks = build_chunks(docs)
    index = build_index(chunks)
    print(f"Indexed {len(docs)} docs -> {len(chunks)} chunks.\n")

    args = sys.argv[1:]
    if args == ["--eval"]:
        run_eval(chunks, index)
    elif args:
        print(answer(" ".join(args), chunks, index))
    else:
        for q in DEMO_QUESTIONS:
            print(answer(q, chunks, index))
            print("\n" + "-" * 70 + "\n")
        print("Mini eval:")
        run_eval(chunks, index)


if __name__ == "__main__":
    main()
