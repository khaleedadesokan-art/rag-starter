# RAG Starter — portfolio project

A working Retrieval-Augmented Generation system in one Python file, built on
aerospace documentation. **Portfolio project and starter, not a production
system** — and the sample docs are short public summaries written for this
repo, not reproductions of the real standards.

## The idea in one paragraph

A language model answers from its training data, which goes stale and can't
cite sources. RAG fixes that: when a question arrives, the system **retrieves**
the most relevant passages from your documents and **generates** the answer
from those passages, with citations. On safety-critical docs, a wrong answer
isn't a UX bug — so every answer here shows its sources.

## Architecture

```
question ──► RETRIEVE ──► top-3 chunks ──► ANSWER ──► cited answer
                  ▲
           TF-IDF index of
           chunked docs
```

1. **Load** — reads the `.txt` files in `docs/`
2. **Chunk** — splits each doc into 500-char windows with 100-char overlap
3. **Index** — builds a TF-IDF vector per chunk (rare words like "MC/DC"
   weigh more than common ones like "software"), normalized for cosine search
4. **Retrieve** — converts the question to a vector, ranks chunks by cosine
   similarity, returns top 3
5. **Answer** — extractive: quotes the retrieved chunks with
   `[Doc title — chunk n, score]` citations. A generator LLM plugs in here.

## Run it

Needs only Python 3 + numpy (no pip install).

```bash
python3 rag.py                   # demo: 3 questions + mini eval
python3 rag.py "your question"   # ask one question
python3 rag.py --eval            # retrieval hit-rate on 6 hand-written Q/A pairs
```

Current result: **Hit@3 = 6/6 (100%)** on the starter eval set.

## How to demo it (60 seconds)

1. Run `python3 rag.py` — point at the three questions and the cited chunks.
2. Ask your own: `python3 rag.py "What is TQL-1?"` — note the score column.
3. Run `python3 rag.py --eval` — "and I measure retrieval, not eyeball it."

## Failure modes (what breaks at scale)

- **TF-IDF is keyword matching.** It misses paraphrase ("coverage needed
  for the strictest level" won't match "MC/DC"). Fix: dense embeddings.
- **Fixed-size chunks split ideas.** A requirement can straddle a boundary.
  Fix: semantic/recursive chunking, or larger windows with a reranker.
- **No reranker.** TF-IDF ranks, but a cross-encoder rerank of the top 20
  would sharpen precision a lot.
- **Extractive, not generative.** Quoting chunks is honest but clunky; an
  LLM generator would synthesize — and then needs faithfulness checks.
- **Tiny corpus.** 6 docs, 24 chunks. Real corpora need metadata filters,
  dedup, and incremental indexing.

## Next steps

1. Swap TF-IDF for `sentence-transformers` dense embeddings (keep the
   same `retrieve()` interface — that's the point of the seam).
2. Add BM25 + reciprocal rank fusion (hybrid retrieval).
3. Plug an LLM into `answer()` with a "cite every claim" prompt.
4. Run the eval-harness build against this system: retrieval hit-rate,
   faithfulness, latency per change. The `--eval` flag is the seed.
