# Querying an index

How the beam descends the tree, what `query.py` takes and returns, and
which encoders a query can come from.

- [Calling `query.py`](#calling-querypy)
- [How the beam descends](#how-the-beam-descends)
- [Embedding providers](#embedding-providers)

---

## Calling `query.py`

Querying is a handful of scalar products. **The embedding happens outside**:
you feed a finished query vector in, you get chunk file paths back.

```bash
# JSON array from file, top 5 chunks
python3 rag_index_001/query.py --embedding query_emb.json -k 5

# or via stdin, with encoder guard and full JSON output
echo "[0.12, -0.03, ...]" | python3 rag_index_001/query.py -k 5 \
    --encoder nomic-embed-text --json
```

## How the beam descends

The descent follows the tree with a beam ("save strategy"): the best
`--beam` branches survive each level (default 8), and the beam widens
automatically until at least `--min-candidates` (default 3·k) chunks reach the
exact final ranking. One beam slot is **pinned to the greedy path** — the
exact per-node decision, which is the side of `q·ŵ = τ` on a plane tree and
the nearer centroid otherwise; the remaining slots go by cosine against the
child means. Without pinning both slots can fall into the same subtree and
evict the branch the query belongs to. The two scores answer different
questions and that is why both are used: the plane says which side of *this*
node the query is on, the cosine says how well a node matches the query at all
— and only the latter is comparable across different parents, which the beam
requires. Measured, pinning lifts chunk retrieval from 98.2 % to 100 % at a
cost of 0.4 points of recall@5, and it works on existing indexes too.
Exit code 2 signals an encoder/dimension mismatch.

`query.py --json` reports `index_chunks` and `encoder` of the knowledge base
it serves.

## Embedding providers

The query is embedded with whatever encoder the
knowledge base was built with (`index_meta.json`). Supported out of the box:
- `ollama` — any local model (nomic-embed-text, bge-m3, embeddinggemma,
  mxbai-embed-large, snowflake-arctic-embed, multilingual-e5, qwen3-embedding, …);
  the connection field holds the base URL.
- `gemini` — Google AI Studio API (gemini-embedding-001, text-embedding-004);
  the connection field turns into a masked API-key input, prefilled from
  `$GEMINI_API_KEY` / `$GOOGLE_API_KEY`. The query embedding is requested
  with `outputDimensionality` = index dimension, so truncated
  gemini-embedding-001 indexes rank correctly (cosine normalizes).
  Free-Tier note: Google trains on free-tier data — keep sensitive corpora
  on paid tier or local Ollama.

## Recall against speed

Widening the beam trades candidates for quality, and it is by far the cheapest
knob in the whole project — it costs nothing at build time and needs no
rebuild, because the beam is chosen per query. On the 4178-chunk reference
base (`knn-da`, one mean per node):

| `--beam` | recall@5 | scalar products per query |
|---|---|---|
| 1 | 39.5 % | 34 |
| 2 | 59.6 % | 65 |
| 4 | 75.9 % | 123 |
| **8** (default) | **86.7 %** | **232** |
| 16 | 92.5 % | 431 |
| 32 | 96.0 % | 787 |

Raise `--beam` / `--min-candidates` when recall matters more than
microseconds; the full numbers, including query times and both encoders, are
in [Benchmarks](benchmarks.md#the-cheapest-quality-knob-is-the-beam).
