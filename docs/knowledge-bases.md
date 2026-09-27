# Knowledge bases and the input contract

What this indexer consumes, which parts of the producer's contract it
relies on, and the reference bases every measurement is made on.

---

This indexer does not chunk or embed. It consumes finished runs of the
companion project **[text-embedder](../../text-embedder/)**, and the handover is
specified: [`RAG_INDEX_CONTRACT.md`](../RAG_INDEX_CONTRACT.md) in this repo is a
**verbatim copy** of the producer's contract — same file on both sides, so
neither can drift without the other noticing.

A run is a directory with three files:

```
<run-dir>/
  chunks.jsonl      {id, source, text, embed_text?, metadata}
  embeddings.jsonl  {id, embedding}
  specs.json        the manifest — encoder, dimension, chunk count, ids
```

What this indexer relies on, and where:

| Contract | Where it is used here |
|---|---|
| §6.1 line alignment / join on `id` | `load_run_folder` joins on `id` and fails loudly on a mismatch, rather than trusting row order |
| §5 `embedding_dim` is authoritative | the dimension is read from `specs.json` and checked against the actual vectors — never inferred from the model name |
| §5 `ids_by_file` in document order | `benchmarks/make_subsets.py` cuts smaller knowledge bases **by whole documents**, which needs exactly this grouping |
| §2 the vector came from `embed_text` when present | the ε calibration perturbs `embed_text or text`, so it measures the encoder and not a text mismatch |
| §7.1 match the encoder | `--eps auto` re-embeds through `specs.embedding`; `query.py --encoder NAME` refuses a query from a different model |
| §7.3 serve `text`, never `embed_text` | the chunk files written to `chunks/` contain `text` |
| §1 chunk-only runs | a run with `files.embeddings = null` is rejected with a clear message instead of a traceback |

**Reference knowledge bases.** `reference-kb/RefKB_bge-m3` (4178 chunks, 1024d)
ships with this repo and is what every number below was measured on. Its
sibling `RefKB_qwen3-embedding-4b-fp16` (same chunks, same IDs, 2560d) is
144 MB and therefore git-ignored — reproduce it locally with the re-embed
command in the contract (§8). Because siblings share their chunk IDs, the two
are comparable chunk by chunk: same questions, same chunks, different encoder.
