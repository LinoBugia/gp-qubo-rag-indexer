# RAG Index Contract v1

This is the interchange format produced by **text-embedder** and consumed by
downstream indexers (e.g. `gp-qubo-rag-indexer`). Any producer that follows
this contract can be swapped in; any consumer that relies only on what is
written here will keep working.

## 1. Directory layout

One *run* is one directory. Its name carries no meaning for consumers — read
`specs.json`, never the folder name.

```
<run-dir>/
  chunks.jsonl        required   one JSON object per line, one per chunk
  embeddings.jsonl    optional   one JSON object per line, one per chunk
  specs.json          required   manifest of the run
```

`embeddings.jsonl` is absent for a chunk-only (dry) run. In that case
`specs.json` has `files.embeddings = null` and `embedding = null`.

All files are UTF-8. `.jsonl` files are JSON Lines: exactly one JSON object
per line, newline-terminated, no wrapping array, no trailing commas.
Non-ASCII characters are written literally (not `\u`-escaped).

## 2. `chunks.jsonl`

```json
{
  "id": "jshtcBq6CmjcBHj9",
  "source": "notes/model-cards/colbert.md",
  "text": "# Handbook\n## Installation\n...",
  "embed_text": "…optional…",
  "metadata": { "...": "see below" }
}
```

| Field | Type | Presence | Meaning |
|-------|------|----------|---------|
| `id` | string | always | 16 characters from `[A-Za-z0-9]`, ~95 bits of entropy. Unique within a run and, in practice, across runs — several knowledge bases can be merged without collisions. |
| `source` | string | always | Source document as a POSIX path **relative to the input root**, e.g. `sub/dir/file.md`. Not unique per chunk. |
| `text` | string | always | **The payload.** This is what a consumer serves to the LLM or shows to the user. It already contains the heading breadcrumb where the producer added one. |
| `embed_text` | string | only when it differs | The text that was actually sent to the embedding model. Present only when the producer transformed the text for embedding (e.g. removed URLs). |
| `metadata` | object | always | See §3. Consumers must tolerate unknown keys. |

**Rule:** the vector in `embeddings.jsonl` was computed from `embed_text` when
that field is present, otherwise from `text`. A consumer that re-embeds a
chunk (for example to migrate models) must apply the same rule to stay
consistent.

## 3. `metadata`

Which keys appear depends on the chunking mode. Consumers must treat every
key as optional and must not fail on unknown keys.

| Key | Type | Appears in | Meaning |
|-----|------|-----------|---------|
| `mode` | `"marker"` \| `"token"` \| `"markdown"` | always | Chunking mode used for **this** document. A run may mix modes across documents. |
| `chunk_index` | int | always | 0-based position **within its source document**, not within the run. |
| `token_count` | int | `token`, `markdown` | Token count of the finished chunk including the heading breadcrumb, as counted by the producer's tokenizer (see `specs.chunking.tokenizer`). An estimate, not a guarantee. |
| `heading_path` | string[] | `markdown` | Heading titles from the top level down to this chunk's own heading, without `#` marks. Empty for text before the first heading. |
| `part` | int \| null | `markdown` | Part number when one section had to be split; `null` when the section fit into a single chunk. |
| `total_parts` | int | `markdown` | Number of parts the section was split into; `1` when unsplit. |
| `redacted` | `true` | optional | Markdown marks (`#`, `*`) were stripped from `text`. |
| `yaml_header_stripped` | `true` | optional | The source document had YAML front matter, which was removed before chunking. |
| `links_stripped_for_embedding` | `true` | optional | URLs were removed for the embedding; implies `embed_text` is present. |

## 4. `embeddings.jsonl`

```json
{"id": "jshtcBq6CmjcBHj9", "embedding": [0.0123, -0.0456, "..."]}
```

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | The chunk ID this vector belongs to. |
| `embedding` | float[] | The vector. Length equals `specs.embedding_dim` for every row in the file. |

Vectors are **not** normalised by the producer; a consumer that needs unit
vectors must normalise them itself.

## 5. `specs.json`

```json
{
  "created_utc": "2026-09-23T14:45:42+00:00",
  "chunking": { "mode": "markdown", "max_tokens": 600, "title_depth": 3, "...": "..." },
  "modes_by_file": { "file.md": "markdown" },
  "embedding": { "provider": "ollama", "model": "bge-m3", "...": "..." },
  "embedding_dim": 1024,
  "input_files": ["file.md", "sub/other.md"],
  "chunk_count": 4178,
  "files": { "chunks": "chunks.jsonl", "embeddings": "embeddings.jsonl" },
  "ids_by_file": { "file.md": ["jshtcBq6CmjcBHj9", "..."] },
  "ids": ["jshtcBq6CmjcBHj9", "..."],
  "reembedded_from": "RefKB_bge-m3"
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `created_utc` | string | ISO 8601, UTC, second precision. |
| `chunking` | object | The full chunking configuration of the run — enough to reproduce it. |
| `modes_by_file` | object | Actual mode per source document (a run may mix them). |
| `embedding` | object \| null | Provider, model and its parameters. `null` for a chunk-only run. **Never contains an API key** — producers must strip credentials. |
| `embedding_dim` | int \| null | Vector length. `null` for a chunk-only run. This is the authoritative dimension; do not infer it from the model name. |
| `input_files` | string[] | All processed documents, as relative paths. |
| `chunk_count` | int | Number of chunks; equals the line count of `chunks.jsonl`. |
| `files` | object | File names inside the run directory. `embeddings` is `null` for a chunk-only run. |
| `ids_by_file` | object | Chunk IDs grouped by source document, **in document order** — this is how you find a chunk's neighbours without parsing `chunks.jsonl`. |
| `ids` | string[] | All chunk IDs, flat, in file order. |
| `reembedded_from` | string | Optional. Present when the run was produced by re-embedding another run's chunks; names that source run. |

## 6. Invariants a consumer may rely on

1. **Line alignment.** Line *i* of `embeddings.jsonl` belongs to line *i* of
   `chunks.jsonl`, and `specs.ids[i]` is that same ID. A consumer may either
   zip the files positionally or join on `id`; both give the same result.
2. **Completeness.** When `embeddings.jsonl` exists it has exactly one row
   per chunk — no gaps, no extras.
3. **Uniform dimension.** Every vector in a run has length
   `specs.embedding_dim`.
4. **Unique IDs.** `id` is unique within a run.
5. **Stable IDs across model variants.** When runs are produced by
   re-embedding (`reembedded_from`), the chunk set, the IDs and their order
   are identical to the source run. Vectors from such sibling runs are
   comparable chunk by chunk via the ID.
6. **No credentials.** `specs.json` never contains an API key.

## 7. Obligations of a consumer

1. **Match the encoder.** Query embeddings must be produced with the provider
   and model named in `specs.embedding`. Vectors from different models live
   in different spaces and must never be mixed or compared.
2. **Read the dimension from `specs.embedding_dim`,** not from a hard-coded
   table of model names.
3. **Serve `text`, not `embed_text`.** `embed_text` exists only to explain
   what the vector was computed from.
4. **Tolerate absent optional fields** and unknown metadata keys.
5. **Do not rely on run directory names.** They are informational.

## 8. Reference knowledge bases

A *RefKB* is a run shipped for benchmarking, named `RefKB_<model>`. Sibling
variants of the same RefKB share one chunk set and one set of IDs and differ
only in the embedding model, which makes them directly comparable (same
question, same chunks, different encoder).

| Variant | Model | Dim |
|---------|-------|-----|
| `RefKB_bge-m3` | `bge-m3` (Ollama) | 1024 |
| `RefKB_qwen3-embedding-4b-fp16` | `qwen3-embedding:4b-fp16` (Ollama) | 2560 |

Only `RefKB_bge-m3` is small enough to ship inside a Git repository; the
other variants are reproduced locally with:

```bash
python main.py --reembed <path-to-RefKB_bge-m3> \
               --out-dir reference-kb --out-name RefKB_<model>
```

## 9. Versioning

This document describes **contract v1**. Additive changes (new optional
fields, new metadata keys) stay within v1. Any change to the meaning of an
existing field, or to an invariant in §6, requires a new version number.
