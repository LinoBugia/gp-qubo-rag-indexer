# gp-qubo-rag-indexer

An approximate-nearest-neighbour index for RAG, built as a **tree** rather
than a **graph**. The chunk embeddings get cut in half, each half again, and
so on until every group is small — and each of those cuts is a
graph-partitioning problem, solved by **Lloyd's algorithm** or by a **digital
annealer** on a QUBO.

What comes out is **one folder that is both the index and the database**: the
split planes, the embeddings, the chunk files, and a `query.py` next to them.
That script is the whole interface — hand it a finished query embedding and it
returns the paths of the best chunks. It imports nothing but numpy, holds no
graph in memory, and runs **entirely locally**: no server, no daemon, nothing
to keep warm, so it can be dropped into a pipeline or handed to an agent as
it is. What goes in is the output of
**[text-embedder](../text-embedder/)**, and that handover is a written
contract — [`RAG_INDEX_CONTRACT.md`](RAG_INDEX_CONTRACT.md), the same file in
both repositories.

![Recall against work: HNSW, this index with and without the annealer, FAISS IVF](docs/img/pareto.svg)

It is a working, efficient index, and it is not as strong as the
neighbour-graph indexes: HNSW reaches the same recall with **2.8–4.0× fewer
scalar products**. Against FAISS IVF it comes out ahead throughout. All of it
is measured in [docs/benchmarks.md](docs/benchmarks.md).

## Core concept

**Divide and conquer through recursive bisection:** take a group of
embeddings — or arbitrary points in a space — cut it in two, recurse into both
halves, stop when a group is small enough to read exhaustively. The tree of
those cuts **is** the index. Its inner nodes hold no data at all, only a
decision — one plane, stored as a normal vector — so a query walks from the
root answering one question per level, each question a single dot product.

Nothing about that is specific to retrieval. The same recursive bisection is
used on the travelling salesman problem, and generally wherever a point set
has to be split and each part solved on its own.

![Recursive bisection: each level splits every group in two](docs/img/concept-bisection.svg)

Which means the quality lives entirely in **where the cuts go**. Slice through
a dense region and every query landing there follows the wrong branch with no
way back. Three levels of effort address that, and each can be switched on by
itself:

| Level | What it does | Cost on 4178 chunks |
|---|---|---|
| **Lloyd (k=2)** | The naive cut, and a decent one — a handful of iterations per group and most splits are already sensible. The baseline everything else has to beat | 14 s |
| **QUBO + digital annealing** | One binary variable per chunk, minimising an energy over *all* pairwise similarities at once instead of iterating centroids. Warm-started from Lloyd | 223 s |
| **Post-processing on the finished split** | The partition an optimiser likes is not automatically one a greedy descent can follow, so the boundary is re-derived as a plane, shifted into the emptiest gap, and given an overlap zone plus a tolerance ε measured from the encoder's own noise | free |

→ **[docs/concept.md](docs/concept.md)** for the reasoning and the figures,
**[docs/pipeline.md](docs/pipeline.md)** for one bisection step by step.

## Quickstart

```bash
pip install numpy          # indexing and querying need nothing else
```

> The GUI additionally needs `customtkinter`, and reproducing the baseline
> comparison needs `faiss-cpu`. Neither is required to build or query an index.

The repository ships a reference knowledge base — 4178 chunks, 1024d, bge-m3 —
so the first index needs no encoder and no downloads:

```bash
# build: Lloyd only, 14 s on an idle machine
python3 rag_indexer.py reference-kb/RefKB_bge-m3 -o rag_index_lloyd --no-anneal

# the same tree with the digital annealer, 223 s
python3 rag_indexer.py reference-kb/RefKB_bge-m3 -o rag_index_da
```

Querying takes a finished query vector — the embedding happens outside. One of
the 300 benchmark questions makes a copy-pasteable example:

```bash
python3 -c "import numpy, json; print(json.dumps(numpy.load(
  'benchmarks/queries/RefKB_bge-m3.npz')['queries'][0].tolist()))" \
  | python3 rag_index_lloyd/query.py -k 5
```

It prints the paths of the five best chunks. `--json` adds scores and the
encoder the index was built with; exit code 2 means the query came from a
different model.

## Commands

| Command | What it does |
|---|---|
| `python3 rag_indexer.py <run-dir> -o <out>` | Build an index from a [text-embedder](../text-embedder/) run folder. `--no-anneal` for Lloyd only, `-g` for the leaf size, `--objective` for what the bisection minimises — every flag in [configuration.md](docs/configuration.md) |
| `python3 <index>/query.py -k 5` | Query a finished index. `--beam` trades work for recall, `--json` returns scores |
| `python3 rag_gui.py` | The GUI: pipeline dialog, partition map, live queries |
| `python3 benchmarks/run_matrix.py --kbs …` | Rebuild the benchmark matrix; `evaluate_full.py` and `baselines.py` produce the tables |
| `python3 docs/img/make_figures.py` | Regenerate all nine figures from the measured results |

## What an index looks like

Everything the query needs, in one place:

```
rag_index_lloyd/
├── query.py               # standalone query script (numpy only)
├── ann_tree.json          # the index: split planes + leaf chunk rows
├── embeddings.npy         # (n, d) float32 chunk embeddings
├── index_meta.json        # encoder spec + partition parameters
├── indexing_stats.json    # per-bisection KPIs — energy, steps, times
├── chunks_manifest.json   # chunk id → path / source / heading
└── chunks/<id>.md         # one file per chunk — these paths are returned
```

[docs/knowledge-bases.md](docs/knowledge-bases.md) says which clauses of the
input contract this indexer relies on, and which reference bases the numbers
come from.

## Documentation

| File | Contains |
|---|---|
| [docs/concept.md](docs/concept.md) | Why a tree, how recursive bisection builds it, what the two bisectors do — with the figures |
| [docs/pipeline.md](docs/pipeline.md) | One bisection step by step: split modes, overlap, the ε calibration, node representation |
| [docs/configuration.md](docs/configuration.md) | Every setting in one table, and the measurement behind each chosen default |
| [docs/query.md](docs/query.md) | The beam descent, `query.py`'s interface, embedding providers, recall against speed |
| [docs/benchmarks.md](docs/benchmarks.md) | The full measurement chapter: HNSW and FAISS IVF, the variant matrix, scaling, query cost, and the measurement traps |
| [docs/knowledge-bases.md](docs/knowledge-bases.md) | What goes in, which contract clauses are used, and the reference bases |
| [docs/gram-matrix.md](docs/gram-matrix.md) | The sparse Gram matrix builder underneath: CLI, Python API, storage format |
| [RAG_INDEX_CONTRACT.md](RAG_INDEX_CONTRACT.md) | Normative input spec — shared verbatim with [text-embedder](../text-embedder/) |

## Project structure

| Path | Purpose |
|---|---|
| `rag_indexer.py` | Run folder → index folder. The pipeline described in `docs/pipeline.md` |
| `rag_query.py` | Template for the `query.py` written into every index |
| `rag_gui.py` | CustomTkinter GUI: pipeline dialog, partition map, queries |
| `graph_partitioning.py` | Recursive bisection → partition tree; both objectives, both bisectors |
| `clustering.py`, `gram_matrix.py`, `cli.py`, `example.py` | The sparse Gram matrix component and its interfaces |
| `benchmarks/` | Measurement scripts and their results as JSON — every number in the docs comes from here |
| `experiments/` | Diagnostics and measured dead ends, kept because they answer why a tree cannot replace a graph |
| `docs/` | Documentation; `docs/img/` holds the figures and the script that generates them |
| `reference-kb/` | The reference knowledge bases every measurement is made on |

## License

MIT — see [LICENSE](LICENSE).
