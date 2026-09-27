#!/usr/bin/env python3
"""
query.py — standalone RAG query over the ANN partition tree in this folder
===========================================================================

Lives inside a rag_index/ folder produced by rag_indexer.py and needs only
numpy + stdlib.  The embedding is produced OUTSIDE this script — you feed a
finished query vector in, you get chunk file paths back.

Usage
-----
    python query.py --embedding query_emb.json -k 5
    echo "[0.12, -0.03, ...]" | python query.py -k 5
    python query.py --embedding q.json -k 5 --beam 2 --encoder nomic-embed-text --json

Input
-----
    --embedding PATH   JSON array with the query embedding
                       (or `-` / omitted → read the array from stdin)
    -k, --top-k N      how many chunks to retrieve            (default 5)
    --beam B           search width during tree descent: at every level the
                       best B branches are followed ("save strategy" — B=2
                       keeps the second path when the query sits close to a
                       partition boundary)                     (default 2)
    --encoder NAME     optional guard: error out if NAME does not match the
                       embedding model recorded in index_meta.json
    --json             full JSON output (id, path, score, source, heading)
                       instead of one path per line

Output
------
    default : one chunk file path per line (absolute), best match first
    --json  : {"k": …, "beam": …, "candidates_ranked": …,
               "candidate_rows": [rows the beam actually searched],
               "results": [{"rank", "id", "path", "score",
               "source", "heading_path"}, …]}

Exit codes:  0 ok · 1 bad input · 2 encoder/dimension mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index() -> tuple[dict, dict, np.ndarray, dict]:
    """Load meta, tree, embedding matrix and chunk manifest from this folder."""
    with open(HERE / "index_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(HERE / "ann_tree.json", encoding="utf-8") as fh:
        tree = json.load(fh)
    attach_anchors(tree, HERE / "anchors.npy")
    embeddings = np.load(HERE / "embeddings.npy")
    with open(HERE / "chunks_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return meta, tree, embeddings, manifest


# ── Query embedding input ─────────────────────────────────────────────────────

def read_embedding(source: str | None) -> np.ndarray:
    """Read the query embedding: JSON array from a file or stdin."""
    if source is None or source == "-":
        raw = sys.stdin.read()
    else:
        with open(source, encoding="utf-8") as fh:
            raw = fh.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: query embedding is not valid JSON ({exc})")

    # Accept [..] as well as {"embedding": [..]}
    if isinstance(data, dict) and "embedding" in data:
        data = data["embedding"]
    vec = np.asarray(data, dtype=float)
    if vec.ndim != 1:
        sys.exit(f"Error: expected a flat embedding vector, got shape {vec.shape}")
    return vec


# ── Tree descent (beam search over cluster means) ─────────────────────────────

def cosine(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cosine similarity of q against each row of M (zero-safe)."""
    qn = np.linalg.norm(q)
    Mn = np.linalg.norm(M, axis=1)
    denom = qn * Mn
    denom[denom == 0.0] = 1.0
    return (M @ q) / denom


def _plane(node: dict):
    """`w` einmal in ein Array wandeln und am Knoten behalten."""
    v = node.get("_w")
    if v is None:
        raw = node.get("w")
        v = np.asarray(raw, dtype=float) if raw is not None else False
        node["_w"] = v
    return v


def _kids(node: dict):
    """
    Die beiden Kind-Mittelpunkte als Matrix (+ ihre Normen), gecacht.

    Ohne den Cache baut jeder Abstieg an JEDEM Knoten ein neues Array aus
    Python-Listen — bei d=1024 ist das der teuerste Teil der ganzen Query
    (gemessen 24 ms statt 0.2 ms).  Der Baum kommt als JSON herein, also
    kostet die Umwandlung einmalig, danach ist sie geschenkt.
    """
    M = node.get("_kidsM")
    if M is None:
        M = np.asarray([c["mean"] for c in node["children"]], dtype=float)
        node["_kidsM"] = M
        node["_kidsN"] = np.maximum(np.linalg.norm(M, axis=1), 1e-30)
    return node["_kidsM"], node["_kidsN"]


def _self_vec(node: dict):
    v = node.get("_meanV")
    if v is None:
        v = np.asarray(node["mean"], dtype=float)
        node["_meanV"] = v
        node["_meanN"] = max(float(np.linalg.norm(v)), 1e-30)
    return node["_meanV"], node["_meanN"]


#  Anker liegen in anchors.npy, der Knoten trägt nur [offset, count] —
#  als JSON-Text wären sie ein Vielfaches des ganzen Baums.  attach_anchors()
#  hängt die Matrix einmal an den Baum, danach ist der Zugriff ein Slice.
_ANCHORS: dict = {"M": None}


def attach_anchors(tree: dict, path) -> None:
    """anchors.npy laden, falls vorhanden — sonst bleibt alles beim Alten."""
    import pathlib
    f = pathlib.Path(path)
    if f.exists():
        A = np.load(f).astype(float)
        _ANCHORS["M"] = A / np.maximum(
            np.linalg.norm(A, axis=1, keepdims=True), 1e-30)


def _anchors(node: dict):
    """
    Die Anker eines Knotens als (k, d)-Matrix, zeilenweise normiert — oder
    None, wenn der Baum ohne `--anchors` gebaut wurde.
    """
    a = node.get("anchors")
    if a is None or _ANCHORS["M"] is None:
        return None
    off, cnt = a
    return _ANCHORS["M"][off:off + cnt]


def node_score(node: dict, q: np.ndarray, qn: float) -> float:
    """
    Wie gut passt dieser Teilbaum zur Query?  Das ist die Frage, die der
    BEAM stellt (nicht: auf welcher Seite liegt die Query — das entscheidet
    die Ebene in child_scores).

    Mit Ankern ist die Antwort `max_j cos(q, a_j)`, also eine Annäherung an
    `max_p cos(q, p)` über die Punkte des Teilbaums.  Ohne Anker bleibt es
    `cos(q, mean)`: ein einziger Mittelwert für unter Umständen tausende
    Punkte — gemessen zeigen die beiden Wurzelkinder eines 4178er-Baums
    eine Zentroid-Cosine von 0.905, sind für den Beam also kaum
    unterscheidbar.
    """
    A = _anchors(node)
    if A is not None:
        return float((A @ q).max()) / qn
    v, vn = _self_vec(node)
    return float(v @ q) / (qn * vn)


def child_scores(node: dict, q: np.ndarray) -> list[float]:
    """
    Score both children of *node* for the query.

    Trees built with a max-margin separating plane carry `w`, `tau` and
    `sigma` per node: the split is the hyperplane `p·w = tau`, so the
    signed distance `q·w - tau` says both WHICH side the query is on and
    HOW confidently.  Dividing by `sigma` (the spread of the projections
    in that node) makes the score comparable across nodes, which the beam
    needs — it ranks children of different parents against each other.

    Older trees have no plane; there we fall back to cosine against the
    child means, which is the same decision without the offset.
    """
    w = _plane(node)
    if w is False:
        M, Mn = _kids(node)
        return list((M @ q) / (max(float(np.linalg.norm(q)), 1e-30) * Mn))
    s = (float(w @ q) - node["tau"]) / max(node.get("sigma") or 0.0, 1e-12)
    return [s, -s]                      # child 0 = side t > tau


def greedy_child(node: dict, q: np.ndarray) -> dict:
    """
    The one child the query belongs to.  With a max-margin plane this is
    decided by the side of `q·w = tau`; that decision is exact in the sense
    that every point of the node lies on the side its own centroid is on,
    so a query within the node's margin cannot be sent the wrong way.
    Without a plane it falls back to the nearer centroid.
    """
    sc = child_scores(node, q)
    return node["children"][0 if sc[0] >= sc[1] else 1]


def descend(tree: dict, q: np.ndarray, beam: int) -> list[dict]:
    """
    Beam-search the partition tree, keeping the *beam* best branches per
    level.  Returns the reached leaves.

    The greedy path is PINNED: one beam slot always follows the exact
    per-node decision (plane side, or nearer centroid), the remaining
    slots are filled by cosine against the child means.  Without pinning,
    both slots can fall into the same subtree and evict the very branch
    the query belongs to — measured, pinning lifts chunk retrieval from
    98.2 % to 100 % at a cost of 0.4 points of recall@5.

    The two scores answer different questions, which is why both are
    needed: the plane says which side of THIS node the query is on, the
    cosine says how well a node matches the query at all — and only the
    latter is comparable across different parents, as the beam requires.
    """
    frontier = [tree]
    greedy = tree
    qn = max(float(np.linalg.norm(q)), 1e-30)
    while any(node["children"] is not None for node in frontier):
        if greedy["children"] is not None:
            greedy = greedy_child(greedy, q)

        candidates: list[dict] = []
        scores: list[float] = []
        for node in frontier:
            if node["children"] is None:
                candidates.append(node)                  # leaf rides along
                scores.append(node_score(node, q, qn))
            elif "anchors" in node["children"][0]:
                # Anker-Baum: jedes Kind wird einzeln bewertet
                for c in node["children"]:
                    candidates.append(c)
                    scores.append(node_score(c, q, qn))
            else:
                M, Mn = _kids(node)                      # gecachter Schnellpfad
                candidates.extend(node["children"])
                scores.extend(((M @ q) / (qn * Mn)).tolist())

        pick = [i for i, nd in enumerate(candidates) if nd is greedy][:1]
        for i in np.argsort(-np.asarray(scores)):
            if len(pick) >= beam:
                break
            if int(i) not in pick:
                pick.append(int(i))
        frontier = [candidates[i] for i in pick]
    return frontier


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retrieve the semantically nearest chunks for a query embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1],
    )
    ap.add_argument("--embedding", "-e", default=None,
                    help="JSON file with the query embedding ('-'/omitted = stdin)")
    ap.add_argument("--top-k", "-k", type=int, default=5,
                    help="number of chunks to retrieve (default 5)")
    ap.add_argument("--beam", "-b", type=int, default=8,
                    help="beam width during tree descent (default 8). "
                         "Gemessen auf 4178 Chunks: beam 2 = 58 %% recall@5, "
                         "beam 8 = 87 %%, beam 16 = 92 %% — bei 0.06 bis "
                         "0.13 ms je Query. Der Beam ist der mit Abstand "
                         "billigste Qualitätshebel, weil er weder Speicher "
                         "noch Neubau kostet.")
    ap.add_argument("--min-candidates", type=int, default=None,
                    help="widen the beam until at least this many candidates "
                         "reach the exact ranking (default: 3·k; recall knob)")
    ap.add_argument("--encoder", default=None,
                    help="assert that this encoder name matches the index")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="full JSON output instead of one path per line")
    args = ap.parse_args()

    meta, tree, embeddings, manifest = load_index()

    # ── Encoder / dimension guard ────────────────────────────────────────────
    if args.encoder is not None and args.encoder != meta["encoder"]["model"]:
        print(
            f"Error: encoder mismatch — index was built with "
            f"'{meta['encoder']['model']}', query claims '{args.encoder}'. "
            f"Same encoder required.",
            file=sys.stderr,
        )
        sys.exit(2)

    q = read_embedding(args.embedding)
    dim = meta["encoder"]["dim"]
    if q.shape[0] != dim:
        print(
            f"Error: embedding dimension mismatch — index has {dim} dims "
            f"({meta['encoder']['model']}), query has {q.shape[0]}. "
            f"Was the same encoder used?",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── Descent → candidate rows → exact ranking ────────────────────────────
    # Save strategy: descend with the requested beam; if too few candidates
    # reach the exact ranking, widen the beam and descend again (a descent
    # costs a handful of scalar products, so re-running is free).
    min_cand = args.min_candidates if args.min_candidates is not None else 3 * args.top_k
    min_cand = min(max(min_cand, args.top_k), embeddings.shape[0])

    beam = max(1, args.beam)
    while True:
        leaves = descend(tree, q, beam)
        candidate_rows = sorted({row for leaf in leaves for row in leaf["rows"]})
        if len(candidate_rows) >= min_cand or len(candidate_rows) == embeddings.shape[0]:
            break
        beam += 1

    sims = cosine(q, embeddings[candidate_rows].astype(float))
    order = np.argsort(-sims)[: args.top_k]

    chunk_ids = manifest["row_to_id"]
    results = []
    for rank, ci in enumerate(order, start=1):
        row = candidate_rows[int(ci)]
        cid = chunk_ids[row]
        info = manifest["chunks"][cid]
        results.append({
            "rank": rank,
            "id": cid,
            "path": str((HERE / info["path"]).resolve()),
            "score": round(float(sims[int(ci)]), 6),
            "source": info["source"],
            "heading_path": info.get("heading_path"),
        })

    if args.as_json:
        print(json.dumps({"k": args.top_k, "beam": beam,
                          "index_chunks": int(embeddings.shape[0]),
                          "encoder": meta["encoder"]["model"],
                          "candidates_ranked": len(candidate_rows),
                          "candidate_rows": candidate_rows,
                          "results": results}, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(r["path"])


if __name__ == "__main__":
    main()
