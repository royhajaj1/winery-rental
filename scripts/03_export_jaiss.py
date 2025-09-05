# scripts/export_from_faiss.py
import argparse, json, sys
from pathlib import Path

# faiss import works for both faiss and faiss-cpu wheel names
try:
    import faiss
except Exception as e:
    print("ERROR: faiss not installed. Try: pip install faiss-cpu", file=sys.stderr)
    raise

import numpy as np


def read_meta_as_chunks(meta_path: Path, out_path: Path, id_field="id", text_field="text"):
    n_lines = 0
    with meta_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            obj = json.loads(line)
            # Resolve id
            _id = obj.get(id_field, i)
            # Resolve text (common alternates)
            _text = obj.get(text_field) or obj.get("page_content") or obj.get("content") or obj.get("chunk") or obj.get("text")
            if _text is None:
                _text = ""  # keep line, but empty text

            # Carry a few helpful fields if present
            chunk = {
                "id": int(_id),
                "text": _text,
            }
            for k in ("source", "title", "doc_id", "page", "metadata"):
                if k in obj:
                    chunk[k] = obj[k]

            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            n_lines += 1
    return n_lines


def export_embeddings(index_path: Path, ids_out: Path, emb_out: Path):
    index = faiss.read_index(str(index_path))
    N, D = index.ntotal, index.d
    if N == 0:
        raise RuntimeError("Index is empty (ntotal=0).")

    # Try the fastest reconstruction first
    vecs = None
    try:
        # reconstruct_n returns a (N, D) float32 array for many index types
        vecs = index.reconstruct_n(0, N)
    except Exception:
        # Fallback: per-vector reconstruction
        vecs = np.vstack([index.reconstruct(i) for i in range(N)]).astype("float32")

    # IDs: default to 0..N-1; try to pull mapped IDs if available
    ids = np.arange(N, dtype="int64")
    try:
        # Many IndexIDMap/IndexIDMap2 expose id_map; vector_to_array works when present
        if hasattr(index, "id_map"):
            maybe_ids = faiss.vector_to_array(index.id_map)
            if maybe_ids.size == N:
                ids = maybe_ids.astype("int64", copy=False)
    except Exception:
        pass

    np.save(emb_out, vecs.astype("float32", copy=False))
    np.save(ids_out, ids)
    return N, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to index.faiss")
    ap.add_argument("--meta", required=True, help="Path to meta.jsonl (line-delimited JSON)")
    ap.add_argument("--outdir", required=True, help="Output directory (will be created if missing)")
    ap.add_argument("--id-field", default="id", help="Field name in meta for the chunk id (default: id)")
    ap.add_argument("--text-field", default="text", help="Field name in meta for the chunk text (default: text)")
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    meta_path = Path(args.meta).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    emb_out = outdir / "embeddings.npy"
    ids_out = outdir / "ids.npy"
    chunks_out = outdir / "chunks.jsonl"

    print(f"[1/3] Exporting embeddings from: {index_path}")
    N, D = export_embeddings(index_path, ids_out, emb_out)
    print(f"    -> embeddings.npy shape: [{N}, {D}]")
    print(f"    -> ids.npy shape: [{N}]")

    print(f"[2/3] Normalizing meta -> chunks.jsonl from: {meta_path}")
    n_lines = read_meta_as_chunks(meta_path, chunks_out, id_field=args.id_field, text_field=args.text_field)
    print(f"    -> chunks.jsonl lines: {n_lines}")

    if n_lines != N:
        print(f"WARNING: meta lines ({n_lines}) != index vectors ({N}). "
              f"This can still work if ids match, but double-check.", file=sys.stderr)

    print(f"[3/3] Done. Files:\n  {emb_out}\n  {ids_out}\n  {chunks_out}")


if __name__ == "__main__":
    main()
