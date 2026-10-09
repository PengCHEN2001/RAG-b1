"""Build persistent dense (Chroma) and BM25 indexes from chunk JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_COLLECTION = "core_rules_bge_base_en_v1_5"


def tokenize_for_bm25(text: str) -> list[str]:
    """Use one predictable tokenization rule for BM25 indexing and querying."""
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate canonical chunk records."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError(f"Chunk file is empty: {path}")
    ids = [chunk.get("chunk_id") for chunk in chunks]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("Each chunk must have a unique chunk_id.")
    if any(not chunk.get("text", "").strip() for chunk in chunks):
        raise ValueError("Every chunk must contain embedding text.")
    return chunks


def content_hash(chunk: dict[str, Any]) -> str:
    payload = json.dumps({
        "text": chunk["text"], "section_path": chunk.get("section_path", []),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(path: Path, model: str) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    cache: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("embedding_model") == model:
            cache[record["content_hash"]] = record["embedding"]
    return cache


def _write_cache(path: Path, model: str, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for digest, embedding in cache.items():
            file.write(json.dumps({
                "content_hash": digest,
                "embedding_model": model,
                "embedding": embedding,
            }) + "\n")


def build_embeddings(
    chunks: list[dict[str, Any]], *, model_name: str, batch_size: int,
    cache_path: Path,
) -> tuple[list[list[float]], int]:
    """Encode only new content hashes; return vectors in chunk-file order."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - depends on runtime install
        raise RuntimeError("Install sentence-transformers before building the index.") from error
    cache = _load_cache(cache_path, model_name)
    missing = [chunk for chunk in chunks if content_hash(chunk) not in cache]
    if missing:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            [chunk["text"] for chunk in missing], batch_size=batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        )
        for chunk, embedding in zip(missing, embeddings, strict=True):
            cache[content_hash(chunk)] = embedding.tolist()
        _write_cache(cache_path, model_name, cache)
    vectors = [cache[content_hash(chunk)] for chunk in chunks]
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embedding cache contains inconsistent vector dimensions.")
    return vectors, dimension


def chroma_metadata(chunk: dict[str, Any], *, model_name: str, dimension: int) -> dict[str, Any]:
    """Return scalar metadata supported by every Chroma deployment.

    Full provenance stays in JSONL; Chroma only needs filter and display fields.
    """
    pages = chunk.get("pdf_page_numbers", [])
    return {
        "logical_unit_id": chunk["logical_unit_id"],
        "part_index": int(chunk["part_index"]),
        "part_count": int(chunk["part_count"]),
        "content_type": chunk["content_type"],
        "section_path": " > ".join(chunk.get("section_path", [])),
        "source": chunk["source"],
        "page_numbers": json.dumps(pages),
        "page_start": int(min(pages)) if pages else -1,
        "page_end": int(max(pages)) if pages else -1,
        "token_count": int(chunk["token_count"]),
        "content_hash": content_hash(chunk),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
    }


def write_chroma(
    chunks: list[dict[str, Any]], vectors: list[list[float]], *,
    output_dir: Path, collection_name: str, model_name: str, dimension: int,
    recreate: bool,
) -> None:
    try:
        import chromadb
    except ImportError as error:  # pragma: no cover - depends on runtime install
        raise RuntimeError("Install chromadb before building the index.") from error
    client = chromadb.PersistentClient(path=str(output_dir / "chroma"))
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:  # Collection absent is safe on a first build.
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "embedding_dimension": dimension,
        },
        embedding_function=None,
    )
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        stop = start + batch_size
        batch = chunks[start:stop]
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=vectors[start:stop],
            metadatas=[chroma_metadata(chunk, model_name=model_name, dimension=dimension)
                       for chunk in batch],
        )
    if collection.count() != len(chunks):
        raise RuntimeError("Chroma collection count does not match the chunk count.")


def write_bm25(chunks: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as error:  # pragma: no cover - depends on runtime install
        raise RuntimeError("Install rank-bm25 before building the index.") from error
    tokens = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
    if any(not document for document in tokens):
        raise ValueError("BM25 tokenization produced an empty chunk.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bm25.pkl").open("wb") as file:
        pickle.dump(BM25Okapi(tokens), file)
    (output_dir / "bm25_chunk_ids.json").write_text(
        json.dumps([chunk["chunk_id"] for chunk in chunks], indent=2), encoding="utf-8",
    )


def write_manifest(
    output_dir: Path, *, chunks_path: Path, chunks: list[dict[str, Any]],
    model_name: str, dimension: int, collection_name: str,
) -> None:
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "chunk_file": str(chunks_path),
        "chunk_file_hash": chunk_file_hash(chunks_path),
        "chunk_count": len(chunks),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "chroma_collection": collection_name,
        "vector_distance_space": "cosine",
        "bm25_tokenizer_version": "regex-v1",
    }
    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path, help="Chunk JSONL created by ingestion.chunker")
    parser.add_argument("--output-dir", type=Path, default=Path("data/indexes/core_rules"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true", help="Replace the named Chroma collection")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vectors, dimension = build_embeddings(
        chunks, model_name=args.model, batch_size=args.batch_size,
        cache_path=args.output_dir / "embedding_cache.jsonl",
    )
    write_chroma(
        chunks, vectors, output_dir=args.output_dir, collection_name=args.collection,
        model_name=args.model, dimension=dimension, recreate=args.recreate,
    )
    write_bm25(chunks, args.output_dir)
    write_manifest(
        args.output_dir, chunks_path=args.chunks, chunks=chunks,
        model_name=args.model, dimension=dimension, collection_name=args.collection,
    )
    print(f"Indexed {len(chunks)} chunks: Chroma collection '{args.collection}' and BM25.")


if __name__ == "__main__":
    main()
