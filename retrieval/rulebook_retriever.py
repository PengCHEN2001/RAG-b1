"""Hybrid retrieval over the processed Core Rules rulebook.

This module deliberately does not rewrite user questions or choose routes.  A
future Router supplies a :class:`RulebookQuerySet`; this retriever executes the
fixed dense, BM25, RRF, and reranking pipeline for that input.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from ingestion.build_index import DEFAULT_COLLECTION, DEFAULT_MODEL, load_chunks, tokenize_for_bm25


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_INDEX_DIR = Path("data/indexes/core_rules")
DEFAULT_CHUNKS_PATH = Path("data/processed/Core Rules.chunks.jsonl")
DEFAULT_RETRIEVAL_K = 12
DEFAULT_FINAL_K = 5
RRF_CONSTANT = 60


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


class Reranker(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RulebookQuerySet:
    """Queries planned by the Router for one rulebook retrieval request."""

    original_query: str
    semantic_query: str | None = None
    keyword_query: tuple[str, ...] | list[str] | None = None

    def normalised(self) -> "RulebookQuerySet":
        """Provide safe fallbacks if a future Router returns partial output."""
        original = self.original_query.strip()
        if not original:
            raise ValueError("original_query must not be empty.")
        semantic = (self.semantic_query or "").strip() or original
        keywords = tuple(item.strip() for item in (self.keyword_query or ()) if item.strip())
        return RulebookQuerySet(
            original_query=original,
            semantic_query=semantic,
            keyword_query=keywords or (original,),
        )

    @property
    def keyword_text(self) -> str:
        return " ".join(self.normalised().keyword_query or ())


@dataclass(frozen=True)
class RankedCandidate:
    chunk_id: str
    rank: int
    score: float | None = None


@dataclass
class FusedCandidate:
    chunk_id: str
    dense_rank: int | None = None
    bm25_original_rank: int | None = None
    bm25_keyword_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def best_rank(self) -> int:
        return min(rank for rank in (
            self.dense_rank, self.bm25_original_rank, self.bm25_keyword_rank,
        ) if rank is not None)


@dataclass(frozen=True)
class RetrievedChunk:
    """A source chunk enriched with retrieval scores and provenance."""

    chunk: dict[str, Any]
    dense_rank: int | None
    bm25_original_rank: int | None
    bm25_keyword_rank: int | None
    rrf_score: float
    reranker_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.chunk,
            "dense_rank": self.dense_rank,
            "bm25_original_rank": self.bm25_original_rank,
            "bm25_keyword_rank": self.bm25_keyword_rank,
            "rrf_score": self.rrf_score,
            "reranker_score": self.reranker_score,
        }


@dataclass(frozen=True)
class RetrievedRulebookContext:
    """Evidence returned to the later answer-generation layer."""

    query_set: RulebookQuerySet
    results: tuple[RetrievedChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.query_set.original_query,
            "semantic_query": self.query_set.semantic_query,
            "keyword_query": list(self.query_set.keyword_query or ()),
            "results": [result.to_dict() for result in self.results],
        }


def _as_ranked_candidates(ids: Iterable[str]) -> list[RankedCandidate]:
    """Assign one-based positions, skipping accidental duplicates."""
    seen: set[str] = set()
    result: list[RankedCandidate] = []
    for identifier in ids:
        if identifier not in seen:
            seen.add(identifier)
            result.append(RankedCandidate(identifier, len(result) + 1))
    return result


def reciprocal_rank_fusion(
    dense: Iterable[RankedCandidate],
    bm25_original: Iterable[RankedCandidate],
    bm25_keyword: Iterable[RankedCandidate],
    *,
    constant: int = RRF_CONSTANT,
) -> list[FusedCandidate]:
    """Fuse ranking positions without mixing incomparable raw scores."""
    if constant < 0:
        raise ValueError("RRF constant must be non-negative.")
    fused: dict[str, FusedCandidate] = {}
    route_attributes = (
        (dense, "dense_rank"),
        (bm25_original, "bm25_original_rank"),
        (bm25_keyword, "bm25_keyword_rank"),
    )
    for ranking, attribute in route_attributes:
        for candidate in ranking:
            if candidate.rank < 1:
                raise ValueError("Ranking positions must start at one.")
            record = fused.setdefault(candidate.chunk_id, FusedCandidate(candidate.chunk_id))
            if getattr(record, attribute) is None:
                setattr(record, attribute, candidate.rank)
                record.rrf_score += 1 / (constant + candidate.rank)
    return sorted(fused.values(), key=lambda item: (-item.rrf_score, item.best_rank, item.chunk_id))


class RulebookRetriever:
    """Retrieve rulebook evidence through Dense + BM25 + RRF + reranking."""

    def __init__(
        self,
        *,
        collection: Any,
        bm25: Any,
        bm25_chunk_ids: Sequence[str],
        chunk_records: dict[str, dict[str, Any]],
        embedding_model: Embedder | None = None,
        reranker: Reranker | None = None,
        embedding_model_name: str = DEFAULT_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        if len(bm25_chunk_ids) != len(set(bm25_chunk_ids)):
            raise ValueError("BM25 chunk IDs must be unique.")
        missing = set(bm25_chunk_ids) - set(chunk_records)
        if missing:
            raise ValueError("BM25 chunk IDs are missing from the chunk records.")
        self.collection = collection
        self.bm25 = bm25
        self.bm25_chunk_ids = list(bm25_chunk_ids)
        self.chunk_records = chunk_records
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

    @classmethod
    def from_paths(
        cls,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model_name: str = DEFAULT_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> "RulebookRetriever":
        """Load persistent indexes and metadata without loading ML models yet."""
        try:
            import chromadb
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("Install chromadb before using RulebookRetriever.") from error
        if not (index_dir / "bm25.pkl").is_file():
            raise FileNotFoundError(f"BM25 index not found in: {index_dir}")
        with (index_dir / "bm25.pkl").open("rb") as file:
            bm25 = pickle.load(file)
        bm25_chunk_ids = json.loads((index_dir / "bm25_chunk_ids.json").read_text(encoding="utf-8"))
        chunks = load_chunks(chunks_path)
        records = {chunk["chunk_id"]: chunk for chunk in chunks}
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        collection = client.get_collection(collection_name, embedding_function=None)
        return cls(
            collection=collection,
            bm25=bm25,
            bm25_chunk_ids=bm25_chunk_ids,
            chunk_records=records,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
        )

    def _get_embedding_model(self) -> Embedder:
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - runtime dependency
                raise RuntimeError("Install sentence-transformers before retrieval.") from error
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model

    def _get_reranker(self) -> Reranker:
        if self.reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:  # pragma: no cover - runtime dependency
                raise RuntimeError("Install sentence-transformers before reranking.") from error
            self.reranker = CrossEncoder(
                self.reranker_model_name, max_length=DEFAULT_RERANKER_MAX_LENGTH,
            )
        return self.reranker

    def _dense_search(self, query: str, limit: int) -> list[RankedCandidate]:
        embedding = self._get_embedding_model().encode(
            [BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True,
        )[0]
        response = self.collection.query(
            query_embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding],
            n_results=limit,
            include=[],
        )
        return _as_ranked_candidates(response["ids"][0])

    def _bm25_search(self, query: str, limit: int) -> list[RankedCandidate]:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda item: (-float(item[1]), self.bm25_chunk_ids[item[0]]),
        )
        return [
            RankedCandidate(self.bm25_chunk_ids[index], rank + 1, float(score))
            for rank, (index, score) in enumerate(ranked)
            if score > 0
        ][:limit]

    def retrieve(
        self,
        query_set: RulebookQuerySet,
        *,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        final_k: int = DEFAULT_FINAL_K,
    ) -> RetrievedRulebookContext:
        """Run three retrieval routes, fuse them, rerank, and return evidence."""
        if retrieval_k < 1 or final_k < 1:
            raise ValueError("retrieval_k and final_k must be positive.")
        planned = query_set.normalised()
        dense = self._dense_search(planned.semantic_query or planned.original_query, retrieval_k)
        bm25_original = self._bm25_search(planned.original_query, retrieval_k)
        bm25_keyword = self._bm25_search(planned.keyword_text, retrieval_k)
        fused = reciprocal_rank_fusion(dense, bm25_original, bm25_keyword)[:retrieval_k]
        if not fused:
            return RetrievedRulebookContext(planned, ())
        pairs = [
            (planned.semantic_query or planned.original_query, self.chunk_records[item.chunk_id]["text"])
            for item in fused
        ]
        scores = self._get_reranker().predict(pairs, show_progress_bar=False)
        reranked = sorted(
            zip(fused, scores, strict=True),
            key=lambda item: (-float(item[1]), -item[0].rrf_score, item[0].chunk_id),
        )[:final_k]
        results = tuple(
            RetrievedChunk(
                chunk=self.chunk_records[candidate.chunk_id],
                dense_rank=candidate.dense_rank,
                bm25_original_rank=candidate.bm25_original_rank,
                bm25_keyword_rank=candidate.bm25_keyword_rank,
                rrf_score=candidate.rrf_score,
                reranker_score=float(score),
            )
            for candidate, score in reranked
        )
        return RetrievedRulebookContext(planned, results)


def main() -> None:
    """Run one local retrieval request before the Router is implemented."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Original user question")
    parser.add_argument("--semantic-query", help="Router-produced English semantic query")
    parser.add_argument(
        "--keyword-query", nargs="*", default=None,
        help="Router-produced rulebook keywords or phrases",
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()
    retriever = RulebookRetriever.from_paths(
        index_dir=args.index_dir, chunks_path=args.chunks, collection_name=args.collection,
    )
    result = retriever.retrieve(RulebookQuerySet(
        original_query=args.query,
        semantic_query=args.semantic_query,
        keyword_query=args.keyword_query,
    ))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
