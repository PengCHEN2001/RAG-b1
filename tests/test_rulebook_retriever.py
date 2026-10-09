"""Unit tests for hybrid rulebook retrieval without model downloads."""

import unittest

from retrieval.rulebook_retriever import (
    RankedCandidate,
    RulebookQuerySet,
    RulebookRetriever,
    reciprocal_rank_fusion,
)


class FakeEmbedding:
    def encode(self, sentences, **_kwargs):
        self.sentences = sentences
        return [[0.1, 0.2, 0.3]]


class FakeCollection:
    def query(self, **_kwargs):
        return {"ids": [["chunk-2", "chunk-1"]]}


class FakeBM25:
    def get_scores(self, tokens):
        if "charge" in tokens:
            return [3.0, 2.0, 0.0]
        return [0.0, 1.0, 4.0]


class FakeReranker:
    def predict(self, pairs, **_kwargs):
        self.pairs = pairs
        return [0.9, 0.2, 0.1][:len(pairs)]


def records():
    return {
        f"chunk-{number}": {
            "chunk_id": f"chunk-{number}",
            "text": f"Rule text {number}",
            "section_path": ["Core Rules"],
            "pdf_page_numbers": [number],
        }
        for number in range(1, 4)
    }


class RulebookRetrieverTests(unittest.TestCase):
    def test_query_set_falls_back_to_original_query(self):
        query_set = RulebookQuerySet("Can I charge?", "", []).normalised()
        self.assertEqual(query_set.semantic_query, "Can I charge?")
        self.assertEqual(query_set.keyword_query, ("Can I charge?",))

    def test_rrf_deduplicates_and_rewards_multiple_routes(self):
        results = reciprocal_rank_fusion(
            [RankedCandidate("chunk-1", 2), RankedCandidate("chunk-2", 1)],
            [RankedCandidate("chunk-1", 1)],
            [RankedCandidate("chunk-3", 1)],
        )
        self.assertEqual([result.chunk_id for result in results], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(results[0].dense_rank, 2)
        self.assertEqual(results[0].bm25_original_rank, 1)

    def test_retrieves_fuses_reranks_and_returns_metadata(self):
        embedder = FakeEmbedding()
        reranker = FakeReranker()
        retriever = RulebookRetriever(
            collection=FakeCollection(),
            bm25=FakeBM25(),
            bm25_chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
            chunk_records=records(),
            embedding_model=embedder,
            reranker=reranker,
        )
        result = retriever.retrieve(
            RulebookQuerySet(
                original_query="Can I charge?",
                semantic_query="Can a unit declare a charge?",
                keyword_query=["charge", "Charge phase"],
            ),
            retrieval_k=3,
            final_k=2,
        )

        self.assertEqual(embedder.sentences, [
            "Represent this sentence for searching relevant passages: "
            "Can a unit declare a charge?"
        ])
        self.assertEqual(len(reranker.pairs), 2)
        self.assertEqual([chunk.chunk["chunk_id"] for chunk in result.results], ["chunk-1", "chunk-2"])
        self.assertEqual(result.results[0].chunk["pdf_page_numbers"], [1])
        self.assertIsNotNone(result.results[0].rrf_score)
        self.assertEqual(result.results[0].reranker_score, 0.9)


if __name__ == "__main__":
    unittest.main()
