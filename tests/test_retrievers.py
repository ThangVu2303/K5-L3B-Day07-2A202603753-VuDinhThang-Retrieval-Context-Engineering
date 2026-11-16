import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrievers import BM25Retriever, HybridRRF, match_filters, tokenize  # noqa: E402


def chunk(i, text, district):
    return {"id": f"c{i}", "text": text, "source_id": "doc", "source": "doc.pdf", "page": i,
            "metadata": {"district": district}}


CHUNKS = [
    chunk(1, "Diesel prices rose sharply for trucking firms", "Dallas"),
    chunk(2, "Wage growth remained moderate across retail", "Boston"),
    chunk(3, "Diesel shortages hit rural Colorado fuel buyers", "Kansas City"),
    chunk(4, "Data center construction drove industrial demand", "New York"),
]


class FakeRetriever:
    def __init__(self, order):
        self.order = order

    def search(self, query, *, filters=None, top_k=5):
        hits = [c for c in (CHUNKS[i] for i in self.order) if match_filters(c, filters)]
        return [{**c, "score": 1.0, "rank": r} for r, c in enumerate(hits[:top_k], 1)]


class RetrieverTests(unittest.TestCase):
    def test_tokenize_drops_stopwords(self):
        self.assertEqual(tokenize("What are the Diesel prices?"), ["diesel", "prices"])

    def test_bm25_ranks_lexical_match_first(self):
        hits = BM25Retriever(CHUNKS).search("diesel shortages Colorado", top_k=2)
        self.assertEqual(hits[0]["id"], "c3")
        self.assertEqual([h["rank"] for h in hits], [1, 2])

    def test_bm25_respects_filter(self):
        hits = BM25Retriever(CHUNKS).search("diesel", filters={"district": "Dallas"})
        self.assertEqual([h["id"] for h in hits], ["c1"])

    def test_bm25_rejects_zero_score(self):
        self.assertEqual(BM25Retriever(CHUNKS).search("zebra"), [])

    def test_rrf_rewards_agreement(self):
        # c2 đứng thứ 2 ở cả hai danh sách -> vượt c1 và c4 (mỗi cái chỉ đứng đầu một danh sách)
        hybrid = HybridRRF([FakeRetriever([0, 1, 2]), FakeRetriever([3, 1, 2])], candidates=3, k=60)
        self.assertEqual(hybrid.search("q", top_k=1)[0]["id"], "c2")

    def test_rrf_passes_filter_to_children(self):
        hybrid = HybridRRF([FakeRetriever([0, 1, 2, 3])], candidates=4)
        hits = hybrid.search("q", filters={"district": "Boston"})
        self.assertEqual([h["id"] for h in hits], ["c2"])


if __name__ == "__main__":
    unittest.main()
