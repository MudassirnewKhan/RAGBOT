"""
BM25 Keyword Search Index.
Implements Okapi BM25 for sparse keyword-based lexical retrieval.
"""
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Basic alphanumeric lowercased tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 lexical index to complement vector embeddings."""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.corpus: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def build(self, texts: List[str], metadata_list: List[Dict[str, Any]]):
        """Build BM25 index from text corpus and corresponding metadata."""
        if not texts:
            self.bm25 = None
            self.corpus = []
            self.metadata = []
            return

        self.corpus = texts
        self.metadata = metadata_list

        tokenized_corpus = [simple_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[INFO] Built BM25 index over {len(texts)} document chunks.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search BM25 index with a natural language query string."""
        if self.bm25 is None or not self.corpus:
            return []

        tokens = simple_tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[-top_k:][::-1]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            entry = self.metadata[idx]
            results.append({
                "score": score,
                "text": self.corpus[idx],
                "metadata": {k: v for k, v in entry.items() if k != "text"},
            })

        return results
