"""Task 764 - Ranking functions: TF-IDF + BM25 scoring and evaluation.  (Tier: Expert, Beyond - Machine Learning & AI)

Two classic bag-of-words rankers over the same inverted statistics:

    TF-IDF   tf-idf(t, d) = (1 + log tf) * log(N / df), summed over the query terms - the sublinear
             (logarithmic) term frequency keeps a word that appears ten times from scoring ten times
             as much, and the inverse document frequency discounts words that are everywhere.
    BM25     idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d)/avg_len)), with the
             probabilistic idf log(1 + (N - df + 0.5)/(df + 0.5)) that stays positive even for terms
             appearing in most documents.  k1 controls the saturation of the term frequency and b
             the strength of the document-length normalisation.

The properties that make BM25 what it is are exactly what the checks probe: additivity over query
terms, monotonicity in the term frequency, saturation (k1), length normalisation that disappears at
b = 0 and penalises long documents at b = 1, and a zero contribution from terms absent from the
document.  BM25 is cross-checked against an independently written transcription of the formula, and
the document scoring is validated on hand-built document pairs instead of on the implementation's own
output.

Evaluation covers the standard offline measures: precision@k, recall@k, average precision, mean
reciprocal rank and graded nDCG@k - the last one verified against the brute-force double sum
sum((2^grade - 1) / log2(position + 1)) divided by the ideal ranking's value, plus the sanity
properties (nDCG = 1 for the ideal order, lower for the reversed one, monotone in k, capped at k).

Simplified versus a production ranker: one in-memory corpus of eight tiny documents, no stemming or
stopword list, no field boosts, no document-length priors beyond b, no learning of k1/b (the defaults
are 1.2/0.75), and the relevance judgements are hand-written instead of pooled.  Verified: the exact
scoring formulas, every ranking property above, the metric values against hand-computed cases and a
brute-force nDCG, and the boundary cases (empty query, unknown term, k beyond the result set).
"""
import math

DOCUMENTS = {
    0: "the quick brown fox jumps over the lazy dog",
    1: "quick brown foxes leap over lazy dogs",
    2: "the fox and the dog are friends",
    3: "brown bears and black dogs",
    4: "jumps jumps jumps over the fence",
    5: "",
    6: "a quick quick brown fox",
    7: "dog dog dog lazy dog",
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def tokenize(text):
    return [word for word in text.lower().split() if word]


def build_statistics(documents):
    """Returns per-document term frequencies, document frequencies and lengths."""
    frequencies = {}
    for doc_id, text in documents.items():
        counts = {}
        for word in tokenize(text):
            counts[word] = counts.get(word, 0) + 1
        frequencies[doc_id] = counts
    document_frequency = {}
    for counts in frequencies.values():
        for word in counts:
            document_frequency[word] = document_frequency.get(word, 0) + 1
    lengths = {doc_id: sum(counts.values()) for doc_id, counts in frequencies.items()}
    return frequencies, document_frequency, lengths


class Ranker:
    """TF-IDF and BM25 over one corpus, with the evaluation measures attached."""

    def __init__(self, documents):
        self.documents = dict(documents)
        self.frequencies, self.document_frequency, self.lengths = build_statistics(documents)
        self.total = len(documents)
        self.average_length = (sum(self.lengths.values()) / float(self.total)) if self.total else 0.0

    # -------------------------------------------------------------------------- scoring
    def idf(self, term):
        documents_with_term = self.document_frequency.get(term, 0)
        return math.log(self.total / documents_with_term) if documents_with_term else 0.0

    def bm25_idf(self, term):
        documents_with_term = self.document_frequency.get(term, 0)
        return math.log(1 + (self.total - documents_with_term + 0.5) / (documents_with_term + 0.5))

    def tf_idf_score(self, term, doc_id):
        count = self.frequencies[doc_id].get(term, 0)
        return 0.0 if not count else (1 + math.log(count)) * self.idf(term)

    def bm25_term(self, term, doc_id, k1=1.2, b=0.75):
        count = self.frequencies[doc_id].get(term, 0)
        if not count:
            return 0.0
        length = self.lengths[doc_id]
        normalisation = 1 - b + b * (length / self.average_length if self.average_length else 1.0)
        return self.bm25_idf(term) * (count * (k1 + 1)) / (count + k1 * normalisation)

    def score(self, query, doc_id, model="bm25", k1=1.2, b=0.75):
        terms = tokenize(query) if isinstance(query, str) else list(query)
        if model == "bm25":
            return sum(self.bm25_term(term, doc_id, k1, b) for term in terms)
        return sum(self.tf_idf_score(term, doc_id) for term in terms)

    def rank(self, query, limit=None, model="bm25", k1=1.2, b=0.75):
        terms = tokenize(query) if isinstance(query, str) else list(query)
        if not terms:
            return []
        scored = [(doc_id, self.score(terms, doc_id, model, k1, b)) for doc_id in sorted(self.documents)]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit] if limit is not None else scored


# ------------------------------------------------------------------------------ evaluation

def precision_at_k(ranking, relevant, k):
    if k <= 0:
        return 0.0
    window = ranking[:k]
    return sum(1 for doc in window if doc in relevant) / float(len(window)) if window else 0.0


def recall_at_k(ranking, relevant, k):
    if not relevant:
        return 0.0
    window = ranking[:k]
    return sum(1 for doc in window if doc in relevant) / float(len(relevant))


def average_precision(ranking, relevant):
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for position, doc in enumerate(ranking, start=1):
        if doc in relevant:
            hits += 1
            total += hits / float(position)
    return total / len(relevant)


def reciprocal_rank(ranking, relevant):
    for position, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def discounted_gain(gains):
    return sum(gain / math.log(position + 1, 2) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(ranking, grades, k):
    """Graded nDCG@k; `grades` maps doc id -> relevance grade (0 means not relevant)."""
    if k <= 0 or not grades:
        return 0.0
    gains = [(2 ** grades.get(doc, 0)) - 1 for doc in ranking[:k]]
    ideal = sorted(((2 ** grade) - 1 for grade in grades.values()), reverse=True)[:k]
    if not sum(ideal):
        return 0.0
    return discounted_gain(gains) / discounted_gain(ideal)


def main():
    ranker = Ranker(DOCUMENTS)

    # 1. statistics: document frequency, lengths and the two idf definitions
    check(ranker.document_frequency["jumps"] == 2 and ranker.document_frequency["quick"] == 3,
          "document frequencies must count documents, not occurrences")
    check(ranker.lengths[4] == 6 and ranker.lengths[5] == 0, "document lengths come from the tokens")
    check(ranker.total == 8 and abs(ranker.average_length - sum(ranker.lengths.values()) / 8.0) < 1e-12,
          "corpus size and average length")
    check(abs(ranker.idf("jumps") - math.log(4.0)) < 1e-12,
          "idf must be log(N / df) = log(8/2)")
    check(ranker.idf("jumps") < ranker.idf("bears"), "rarer terms must have a larger idf")
    check(all(ranker.bm25_idf(term) > 0 for term in ranker.document_frequency),
          "the probabilistic idf stays positive even for frequent terms")
    for term in ("jumps", "the", "dog"):
        frequency = ranker.document_frequency[term]
        check(abs(ranker.bm25_idf(term) - math.log(1 + (8 - frequency + 0.5) / (frequency + 0.5)))
              < 1e-12, "the BM25 idf must use the shifted formula for %r" % term)

    # 2. BM25 against an independently transcribed formula, term by term
    for term in ("quick", "jumps", "dog", "bears"):
        for doc_id in sorted(DOCUMENTS):
            count = ranker.frequencies[doc_id].get(term, 0)
            if not count:
                expected = 0.0
            else:
                length = len(tokenize(DOCUMENTS[doc_id]))
                normalised = count + 1.2 * (1 - 0.75 + 0.75 * length / ranker.average_length)
                expected = math.log(1 + (8 - ranker.document_frequency[term] + 0.5)
                                    / (ranker.document_frequency[term] + 0.5)) \
                    * (count * (1.2 + 1)) / normalised
            check(abs(ranker.bm25_term(term, doc_id) - expected) < 1e-12,
                  "BM25(%r, %d) must match the formula" % (term, doc_id))

    # 3. additivity over query terms, and no contribution from absent terms
    multi = ranker.score(["quick", "brown"], 6)
    check(abs(multi - (ranker.bm25_term("quick", 6) + ranker.bm25_term("brown", 6))) < 1e-12,
          "a multi-term query is the sum of its term scores")
    check(ranker.score(["quick", "zebra"], 6) == ranker.score(["quick"], 6),
          "a term absent from the corpus contributes nothing")
    check(ranker.bm25_term("zebra", 6) == 0.0 and ranker.tf_idf_score("zebra", 6) == 0.0,
          "absent terms score zero in both models")

    # 4. monotonicity and saturation in the term frequency (same length, same document)
    repeated = Ranker({0: "word", 1: "word word", 2: "word word word word word",
                       3: "word word word word word word word word word word",
                       4: "other filler tokens here as well"})
    scores = [repeated.bm25_term("word", doc_id) for doc_id in range(4)]
    check(scores == sorted(scores) and scores[0] < scores[-1],
          "more occurrences must score higher: %r" % (scores,))
    check(scores[3] - scores[2] < scores[1] - scores[0],
          "the term-frequency contribution must saturate: %r" % (scores,))
    equal_length = Ranker({0: "word filler", 1: "word word"})
    saturated = equal_length.bm25_term("word", 1, k1=1.2) / equal_length.bm25_term("word", 0, k1=1.2)
    linear = equal_length.bm25_term("word", 1, k1=1000.0) / equal_length.bm25_term("word", 0, k1=1000.0)
    check(saturated < 1.5, "doubling the term frequency must not double the score: %.3f" % saturated)
    check(abs(linear - 2.0) < 0.01,
          "a huge k1 removes the saturation and the score becomes linear in tf: %.3f" % linear)

    # 5. length normalisation: the same frequency in a longer document scores lower when b > 0
    lengths = Ranker({0: "alpha beta gamma", 1: "alpha " + " ".join("w%d" % i for i in range(30))})
    check(lengths.bm25_term("alpha", 0) > lengths.bm25_term("alpha", 1),
          "with b = 0.75 a longer document must be penalised")
    check(abs(lengths.bm25_term("alpha", 0, b=0.0) - lengths.bm25_term("alpha", 1, b=0.0)) < 1e-12,
          "with b = 0 the length must not matter at all")
    check(lengths.bm25_term("alpha", 0, b=1.0) > lengths.bm25_term("alpha", 0, b=0.0),
          "full normalisation must boost a short document further")

    # 6. ranking: order, determinism, tie-breaking and the query interface
    ranking = ranker.rank("quick brown", limit=None)
    check(all(ranking[i][1] >= ranking[i + 1][1] for i in range(len(ranking) - 1)),
          "ranked scores must be non-increasing")
    check(ranking[0][0] == 6, "document 6 repeats 'quick', so it must rank first")
    check([doc for doc, _score in ranker.rank("quick brown", limit=2)] ==
          [doc for doc, _score in ranker.rank("quick brown")][:2],
          "the limit must cut the list without reordering")
    check(ranker.rank("") == [] and ranker.rank([]) == [], "an empty query ranks nothing")
    check(ranker.rank("zebra") == [], "a query with no hits ranks nothing")
    check([doc for doc, _score in ranker.rank("jumps over")] ==
          [doc for doc, _score in ranker.rank("over jumps")],
          "the term order in the query must not change the ranking")

    # 7. TF-IDF and BM25 agree on the obvious cases and differ elsewhere
    check(ranker.rank("quick brown", model="tfidf")[0][0] == 6,
          "the TF-IDF ranker must also put document 6 first")
    tfidf_scores = [ranker.score(["quick", "brown"], doc, "tfidf") for doc in sorted(DOCUMENTS)]
    bm25_scores = [ranker.score(["quick", "brown"], doc, "bm25") for doc in sorted(DOCUMENTS)]
    check(any(abs(a - b) > 1e-6 for a, b in zip(tfidf_scores, bm25_scores)),
          "the two models must not be the same function of the statistics")
    check(all(a >= 0 and b >= 0 for a, b in zip(tfidf_scores, bm25_scores)),
          "both models must produce non-negative scores for positive terms")

    # 8. evaluation measures against hand-computed cases
    check(precision_at_k([0, 1, 2, 3], {0, 2}, 2) == 0.5, "precision@2 = 1 hit in 2")
    check(precision_at_k([0, 1, 2, 3], {0, 2}, 4) == 0.5, "precision@4 = 2 hits in 4")
    check(abs(recall_at_k([0, 1, 2, 3], {0, 2, 9}, 2) - 1.0 / 3) < 1e-12,
          "recall@2 = 1 of 3 relevant")
    check(abs(recall_at_k([0, 1, 2, 3], {0, 2, 9}, 4) - 2.0 / 3) < 1e-12,
          "recall@4 = 2 of 3 relevant")
    check(average_precision([0, 1, 2, 3], {0, 2}) == (1.0 + 2.0 / 3) / 2,
          "average precision must average the precision at each hit")
    check(average_precision([2, 0, 1], {2}) == 1.0, "a relevant document first gives AP 1")
    check(reciprocal_rank([1, 0, 2], {0, 2}) == 0.5 and reciprocal_rank([1, 2], {0}) == 0.0,
          "reciprocal rank is 1/position of the first hit")
    check(precision_at_k([], {0}, 3) == 0.0 and recall_at_k([], {0}, 3) == 0.0,
          "an empty ranking scores zero")
    check(average_precision([0], set()) == 0.0 and recall_at_k([0], set(), 1) == 0.0,
          "no relevant documents means no score")

    # 9. nDCG: brute-force double sum, the ideal order, the reversed order, graded gains
    grades = {0: 3, 1: 2, 2: 1, 6: 2}
    order = [0, 1, 6, 2]                       # the ideal order: grades 3, 2, 2, 1
    reversed_order = [2, 6, 1, 0]
    brute = sum(((2 ** grades[doc]) - 1) / math.log(position + 1, 2)
                for position, doc in enumerate(order, start=1))
    ideal = sum(((2 ** grade) - 1) / math.log(position + 1, 2)
                for position, grade in enumerate(sorted(grades.values(), reverse=True), start=1))
    check(abs(ndcg_at_k(order, grades, 4) - brute / ideal) < 1e-12,
          "nDCG must equal the brute-force double sum divided by the ideal gain")
    check(abs(ndcg_at_k(order, grades, 4) - 1.0) < 1e-12,
          "the ideal order must give nDCG 1")
    check(ndcg_at_k(reversed_order, grades, 4) < ndcg_at_k(order, grades, 4) - 0.2,
          "the reversed order must score clearly lower")
    check(abs(ndcg_at_k(order, grades, 1) - 1.0) < 1e-12,
          "a single ideal document gives nDCG@1 = 1")
    check(ndcg_at_k(order, grades, 2) <= ndcg_at_k(order, grades, 3) <= 1.0,
          "nDCG must be bounded by 1 and monotone in k here")
    check(ndcg_at_k([1, 0], {0: 2, 1: 2}, 2) == 1.0,
          "two equally graded documents give nDCG 1 in either order")
    check(ndcg_at_k([2, 0], {0: 2, 2: 1}, 1) < 1.0,
          "a less relevant document first must cost nDCG")
    check(ndcg_at_k([0], {}, 1) == 0.0 and ndcg_at_k([0], grades, 0) == 0.0,
          "nDCG is zero without relevance or without k")
    check(ndcg_at_k([9, 9], grades, 5) == 0.0, "documents with no grade contribute nothing")

    # 10. a whole evaluation run: the honest ranker beats a reversed ranking
    query = "quick brown fox"
    relevant = {0, 1, 6}
    ranked = [doc for doc, _score in ranker.rank(query)][:5]
    reversed_ranking = list(reversed(ranked))
    check(recall_at_k(ranked, relevant, 5) >= recall_at_k(reversed_ranking, relevant, 5),
          "the ranker's order must not lose relevant documents")
    check(average_precision(ranked, relevant) > average_precision(reversed_ranking, relevant),
          "average precision must favour the ranker")
    check(reciprocal_rank(ranked, relevant) >= reciprocal_rank(reversed_ranking, relevant),
          "MRR must favour the ranker")
    check(ndcg_at_k(ranked, {0: 3, 1: 2, 6: 2}, 5) > ndcg_at_k(reversed_ranking, {0: 3, 1: 2, 6: 2}, 5),
          "nDCG must favour the ranker")

    # 11. determinism and the boundary documents
    check(ranker.rank("quick brown") == Ranker(DOCUMENTS).rank("quick brown"),
          "ranking must be reproducible")
    check(ranker.score("the", 5) == 0.0 and 5 not in [doc for doc, _s in ranker.rank("dog")],
          "the empty document never scores or ranks")
    check(len(ranker.rank("dog", limit=99)) == len(ranker.rank("dog")),
          "a limit above the result size is clamped")

    print("764 corpus=%d docs, average length %.2f, vocabulary %d"
          % (ranker.total, ranker.average_length, len(ranker.document_frequency)))
    print("764 bm25('quick brown') -> %r" % ([(doc, round(score, 4)) for doc, score in ranking[:3]],))
    print("764 bm25 vs tf-idf on doc 6: %.4f vs %.4f" % (ranker.score("quick brown", 6),
                                                          ranker.score("quick brown", 6, "tfidf")))
    print("764 evaluation: AP=%.3f MRR=%.3f nDCG@5=%.3f"
          % (average_precision(ranked, relevant), reciprocal_rank(ranked, relevant),
             ndcg_at_k(ranked, {0: 3, 1: 2, 6: 2}, 5)))
    print("PASS")


if __name__ == "__main__":
    main()
