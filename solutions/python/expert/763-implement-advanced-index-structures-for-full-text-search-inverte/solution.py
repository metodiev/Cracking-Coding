"""Task 763 - Advanced index structures for full-text search (inverted index with positions).  (Tier: Expert, Beyond - Machine Learning & AI)

A positional inverted index maps every term to the sorted list of documents containing it, and for
each document to the *positions* of the term.  Positions turn a bag-of-words index into a real
search engine: a phrase query is an intersection of the posting lists followed by a positional
check, and a proximity query is a positional check with a distance bound.

The index here is built the way an on-disk engine would store it: for every term, the postings are
delta-encoded and written as varbyte numbers (doc-id deltas, then the count and the position deltas),
so the encoded blob can be decoded back to the exact postings; a skip list (one entry every
`SKIP_STEP` postings) lets `advance_to` jump over whole blocks instead of scanning.  Boolean queries
(AND / OR / NOT), phrase and proximity queries, prefix (wildcard) expansion over the sorted
vocabulary using binary search, and a TF-IDF-style top-k ranking are then answered from that
structure.

Every answer is cross-checked against an independent brute-force oracle: the postings are compared
with a second tokenizer built on `re.findall`, phrase/proximity results with a scan over each
document's token list, boolean results with plain set algebra over the matching document sets,
prefix expansion with a linear scan of the vocabulary, and `advance_to` with a naive linear walk of
the same posting list (the checks also count the skip hops to show the skip list is really used).

Simplified versus Lucene/Sphinx: everything is in memory, there is no compression beyond varbyte
deltas, no term dictionary with front coding, no WAND/block-max skipping, no scoring beyond a toy
TF-IDF, no stemming and no stopwords - the tokenizer is a lower-casing alphanumeric split.  Verified:
posting correctness, delta/varbyte round-trips (including multi-byte boundaries), phrase/proximity
correctness, boolean algebra, skip-list equivalence with a linear walk, prefix expansion, ranking
order, and the boundary cases (empty document, empty query, absent term, k larger than the result).
"""
import bisect
import math
import re
from itertools import islice

SKIP_STEP = 2

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
    """Lower-casing alphanumeric split; positions are token offsets in the document."""
    tokens = []
    current = ""
    for character in text.lower():
        if character.isalnum():
            current += character
        elif current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens


def oracle_tokens(text):
    """Second, independent tokenizer (regular expression) used to verify the postings."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ------------------------------------------------------------------ varbyte delta encoding

def encode_numbers(numbers):
    out = bytearray()
    for number in numbers:
        if number < 0:
            raise ValueError("varbyte encoding needs non-negative integers")
        while True:
            chunk = number & 0x7F
            number >>= 7
            if number:
                out.append(chunk | 0x80)
            else:
                out.append(chunk)
                break
    return bytes(out)


def decode_numbers(blob):
    numbers = []
    value = 0
    shift = 0
    for byte in blob:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            numbers.append(value)
            value = 0
            shift = 0
    if shift:
        raise ValueError("truncated varbyte sequence")
    return numbers


def encode_postings(postings):
    """postings: [(doc_id, [positions])] -> delta-encoded varbyte blob."""
    numbers = []
    previous_doc = 0
    for doc_id, positions in postings:
        numbers.append(doc_id - previous_doc)
        previous_doc = doc_id
        numbers.append(len(positions))
        previous_position = 0
        for position in positions:
            numbers.append(position - previous_position)
            previous_position = position
    return encode_numbers(numbers)


def decode_postings(blob):
    numbers = decode_numbers(blob)
    postings = []
    previous_doc = 0
    index = 0
    while index < len(numbers):
        doc_id = previous_doc + numbers[index]
        previous_doc = doc_id
        index += 1
        count = numbers[index]
        index += 1
        positions = []
        previous_position = 0
        for _ in range(count):
            previous_position += numbers[index]
            positions.append(previous_position)
            index += 1
        postings.append((doc_id, positions))
    return postings


class PositionalIndex:
    """Inverted index with positions, delta-varbyte postings, skips and query operators."""

    def __init__(self, documents, skip_step=SKIP_STEP):
        self.documents = dict(documents)
        self.skip_step = skip_step
        self.reverse = {}                      # term -> {doc_id: [positions]}
        for doc_id in sorted(self.documents):
            for position, term in enumerate(tokenize(self.documents[doc_id])):
                self.reverse.setdefault(term, {}).setdefault(doc_id, []).append(position)
        self.postings = {term: sorted(docs.items()) for term, docs in self.reverse.items()}
        self.vocabulary = sorted(self.postings)
        self.blobs = {term: encode_postings(postings) for term, postings in self.postings.items()}
        self.doc_lengths = {doc_id: len(tokenize(text)) for doc_id, text in self.documents.items()}
        self.hops = 0
        self.comparisons = 0

    # ------------------------------------------------------------------------ postings access
    def posting_list(self, term):
        return self.postings.get(term, [])

    def docs_with(self, term):
        return [doc_id for doc_id, _positions in self.posting_list(term)]

    def decode(self, term):
        return decode_postings(self.blobs[term]) if term in self.blobs else []

    def skips_for(self, term):
        return [(doc_id, index) for index, (doc_id, _p) in enumerate(self.posting_list(term))
                if index % self.skip_step == 0]

    def advance_to(self, term, target):
        """First posting whose doc id is >= target, using the skip list."""
        postings = self.posting_list(term)
        skips = self.skips_for(term)
        start = 0
        for doc_id, index in skips:
            self.comparisons += 1
            if doc_id <= target:
                start = index
                self.hops += 1
            else:
                break
        index = start
        while index < len(postings):
            self.comparisons += 1
            if postings[index][0] >= target:
                return postings[index]
            index += 1
        return None

    def linear_advance(self, term, target):
        """The oracle: walk the posting list from the beginning."""
        for doc_id, positions in self.posting_list(term):
            if doc_id >= target:
                return (doc_id, positions)
        return None

    # ---------------------------------------------------------------------------- retrieval
    def and_docs(self, terms):
        if not terms:
            return []
        sets = [set(self.docs_with(term)) for term in terms]
        result = sets[0]
        for other in sets[1:]:
            result &= other
        return sorted(result)

    def or_docs(self, terms):
        result = set()
        for term in terms:
            result |= set(self.docs_with(term))
        return sorted(result)

    def not_docs(self, terms):
        return sorted(set(self.documents) - set(self.or_docs(terms)))

    def phrase(self, text):
        """Phrase query: intersect the postings, then check consecutive positions."""
        terms = tokenize(text)
        if not terms:
            return []
        if len(terms) == 1:
            return self.docs_with(terms[0])
        candidates = set(self.docs_with(terms[0]))
        for term in terms[1:]:
            candidates &= set(self.docs_with(term))
            if not candidates:
                return []
        matches = []
        for doc_id in sorted(candidates):
            positions = [self.reverse[term][doc_id] for term in terms]
            first = positions[0]
            for offset in first:
                if all(offset + index in set(positions[index])
                       for index in range(1, len(terms))):
                    matches.append(doc_id)
                    break
        return matches

    def proximity(self, left, right, distance):
        """Documents where the two terms occur at most `distance` tokens apart."""
        matches = []
        for doc_id in sorted(set(self.docs_with(left)) & set(self.docs_with(right))):
            first = self.reverse[left][doc_id]
            second = self.reverse[right][doc_id]
            for position in first:
                if any(abs(position - other) <= distance for other in second):
                    matches.append(doc_id)
                    break
        return matches

    def prefix(self, prefix):
        """Wildcard expansion over the sorted vocabulary using binary search."""
        start = bisect.bisect_left(self.vocabulary, prefix)
        terms = []
        for term in self.vocabulary[start:]:
            if not term.startswith(prefix):
                break
            terms.append(term)
        return terms

    def score(self, term, doc_id):
        """Toy TF-IDF: sublinear term frequency times inverse document frequency."""
        positions = self.reverse.get(term, {}).get(doc_id)
        if not positions:
            return 0.0
        frequency = len(positions)
        documents_with_term = len(self.postings[term])
        return (1 + math.log(frequency)) * math.log(len(self.documents) / documents_with_term)

    def rank(self, terms, limit):
        scores = {}
        for doc_id in set(self.or_docs(terms)):
            total = sum(self.score(term, doc_id) for term in set(terms))
            if total > 0:
                scores[doc_id] = total
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return list(islice(ordered, limit))


# ------------------------------------------------------------------- brute-force oracles

def brute_phrase(documents, text):
    terms = tokenize(text)
    hits = []
    for doc_id in sorted(documents):
        tokens = tokenize(documents[doc_id])
        for start in range(len(tokens) - len(terms) + 1):
            if tokens[start:start + len(terms)] == terms and terms:
                hits.append(doc_id)
                break
    return hits


def brute_proximity(documents, left, right, distance):
    hits = []
    for doc_id in sorted(documents):
        tokens = tokenize(documents[doc_id])
        positions_left = [i for i, token in enumerate(tokens) if token == left]
        positions_right = [i for i, token in enumerate(tokens) if token == right]
        if any(abs(a - b) <= distance for a in positions_left for b in positions_right):
            hits.append(doc_id)
    return hits


def main():
    index = PositionalIndex(DOCUMENTS)

    # 1. the postings agree with an independently written tokenizer
    expected = {}
    for doc_id in sorted(DOCUMENTS):
        for position, term in enumerate(oracle_tokens(DOCUMENTS[doc_id])):
            expected.setdefault(term, {}).setdefault(doc_id, []).append(position)
    check(sorted(index.postings) == sorted(expected), "the vocabulary must match the oracle")
    for term, docs in expected.items():
        check(index.postings[term] == sorted(docs.items()),
              "postings of %r must match the oracle" % term)
    check(len(tokenize(DOCUMENTS[3])) == 5 and tokenize(DOCUMENTS[5]) == [],
          "tokenization boundaries: punctuation and an empty document")
    check(index.reverse["jumps"][4] == [0, 1, 2], "repeated terms keep every position")

    # 2. delta + varbyte encoding round-trips exactly, including multi-byte boundaries
    for term in index.vocabulary:
        check(decode_postings(index.blobs[term]) == index.postings[term],
              "the encoded postings of %r must decode back" % term)
    for value in (0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 2 ** 31):
        check(decode_numbers(encode_numbers([value])) == [value],
              "varbyte round-trip for %d" % value)
    check(decode_numbers(encode_numbers([0, 127, 128, 300, 2 ** 20])) == [0, 127, 128, 300, 2 ** 20],
          "a mixed varbyte sequence must round-trip")
    check(len(index.blobs["the"]) < 16, "delta encoding keeps the postings small")
    try:
        encode_numbers([-1])
        raise AssertionError("a negative number must be rejected")
    except ValueError:
        pass
    try:
        decode_numbers(encode_numbers([200]) + b"\x80")
        raise AssertionError("a truncated varbyte sequence must be rejected")
    except ValueError:
        pass

    # 3. the skip list returns exactly what a linear walk returns, and it really skips
    for term in ("the", "quick", "brown", "dog", "fox"):
        for target in range(0, 9):
            check(index.advance_to(term, target) == index.linear_advance(term, target),
                  "advance_to(%r, %d) must match the linear walk" % (term, target))
    check(index.hops > 0, "the skip list must actually be used: %d hops" % index.hops)
    check(index.comparisons > 0, "and the comparisons must be counted")
    long_term = max(index.postings, key=lambda term: len(index.postings[term]))
    postings = index.posting_list(long_term)
    check(index.skips_for(long_term) ==
          [(doc_id, position) for position, (doc_id, _p) in enumerate(postings)
           if position % SKIP_STEP == 0],
          "skip entries must sit every %d postings" % SKIP_STEP)
    check(index.advance_to("absent-term", 0) is None, "an absent term advances to nothing")

    # 4. phrase queries against a brute-force scan of the token lists
    for phrase in ("quick brown", "brown fox", "the lazy dog", "lazy dog", "dog dog",
                   "quick quick brown", "jumps over the lazy", "brown quick", "dog lazy"):
        check(index.phrase(phrase) == brute_phrase(DOCUMENTS, phrase),
              "phrase %r: %r vs %r" % (phrase, index.phrase(phrase), brute_phrase(DOCUMENTS, phrase)))
    check(index.phrase("quick quick brown") == [6], "a phrase with a repeated term must be found")
    check(index.phrase("dog dog") == [7], "a repeated word phrase matches document 7 only")
    check(index.phrase("") == [] and index.phrase("   ") == [], "an empty phrase matches nothing")
    check(index.phrase("quick") == sorted(index.docs_with("quick")),
          "a single-word phrase is just the posting list")

    # 5. proximity queries against the brute-force scan
    for left, right, distance in (("quick", "fox", 1), ("quick", "fox", 3), ("lazy", "dog", 0),
                                  ("jumps", "fence", 2), ("brown", "bears", 2)):
        check(index.proximity(left, right, distance) ==
              brute_proximity(DOCUMENTS, left, right, distance),
              "proximity %r/%r within %d" % (left, right, distance))
    check(index.proximity("quick", "fox", 100) == index.and_docs(["quick", "fox"]),
          "a huge window degenerates to a conjunction")
    check(3 not in index.proximity("quick", "fox", 1), "document 3 has neither term in range")

    # 6. boolean algebra against plain set operations
    check(index.and_docs(["quick", "fox"]) ==
          sorted(set(index.docs_with("quick")) & set(index.docs_with("fox"))), "AND must intersect")
    check(index.or_docs(["fox", "bears"]) ==
          sorted(set(index.docs_with("fox")) | set(index.docs_with("bears"))), "OR must union")
    check(index.not_docs(["the"]) == sorted(set(DOCUMENTS) - set(index.docs_with("the"))),
          "NOT must complement")
    check(index.and_docs(["quick", "zebra"]) == [], "AND with an absent term is empty")
    check(index.or_docs(["zebra", "bears"]) == index.docs_with("bears"),
          "OR with an absent term keeps the present one")
    check(index.and_docs([]) == [] and index.or_docs([]) == [] and index.not_docs([]) ==
          sorted(DOCUMENTS), "the empty conjunction/disjunction/complement")

    # 7. prefix expansion against a linear scan of the vocabulary
    for prefix in ("do", "b", "qu", "zz", "", "the"):
        expected_terms = [term for term in index.vocabulary if term.startswith(prefix)]
        check(index.prefix(prefix) == expected_terms, "prefix %r expansion" % prefix)
    check(index.prefix("do") == ["dog", "dogs"], "the dog prefix expands to both forms")
    check(index.prefix("zz") == [], "an unmatched prefix expands to nothing")
    check(index.prefix("") == index.vocabulary, "the empty prefix lists the whole vocabulary")

    # 8. ranking: TF-IDF order, the limit, and the clamped k
    ranked = index.rank(["quick", "brown"], 3)
    check(len(ranked) == 3, "the limit must be honoured")
    check(all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1)),
          "scores must be non-increasing")
    check([doc for doc, _score in ranked] == [6, 0, 1], "document 6 repeats 'quick' twice")
    check(len(index.rank(["quick"], 99)) == len(index.docs_with("quick")),
          "a limit above the result size is clamped")
    check(index.rank(["zebra"], 5) == [], "an absent term scores nothing")
    check(index.score("zebra", 0) == 0.0, "an absent term has no score")
    check(index.score("jumps", 4) > index.score("jumps", 0),
          "more occurrences must score higher in the same document set")

    # 9. boundary cases
    single = PositionalIndex({0: "a", 1: "b b"})
    check(single.docs_with("a") == [0] and single.postings["b"] == [(1, [0, 1])],
          "single character terms are indexed")
    check(single.phrase("b b") == [1], "a two-token document supports a repeated phrase")
    check(single.advance_to("b", 1) == (1, [0, 1]) and single.advance_to("b", 2) is None,
          "advance past the last posting returns nothing")
    check(PositionalIndex({}).vocabulary == [] and PositionalIndex({}).rank(["x"], 1) == [],
          "an empty corpus has an empty vocabulary and no results")
    check(index.doc_lengths[5] == 0 and index.docs_with("the") == [0, 2, 4],
          "the empty document contributes no postings")

    # 10. determinism and payload independence
    twin = PositionalIndex(DOCUMENTS)
    check(twin.blobs == index.blobs and twin.postings == index.postings,
          "index construction must be reproducible")
    check(twin.phrase("quick brown") == index.phrase("quick brown"), "queries must be reproducible")

    print("763 documents=%d terms=%d postings=%d encoded=%d bytes"
          % (len(DOCUMENTS), len(index.vocabulary),
             sum(len(p) for p in index.postings.values()), sum(len(b) for b in index.blobs.values())))
    print("763 phrase 'quick brown' -> %r; proximity(quick, fox, 3) -> %r"
          % (index.phrase("quick brown"), index.proximity("quick", "fox", 3)))
    print("763 skips: %d hops, %d comparisons for the advance probes" % (index.hops, index.comparisons))
    print("763 rank(quick, brown, 3) -> %r; vocabulary %r" % (ranked, index.vocabulary))
    print("PASS")


if __name__ == "__main__":
    main()
