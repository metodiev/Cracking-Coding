"""Task 762 - Searchable encryption demo: deterministic encryption and its leakage.  (Tier: Expert, Beyond - Security & Privacy)

Searchable encryption lets a server answer keyword queries over encrypted documents.  The cheapest
construction - deterministic encryption of each keyword, used as the index key - is also the
textbook example of *leakage*: the index is a function of the keyword alone, so

  * the *equality pattern* leaks: two documents share a keyword exactly when they share a token, so
    an adversary holding only the index recovers which documents have keywords in common;
  * the *frequency distribution* leaks: the number of documents per token is the document frequency
    of the keyword, so an adversary who knows the corpus statistics recovers the keyword behind
    every token;
  * the *search pattern* leaks: the same query produces the same token, so the server learns that
    two queries are equal (and, across time, what a client is interested in).

This file implements that scheme (`det_token`), a per-document variant (`doc_token`, token =
H(key, doc id, keyword)) and a randomized variant, and then *runs the adversaries* rather than
asserting the leaks: a co-occurrence adversary that reconstructs the document/keyword equality
partition and is scored with the balanced accuracy of its pairwise "shares a keyword" predictions,
and a frequency adversary that maps observed token counts to keywords using the corpus statistics.
The measured results are the point of the exercise: balanced accuracy 1.00 and an exact token ->
keyword recovery for the deterministic scheme; ~0.50 (chance) for the per-document variant, which
pays for it with N tokens per query instead of one and hands the server the full result set.

Simplified versus the literature: tokens are truncated SHA-256 values (a real hash function used as
the "encryption" of a keyword), there is no trapdoor, no ORAM and no forward privacy; the index is a
dictionary in memory instead of an encrypted data structure; the adversary is given the exact
corpus statistics (the strongest case) and the documents are tiny.  Verified: query correctness
against a brute-force scan, token determinism and collision freedom, both leaks quantified,
the trade-offs of the alternatives, and that randomized encryption removes the equality leak while
destroying searchability.
"""
import hashlib

DEMO_KEY = b"demo-search-key-not-secret"
DOCUMENTS = {
    0: "",
    1: "alpha",
    2: "alpha beta",
    3: "alpha beta gamma",
    4: "alpha beta gamma delta",
    5: "alpha beta gamma delta epsilon",
    6: "alpha beta gamma delta epsilon zeta",
    7: "alpha beta gamma delta epsilon zeta",
    8: "alpha beta gamma delta epsilon zeta",
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def token(key, *parts):
    """Deterministic token: the 'encryption' of a keyword under a key (truncated SHA-256)."""
    digest = hashlib.sha256(key)
    for part in parts:
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.hexdigest()[:16]


def keywords_of(text):
    return {word for word in text.split() if word}


def det_token(keyword, key=DEMO_KEY):
    return token(key, keyword.encode())


def doc_token(doc_id, keyword, key=DEMO_KEY):
    return token(key, keyword.encode(), b"doc", str(doc_id).encode())


def rand_token(keyword, nonce, key=DEMO_KEY):
    return token(key, keyword.encode(), b"rand", nonce)


def build_index(documents, key=DEMO_KEY, scheme="deterministic"):
    """token -> sorted document ids; equality between keywords is preserved or not by design."""
    index = {}
    for doc_id in sorted(documents):
        for keyword in keywords_of(documents[doc_id]):
            if scheme == "deterministic":
                entry = det_token(keyword, key)
            elif scheme == "per-document":
                entry = doc_token(doc_id, keyword, key)
            else:
                raise ValueError("unknown scheme %r" % (scheme,))
            index.setdefault(entry, []).append(doc_id)
    return {entry: sorted(docs) for entry, docs in index.items()}


def search(index, keyword, doc_ids, key=DEMO_KEY, scheme="deterministic"):
    """One lookup for the deterministic scheme; N tokens have to be derived for the other one."""
    if scheme == "deterministic":
        entry = det_token(keyword, key)
        return index.get(entry, []), [entry]
    tokens = [doc_token(doc_id, keyword, key) for doc_id in doc_ids]
    return sorted(doc_id for doc_id, entry in zip(doc_ids, tokens)
                  if entry in index), tokens


def brute_search(documents, keyword):
    return sorted(doc_id for doc_id, text in documents.items() if keyword in keywords_of(text))


def spearman(left, right):
    """Rank correlation of two sequences (1.0 = identical order, 0 = unrelated)."""
    if len(left) != len(right):
        raise ValueError("the two sequences must have the same length")
    if len(left) < 2:
        return 0.0
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0] * len(values)
        for position, index in enumerate(order):
            out[index] = position
        return out
    a, b = ranks(left), ranks(right)
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    va = sum((x - mean_a) ** 2 for x in a) ** 0.5
    vb = sum((y - mean_b) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def doc_profiles(index):
    """doc id -> the set of tokens it contains (what the server can see)."""
    profiles = {}
    for entry, docs in index.items():
        for doc_id in docs:
            profiles.setdefault(doc_id, set()).add(entry)
    return profiles


def pairwise_shares(profiles, doc_ids):
    """The adversary's prediction: two documents 'share a keyword' iff they share a token."""
    predictions = {}
    for left in doc_ids:
        for right in doc_ids:
            if left < right:
                predictions[(left, right)] = bool(profiles.get(left, set())
                                                  & profiles.get(right, set()))
    return predictions


def true_shares(documents, doc_ids):
    truth = {}
    for left in doc_ids:
        for right in doc_ids:
            if left < right:
                truth[(left, right)] = bool(keywords_of(documents[left])
                                            & keywords_of(documents[right]))
    return truth


def balanced_accuracy(predictions, truth):
    """Mean of the true-positive and true-negative rates: 0.5 means 'no information'."""
    positives = [key for key, value in truth.items() if value]
    negatives = [key for key, value in truth.items() if not value]
    tpr = sum(1 for key in positives if predictions[key]) / float(len(positives)) if positives else 0.0
    tnr = sum(1 for key in negatives if not predictions[key]) / float(len(negatives)) if negatives else 0.0
    return 0.5 * (tpr + tnr)


def frequency_attack(index):
    """Adversary with the corpus statistics: match observed token counts to keyword counts."""
    observed = sorted(((len(docs), entry) for entry, docs in index.items()), reverse=True)
    counts = {}
    for keyword, text in DOCUMENTS.items():
        for word in keywords_of(text):
            counts[word] = counts.get(word, 0) + 1
    known = sorted(((count, keyword) for keyword, count in counts.items()), reverse=True)
    return {entry: keyword for (count, entry), (other, keyword) in zip(observed, known)}


def main():
    doc_ids = sorted(DOCUMENTS)
    corpus_keywords = sorted({word for text in DOCUMENTS.values() for word in keywords_of(text)})
    counts = {}
    for text in DOCUMENTS.values():
        for word in keywords_of(text):
            counts[word] = counts.get(word, 0) + 1
    check(len(set(counts.values())) == len(counts) and len(counts) == 6,
          "the demo corpus must have six keywords with pairwise distinct document counts")

    # 1. search correctness against a brute-force scan, for every keyword and for absent ones
    index = build_index(DOCUMENTS)
    for keyword in corpus_keywords:
        hits, touched = search(index, keyword, doc_ids)
        check(hits == brute_search(DOCUMENTS, keyword),
              "search(%r) = %r, brute force says %r" % (keyword, hits, brute_search(DOCUMENTS, keyword)))
        check(len(touched) == 1, "the deterministic scheme needs exactly one token per query")
    check(search(index, "not-in-corpus", doc_ids)[0] == [], "an absent keyword must return nothing")
    check(search(index, "", doc_ids)[0] == [], "the empty query returns nothing")

    # 2. token determinism, and tokens are injective over the keyword set
    check(det_token("alpha") == det_token("alpha"), "tokens must be deterministic")
    check(det_token("alpha") != det_token("beta"), "different keywords must give different tokens")
    check(len({det_token(word) for word in corpus_keywords}) == len(corpus_keywords),
          "the token map must be injective on the corpus")
    check(det_token("alpha", b"other-key") != det_token("alpha"),
          "a different index key must give a different token")
    check(len({det_token("word%d" % i) for i in range(500)}) == 500,
          "no collisions over 500 synthetic keywords")

    # 3. the equality leak, measured: the adversary recovers the whole co-occurrence pattern
    profiles = doc_profiles(index)
    accuracy = balanced_accuracy(pairwise_shares(profiles, doc_ids), true_shares(DOCUMENTS, doc_ids))
    check(accuracy == 1.0, "the deterministic index leaks equality perfectly: %.3f" % accuracy)
    plain_signature = sorted(frozenset(keywords_of(DOCUMENTS[doc])) for doc in doc_ids)
    cipher_signature = sorted(frozenset(profiles.get(doc, set())) for doc in doc_ids)
    distinct_plain = len({frozenset(x) for x in plain_signature})
    distinct_cipher = len({frozenset(x) for x in cipher_signature})
    check(len(plain_signature) == len(cipher_signature) and distinct_plain == distinct_cipher,
          "the number of distinct document keyword-sets is leaked exactly (%d)" % distinct_plain)

    # 4. the frequency leak, measured: the token -> keyword mapping is recovered exactly
    recovered = frequency_attack(index)
    true_map = {det_token(word): word for word in corpus_keywords}
    check(recovered == true_map,
          "frequency analysis must recover every keyword: %r" % (recovered,))
    check(len(recovered) == len(true_map) == 6, "all six keywords must be recovered")
    check(sorted(len(docs) for docs in index.values()) == sorted(counts.values()),
          "the number of documents per token is exactly the document frequency")

    # 5. what deterministic encryption does *not* leak: the token order says nothing about the words
    by_keyword = [det_token(word) for word in corpus_keywords]
    order_correlation = spearman(by_keyword, corpus_keywords)
    check(abs(order_correlation) < 0.6,
          "tokens must not be order revealing: correlation %.3f" % order_correlation)
    check(abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
          and abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9,
          "the rank-correlation helper must be exact on known inputs")

    # 6. the search pattern leaks: identical queries produce identical transcripts
    first_hits, first_tokens = search(index, "gamma", doc_ids)
    second_hits, second_tokens = search(index, "gamma", doc_ids)
    other_hits, other_tokens = search(index, "delta", doc_ids)
    check(first_tokens == second_tokens, "a repeated query must look identical to the server")
    check(first_tokens != other_tokens, "different keywords must look different")
    check(first_hits == second_hits and other_hits != first_hits, "and the results stay correct")

    # 7. per-document tokens remove the equality leak - and cost one token per document
    per_doc = build_index(DOCUMENTS, scheme="per-document")
    per_profiles = doc_profiles(per_doc)
    per_accuracy = balanced_accuracy(pairwise_shares(per_profiles, doc_ids),
                                     true_shares(DOCUMENTS, doc_ids))
    check(per_accuracy <= 0.6,
          "per-document tokens must hide equality (balanced accuracy %.3f)" % per_accuracy)
    check(accuracy - per_accuracy >= 0.4,
          "the leak must be measurably worse for the deterministic scheme")
    hits, tokens_sent = search(per_doc, "gamma", doc_ids, scheme="per-document")
    check(hits == brute_search(DOCUMENTS, "gamma"), "the per-document variant is still correct")
    check(len(tokens_sent) == len(doc_ids),
          "the per-document variant must send %d tokens instead of 1" % len(doc_ids))
    observed_hits = [entry for entry in tokens_sent if entry in per_doc]
    check(len(observed_hits) == len(hits),
          "the server sees exactly the result set in the per-document scheme")
    check(len(per_doc) == sum(len(keywords_of(text)) for text in DOCUMENTS.values()),
          "per-document tokens never merge two documents' keywords")

    # 8. randomized encryption kills the equality leak too - and kills searchability with it
    randomized = {rand_token(word, b"nonce%d" % index_number): [doc]
                  for index_number, doc in enumerate([0, 1, 2]) for word in ["alpha", "beta"]}
    check(len(randomized) == 6, "randomized tokens never collide even for equal keywords")
    check(rand_token("alpha", b"n1") != rand_token("alpha", b"n2"),
          "randomized encryption hides equality")
    check(search(randomized, "alpha", doc_ids, scheme="deterministic")[0] == [],
          "a deterministic query token cannot find a randomized index entry")

    # 9. boundary: an empty document, a keyword in every document, duplicates within a document
    check(search(index, "alpha", doc_ids)[0] == brute_search(DOCUMENTS, "alpha"),
          "the most frequent keyword must be handled")
    check(0 not in [doc for doc in search(index, "alpha", doc_ids)[0]],
          "the empty document must never match")
    duplicate = dict(DOCUMENTS)
    duplicate[9] = "alpha alpha beta"
    duplicate_index = build_index(duplicate)
    check(duplicate_index[det_token("alpha")] == [1, 2, 3, 4, 5, 6, 7, 8, 9],
          "a keyword repeated inside one document is indexed once")
    empty_corpus = build_index({})
    check(empty_corpus == {} and search(empty_corpus, "alpha", doc_ids)[0] == [],
          "an empty corpus has an empty index")

    # 10. determinism of the whole pipeline
    check(build_index(DOCUMENTS) == index, "index construction must be reproducible")
    check(frequency_attack(build_index(DOCUMENTS)) == recovered, "the attack must be reproducible")
    check(doc_profiles(build_index(DOCUMENTS, scheme="per-document")) == per_profiles,
          "the per-document index must be reproducible")

    print("762 corpus: %d documents, %d keywords, counts %r"
          % (len(doc_ids), len(corpus_keywords), sorted(counts.values(), reverse=True)))
    print("762 equality leak: deterministic balanced accuracy %.3f, per-document %.3f"
          % (accuracy, per_accuracy))
    print("762 frequency attack: %d/%d keywords recovered exactly; order correlation %.3f"
          % (len(recovered), len(true_map), order_correlation))
    print("762 query cost: 1 token (deterministic) vs %d tokens (per-document)"
          % len(tokens_sent))
    print("PASS")


if __name__ == "__main__":
    main()
