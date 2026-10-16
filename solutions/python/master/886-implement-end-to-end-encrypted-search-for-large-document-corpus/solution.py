"""Task 886 - Implement end-to-end encrypted search for large document corpus with privacy budget.  (Tier: Master, CTCI Beyond - Security, Privacy & Cryptography)

The corpus never leaves the client in the clear.  Each document is serialised as its sorted
term list and sealed with a hand-rolled encrypt-then-MAC scheme built on stdlib primitives:
the keystream is SHA-256 in counter mode over (key, nonce), the ciphertext is XORed against it
and authenticated with HMAC-SHA256 truncated to 128 bits (a flipped ciphertext bit, or the
wrong nonce, is rejected).  The searchable index is a blind index: the client tokenises each
term as HMAC(index_key, "T|term") so the server's postings table is keyed by 128-bit
pseudorandom tokens, and document identifiers are blinded the same way.  The server therefore
performs a plain set intersection over opaque tokens, returns opaque payloads, and can never
hold a keyword or a document id in clear - the API refuses a non-bytes token outright.

Search is a conjunction: the client submits the blinded tokens for the query terms, the server
returns the postings of the rarest token (the cost bound: work is proportional to the shortest
posting list, never to the corpus), and the client decrypts, re-checks the terms against the
plaintext (so a token collision cannot produce a false positive) and ranks by term frequency
with a deterministic tie-break.  Results are compared against a brute-force plaintext oracle.

The privacy budget is the second half of the story.  Each query releases a differentially
private match count through the Laplace mechanism with scale 1/epsilon, and the accountant
charges that epsilon by basic composition.  The budget is enforced exactly: the spent total is
the sum of the released epsilons, remaining budget never goes negative, and once the budget is
exhausted every further query raises BudgetExhausted instead of silently leaking.  The
mechanism itself is calibrated: its empirical variance matches 2 * (1/epsilon)^2.

Simplified versus a deployed system: no ORAM/oblivious access (the server sees which token was
queried, which is exactly why an epsilon is charged per query), no homomorphic scoring, single
server, no concurrency, the corpus is a small generated vocabulary rather than gigabytes of
text, and the Laplace noise comes from a seeded PRNG rather than a hardware entropy source.
Verified invariants: encryption/decryption round-trips exactly, tampering is detected, the
server's entire state contains no plaintext term or document id, search recall and ranking are
exact against the brute-force oracle, query cost stays proportional to the rarest posting
list, the budget accounting is exact and refuses over-spending, and every run is reproducible.
"""
import hashlib
import hmac
import math

MASK64 = (1 << 64) - 1
TOKEN_SIZE = 16
TAG_SIZE = 16
BAND_TOLERANCE = 0.25


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def uniform(self):
        return (self.u64() >> 11) / 9007199254740992.0


class IntegrityError(Exception):
    """Raised when an authenticated packet fails verification."""


class PlaintextLeak(Exception):
    """Raised when a caller tries to hand the server a non-blinded value."""


class BudgetExhausted(Exception):
    """Raised when a query would exceed the differential-privacy budget."""


# ----------------------------------------------------------- encryption layer
def keystream(key, nonce, length):
    """SHA-256 in counter mode; deterministic and independent for each (key, nonce)."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:length])


def seal(key, nonce, plaintext):
    """Encrypt-then-MAC: ciphertext followed by a 16 byte tag over (nonce, ciphertext)."""
    stream = keystream(key, nonce, len(plaintext))
    body = bytes(p ^ s for p, s in zip(plaintext, stream))
    tag = hmac.new(key, b"MAC" + nonce + body, hashlib.sha256).digest()[:TAG_SIZE]
    return body + tag


def unseal(key, nonce, packet):
    body, tag = packet[:-TAG_SIZE], packet[-TAG_SIZE:]
    expected = hmac.new(key, b"MAC" + nonce + body, hashlib.sha256).digest()[:TAG_SIZE]
    if not hmac.compare_digest(tag, expected):
        raise IntegrityError("packet failed authentication")
    stream = keystream(key, nonce, len(body))
    return bytes(c ^ s for c, s in zip(body, stream))


def blind(key, purpose, value):
    """Keyed 128-bit token: the server side only ever sees one of these."""
    payload = purpose.encode() + b"|" + value.encode()
    return hmac.new(key, payload, hashlib.sha256).digest()[:TOKEN_SIZE]


# ------------------------------------------------------- privacy budget layer
def laplace(scale, rng):
    """Inverse-CDF Laplace sampler driven by the seeded PRNG (no wall-clock entropy)."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    u = rng.uniform() - 0.5
    magnitude = -scale * math.log(1.0 - 2.0 * abs(u))
    return magnitude if u >= 0 else -magnitude


class QueryBudget:
    """Basic composition over per-query epsilons; never lets the total exceed the budget."""

    def __init__(self, total_epsilon):
        if total_epsilon <= 0.0:
            raise ValueError("the total privacy budget must be positive")
        self.total_epsilon = float(total_epsilon)
        self.spent = 0.0
        self.releases = []

    @property
    def remaining(self):
        return self.total_epsilon - self.spent

    def charge(self, epsilon, label=""):
        if epsilon <= 0.0:
            raise ValueError("a query epsilon must be strictly positive")
        if epsilon > self.remaining + 1e-12:
            raise BudgetExhausted(
                "query %r needs epsilon=%.3f but only %.3f remain" % (label, epsilon, self.remaining))
        self.spent += epsilon
        self.releases.append((label, epsilon))
        return epsilon


class NoisyCount:
    """Differentially private count release: exact value plus Laplace(sensitivity/epsilon)."""

    def __init__(self, accountant, rng):
        self.accountant = accountant
        self.rng = rng
        self.releases_made = 0

    def release(self, exact_count, epsilon, label=""):
        self.accountant.charge(epsilon, label)
        self.releases_made += 1
        return exact_count + laplace(1.0 / epsilon, self.rng)


# ------------------------------------------------------------- corpus + index
def generate_corpus(rng, doc_count, vocabulary_size, min_terms=6, max_terms=12):
    """Each document is a sorted set of distinct terms drawn from a fixed vocabulary."""
    if max_terms > vocabulary_size:
        raise ValueError("max_terms cannot exceed the vocabulary size")
    vocabulary = ["term%04d" % index for index in range(vocabulary_size)]
    documents = []
    for _ in range(doc_count):
        wanted = min_terms + rng.below(max_terms - min_terms + 1)
        terms = set()
        while len(terms) < wanted:
            terms.add(vocabulary[rng.below(vocabulary_size)])
        documents.append(sorted(terms))
    return vocabulary, documents


class Client:
    """Holds the keys, builds the blind index and issues queries."""

    def __init__(self, keys, documents):
        self.index_key = keys[0]
        self.id_key = keys[1]
        self.payload_key = keys[2]
        self.documents = [sorted(set(terms)) for terms in documents]
        self.tokens = [self._token(term) for term in sorted({t for d in self.documents for t in d})]
        self.payloads = {}
        self.token_map = {}
        self._build()

    def _token(self, term):
        return blind(self.index_key, "term", term)

    def _doc_token(self, index):
        return blind(self.id_key, "doc", "%08d" % index)

    def _nonce(self, index):
        return hashlib.sha256(self.id_key + b"nonce" + index.to_bytes(4, "big")).digest()[:12]

    def _build(self):
        for index, terms in enumerate(self.documents):
            doc_token = self._doc_token(index)
            payload = seal(self.payload_key, self._nonce(index), " ".join(terms).encode())
            self.payloads[doc_token] = (self._nonce(index), payload)
            self.token_map[doc_token] = index

    def index_for_server(self):
        """The only structure the server ever receives: opaque tokens to opaque payloads."""
        postings = {}
        for index, terms in enumerate(self.documents):
            doc_token = self._doc_token(index)
            _, payload = self.payloads[doc_token]
            for term in terms:
                postings.setdefault(self._token(term), []).append((doc_token, payload))
        for token in postings:
            postings[token].sort()
        return postings

    def query_tokens(self, terms):
        return tuple(sorted({self._token(term) for term in terms}))

    def decode(self, entries, terms):
        """Decrypt a server reply and keep only documents that really contain every term."""
        wanted = set(terms)
        scored = []
        for doc_token, payload in entries:
            nonce, _ = self.payloads[doc_token]
            plaintext = unseal(self.payload_key, nonce, payload).decode().split()
            present = set(plaintext)
            if wanted <= present:
                scored.append((self.token_map[doc_token], plaintext.count(terms[0]) if terms else 0))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


class EncryptedServer:
    """Blind server: intersects opaque tokens and can only ever see ciphertext."""

    def __init__(self, postings):
        self.postings = postings
        self.queries_served = 0
        self.postings_scanned = 0

    def search(self, tokens):
        if not isinstance(tokens, tuple):
            tokens = tuple(tokens)
        for token in tokens:
            if not isinstance(token, bytes):
                raise PlaintextLeak("the server accepts blinded bytes only, got %r" % (token,))
        if not tokens:
            return []
        lists = [self.postings.get(token, []) for token in tokens]
        smallest = min(range(len(lists)), key=lambda position: len(lists[position]))
        self.queries_served += 1
        self.postings_scanned += len(lists[smallest])
        return list(lists[smallest])

    def store_size(self):
        return len(self.postings), sum(len(entries) for entries in self.postings.values())


# ---------------------------------------------------------------- plain oracle
def brute_force(documents, terms):
    """Exact ranked answer, computed without any index or encryption."""
    scored = []
    for index, document in enumerate(documents):
        if set(terms) <= set(document):
            scored.append((index, document.count(terms[0]) if terms else 0))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def run_query(client, server, terms, budget, rng, epsilon):
    """One end-to-end query: charge budget, release a noisy count, run the blind search."""
    tokens = client.query_tokens(terms)
    exact = len(server.search(tokens))
    noisy = NoisyCount(budget, rng).release(exact, epsilon, "count:%s" % ",".join(terms))
    answer = client.decode(server.search(tokens), terms)
    return {"noisy_count": noisy, "exact_count": exact, "answer": answer, "tokens": tokens}


# ----------------------------------------------------------------------- main
def main():
    keys = (b"index-key-886", b"id-key-886", b"payload-key-886")
    rng = Rng(886001)
    vocabulary, documents = generate_corpus(rng, 300, 40)
    client = Client(keys, documents)
    server = EncryptedServer(client.index_for_server())

    # --- 1. encryption round-trips and authenticates -----------------------
    nonce = b"nonce-000001"
    message = b"the quick brown fox jumps over the lazy dog"
    packet = seal(keys[2], nonce, message)
    check(unseal(keys[2], nonce, packet) == message, "encryption must round-trip exactly")
    check(len(packet) == len(message) + TAG_SIZE, "the packet must be ciphertext plus tag")
    check(seal(keys[2], nonce, message) == packet, "sealing must be deterministic")
    check(seal(keys[2], b"nonce-000002", message) != packet, "a nonce change must change the output")
    check(seal(b"other-key", nonce, message) != packet, "a key change must change the output")
    check(b"quick brown" not in packet, "the plaintext must not appear in the packet")
    flipped = bytearray(packet)
    flipped[3] ^= 0x01
    raised = False
    try:
        unseal(keys[2], nonce, bytes(flipped))
    except IntegrityError:
        raised = True
    check(raised, "a flipped ciphertext bit must fail authentication")
    raised = False
    try:
        unseal(keys[2], b"nonce-000002", packet)
    except IntegrityError:
        raised = True
    check(raised, "the tag must cover the nonce")
    check(unseal(keys[2], nonce, seal(keys[2], nonce, b"")) == b"",
          "an empty plaintext must round-trip")

    # --- 2. the keystream behaves like a PRF -------------------------------
    stream_a = keystream(keys[0], nonce, 256)
    stream_b = keystream(keys[0], b"nonce-000002", 256)
    check(len(stream_a) == 256 and len(keystream(keys[0], nonce, 100)) == 100,
          "the keystream must honour the requested length")
    check(keystream(keys[0], nonce, 100) == stream_a[:100], "the keystream must be a prefix")
    differing = sum(1 for a, b in zip(stream_a, stream_b) if a != b)
    check(differing > 0.40 * 256,
          "a one-symbol nonce change must avalanche, only %d/256 bytes differ" % differing)

    # --- 3. the server never holds plaintext -------------------------------
    server_blob = repr(server.postings).encode()
    for term in vocabulary[:40]:
        check(term.encode() not in server_blob,
              "the served index must not contain the plaintext term %r" % term)
    for index in range(len(documents)):
        check(("%08d" % index).encode() not in server_blob,
              "the served index must not contain a cleartext document id")
    token_count, entry_count = server.store_size()
    check(token_count == len(client.tokens), "every indexed term must have exactly one token")
    check(entry_count == sum(len(document) for document in documents),
          "the postings must cover every (document, term) pair")

    # --- 4. tokens are key dependent ---------------------------------------
    other_client = Client((b"index-key-other", keys[1], keys[2]), documents)
    shared = set(client.index_for_server()) & set(other_client.index_for_server())
    check(not shared, "a different index key must yield a disjoint token space")

    # --- 5. recall and ranking against the brute-force oracle --------------
    vocabulary_terms = vocabulary
    checked_queries = 0
    for _ in range(60):
        size = 1 + rng.below(3)
        terms = [vocabulary_terms[rng.below(len(vocabulary_terms))] for _ in range(size)]
        expected = brute_force(documents, terms)
        answer = client.decode(server.search(client.query_tokens(terms)), terms)
        check(answer == expected,
              "search must match the brute-force oracle for %r: %r vs %r" % (terms, answer, expected))
        check(len(answer) <= len(server.search(client.query_tokens(terms))),
              "the post-filter must never invent documents")
        checked_queries += 1
    check(checked_queries == 60, "all randomised queries must run")

    # --- 6. cost is bounded by the rarest posting list ---------------------
    worst_scan = 0
    for _ in range(40):
        size = 2 + rng.below(2)
        terms = [vocabulary_terms[rng.below(len(vocabulary_terms))] for _ in range(size)]
        tokens = client.query_tokens(terms)
        before = server.postings_scanned
        server.search(tokens)
        scanned = server.postings_scanned - before
        rarest = min(len(server.postings.get(token, [])) for token in tokens)
        worst_scan = max(worst_scan, scanned)
        check(scanned == rarest, "the server must scan the shortest posting list")
        check(scanned < len(documents),
              "a query must be cheaper than a linear scan: %d vs %d" % (scanned, len(documents)))

    # --- 7. degenerate corpora and queries ---------------------------------
    empty_server = EncryptedServer(Client(keys, []).index_for_server())
    check(empty_server.search(()) == [] and empty_server.search(
        (blind(keys[0], "term", "term0000"),)) == [],
        "an empty corpus must answer every query with nothing")
    single_client = Client(keys, [["term0000", "term0001"]])
    single_server = EncryptedServer(single_client.index_for_server())
    check(single_client.decode(single_server.search(single_client.query_tokens(["term0000"])),
                               ["term0000"]) == [(0, 1)],
          "a single-document corpus must answer correctly")
    check(single_client.decode(single_server.search(single_client.query_tokens(["nope"])),
                               ["nope"]) == [],
          "a term outside the corpus must return an empty result")
    missing = "term9999"
    check(client.decode(server.search(client.query_tokens([missing])), [missing]) == [],
          "an absent term must return an empty result on the large corpus too")
    duplicated = client.decode(server.search(client.query_tokens(["term0000", "term0000"])),
                               ["term0000"])
    check(duplicated == brute_force(documents, ["term0000"]),
          "a repeated query term must be deduplicated")
    check(server.search(()) == [], "an empty query must return nothing without touching postings")

    # --- 8. the server refuses anything that is not blinded -----------------
    raised = False
    try:
        server.search(("term0000",))
    except PlaintextLeak:
        raised = True
    check(raised, "the server API must reject a cleartext token")

    # --- 9. exact budget accounting and refusal ----------------------------
    budget = QueryBudget(1.0)
    check(abs(budget.remaining - 1.0) < 1e-15, "a fresh budget must be untouched")
    release_rng = Rng(886500)
    run_query(client, server, ["term0000"], budget, release_rng, 0.25)
    run_query(client, server, ["term0001"], budget, release_rng, 0.25)
    check(abs(budget.spent - 0.5) < 1e-12, "composition must add the epsilons exactly")
    check(abs(budget.remaining - 0.5) < 1e-12, "remaining budget must be total minus spent")
    check([epsilon for _, epsilon in budget.releases] == [0.25, 0.25],
          "every release must be recorded for auditing")
    run_query(client, server, ["term0002"], budget, release_rng, 0.5)
    check(abs(budget.remaining) < 1e-12, "the budget must be exactly exhausted")
    raised = False
    try:
        run_query(client, server, ["term0003"], budget, release_rng, 0.25)
    except BudgetExhausted:
        raised = True
    check(raised, "a query beyond the budget must be refused, never silently released")
    check(budget.spent <= budget.total_epsilon + 1e-12,
          "spending must never exceed the advertised budget")
    check(abs(sum(epsilon for _, epsilon in budget.releases) - budget.spent) < 1e-12,
          "the audit trail must sum to the spent budget")
    raised = False
    try:
        QueryBudget(0.0)
    except ValueError:
        raised = True
    check(raised, "a non-positive total budget must be rejected")
    raised = False
    try:
        budget.charge(0.0)
    except ValueError:
        raised = True
    check(raised, "a non-positive query epsilon must be rejected")

    # --- 10. the Laplace mechanism is calibrated to epsilon ----------------
    measured = {}
    for epsilon in (0.5, 2.0):
        draws = [laplace(1.0 / epsilon, rng) for _ in range(4000)]
        mean = sum(draws) / len(draws)
        variance = sum((value - mean) ** 2 for value in draws) / len(draws)
        measured[epsilon] = (mean, variance)
        expected_variance = 2.0 * (1.0 / epsilon) ** 2
        check(abs(mean) < BAND_TOLERANCE * (1.0 / epsilon),
              "epsilon=%.2f noise must be centred, mean %.4f" % (epsilon, mean))
        check(abs(variance - expected_variance) < BAND_TOLERANCE * expected_variance,
              "epsilon=%.2f variance %.4f must be close to 2/epsilon^2 = %.4f"
              % (epsilon, variance, expected_variance))
    check(measured[0.5][1] > measured[2.0][1],
          "a smaller epsilon must inject strictly more noise")
    mechanism = NoisyCount(QueryBudget(50.0), Rng(886700))
    noisy = [mechanism.release(100, 1.0, "repeat") for _ in range(50)]
    check(len(set(noisy)) > 40, "the mechanism must actually randomise its answers")
    check(mechanism.releases_made == 50, "every release must be counted")

    # --- 11. reproducibility ----------------------------------------------
    def replay(seed):
        local = Rng(seed)
        vocab, docs = generate_corpus(local, 120, 25)
        own_client = Client(keys, docs)
        own_server = EncryptedServer(own_client.index_for_server())
        own_budget = QueryBudget(5.0)
        answers = []
        for _ in range(15):
            terms = [vocab[local.below(len(vocab))] for _ in range(2)]
            answers.append(run_query(own_client, own_server, terms, own_budget, local, 0.3))
        return answers, own_budget.spent, own_server.postings_scanned

    first = replay(886800)
    second = replay(886800)
    check(first == second, "an identical seed must replay the corpus, index and query stream")
    check(abs(first[1] - 15 * 0.3) < 1e-12, "the replayed budget must be accounted identically")

    # --- 12. scale: a 2000 document corpus ---------------------------------
    scale_rng = Rng(886900)
    scale_vocabulary, scale_documents = generate_corpus(scale_rng, 2000, 60)
    scale_client = Client(keys, scale_documents)
    scale_server = EncryptedServer(scale_client.index_for_server())
    scale_budget = QueryBudget(4.0)
    selective = 0
    mismatches = 0
    for _ in range(10):
        size = 2 + scale_rng.below(2)
        terms = [scale_vocabulary[scale_rng.below(len(scale_vocabulary))] for _ in range(size)]
        tokens = scale_client.query_tokens(terms)
        before = scale_server.postings_scanned
        reply = scale_server.search(tokens)
        scanned = scale_server.postings_scanned - before
        answer = scale_client.decode(reply, terms)
        if answer != brute_force(scale_documents, terms):
            mismatches += 1
        if scanned * 4 < len(scale_documents):
            selective += 1
        run_query(scale_client, scale_server, terms, scale_budget, scale_rng, 0.4)
    check(mismatches == 0, "the large corpus must answer every query exactly")
    check(selective > 0, "at least one conjunctive query must scan under a quarter of the corpus")
    check(abs(scale_budget.spent - 4.0) < 1e-12,
          "the scaled workload must consume the planned budget exactly")
    check(scale_server.queries_served >= 20, "each query must hit the server at least once")
    documents_stored = sum(len(document) for document in scale_documents)
    check(documents_stored > 15000, "the scaled corpus must be substantial, got %d postings"
          % documents_stored)

    print("corpus: %d documents, %d distinct terms, %d postings"
          % (len(documents), token_count, entry_count))
    print("recall: 60 random conjunctive queries matched the brute-force oracle exactly")
    print("query cost: worst scan %d postings over %d documents (linear scan would touch all)"
          % (worst_scan, len(documents)))
    print("budget audit: total %.2f, spent %.2f, releases %d"
          % (budget.total_epsilon, budget.spent, len(budget.releases)))
    print("laplace variance: epsilon=0.5 -> %.4f, epsilon=2.0 -> %.4f"
          % (measured[0.5][1], measured[2.0][1]))
    print("scaled corpus: %d documents, %d postings, %d selective queries, budget spent %.2f"
          % (len(scale_documents), documents_stored, selective, scale_budget.spent))
    print("PASS")


if __name__ == "__main__":
    main()
