"""Task 814 - Implement searchable encrypted index with limited leakage and proof-of-correctness.  (Tier: Master, CTCI Beyond - Security & Privacy)

A searchable symmetric encryption (SSE) index lets a client hand an untrusted server a set of opaque
entries such that the server can answer equality searches without ever seeing a keyword or a document
identifier.  The construction here is the classic inverted-index layout:

  * the client derives a *search token* per (keyword, epoch), `search_token(key, keyword, epoch)` - a
    keyed digest - so the server can locate an entry but cannot invert the handle into the keyword;
  * the posting list is sealed (keystream pad plus keyed tag) under a key derived from that token, so
    only the holder of the token can open it;
  * every posting list carries a *commitment*, a Merkle root over the sorted document ids folded with
    their count, authenticated with a keyed digest.  The client therefore verifies exactly what it
    asked for: dropping, injecting or truncating ids changes the root, and forging the commitment
    breaks its tag;
  * a *directory commitment* over all entry handles makes withholding a whole entry detectable: the
    client re-derives the root over the server's token set and compares it with the root committed at
    upload time.

Updates are appended, never rewritten: a batch of new documents is uploaded at a fresh epoch, so the
index is forward private in the measurable sense - a token captured before an update returns exactly
the list it returned before and cannot be used to learn which new documents contain that keyword.  The
price is explicit: a query costs one lookup per surviving epoch per term.  The leakage that remains is
the leakage an inverted-index SSE cannot avoid, and it is *measured* rather than claimed: the number
of distinct (keyword, epoch) handles (keyword cardinality), the search pattern (a repeated query
produces an identical token sequence) and the access pattern (which handles a query touches, and how
many ids each returned list carries).  Conjunctive queries are answered by fetching one list per term
and intersecting on the client, so the server never learns which documents matched the conjunction.

Verified: token determinism and epoch domain separation, handles and payloads containing no plaintext
keyword, exact agreement with a plaintext oracle for single-term and conjunctive queries (recall and
precision 1.0), the proof accepting an honest answer and a re-encoded but content-identical list,
rejecting a flipped payload, a truncated payload and a forged commitment, directory verification
detecting a withheld or injected entry, the forward-privacy measurement together with its lookup-cost
bound, an unknown keyword returning empty, a repeated term collapsing to one id per document, and a
scale run over 200 documents where recall stays exact and the cost stays bounded by terms times epochs.

Simplified: all cryptography is toy-scale - keyed splitmix64 digests and a keystream pad, no HMAC, no
block cipher, no real key material - so every security statement is a measurement over this
implementation, never a reduction; client, server and network live in one process and are driven
synchronously; there is no oblivious transfer, no padding and no ORAM layer, so the leakage metrics
printed are exactly the ones this layout exposes; and a server that answers "no such entry" for a
handle it did store is caught only by the directory commitment, not by the per-entry proof.
"""
MASK64 = (1 << 64) - 1
VOCABULARY = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
              "lambda", "mu", "nu", "xi", "omicron", "pi")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def digest(key, payload):
    """Keyed 64-bit digest: the toy stand-in for HMAC."""
    h = (key ^ 0x5A5A5A5A5A5A5A5A) & MASK64
    for byte in payload:
        h, r = splitmix64((h + byte + 0x9E37) & MASK64)
        h = (h ^ r) & MASK64
    return h


def hex16(value):
    return format(value & MASK64, "016x")


def keystream(key, length):
    state = (key ^ 0x0123456789ABCDEF) & MASK64
    out = bytearray()
    while len(out) < length:
        state, word = splitmix64(state)
        out.extend(word.to_bytes(8, "little"))
    return bytes(out[:length])


def seal(key, plaintext):
    """Pad-and-tag: XOR with the keystream, then tag the ciphertext with the key."""
    raw = plaintext.encode("utf-8")
    cipher = bytes(mask ^ byte for mask, byte in zip(keystream(key, len(raw)), raw))
    return cipher.hex(), digest(key, cipher)


def unseal(key, cipher_hex, tag):
    cipher = bytes.fromhex(cipher_hex)
    if digest(key, cipher) != tag:
        raise ValueError("tag mismatch: tampered payload or wrong key")
    raw = bytes(mask ^ byte for mask, byte in zip(keystream(key, len(cipher)), cipher))
    return raw.decode("utf-8")


class Rng:
    """Fixed-seed splitmix64 stream so every run builds the same corpus."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# ------------------------------------------------------------------ the scheme
def search_token(key, keyword, epoch):
    """The server-side handle for (keyword, epoch); a keyed digest, never the keyword itself."""
    if not keyword:
        raise ValueError("an empty keyword has no token")
    return hex16(digest(key, f"tok|{keyword}|{epoch}".encode("utf-8")))


def entry_key(key, token):
    """Per-entry key, so opening one entry does not open the others."""
    return digest(key, f"key|{token}".encode("utf-8")) | 1


def commitment_key(key, token):
    return digest(key, f"com|{token}".encode("utf-8")) | 1


def leaf_hash(key, token, doc_id):
    return digest(key, f"leaf|{token}|{doc_id}".encode("utf-8"))


def list_root(key, token, doc_ids):
    """Merkle root over the sorted ids; the count is folded in so truncation is caught as well."""
    level = [leaf_hash(key, token, doc_id) for doc_id in sorted(doc_ids)]
    if not level:
        return digest(key, f"empty|{token}".encode("utf-8"))
    while len(level) > 1:
        nxt = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            nxt.append(digest(key, f"node|{left:016x}{right:016x}".encode("utf-8")))
        level = nxt
    return digest(key, f"count|{token}|{len(doc_ids)}|{level[0]:016x}".encode("utf-8"))


def root_tag(key, token, root):
    """The authenticated commitment the client keeps next to the sealed list."""
    return digest(commitment_key(key, token), hex16(root).encode("utf-8"))


def directory_root(key, tokens):
    """Commitment over every entry handle, so a withheld entry is detectable too."""
    level = [digest(key, f"entry|{token}".encode("utf-8")) for token in sorted(tokens)]
    if not level:
        return digest(key, b"directory|empty")
    while len(level) > 1:
        nxt = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            nxt.append(digest(key, f"dir|{left:016x}{right:016x}".encode("utf-8")))
        level = nxt
    return level[0]


def tokenize(text):
    """Lower-case alphanumeric words; the only notion of a term in this index."""
    words = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


# ------------------------------------------------------------------ the server
class EncryptedIndexServer:
    """Stores opaque entries keyed by search token; it can neither read nor verify them."""

    def __init__(self):
        self.entries = {}
        self.lookups = 0
        self.access_log = []

    def store(self, token, payload_hex, tag, sealed_root):
        check(token not in self.entries, "an entry is write-once; updates go to a new epoch handle")
        self.entries[token] = (payload_hex, tag, sealed_root)

    def fetch(self, token):
        self.lookups += 1
        self.access_log.append(token)
        return self.entries.get(token)

    def tokens(self):
        return sorted(self.entries)


class TamperingServer(EncryptedIndexServer):
    """Modes exercised by the checks: flip a payload byte, truncate a payload, forge a commitment."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def fetch(self, token):
        entry = super().fetch(token)
        if entry is None or self.mode == "honest":
            return entry
        payload_hex, tag, sealed_root = entry
        if self.mode == "flip":
            raw = bytearray(bytes.fromhex(payload_hex))
            raw[0] ^= 0x01
            return raw.hex(), tag, sealed_root
        if self.mode == "truncate":
            return payload_hex[:-2], tag, sealed_root
        if self.mode == "commitment":
            return payload_hex, tag, hex16(digest(0xBAD, b"forged commitment"))
        if self.mode == "drop":
            return None
        raise AssertionError(f"unknown tampering mode {self.mode!r}")


# ------------------------------------------------------------------ the client
class SearchClient:
    """Derives handles, uploads sealed entries, verifies proofs and answers queries."""

    def __init__(self, key):
        self.key = key
        self.directory_root = None

    def upload(self, server, keyword, epoch, doc_ids):
        """Write the (keyword, epoch) entry: sealed posting list plus an authenticated commitment."""
        token = search_token(self.key, keyword, epoch)
        payload, tag = seal(entry_key(self.key, token),
                            ",".join(str(doc_id) for doc_id in sorted(doc_ids)))
        server.store(token, payload, tag,
                     hex16(root_tag(self.key, token, list_root(self.key, token, doc_ids))))
        return token

    def upload_batch(self, server, documents, epoch):
        """One update batch: only the keywords the batch touches, only the batch's own documents."""
        inverted = {}
        for doc_id, text in sorted(documents.items()):
            for word in set(tokenize(text)):
                inverted.setdefault(word, []).append(doc_id)
        for word, doc_ids in sorted(inverted.items()):
            self.upload(server, word, epoch, doc_ids)

    def commit_directory(self, server):
        self.directory_root = directory_root(self.key, server.tokens())
        return self.directory_root

    def verify_directory(self, server):
        check(self.directory_root is not None, "the directory must be committed before verification")
        return directory_root(self.key, server.tokens()) == self.directory_root

    def read_entry(self, server, keyword, epoch):
        """Fetch one entry and verify its proof; None means the handle was never written."""
        token = search_token(self.key, keyword, epoch)
        entry = server.fetch(token)
        if entry is None:
            return None
        payload_hex, tag, sealed_root = entry
        text = unseal(entry_key(self.key, token), payload_hex, tag)
        doc_ids = sorted(int(part) for part in text.split(",") if part)
        if int(sealed_root, 16) != root_tag(self.key, token, list_root(self.key, token, doc_ids)):
            raise ValueError("proof of correctness failed: the commitment does not match the list")
        return doc_ids

    def search(self, server, keyword, epochs):
        """Single-term search: union the per-epoch entries (the epoch count is the privacy cost)."""
        found = set()
        for epoch in range(epochs):
            found.update(self.read_entry(server, keyword, epoch) or ())
        return sorted(found)

    def conjunctive(self, server, terms, epochs):
        """AND: one list per term is fetched and the intersection is computed on the client."""
        lists = [set(self.search(server, term, epochs)) for term in terms]
        if not lists:
            return []
        return sorted(set.intersection(*lists))


def build_index(key, corpus, updates=()):
    """Base corpus at epoch 0, then one epoch per update batch; the client commits the directory."""
    server = EncryptedIndexServer()
    client = SearchClient(key)
    client.upload_batch(server, corpus, 0)
    for index, batch in enumerate(updates, start=1):
        client.upload_batch(server, batch, index)
    client.commit_directory(server)
    return server, client


def oracle(corpus, terms):
    """Plaintext oracle: the documents containing every term."""
    return [doc_id for doc_id, text in sorted(corpus.items())
            if all(term in set(tokenize(text)) for term in terms)]


def make_corpus(rng, documents, words_per_document):
    return {doc_id: " ".join(VOCABULARY[rng.below(len(VOCABULARY))]
                             for _ in range(words_per_document))
            for doc_id in range(documents)}


def main():
    key = 0x5EED1234ABCD
    corpus = {0: "alpha beta gamma", 1: "beta delta", 2: "alpha gamma epsilon",
              3: "zeta eta theta", 4: "alpha beta delta epsilon", 5: ""}
    server, client = build_index(key, corpus)

    # 1. handles: deterministic, domain-separated across keywords and epochs, never the keyword
    check(search_token(key, "alpha", 0) == search_token(key, "alpha", 0),
          "a handle must be deterministic")
    check(search_token(key, "alpha", 0) != search_token(key, "alpha", 1),
          "the handle must change with the epoch (forward privacy)")
    tokens = {search_token(key, word, 0) for word in VOCABULARY}
    check(len(tokens) == len(VOCABULARY), "distinct keywords must give distinct handles")
    check(all(word.encode() not in bytes.fromhex(token) for word in VOCABULARY for token in tokens),
          "a handle must not contain its own keyword")
    try:
        search_token(key, "", 0)
        check(False, "an empty keyword must be rejected")
    except ValueError:
        pass

    # 2. stored payloads are opaque: no keyword survives in the clear
    keywords = {word for text in corpus.values() for word in tokenize(text)}
    raw_entries = bytes.fromhex("".join(payload for payload, _, _ in server.entries.values()))
    check(all(word.encode() not in raw_entries for word in keywords),
          "no plaintext keyword may occur in the stored payloads")
    check(len(server.entries) == len(keywords),
          f"one entry per distinct keyword, got {len(server.entries)} for {len(keywords)}")

    # 3. single-term recall and precision against the plaintext oracle
    for word in VOCABULARY:
        expected = oracle(corpus, [word])
        got = client.search(server, word, epochs=1)
        check(got == expected, f"search({word!r}) gave {got}, the oracle says {expected}")
    check(client.search(server, "absent", epochs=1) == [],
          "an unknown keyword must return an empty answer")

    # 4. conjunctive queries agree exactly, and the server never learns the intersection
    for terms in (["alpha", "beta"], ["alpha", "gamma"], ["beta", "epsilon"], ["alpha", "zeta"],
                  ["alpha", "beta", "delta"], ["alpha", "beta", "epsilon"]):
        expected = oracle(corpus, terms)
        got = client.conjunctive(server, terms, epochs=1)
        check(got == expected, f"conjunctive {terms} gave {got}, the oracle says {expected}")
    server.access_log.clear()
    server.lookups = 0
    check(client.conjunctive(server, ["alpha", "beta"], epochs=1) == [0, 4],
          "the conjunctive answer must be exact")
    check(server.access_log == [search_token(key, "alpha", 0), search_token(key, "beta", 0)],
          "the server must only see one handle per term, never the intersection")
    check(server.lookups == 2, f"an AND query must cost one lookup per term, got {server.lookups}")

    # 5. the proof accepts an honest answer and a re-encoded but content-identical list
    check(client.read_entry(server, "alpha", 0) == [0, 2, 4],
          "the verifier must return the committed ids")
    check(client.read_entry(server, "absent", 0) is None, "an unwritten handle returns nothing")
    token = search_token(key, "alpha", 0)
    permuted = EncryptedIndexServer()
    for handle, entry in server.entries.items():
        permuted.entries[handle] = entry
    payload, tag = seal(entry_key(key, token), "4,2,0")            # same set, different order
    permuted.entries[token] = (payload, tag,
                               hex16(root_tag(key, token, list_root(key, token, [4, 2, 0]))))
    check(client.read_entry(permuted, "alpha", 0) == [0, 2, 4],
          "the commitment is over the sorted list, so a re-encoded permutation still verifies")
    empty_token = search_token(key, "absent", 0)
    empty_root = list_root(key, empty_token, [])
    check(empty_root == digest(key, f"empty|{empty_token}".encode("utf-8")),
          "an empty posting list must have a well-defined commitment")
    check(root_tag(key, empty_token, empty_root) != root_tag(key, token, empty_root),
          "the commitment tag must be bound to the handle")

    # 6. tampering is caught: flipped payload, truncated payload, forged commitment
    for mode in ("flip", "truncate", "commitment"):
        broken = TamperingServer(mode)
        for handle, entry in server.entries.items():
            broken.entries[handle] = entry
        try:
            SearchClient(key).read_entry(broken, "alpha", 0)
            check(False, f"tampering mode {mode!r} must be detected")
        except ValueError:
            pass
    dropped = TamperingServer("drop")
    for handle, entry in server.entries.items():
        dropped.entries[handle] = entry
    check(SearchClient(key).read_entry(dropped, "alpha", 0) is None,
          "a withheld handle is indistinguishable from an absent one at the entry level")

    # 7. withholding or injecting a whole entry is detected by the directory commitment
    honest = EncryptedIndexServer()
    for handle, entry in server.entries.items():
        honest.entries[handle] = entry
    check(client.verify_directory(honest), "the directory must verify when nothing is withheld")
    honest.entries.pop(token)
    check(not client.verify_directory(honest), "a withheld entry must break the directory root")
    honest.entries[search_token(key, "forged", 0)] = ("00", 0, "00")
    check(not client.verify_directory(honest), "an injected entry must break the directory root")

    # 8. forward privacy: a handle captured before an update cannot see the new documents
    updates = ({6: "alpha omega"}, {7: "beta omega"})
    grown = dict(corpus)
    for batch in updates:
        grown.update(batch)
    update_server, update_client = build_index(key, corpus, updates)
    check(len(update_server.entries) == len(keywords) + 4,
          f"each update adds only the handles it touches, got {len(update_server.entries)}")
    stale = update_client.search(update_server, "alpha", epochs=1)
    check(stale == oracle(corpus, ["alpha"]),
          f"the pre-update handle must return exactly the old list, got {stale}")
    check(6 not in stale, "a document added later must be invisible to the captured handle")
    fresh = update_client.search(update_server, "alpha", epochs=3)
    check(fresh == oracle(grown, ["alpha"]), f"the current handles must see everything, got {fresh}")
    check(6 in fresh and len(fresh) == len(stale) + 1,
          "the current handles must gain exactly the new document")
    check(update_client.verify_directory(update_server), "the updated directory must verify")

    # 9. the leakage and cost metrics this layout exposes are measured, not claimed
    repeated = [update_client.search(update_server, "alpha", epochs=3) for _ in range(3)]
    check(repeated[0] == repeated[1] == repeated[2], "the answer must be deterministic")
    epoch_handles = [search_token(key, "alpha", epoch) for epoch in range(3)]
    check(len(set(epoch_handles)) == 3, "each epoch leaks a separate handle for the same keyword")
    update_server.lookups = 0
    update_client.conjunctive(update_server, ["alpha", "beta", "gamma"], epochs=3)
    check(update_server.lookups == 9,
          f"an AND over three terms with three epochs must cost nine lookups, "
          f"got {update_server.lookups}")
    check(len(set(update_server.tokens())) == len(update_server.entries),
          "the server observes exactly one handle per (keyword, epoch)")

    # 10. scale: 200 documents, recall still exact, cost still bounded by terms times epochs
    base = make_corpus(Rng(0x2024), documents=200, words_per_document=8)
    scale_updates = ({200: "alpha omega"}, {201: "beta omega"})
    full = dict(base)
    for batch in scale_updates:
        full.update(batch)
    big_server, big_client = build_index(key, base, scale_updates)
    sampled = sorted({word for text in full.values() for word in tokenize(text)})
    check(len(sampled) >= 12, f"the sampled corpus must cover most of the vocabulary, {sampled}")
    for word in sampled:
        check(big_client.search(big_server, word, epochs=3) == oracle(full, [word]),
              f"scale search({word!r}) must match the oracle")
    for terms in (sampled[:2], sampled[:3], sampled[:4]):
        check(big_client.conjunctive(big_server, terms, epochs=3) == oracle(full, terms),
              f"scale conjunctive {terms} must match the oracle")
    check(big_client.verify_directory(big_server), "the scaled directory must verify")
    check(len(big_server.entries) <= 3 * len(sampled),
          "the entry count must stay bounded by keywords times epochs")
    big_server.lookups = 0
    big_client.conjunctive(big_server, ["alpha", "beta"], epochs=3)
    check(big_server.lookups == 6,
          f"the cost must be terms times epochs, independent of the corpus size, "
          f"got {big_server.lookups}")

    # 11. degenerate inputs
    empty_server, empty_client = build_index(key, {})
    check(empty_client.search(empty_server, "alpha", epochs=1) == [],
          "an empty corpus must answer every query with nothing")
    check(empty_client.verify_directory(empty_server), "an empty directory must verify")
    check(empty_client.conjunctive(empty_server, [], epochs=1) == [],
          "a conjunctive query with no terms returns nothing")
    check(tokenize("") == [] and tokenize("...") == [], "punctuation carries no terms")
    check(tokenize("Alpha, BETA!") == ["alpha", "beta"], "tokenisation must lower-case and split")
    single_server, single_client = build_index(key, {9: "alpha alpha", 10: "alpha"})
    check(single_client.search(single_server, "alpha", epochs=1) == [9, 10],
          "a repeated term must appear once per document in the posting list")

    print(f"checks: entries(small corpus)={len(server.entries)}, "
          f"entries(200 docs, 3 epochs)={len(big_server.entries)}, lookups per 2-term AND=6")
    print(f"leakage: keyword_cardinality={len(sampled)}, deterministic_answers=True, "
          f"handles_per_keyword=3, server_sees_intersection=False")
    print("PASS")


if __name__ == "__main__":
    main()
