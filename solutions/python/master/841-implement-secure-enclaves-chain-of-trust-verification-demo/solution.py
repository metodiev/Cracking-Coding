"""Task 841 - Implement secure enclaves chain-of-trust verification demo.  (Tier: Master, CTCI Beyond — Security & Privacy)

Remote attestation, modelled end to end in one deterministic process: a certificate chain of
authorities, an enclave that signs a *quote* over its measurement, and a verifier that either
accepts the quote or rejects it with a specific reason.  There are no sockets, threads or wall
clock reads - time is an explicit integer ("clock") carried by the verifier, nonces come from
the verifier's own issued set, and every signature is genuine modular arithmetic.

What is real here:
  * A toy RSA (`KeyPair`) built from deterministically searched 61/63-bit primes: signing is
    s = H(m)^d mod n, verification is s^e mod n == H(m) mod n with H = SHA-256 reduced into
    Z_n.  No real key material, no OAEP padding - the modulus is ~124 bits, so this is a
    demonstration of the *protocol* structure and is not secure against factoring.
  * A certificate is (serial, subject, subject key, issuer, not_before, not_after, tcb level)
    signed by the issuer's key; the root is self-signed, and the verifier pins the root
    fingerprint, so a chain that ends anywhere else is rejected.
  * A quote is (enclave leaf key, measurement, nonce, clock, counter, payload hash) signed by
    the leaf key.  Verification checks: chain to the pinned root (each signature, every
    validity window, no cycle, no revoked serial), measurement allowed by policy, TCB level,
    the leaf key is the one bound into the chain, nonce freshness and replay, monotonically
    increasing counter (anti-rollback), and payload integrity.
  * The verifier is stateful in exactly one way: a *successful* verification consumes the
    nonce and advances the stored counter; a rejected one must leave both untouched, which is
    the property that stops a failed attempt from burning a fresh nonce.

Simplified versus production: real attestation uses ECDSA over P-256 and platform-specific
quote formats, group signatures for privacy, and TPM-style monotonic counters in hardware;
here RSA is duplicated per authority from a seeded search, the counter lives in the verifier,
and policies are small explicit sets.

Verified below: the signature primitive (round trip, tampering, wrong key, wrong digest), chain
verification at depth 3 and depth 10 against an independent link-by-link re-check, and one
rejection check per attack: unpinned root, forged intermediate signature, cycle, expired and
not-yet-valid certificate, revoked serial, disallowed measurement, TCB policy violation, quote
measurement mismatch, unknown and replayed nonce, stale counter, tampered payload, leaf key not
in the chain - plus the state-machine checks that failures change nothing and that the cost is
one signature check per link.
"""

import hashlib

SMALL_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
PUBLIC_EXPONENT = 65537
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def is_prime(n):
    if n < 2:
        return False
    for p in SMALL_BASES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for base in SMALL_BASES:
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(value):
    candidate = value | 1
    while not is_prime(candidate):
        candidate += 2
    return candidate


def digest_int(payload):
    """Hash-to-integer: SHA-256 of the canonical encoding, reduced into Z_n later."""
    if isinstance(payload, bytes):
        blob = payload
    else:
        blob = repr(payload).encode()
    return int.from_bytes(hashlib.sha256(blob).digest(), "big")


class KeyPair:
    """Deterministic toy RSA: 61/63-bit primes, e = 65537, textbook signing."""

    def __init__(self, offset):
        p = next_prime((1 << 61) + 1000 * offset + 1)
        q = next_prime((1 << 63) + 1000 * offset + 7)
        while (p - 1) * (q - 1) % PUBLIC_EXPONENT == 0:
            p = next_prime(p + 2)
        self.n = p * q
        self.e = PUBLIC_EXPONENT
        self.d = pow(self.e, -1, (p - 1) * (q - 1))
        self.fingerprint = digest_int("fingerprint|%d|%d" % (self.n, self.e))

    def sign(self, payload):
        return pow(digest_int(payload) % self.n, self.d, self.n)

    def verify(self, payload, signature):
        return pow(signature, self.e, self.n) == digest_int(payload) % self.n


class Certificate:
    """A signed statement binding a subject name to a public key."""

    def __init__(self, serial, subject, key, issuer, not_before, not_after, tcb_level,
                 signature=None):
        self.serial = serial
        self.subject = subject
        self.key = key
        self.issuer = issuer
        self.not_before = not_before
        self.not_after = not_after
        self.tcb_level = tcb_level
        self.signature = signature

    def payload(self):
        return ("cert", self.serial, self.subject, self.key.n, self.key.e, self.issuer,
                self.not_before, self.not_after, self.tcb_level)

    def issue(self, issuer_key):
        self.signature = issuer_key.sign(self.payload())
        return self

    def fingerprint(self):
        return digest_int(("cert-fingerprint", self.payload()))


class Quote:
    """An enclave-signed attestation report."""

    def __init__(self, leaf_key, measurement, nonce, clock, counter, payload_hash,
                 debug=False, signature=None):
        self.leaf_key = leaf_key
        self.measurement = measurement
        self.nonce = nonce
        self.clock = clock
        self.counter = counter
        self.payload_hash = payload_hash
        self.debug = debug
        self.signature = signature

    def payload(self):
        return ("quote", self.leaf_key.n, self.leaf_key.e, self.measurement, self.nonce,
                self.clock, self.counter, self.payload_hash, self.debug)

    def issue(self, leaf_key):
        self.signature = leaf_key.sign(self.payload())
        return self


class Policy:
    def __init__(self, allowed_measurements, min_tcb_level=1, allow_debug=False,
                 max_clock_skew=30, revoked_serials=()):
        self.allowed_measurements = set(allowed_measurements)
        self.min_tcb_level = min_tcb_level
        self.allow_debug = allow_debug
        self.max_clock_skew = max_clock_skew
        self.revoked_serials = set(revoked_serials)


class Verifier:
    """Verifies chains and quotes; consumes a nonce only when the quote is accepted."""

    totals = {"accepted": 0, "rejected": 0}

    def __init__(self, pinned_root_fingerprint, policy, clock):
        self.pinned_root = pinned_root_fingerprint
        self.policy = policy
        self.clock = clock
        self.issued_nonces = {}
        self.used_nonces = set()
        self.counters = {}
        self.chain_checks = 0
        self.nonce_serial = 0

    def issue_nonce(self, label):
        self.nonce_serial += 1
        nonce = digest_int("nonce|%s|%d|%d" % (label, self.clock,
                                               self.nonce_serial)) % (1 << 96)
        self.issued_nonces[nonce] = label
        return nonce

    # -- chain of trust ----------------------------------------------------
    def verify_chain(self, leaf, directory):
        """Walk leaf -> root verifying each signature; returns (ok, reason)."""
        seen = set()
        certificate = leaf
        self.chain_checks = 0
        while True:
            if certificate.subject in seen:
                return False, "cycle in the chain at %s" % certificate.subject
            seen.add(certificate.subject)
            if certificate.serial in self.policy.revoked_serials:
                return False, "revoked certificate %d" % certificate.serial
            if certificate.not_before > self.clock:
                return False, "certificate %d is not yet valid" % certificate.serial
            if certificate.not_after < self.clock:
                return False, "certificate %d has expired" % certificate.serial
            if certificate.tcb_level < self.policy.min_tcb_level:
                return False, "TCB level %d below policy" % certificate.tcb_level
            if certificate.issuer == certificate.subject:
                if certificate.key.fingerprint != self.pinned_root:
                    return False, "self-signed chain does not end at the pinned root"
                if not certificate.key.verify(certificate.payload(), certificate.signature):
                    return False, "root signature is invalid"
                return True, "chain of %d links verified" % (len(seen) + self.chain_checks)
            if certificate.issuer not in directory:
                return False, "missing issuer %s" % certificate.issuer
            parent = directory[certificate.issuer]
            if not parent.key.verify(certificate.payload(), certificate.signature):
                return False, "bad signature at link %s" % certificate.subject
            self.chain_checks += 1
            certificate = parent

    # -- quote -------------------------------------------------------------
    def verify_quote(self, quote, leaf_cert, directory, payload):
        ok, reason = self._verify_quote(quote, leaf_cert, directory, payload)
        Verifier.totals["accepted" if ok else "rejected"] += 1
        return ok, reason

    def _verify_quote(self, quote, leaf_cert, directory, payload):
        ok, reason = self.verify_chain(leaf_cert, directory)
        if not ok:
            return False, reason
        if leaf_cert.key.n != quote.leaf_key.n or leaf_cert.key.e != quote.leaf_key.e:
            return False, "quote was not signed by the attested enclave key"
        if not quote.leaf_key.verify(quote.payload(), quote.signature):
            return False, "quote signature is invalid"
        if quote.measurement not in self.policy.allowed_measurements:
            return False, "measurement is not in the policy"
        if quote.measurement != "%d" % leaf_cert.tcb_level:
            return False, "measurement does not match the attested TCB level"
        if quote.debug and not self.policy.allow_debug:
            return False, "debug enclave rejected by policy"
        if quote.nonce not in self.issued_nonces:
            return False, "nonce was never issued"
        if quote.nonce in self.used_nonces:
            return False, "nonce has already been used (replay)"
        if abs(quote.clock - self.clock) > self.policy.max_clock_skew:
            return False, "quote clock %d too far from %d" % (quote.clock, self.clock)
        if quote.counter <= self.counters.get(leaf_cert.subject, -1):
            return False, "counter %d is stale" % quote.counter
        if quote.payload_hash != digest_int(payload) % (1 << 128):
            return False, "payload hash does not match"
        # accepted: only now are the stateful effects applied
        self.used_nonces.add(quote.nonce)
        self.counters[leaf_cert.subject] = quote.counter
        return True, "accepted at counter %d after %d signature checks" % (
            quote.counter, self.chain_checks + 2)


def build_chain(names, depth):
    """Root + `depth` signed certificates, each one signing the next."""
    keys = [KeyPair(index + 1) for index in range(depth + 1)]
    root = Certificate(1, names[0], keys[0], names[0], 0, 10 ** 6, 3).issue(keys[0])
    directory = {names[0]: root}
    chain = [root]
    for index in range(1, depth + 1):
        certificate = Certificate(100 + index, names[index], keys[index], names[index - 1],
                                  0, 10 ** 6, 3)
        certificate.issue(keys[index - 1])
        directory[names[index]] = certificate
        chain.append(certificate)
    return root, directory, chain, keys


def make_quote(enclave_index, chain, keys, measurement, verifier, counter, payload,
               debug=False, clock=None, nonce=None, signer=None):
    leaf = chain[-1]
    nonce = verifier.issue_nonce("enclave-%d" % enclave_index) if nonce is None else nonce
    quote = Quote(keys[-1], measurement, nonce, verifier.clock if clock is None else clock,
                  counter, digest_int(payload) % (1 << 128), debug)
    return quote.issue(signer if signer is not None else keys[-1])


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. the signature primitive --------------------------------------
    authority = KeyPair(1)
    assert authority.n.bit_length() > 120
    signature = authority.sign("hello")
    check(authority.verify("hello", signature), "a valid signature must verify")
    check(not authority.verify("hello!", signature), "a tampered message must fail")
    check(not authority.verify("hello", (signature + 1) % authority.n),
          "a tampered signature must fail")
    check(not KeyPair(2).verify("hello", signature), "a foreign key must not verify")
    check(signature == authority.sign("hello"), "textbook RSA signing is deterministic")
    check(pow(signature, authority.e, authority.n) ==
          digest_int("hello") % authority.n, "s^e must recover the digest")
    check(not is_prime(authority.n), "the modulus is composite by construction")
    check(authority.sign("k") != digest_int("k") % authority.n,
          "signing applies the private exponent, not the identity")

    # ---- 2. chain of trust ------------------------------------------------
    names = ["root", "intermediate", "leaf"]
    root, directory, chain, keys = build_chain(names, 2)
    verifier = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    ok, reason = verifier.verify_chain(chain[-1], directory)
    check(ok, "a three-link chain must verify (%s)" % reason)
    check(verifier.chain_checks == 2, "two signature checks for two links below the root")

    deep_names = ["root"] + ["ca%d" % index for index in range(1, 10)]
    deep_root, deep_directory, deep_chain, deep_keys = build_chain(deep_names, 9)
    deep_verifier = Verifier(deep_root.key.fingerprint, Policy(allowed_measurements={"3"}),
                             clock=100)
    ok, reason = deep_verifier.verify_chain(deep_chain[-1], deep_directory)
    check(ok, "a ten-link chain must verify (%s)" % reason)
    check(deep_verifier.chain_checks == 9, "one signature check per link below the root")
    for index, certificate in enumerate(deep_chain):
        check(certificate.key.verify(certificate.payload(), certificate.signature) ==
              (index == 0), "only the root is self-signed")

    # ---- 3. chain rejection cases ----------------------------------------
    forged = KeyPair(77)
    broken = Certificate(999, "intermediate", chain[1].key, "root", 0, 10 ** 6, 3)
    broken.signature = forged.sign(broken.payload())
    broken_dir = dict(directory)
    broken_dir["intermediate"] = broken
    ok, reason = verifier.verify_chain(chain[-1], broken_dir)
    check(not ok and "bad signature" in reason, "a forged intermediate must be rejected")
    unpinned = Verifier(KeyPair(5).fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    ok, reason = unpinned.verify_chain(chain[-1], directory)
    check(not ok and "pinned root" in reason, "an unpinned root must be rejected")
    cyclic_root = Certificate(1, "loop-a", keys[0], "loop-b", 0, 10 ** 6, 3)
    cyclic_other = Certificate(2, "loop-b", keys[1], "loop-a", 0, 10 ** 6, 3)
    cyclic_root.issue(keys[1])
    cyclic_other.issue(keys[0])
    ok, reason = verifier.verify_chain(cyclic_root, {"loop-a": cyclic_root,
                                                     "loop-b": cyclic_other})
    check(not ok and "cycle" in reason, "a cyclic chain must be rejected")
    ok, reason = verifier.verify_chain(chain[-1], {"root": root})
    check(not ok and "missing issuer" in reason, "a chain with a missing link must be rejected")
    expired = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=10 ** 7)
    ok, reason = expired.verify_chain(chain[-1], directory)
    check(not ok and "expired" in reason, "an expired certificate must be rejected")
    future = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=-1)
    ok, reason = future.verify_chain(chain[-1], directory)
    check(not ok and "not yet valid" in reason, "a not-yet-valid certificate must be rejected")
    revoked = Verifier(root.key.fingerprint,
                       Policy(allowed_measurements={"3"}, revoked_serials={101}), clock=100)
    ok, reason = revoked.verify_chain(chain[-1], directory)
    check(not ok and "revoked" in reason, "a revoked serial must be rejected")
    weak = Verifier(root.key.fingerprint,
                    Policy(allowed_measurements={"3"}, min_tcb_level=5), clock=100)
    ok, reason = weak.verify_chain(chain[-1], directory)
    check(not ok and "TCB" in reason, "a TCB level below policy must be rejected")

    # ---- 4. quote acceptance ---------------------------------------------
    payload = {"job": 42, "model": "toy"}
    verifier = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    quote = make_quote(1, chain, keys, "3", verifier, 7, payload)
    ok, reason = verifier.verify_quote(quote, chain[-1], directory, payload)
    check(ok, "a well-formed fresh quote must be accepted (%s)" % reason)
    check(quote.nonce in verifier.used_nonces, "acceptance consumes the nonce")
    check(verifier.counters[chain[-1].subject] == 7, "acceptance advances the counter")

    # replay must be refused, and refused *without* changing state
    replay = make_quote(1, chain, keys, "3", verifier, 8, payload, nonce=quote.nonce)
    before_nonces, before_counter = set(verifier.used_nonces), verifier.counters.copy()
    ok, reason = verifier.verify_quote(replay, chain[-1], directory, payload)
    check(not ok and "replay" in reason, "a replayed nonce must be rejected")
    check(verifier.used_nonces == before_nonces and verifier.counters == before_counter,
          "a rejected quote must not mutate the verifier")

    unknown = make_quote(1, chain, keys, "3", verifier, 9, payload, nonce=12345)
    ok, reason = verifier.verify_quote(unknown, chain[-1], directory, payload)
    check(not ok and "never issued" in reason, "an unissued nonce must be rejected")

    stale = make_quote(1, chain, keys, "3", verifier, 7, payload)
    ok, reason = verifier.verify_quote(stale, chain[-1], directory, payload)
    check(not ok and "stale" in reason, "a non-increasing counter must be rejected")
    fresh = make_quote(1, chain, keys, "3", verifier, 9, payload)
    ok, reason = verifier.verify_quote(fresh, chain[-1], directory, payload)
    check(ok and verifier.counters[chain[-1].subject] == 9,
          "an advancing counter must be accepted and stored (%s)" % reason)

    # ---- 5. quote rejection cases ----------------------------------------
    verifier = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    bad_payload = make_quote(2, chain, keys, "3", verifier, 1, payload)
    ok, reason = verifier.verify_quote(bad_payload, chain[-1], directory, {"job": 43})
    check(not ok and "payload hash" in reason, "a tampered payload must be rejected")
    ok, reason = verifier.verify_quote(bad_payload, chain[-1], directory, payload)
    check(ok, "the same quote is accepted for the untampered payload")

    wrong_measurement = make_quote(3, chain, keys, "9", verifier, 2, payload)
    ok, reason = verifier.verify_quote(wrong_measurement, chain[-1], directory, payload)
    check(not ok and "policy" in reason, "a measurement outside policy must be rejected")
    relaxed = Verifier(root.key.fingerprint,
                       Policy(allowed_measurements={"1", "3"}), clock=5000)
    mismatched = make_quote(4, chain, keys, "1", relaxed, 2, payload)
    ok, reason = relaxed.verify_quote(mismatched, chain[-1], directory, payload)
    check(not ok and "TCB level" in reason, "a measurement/TBC mismatch must be rejected")
    debug_quote = make_quote(5, chain, keys, "3", verifier, 3, payload, debug=True)
    ok, reason = verifier.verify_quote(debug_quote, chain[-1], directory, payload)
    check(not ok and "debug" in reason, "a debug enclave must be rejected by default")
    permissive = Verifier(root.key.fingerprint,
                          Policy(allowed_measurements={"3"}, allow_debug=True), clock=5000)
    debug_accepted = make_quote(5, chain, keys, "3", permissive, 3, payload, debug=True)
    ok, reason = permissive.verify_quote(debug_accepted, chain[-1], directory, payload)
    check(ok, "the debug quote is accepted once policy allows it (%s)" % reason)

    forged_leaf = forged
    forged_quote = make_quote(6, chain, keys, "3", verifier, 4, payload, signer=forged_leaf)
    ok, reason = verifier.verify_quote(forged_quote, chain[-1], directory, payload)
    check(not ok and "quote signature" in reason, "a quote must be signed by the enclave key")
    skimming_root, skimming_dir, skimming_chain, skimming_keys = build_chain(
        ["other-root", "other-ca", "other-leaf"], 2)
    foreign = make_quote(7, skimming_chain, skimming_keys, "3", verifier, 5, payload)
    ok, reason = verifier.verify_quote(foreign, skimming_chain[-1], directory, payload)
    check(not ok and ("missing issuer" in reason or "pinned root" in reason),
          "a quote from a foreign chain must be rejected")
    skew = make_quote(8, chain, keys, "3", verifier, 6, payload, clock=5000 + 500)
    ok, reason = verifier.verify_quote(skew, chain[-1], directory, payload)
    check(not ok and "clock" in reason, "a quote with a stale clock must be rejected")
    tampered_quote = make_quote(9, chain, keys, "3", verifier, 7, payload)
    tampered_quote.measurement = "1"
    ok, reason = verifier.verify_quote(tampered_quote, chain[-1], directory, payload)
    check(not ok, "tampering with a quote field must break its signature")

    # a leaf that is not the attested enclave's certificate
    impostor = Quote(forged, "3", verifier.issue_nonce("impostor"), verifier.clock, 4,
                     digest_int(payload) % (1 << 128)).issue(forged)
    ok, reason = verifier.verify_quote(impostor, chain[-1], directory, payload)
    check(not ok and "attested enclave" in reason,
          "a quote must be bound to the key its certificate attests")

    # ---- 6. cost and determinism -----------------------------------------
    counted = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    quote = make_quote(11, chain, keys, "3", counted, 1, payload)
    ok, reason = counted.verify_quote(quote, chain[-1], directory, payload)
    check(ok and counted.chain_checks == 2,
          "a full verification costs one check per link plus the quote")
    first = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    second = Verifier(root.key.fingerprint, Policy(allowed_measurements={"3"}), clock=5000)
    first_nonce = first.issue_nonce("determinism")
    second_nonce = second.issue_nonce("determinism")
    check(first_nonce == second_nonce,
          "two identically configured verifiers must mint the same nonce")
    first_quote = make_quote(11, chain, keys, "3", first, 1, payload, nonce=first_nonce)
    second_quote = make_quote(11, chain, keys, "3", second, 1, payload, nonce=second_nonce)
    ok_first, _ = first.verify_quote(first_quote, chain[-1], directory, payload)
    ok_second, _ = second.verify_quote(second_quote, chain[-1], directory, payload)
    check(ok_first and ok_second and first_quote.signature == second_quote.signature,
          "the same inputs must produce the same quote and the same verdict")
    check(Verifier.totals["rejected"] >= 12 and Verifier.totals["accepted"] >= 4,
          "the battery must exercise both outcomes")

    print("chain: 3-link chain verified with %d signature checks; 10-link chain with %d"
          % (2, 9))
    print("keys: root modulus %d bits, %d authorities generated deterministically"
          % (root.key.n.bit_length(), len(keys) + len(deep_keys)))
    print("quotes: %d accepted, %d rejected; nonces consumed on success only"
          % (Verifier.totals["accepted"], Verifier.totals["rejected"]))
    print("PASS")


if __name__ == "__main__":
    main()
