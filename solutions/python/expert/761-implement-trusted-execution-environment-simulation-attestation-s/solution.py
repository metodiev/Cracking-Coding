"""Task 761 - Trusted execution environment simulation (attestation sketch).  (Tier: Expert, Beyond - Security & Privacy)

A TEE attests *what code is running*: the enclave measures its own code, signs a quote containing
that measurement together with a fresh challenge (nonce) and a version, and a remote verifier checks
the quote against a policy before releasing anything to it.  The pieces here are:

    measure(code)            the enclave's identity: a hash of the loaded code
    Enclave.seal/unseal      sealed storage: the payload is encrypted with a key derived from the
                             measurement and authenticated, so another enclave (or the host) cannot
                             open it
    Enclave.quote            (measurement, challenge nonce, version, tag); the tag is a keyed hash
                             over exactly those fields
    Verifier.verify          freshness (challenge never seen before) -> version policy -> tag ->
                             allow-list -> revocation list, recording the nonce when it accepts
    session_key              binds a session key to the attested identity plus both nonces

Attestation and authorisation are deliberately two different steps, and the checks show why: a quote
that is *cryptographically perfect* for an enclave that is not on the allow-list (or has been
revoked) must still be rejected, while a replayed quote for an approved enclave must be rejected
because its challenge was already used.

Simplified versus real hardware attestation: there is no CPU signing key, no endorsement certificate
chain and no ECDSA - the "signature" is a hash-based MAC with a hard-coded *demo* key that stands for
the hardware-protected attestation key (nothing in this file is real key material, and no claim is
made that a MAC is a signature); there is no real encryption primitive, only a SHA-256-based stream
cipher, and the sealing key derivation is a toy.  The checks are still exact: a tampered field, a
quote forged with the wrong key, a stale nonce, a downgraded version, a revoked measurement and a
seal opened under a different measurement are all shown to be rejected, while the honest flows
succeed end to end - and a forced *nonce reuse* is shown to break the stream cipher
(ciphertext1 XOR ciphertext2 = plaintext1 XOR plaintext2), which is exactly why the nonce must be
unique per sealing operation.
"""
import hashlib

DEMO_KEY = b"demo-attestation-key-not-secret"
ATTACKER_KEY = b"attacker-controlled-key"
NONCE_LENGTH = 8
VERSION = 2


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class AttestationError(ValueError):
    """Raised when a sealed blob or a quote cannot be authenticated."""


def sha(*parts):
    """Hash of the given byte strings, joined with a delimiter that cannot occur ambiguously."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def measure(code):
    return sha(code).hex()[:32]


def mac(key, *parts):
    return sha(key, *parts).hex()[:32]


def keystream(key, measurement, nonce, length):
    """SHA-256 counter mode: a byte stream that depends on the key, the measurement and the nonce."""
    out = b""
    counter = 0
    while len(out) < length:
        out += sha(key, measurement.encode(), nonce, counter.to_bytes(4, "big"))
        counter += 1
    return out[:length]


def xor_bytes(left, right):
    return bytes(a ^ b for a, b in zip(left, right))


class Enclave:
    """A simulated enclave: it knows its own measurement and can seal, unseal and attest."""

    def __init__(self, code, key=DEMO_KEY, version=VERSION):
        self.code = code
        self.measurement = measure(code)
        self.key = key
        self.version = version
        self.seal_counter = 0

    def next_nonce(self):
        """Deterministic nonce source: unique per sealing operation (a real one uses a CSPRNG)."""
        self.seal_counter += 1
        return sha(self.key, b"nonce", self.seal_counter.to_bytes(4, "big"))[:NONCE_LENGTH]

    def seal(self, data, nonce=None):
        nonce = self.next_nonce() if nonce is None else nonce
        cipher = xor_bytes(data, keystream(self.key, self.measurement, nonce, len(data)))
        tag = mac(self.key, self.measurement.encode(), nonce, cipher)
        return {"measurement": self.measurement, "nonce": nonce, "cipher": cipher, "tag": tag}

    def unseal(self, blob):
        return unseal(blob, self.measurement, self.key)

    def quote(self, challenge):
        if len(challenge) < NONCE_LENGTH:
            raise AttestationError("the challenge must be at least %d bytes" % NONCE_LENGTH)
        tag = mac(self.key, self.measurement.encode(), challenge, bytes([self.version]))
        return {"measurement": self.measurement, "nonce": challenge,
                "version": self.version, "mac": tag}


def unseal(blob, measurement, key=DEMO_KEY):
    """Open a sealed blob, verifying first that it was sealed *for this measurement*."""
    if blob["measurement"] != measurement:
        raise AttestationError("the blob was sealed for a different enclave")
    expected = mac(key, measurement.encode(), blob["nonce"], blob["cipher"])
    if expected != blob["tag"]:
        raise AttestationError("the authentication tag does not match (host tampering?)")
    return xor_bytes(blob["cipher"], keystream(key, measurement, blob["nonce"], len(blob["cipher"])))


def forge_quote(measurement, challenge, version, key=ATTACKER_KEY):
    """What an attacker without the hardware key can produce: a quote with a wrong tag."""
    tag = mac(key, measurement.encode(), challenge, bytes([version]))
    return {"measurement": measurement, "nonce": challenge, "version": version, "mac": tag}


class Verifier:
    """Remote verifier: freshness, version policy, tag, allow-list and revocation list."""

    def __init__(self, allowed, revoked=(), min_version=1):
        self.allowed = set(allowed)
        self.revoked = set(revoked)
        self.min_version = min_version
        self.seen = set()
        self.accepted = 0
        self.rejected = {}

    def reject(self, reason):
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        return False, reason

    def verify(self, quote_value, challenge):
        if quote_value["nonce"] != challenge:
            return self.reject("challenge mismatch")
        if len(quote_value["nonce"]) < NONCE_LENGTH:
            return self.reject("challenge too short")
        if quote_value["nonce"] in self.seen:
            return self.reject("replayed challenge")
        if quote_value["version"] < self.min_version:
            return self.reject("version downgrade")
        expected = mac(DEMO_KEY, quote_value["measurement"].encode(), quote_value["nonce"],
                       bytes([quote_value["version"]]))
        if expected != quote_value["mac"]:
            return self.reject("invalid attestation mac")
        if quote_value["measurement"] in self.revoked:
            return self.reject("revoked enclave")
        if quote_value["measurement"] not in self.allowed:
            return self.reject("measurement not on the allow-list")
        self.seen.add(quote_value["nonce"])
        self.accepted += 1
        return True, "ok"


def session_key(measurement, enclave_nonce, client_nonce):
    """Channel binding: only a session with the attested identity derives this key."""
    return sha(DEMO_KEY, measurement.encode(), enclave_nonce, client_nonce).hex()[:32]


def main():
    good_code = b"load /srv/app; verify signatures; serve"
    evil_code = b"load /srv/app; exfiltrate keys"
    good = Enclave(good_code)
    evil = Enclave(evil_code)
    verifier = Verifier([good.measurement], revoked=(), min_version=1)

    # 1. the measurement is a deterministic identity of the code
    check(good.measurement == measure(good_code) and len(good.measurement) == 32,
          "the measurement must be reproducible and 32 hex characters")
    check(good.measurement != evil.measurement, "different code must measure differently")
    check(Enclave(good_code).measurement == good.measurement, "identical code measures identically")

    # 2. the honest flow: a fresh quote verifies and the session key agrees on both sides
    client_nonce = b"client-challenge"
    attestation = good.quote(client_nonce)
    ok, reason = verifier.verify(attestation, client_nonce)
    check(ok and reason == "ok", "an honest quote must verify: %s" % reason)
    server_key = session_key(attestation["measurement"], attestation["nonce"], client_nonce)
    client_key = session_key(good.measurement, attestation["nonce"], client_nonce)
    check(server_key == client_key, "both sides must derive the same bound session key")

    # 3. replay: the same quote (or the same challenge) may only be used once
    ok, reason = verifier.verify(attestation, client_nonce)
    check(not ok and reason == "replayed challenge", "a replayed quote must be rejected: %s" % reason)
    fresh = good.quote(b"second-challenge")
    check(verifier.verify(fresh, b"second-challenge")[0], "a new challenge must be accepted")
    check(verifier.verify(fresh, b"third-challenge")[1] == "challenge mismatch",
          "a quote must be bound to the challenge it answers")

    # 4. tampering with any attested field invalidates the tag
    tampered = dict(good.quote(b"tamper-challenge"))
    tampered["measurement"] = evil.measurement
    check(verifier.verify(tampered, b"tamper-challenge")[1] == "invalid attestation mac",
          "swapping the measurement into a valid quote must break the tag")
    bumped = dict(good.quote(b"bump-challenge"))
    bumped["version"] = bumped["version"] + 1
    check(verifier.verify(bumped, b"bump-challenge")[1] == "invalid attestation mac",
          "changing the version must break the tag")
    retagged = dict(good.quote(b"nonce-challenge"))
    retagged["nonce"] = b"other-nonce-8"
    check(verifier.verify(retagged, b"other-nonce-8")[1] == "invalid attestation mac",
          "changing the nonce must break the tag")

    # 5. an attacker without the hardware key cannot mint a quote for an approved measurement
    with_key = Verifier([good.measurement])
    forged = forge_quote(good.measurement, b"forged-challenge", VERSION)
    check(with_key.verify(forged, b"forged-challenge")[1] == "invalid attestation mac",
          "a quote forged with another key must be rejected")
    honest_forgery = forge_quote(good.measurement, b"forged-challenge-2", VERSION, key=DEMO_KEY)
    check(with_key.verify(honest_forgery, b"forged-challenge-2")[0],
          "only the holder of the demo key can satisfy the tag check (the key stands for hardware)")

    # 6. attestation and authorisation are separate: a valid quote for unapproved code fails policy
    policy = Verifier([good.measurement])
    evil_quote = evil.quote(b"evil-challenge")
    ok, reason = policy.verify(evil_quote, b"evil-challenge")
    check(not ok and reason == "measurement not on the allow-list",
          "cryptographically valid attestation of unapproved code must still be refused: %s" % reason)

    # 7. revocation and version policy
    revoked = Verifier([good.measurement, evil.measurement], revoked=[evil.measurement])
    check(revoked.verify(evil.quote(b"revoked-challenge"), b"revoked-challenge")[1] == "revoked enclave",
          "a revoked measurement must be refused even when it is allowed")
    strict = Verifier([good.measurement], min_version=VERSION + 1)
    check(strict.verify(good.quote(b"old-challenge"), b"old-challenge")[1] == "version downgrade",
          "an enclave below the minimum version must be refused")
    check(Verifier([good.measurement], min_version=VERSION).verify(
        good.quote(b"current-challenge"), b"current-challenge")[0],
        "the current version must be accepted")

    # 8. short challenges are refused before anything else can go wrong
    short_quote = {"measurement": good.measurement, "nonce": b"short", "version": VERSION,
                   "mac": mac(DEMO_KEY, good.measurement.encode(), b"short", bytes([VERSION]))}
    check(verifier.verify(short_quote, b"short")[1] == "challenge too short",
          "a challenge shorter than the policy minimum must be refused")
    try:
        good.quote(b"tiny")
        raise AssertionError("the enclave must refuse to attest a too short challenge")
    except AttestationError:
        pass

    # 9. sealed storage round-trips, and the tag protects it against the host
    for payload in (b"", b"k", b"session-keys-and-attestation-tokens" * 3):
        blob = good.seal(payload)
        check(good.unseal(blob) == payload, "seal/unseal must round-trip %d bytes" % len(payload))
        check(blob["cipher"] != payload or payload == b"", "the sealed payload must be encrypted")
    blob = good.seal(b"top secret material")
    flipped = dict(blob)
    flipped["cipher"] = blob["cipher"][:-1] + bytes([blob["cipher"][-1] ^ 0x01])
    try:
        unseal(flipped, good.measurement)
        raise AssertionError("a tampered ciphertext must be rejected")
    except AttestationError:
        pass

    # 10. sealing is bound to the measurement: another enclave (or the host) cannot open it
    other_blob = evil.seal(b"evil payload")
    try:
        good.unseal(other_blob)
        raise AssertionError("a blob sealed by another enclave must not open")
    except AttestationError:
        pass
    try:
        unseal(blob, evil.measurement)
        raise AssertionError("a blob must not open under a different measurement")
    except AttestationError:
        pass
    check(unseal(blob, good.measurement) == b"top secret material",
          "the rightful enclave must still open its own blob")

    # 11. nonce reuse breaks the stream cipher - the reason each sealing gets a fresh nonce
    reuse = b"fixed-nonce!!"
    first = xor_bytes(b"attack at dawn", keystream(DEMO_KEY, good.measurement, reuse, 14))
    second = xor_bytes(b"retreat at 09", keystream(DEMO_KEY, good.measurement, reuse, 14))
    check(xor_bytes(first, second) == xor_bytes(b"attack at dawn", b"retreat at 09"),
          "with a reused nonce the ciphertext XOR reveals the plaintext XOR")
    recovered = xor_bytes(second, xor_bytes(first, b"attack at dawn"))
    check(recovered == b"retreat at 09",
          "a known plaintext recovers the keystream, hence the other message entirely")
    check(good.seal(b"x")["nonce"] != good.seal(b"y")["nonce"],
          "every sealing operation must use a fresh nonce")

    # 12. channel binding: a man in the middle with different code derives a different key
    evil_quote = evil.quote(b"mitm-challenge")
    check(session_key(evil_quote["measurement"], evil_quote["nonce"], b"mitm-challenge") !=
          session_key(good.measurement, good.quote(b"mitm-challenge")["nonce"], b"mitm-challenge"),
          "a different attested identity must derive a different session key")
    check(policy.verify(evil_quote, b"mitm-challenge")[0] is False,
          "so the client's policy check stops the man in the middle")

    # 13. the verifier keeps an honest ledger of accepted and rejected quotes
    check(verifier.accepted >= 2, "the verifier must have accepted the honest quotes")
    check(sum(verifier.rejected.values()) >= 5, "and counted the rejections: %r" % (verifier.rejected,))
    check(set(verifier.rejected) <= {"replayed challenge", "challenge mismatch", "challenge too short",
                                     "version downgrade", "invalid attestation mac",
                                     "measurement not on the allow-list", "revoked enclave"},
          "only policy reasons may appear: %r" % (sorted(verifier.rejected),))
    check(len(verifier.seen) == verifier.accepted, "one recorded challenge per accepted quote")

    print("761 enclaves: approved=%s revoked=%s"
          % (good.measurement[:8], evil.measurement[:8]))
    print("761 verifier: accepted=%d rejected=%r" % (verifier.accepted, verifier.rejected))
    print("761 sealing: %d blobs round-tripped, nonce reuse recovers %r from a known plaintext"
          % (good.seal_counter, recovered))
    print("761 policy: unapproved valid quote -> %r, replay -> %r"
          % (policy.verify(evil.quote(b"policy-challenge"), b"policy-challenge")[1],
             "replayed challenge"))
    print("PASS")


if __name__ == "__main__":
    main()
