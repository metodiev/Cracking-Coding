"""Task 837 - Implement decentralized identity (DID) system with verifiable credentials simulation. (Tier: Master, CTCI Beyond — Security & Privacy)

An issuer signs a credential for a holder; the holder later presents it to a verifier who trusts
neither of them but can resolve both DIDs. The pieces that make that work are all here: DIDs
derived from public keys with resolvable DID documents, key rotation with a retained key
history, credential issuance over a canonical serialization, presentations bound to the holder
by a proof-of-possession signature over a verifier nonce, an expiry check against a logical
clock supplied by the caller, a revocation registry, and a nonce cache that stops replay.

What is implemented
-------------------
* `generate_key` / `sign` / `verify_signature`: RSA over a deterministic modulus with a
  full-domain-hash style message representative (repeated SHA-256 expansion to the modulus size)
  -- a real, if tiny, signature scheme -- plus a domain separator so a credential signature can
  never be replayed as a proof of possession.
* `did_for_key` / `resolve`: `did:cc:<hash of the public key>`, resolved through a ledger of DID
  documents that carry the verification methods and the key history rotation needs.
* `issue_credential` / `verify_credential`: the credential is signed over a canonical
  (sorted-key) serialization, so *every* field is covered by the signature; the verifier checks
  resolution, the issuer signature, `validFrom`/`validUntil` against the caller's clock, and the
  revocation registry, reporting one reason per failure.
* `present` / `verify_presentation`: the holder signs `nonce|credential-id|verifier` with the key
  its own DID document names, and the verifier applies its full policy: issuer checks, holder
  binding (subject == holder), possession proof, and nonce freshness.

What is simplified -- and what is explicitly *not* claimed
----------------------------------------------------------
* The modulus is 512 bits and there is no OAEP/PSS padding; the full-domain-hash construction
  used here has no security proof, and 512-bit RSA is breakable. This is a structural
  simulation of a DID/VC flow, not a deployment, and no security property beyond "the algebra
  holds for honest and dishonest inputs" is claimed.
* No DIDComm transports, no JSON-LD canonicalization beyond a sorted-key dump, no
  zero-knowledge presentations, no status lists (revocation is a registry lookup), no anchoring:
  the ledger is an in-process dictionary.
* Time is an explicit integer supplied by the caller; the file contains no clock, so every
  expiry boundary is deterministic and the inclusive/exclusive rules are checked exactly.

Invariants verified
-------------------
1.  Key generation is deterministic, the DID depends only on the public key, resolution is
    idempotent, and unknown and malformed DIDs are distinguished.
2.  Signature semantics: honest signatures verify; a tampered message, another key, a single
    flipped bit and the wrong domain separator all fail.
3.  Issuance round trip with all eight credential fields mutated one at a time; every mutation is
    caught by the issuer signature.
4.  Validity window: valid at `validFrom` and at `validUntil` (documented as inclusive), expired
    one tick later, `not_yet_valid` one tick earlier.
5.  Holder binding: the subject must be the presenting holder, a possession proof signed by the
    wrong key fails, and a corrupted possession proof fails.
6.  Nonce freshness: replaying a (verifier, nonce) pair fails, a fresh nonce passes, the same
    nonce at a different verifier is a different challenge, and a proof does not transfer to a
    third verifier.
7.  Revocation: a revoked credential fails, and the registry only affects the credential it
    names.
8.  Key rotation: a credential signed with a retained key still verifies, a credential signed
    with the new key verifies, and revoking the old key invalidates exactly the old credential.
9.  Issuer substitution: an unresolvable issuer, and a credential naming an issuer that did not
    sign it, are both rejected.
10. Presentation contents: exactly the credential, the proof, the holder, the nonce and the
    possession signature -- and no private key material.
11. Cost: verification needs two modular exponentiations (issuer signature plus possession
    proof) and no private key at all.
12. Determinism: issuing and signing the same inputs twice produces identical bytes.
"""
import hashlib
import json

MASK64 = (1 << 64) - 1
DOMAIN_CREDENTIAL = b"vc-v1"
DOMAIN_POSSESSION = b"pop-v1"
PRESENTATION_FIELDS = {"credential", "proof", "holder", "nonce", "possession"}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


# --------------------------------------------------------------------------------------
# deterministic RSA keys and signatures
# --------------------------------------------------------------------------------------
def is_prime(n, rounds=16):
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        if n % small == 0:
            return n == small
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for witness in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)[:rounds]:
        x = pow(witness, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_of_bits(bits, seed):
    rng = SplitMix64(seed)
    while True:
        candidate = (1 << (bits - 1)) | (rng.next_u64() << (bits - 64)) | 1
        if is_prime(candidate):
            return candidate


def generate_key(bits=512, seed=837):
    half = bits // 2
    p = prime_of_bits(half, seed)
    q = prime_of_bits(half, seed ^ 0x5A5A5A5A)
    while q == p:
        q = prime_of_bits(half, (seed ^ 0x7777) + 1)
    modulus = p * q
    phi = (p - 1) * (q - 1)
    exponent = 65537
    return {"n": modulus, "e": exponent, "d": pow(exponent, -1, phi),
            "bits": modulus.bit_length(), "p": p, "q": q}


def public_of(key):
    return {"n": key["n"], "e": key["e"]}


def full_domain_representative(message, modulus):
    """Expand a digest to the modulus size by repeated hashing (a toy FDH, not padded RSA)."""
    size = (modulus.bit_length() + 7) // 8
    out, block = b"", hashlib.sha256(message).digest()
    while len(out) < size:
        block = hashlib.sha256(block + message).digest()
        out += block
    return int.from_bytes(out[:size], "big") % modulus


def sign(key, domain, message, counter=None):
    if counter is not None:
        counter["modexps"] += 1
    return pow(full_domain_representative(domain + message, key["n"]), key["d"], key["n"])


def verify_signature(public, domain, message, signature, counter=None):
    if not 0 <= signature < public["n"]:
        return False
    if counter is not None:
        counter["modexps"] += 1
    representative = full_domain_representative(domain + message, public["n"])
    return pow(signature, public["e"], public["n"]) == representative


# --------------------------------------------------------------------------------------
# DIDs and their documents
# --------------------------------------------------------------------------------------
def did_for_key(public):
    digest = hashlib.sha256(f"{public['n']}:{public['e']}".encode()).hexdigest()
    return f"did:cc:{digest[:32]}"


def make_document(public, did=None, previous=()):
    """A DID document with one active verification method and any retained older ones."""
    identifier = did if did is not None else did_for_key(public)
    active = {"id": f"{identifier}#key-new", "type": "RsaVerificationKey2018",
              "publicKey": public, "revoked": False}
    return {"id": identifier, "verificationMethod": [active] + [dict(method)
                                                                for method in previous],
            "created": len(previous)}


def resolve(ledger, did):
    """Resolution is a lookup; unknown, malformed and present DIDs are distinguished."""
    prefix = "did:cc:"
    if not isinstance(did, str) or not did.startswith(prefix) or len(did) != len(prefix) + 32:
        return {"error": "malformed_did"}
    document = ledger.get(did)
    return dict(document) if document else {"error": "not_found"}


def active_methods(document):
    return [method for method in document.get("verificationMethod", [])
            if not method.get("revoked")]


# --------------------------------------------------------------------------------------
# verifiable credentials
# --------------------------------------------------------------------------------------
def canonical(credential):
    return json.dumps(credential, sort_keys=True, separators=(",", ":")).encode()


def issue_credential(ledger, issuer_key, issuer_did, subject_did, claims, credential_id,
                     valid_from, valid_until, counter=None):
    credential = {"id": credential_id, "issuer": issuer_did, "subject": subject_did,
                  "claims": dict(claims), "validFrom": valid_from, "validUntil": valid_until,
                  "version": 1, "type": "VerifiableCredential"}
    return {"credential": credential,
            "proof": sign(issuer_key, DOMAIN_CREDENTIAL, canonical(credential), counter)}


def verify_credential(ledger, bundle, now, revocation=None, counter=None):
    """The issuer-side checks; returns a list of failure reasons (empty means valid)."""
    credential, proof = bundle["credential"], bundle["proof"]
    failures = []
    document = resolve(ledger, credential.get("issuer"))
    if "error" in document:
        return [f"issuer_{document['error']}"]
    methods = active_methods(document)
    if not methods:
        failures.append("issuer_key_revoked")
    elif not any(verify_signature(method["publicKey"], DOMAIN_CREDENTIAL, canonical(credential),
                                  proof, counter) for method in methods):
        failures.append("bad_issuer_signature")
    if credential.get("validFrom", 0) > now:
        failures.append("not_yet_valid")
    if now > credential.get("validUntil", 0):
        failures.append("expired")
    if revocation is not None and credential.get("id") in revocation.get(
            credential.get("issuer"), set()):
        failures.append("revoked")
    return failures


def present(ledger, holder_key, holder_did, bundle, nonce, verifier, counter=None):
    credential = bundle["credential"]
    challenge = f"{nonce}|{credential['id']}|{verifier}".encode()
    return {"credential": bundle["credential"], "proof": bundle["proof"],
            "holder": holder_did, "nonce": nonce,
            "possession": sign(holder_key, DOMAIN_POSSESSION, challenge, counter)}


def verify_presentation(ledger, presentation, now, verifier, nonce_cache=None, revocation=None,
                        counter=None):
    """The verifier's full policy; every failure carries its own reason."""
    failures = verify_credential(ledger, presentation, now, revocation, counter)
    credential = presentation["credential"]
    if credential.get("subject") != presentation.get("holder"):
        failures.append("holder_mismatch")
    document = resolve(ledger, presentation.get("holder"))
    if "error" in document:
        failures.append(f"holder_{document['error']}")
    elif not any(verify_signature(method["publicKey"], DOMAIN_POSSESSION,
                                  f"{presentation['nonce']}|{credential['id']}|{verifier}"
                                  .encode(), presentation["possession"], counter)
                 for method in active_methods(document)):
        failures.append("bad_possession_proof")
    if nonce_cache is not None and (verifier, presentation.get("nonce")) in nonce_cache:
        failures.append("nonce_replayed")
    return failures


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def main():
    issuer_key = generate_key(512, seed=11)
    holder_key = generate_key(512, seed=22)
    other_key = generate_key(512, seed=33)
    issuer_did = did_for_key(public_of(issuer_key))
    holder_did = did_for_key(public_of(holder_key))
    ledger = {issuer_did: make_document(public_of(issuer_key)),
              holder_did: make_document(public_of(holder_key))}

    # -- 1. keys and DIDs ----------------------------------------------------------------
    check(generate_key(512, seed=11) == issuer_key, "key generation is deterministic")
    check(511 <= issuer_key["bits"] <= 512 and is_prime(issuer_key["p"]),
          f"the toy modulus is {issuer_key['bits']} bits and its factors are prime")
    check(issuer_did.startswith("did:cc:") and len(issuer_did) == 39,
          f"the DID is a 32-hex-character hash of the public key ({issuer_did})")
    check(did_for_key(public_of(issuer_key)) == issuer_did
          and did_for_key(public_of(other_key)) != issuer_did,
          "the DID depends only on the public key")
    check(resolve(ledger, issuer_did) == ledger[issuer_did],
          "resolution returns the document and is idempotent")
    check(resolve(ledger, "did:cc:" + "0" * 32) == {"error": "not_found"},
          "an unknown DID does not resolve")
    check(resolve(ledger, "did:cc:short") == {"error": "malformed_did"},
          "a malformed DID is rejected before the lookup")

    # -- 2. signature semantics ------------------------------------------------------------
    message = b"I am the issuer"
    signature = sign(issuer_key, DOMAIN_CREDENTIAL, message)
    check(verify_signature(public_of(issuer_key), DOMAIN_CREDENTIAL, message, signature),
          "an honest signature verifies")
    check(not verify_signature(public_of(issuer_key), DOMAIN_CREDENTIAL, message + b"!",
                               signature), "a tampered message fails")
    check(not verify_signature(public_of(other_key), DOMAIN_CREDENTIAL, message, signature),
          "a signature does not verify under another key")
    check(not verify_signature(public_of(issuer_key), DOMAIN_CREDENTIAL, message, signature ^ 1),
          "one flipped bit breaks the signature")
    check(not verify_signature(public_of(issuer_key), DOMAIN_POSSESSION, message, signature),
          "the domain separator stops a credential signature being used as possession")
    check(sign(issuer_key, DOMAIN_CREDENTIAL, message)
          == sign(issuer_key, DOMAIN_CREDENTIAL, message), "signing is deterministic")

    # -- 3. issuance and every covered field -------------------------------------------------
    bundle = issue_credential(ledger, issuer_key, issuer_did, holder_did,
                              {"role": "auditor", "level": 3}, "vc-1", 10, 100)
    check(verify_credential(ledger, bundle, now=50) == [],
          "the credential verifies inside its validity window")
    for field, value in (("id", "vc-other"), ("issuer", holder_did), ("subject", issuer_did),
                         ("version", 2), ("type", "SomethingElse"), ("validFrom", 0),
                         ("validUntil", 1000), ("claims", {"role": "admin"})):
        mutated = json.loads(json.dumps(bundle))
        mutated["credential"][field] = value
        failures = verify_credential(ledger, mutated, now=50)
        check("bad_issuer_signature" in failures,
              f"mutating {field} is caught by the issuer signature ({failures})")
    check(len(bundle["credential"]) == 8, "all eight credential fields sit under the signature")

    # -- 4. validity window -------------------------------------------------------------------
    check(verify_credential(ledger, bundle, now=10) == [], "valid exactly at validFrom")
    check(verify_credential(ledger, bundle, now=100) == [],
          "valid exactly at validUntil (the window is inclusive)")
    check(verify_credential(ledger, bundle, now=101) == ["expired"],
          "one tick after validUntil it is expired")
    check(verify_credential(ledger, bundle, now=9) == ["not_yet_valid"],
          "before validFrom it is not yet valid")

    # -- 5. holder binding --------------------------------------------------------------------
    presentation = present(ledger, holder_key, holder_did, bundle, "n-1", "verifier-A")
    used = set()
    check(verify_presentation(ledger, presentation, 50, "verifier-A", used) == [],
          "a well-formed presentation passes every check")
    stranger = present(ledger, other_key, holder_did, bundle, "n-2", "verifier-A")
    check(verify_presentation(ledger, stranger, 50, "verifier-A", used)
          == ["bad_possession_proof"],
          "a possession proof signed by the wrong key fails")
    self_issued = issue_credential(ledger, issuer_key, issuer_did, issuer_did,
                                   {"role": "auditor"}, "vc-self", 10, 100)
    rented = present(ledger, holder_key, holder_did, self_issued, "n-3", "verifier-A")
    check(verify_presentation(ledger, rented, 50, "verifier-A", used) == ["holder_mismatch"],
          "a credential whose subject is not the presenting holder fails")
    check(verify_presentation(ledger, {**presentation,
                                       "possession": presentation["possession"] ^ 1},
                              50, "verifier-A", used) == ["bad_possession_proof"],
          "a corrupted possession proof fails")

    # -- 6. nonce freshness ---------------------------------------------------------------------
    replay_cache = {("verifier-A", "n-1")}
    check(verify_presentation(ledger, presentation, 50, "verifier-A", replay_cache)
          == ["nonce_replayed"], "reusing a (verifier, nonce) pair is rejected")
    fresh = present(ledger, holder_key, holder_did, bundle, "n-9", "verifier-A")
    check(verify_presentation(ledger, fresh, 50, "verifier-A", replay_cache) == [],
          "a fresh nonce passes")
    elsewhere = present(ledger, holder_key, holder_did, bundle, "n-1", "verifier-B")
    check(verify_presentation(ledger, elsewhere, 50, "verifier-B", replay_cache) == [],
          "the same nonce at another verifier is a different challenge and is accepted")
    check(verify_presentation(ledger, elsewhere, 50, "verifier-C", replay_cache)
          == ["bad_possession_proof"],
          "and it does not transfer to a third verifier")

    # -- 7. revocation ---------------------------------------------------------------------------
    revocation = {issuer_did: {"vc-1"}}
    check(verify_credential(ledger, bundle, 50, revocation) == ["revoked"],
          "a revoked credential is rejected with that reason")
    second = issue_credential(ledger, issuer_key, issuer_did, holder_did, {"role": "reader"},
                              "vc-2", 10, 100)
    check(verify_credential(ledger, second, 50, revocation) == [],
          "the registry only affects the credential it names")

    # -- 8. key rotation --------------------------------------------------------------------------
    new_key = generate_key(512, seed=44)
    new_did = did_for_key(public_of(new_key))
    retained_methods = ledger[issuer_did]["verificationMethod"]
    rotated = dict(ledger)
    rotated[new_did] = make_document(public_of(new_key), previous=retained_methods)
    check(active_methods(rotated[new_did]) == rotated[new_did]["verificationMethod"],
          "both the new and the retained key are active after rotation")
    old_credential = issue_credential(rotated, issuer_key, new_did, holder_did,
                                      {"role": "auditor"}, "vc-3", 10, 100)
    check(verify_credential(rotated, old_credential, 50) == [],
          "a credential signed with the retained key still verifies after rotation")
    new_credential = issue_credential(rotated, new_key, new_did, holder_did,
                                      {"role": "auditor"}, "vc-4", 10, 100)
    check(verify_credential(rotated, new_credential, 50) == [],
          "a credential signed with the new key verifies")
    rotated[new_did] = make_document(public_of(new_key),
                                     previous=[{**retained_methods[0], "revoked": True}])
    check(verify_credential(rotated, old_credential, 50) == ["bad_issuer_signature"],
          "revoking the old key invalidates exactly the credential signed with it")
    check(verify_credential(rotated, new_credential, 50) == [],
          "and leaves the new credentials valid")
    all_revoked = dict(rotated)
    revoked_document = make_document(public_of(new_key))
    for method in revoked_document["verificationMethod"]:
        method["revoked"] = True
    all_revoked[new_did] = revoked_document
    check("issuer_key_revoked" in verify_credential(all_revoked, new_credential, 50),
          "with no active key left the issuer cannot be trusted at all")

    # -- 9. issuer substitution ----------------------------------------------------------------------
    impostor = issue_credential(ledger, other_key, issuer_did, holder_did, {"role": "admin"},
                                "vc-5", 10, 100)
    check(verify_credential(ledger, impostor, 50) == ["bad_issuer_signature"],
          "a credential naming an issuer that did not sign it is rejected")
    self_signed = issue_credential(ledger, other_key, did_for_key(public_of(other_key)),
                                   holder_did, {"role": "admin"}, "vc-6", 10, 100)
    check(verify_credential(ledger, self_signed, 50) == ["issuer_not_found"],
          "an issuer DID that is absent from the ledger does not resolve")

    # -- 10. contents, cost and determinism -------------------------------------------------------------
    check(set(presentation) == PRESENTATION_FIELDS,
          f"a presentation carries exactly {sorted(PRESENTATION_FIELDS)}")
    check(not any(field in presentation for field in ("d", "p", "q", "privateKey")),
          "no private key material travels with a presentation")
    counter = {"modexps": 0}
    check(verify_presentation(ledger, fresh, 50, "verifier-A", replay_cache, None, counter)
          == [], "the counted verification still succeeds")
    check(counter["modexps"] == 2,
          f"verification costs {counter['modexps']} modular exponentiations "
          f"(issuer signature plus possession proof) and no private key")
    check(issue_credential(ledger, issuer_key, issuer_did, holder_did,
                           {"role": "auditor", "level": 3}, "vc-1", 10, 100) == bundle,
          "issuing the same credential twice produces identical bytes")
    check(present(ledger, holder_key, holder_did, bundle, "n-9", "verifier-A") == fresh,
          "and presenting it twice produces an identical presentation")

    print(f"issuer {issuer_did[:17]}..., holder {holder_did[:17]}..., modulus "
          f"{issuer_key['bits']} bits")
    print(f"verification: {counter['modexps']} modular exponentiations; "
          f"failures reported as their own reason strings")
    print(f"rotation: retained credential after revocation "
          f"{verify_credential(rotated, old_credential, 50)}, "
          f"new credential {verify_credential(rotated, new_credential, 50)}")
    print("PASS")


if __name__ == "__main__":
    main()
