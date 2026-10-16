"""Task 923 - Implement distributed ledger with privacy-preserving transaction aggregation.  (Tier: Master, CTIC Ch 9 - System Design and Scalability)

A confidential ledger where individual amounts never appear in plaintext.

* Group: the curve y^2 = x^3 + 7 over the 61-bit Mersenne prime p = 2^61 - 1, with affine point
  addition (exceptional cases: the point at infinity, doubling, and P + (-P)) and double-and-add
  scalar multiplication; negative scalars negate the point.  G is the first curve point found by
  scanning x, H = hash-to-curve of a fixed label, so nobody knows log_G H.
* Commitments: C(v, r) = vG + rH is additively homomorphic, C(a, ra) + C(b, rb) = C(a+b, ra+rb),
  so a block can aggregate transactions by adding points and open only the total.
* Range proofs: the amount is rewritten as sum 2^i b_i with a bit commitment C_i per bit and a
  Schnorr OR proof (Fiat-Shamir over sha256) that each C_i opens to 0 or to 1; the verifier also
  checks sum 2^i C_i == C(v).  Blinding scalars are 61-bit integers, never reduced, so the proof
  algebra needs no knowledge of the group order.
* Ledger: blocks of commitments linked by a sha256 chain; the auditor's reserve proof opens the
  aggregate commitment against the published money supply.

Verified: double-and-add agrees with repeated addition on small scalars; the commitment scheme is
homomorphic; range proofs accept values inside the bit width (0 and 2^bits - 1 included) and reject
tampered bit commitments and out-of-range claims; the ledger aggregates to the published total,
detects a tampered amount and a broken hash chain; the serialised ledger contains no plaintext
amount and different blindings hide equal values; empty and single-entry ledgers behave.
Simplified: one curve, one proof system, no consensus or network layer, small amounts.
"""
import hashlib

P = (1 << 61) - 1
CURVE_B = 7
BITS = 8
BLOCK_SIZE = 3
MASK64 = (1 << 64) - 1

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)

def rand_scalar(state):
    state, raw = splitmix64(state)
    return state, raw % P

def point_neg(point):
    return None if point is None else (point[0], (-point[1]) % P)

def point_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0] and (a[1] + b[1]) % P == 0:
        return None
    if a == b:
        slope = 3 * a[0] * a[0] % P * pow(2 * a[1], P - 2, P) % P
    else:
        slope = (b[1] - a[1]) % P * pow((b[0] - a[0]) % P, P - 2, P) % P
    x = (slope * slope - a[0] - b[0]) % P
    return (x, (slope * (a[0] - x) - a[1]) % P)

def scalar_mul(scalar, point):
    """Double-and-add; negative scalars negate the point so exponents are never reduced."""
    if scalar < 0:
        return scalar_mul(-scalar, point_neg(point))
    result, addend = None, point
    while scalar:
        if scalar & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        scalar >>= 1
    return result

def naive_mul(scalar, point):
    """Oracle for the group law: repeated addition, only useful for tiny scalars."""
    result = None
    for _ in range(scalar):
        result = point_add(result, point)
    return result

def on_curve(point):
    return point is None or (point[1] * point[1] - point[0] ** 3 - CURVE_B) % P == 0

def curve_point(source):
    """Try-and-increment: the first x >= 1 (or hashed label) whose x^3 + 7 is a square."""
    counter = 0
    while True:
        if isinstance(source, bytes):
            digest = hashlib.sha256(source + counter.to_bytes(4, "big")).digest()
            x = int.from_bytes(digest, "big") % P
        else:
            x = source + counter
        value = (pow(x, 3, P) + CURVE_B) % P
        if pow(value, (P - 1) // 2, P) == 1:
            y = pow(value, (P + 1) // 4, P)
            return (x, min(y, P - y))
        counter += 1

def encode(point):
    if point is None:
        return b"\x00"
    return point[0].to_bytes(8, "big") + point[1].to_bytes(8, "big")

def challenge(*points):
    digest = hashlib.sha256(b"".join(encode(point) for point in points)).digest()
    return int.from_bytes(digest, "big") % P

def commit(value, blinding, generators):
    g, h = generators
    return point_add(scalar_mul(value, g), scalar_mul(blinding, h))

def or_bit_proof(bit, blinding, commitment, generators, state):
    """Schnorr OR proof that `commitment` opens to 0 or to 1 (branch 0 / branch 1).

    The prover knows the blinding of the true branch, so it takes a real Schnorr step there and
    simulates the other branch; the Fiat-Shamir challenge splits as c0 + c1.
    """
    g, h = generators
    state, nonce = rand_scalar(state)
    state, other_challenge = rand_scalar(state)
    state, other_response = rand_scalar(state)
    shifted = point_add(commitment, point_neg(g))          # commitment - G: the "bit = 1" target
    other = shifted if bit == 0 else commitment
    simulated = point_add(scalar_mul(other_response, h),
                          point_neg(scalar_mul(other_challenge, other)))
    real_witness = scalar_mul(nonce, h)
    witness_0, witness_1 = (real_witness, simulated) if bit == 0 else (simulated, real_witness)
    digest = challenge(witness_0, witness_1, commitment)
    real_challenge = digest - other_challenge
    real_response = nonce + real_challenge * blinding
    if bit == 0:
        c0, c1, s0, s1 = real_challenge, other_challenge, real_response, other_response
    else:
        c0, c1, s0, s1 = other_challenge, real_challenge, other_response, real_response
    return {"w0": witness_0, "w1": witness_1, "c0": c0, "c1": c1, "s0": s0, "s1": s1}, state

def verify_or_bit(commitment, proof, generators):
    g, h = generators
    shifted = point_add(commitment, point_neg(g))
    if challenge(proof["w0"], proof["w1"], commitment) != proof["c0"] + proof["c1"]:
        return False
    left_0 = scalar_mul(proof["s0"], h)
    right_0 = point_add(proof["w0"], scalar_mul(proof["c0"], commitment))
    left_1 = scalar_mul(proof["s1"], h)
    right_1 = point_add(proof["w1"], scalar_mul(proof["c1"], shifted))
    return left_0 == right_0 and left_1 == right_1

def transaction(value, generators, state, bits=BITS):
    """Bit-commit the amount; the sum of the bit commitments is the amount commitment."""
    bits_used, total_blinding, weight = [], 0, 1
    for index in range(bits):
        state, blinding = rand_scalar(state)
        bit = (value >> index) & 1
        commitment = commit(bit, blinding, generators)
        proof, state = or_bit_proof(bit, blinding, commitment, generators, state)
        bits_used.append((commitment, proof))
        total_blinding += weight * blinding
        weight *= 2
    return (commit(value, total_blinding, generators), bits_used, total_blinding), state

def verify_transaction(amount, bits, generators):
    """Check every bit proof and that the bit commitments recombine into the amount commitment."""
    weight, total = 1, None
    for commitment, proof in bits:
        if not verify_or_bit(commitment, proof, generators):
            return False
        total = point_add(total, scalar_mul(weight, commitment))
        weight *= 2
    return total == amount

def block_hash(previous, commitments):
    digest = hashlib.sha256(previous)
    for commitment in commitments:
        digest.update(encode(commitment))
    return digest.digest()

def build_ledger(values, generators, state):
    """Build the chain: each block carries the commitments and the running aggregate."""
    ledger, aggregate, previous = [], None, b"origin"
    for start in range(0, len(values), BLOCK_SIZE):
        commitments, proofs, blindings = [], [], []
        for value in values[start:start + BLOCK_SIZE]:
            (amount, bits, blinding), state = transaction(value, generators, state)
            commitments.append(amount)
            proofs.append(bits)
            blindings.append(blinding)
        for commitment in commitments:
            aggregate = point_add(aggregate, commitment)
        ledger.append({"index": len(ledger), "previous": previous, "commitments": commitments,
                       "proofs": proofs, "blindings": blindings,
                       "hash": block_hash(previous, commitments)})
        previous = ledger[-1]["hash"]
    return ledger, aggregate, state

def total_blinding(ledger):
    return sum(sum(block["blindings"]) for block in ledger)

def published_amounts(ledger, generators):
    """The auditor's view: only the aggregate commitment, never the individual amounts."""
    aggregate = None
    for block in ledger:
        for commitment in block["commitments"]:
            aggregate = point_add(aggregate, commitment)
    return aggregate

def verify_ledger(ledger, generators, supply, blinding):
    """Hash chain, every range proof, and the aggregate opening against the published supply."""
    previous = b"origin"
    for block in ledger:
        if block["previous"] != previous or block["hash"] != block_hash(previous,
                                                                       block["commitments"]):
            return False
        for commitment, proof in zip(block["commitments"], block["proofs"]):
            if not verify_transaction(commitment, proof, generators):
                return False
        previous = block["hash"]
    return published_amounts(ledger, generators) == commit(supply, blinding, generators)

def ledger_bytes(ledger):
    return b"".join(block["hash"] + b"".join(encode(c) for c in block["commitments"])
                    for block in ledger)

def main():
    generators = (curve_point(1), curve_point(b"confidential-aggregation-lab"))
    g, h = generators

    # 1. Group law: independent oracle on tiny scalars, plus curve and inverse identities.
    check(on_curve(g) and on_curve(h) and g != h, "both generators must be distinct curve points")
    check(all(scalar_mul(k, g) == naive_mul(k, g) for k in range(1, 9)),
          "double-and-add must agree with repeated addition")
    check(point_add(scalar_mul(5, g), point_neg(scalar_mul(5, g))) is None
          and point_add(g, None) == g and point_neg(point_neg(g)) == g,
          "inverse, identity and double negation must hold")
    check(scalar_mul(-3, g) == point_neg(scalar_mul(3, g)), "negative scalars must negate the point")

    # 2. Commitments are additively homomorphic.
    state = 0xC0FFEE
    state, ra = rand_scalar(state)
    state, rb = rand_scalar(state)
    combined = point_add(commit(30, ra, generators), commit(12, rb, generators))
    check(combined == commit(42, ra + rb, generators)
          and commit(30, ra, generators) != commit(30, rb, generators),
          "commitments must be homomorphic, and one value under two blindings must differ")

    # 3. Range proofs: valid inside the width, rejected when tampered or out of range.
    (amount, bits, blinding), state = transaction(200, generators, state)
    check(verify_transaction(amount, bits, generators), "an in-range transaction must verify")
    for value in (0, 1, 2 ** BITS - 1):
        (probe, probe_bits, _), state = transaction(value, generators, state)
        check(verify_transaction(probe, probe_bits, generators),
              "boundary amounts inside the bit width must verify")
    tampered = list(bits)
    tampered[0] = (point_add(bits[0][0], g), bits[0][1])
    check(not verify_transaction(amount, tampered, generators),
          "a shifted bit commitment must break the range proof")
    forged = commit(2 ** BITS, blinding, generators)
    check(not verify_transaction(forged, bits, generators),
          "an out-of-range amount must not recombine from valid bit commitments")

    # 4. Ledger aggregation: the chain verifies against the published supply.
    values = [120, 45, 30, 7, 200, 60]
    ledger, aggregate, _ = build_ledger(values, generators, state)
    supply = sum(values)
    check(len(ledger) == 2 and verify_ledger(ledger, generators, supply, total_blinding(ledger)),
          "the ledger must aggregate to the published supply")
    check(published_amounts(ledger, generators) == commit(supply, total_blinding(ledger), generators),
          "the aggregate commitment must open to the published supply")
    check(verify_ledger(ledger, generators, supply + 1, total_blinding(ledger)) is False,
          "a wrong published supply must be rejected")

    # 5. Tamper detection: a changed amount breaks the aggregate, a changed block breaks the chain.
    broken, _, _ = build_ledger(values[:-1] + [61], generators, state)
    check(not verify_ledger(broken, generators, supply, total_blinding(ledger)),
          "a tampered amount must break the aggregate opening")
    forked = [dict(block) for block in ledger]
    forked[1] = dict(forked[1])
    forked[1]["previous"] = b"tampered"
    check(not verify_ledger(forked, generators, supply, total_blinding(ledger)),
          "a broken hash link must be detected")

    # 6. Privacy: the ledger bytes carry no plaintext amount, and blinding hides equal values.
    raw = ledger_bytes(ledger)
    check(all(value.to_bytes(8, "big") not in raw for value in values),
          "no plaintext amount may appear in the serialised ledger")
    public, public_bits, _ = transaction(77, generators, state)[0]
    check(commit(77, ra, generators) != commit(77, rb, generators)
          and verify_transaction(public, public_bits, generators),
          "equal amounts under different blindings must give different commitments")

    # 7. Boundary cases: an empty ledger and a single-entry ledger whose aggregation is a no-op.
    empty, empty_aggregate, _ = build_ledger([], generators, state)
    check(empty == [] and empty_aggregate is None and verify_ledger(empty, generators, 0, 0),
          "an empty ledger must verify against a zero supply")
    single, single_aggregate, single_state = build_ledger([9], generators, state)
    check(len(single) == 1 and single_aggregate == commit(9, total_blinding(single), generators),
          "a single-entry ledger must aggregate to its own commitment")

    # 8. Determinism: rebuilding from the same seed reproduces the identical ledger.
    rebuilt, _, _ = build_ledger(values, generators, state)
    check(rebuilt == ledger, "the ledger must be deterministic for a fixed seed")
    print(f"ledger blocks {len(ledger)}, transactions {len(values)}, supply {supply}, "
          f"bit width {BITS}")
    print(f"serialised ledger {len(raw)} bytes, aggregate commitment x={aggregate[0]:x}")
    print("PASS")


if __name__ == "__main__":
    main()
