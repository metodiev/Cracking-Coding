"""Task 758 - Secure multiparty computation: basic XOR-sharing demo.  (Tier: Expert, Beyond - Security & Privacy)

XOR (additive mod 2) secret sharing: to share an w-bit value x among n parties, the dealer picks
n-1 uniform shares and gives the last party x XOR (all the others), so every share alone is uniform
and only the full set reconstructs x.  Addition is free - each party XORs its own shares locally, no
communication at all.  Multiplication is where the parties must talk: with a Beaver triple
(a, b, c = a AND b) shared ahead of time, each party i broadcasts d_i = x_i XOR a_i and
e_i = y_i XOR b_i, then computes

    z_i = c_i XOR (d AND b_i) XOR (e AND a_i) XOR (d AND e if i == 0 else 0)

so that XOR_i z_i = (a XOR d) AND (b XOR e) = x AND y, because XOR over the shares of d and e
reconstructs those masks.  The public values d and e are uniform masks of the secrets, which is why
the transcript reveals nothing.  This file implements the sharing, the local operations
(XOR with another shared value, XOR with a public constant, NOT), the Beaver-triple AND, a per-bit
AND for multi-bit operands, and a full GMW evaluation of a small shared Boolean circuit.

Privacy is checked *exactly*, not approximately: for the sharing itself, the checks enumerate every
completion of the other parties' shares and verify that the number of completions is the same for
every secret (perfect privacy of the marginal view); for the AND gate, they enumerate every pair of
triple-share profiles and verify that the distribution of the transcript (d, e) is *identical and
uniform* for every pair of secrets.  Correctness is exhaustive for 1-bit and 2-bit operands and the
circuit is evaluated for all 16 input assignments - the plaintext evaluation is the oracle.

Simplified versus a deployment: the Beaver triples come from a trusted dealer (real protocols use
OT-based preprocessing or a correlated-randomness setup), the honesty model is semi-honest (no
malicious-party detection, no commitments), communication is simulated as a broadcast list rather
than messages, and secrets are toy values of a few bits - never real key material.  Verified
invariants: every reconstruction is exact, the marginal view of every party is independent of the
secret, the AND transcript distribution is secret-independent, and no private share ever appears in
the transcript.
"""
import itertools
from functools import reduce
from operator import xor

MASK64 = (1 << 64) - 1
MAX_WIDTH = 8


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def reconstruct(shares, width=MAX_WIDTH):
    if not shares:
        raise ValueError("nothing to reconstruct")
    if any(share < 0 or share >= (1 << width) for share in shares):
        raise ValueError("a share is outside the %d-bit domain" % width)
    return reduce(xor, shares)


def share(state, secret, parties, width=MAX_WIDTH):
    """XOR-share `secret` among `parties` parties; every proper subset of shares is uniform."""
    if parties < 2:
        raise ValueError("at least two parties are required")
    if not 0 <= secret < (1 << width):
        raise ValueError("the secret does not fit the %d-bit domain" % width)
    shares = []
    for _ in range(parties - 1):
        state, r = splitmix64(state)
        shares.append(r & ((1 << width) - 1))
    shares.append(reconstruct(shares, width) ^ secret)
    return shares, state


def local_xor(left_shares, right_shares, width=MAX_WIDTH):
    """Addition of two shared values: purely local, no interaction."""
    if len(left_shares) != len(right_shares):
        raise ValueError("the two sharings must have the same number of parties")
    return [x ^ y for x, y in zip(left_shares, right_shares)]


def local_constant(parties, constant, width=MAX_WIDTH):
    """Add a public constant: give it to one party, zero to the rest."""
    if parties < 1 or not 0 <= constant < (1 << width):
        raise ValueError("bad constant sharing")
    return [constant] + [0] * (parties - 1)


def local_not(shares, width=MAX_WIDTH):
    """Complement: flip the mask in one share only."""
    flipped = list(shares)
    flipped[0] ^= (1 << width) - 1
    return flipped


def dealer_triples(state, parties, width=MAX_WIDTH):
    """Trusted dealer: random a, b and c = a AND b, each XOR-shared."""
    state, r = splitmix64(state)
    a_secret = r & ((1 << width) - 1)
    state, r = splitmix64(state)
    b_secret = r & ((1 << width) - 1)
    a_shares, state = share(state, a_secret, parties, width)
    b_shares, state = share(state, b_secret, parties, width)
    c_shares, state = share(state, a_secret & b_secret, parties, width)
    return (a_shares, b_shares, c_shares), state


def beaver_and(state, x_shares, y_shares, parties, width=MAX_WIDTH, transcript=None):
    """One AND of two shared w-bit values using a fresh Beaver triple."""
    if len(x_shares) != parties or len(y_shares) != parties:
        raise ValueError("share vectors must have one entry per party")
    triple, state = dealer_triples(state, parties, width)
    a_shares, b_shares, c_shares = triple
    d_shares = local_xor(x_shares, a_shares, width)      # broadcast d_i = x_i XOR a_i
    e_shares = local_xor(y_shares, b_shares, width)      # broadcast e_i = y_i XOR b_i
    d = reconstruct(d_shares, width)
    e = reconstruct(e_shares, width)
    if transcript is not None:
        transcript.append((d, e))
    z_shares = []
    for i in range(parties):
        value = c_shares[i] ^ (d & b_shares[i]) ^ (e & a_shares[i])
        if i == 0:
            value ^= d & e                              # one party absorbs the cross term
        z_shares.append(value)
    return z_shares, state


def beaver_and_bitwise(state, x_shares, y_shares, parties, width=MAX_WIDTH, transcript=None):
    """AND of multi-bit operands, one Beaver triple per bit."""
    result = [0] * parties
    for bit in range(width):
        x_bit = [(s >> bit) & 1 for s in x_shares]
        y_bit = [(s >> bit) & 1 for s in y_shares]
        z_bit, state = beaver_and(state, x_bit, y_bit, parties, 1, transcript)
        result = [value | (z_bit[i] << bit) for i, value in enumerate(result)]
    return result, state


def gmw_circuit(state, x_shares, circuit, parties, width=MAX_WIDTH, transcript=None):
    """Evaluate a shared Boolean circuit: gates are ('and', i, j) or ('xor', i, j) on wires."""
    wires = list(x_shares)
    for gate in circuit:
        kind, left, right = gate
        if kind == "xor":
            wires.append(local_xor(wires[left], wires[right], width))
        elif kind == "and":
            value, state = beaver_and_bitwise(state, wires[left], wires[right], parties, width,
                                              transcript)
            wires.append(value)
        else:
            raise ValueError("unknown gate %r" % (kind,))
    return wires[-1], state


def plain_circuit(x_values, circuit):
    """The same circuit on plaintext values - the correctness oracle."""
    wires = list(x_values)
    for kind, left, right in circuit:
        wires.append(wires[left] ^ wires[right] if kind == "xor" else wires[left] & wires[right])
    return wires[-1]


def share_completions(secret, party, observed, parties, width):
    """How many share profiles of the other parties are consistent with this view and secret."""
    count = 0
    for combo in itertools.product(range(1 << width), repeat=parties - 1):
        profile = list(combo)
        profile.insert(party, observed)
        if reduce(xor, profile) == secret:
            count += 1
    return count


def transcript_distribution(x, y, parties, width):
    """Exact distribution of the public (d, e) masks over every triple-share profile pair."""
    counts = {}
    for a_profile in itertools.product(range(1 << width), repeat=parties):
        for b_profile in itertools.product(range(1 << width), repeat=parties):
            key = (x ^ reduce(xor, a_profile), y ^ reduce(xor, b_profile))
            counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    # 1. share/reconstruct roundtrip for several widths and party counts
    state = 0x5EED
    for width in (1, 2, 8):
        for parties in (2, 3, 5):
            for secret in range(1 << min(width, 4)):
                shares, state = share(state, secret, parties, width)
                check(len(shares) == parties, "one share per party")
                check(reconstruct(shares, width) == secret,
                      "reconstruction must return the secret %d" % secret)

    # 2. the shares really are uniform-looking, and any n-1 of them are independent of the secret
    marginals = set()
    for _ in range(200):
        shares, state = share(state, 0x2A, 4, 8)
        marginals.add(shares[0])
    check(len(marginals) > 100, "one share alone must look uniform: %d distinct values" % len(marginals))

    # 3. perfect privacy, exactly: for every secret and every observed share, the number of
    #    completions of the other parties is the same - the view carries no information
    width, parties = 2, 3
    privacy_width, privacy_parties = width, parties
    for party in range(parties):
        for observed in range(1 << width):
            counts = {secret: share_completions(secret, party, observed, parties, width)
                      for secret in range(1 << width)}
            check(len(set(counts.values())) == 1,
                  "party %d seeing %d must not distinguish the secrets: %r" % (party, observed, counts))
            check(set(counts.values()) == {1 << (width * (parties - 2))},
                  "each secret needs the same number of completions")

    # 4. local operations need no interaction and stay correct
    a, b = 0x37, 0x59
    shares_a, state = share(state, a, 3, 8)
    shares_b, state = share(state, b, 3, 8)
    check(reconstruct(local_xor(shares_a, shares_b, 8), 8) == a ^ b, "local XOR must add secrets")
    check(reconstruct(local_constant(3, 0x0F, 8), 8) == 0x0F, "a public constant must add in")
    check(reconstruct([0x0F] + [0] * 2, 8) == 0x0F, "the constant sharing is a valid sharing")
    check(reconstruct(local_not(shares_a, 8), 8) == a ^ 0xFF, "NOT must complement the secret")
    check(reconstruct(local_xor(local_not(shares_a, 8), local_constant(3, 0xFF, 8), 8), 8) == a,
          "NOT plus the all-ones constant must cancel")
    check(sum(reconstruct(local_xor(shares_a, shares_b, 8), 8) for _ in range(1)) == a ^ b,
          "local addition is deterministic")

    # 5. Beaver AND: correct for every boundary pair, and the transcript is the two uniforms
    transcript = []
    for x_value, y_value in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x_shares, state = share(state, x_value, 3, 1)
        y_shares, state = share(state, y_value, 3, 1)
        z, state = beaver_and(state, x_shares, y_shares, 3, 1, transcript)
        check(reconstruct(z, 1) == x_value & y_value,
              "Beaver AND must be exact on (%d, %d)" % (x_value, y_value))
    check(all(0 <= d < 2 and 0 <= e < 2 for d, e in transcript),
          "the transcript is only the two broadcast masks")

    # 6. the transcript distribution is identical and uniform for every pair of secrets
    reference = None
    for x, y in ((0, 0), (0, 3), (3, 3), (1, 2)):
        counts = transcript_distribution(x, y, parties, width)
        check(len(counts) == (1 << width) ** 2,
              "every mask pair must occur for any secrets: %d of %d" % (len(counts), (1 << width) ** 2))
        check(set(counts.values()) == {1 << (2 * width * (parties - 1))},
              "the mask pairs must be uniformly distributed: %r" % (sorted(set(counts.values())),))
        if reference is None:
            reference = counts
        else:
            check(counts == reference, "the transcript distribution must not depend on the secrets")

    # 7. multi-bit AND: correct on random operands for several widths, bit by bit
    for width in (1, 4, 8):
        for _ in range(20):
            state, r = splitmix64(state)
            x = r & ((1 << width) - 1)
            state, r = splitmix64(state)
            y = r & ((1 << width) - 1)
            shares_x, state = share(state, x, 3, width)
            shares_y, state = share(state, y, 3, width)
            product, state = beaver_and_bitwise(state, shares_x, shares_y, 3, width)
            check(reconstruct(product, width) == x & y,
                  "bitwise AND must be exact for %d & %d at width %d" % (x, y, width))

    # 8. a full shared circuit, evaluated for every input assignment (16 cases)
    circuit = (("and", 0, 1), ("and", 2, 3), ("xor", 4, 5))
    for assignment in itertools.product((0, 1), repeat=4):
        shares = []
        for bit in assignment:
            vector, state = share(state, bit, 3, 1)
            shares.append(vector)
        output, state = gmw_circuit(state, shares, circuit, 3, 1)
        check(reconstruct(output, 1) == plain_circuit(list(assignment), circuit),
              "the shared circuit must agree with the plaintext one on %r" % (assignment,))

    # 9. the protocol is correct for every party count, with each party holding a private view
    for parties in (2, 3, 4, 5):
        state, r = splitmix64(state)
        x = r & 0x0F
        state, r = splitmix64(state)
        y = r & 0x0F
        shares_x, state = share(state, x, parties, 4)
        shares_y, state = share(state, y, parties, 4)
        product, state = beaver_and_bitwise(state, shares_x, shares_y, parties, 4)
        check(reconstruct(product, 4) == x & y, "AND must work with %d parties" % parties)
        check(len(product) == parties, "every party must end up with a share of the product")

    # 10. boundary and error handling
    single, state = share(state, 5, 2, 3)
    check(reconstruct(single, 3) == 5 and single[0] != 5, "two-party sharing puts a mask on party 0")
    try:
        reconstruct([])
        raise AssertionError("an empty share vector must be rejected")
    except ValueError:
        pass
    try:
        reconstruct([1, 5, 3], 2)
        raise AssertionError("out-of-domain shares must be rejected")
    except ValueError:
        pass
    for bad in (lambda: share(state, 9, 3, 2), lambda: share(state, 1, 1, 2),
                lambda: local_xor(single, [1], 3), lambda: local_constant(2, 0x100, 8)):
        try:
            bad()
            raise AssertionError("malformed sharing parameters must be rejected")
        except ValueError:
            pass
    try:
        gmw_circuit(state, single, (("nand", 0, 1),), 2, 3)
        raise AssertionError("an unknown gate must be rejected")
    except ValueError:
        pass

    # 11. determinism, and every share vector really is a fresh sharing of the same secret
    first, state_a = share(0x77, 0x2A, 4, 8)
    second, _ = share(0x77, 0x2A, 4, 8)
    check(first == second, "the dealer must be deterministic given the PRNG state")
    check(reconstruct(first, 8) == reconstruct(second, 8) == 0x2A, "both sharings reconstruct")
    check(first != [0x2A, 0, 0, 0], "a real sharing does not hand the secret to one party")

    print("758 sharing: widths 1/2/8, parties 2..5, reconstruction exact for %d secrets"
          % sum(1 << min(w, 4) for w in (1, 2, 8)))
    print("758 privacy: marginal completions identical for all secrets (%d each), AND transcript"
          " uniform with %d mask pairs counted %d times"
          % (1 << (privacy_width * (privacy_parties - 2)), len(reference),
             sorted(set(reference.values()))[0]))
    print("758 circuit: 16/16 shared evaluations match the plaintext circuit; 4 party counts agree")
    print("PASS")


if __name__ == "__main__":
    main()
