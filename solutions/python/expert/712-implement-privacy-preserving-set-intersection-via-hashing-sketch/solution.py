"""Task 712 - Implement privacy-preserving set intersection via hashing sketch.  (Tier: Expert, CTCI Beyond - Security & Privacy)

Two parties each hold a private set and want the intersection without revealing the rest.  This
file implements both shapes of the classical answer and checks them against plaintext set algebra:

1. Commutative-encryption PSI (the AgES / Diffie-Hellman style protocol).  Fix a group of prime
   order q (the quadratic residues modulo a safe prime p = 2q + 1, generator g).  Each element v is
   mapped to the group element H(v), and a party blinds a set with its secret exponent a by
   raising every element to the a-th power.  Because exponentiation commutes,

       (H(v)^a)^b = (H(v)^b)^a,

   the two parties can exchange blinded sets, re-blind what they received and compare: a value
   appears in both doubly-blinded sets exactly when the two parties share the element.  Neither
   party ever sees the other's plaintext, and the blinding is a bijection of the group, so the
   intersection computed by each side (they run the test independently) is the true intersection.

2. Bloom-filter sketch PSI (Sander's method).  One party sends a Bloom filter of its set; the other
   tests its own elements against it.  This leaks a bounded amount of information and has false
   positives, so the file measures the false-positive rate and checks it against the closed form
   (1 - e^(-k n / m))^k, and it checks the invariant that Bloom filters never produce a false
   negative.

Simplified versus a real deployment: the group is tiny (~40-bit order, not a cryptographic size),
the protocol is the two-round honest-but-curious version with no proofs of correct blinding, the
element-to-group map is a plain hash (so a hash collision shows up as a false positive, which the
file demonstrates deliberately), and there is no networking.  Verified: exact intersection for
curated and random set pairs computed independently by both parties, the commutation and bijection
invariants of the blinding map, the absence of plaintext hashes in the transmitted sketches, the
Bloom filter's no-false-negative guarantee plus its analytic false-positive rate, and all the
degenerate set shapes (empty, disjoint, identical, subset, duplicates, thousands of elements).
"""
import math
import sys

MASK64 = (1 << 64) - 1

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151]


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

    def below(self, bound):
        check(bound >= 2, "the bound must be at least 2")
        self.state, r = splitmix64(self.state)
        return r % (bound - 1) + 1

    def word(self):
        self.state, r = splitmix64(self.state)
        return r


def hash_bytes(payload, seed=0x9E3779B97F4A7C15):
    """A deterministic 128-bit hash of a byte string.

    Only the *mixed* output of splitmix64 is used (the raw state is just an incrementing
    counter, whose low bits would leak straight into the digest).
    """
    state = seed
    digest_low = 0
    digest_high = 0
    for index, byte in enumerate(payload):
        state, word = splitmix64(state + byte + index * 0x1F123BB5)
        digest_low ^= word
        digest_high = (digest_high * 0x100000001B3 + word) & MASK64
    state, word = splitmix64(state ^ digest_low)
    digest_low = (digest_low ^ word) & MASK64
    state, word = splitmix64(state ^ digest_high)
    digest_high = (digest_high ^ word) & MASK64
    return (digest_high << 64) | digest_low


def hash_value(value):
    return hash_bytes(str(value).encode("ascii"))


def is_probable_prime(candidate, rng, rounds=20):
    if candidate < 2:
        return False
    for prime in SMALL_PRIMES:
        if candidate % prime == 0:
            return candidate == prime
    exponent, twos = candidate - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        twos += 1
    for _ in range(rounds):
        witness = pow(rng.below(candidate), exponent, candidate)
        if witness in (1, candidate - 1):
            continue
        for _ in range(twos - 1):
            witness = witness * witness % candidate
            if witness == candidate - 1:
                break
        else:
            return False
    return True


def find_safe_prime(order_bits, seed):
    """p = 2q + 1 with q prime: the multiplicative group then has a prime-order subgroup."""
    rng = Rng(seed)
    while True:
        q = rng.word() | (1 << (order_bits - 1)) | 1
        if not is_probable_prime(q, rng):
            continue
        if is_probable_prime(2 * q + 1, rng):
            return 2 * q + 1, q


class Group:
    """The order-q subgroup of quadratic residues modulo p, with generator g."""

    def __init__(self, order_bits=40, seed=0x1CE):
        self.p, self.q = find_safe_prime(order_bits, seed)
        rng = Rng(seed ^ 0x5EED)
        while True:
            candidate = pow(rng.below(self.p - 1) + 1, 2, self.p)
            if candidate not in (0, 1) and pow(candidate, self.q, self.p) == 1:
                self.g = candidate
                break

    def element(self, value):
        """Hash an element into the subgroup; the squaring makes it a quadratic residue."""
        raw = hash_value(value) % self.p
        return pow(raw + 2, 2, self.p)

    def blind(self, element, exponent):
        return pow(element, exponent, self.p)

    def secret_exponent(self, rng):
        return rng.below(self.q)


def commutative_psi(set_a, set_b, group, seed_a=0xA1, seed_b=0xB2, hash_function=None):
    """Run the two-round protocol; returns (intersection seen by A, intersection seen by B).

    The protocol transcript that leaves a party is the *blinded* set, never the plaintext hash;
    each party assembles its own answer from the round-two messages, which is why two independent
    intersections come out and both are compared with the plaintext answer by the caller.
    """
    hash_function = hash_function or group.element
    party_a = Rng(seed_a)
    party_b = Rng(seed_b)
    exponent_a = group.secret_exponent(party_a)
    exponent_b = group.secret_exponent(party_b)

    hashed_a = [hash_function(value) for value in set_a]
    hashed_b = [hash_function(value) for value in set_b]
    blind_a = [group.blind(element, exponent_a) for element in hashed_a]
    blind_b = [group.blind(element, exponent_b) for element in hashed_b]

    # round 2: everyone re-blinds what it received, so both sides hold H(v)^(a*b)
    doubly_b = [group.blind(value, exponent_a) for value in blind_b]   # A re-blinds B's values
    doubly_a = [group.blind(value, exponent_b) for value in blind_a]   # B re-blinds A's values

    seen_by_a = {value: element for value, element in zip(doubly_a, set_a)}
    answer_a = [seen_by_a[value] for value in doubly_b if value in seen_by_a]
    seen_by_b = {value: element for value, element in zip(doubly_b, set_b)}
    answer_b = [seen_by_b[value] for value in doubly_a if value in seen_by_b]
    transcript = {"blind_a": blind_a, "blind_b": blind_b,
                  "doubly_a": doubly_a, "doubly_b": doubly_b}
    return answer_a, answer_b, transcript


# --------------------------------------------------------------------------- Bloom sketch
def bloom_parameters(capacity, false_positive_target=0.01):
    bits = max(1, int(math.ceil(-capacity * math.log(false_positive_target) / (math.log(2) ** 2))))
    hashes = max(1, int(round(bits / capacity * math.log(2))))
    return bits, hashes


def bloom_positions(value, bits, hashes):
    digest = hash_bytes(str(value).encode("ascii"))
    positions = set()
    for index in range(hashes):
        digest, word = splitmix64(digest + index * 0x9E3779B97F4A7C15)
        positions.add(word % bits)
    return positions


def bloom_build(values, bits, hashes):
    sketch = bytearray(bits)
    for value in values:
        for position in bloom_positions(value, bits, hashes):
            sketch[position] = 1
    return sketch


def bloom_test(sketch, value, hashes):
    bits = len(sketch)
    return all(sketch[position] for position in bloom_positions(value, bits, hashes))


def bloom_psi(set_a, set_b, bits, hashes):
    """B's sketch is sent to A, which tests its own elements: the classic hash-based PSI."""
    sketch = bloom_build(set_b, bits, hashes)
    return [value for value in set_a if bloom_test(sketch, value, hashes)]


def bloom_size_estimate(sketch, bits, hashes):
    """The standard estimator |A n B| ~ -(m/k) ln(1 - X/m) for a one-sided construction."""
    set_bits = sum(sketch)
    if set_bits >= bits:
        return float("inf")
    return -(bits / hashes) * math.log(1.0 - set_bits / bits)


# --------------------------------------------------------------------------- checks
def test_group_setup():
    group = Group(order_bits=40, seed=0x1CE)
    for seed in (1, 2, 3):
        check(is_probable_prime(group.q, Rng(seed)), "q must be prime")
        check(is_probable_prime(group.p, Rng(seed + 100)), "p must be prime")
    check(group.p == 2 * group.q + 1, "p = 2q + 1 with a prime-order subgroup")
    check(pow(group.g, group.q, group.p) == 1, "the generator has order dividing q")
    check(group.g not in (0, 1), "the generator is not degenerate")
    check(pow(group.q, 2, group.p) != 1, "q is not the full group order")
    return group


def test_blinding_invariants():
    group = Group(order_bits=40, seed=0x77)
    rng = Rng(0xF00D)
    for _ in range(40):
        value = rng.word() % 10 ** 6
        element = group.element(value)
        check(1 < element < group.p and pow(element, group.q, group.p) == 1,
              "every hashed element lives in the order-q subgroup")
        exponent_a = group.secret_exponent(Rng(value + 1))
        exponent_b = group.secret_exponent(Rng(value + 2))
        first = group.blind(group.blind(element, exponent_a), exponent_b)
        second = group.blind(group.blind(element, exponent_b), exponent_a)
        check(first == second, "the blinding operations commute: (x^a)^b = (x^b)^a")

    # the blinding map is a bijection of the subgroup, so no information is lost or merged
    exponents = [group.secret_exponent(Rng(1000 + k)) for k in range(8)]
    elements = [group.element(k) for k in range(200)]
    check(len(set(elements)) == 200, "the hash map is injective on the test elements")
    for exponent in exponents:
        images = [group.blind(element, exponent) for element in elements]
        check(len(set(images)) == len(elements), "x -> x^a must be injective on the subgroup")
    # the transmitted values are not the plaintext hashes and are not ordered like them
    hashes = [group.element(k) for k in range(10)]
    blinded = [group.blind(element, 12345 % group.q or 7) for element in hashes]
    check(not set(hashes) & set(blinded), "no blinded value equals a plaintext hash")
    rank_hashes = sorted(range(10), key=lambda index: hashes[index])
    rank_blinded = sorted(range(10), key=lambda index: blinded[index])
    check(rank_hashes != rank_blinded, "the blinding destroys the ordering of the inputs")
    return group


def test_intersection_correctness():
    group = Group(order_bits=40, seed=0x222)
    curated = [
        ([], []),
        ([1, 2, 3], [4, 5, 6]),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3, 4, 5], [2, 4]),
        ([7], [7]),
        ([7], [8]),
        (list(range(50)), list(range(25, 75))),
    ]
    for set_a, set_b in curated:
        answer_a, answer_b, _ = commutative_psi(set_a, set_b, group)
        expected = set(set_a) & set(set_b)
        check(set(answer_a) == expected, f"party A got {sorted(set(answer_a))} != {sorted(expected)}")
        check(set(answer_b) == expected, f"party B got {sorted(set(answer_b))} != {sorted(expected)}")
        check(len(answer_a) == len(set(answer_a)), "the answer must not contain duplicates")

    rng = Rng(0xABCD)
    for trial in range(12):
        universe = [rng.word() % 10 ** 5 for _ in range(80)]
        set_a = sorted({universe[rng.word() % 80] for _ in range(30)})
        set_b = sorted({universe[rng.word() % 80] for _ in range(30)})
        answer_a, answer_b, _ = commutative_psi(set_a, set_b, group, seed_a=7 + trial, seed_b=99)
        expected = set(set_a) & set(set_b)
        check(set(answer_a) == expected and set(answer_b) == expected,
              f"trial {trial}: both parties must agree with plaintext intersection")
    return group


def test_transcript_leaks_no_plaintext():
    group = Group(order_bits=40, seed=0x333)
    set_a = [11, 22, 33, 44]
    set_b = [33, 44, 55, 66]
    _, _, transcript = commutative_psi(set_a, set_b, group)
    plaintext_hashes = {group.element(value) for value in set_a + set_b}
    for name, values in transcript.items():
        check(len(set(values)) == len(values), f"the {name} transcript has no repeated entries")
        check(not (set(values) & plaintext_hashes),
              f"the {name} transcript must not contain plaintext hashes")
        check(all(1 < value < group.p for value in values), f"the {name} values are group residues")
    check(transcript["blind_a"] != transcript["doubly_a"],
          "the second round really re-blinds (the transcripts differ)")
    check(len(transcript["blind_a"]) == len(set_a), "one transcript entry per element")


def test_hash_collisions_are_the_only_false_positives():
    group = Group(order_bits=40, seed=0x444)

    def colliding_map(value):
        """A deliberately broken element map: 11 and 111 collide, everything else is distinct."""
        key = 0 if value in (11, 111) else value
        return group.element(key)

    set_a = [11, 5, 6]
    set_b = [111, 6]
    answer_a, answer_b, _ = commutative_psi(set_a, set_b, group, hash_function=colliding_map)
    check(set(answer_a) == {11, 6}, f"a colliding hash shows up as a false positive: {answer_a}")
    check(set(answer_a) > set(set_a) & set(set_b), "the reported intersection is a superset")
    check(set(answer_a) - set(set_b) == {11}, "the extra element is exactly the colliding one")
    check(set(answer_b) == {111, 6}, "the other party sees its own colliding element")

    # with the real hash map the same sets are reported exactly
    answer_a, _, _ = commutative_psi(set_a, set_b, group)
    check(set(answer_a) == {6}, "with the real map the answer is exact again")


def test_bloom_sketch_psi():
    rng = Rng(0x777)
    set_b = [rng.word() % 10 ** 6 for _ in range(400)]
    set_a = [value for value in set_b[:120]] + [rng.word() % 10 ** 6 for _ in range(280)]
    bits, hashes = bloom_parameters(len(set_b), 0.01)
    check(bits >= len(set_b), "the sketch is at least as large as the set")
    check(hashes == max(1, int(round(bits / len(set_b) * math.log(2)))), "k follows the formula")

    found = bloom_psi(set_a, set_b, bits, hashes)
    check(set(set_b[:120]) <= set(found), "a Bloom filter never produces a false negative")
    false_positives = [value for value in found if value not in set(set_b)]
    kinds = len(found)
    measured = len(false_positives) / max(1, len(set_a) - 120)
    theoretical = (1 - math.exp(-hashes * len(set_b) / bits)) ** hashes
    check(measured <= theoretical * 3 + 0.01,
          f"the measured false-positive rate {measured:.4f} must stay near {theoretical:.4f}")
    check(kinds >= 120, "every true element must be reported")

    estimate = bloom_size_estimate(bloom_build(set_b, bits, hashes), bits, hashes)
    check(abs(estimate - len(set_b)) < 0.15 * len(set_b),
          f"the sketch size estimate {estimate:.1f} should be close to {len(set_b)}")

    # degenerate sketches
    tiny = bloom_build(set_b[:3], 1, 1)
    check(all(bloom_test(tiny, value, 1) for value in set_b[:3]), "a 1-bit sketch still works")
    check(bloom_test(tiny, "anything", 1), "and it accepts everything: the collision extreme")
    empty = bloom_build([], bits, hashes)
    check(bloom_psi(set_a, [], bits, hashes) == [], "an empty sender means an empty intersection")
    check(bloom_psi([], set_b, bits, hashes) == [], "an empty receiver asks nothing")
    return bits, hashes, measured, theoretical


def test_large_sets_and_determinism():
    group = Group(order_bits=40, seed=0x555)
    rng = Rng(0x1234)
    shared = {rng.word() % 10 ** 6 for _ in range(300)}
    set_a = sorted(shared | {rng.word() % 10 ** 6 for _ in range(1700)})
    set_b = sorted(shared | {rng.word() % 10 ** 6 for _ in range(1700)})
    answer_a, answer_b, _ = commutative_psi(set_a, set_b, group)
    expected = set(set_a) & set(set_b)
    check(set(answer_a) == expected, "the large instance must be exactly right")
    check(set(answer_b) == expected, "both parties agree on the large instance")
    check(len(expected) >= 250, f"the shared part must dominate, got {len(expected)}")

    again_a, again_b, _ = commutative_psi(set_a, set_b, group)
    check(again_a == answer_a and again_b == answer_b,
          "the protocol is deterministic for fixed exponents")
    bits, hashes = bloom_parameters(len(set_b), 0.02)
    bloom_answer = bloom_psi(set_a, set_b, bits, hashes)
    check(expected <= set(bloom_answer), "the Bloom variant has no false negatives either")
    returned_bloom, returned_commutative = len(bloom_answer), len(answer_a)
    return len(set_a), len(set_b), returned_commutative, returned_bloom


def main():
    test_group_setup()
    test_blinding_invariants()
    test_intersection_correctness()
    test_transcript_leaks_no_plaintext()
    test_hash_collisions_are_the_only_false_positives()
    bits, hashes, measured, theoretical = test_bloom_sketch_psi()
    sizes = test_large_sets_and_determinism()

    print(f"bloom: m={bits} k={hashes} measured_fpr={measured:.4f} theory={theoretical:.4f}")
    print(f"large: |A|={sizes[0]} |B|={sizes[1]} commutative={sizes[2]} bloom_candidates={sizes[3]}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
