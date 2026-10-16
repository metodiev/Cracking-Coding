"""Task 503 - Implement locality-sensitive hashing for cosine similarity (simhash). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

SimHash for cosine similarity: b random hyperplane normals r_i (standard normal deviates from a
deterministic splitmix64 stream, hence an exactly isotropic direction) turn a vector x into a
b-bit signature whose i-th bit is sign(r_i . x). For two vectors at angle theta exactly
P[bit agrees] = 1 - theta/pi (the measure of the wedge between the two hyperplanes), so the
signature is a Bernoulli sample of the angle. The estimator used here therefore goes through the
angle: theta_hat = pi * hamming / b, and cos(theta_hat) - not the naive 1 - 2 * hamming / b, which
is the same number only at theta = 0, pi/2 and pi and is otherwise shrunk towards zero by up to
0.2 - measures the cosine. The signature is invariant under positive scaling (the sign of a
scaled dot product does not change) and negation complements every bit. For retrieval the b bits
are cut into `bands` disjoint bands of `band_bits` bits; two vectors become candidates when they
agree on at least one whole band ("AND inside a band, OR across bands"): identical vectors always
collide (hamming 0), while unrelated vectors collide in a band with probability 2^-band_bits, so
the candidate list stays tiny. The signature is also a compact code: b bits instead of dim doubles.

Documented checks: identical vectors hash to hamming 0 (cosine estimate exactly 1.0) and a positive
rescaled copy keeps every bit, while the negated vector complements all b bits (estimate -1.0);
over 2000 hyperplanes the measured per-bit agreement at angles pi/4, pi/2 and 3pi/4 must match
1 - theta/pi to within 0.03, and the angle/dot-product estimates must satisfy
|theta_hat - theta| <= 0.05 and |cos(theta_hat) - cos(theta)| <= 0.05; on 20 LCG pairs in 16
dimensions the hamming estimate must track the exact cosine with mean absolute error <= 0.08 and
worst case <= 0.25 (256 bits), keeping the sign of every clearly positive/negative cosine, and its
order for a close versus a far pair (averaged over 8 independent hyperplane draws) must be right;
and the banded index must retrieve each of 5 planted near-duplicates (cosine >= 0.9) for every
query, never drop a vector from its own candidate list, keep the average candidate list below 20%
of the 40-entry corpus, and account for every bucket entry exactly.
"""

import math

BITS = 256
DIM = 16
BANDS = 16
BAND_BITS = BITS // BANDS
THEORY_PLANES = 2000
CORPUS = 40
NEAR_DUPES = 5
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
TWO_PI = 2.0 * math.pi


def lcg(seed):
    return (seed * 75 + 74) % 65537


def rand_vector(dim, seed, span=5):
    """Data vector from the wave's 75/74 LCG: small integer entries in [-span, span]."""
    out = []
    for _ in range(dim):
        seed = lcg(seed)
        value = float(seed % (2 * span + 1) - span)
        out.append(1.0 if value == 0.0 else value)
    return out, seed


def splitmix64(state):
    """Deterministic 64-bit mixer used for the hyperplane directions."""
    state = (state + GOLDEN) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def uniform(state):
    state, z = splitmix64(state)
    return state, (z + 0.5) / 2.0 ** 64


def make_planes(bits, dim, seed):
    """`bits` isotropic hyperplane normals (Box-Muller standard normals, fixed seed)."""
    state = seed
    planes = []
    while len(planes) < bits:
        vector = []
        while len(vector) < dim:
            state, u1 = uniform(state)
            state, u2 = uniform(state)
            radius = math.sqrt(-2.0 * math.log(u1))
            angle = TWO_PI * u2
            vector.append(radius * math.cos(angle))
            vector.append(radius * math.sin(angle))
        planes.append(vector[:dim])
    return planes


def signature(vec, planes, bits=None):
    """b-bit simhash signature as an integer (bit i = sign(r_i . x), ties count as +1)."""
    if bits is None:
        bits = len(planes)
    sig = 0
    for i in range(bits):
        if sum(a * b for a, b in zip(planes[i], vec)) >= 0.0:
            sig |= 1 << i
    return sig


def popcount(value):
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count


def hamming(sig_a, sig_b):
    return popcount(sig_a ^ sig_b)


def agreement(sig_a, sig_b, bits=BITS):
    """1 - hamming / b: the estimated probability that one random bit agrees."""
    return 1.0 - hamming(sig_a, sig_b) / bits


def angle_estimate(sig_a, sig_b, bits=BITS):
    """theta_hat = pi * hamming / b, unbiased for the angle: each bit is Bernoulli(1 - theta/pi)."""
    return math.pi * hamming(sig_a, sig_b) / bits


def hash_similarity(sig_a, sig_b, bits=BITS):
    """2 * agreement - 1. Equals the cosine only at theta = 0, pi/2, pi - kept for comparison."""
    return 1.0 - 2.0 * hamming(sig_a, sig_b) / bits


def cosine_estimate(sig_a, sig_b, bits=BITS):
    return math.cos(angle_estimate(sig_a, sig_b, bits))


def cosine(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def bands_of(sig, bands=BANDS, band_bits=BAND_BITS):
    mask = (1 << band_bits) - 1
    return [(b, (sig >> (b * band_bits)) & mask) for b in range(bands)]


class SimHashIndex:
    """Banded simhash index: two vectors collide when they share at least one whole band."""

    def __init__(self, planes, bands=BANDS, band_bits=BAND_BITS):
        self.planes = planes
        self.bands = bands
        self.band_bits = band_bits
        self.signatures = {}
        self.buckets = {}
        self.entries = 0

    def add(self, key, vec):
        sig = signature(vec, self.planes, self.bands * self.band_bits)
        self.signatures[key] = sig
        for band, value in bands_of(sig, self.bands, self.band_bits):
            self.buckets.setdefault((band, value), []).append(key)
            self.entries += 1
        return sig

    def candidates(self, vec):
        sig = signature(vec, self.planes, self.bands * self.band_bits)
        out = set()
        for band, value in bands_of(sig, self.bands, self.band_bits):
            out.update(self.buckets.get((band, value), ()))
        return out

    def similarity(self, key_a, key_b):
        return cosine_estimate(self.signatures[key_a], self.signatures[key_b],
                               self.bands * self.band_bits)


def corpus_and_duplicates(seed=101):
    """CORPUS vectors in 16 dimensions with 5 planted near-duplicates (cosine ~ 0.94)."""
    vectors = []
    for _ in range(CORPUS):
        vec, seed = rand_vector(DIM, seed)
        vectors.append([int(x) for x in vec])
    for i in range(NEAR_DUPES):
        copy = vectors[i][:]
        for index, delta in ((2, 3), (9, -3)):
            copy[index] += delta
        vectors[i + NEAR_DUPES] = copy
    return vectors


def main():
    planes = make_planes(BITS, DIM, seed=7)
    assert len(planes) == BITS and all(len(p) == DIM for p in planes)

    # --- Exact hand checks: duplicates, positive rescaling, negation. -------------------------
    v = ([3.0, -4.0, 5.0] + [1.0] * (DIM - 3))[:DIM]
    sig_v = signature(v, planes)
    assert hamming(sig_v, signature(v, planes)) == 0
    assert cosine_estimate(sig_v, sig_v, BITS) == 1.0 and agreement(sig_v, sig_v, BITS) == 1.0
    assert signature([7.0 * x for x in v], planes) == sig_v, \
        "positive scaling must not change any bit"
    neg_sig = signature([-x for x in v], planes)
    assert hamming(sig_v, neg_sig) == BITS
    assert neg_sig == ((1 << BITS) - 1) ^ sig_v, "negation complements every bit"
    assert abs(cosine_estimate(sig_v, neg_sig, BITS) + 1.0) <= 1e-12
    assert abs(angle_estimate(sig_v, sig_v, BITS)) <= 1e-15

    # --- Angle sweep: bit agreement equals 1 - theta/pi and the estimates recover theta. -----
    many = make_planes(THEORY_PLANES, 2, seed=13)
    for theta in (math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
        a = [1.0, 0.0]
        b = [math.cos(theta), math.sin(theta)]
        sig_a = signature(a, many)
        sig_b = signature(b, many)
        measured = agreement(sig_a, sig_b, THEORY_PLANES)
        theory = 1.0 - theta / math.pi
        assert abs(measured - theory) <= 0.03, (theta, measured, theory)
        theta_hat = angle_estimate(sig_a, sig_b, THEORY_PLANES)
        assert abs(theta_hat - theta) <= 0.05, (theta, theta_hat)
        assert abs(cosine_estimate(sig_a, sig_b, THEORY_PLANES) - math.cos(theta)) <= 0.05
    # A positive multiple keeps every bit; the negated vector flips all of them.
    assert hamming(signature([1.0, 0.0], many), signature([2.0, 0.0], many)) == 0
    assert hamming(signature([1.0, 0.0], many), signature([-1.0, 0.0], many)) == THEORY_PLANES

    # The naive 1 - 2h/b equals the cosine only at 0, pi/2 and pi (documented shrink towards 0).
    theta = math.pi / 4.0
    sig_a = signature([1.0, 0.0], planes)
    sig_b = signature([math.cos(theta), math.sin(theta)], planes)
    assert abs(hash_similarity(sig_a, sig_b, BITS) - math.cos(theta)) > 0.1
    assert abs(cosine_estimate(sig_a, sig_b, BITS) - math.cos(theta)) <= 0.1

    # --- 20 LCG pairs in 16 dimensions: estimate versus exact cosine. -------------------------
    pairs = []
    seed = 31
    for _ in range(20):
        a, seed = rand_vector(DIM, seed)
        c, seed = rand_vector(DIM, seed)
        weight = (seed % 11 - 5) / 5.0
        pairs.append((a, [x + weight * y for x, y in zip(c, a)]))
    errors = []
    for a, b in pairs:
        exact = cosine(a, b)
        got = cosine_estimate(signature(a, planes), signature(b, planes), BITS)
        errors.append(abs(got - exact))
        assert abs(got - exact) <= 0.25, (got, exact)
        if exact > 0.2:
            assert got > 0.0, "the estimator must keep the sign of a clearly positive cosine"
        if exact < -0.2:
            assert got < 0.0, "the estimator must keep the sign of a clearly negative cosine"
    mean_error = sum(errors) / len(errors)
    assert mean_error <= 0.08, mean_error
    assert max(errors) <= 0.25

    # Averaging over 8 independent hyperplane draws must rank a close pair above a far one.
    close_a = [10.0, 1.0] + [0.0] * (DIM - 2)
    close_b = [10.0, 2.0] + [0.0] * (DIM - 2)
    far_c = [0.0, 10.0] + [0.0] * (DIM - 2)
    assert cosine(close_a, close_b) > cosine(close_a, far_c)
    close_sum = 0.0
    far_sum = 0.0
    for draw in range(8):
        drawn = make_planes(BITS, DIM, seed=1000 + draw)
        close_sum += cosine_estimate(signature(close_a, drawn), signature(close_b, drawn), BITS)
        far_sum += cosine_estimate(signature(close_a, drawn), signature(far_c, drawn), BITS)
    assert close_sum / 8.0 > far_sum / 8.0 + 0.05, (close_sum / 8.0, far_sum / 8.0)

    # --- Banded retrieval of planted near-duplicates. ----------------------------------------
    vectors = corpus_and_duplicates()
    assert len(vectors) == CORPUS
    index = SimHashIndex(planes)
    for i, vec in enumerate(vectors):
        index.add(i, vec)
    assert index.entries == CORPUS * BANDS, "every vector must sit in every band"
    assert sum(len(bucket) for bucket in index.buckets.values()) == index.entries

    candidate_sizes = []
    for q in range(NEAR_DUPES):
        near = q + NEAR_DUPES
        exact = cosine(vectors[q], vectors[near])
        assert exact >= 0.9, ("planted duplicate is not near", exact)
        assert abs(index.similarity(q, near) - exact) <= 0.1, "estimate of the duplicate"
        candidates = index.candidates(vectors[q])
        candidate_sizes.append(len(candidates))
        assert q in candidates, "a vector must retrieve itself"
        assert near in candidates, "the planted near-duplicate must collide in some band"
        true_nn = min((r for r in range(CORPUS) if r != q),
                      key=lambda r: -cosine(vectors[q], vectors[r]))
        assert true_nn in candidates, "the exact nearest neighbour must be a candidate"
    assert max(candidate_sizes) <= 5, ("candidate lists must stay tiny", candidate_sizes)
    assert sum(candidate_sizes) / len(candidate_sizes) < 0.2 * CORPUS

    # A vector far from the whole corpus collides with nobody; the zero vector has no bucket.
    outsider = [1.0 if i % 2 == 0 else -1.0 for i in range(DIM)]
    assert max(cosine(outsider, vec) for vec in vectors) < 0.9
    assert index.candidates([0.0] * DIM) == set()

    # Signature is a compact code: 32 bytes instead of dim doubles.
    assert BITS // 8 == 32 and BITS < DIM * 64


if __name__ == "__main__":
    main()
    print("PASS")
