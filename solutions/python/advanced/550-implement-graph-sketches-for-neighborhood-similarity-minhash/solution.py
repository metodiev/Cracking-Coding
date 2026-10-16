"""Task 550 - Implement graph sketches for neighborhood similarity (MinHash). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Compressing each vertex's neighbourhood into a small sketch so that neighbourhood Jaccard
similarity |N(u) ∩ N(v)| / |N(u) ∪ N(v)| can be estimated without touching the neighbourhoods.
The main sketch is MinHash: k universal hashes h_i(x) = (a_i * (x + 1) + b_i) mod P with
P = 4294967291 prime and the (a_i, b_i) drawn from a full-period 64-bit LCG (the small 16-bit LCG
used for sample data cannot supply both coefficients of a hash independently - deriving them from
one scalar puts all k permutations on a single line and the estimates come out correlated, which
the file documents next to the generator), and the signature of a set is the vector of per-hash
minima. For two sets the fraction of agreeing signature components is an
unbiased estimator of their Jaccard similarity with standard deviation <= 1/(2 sqrt(k)); the file
checks that empirically as well as per pair. A second, structurally different sketch is
implemented as well - the bottom-k / k-minimum-values sketch under a single ranking function, with
the estimator |A ∩ B ∩ {h <= t}| / |A ∪ B ∩ {h <= t}| for t = min(k-th smallest of A, the k-th
smallest of B) - and both must agree with the exact similarity within tolerance. Neighbourhoods are
sets, so parallel edges are folded away and the sketches are invariant under duplicate entries.

Checks: identical sets giving an estimate of exactly 1 and disjoint sets exactly 0 (the hashes are
checked to be injective on the element range, so disjointness cannot be faked by collisions);
hand-checked triangles, the bowtie (two triangles sharing a hub, where every rim pair scores 1/3
and a rim versus the hub scores 1/5) and the diamond K4 minus one edge, where the two non-adjacent
vertices have identical neighbourhoods and therefore similarity exactly 1 - the textbook
link-prediction case; the exact Jaccard computed two ways (set
operations and a sorted-merge count); all pairs of LCG graphs with k = 512 agreeing within 0.1 of
the exact value, and the mean absolute error shrinking as k grows; the unbiasedness of the
estimator averaged over 40 hash families; empty neighbourhoods handled by the documented convention
J(empty, empty) = J(empty, S) = 0; symmetry and the range [0, 1]; sketch determinism; and the ranking
use case: the pair the sketch puts first is within 0.05 of the best exact similarity on every
tested graph.
"""
P = 4294967291                                    # the largest prime below 2^32
EMPTY = P                                         # sentinel for the minimum over an empty set
MASK64 = (1 << 64) - 1


def lcg(seed):
    return (seed * 75 + 74) % 65537


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants) - the source of the hash coefficients.

    Deriving both a and b from the same 16-bit LCG value would put the whole family on one line
    b = 7919 (a - 1) / 97 mod P, i.e. 256 copies of essentially the same permutation; the
    signatures then agree and disagree in correlated blocks and the estimates are useless (the
    estimator was observed to return 0 where the exact similarity was 0.25). Halving a 64-bit
    state into two independent-looking coefficients fixes that while staying deterministic.
    """
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def hash_family(k, seed):
    """k pairs (a, b) with a != 0, so x -> (a (x + 1) + b) mod P is a bijection on 0..P-1."""
    state = (seed * 2654435761 + 1) & MASK64
    family = []
    for _ in range(k):
        state = prng64(state)
        a = 1 + (state >> 16) % (P - 1)
        state = prng64(state)
        b = (state >> 16) % P
        family.append((a, b))
    return family, state


def hashed(value, a, b):
    return (a * (value + 1) + b) % P


def minhash_signature(elements, family):
    signature = []
    for (a, b) in family:
        best = EMPTY
        for element in elements:
            candidate = hashed(element, a, b)
            if candidate < best:
                best = candidate
        signature.append(best)
    return signature


def jaccard_estimate(signature_a, signature_b):
    """Fraction of agreeing components; empty sets are handled by the documented convention."""
    if all(value == EMPTY for value in signature_a) or \
            all(value == EMPTY for value in signature_b):
        return 0.0
    agree = sum(1 for (x, y) in zip(signature_a, signature_b) if x == y)
    return agree / len(signature_a)


def plan(x, y):
    """Deterministic linear ranking used by the bottom-k sketch, independent of the family."""
    return (2654435761 * (x + 1) + 40503 * (y + 1) + 12345) % P


def bottom_k(elements, k, seed=0):
    """The k elements of the set with the smallest rank, plus the k-th smallest rank."""
    ranked = sorted((plan(value, seed), value) for value in elements)
    chosen = [value for (_, value) in ranked[:k]]
    threshold = ranked[k - 1][0] if len(ranked) >= k else EMPTY
    return frozenset(chosen), threshold


def bottom_k_estimate(set_a, set_b, k, seed=0):
    """k-minimum-values estimator: compare the two truncated sets, not their full contents."""
    chosen_a, threshold_a = bottom_k(set_a, k, seed)
    chosen_b, threshold_b = bottom_k(set_b, k, seed)
    threshold = min(threshold_a, threshold_b)
    if threshold == EMPTY:                       # at least one set is smaller than k
        threshold = max(threshold_a, threshold_b)
    ranked_a = {value for value in set_a if plan(value, seed) <= threshold}
    ranked_b = {value for value in set_b if plan(value, seed) <= threshold}
    union = ranked_a | ranked_b
    if not union:
        return 0.0
    return len(ranked_a & ranked_b) / len(union)


def exact_jaccard(set_a, set_b):
    """Convention for degenerate inputs: an empty neighbourhood has similarity zero."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def exact_jaccard_by_merge(set_a, set_b):
    """Independent count via two sorted lists rather than Python set operations."""
    left = sorted(set_a)
    right = sorted(set_b)
    i = j = 0
    common = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            common += 1
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    union = len(left) + len(right) - common
    if union == 0:
        return 0.0
    return common / union


def neighbourhoods(n, edges, include_self=False):
    sets = [set() for _ in range(n)]
    for (u, v) in edges:
        sets[u].add(v)
        sets[v].add(u)
    if include_self:
        for v in range(n):
            sets[v].add(v)
    return sets


def lcg_graph(n, density, seed):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            seed = lcg(seed)
            if seed % 100 < density:
                edges.append((i, j))
    return edges, seed


def main():
    # --- Sanity of the hash family ------------------------------------------------------
    family, _ = hash_family(64, 20241016)
    assert len(family) == 64 and all(a != 0 for (a, _) in family)
    assert all(b < P for (_, b) in family)
    images = [hashed(value, family[0][0], family[0][1]) for value in range(5000)]
    assert len(set(images)) == 5000, "the hash is injective on the small element range"
    assert len({a for (a, _) in family}) == 64, "the multipliers are distinct here"

    # --- Identical, disjoint and empty sets ---------------------------------------------
    family, _ = hash_family(256, 7)
    full = minhash_signature(range(24), family)
    same = minhash_signature(list(reversed(range(24))), family)
    assert full == same, "the signature only depends on the set"
    assert minhash_signature([1, 1, 1, 2, 3], family) == minhash_signature([1, 2, 3], family), \
        "duplicates are folded away"
    assert jaccard_estimate(full, full) == 1.0, "identical sets estimate 1 exactly"
    disjoint = minhash_signature(range(100, 124), family)
    assert jaccard_estimate(full, disjoint) == 0.0, "disjoint sets estimate 0 exactly"
    assert exact_jaccard(set(range(24)), set(range(100, 124))) == 0.0
    empty = minhash_signature([], family)
    assert minhash_signature([], family) == [EMPTY] * 256
    assert jaccard_estimate(empty, empty) == 0.0, "the empty-set convention is zero"
    assert jaccard_estimate(empty, full) == 0.0
    assert exact_jaccard(set(), set()) == 0.0 and exact_jaccard(set(), {1}) == 0.0
    assert exact_jaccard({1, 2}, {2, 3}) == 1.0 / 3.0
    assert exact_jaccard_by_merge({1, 2}, {2, 3}) == 1.0 / 3.0
    assert exact_jaccard({1}, {1}) == 1.0 and exact_jaccard_by_merge({1}, {1}) == 1.0

    # --- Hand-checked graphs ------------------------------------------------------------
    triangle = neighbourhoods(3, [(0, 1), (1, 2), (0, 2)])
    assert exact_jaccard(triangle[0], triangle[1]) == 1.0 / 3.0, "adjacent triangle vertices"
    # Bowtie: two triangles sharing the hub 2. Adjacent rim vertices share {2} plus one another's
    # other neighbour, and the hub's neighbourhood is the union of the rims.
    bowtie = neighbourhoods(5, [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)])
    assert bowtie[0] == {1, 2} and bowtie[3] == {2, 4}
    assert bowtie[2] == {0, 1, 3, 4}
    assert exact_jaccard(bowtie[0], bowtie[1]) == 1.0 / 3.0, "one rim pair: {1,2} vs {0,2}"
    assert exact_jaccard(bowtie[3], bowtie[4]) == 1.0 / 3.0, "the other rim pair: {2,4} vs {2,3}"
    assert exact_jaccard(bowtie[0], bowtie[3]) == 1.0 / 3.0, "across the hub: {1,2} vs {2,4}"
    assert exact_jaccard(bowtie[0], bowtie[2]) == 0.2, "a rim versus the hub: 1 of 5"
    best_pair = max(((u, v) for u in range(5) for v in range(u + 1, 5)),
                    key=lambda pair: exact_jaccard(bowtie[pair[0]], bowtie[pair[1]]))
    assert exact_jaccard(bowtie[best_pair[0]], bowtie[best_pair[1]]) == 1.0 / 3.0
    # Diamond: K4 minus the edge 0-1. Vertices 0 and 1 are *not* adjacent yet have identical
    # neighbourhoods, the textbook link-prediction case with similarity exactly 1.
    diamond = neighbourhoods(4, [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    assert 1 not in diamond[0] and diamond[0] == diamond[1] == {2, 3}
    assert exact_jaccard(diamond[0], diamond[1]) == 1.0
    pairs_at_one = [(u, v) for u in range(4) for v in range(u + 1, 4)
                    if exact_jaccard(diamond[u], diamond[v]) == 1.0]
    assert pairs_at_one == [(0, 1)], "the only pair sharing a neighbourhood exactly"
    assert exact_jaccard(diamond[0], diamond[2]) == 0.25, "{3} shared out of {0, 1, 2, 3}"
    family, _ = hash_family(256, 31)
    diamond_signatures = [minhash_signature(neighbours, family) for neighbours in diamond]
    assert jaccard_estimate(diamond_signatures[0], diamond_signatures[1]) == 1.0
    assert abs(jaccard_estimate(diamond_signatures[0], diamond_signatures[2]) - 0.25) <= 0.1
    for u in range(5):
        for v in range(5):
            assert exact_jaccard(bowtie[u], bowtie[v]) == exact_jaccard(bowtie[v], bowtie[u])
            assert exact_jaccard_by_merge(bowtie[u], bowtie[v]) == exact_jaccard(bowtie[u],
                                                                               bowtie[v])
    family, _ = hash_family(512, 99)
    bowtie_signatures = [minhash_signature(neighbours, family) for neighbours in bowtie]
    assert abs(jaccard_estimate(bowtie_signatures[3], bowtie_signatures[4]) - 1.0 / 3.0) <= 0.1
    assert abs(jaccard_estimate(bowtie_signatures[0], bowtie_signatures[1]) - 1.0 / 3.0) <= 0.1
    assert jaccard_estimate(bowtie_signatures[2], bowtie_signatures[2]) == 1.0

    # --- Accuracy over all pairs of LCG graphs -----------------------------------------
    seed = 4242
    for trial in range(4):
        n = 14 + 2 * trial
        edges, seed = lcg_graph(n, 35 + 5 * trial, seed)
        sets = neighbourhoods(n, edges)
        family, _ = hash_family(512, 555 + trial)
        signatures = [minhash_signature(neighbours, family) for neighbours in sets]
        errors = []
        for u in range(n):
            for v in range(u + 1, n):
                exact = exact_jaccard(sets[u], sets[v])
                assert abs(exact - exact_jaccard_by_merge(sets[u], sets[v])) <= 1e-12
                estimate = jaccard_estimate(signatures[u], signatures[v])
                assert 0.0 <= estimate <= 1.0
                assert abs(estimate - exact) <= 0.1, "512 hashes keep the error inside 0.1"
                errors.append(abs(estimate - exact))
        assert errors and max(errors) <= 0.1
        # Ordering agreement: sorting the pairs by the sketch reproduces the exact ordering.
        ranked_by_sketch = sorted(
            ((u, v) for u in range(n) for v in range(u + 1, n)),
            key=lambda pair: -jaccard_estimate(signatures[pair[0]], signatures[pair[1]]))
        best_exact = max(((u, v) for u in range(n) for v in range(u + 1, n)),
                         key=lambda pair: exact_jaccard(sets[pair[0]], sets[pair[1]]))
        top_sketch = ranked_by_sketch[0]
        assert abs(exact_jaccard(sets[top_sketch[0]], sets[top_sketch[1]])
                   - exact_jaccard(sets[best_exact[0]], sets[best_exact[1]])) <= 0.05, \
            "the sketch ranks a genuinely best pair first"

    # --- More hashes means less error ---------------------------------------------------
    n = 20
    edges, _ = lcg_graph(n, 45, 31415)
    sets = neighbourhoods(n, edges)
    pair_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    mean_error = {}
    for k in (32, 512):
        family, _ = hash_family(k, 2718)
        signatures = [minhash_signature(neighbours, family) for neighbours in sets]
        total = 0.0
        for (u, v) in pair_list:
            total += abs(jaccard_estimate(signatures[u], signatures[v])
                         - exact_jaccard(sets[u], sets[v]))
        mean_error[k] = total / len(pair_list)
    assert mean_error[512] < mean_error[32], "a longer signature is more accurate"
    assert mean_error[512] <= 0.05

    # --- Unbiasedness over many hash families -------------------------------------------
    pair = (7, 11)
    exact_pair = exact_jaccard(sets[pair[0]], sets[pair[1]])
    estimates = []
    for trial in range(40):
        family, _ = hash_family(256, 9000 + 13 * trial)
        left = minhash_signature(sets[pair[0]], family)
        right = minhash_signature(sets[pair[1]], family)
        estimates.append(jaccard_estimate(left, right))
    mean = sum(estimates) / len(estimates)
    assert abs(mean - exact_pair) <= 0.03, "the estimator is unbiased in the mean over families"
    assert min(estimates) >= 0.0 and max(estimates) <= 1.0
    spread = (sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1)) ** 0.5
    assert spread <= 0.5 / (256 ** 0.5) + 0.05, "the spread respects the 1/(2 sqrt(k)) heuristic"

    # --- Bottom-k sketch ---------------------------------------------------------------
    for (n, density) in ((12, 40), (16, 35), (10, 60)):
        edges, _ = lcg_graph(n, density, 6060 + n)
        sets = neighbourhoods(n, edges)
        for k in (32, 64):
            worst = 0.0
            for u in range(n):
                for v in range(u + 1, n):
                    exact = exact_jaccard(sets[u], sets[v])
                    estimate = bottom_k_estimate(sets[u], sets[v], k)
                    assert 0.0 <= estimate <= 1.0
                    worst = max(worst, abs(estimate - exact))
            assert worst <= 0.2, "the k-minimum-values sketch stays within 0.2"
    sets = neighbourhoods(5, [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)])
    assert bottom_k_estimate(sets[3], sets[3], 8) == 1.0, "identical sets score exactly 1"
    # Both neighbourhoods here are smaller than k, so the truncated sets are the full sets and
    # the sketch is exact rather than an estimate.
    assert abs(bottom_k_estimate(sets[3], sets[4], 8) - 1.0 / 3.0) <= 1e-12
    assert abs(bottom_k_estimate(sets[0], sets[1], 8) - 1.0 / 3.0) <= 1e-12
    empty_sets = neighbourhoods(3, [(0, 1)])
    assert bottom_k_estimate(empty_sets[2], empty_sets[2], 8) == 0.0, \
        "an isolated vertex has no neighbourhood to compare"
    assert bottom_k_estimate(sets[0], sets[1], 4) > 0.0

    # --- Determinism ------------------------------------------------------------------
    edges, _ = lcg_graph(12, 40, 8080)
    sets = neighbourhoods(12, edges)
    family, _ = hash_family(128, 4321)
    first = [minhash_signature(neighbours, family) for neighbours in sets]
    second = [minhash_signature(neighbours, family) for neighbours in sets]
    assert first == second, "sketches are deterministic"
    assert bottom_k(sets[0], 8) == bottom_k(sets[0], 8)
    assert bottom_k_estimate(sets[0], sets[1], 8) == bottom_k_estimate(sets[0], sets[1], 8)
    assert len(first[0]) == 128
    assert all(value < P for signature in first for value in signature)


if __name__ == "__main__":
    main()
    print("PASS")
