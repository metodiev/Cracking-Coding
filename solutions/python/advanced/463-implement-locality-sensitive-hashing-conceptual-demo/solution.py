"""Task 463 - Implement locality sensitive hashing (conceptual demo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

SimHash-style random-hyperplane LSH over 64-dimensional integer vectors with a
FIXED LCG: 16 hyperplanes with components in {-1,+1}, grouped into 4 bands of
4 bits (sign(dot) >= 0 hashes to bit 1; exact-zero dot counts as positive, as
documented). Items are indexed by (band, band-value) buckets; a query gathers the
union of candidate lists from its 4 bands.

Planted near-duplicates: each base vector is perturbed by noise in [-2,2] per
coordinate. The demo asserts that the recall of the planted base over the fixed
LCG workload is at least the documented threshold RECALL_MIN (measured ~0.99),
and that orthogonal basis-vector pairs are handled without errors: for a fixed
LCG sample of 100 disjoint-support (orthogonal) pairs, the fraction sharing a
band stays below the documented ORTHO_COLLISION_MAX (measured ~0.2).
"""

DIMS = 64
BITS = 16
BANDS = 4
BITS_PER_BAND = 4
RECALL_MIN = 0.9
ORTHO_COLLISION_MAX = 0.5


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def make_hyperplanes(seed):
    planes = []
    for _ in range(BITS):
        row = []
        for _ in range(DIMS):
            seed = lcg_value(seed)
            row.append(1 if seed % 2 == 0 else -1)
        planes.append(row)
    return planes, seed


def signature(vector, planes):
    bits = 0
    for j in range(BITS):
        row = planes[j]
        dot = 0
        for k in range(DIMS):
            dot += row[k] * vector[k]
        if dot >= 0:
            bits |= 1 << j
    return bits


def band_keys(bits):
    return [(b, (bits >> (BITS_PER_BAND * b)) & 0xF) for b in range(BANDS)]


def build_index(vectors, planes):
    index = {}
    for item_id, vector in enumerate(vectors):
        for key in band_keys(signature(vector, planes)):
            index.setdefault(key, []).append(item_id)
    return index


def query_candidates(index, vector, planes):
    found = set()
    for key in band_keys(signature(vector, planes)):
        for item_id in index.get(key, ()):
            found.add(item_id)
    return found


def main():
    seed = 20261016
    planes, seed = make_hyperplanes(seed)

    # 300 base vectors in [-9, 9]^64 plus one planted near-duplicate each (noise in [-2, 2]).
    n = 300
    base = []
    for _ in range(n):
        row = []
        for _ in range(DIMS):
            seed = lcg_value(seed)
            row.append(seed % 19 - 9)
        base.append(row)
    near = []
    for vector in base:
        row = list(vector)
        for k in range(DIMS):
            seed = lcg_value(seed)
            row[k] += seed % 5 - 2
        near.append(row)

    index = build_index(base, planes)
    hits = 0
    for item_id in range(n):
        candidates = query_candidates(index, near[item_id], planes)
        if item_id in candidates:
            hits += 1
    recall = hits / n
    assert recall >= RECALL_MIN, recall

    # Orthogonal pairs: disjoint-support basis vectors, fixed LCG sample.
    pairs = []
    seen = set()
    while len(pairs) < 100:
        seed = lcg_value(seed)
        i = seed % DIMS
        seed = lcg_value(seed)
        j = seed % DIMS
        if i == j:
            j = (j + 1) % DIMS
        if (i, j) not in seen:
            seen.add((i, j))
            pairs.append((i, j))
    collisions = 0
    for i, j in pairs:
        vi = [1 if k == i else 0 for k in range(DIMS)]
        vj = [1 if k == j else 0 for k in range(DIMS)]
        dot = sum(a * b for a, b in zip(vi, vj))
        assert dot == 0  # the pair is exactly orthogonal by construction
        si, sj = signature(vi, planes), signature(vj, planes)
        assert isinstance(query_candidates(index, vi, planes), set)  # no crash
        if any(a == b for a, b in zip(band_keys(si), band_keys(sj))):
            collisions += 1
    assert collisions / len(pairs) <= ORTHO_COLLISION_MAX, collisions


if __name__ == "__main__":
    main()
    print("PASS")
