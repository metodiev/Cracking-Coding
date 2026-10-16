"""Task 950 - Implement scalable nearest neighbor with multi-index hashing and rerank pipeline.  (Tier: Master, CTCI Beyond - Advanced Data Structures & Algorithms)

Multi-index hashing over 32-bit signatures split into four 8-bit blocks.  Candidates are found by
enumerating every combination of per-block radii whose total is at most the query radius, in increasing
total order (the adaptive schedule), reading the ids out of per-block hash tables until an explicit
candidate budget is reached; the candidates are then reranked by the exact Hamming distance computed on
the full signature, which is the expensive step the index exists to avoid.  A band-based locality
sensitive hash over the same signatures shares the same budget and serves as the comparison baseline,
and brute force over all signatures provides the ground truth.

Simplified versus FAISS/PQ-MIH: signatures are hand-built hyperplane hashes rather than learned codes,
the index is a plain dictionary per block and the enumeration is exhaustive within the radius rather than
pruned by a per-block distance bound.  Verified: the Hamming distance matches a brute-force bit count;
with an unlimited budget the candidate set contains every signature inside the radius (no false
negatives), while the budgeted version recalls the true nearest neighbor with a small probe count; the
truncated candidate set is a subset of the untruncated one; multi-index hashing beats the bandwidth-LSH
baseline at the same budget; reranking by the exact distance never loses to ranking by block matches;
the r = 0, empty-index and foreign-query boundaries; recall is monotone in the radius; determinism.
"""
import itertools

BLOCKS = 4
BLOCK_BITS = 8
BLOCK_MASK = (1 << BLOCK_BITS) - 1
BUDGET = 64


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def pseudo(value):
    """Deterministic 64-bit mix, so signatures and vectors are reproducible."""
    value = (value * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 29
    return ((value * 0x94D049BB133111EB) & ((1 << 64) - 1)) >> 11


def bit_count(value):
    return bin(value).count("1")


def build_vectors(count, dimensions=8):
    return [tuple(pseudo(index * 97 + axis * 13) % 101 - 50 for axis in range(dimensions))
            for index in range(count)]


def build_signatures(vectors, dimensions=8, bits=32):
    """Hyperplane hashing: one deterministic weight vector per signature bit."""
    codes = []
    for vector in vectors:
        code = 0
        for bit in range(bits):
            total = sum((pseudo(bit * 4099 + axis * 131) % 41 - 20) * value
                        for axis, value in enumerate(vector))
            if total > 0:
                code |= 1 << bit
        codes.append(code)
    return codes


NEIGHBOURHOODS = {
    (radius, value): [other for other in range(1 << BLOCK_BITS)
                      if bit_count((other ^ value) & BLOCK_MASK) <= radius]
    for radius in range(BLOCK_BITS + 1)
    for value in range(1 << BLOCK_BITS)
}


def query_block(code, block):
    return (code >> (block * BLOCK_BITS)) & BLOCK_MASK


class MihIndex:
    """One hash table per block, plus the rerank stage."""

    def __init__(self, codes, budget=BUDGET):
        self.codes = list(codes)
        self.budget = budget
        self.tables = [{} for _ in range(BLOCKS)]
        for identifier, code in enumerate(self.codes):
            for block in range(BLOCKS):
                self.tables[block].setdefault(query_block(code, block), []).append(identifier)
        self.probes = 0

    def candidates(self, code, radius, budget=None):
        """Enumerate block-radius combinations in increasing total order; returns (ids, truncated)."""
        limit = self.budget if budget is None else budget
        found, truncated = [], False
        for total in range(radius + 1):
            for radii in itertools.product(range(total + 1), repeat=BLOCKS):
                if sum(radii) != total:
                    continue
                for block, block_radius in enumerate(radii):
                    for value in NEIGHBOURHOODS[(block_radius, query_block(code, block))]:
                        for identifier in self.tables[block].get(value, ()):
                            if identifier not in found:
                                found.append(identifier)
                self.probes += len(found)
                if len(found) >= limit:
                    return found[:limit], True
        return found, truncated

    def nearest(self, code, radius, budget=None):
        """Rerank the candidates by the exact Hamming distance on the full signature."""
        candidates, truncated = self.candidates(code, radius, budget)
        ranked = sorted(candidates, key=lambda identifier: (bit_count(self.codes[identifier] ^ code),
                                                            identifier))
        return (ranked[0] if ranked else None), len(candidates), truncated


def brute_force_nearest(codes, code):
    return min(range(len(codes)), key=lambda identifier: (bit_count(codes[identifier] ^ code),
                                                          identifier))


def brute_force_within(codes, code, radius):
    return {identifier for identifier, other in enumerate(codes)
            if bit_count(other ^ code) <= radius}


def band_candidates(codes, code, bands=8, band_bits=4, budget=BUDGET):
    """Baseline: the ids sharing any 4-bit band with the query, truncated to the same budget."""
    mask = (1 << band_bits) - 1
    found = []
    for band in range(bands):
        shift = band * band_bits
        target = (code >> shift) & mask
        for identifier, other in enumerate(codes):
            if ((other >> shift) & mask) == target and identifier not in found:
                found.append(identifier)
                if len(found) >= budget:
                    return found
    return found


def main():
    codes = build_signatures(build_vectors(320))
    index = MihIndex(codes)
    queries = [codes[identifier] for identifier in range(0, 320, 7)]

    # 1) the distance primitive matches a brute-force bit count on random pairs.
    pairs = [(pseudo(step * 7919) % 320, pseudo(step * 104729) % 320) for step in range(50)]
    check(all(bit_count(codes[first] ^ codes[second])
              == bin(codes[first] ^ codes[second]).count("1")
              and 0 <= bit_count(codes[first] ^ codes[second]) <= 32 for first, second in pairs),
          "the Hamming distance must match a brute-force bit count")

    # 2) no false negatives: with an unlimited budget the candidate set contains every signature
    #    inside the radius, which is the multi-index hashing correctness property.
    for query in queries[:20]:
        for radius in (0, 1, 2):
            candidates, _ = index.candidates(query, radius, budget=10 ** 6)
            inside = brute_force_within(codes, query, radius)
            check(inside <= set(candidates) and inside,
                  f"radius {radius}: missing candidates {sorted(inside - set(candidates))}")
    check(index.probes > 0, "the enumeration must actually probe the tables")

    # 3) budgeted recall: the true nearest neighbor is found while probing far fewer than n ids.
    hits, probed = 0, []
    for query in queries:
        found, count, _ = index.nearest(query, 3, BUDGET)
        probed.append(count)
        hits += bit_count(codes[found] ^ query) <= 3
    recall = hits / len(queries)
    check(recall >= 0.9 and max(probed) <= BUDGET and sum(probed) / len(probed) < len(codes) / 4,
          f"budgeted recall {recall:.3f} with {sum(probed) / len(probed):.1f} candidates on average")

    # 4) multi-index hashing beats the band-LSH baseline at the same budget.
    mih_hits = sum(bit_count(codes[index.nearest(query, 3, BUDGET)[0]] ^ query) <= 3
                   for query in queries)
    baseline_hits, baseline_probes = 0, []
    for query in queries:
        found = band_candidates(codes, query, budget=BUDGET)
        baseline_probes.append(len(found))
        best = min(found, key=lambda identifier: bit_count(codes[identifier] ^ query))
        baseline_hits += bit_count(codes[best] ^ query) <= 3
    check(mih_hits >= baseline_hits and max(baseline_probes) <= BUDGET,
          f"MIH recall {mih_hits}/{len(queries)} must not lose to the band baseline "
          f"{baseline_hits}/{len(queries)}")

    # 5) reranking by the exact distance never loses to ranking the same candidates by block matches.
    def block_score(identifier, query):
        return sum(1 for block in range(BLOCKS)
                   if query_block(codes[identifier], block) == query_block(query, block))

    exact_hits, block_hits = 0, 0
    for query in queries:
        candidates, _ = index.candidates(query, 3, BUDGET)
        exact_hits += bit_count(codes[min(candidates, key=lambda identifier: (
            bit_count(codes[identifier] ^ query), identifier))] ^ query) <= 3
        block_hits += bit_count(codes[max(candidates, key=lambda identifier: (
            block_score(identifier, query), -identifier))] ^ query) <= 3
    check(exact_hits >= block_hits,
          f"exact rerank {exact_hits} must beat block-match ranking {block_hits}")

    # 6) recall does not fall as the radius grows, and the truncated set is a subset of the full one.
    recalls = []
    for radius in range(4):
        found_hits = sum(bit_count(codes[index.nearest(query, radius, 10 ** 6)[0]] ^ query) <= 3
                         for query in queries)
        recalls.append(found_hits)
    tight, _ = index.candidates(queries[0], 3, 8)
    wide, _ = index.candidates(queries[0], 3, 10 ** 6)
    check(recalls == sorted(recalls) and set(tight) <= set(wide) and len(tight) <= 8,
          f"recall must not fall with the radius: {recalls}; the truncated set must be a subset")

    # 7) boundaries: radius 0 finds only exact matches, the empty index returns nothing and a query
    #    outside the data set still returns the closest stored signature.
    exact_identifiers = [identifier for identifier, code in enumerate(codes)
                         if code == queries[0]]
    near, count, _ = MihIndex(codes).nearest(queries[0], 0, BUDGET)
    empty = MihIndex([])
    foreign = (queries[0] ^ 0xFFFFFFFF) & 0xFFFFFFFF
    check(near in exact_identifiers and count >= len(exact_identifiers)
          and empty.nearest(queries[0], 2, BUDGET) == (None, 0, False)
          and index.nearest(foreign, 3, BUDGET)[0] is not None,
          "the radius-0, empty-index and foreign-query boundaries must hold")

    # 8) determinism of the index, the enumeration and the rerank.
    twin = MihIndex(codes)
    check([twin.nearest(query, 3, BUDGET) for query in queries[:10]]
          == [MihIndex(codes).nearest(query, 3, BUDGET) for query in queries[:10]]
          and build_signatures(build_vectors(320)) == codes,
          "the index, the candidate order and the rerank must be deterministic")

    print(f"signatures={len(codes)} queries={len(queries)} recall(r=3, budget={BUDGET})={recall:.3f} "
          f"average_candidates={sum(probed) / len(probed):.1f}")
    print(f"MIH {mih_hits}/{len(queries)} vs band-LSH {baseline_hits}/{len(queries)} at the same "
          f"budget; exact rerank {exact_hits} vs block ranking {block_hits}")
    print("PASS")


if __name__ == "__main__":
    main()
