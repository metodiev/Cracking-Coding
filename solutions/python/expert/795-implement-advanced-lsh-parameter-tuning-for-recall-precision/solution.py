"""Task 795 - Implement advanced LSH parameter tuning for recall/precision.  (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)

SimHash (random hyperplane LSH) hashes a vector to one bit per random hyperplane: the sign
of the projection.  For two vectors separated by the angle theta the probability that a
single hyperplane separates them on the same side is

    p(theta) = 1 - theta / pi

and with b bands of r bits each, two vectors become *candidates* as soon as one band agrees
completely, so

    P_candidate(theta) = 1 - (1 - p(theta)^r)^b.

That closed form is exactly the quantity a parameter search needs: recall is driven by
P_candidate at the largest angle that still counts as a neighbour, while precision (and the
number of candidate pairs to verify) grows with b and shrinks with r.  This file implements
the index, an exact pair-set recomputation for validation, a grid search over (b, r) under a
budget of total hash bits, and the tuning criterion "maximise measured recall subject to the
candidate count staying under the budget".

Verified: the measured bit-agreement frequency matches 1 - theta/pi for several controlled
angles, the measured candidate frequency matches 1 - (1 - p^r)^b, the band-hash candidate set
equals the exhaustive pairwise comparison of band keys, the tuned configuration reaches a
recall no worse than the untuned default at no more than the allowed candidate cost, recall
reaches 1.0 for a configuration whose theoretical candidate probability is above 0.999,
near-orthogonal pairs stay out of the candidate set, degenerate inputs (zero vectors, a
single item, mismatched dimensions) are rejected, and the index is reproducible for a fixed
seed.

Simplified versus a production ANN system: no multi-probe, no query-adaptive early
termination, no buckets stored off-process, and the quality measures are computed by brute
force on a few dozen low-dimensional vectors so that the whole sweep finishes in well under a
second.
"""
import math

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


def gauss(state):
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / 9007199254740992.0
    p = (u2 >> 11) / 9007199254740992.0
    if r < 1e-12:
        r = 1e-12
    return state, math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * p)


# ------------------------------------------------------------------ LSH basics
def random_hyperplanes(dim, count, seed):
    """count unit vectors drawn isotropically in R^dim."""
    planes = []
    state = seed
    for _ in range(count):
        raw = []
        for _c in range(dim):
            state, value = gauss(state)
            raw.append(value)
        norm = math.sqrt(math.fsum(value * value for value in raw))
        planes.append([value / norm for value in raw])
    return planes, state


def signature(vector, planes):
    """SimHash signature: one bit per hyperplane, 1 when the projection is non-negative."""
    if len(vector) != len(planes[0]):
        raise ValueError("dimension mismatch between vector and hyperplanes")
    return tuple(1 if math.fsum(a * b for a, b in zip(vector, plane)) >= 0.0 else 0
                 for plane in planes)


def band_keys(bits, bands, rows):
    if len(bits) != bands * rows:
        raise ValueError("bit budget must equal bands * rows")
    return tuple(tuple(bits[k * rows:(k + 1) * rows]) for k in range(bands))


def cosine(u, v):
    nu = math.sqrt(math.fsum(x * x for x in u))
    nv = math.sqrt(math.fsum(x * x for x in v))
    if nu < 1e-12 or nv < 1e-12:
        raise ValueError("cosine is undefined for a zero vector")
    return math.fsum(a * b for a, b in zip(u, v)) / (nu * nv)


def angle_between(u, v):
    return math.acos(max(-1.0, min(1.0, cosine(u, v))))


def bit_agreement_probability(theta):
    """The SimHash collision probability for a pair separated by theta."""
    return 1.0 - theta / math.pi


def candidate_probability(theta, bands, rows):
    p = bit_agreement_probability(theta)
    return 1.0 - (1.0 - p ** rows) ** bands


# --------------------------------------------------------------------- index
class LSHIndex:
    """b bands of r bits; two items are candidates when one band collides exactly."""

    def __init__(self, dim, bands, rows, seed):
        if bands < 1 or rows < 1:
            raise ValueError("bands and rows must be positive")
        self.bands = bands
        self.rows = rows
        self.dim = dim
        self.planes, _state = random_hyperplanes(dim, bands * rows, seed)
        self.items = []
        self.band_tables = [{} for _ in range(bands)]
        self.signatures = []

    def add(self, item):
        bits = signature(item, self.planes)
        if len(bits) != self.bands * self.rows:
            raise ValueError("signature length must equal bands * rows")
        keys = band_keys(bits, self.bands, self.rows)
        index = len(self.items)
        self.items.append(item)
        self.signatures.append(bits)
        for band, key in enumerate(keys):
            self.band_tables[band].setdefault(key, []).append(index)
        return index

    def candidate_pairs(self):
        """Every pair sharing at least one band key, via the band tables."""
        pairs = set()
        for table in self.band_tables:
            for bucket in table.values():
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        pairs.add((bucket[i], bucket[j]))
        return pairs

    def brute_force_candidate_pairs(self):
        """The same pair set computed by direct comparison of signatures (independent path)."""
        pairs = set()
        keys = [band_keys(bits, self.bands, self.rows) for bits in self.signatures]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if any(a == b for a, b in zip(keys[i], keys[j])):
                    pairs.add((i, j))
        return pairs


def true_neighbour_pairs(vectors, threshold_angle):
    """All pairs whose cosine angle is at most the threshold."""
    pairs = set()
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if angle_between(vectors[i], vectors[j]) <= threshold_angle:
                pairs.add((i, j))
    return pairs


def evaluate(bands, rows, vectors, seed, threshold_angle):
    """Recall, candidate count and cost for one (bands, rows) configuration."""
    index = LSHIndex(len(vectors[0]), bands, rows, seed)
    for vector in vectors:
        index.add(vector)
    candidates = index.candidate_pairs()
    truth = true_neighbour_pairs(vectors, threshold_angle)
    hits = len(candidates & truth)
    recall = hits / len(truth) if truth else 1.0
    precision = hits / len(candidates) if candidates else 1.0
    return {"bands": bands, "rows": rows, "recall": recall, "precision": precision,
            "candidates": len(candidates), "hits": hits, "truth": len(truth),
            "cost": bands * rows}


def tune(vectors, seed, threshold_angle, bit_budget, candidate_budget,
         grid=((1, 16), (2, 8), (4, 4), (8, 2), (16, 1), (4, 8), (8, 4), (2, 12))):
    """Grid search: best recall under the candidate budget, then the cheapest such config."""
    affordable = [(bands, rows) for bands, rows in grid if bands * rows <= bit_budget]
    check(bool(affordable), "the grid must contain at least one configuration inside the budget")
    results = [evaluate(bands, rows, vectors, seed, threshold_angle) for bands, rows in affordable]
    feasible = [entry for entry in results if entry["candidates"] <= candidate_budget]
    check(bool(feasible), "the grid must contain at least one affordable configuration")
    feasible.sort(key=lambda entry: (-entry["recall"], entry["candidates"], entry["cost"]))
    return feasible[0], results


# ------------------------------------------------------------------ data sets
def make_vectors(seed, count, dim, clusters, spread=0.35):
    """A few tight clusters, so the neighbour structure is non-trivial but well defined."""
    vectors = []
    state = seed
    centres = []
    for _ in range(clusters):
        centre = []
        for _c in range(dim):
            state, value = gauss(state)
            centre.append(value)
        centres.append(centre)
    for item in range(count):
        centre = centres[item % clusters]
        vector = []
        for component in range(dim):
            state, noise = gauss(state)
            vector.append(centre[component] + spread * noise)
        vectors.append(vector)
    return vectors, centres


def controlled_pair(theta, seed):
    """Two unit vectors with an exactly known angle."""
    state, noise = splitmix64(seed)
    return ([1.0, 0.0], [math.cos(theta), math.sin(theta)])


def main():
    DIM = 12
    THETA = 0.55                                     # neighbours are pairs within ~31.5 degrees

    # --- 1. the theory behind SimHash: bit agreement vs 1 - theta/pi --------
    planes, _state = random_hyperplanes(2, 4000, 795001)
    deviations = []
    for theta in (0.1, 0.5, 1.0, 2.0):
        u, v = controlled_pair(theta, 795002)
        agree = 0
        for plane in planes:
            agree += signature(u, [plane])[0] == signature(v, [plane])[0]
        measured = agree / len(planes)
        deviations.append(abs(measured - bit_agreement_probability(theta)))
        check(deviations[-1] < 0.03,
              "measured bit agreement %.4f must match 1 - theta/pi = %.4f for theta=%.2f"
              % (measured, bit_agreement_probability(theta), theta))
    check(len(deviations) == 4, "four angles must be probed")

    # --- 2. candidate probability matches 1 - (1 - p^r)^b -------------------
    vectors, _centres = make_vectors(795010, 60, DIM, clusters=6)
    index = LSHIndex(DIM, 4, 3, 795020)
    for vector in vectors:
        index.add(vector)
    measured_pairs = 0
    predicted = 0.0
    sampled = 0
    for i in range(0, len(vectors), 3):
        for j in range(i + 1, len(vectors), 4):
            sampled += 1
            keys_i = band_keys(index.signatures[i], 4, 3)
            keys_j = band_keys(index.signatures[j], 4, 3)
            measured_pairs += any(a == b for a, b in zip(keys_i, keys_j))
            predicted += candidate_probability(angle_between(vectors[i], vectors[j]), 4, 3)
    measured_rate = measured_pairs / sampled
    predicted_rate = predicted / sampled
    check(sampled >= 40, "the candidate-rate probe must cover enough pairs")
    check(abs(measured_rate - predicted_rate) < 0.15,
          "measured candidate rate %.4f must track the closed form %.4f"
          % (measured_rate, predicted_rate))
    check(index.brute_force_candidate_pairs() == index.candidate_pairs(),
          "the band tables must reproduce the exhaustive band-key comparison")

    # --- 3. tuning: recall against a candidate budget -----------------------
    bit_budget = 32
    candidate_budget = 400
    best, results = tune(vectors, 795030, THETA, bit_budget, candidate_budget)
    check(best["cost"] <= bit_budget,
          "the tuned configuration must respect the bit budget: %d" % best["cost"])
    check(best["candidates"] <= candidate_budget,
          "the tuned configuration must respect the candidate budget: %d" % best["candidates"])
    check(all(entry["recall"] <= best["recall"] + 1e-12 for entry in results
              if entry["candidates"] <= candidate_budget),
          "no affordable configuration may beat the tuned recall")
    default = evaluate(2, 8, vectors, 795030, THETA)
    check(default["cost"] < bit_budget, "the default configuration must be cheaper in bits")
    check(best["recall"] >= 1.5 * default["recall"],
          "tuning must improve recall substantially: tuned %.4f vs default %.4f"
          % (best["recall"], default["recall"]))
    check(best["recall"] > 0.6, "the tuned recall must be high, got %.4f" % best["recall"])

    # --- 4. a configuration with P_candidate > 0.999 finds everything -------
    high = evaluate(16, 1, vectors, 795030, THETA)
    guarantee = candidate_probability(THETA, 16, 1)
    check(guarantee > 0.999, "the 16x1 configuration must guarantee the candidate property")
    check(high["recall"] == 1.0,
          "recall must be exactly 1.0 when the theory guarantees candidacy, got %.4f"
          % high["recall"])
    orthogonal = candidate_probability(math.pi / 2, 4, 8)
    check(orthogonal < 0.05,
          "near-orthogonal pairs must stay out of the candidate set with high probability, "
          "got %.6f" % orthogonal)
    index4 = LSHIndex(DIM, 4, 8, 795040)
    for vector in vectors:
        index4.add(vector)
    far_pairs = {(i, j) for i in range(len(vectors)) for j in range(i + 1, len(vectors))
                 if angle_between(vectors[i], vectors[j]) > 1.4}
    hits = len(index4.candidate_pairs() & far_pairs)
    rate = hits / max(1, len(far_pairs))
    check(rate < 4.0 * orthogonal + 0.02,
          "the empirical near-orthogonal collision rate %.4f must stay near the closed form %.4f"
          % (rate, orthogonal))

    # --- 5. boundary cases --------------------------------------------------
    same = [0.3] * DIM
    twin = LSHIndex(DIM, 4, 4, 795050)
    twin.add(same)
    twin.add(list(same))
    check(twin.candidate_pairs() == {(0, 1)},
          "identical vectors must always share a full signature")
    check((0, 1) in twin.candidate_pairs(), "identical vectors must be candidates")
    raised = False
    try:
        cosine([0.0] * DIM, same)
    except ValueError:
        raised = True
    check(raised, "the cosine of a zero vector must be rejected")
    raised = False
    try:
        signature([0.0] * (DIM + 1), twin.planes)
    except ValueError:
        raised = True
    check(raised, "a dimension mismatch must be rejected")
    raised = False
    try:
        LSHIndex(DIM, 0, 4, 795051)
    except ValueError:
        raised = True
    check(raised, "a zero band count must be rejected")
    lone = LSHIndex(DIM, 3, 4, 795052)
    lone.add(same)
    check(lone.candidate_pairs() == set(), "a single item must produce no candidate pairs")

    # --- 6. determinism ----------------------------------------------------
    repeat = evaluate(best["bands"], best["rows"], vectors, 795030, THETA)
    check(repeat["candidates"] == best["candidates"] and repeat["recall"] == best["recall"],
          "an identical seed must reproduce the index and the measurements")
    different = evaluate(best["bands"], best["rows"], vectors, 795031, THETA)
    check(different["hits"] > 0, "a different plane seed must still find true candidates")

    print("bit agreement deviation from 1-theta/pi: %s"
          % [round(value, 4) for value in deviations])
    print("candidate rate: measured %.4f vs closed form %.4f over %d pairs"
          % (measured_rate, predicted_rate, sampled))
    print("tuned configuration b=%d r=%d: recall %.4f, precision %.4f, candidates %d (budget %d)"
          % (best["bands"], best["rows"], best["recall"], best["precision"],
             best["candidates"], candidate_budget))
    print("default b=2 r=8: recall %.4f, candidates %d" % (default["recall"],
                                                           default["candidates"]))
    print("16x1 configuration: recall %.4f with P_candidate %.6f"
          % (high["recall"], guarantee))
    print("PASS")


if __name__ == "__main__":
    main()
