"""Task 464 - Implement reservoir sampling with weighted probabilities. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Weighted reservoir sampling with the A-Res algorithm (Efraimidis & Spirakis):
each item is assigned the key u^(1/w) with u uniform in (0, 1] drawn from a FIXED
LCG as u = (seed + 1) / 65537, and the reservoir keeps the k largest keys (a
min-heap of (key, item); an equal key is displaced only by a smaller item index,
as documented). For k = 1 an item with weight w_i is selected with probability
exactly w_i / W, which the statistical checks below exercise.

Checks (all deterministic: fixed LCG seed, fixed trial counts):
  * weights [1,2,3,4], 20000 trials, tolerance TOL = 0.02: every empirical
    selection frequency must lie within TOL of the weight ratio (measured max
    deviation ~0.005); the per-trial reservoir holds exactly one item and every
    item appears at most once per trial.
  * equal weights [1,1,1]: frequencies must match the uniform ratio 1/3 +- TOL.
  * weights 1..10 with k = 3, 3000 trials: each trial yields 3 DISTINCT items,
    the selection counts sum to 3000 * k, and heavier items are selected more
    often than lighter ones (a documented monotonicity sanity check).
"""
import heapq

TOL = 0.02


class WeightedReservoir:
    def __init__(self, size):
        self.size = size
        self.heap = []  # (key, item) min-heap; kept as the `size` largest keys

    def add(self, item, weight, u):
        key = u ** (1.0 / weight)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, (key, item))
        elif (key, item) > self.heap[0]:
            heapq.heapreplace(self.heap, (key, item))

    def sample(self):
        return sorted(item for _, item in self.heap)


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def draw_u(seed):
    seed = lcg_value(seed)
    return (seed + 1) / 65537.0, seed


def sample_once(items, weights, size, seed):
    reservoir = WeightedReservoir(size)
    for item, weight in zip(items, weights):
        u, seed = draw_u(seed)
        reservoir.add(item, weight, u)
    return reservoir.sample(), seed


def main():
    # Weights [1,2,3,4]: selection frequencies must track the ratios 0.1/0.2/0.3/0.4.
    weights = [1, 2, 3, 4]
    items = list(range(len(weights)))
    trials = 20000
    seed = 99
    counts = [0] * len(items)
    for _ in range(trials):
        picked, seed = sample_once(items, weights, 1, seed)
        assert len(picked) == 1  # exactly one item per trial
        counts[picked[0]] += 1
    total = sum(weights)
    max_dev = 0.0
    for i, w in enumerate(weights):
        expected = w / total
        observed = counts[i] / trials
        max_dev = max(max_dev, abs(observed - expected))
        assert abs(observed - expected) <= TOL, (i, observed, expected)
    assert sum(counts) == trials  # every item selected-exactly-once-per-trial
    assert counts[3] > counts[0]
    assert max_dev <= 0.01  # measured max deviation stays well inside TOL

    # Equal weights: the same machinery must produce the uniform ratio 1/3.
    seed = 12345
    counts = [0, 0, 0]
    trial = 0
    while trial < 20000:
        picked, seed = sample_once([0, 1, 2], [1, 1, 1], 1, seed)
        counts[picked[0]] += 1
        trial += 1
    for i in range(3):
        assert abs(counts[i] / 20000 - 1.0 / 3) <= TOL, (i, counts[i])

    # Weights 1..10 with k = 3: distinctness, count bookkeeping, monotonicity.
    ws = list(range(1, 11))
    size = 3
    trials = 3000
    seed = 777
    counts = [0] * len(ws)
    for _ in range(trials):
        picked, seed = sample_once(list(range(len(ws))), ws, size, seed)
        assert len(picked) == size
        assert len(set(picked)) == size  # each item at most once per trial
        for item in picked:
            counts[item] += 1
    assert sum(counts) == trials * size
    assert counts[9] > counts[0]


if __name__ == "__main__":
    main()
    print("PASS")
