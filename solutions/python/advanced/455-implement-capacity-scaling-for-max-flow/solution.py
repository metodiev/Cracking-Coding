"""Task 455 - Implement capacity scaling for max flow. (Tier: Advanced, CTCI Ch 4)

Capacity scaling (Edmonds-Karp variant): let D be the largest power of 2 not exceeding the
largest capacity. Phase D searches for augmenting paths that use only residual arcs of capacity
>= D and pushes along them until none remains; D halves each phase until D = 1, after which the
residual graph is the plain one. Each phase therefore works on a sparser "big-capacity" graph,
and there are floor(log2 C) + 1 phases with at most 2E augmentations each.

The hand cases document the phases: a single edge of capacity 6 is saturated by one D=4 path push
of amount 6 (bottleneck pushes may exceed D); parallel arcs of 3 and 4 push 4 (D=4) then 3
(D=2); a chain with capacities 1000/999 needs phases 512..1 = 10 phases and yields 999 in one
push. On LCG graphs the value always equals Edmonds-Karp; invariants checked for every
augmentation: its amount is at least the current phase D, the amounts sum to the flow value, and
each graph reports the documented phase count.
"""

from collections import deque

INF = 10 ** 9


class MaxFlow:
    """Arc-list flow network: to/cap arrays, reverse arc index is idx ^ 1."""

    def __init__(self, n):
        self.n = n
        self.to = []
        self.initial = []
        self.cap = []
        self.head = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity):
        idx = len(self.to)
        self.to.append(v)
        self.initial.append(capacity)
        self.cap.append(capacity)
        self.head[u].append(idx)
        self.to.append(u)
        self.initial.append(0)
        self.cap.append(0)
        self.head[v].append(idx + 1)
        return idx

    def augment(self, parent_edge, s, t):
        aug = INF
        v = t
        while v != s:
            aug = min(aug, self.cap[parent_edge[v]])
            v = self.to[parent_edge[v] ^ 1]
        v = t
        while v != s:
            idx = parent_edge[v]
            self.cap[idx] -= aug
            self.cap[idx ^ 1] += aug
            v = self.to[idx ^ 1]
        return aug

    def find_path(self, s, t, min_capacity):
        parent_edge = [-1] * self.n
        parent_edge[s] = -2
        q = deque([s])
        while q:
            u = q.popleft()
            for idx in self.head[u]:
                v = self.to[idx]
                if parent_edge[v] == -1 and self.cap[idx] >= min_capacity:
                    parent_edge[v] = idx
                    q.append(v)
        return parent_edge if parent_edge[t] != -1 else None

    def edmonds_karp(self, s, t):
        flow = 0
        while True:
            parent_edge = self.find_path(s, t, 1)
            if parent_edge is None:
                return flow
            flow += self.augment(parent_edge, s, t)

    def capacity_scaling(self, s, t):
        """Returns (value, phase_deltas, augmentations_per_phase, augmentation_amounts)."""
        max_capacity = 0
        for idx in range(0, len(self.to), 2):
            max_capacity = max(max_capacity, self.initial[idx])
        delta = 0
        if max_capacity >= 1:
            delta = 1
            while delta * 2 <= max_capacity:
                delta *= 2
        deltas = []
        per_phase = []
        amounts = []
        flow = 0
        while delta >= 1:
            deltas.append(delta)
            count = 0
            while True:
                parent_edge = self.find_path(s, t, delta)
                if parent_edge is None:
                    break
                amount = self.augment(parent_edge, s, t)
                assert amount >= delta
                amounts.append(amount)
                flow += amount
                count += 1
            per_phase.append(count)
            delta //= 2
        return flow, deltas, per_phase, amounts


def build_network(n, edges):
    network = MaxFlow(n)
    for u, v, c in edges:
        network.add_edge(u, v, c)
    return network


def lcg_flow_graphs(count, seed, min_n=4, max_n=7):
    graphs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        edges = []
        for node in range(n - 1):
            seed = (seed * 75 + 74) % 65537
            edges.append((node, node + 1, 1 + seed % 500))
        for _ in range(n + 3):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            seed = (seed * 75 + 74) % 65537
            if u != v:
                edges.append((u, v, 1 + seed % 500))
        graphs.append((n, edges))
    return graphs


def phase_count_for(max_capacity):
    delta = 1
    count = 0
    while delta <= max_capacity:
        count += 1
        delta *= 2
    return count


def main():
    single = build_network(2, [(0, 1, 6)])
    value, deltas, per_phase, amounts = single.capacity_scaling(0, 1)
    assert value == 6
    assert deltas == [4, 2, 1]
    assert per_phase == [1, 0, 0]
    assert amounts == [6]

    parallel = build_network(2, [(0, 1, 3), (0, 1, 4)])
    value, deltas, per_phase, amounts = parallel.capacity_scaling(0, 1)
    assert value == 7
    assert deltas == [4, 2, 1]
    assert per_phase == [1, 1, 0]
    assert amounts == [4, 3]
    assert sum(amounts) == value

    chain = build_network(3, [(0, 1, 1000), (1, 2, 999)])
    value, deltas, per_phase, amounts = chain.capacity_scaling(0, 2)
    assert value == 999
    assert len(deltas) == 10 and deltas[0] == 512 and deltas[-1] == 1
    assert sum(amounts) == value
    assert build_network(3, [(0, 1, 1000), (1, 2, 999)]).edmonds_karp(0, 2) == 999

    tiny = build_network(2, [(0, 1, 1)])
    value, deltas, per_phase, amounts = tiny.capacity_scaling(0, 1)
    assert value == 1 and deltas == [1] and amounts == [1]

    zero = build_network(2, [])
    value, deltas, per_phase, amounts = zero.capacity_scaling(0, 1)
    assert value == 0 and deltas == [] and amounts == []

    clrs = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9),
            (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    value, deltas, per_phase, amounts = build_network(6, clrs).capacity_scaling(0, 5)
    assert value == 23
    assert deltas == [16, 8, 4, 2, 1]
    assert sum(amounts) == value
    assert build_network(6, clrs).edmonds_karp(0, 5) == 23

    checked = 0
    late_phase_work = 0
    total_augmentations = 0
    for seed in (4551, 4552, 4553):
        for n, edges in lcg_flow_graphs(8, seed):
            scaled = build_network(n, edges)
            value, deltas, per_phase, amounts = scaled.capacity_scaling(0, n - 1)
            max_capacity = max(c for _, _, c in edges)
            assert len(deltas) == phase_count_for(max_capacity), (n, edges)
            assert deltas[0] * 2 > max_capacity and deltas[0] <= max_capacity
            assert sum(amounts) == value
            assert sum(per_phase) == len(amounts)
            total_augmentations += len(amounts)
            if any(count > 0 for count in per_phase[1:]):
                late_phase_work += 1
            plain = build_network(n, edges)
            expected = plain.edmonds_karp(0, n - 1)
            assert value == expected, (n, edges, value, expected)
            checked += 1
    assert checked == 24
    assert total_augmentations > 0
    assert late_phase_work >= 1


if __name__ == "__main__":
    main()
    print("PASS")
