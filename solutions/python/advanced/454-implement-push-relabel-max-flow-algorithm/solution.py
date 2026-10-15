"""Task 454 - Implement push-relabel max-flow algorithm. (Tier: Advanced, CTCI Ch 4)

Generic FIFO push-relabel (a.k.a. preflow-push): start with a preflow that saturates every arc
out of the source, give the source height n and everything else 0, then repeatedly discharge an
active vertex: push along an admissible arc (h[u] == h[v] + 1), otherwise relabel u to
min(h[v]) + 1 over residual arcs. Not counting saturating pushes, each push/relabel pair moves
excess strictly toward the sink, so the excess accumulating at t is the max flow.

Checked against Edmonds-Karp on LCG graphs (value equality, capacity bounds and conservation of
both flows), plus the CLRS 23 hand graph whose max flow is 23 = 12+11/... re-derived: 0->1(16)
and 0->2(13) feed 1->3(12)+1->2(10) and 2->4(14); the two sink arcs 3->5(20), 4->5(4) bound the
flow at min(16+13, 12+14+... ) = 23 (CLRS example 26.1).
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

    def flow_on(self, idx):
        return self.initial[idx] - self.cap[idx]

    def push_relabel(self, s, t):
        """FIFO preflow-push; returns (max flow, push count, relabel count)."""
        n = self.n
        height = [0] * n
        excess = [0] * n
        height[s] = n
        in_queue = [False] * n
        queue = deque()
        for idx in self.head[s]:
            amount = self.cap[idx]
            if amount > 0:
                v = self.to[idx]
                self.cap[idx] -= amount
                self.cap[idx ^ 1] += amount
                excess[v] += amount
                excess[s] -= amount
        pushes = relabels = 0
        for v in range(n):
            if v != s and v != t and excess[v] > 0:
                queue.append(v)
                in_queue[v] = True
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            while excess[u] > 0:
                pushed = False
                for idx in self.head[u]:
                    v = self.to[idx]
                    if self.cap[idx] > 0 and height[u] == height[v] + 1:
                        amount = min(excess[u], self.cap[idx])
                        self.cap[idx] -= amount
                        self.cap[idx ^ 1] += amount
                        excess[u] -= amount
                        excess[v] += amount
                        pushes += 1
                        if v != s and v != t and not in_queue[v] and excess[v] > 0:
                            queue.append(v)
                            in_queue[v] = True
                        pushed = True
                        break
                if not pushed:
                    best = INF
                    for idx in self.head[u]:
                        if self.cap[idx] > 0 and height[self.to[idx]] < best:
                            best = height[self.to[idx]]
                    height[u] = best + 1
                    relabels += 1
        return excess[t], pushes, relabels

    def edmonds_karp(self, s, t):
        flow = 0
        while True:
            parent_edge = [-1] * self.n
            parent_edge[s] = -2
            q = deque([s])
            while q:
                u = q.popleft()
                for idx in self.head[u]:
                    v = self.to[idx]
                    if parent_edge[v] == -1 and self.cap[idx] > 0:
                        parent_edge[v] = idx
                        q.append(v)
            if parent_edge[t] == -1:
                return flow
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
            flow += aug


def build_network(n, edges):
    network = MaxFlow(n)
    ids = [network.add_edge(u, v, c) for u, v, c in edges]
    return network, ids


def check_flow_valid(n, edges, ids, network, s, t, value):
    balance = [0] * n
    for (u, v, c), idx in zip(edges, ids):
        f = network.flow_on(idx)
        assert 0 <= f <= c, (u, v, f, c)
        balance[u] -= f
        balance[v] += f
    assert balance[t] == value, (balance[t], value)
    assert balance[s] == -value, (balance[s], value)
    for node in range(n):
        if node != s and node != t:
            assert balance[node] == 0, (node, balance[node])


def lcg_flow_graphs(count, seed, min_n=4, max_n=8):
    graphs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        edges = []
        for node in range(n - 1):
            seed = (seed * 75 + 74) % 65537
            edges.append((node, node + 1, 1 + seed % 20))
        for _ in range(n + 3):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            seed = (seed * 75 + 74) % 65537
            if u != v:
                edges.append((u, v, 1 + seed % 20))
        graphs.append((n, edges))
    return graphs


def main():
    clrs = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9),
            (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    network, ids = build_network(6, clrs)
    value, pushes, relabels = network.push_relabel(0, 5)
    assert value == 23, value
    assert pushes > 0 and relabels > 0
    check_flow_valid(6, clrs, ids, network, 0, 5, 23)
    network, ids = build_network(6, clrs)
    assert network.edmonds_karp(0, 5) == 23

    single = [(0, 1, 5)]
    network, ids = build_network(2, single)
    assert network.push_relabel(0, 1)[0] == 5
    check_flow_valid(2, single, ids, network, 0, 1, 5)

    parallel = [(0, 1, 3), (0, 1, 4), (1, 2, 2)]
    network, ids = build_network(3, parallel)
    value = network.push_relabel(0, 2)[0]
    assert value == 2, value
    network, ids = build_network(3, parallel)
    assert network.edmonds_karp(0, 2) == 2

    no_path = [(0, 1, 9), (2, 3, 9)]
    network, ids = build_network(4, no_path)
    assert network.push_relabel(0, 3)[0] == 0
    network, ids = build_network(3, [(0, 1, 0), (1, 2, 5)])
    assert network.push_relabel(0, 2)[0] == 0

    checked = 0
    total_pushes = 0
    total_relabels = 0
    for seed in (4541, 4542, 4543, 4544):
        for n, edges in lcg_flow_graphs(8, seed):
            network, ids = build_network(n, edges)
            value, pushes, relabels = network.push_relabel(0, n - 1)
            total_pushes += pushes
            total_relabels += relabels
            check_flow_valid(n, edges, ids, network, 0, n - 1, value)
            oracle_network, _ = build_network(n, edges)
            expected = oracle_network.edmonds_karp(0, n - 1)
            assert value == expected, (n, edges, value, expected)
            checked += 1
    assert checked == 32
    assert total_pushes > 0 and total_relabels > 0


if __name__ == "__main__":
    main()
    print("PASS")
