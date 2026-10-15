"""Task 453 - Implement circulation with lower bounds (flow transform). (Tier: Advanced, CTCI Ch 4)

A circulation assigns every edge (u,v) a flow in [lower, upper] with conservation at every
vertex. Transform: route the mandatory `lower` part first, which leaves each vertex with a
balance b[v] = (sum of lowers entering v) - (sum of lowers leaving v). Then find a flow in the
residual network (edge capacities upper - lower) that absorbs the balances: a super source SS
feeds every v with b[v] > 0, every v with b[v] < 0 drains into a super sink TT. The circulation
is feasible iff the SS->TT max flow saturates all SS arcs; edge flows are lower + used residual.

The transform also handles s-t flow with a required value F: add the edge t->s with bounds
[F, F] and ask for a feasible circulation. Hand cases below are re-derived by hand (triangle
[1,3] works with 1 per edge; a vertex whose outgoing upper bound is below its incoming lower
bound makes the instance infeasible; the s-t example has max flow 3 = 2 via 0-1-3 + 1 via
0-2-3, so F=3 is feasible and F=4 is not). Feasibility is cross-checked against brute-force
enumeration of all integer flow assignments on small LCG instances.
"""

from collections import deque

INF = 10 ** 9


class MaxFlow:
    """Edmonds-Karp on an explicit arc list; each arc keeps its own reverse residual."""

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

    def used(self, idx):
        return self.initial[idx] - self.cap[idx]

    def max_flow(self, s, t):
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
                idx = parent_edge[v]
                aug = min(aug, self.cap[idx])
                v = self.to[idx ^ 1]
            v = t
            while v != s:
                idx = parent_edge[v]
                self.cap[idx] -= aug
                self.cap[idx ^ 1] += aug
                v = self.to[idx ^ 1]
            flow += aug


def circulation(n, edges):
    """Returns (feasible, flows) for edges = (u, v, lower, upper), or (False, None)."""
    ss = n
    tt = n + 1
    network = MaxFlow(n + 2)
    balance = [0] * n
    arc_ids = []
    for u, v, lower, upper in edges:
        balance[v] += lower
        balance[u] -= lower
        arc_ids.append(network.add_edge(u, v, upper - lower))
    required = 0
    for v in range(n):
        if balance[v] > 0:
            network.add_edge(ss, v, balance[v])
            required += balance[v]
        elif balance[v] < 0:
            network.add_edge(v, tt, -balance[v])
    used = network.max_flow(ss, tt)
    if used != required:
        return False, None
    flows = []
    for (u, v, lower, upper), idx in zip(edges, arc_ids):
        flows.append(lower + network.used(idx))
    return True, flows


def circulation_with_required_value(n, edges, s, t, required_value):
    """s-t flow of exactly `required_value` via the t->s [F, F] circulation transform."""
    extended = list(edges) + [(t, s, required_value, required_value)]
    return circulation(n, extended)


def brute_feasible(n, edges):
    """Enumerate every integer flow assignment in [lower, upper] and test conservation."""
    m = len(edges)
    values = [0] * m
    while True:
        balance = [0] * n
        for i in range(m):
            u, v, lower, upper = edges[i]
            f = lower + values[i]
            balance[u] -= f
            balance[v] += f
        if all(b == 0 for b in balance):
            return True
        j = 0
        while j < m:
            if values[j] + 1 < edges[j][3] - edges[j][2] + 1:
                values[j] += 1
                break
            values[j] = 0
            j += 1
        if j == m:
            return False


def check_flow_valid(n, edges, flows):
    balance = [0] * n
    for (u, v, lower, upper), f in zip(edges, flows):
        assert lower <= f <= upper, (u, v, lower, upper, f)
        balance[u] -= f
        balance[v] += f
    assert all(b == 0 for b in balance), balance


def lcg_instances(count, seed):
    """Small LCG instances: n in 3..4, up to 5 distinct ordered pairs, narrow bounds."""
    instances = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = 3 + seed % 2
        seed = (seed * 75 + 74) % 65537
        m = 3 + seed % 3
        pool = [(u, v) for u in range(n) for v in range(n) if u != v]
        edges = []
        for _ in range(min(m, len(pool))):
            seed = (seed * 75 + 74) % 65537
            idx = seed % len(pool)
            u, v = pool[idx]
            pool[idx] = pool[-1]
            pool.pop()
            seed = (seed * 75 + 74) % 65537
            lower = seed % 3
            seed = (seed * 75 + 74) % 65537
            upper = lower + seed % 3
            edges.append((u, v, lower, upper))
        instances.append((n, edges))
    return instances


def main():
    triangle = [(0, 1, 1, 3), (1, 2, 1, 3), (2, 0, 1, 3)]
    feasible, flows = circulation(3, triangle)
    assert feasible and flows == [1, 1, 1]
    check_flow_valid(3, triangle, flows)
    assert brute_feasible(3, triangle)

    tight_out = [(0, 1, 2, 3), (1, 2, 0, 1), (2, 0, 1, 3)]
    assert not circulation(3, tight_out)[0]
    assert not brute_feasible(3, tight_out)

    two_way = [(0, 1, 3, 5), (1, 0, 0, 5)]
    feasible, flows = circulation(2, two_way)
    assert feasible and flows == [3, 3]
    check_flow_valid(2, two_way, flows)
    assert brute_feasible(2, two_way)

    too_much_in = [(0, 1, 4, 5), (1, 0, 0, 2)]
    assert not circulation(2, too_much_in)[0]
    assert not brute_feasible(2, too_much_in)

    disjoint = [(0, 1, 1, 2), (1, 0, 1, 2), (2, 3, 0, 1), (3, 2, 0, 1)]
    feasible, flows = circulation(4, disjoint)
    assert feasible
    check_flow_valid(4, disjoint, flows)
    assert brute_feasible(4, disjoint)

    dead_end = [(0, 1, 1, 1)]
    assert not circulation(2, dead_end)[0]
    assert not brute_feasible(2, dead_end)

    with_isolated = [(0, 1, 1, 1), (1, 0, 1, 1)]
    feasible, flows = circulation(3, with_isolated)
    assert feasible and flows == [1, 1]
    check_flow_valid(3, with_isolated, flows)
    assert brute_feasible(3, with_isolated)

    st_edges = [(0, 1, 0, 2), (1, 3, 0, 2), (0, 2, 0, 1), (2, 3, 0, 1)]
    feasible, flows = circulation_with_required_value(4, st_edges, 0, 3, 3)
    assert feasible and flows[-1] == 3
    check_flow_valid(4, st_edges + [(3, 0, 3, 3)], flows)
    feasible, flows = circulation_with_required_value(4, st_edges, 0, 3, 4)
    assert not feasible

    feasible_count = 0
    infeasible_count = 0
    for seed in (4531, 4532, 4533):
        for n, edges in lcg_instances(8, seed):
            feasible, flows = circulation(n, edges)
            expected = brute_feasible(n, edges)
            assert feasible == expected, (n, edges, feasible, expected)
            if feasible:
                check_flow_valid(n, edges, flows)
                feasible_count += 1
            else:
                infeasible_count += 1
    assert feasible_count >= 1 and infeasible_count >= 1
    assert feasible_count + infeasible_count == 24


if __name__ == "__main__":
    main()
    print("PASS")
