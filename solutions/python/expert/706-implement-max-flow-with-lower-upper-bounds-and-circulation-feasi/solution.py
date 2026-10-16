"""Task 706 - Implement max flow with lower/upper bounds and circulation feasibility.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)

Every edge carries both a lower bound l and an upper bound u, so a "flow" is an integer vector
with l <= f(e) <= u plus conservation.  The standard reduction turns that into an ordinary max
flow problem, and this file implements all three layers of it:

  * transform -- send l units along every edge unconditionally; f = l + x with 0 <= x <= u - l.
    Node v then has a forced imbalance b(v) = sum of l into v minus sum of l out of v;
  * feasibility -- a super source SS feeds every node with b(v) > 0 exactly b(v) units, every
    node with b(v) < 0 drains to a super sink ST, and a maximum flow saturating all SS edges
    exists iff a feasible circulation exists.  Ford-Fulkerson's theorem then says the demanded
    flow exists iff the min cut is saturated;
  * max flow s->t -- add an artificial t->s edge of infinite capacity, find a feasible
    circulation of the enlarged graph, read off the flow on the artificial edge (the initial
    value), then disable the artificial and the SS/ST edges and keep augmenting s->t in the
    residual network of x.  The final value is that initial value plus the new augmentation.

The invariants re-checked on every produced flow: l <= f(e) <= u for each edge, conservation at
every non-terminal node, value = net outflow of s = net inflow of t, and value >= 0.

Simplified versus production code: Dinic is written directly on arc pairs with explicit flow
bookkeeping instead of a packed array form, capacities are plain Python ints, and instances are
small (tens of nodes) so the answer fits comfortably inside the time budget.  Verified against a
brute-force enumeration of all integral flows inside the bounds (feasibility and optimal value),
against the classical min cut when all lower bounds are zero, and against the cut upper bound
value <= cap(S) - lower(T -> S) for random cuts.
"""
import itertools
import sys

INF = float("inf")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Dinic:
    """Maximum flow on a network of arc pairs; supports disabling an edge after the fact."""

    def __init__(self, n):
        self.n = n
        self.adjacency = [[] for _ in range(n)]
        self.target = []
        self.reverse_of = []
        self.capacity = []
        self.flow = []
        self.active = []
        self.augmentations = 0
        self.phases = 0

    def add_edge(self, u, v, capacity):
        arc = len(self.target)
        self.target.append(v)
        self.reverse_of.append(arc + 1)
        self.capacity.append(capacity)
        self.flow.append(0)
        self.active.append(True)
        self.target.append(u)
        self.reverse_of.append(arc)
        self.capacity.append(0)
        self.flow.append(0)
        self.active.append(True)
        self.adjacency[u].append(arc)
        self.adjacency[v].append(arc + 1)
        return arc

    def residual(self, arc):
        if not self.active[arc]:
            return 0
        return self.capacity[arc] - self.flow[arc]

    def disable_edge(self, arc):
        """Forget an edge and its flow: both residual directions become unusable."""
        self.active[arc] = False
        self.active[self.reverse_of[arc]] = False
        self.flow[arc] = 0
        self.flow[self.reverse_of[arc]] = 0

    def _push(self, arc, amount):
        self.flow[arc] += amount
        self.flow[self.reverse_of[arc]] -= amount

    def _dfs(self, node, sink, limit, level, iterator):
        if node == sink:
            return limit
        adjacency = self.adjacency[node]
        while iterator[node] < len(adjacency):
            arc = adjacency[iterator[node]]
            nxt = self.target[arc]
            available = self.residual(arc)
            if available > 0 and level[nxt] == level[node] + 1:
                pushed = self._dfs(nxt, sink, min(limit, available), level, iterator)
                if pushed:
                    self._push(arc, pushed)
                    self.augmentations += 1
                    return pushed
            iterator[node] += 1
        return 0

    def max_flow(self, source, sink, ceiling=1 << 62):
        """Augment the current flow; returns only the *additional* value."""
        added = 0
        while True:
            level = [-1] * self.n
            level[source] = 0
            queue = [source]
            while queue:
                node = queue.pop(0)
                for arc in self.adjacency[node]:
                    nxt = self.target[arc]
                    if level[nxt] < 0 and self.residual(arc) > 0:
                        level[nxt] = level[node] + 1
                        queue.append(nxt)
            if level[sink] < 0:
                return added
            self.phases += 1
            iterator = [0] * self.n
            while True:
                pushed = self._dfs(source, sink, ceiling, level, iterator)
                if not pushed:
                    break
                added += pushed


def flow_with_bounds(n, edges, source, sink):
    """Max s-t flow with lower/upper bounds; (None, None) when no feasible flow exists.

    `edges` is a list of (u, v, lower, upper); the returned flow list is indexed like `edges`.
    """
    for index, (u, v, lower, upper) in enumerate(edges):
        check(0 <= u < n and 0 <= v < n, f"edge {index} has an endpoint outside 0..{n - 1}")
        check(0 <= lower <= upper, f"edge {index} needs 0 <= lower <= upper, got {lower},{upper}")
    check(0 <= source < n and 0 <= sink < n, "the terminals must be graph nodes")

    super_source, super_sink = n, n + 1
    network = Dinic(n + 2)
    arcs = [network.add_edge(u, v, upper - lower) for u, v, lower, upper in edges]

    balance = [0] * n
    for u, v, lower, _ in edges:
        balance[v] += lower
        balance[u] -= lower

    big = 1 + sum(upper for _, _, _, upper in edges)
    artificial = None
    if source != sink:
        artificial = network.add_edge(sink, source, big)

    demand = 0
    supply_arcs = []
    for node in range(n):
        if balance[node] > 0:
            supply_arcs.append(network.add_edge(super_source, node, balance[node]))
            demand += balance[node]
        elif balance[node] < 0:
            supply_arcs.append(network.add_edge(node, super_sink, -balance[node]))

    delivered = 0
    while True:
        pushed = network.max_flow(super_source, super_sink)
        if not pushed:
            break
        delivered += pushed
    if delivered != demand:
        return None, None

    value = network.flow[artificial] if artificial is not None else 0
    if artificial is not None:
        network.disable_edge(artificial)
    for arc in supply_arcs:
        network.disable_edge(arc)
    if source != sink:
        value += network.max_flow(source, sink)

    flows = [edges[index][2] + network.flow[arcs[index]] for index in range(len(edges))]
    return value, flows


# --------------------------------------------------------------------------- oracles
def flow_invariants(n, edges, flows, source, sink):
    """Check conservation and bounds; returns the value implied by the flow."""
    balance = [0] * n
    for (u, v, lower, upper), f in zip(edges, flows):
        check(lower <= f <= upper, f"edge {u}->{v}: flow {f} outside [{lower}, {upper}]")
        balance[u] -= f
        balance[v] += f
    for node in range(n):
        if node == source or node == sink:
            continue
        check(balance[node] == 0, f"conservation violated at node {node}: {balance[node]}")
    value = balance[sink]
    check(balance[source] == -value, "the source must lose exactly what the sink gains")
    check(value >= 0, f"a feasible flow cannot have value {value}")
    return value


def brute_force_flow(n, edges, source, sink):
    """Enumerate every integral flow inside the bounds: None if infeasible, else the maximum."""
    best = None
    ranges = [range(lower, upper + 1) for _, _, lower, upper in edges]
    for choice in itertools.product(*ranges):
        balance = [0] * n
        for (u, v, _, _), f in zip(edges, choice):
            balance[u] -= f
            balance[v] += f
        feasible = True
        for node in range(n):
            if node == source or node == sink:
                continue
            if balance[node] != 0:
                feasible = False
                break
        if not feasible or balance[sink] != -balance[source]:
            continue
        value = balance[sink]
        if value < 0:
            continue
        if best is None or value > best:
            best = value
    return best


def cut_capacity(edges, side):
    """sum upper(S -> T) - sum lower(T -> S), the classic upper bound on the s-t value."""
    forward = sum(upper for u, v, _, upper in edges if u in side and v not in side)
    backward = sum(lower for u, v, lower, _ in edges if v in side and u not in side)
    return forward - backward


def min_cut(n, edges, source, sink):
    best = INF
    others = [node for node in range(n) if node != source and node != sink]
    for mask in range(1 << len(others)):
        side = {source} | {others[i] for i in range(len(others)) if mask >> i & 1}
        best = min(best, cut_capacity(edges, side))
    return best


# --------------------------------------------------------------------------- checks
def test_handcrafted_circulations():
    # a 3-cycle where every edge demands at least 1: feasible, value 0
    edges = [(0, 1, 1, 3), (1, 2, 1, 3), (2, 0, 1, 3)]
    value, flows = flow_with_bounds(3, edges, 0, 0)
    check(value == 0, "a circulation has value 0")
    check(flow_invariants(3, edges, flows, 0, 0) == 0, "the circulation is conservative")
    check(flows == [1, 1, 1], f"the cheapest circulation sends exactly the lower bounds, got {flows}")

    # same cycle but one edge cannot carry the demand: infeasible
    edges = [(0, 1, 2, 2), (1, 2, 2, 2), (2, 0, 0, 1)]
    value, flows = flow_with_bounds(3, edges, 0, 0)
    check(value is None and flows is None, "the cycle demand exceeds the third edge's bound")
    check(brute_force_flow(3, edges, 0, 0) is None, "the oracle agrees: infeasible")

    # a node cannot absorb its forced inflow
    edges = [(0, 1, 3, 5), (1, 2, 0, 1)]
    check(flow_with_bounds(3, edges, 0, 0)[0] is None, "node 1 is forced to lose 3 but can send 1")
    check(brute_force_flow(3, edges, 0, 0) is None, "oracle agrees")

    # forced flow that is perfectly balanced again
    edges = [(0, 1, 3, 5), (1, 0, 3, 4)]
    value, flows = flow_with_bounds(2, edges, 0, 0)
    check(value == 0 and flows == [3, 3], f"the balanced pair sends 3 each way, got {flows}")


def test_handcrafted_max_flow():
    # classic example with a mandatory unit on the 1->3 edge
    edges = [(0, 1, 0, 4), (0, 2, 0, 3), (1, 3, 1, 5), (2, 3, 0, 2)]
    value, flows = flow_with_bounds(4, edges, 0, 3)
    check(value == 6, f"4 + 2 units can be shipped, got {value}")
    check(flows == [4, 2, 4, 2], f"the tightest routing is expected, got {flows}")
    check(flow_invariants(4, edges, flows, 0, 3) == value, "conservation and bounds hold")
    check(brute_force_flow(4, edges, 0, 3) == value, "the oracle agrees on the optimum")
    check(flows[2] >= 1, "the lower bound on 1->3 must be respected")

    # a lower bound that eats part of the capacity of every path
    edges = [(0, 1, 2, 2), (0, 2, 0, 5), (1, 3, 2, 2), (2, 3, 0, 5)]
    value, flows = flow_with_bounds(4, edges, 0, 3)
    check(value == 7, f"5 + 2 = 7 units can be shipped, got {value}")
    check(flow_invariants(4, edges, flows, 0, 3) == value, "invariants")
    check(brute_force_flow(4, edges, 0, 3) == value, "oracle")

    # a mandatory reverse edge: 1 -> 0 forces flow back, but the other paths still carry 3
    edges = [(0, 1, 0, 5), (1, 0, 2, 3), (1, 2, 0, 4)]
    value, flows = flow_with_bounds(3, edges, 0, 2)
    check(value == 3, f"a = b + c with b >= 2 gives c <= 3, got {value}")
    check(flows[1] >= 2, "the reverse lower bound must be respected")
    check(flow_invariants(3, edges, flows, 0, 2) == value, "invariants with a back edge")
    check(brute_force_flow(3, edges, 0, 2) == value, "oracle")

    # a forced reverse edge leaving the source: the value would have to be negative
    edges = [(1, 0, 5, 8)]
    value, flows = flow_with_bounds(2, edges, 0, 1)
    check(value is None and flows is None,
          "a mandatory flow into the source makes an s-t flow impossible")
    check(brute_force_flow(2, edges, 0, 1) is None, "the oracle agrees")


def test_cut_upper_bound():
    edges = [(0, 1, 1, 4), (1, 3, 0, 5), (0, 2, 0, 3), (2, 3, 1, 2), (1, 2, 0, 2), (2, 1, 1, 2)]
    value, flows = flow_with_bounds(4, edges, 0, 3)
    check(value is not None, "the instance is feasible")
    check(value == 7, f"the forced 2->3 and 2->1 units leave room for 7, got {value}")
    check(flow_invariants(4, edges, flows, 0, 3) == value, "invariants")
    check(value == brute_force_flow(4, edges, 0, 3), "oracle")
    for mask in range(4):
        side = {0} | ({1} if mask & 1 else set()) | ({2} if mask & 2 else set())
        bound = cut_capacity(edges, side)
        check(value <= bound + 1e-9,
              f"value {value} must not exceed the cut bound {bound} for side {sorted(side)}")

    # the generalised max-flow / min-cut theorem for lower bounds:
    # value = min over cuts S of ( cap(S -> T) - lower(T -> S) )
    check(value == min_cut(4, edges, 0, 3),
          f"value {value} != generalised min cut {min_cut(4, edges, 0, 3)}")
    check(min_cut(4, edges, 0, 3) < cut_capacity(edges, {0, 2}),
          "the tight cut must be strictly better than another one")


def test_zero_lowers_match_classic_min_cut():
    mask = (1 << 64) - 1
    seed = 0x51F00D0000BEEF01
    tested = 0
    for trial in range(25):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        n = 3 + seed % 4
        edges = []
        for _ in range(2 + (seed >> 7) % 5):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            u = seed % n
            v = (seed >> 17) % n
            if u == v:
                continue
            edges.append((u, v, 0, 1 + (seed >> 31) % 3))
        if not edges:
            continue
        value, flows = flow_with_bounds(n, edges, 0, n - 1)
        check(value is not None, f"trial {trial}: zero lowers are always feasible")
        check(value == min_cut(n, edges, 0, n - 1),
              f"trial {trial}: max flow {value} != min cut {min_cut(n, edges, 0, n - 1)}")
        check(flow_invariants(n, edges, flows, 0, n - 1) == value, f"trial {trial}: invariants")
        check(value == brute_force_flow(n, edges, 0, n - 1), f"trial {trial}: oracle")
        tested += 1
    check(tested >= 15, f"only {tested} classic instances were exercised")


def test_random_instances_against_brute_force():
    mask = (1 << 64) - 1
    seed = 0x1BADB002DEADBEEF
    feasible, infeasible, suboptimal = 0, 0, 0
    for trial in range(60):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        n = 3 + seed % 4
        edges = []
        for _ in range(2 + (seed >> 11) % 5):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            u = seed % n
            v = (seed >> 19) % n
            upper = (seed >> 33) % 3
            lower = (seed >> 41) % (upper + 1)
            edges.append((u, v, lower, upper))
        if not edges:
            continue
        source, sink = 0, n - 1
        optimum = brute_force_flow(n, edges, source, sink)
        value, flows = flow_with_bounds(n, edges, source, sink)
        if optimum is None:
            check(value is None, f"trial {trial}: the oracle proved infeasibility")
            infeasible += 1
            continue
        check(value is not None, f"trial {trial}: a feasible instance was rejected")
        check(value == optimum, f"trial {trial}: value {value} != brute-force optimum {optimum}")
        check(flow_invariants(n, edges, flows, source, sink) == value,
              f"trial {trial}: the returned flow violates its invariants")
        check(all(isinstance(f, int) for f in flows), f"trial {trial}: flows must stay integral")
        bound = min_cut(n, edges, source, sink)
        check(value <= bound + 1e-9, f"trial {trial}: value above the cut bound {bound}")
        feasible += 1
        suboptimal += value < bound
    check(feasible >= 15 and infeasible >= 3,
          f"the sweep must cover both outcomes, got {feasible} feasible / {infeasible} infeasible")
    return feasible, infeasible, suboptimal


def test_boundaries():
    value, flows = flow_with_bounds(3, [], 0, 2)
    check(value == 0 and flows == [], "the empty instance has value 0")
    check(brute_force_flow(3, [], 0, 2) == 0, "the oracle agrees on the empty instance")

    value, flows = flow_with_bounds(2, [(0, 1, 4, 4)], 0, 1)
    check(value == 4 and flows == [4], "lower == upper forces the flow exactly")

    value, flows = flow_with_bounds(2, [(1, 0, 5, 8)], 0, 1)
    check(value is None, "a mandatory edge into the source admits no s-t flow")
    check(brute_force_flow(2, [(1, 0, 5, 8)], 0, 1) is None, "oracle agrees")

    value, flows = flow_with_bounds(2, [(0, 0, 2, 4), (0, 1, 0, 2)], 0, 1)
    check(value == 2 and flows[0] == 2, "a self loop is feasible and does not hurt the value")
    check(flow_invariants(2, [(0, 0, 2, 4), (0, 1, 0, 2)], flows, 0, 1) == 2, "self-loop invariants")

    value, flows = flow_with_bounds(3, [(0, 1, 0, 2), (0, 1, 0, 3)], 0, 1)
    check(value == 5, f"parallel edges add up to 5, got {value}")
    check(flows[0] + flows[1] == 5, "both parallel edges carry flow")

    value, flows = flow_with_bounds(4, [(0, 3, 1, 1)], 0, 3)
    check(value == 1, "a direct edge with a forced unit")
    value, _ = flow_with_bounds(4, [(0, 1, 0, 1)], 0, 3)
    check(value == 0, "no path means no flow")

    edges = [(0, 1, 2, 2), (1, 2, 0, 5), (2, 1, 0, 1)]
    value, flows = flow_with_bounds(3, edges, 0, 2)
    check(value == 2, f"f(1,2) = 2 + f(2,1) pins the value at 2, got {value}")
    check(flow_invariants(3, edges, flows, 0, 2) == value, "invariants of the detour instance")
    check(brute_force_flow(3, edges, 0, 2) == value, "oracle on the detour instance")

    try:
        flow_with_bounds(3, [(0, 1, 4, 2)], 0, 2)
        raise AssertionError("lower > upper must be rejected")
    except AssertionError as error:
        check("lower <= upper" in str(error), f"unexpected message: {error}")
    try:
        flow_with_bounds(3, [(0, 7, 0, 1)], 0, 2)
        raise AssertionError("an endpoint outside the graph must be rejected")
    except AssertionError as error:
        check("outside" in str(error), f"unexpected message: {error}")

    first = flow_with_bounds(4, [(0, 1, 1, 3), (1, 3, 0, 2), (0, 2, 0, 2), (2, 3, 0, 2)], 0, 3)
    second = flow_with_bounds(4, [(0, 1, 1, 3), (1, 3, 0, 2), (0, 2, 0, 2), (2, 3, 0, 2)], 0, 3)
    check(first == second, "the solver must be deterministic")


def test_larger_instance():
    """A 60-node instance that is feasible by construction: a chain, random extra edges, and a
    directed 3-cycle whose edges carry the only positive lower bounds (its unit circulation is a
    feasible flow, so the instance cannot be infeasible)."""
    n = 60
    mask = (1 << 64) - 1
    seed = 0xC0FFEE123456789
    edges = []
    for node in range(1, n):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        edges.append((node - 1, node, 0, 1 + (seed >> 19) % 6))
    for _ in range(200):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        u = seed % n
        v = (seed >> 21) % n
        if u != v:
            edges.append((u, v, 0, 1 + (seed >> 37) % 8))
    cycle = [5, 31, 44]
    for k in range(3):
        edges.append((cycle[k], cycle[(k + 1) % 3], 1, 3))

    value, flows = flow_with_bounds(n, edges, 0, n - 1)
    check(value is not None, "the instance built around a feasible circulation must be feasible")
    check(flow_invariants(n, edges, flows, 0, n - 1) == value, "invariants on the large instance")
    for index in range(len(edges) - 3, len(edges)):
        check(flows[index] >= 1, "the cycle's lower bounds must be honoured")
    bound = cut_capacity(edges, {0})
    check(value <= bound + 1e-9, f"value {value} above the single-node cut bound {bound}")
    total_upper = sum(upper for _, _, _, upper in edges)
    check(value <= total_upper, "the value cannot exceed the total capacity")
    return n, len(edges), value, bound


def main():
    test_handcrafted_circulations()
    test_handcrafted_max_flow()
    test_cut_upper_bound()
    test_zero_lowers_match_classic_min_cut()
    feasible, infeasible, suboptimal = test_random_instances_against_brute_force()
    test_boundaries()
    large = test_larger_instance()

    print(f"random_sweep: feasible={feasible} infeasible={infeasible} "
          f"cases_below_cut_bound={suboptimal}")
    print(f"large_instance: {large}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
