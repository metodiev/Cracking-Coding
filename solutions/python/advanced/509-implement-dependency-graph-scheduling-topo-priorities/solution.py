"""Task 509 - Implement dependency graph scheduling (topo + priorities). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Dependency-aware task scheduling over a DAG of tasks and prerequisites. Two orders are produced
with the same Kahn-style peeling loop but different ready-set disciplines: a binary heap keyed by
task id yields the lexicographically smallest topological order, and a heap keyed by
(-priority, id) yields, at every step, the highest-priority task among those whose prerequisites are
already done - the classic greedy priority scheduler (and the tie-break is again the smallest id,
so the answer is unique). The graph is stored as a dict of task -> set of prerequisites; unknown
names appearing only as prerequisites are treated as prerequisite-free tasks so a partial graph can
be scheduled without a separate sanitising pass, and duplicate edges collapse because the
prerequisite sets are sets. Cycles are found with a three-colour depth-first search that records the
path, so the reported cycle is a real closed walk of existing edges; a self-loop is a cycle of
length one.

Documented checks: for every graph built here (a chain, a diamond, a wide fan, an LCG DAG on
8 nodes with edges from lower to higher index, and a graph with an isolated node and a partial
node) the returned order is a valid topological order (every prerequisite precedes its dependant)
and, for the priority variant, the order is checked step by step against an independent
"scan the ready set for the maximum priority" implementation; the lexicographically smallest order
is compared against brute force over all n! permutations for n <= 6 - the minimum valid permutation
must equal it - and the number of topological orders returned by a bit-mask subset DP must equal the
brute-force count; the empty graph, a single task and disconnected components are covered; a cyclic
graph makes both schedulers report failure while the DFS must return a genuine cycle (a closed path
of existing edges) and detect length-one self-loops; and priorities must actually change the answer
- the priority order and the lexicographic order must differ on the priority example, while both
remain valid topological orders.
"""

import heapq
from itertools import permutations


def normalise(graph):
    """Tasks -> frozenset of prerequisites, with prerequisite-only names materialised."""
    tasks = {}
    for task, deps in graph.items():
        tasks[task] = set(deps or ())
        for dep in deps or ():
            tasks.setdefault(dep, set())
    return tasks


def topo_order(graph, use_priority=False, priorities=None):
    """Kahn peeling. Returns a list of tasks, or None when the graph is cyclic."""
    tasks = normalise(graph)
    priorities = priorities or {}
    indegree = {task: len(tasks[task]) for task in tasks}
    dependants = {}
    for task, deps in tasks.items():
        for dep in deps:
            dependants.setdefault(dep, []).append(task)
    heap = []
    for task in sorted(tasks):
        if indegree[task] == 0:
            key = (-priorities.get(task, 0), task) if use_priority else (task,)
            heapq.heappush(heap, key)
    order = []
    while heap:
        key = heapq.heappop(heap)
        task = key[-1]
        order.append(task)
        for next_task in sorted(dependants.get(task, ())):
            indegree[next_task] -= 1
            if indegree[next_task] == 0:
                next_key = (-priorities.get(next_task, 0), next_task) if use_priority \
                    else (next_task,)
                heapq.heappush(heap, next_key)
    if len(order) != len(tasks):
        return None
    return order


def is_valid_order(tasks, order):
    position = {task: index for index, task in enumerate(order)}
    if set(order) != set(tasks):
        return False
    return all(position[dep] < position[task] for task, deps in tasks.items() for dep in deps)


def greedy_priority_order(graph, priorities):
    """Independent implementation: repeatedly scan the ready set for the best priority."""
    tasks = normalise(graph)
    done = set()
    order = []
    while len(order) < len(tasks):
        ready = [task for task in tasks if task not in done and tasks[task] <= done]
        if not ready:
            return None
        best = min(ready, key=lambda task: (-priorities.get(task, 0), task))
        order.append(best)
        done.add(best)
    return order


def find_cycle(graph):
    """Three-colour DFS. Returns a list [v0, v1, ..., v0] of a real cycle, or None."""
    tasks = normalise(graph)
    colour = {}
    parent = {}

    def visit(node, stack):
        colour[node] = 1
        stack.append(node)
        for dep in sorted(tasks[node]):
            if colour.get(dep, 0) == 1:
                start = stack.index(dep)
                return stack[start:] + [dep]
            if colour.get(dep, 0) == 0:
                parent[dep] = node
                found = visit(dep, stack)
                if found:
                    return found
        colour[node] = 2
        stack.pop()
        return None

    for task in sorted(tasks):
        if colour.get(task, 0) == 0:
            found = visit(task, [])
            if found:
                return found
    return None


def count_orders(graph):
    """Bit-mask subset DP: how many topological orders the DAG has."""
    tasks = normalise(graph)
    names = sorted(tasks)
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    need = [0] * n
    for name, deps in tasks.items():
        mask = 0
        for dep in deps:
            mask |= 1 << index[dep]
        need[index[name]] = mask
    full = (1 << n) - 1
    counts = [0] * (full + 1)
    counts[0] = 1
    for mask in range(full + 1):
        if not counts[mask]:
            continue
        for i in range(n):
            bit = 1 << i
            if mask & bit:
                continue
            if need[i] & ~mask:
                continue
            counts[mask | bit] += counts[mask]
    return counts[full]


def brute_force_best(graph):
    """Minimum valid permutation (n <= 7 only) and the number of valid permutations."""
    tasks = normalise(graph)
    names = sorted(tasks)
    best = None
    total = 0
    for candidate in permutations(names):
        if is_valid_order(tasks, list(candidate)):
            total += 1
            if best is None or candidate < best:
                best = candidate
    return (list(best) if best else None), total


def lcg(seed):
    return (seed * 75 + 74) % 65537


def random_dag(n, seed, edge_chance=30):
    """Edges only from a lower index to a higher one, so the graph is acyclic by construction."""
    graph = {}
    for i in range(n):
        deps = []
        for j in range(i):
            seed = lcg(seed)
            if seed % 100 < edge_chance:
                deps.append("t%d" % j)
        graph["t%d" % i] = deps
    return graph


def main():
    chain = {"c": ["b"], "b": ["a"], "a": []}
    diamond = {"d": ["b", "c"], "b": ["a"], "c": ["a"], "a": []}
    fan = {"root": [], "x1": ["root"], "x2": ["root"], "x3": ["root"],
           "sink": ["x1", "x2", "x3"]}
    graph_with_orphans = {"a": [], "b": ["a"], "c": [], "e": ["orphan"], "orphan": []}

    for graph in (chain, diamond, fan, graph_with_orphans):
        tasks = normalise(graph)
        prios = _priorities(graph)
        for use_priority in (False, True):
            order = topo_order(graph, use_priority=use_priority, priorities=prios)
            assert order is not None and is_valid_order(tasks, order), (graph, order)
            if use_priority:
                assert order == greedy_priority_order(graph, prios)

    # --- Lexicographically smallest order against brute force. -------------------------------
    for graph in (chain, diamond, fan, graph_with_orphans,
                  {"d": ["a", "b"], "c": ["a"], "b": [], "a": [], "e": ["c", "d"]}):
        best, total = brute_force_best(graph)
        assert topo_order(graph) == best, (graph, topo_order(graph), best)
        assert count_orders(graph) == total, (graph, count_orders(graph), total)
    # Known counts: the diamond has exactly 2 orders, the chain exactly 1.
    assert count_orders(diamond) == 2 and count_orders(chain) == 1
    assert brute_force_best(diamond)[0] == ["a", "b", "c", "d"]
    assert brute_force_best(fan)[1] == 3 * 2 * 1      # x1..x3 in any order, then sink

    # --- Priorities change the answer but never break validity. ------------------------------
    priorities = {"a": 1, "b": 9, "c": 5, "d": 9, "e": 2}
    priority_graph = {"d": ["b", "c"], "c": ["a"], "b": ["a"], "a": [], "e": []}
    ordered = topo_order(priority_graph, use_priority=True, priorities=priorities)
    assert ordered == greedy_priority_order(priority_graph, priorities)
    assert ordered == ["e", "a", "b", "c", "d"], ordered
    assert is_valid_order(normalise(priority_graph), ordered)
    lexicographic = topo_order(priority_graph)
    assert lexicographic == ["a", "b", "c", "d", "e"]
    assert ordered != lexicographic, "priorities must be able to reorder the schedule"
    # Step-by-step greedy check: at each step the chosen task is the best ready task.
    done = set()
    for chosen in ordered:
        ready = [task for task in normalise(priority_graph)
                 if task not in done and normalise(priority_graph)[task] <= done]
        assert chosen == min(ready, key=lambda task: (-priorities.get(task, 0), task))
        done.add(chosen)

    # --- LCG DAGs: validity, agreement with the independent greedy, subset-DP counts. --------
    for n in (4, 5, 6):
        for seed in (11, 29, 53):
            graph = random_dag(n, seed)
            tasks = normalise(graph)
            order = topo_order(graph)
            assert order is not None and is_valid_order(tasks, order)
            best, total = brute_force_best(graph)
            assert order == best and count_orders(graph) == total, (n, seed)
            prios = {task: (lcg(seed + int(task[1:]) * 7) % 5) for task in tasks}
            prio_order = topo_order(graph, use_priority=True, priorities=prios)
            assert is_valid_order(tasks, prio_order)
            assert prio_order == greedy_priority_order(graph, prios)
    graph8 = random_dag(8, seed=101)
    assert is_valid_order(normalise(graph8), topo_order(graph8))
    assert count_orders(graph8) == brute_force_best(graph8)[1]
    assert count_orders(graph8) >= 1

    # --- Degenerate graphs. ------------------------------------------------------------------
    assert topo_order({}) == [] and count_orders({}) == 1
    assert topo_order({"only": []}) == ["only"]
    assert count_orders({"only": []}) == 1
    partial = {"leaf": []}
    assert topo_order(partial) == ["leaf"]
    disconnected = {"a": [], "b": [], "c": ["a"], "d": ["b"]}
    order = topo_order(disconnected)
    assert order == ["a", "b", "c", "d"] and is_valid_order(normalise(disconnected), order)
    duplicated = {"b": ["a", "a"], "a": []}
    assert topo_order(duplicated) == ["a", "b"]

    # --- Cycles: the schedulers refuse, the DFS reports a real closed walk. ------------------
    cyclic = {"a": ["c"], "b": ["a"], "c": ["b"], "d": []}
    assert topo_order(cyclic) is None
    assert topo_order(cyclic, use_priority=True, priorities={"a": 1}) is None
    assert greedy_priority_order(cyclic, {}) is None
    cycle = find_cycle(cyclic)
    assert cycle is not None and cycle[0] == cycle[-1] and len(cycle) >= 2
    for left, right in zip(cycle, cycle[1:]):
        assert left in normalise(cyclic) and right in normalise(cyclic)[left], \
            "every reported edge must exist in the direction of the dependency"
    assert sorted(set(cycle)) == ["a", "b", "c"]
    assert find_cycle({"s": ["s"]}) == ["s", "s"], "a self-loop is a cycle of length one"
    assert topo_order({"s": ["s"]}) is None and count_orders({"s": ["s"]}) == 0
    assert find_cycle(diamond) is None and find_cycle(graph8) is None


def _priorities(graph):
    """A deterministic priority per task, used by the validity sweep."""
    return {task: (len(task) + sum(ord(ch) for ch in task)) % 7 for task in normalise(graph)}


if __name__ == "__main__":
    main()
    print("PASS")
