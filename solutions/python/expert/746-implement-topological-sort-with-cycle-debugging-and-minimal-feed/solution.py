"""Task 746 - Implement topological sort with cycle debugging and minimal feedback arc set heuristics.  (Tier: Expert, Ch 4 — Trees and Graphs)

Three things are asked of a directed graph here.  First a topological order: Kahn's algorithm
peels off zero in-degree nodes and the DFS variant emits nodes in post-order reversal, and the
leftovers of Kahn are exactly the nodes that take part in a cycle.  Second *debugging*: instead of
just reporting "cyclic", the code extracts a concrete cycle (a DFS with white/grey/black colours
returns the grey stack segment as the cycle) and, for bigger residues, condenses the graph into its
strongly connected components with Tarjan's algorithm and reports one real cycle per cyclic
component, which is what a developer needs to see.

Third an ordering that minimises the number of edges pointing backwards (the feedback arc set),
which is NP-hard: the tests compare three orderings - the naive input order, the Eades-Lin-Smyth
greedy (repeatedly peel sinks to the right and sources to the left, then break the remaining cycle
by the best in/out degree difference) and, for tiny graphs, the exact optimum found by trying every
permutation.  Verified: every reported order really is topological for acyclic graphs, every
reported cycle is a real cycle (each consecutive pair is an edge and the last edge closes it), the
Tarjan components equal a brute-force mutual-reachability computation on random digraphs, Kahn's
leftovers coincide exactly with the non-singleton components, the condensation is acyclic, the GR
order never beats the exact optimum and never loses to the naive order, and the cycle-free graphs
stay cycle-free under every routine.
"""
import heapq
from itertools import permutations

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


def build_graph(nodes, edges):
    """Adjacency sets with duplicates collapsed; self-loops are kept (they are cycles)."""
    graph = {node: set() for node in nodes}
    for source, target in edges:
        graph[source].add(target)
    return graph


def reverse_graph(graph):
    reversed_graph = {node: set() for node in graph}
    for node, targets in graph.items():
        for target in targets:
            reversed_graph[target].add(node)
    return reversed_graph


def indegrees(graph):
    degrees = {node: 0 for node in graph}
    for node, targets in graph.items():
        for target in targets:
            degrees[target] += 1
    return degrees


# ---------------------------------------------------------------------------- topological sort
def kahn(graph):
    """Peel zero in-degree nodes.  Returns (order, leftover) where leftover holds the nodes that
    are reachable again inside a cycle."""
    degrees = indegrees(graph)
    ready = [node for node in sorted(graph) if degrees[node] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for target in sorted(graph[node]):
            degrees[target] -= 1
            if degrees[target] == 0:
                heapq.heappush(ready, target)
    leftover = [node for node in sorted(graph) if node not in set(order)]
    return order, leftover


def dfs_order(graph):
    """DFS post-order, reversed; the second element is a real cycle when the graph is cyclic."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {node: WHITE for node in graph}
    order = []
    stack = []

    def visit(node):
        colour[node] = GREY
        stack.append(node)
        for target in sorted(graph[node]):
            if colour[target] == GREY:
                return stack[stack.index(target):] + [target]
            if colour[target] == WHITE:
                cycle = visit(target)
                if cycle is not None:
                    return cycle
        stack.pop()
        colour[node] = BLACK
        order.append(node)
        return None

    for node in sorted(graph):
        if colour[node] == WHITE:
            cycle = visit(node)
            if cycle is not None:
                return None, cycle
    return order[::-1], None


def is_topological(graph, order):
    """Every edge must point forwards in the order."""
    position = {node: index for index, node in enumerate(order)}
    if sorted(position) != sorted(graph):
        return False
    return all(position[source] < position[target]
               for source, targets in graph.items() for target in targets)


def is_cycle(graph, cycle):
    """A real cycle: consecutive pairs are edges, the last edge closes it, no repeats inside."""
    if len(cycle) < 2 or cycle[0] != cycle[-1]:
        return False
    body = cycle[:-1]
    if len(set(body)) != len(body):
        return False
    return all(target in graph[source] for source, target in zip(cycle, cycle[1:]))


def find_cycle(graph):
    return dfs_order(graph)[1]


# ---------------------------------------------------------------------------- components
def tarjan_scc(graph):
    """Tarjan's strongly connected components (iterative, deterministic order)."""
    index_of, low, on_stack, stack, result = {}, {}, {}, [], []
    counter = [0]

    def strong_connect(root):
        work = [(root, iter(sorted(graph[root])))]
        index_of[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            node, targets = work[-1]
            advanced = False
            for target in targets:
                if target not in index_of:
                    index_of[target] = low[target] = counter[0]
                    counter[0] += 1
                    stack.append(target)
                    on_stack[target] = True
                    work.append((target, iter(sorted(graph[target]))))
                    advanced = True
                    break
                if on_stack.get(target):
                    low[node] = min(low[node], index_of[target])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                result.append(sorted(component))

    for node in sorted(graph):
        if node not in index_of:
            strong_connect(node)
    return sorted(result, key=lambda component: component[0])


def reaches(graph, source):
    seen, stack = {source}, [source]
    while stack:
        node = stack.pop()
        for target in graph[node]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def scc_brute(graph):
    """Independent oracle: two nodes are in the same component when each reaches the other."""
    nodes = sorted(graph)
    components, assigned = [], set()
    for node in nodes:
        if node in assigned:
            continue
        forward, backward = reaches(graph, node), reaches(reverse_graph(graph), node)
        component = sorted(other for other in forward if other in backward)
        assigned.update(component)
        components.append(component)
    return sorted(components, key=lambda component: component[0])


def condensation(graph, components):
    """Contract the components; the result must be acyclic."""
    owner = {node: index for index, component in enumerate(components) for node in component}
    edges = set()
    for source, targets in graph.items():
        for target in targets:
            if owner[source] != owner[target]:
                edges.add((owner[source], owner[target]))
    return build_graph(range(len(components)), edges)


# ---------------------------------------------------------------------------- feedback arc sets
def backward_arcs(graph, order):
    position = {node: index for index, node in enumerate(order)}
    return sum(1 for source, targets in graph.items() for target in targets
               if position[source] > position[target])


def gr_order(graph):
    """Eades-Lin-Smyth greedy: peel sinks to the right and sources to the left, then break the
    remaining cycle by the largest out-degree minus in-degree difference."""
    remaining = {node: set(targets) for node, targets in graph.items()}
    left, right = [], []
    while remaining:
        sinks = sorted(node for node, targets in remaining.items()
                       if not any(target in remaining for target in targets))
        while sinks:
            node = sinks.pop(0)
            right.append(node)
            del remaining[node]
            sinks = sorted(node for node, targets in remaining.items()
                           if not any(target in remaining for target in targets))
        if not remaining:
            break
        sources = sorted(node for node in remaining
                         if not any(node in targets for targets in remaining.values()))
        while sources:
            node = sources.pop(0)
            left.append(node)
            del remaining[node]
            sources = sorted(node for node in remaining
                             if not any(node in targets for targets in remaining.values()))
        if not remaining:
            break
        best = max(sorted(remaining),
                   key=lambda node: (len(remaining[node])
                                     - sum(1 for targets in remaining.values() if node in targets),
                                     [-ord(character) for character in str(node)]))
        left.append(best)
        del remaining[best]
    return left + right[::-1]


def min_fas_brute(graph):
    """Exact minimum over all orderings: only for toy graphs (n <= 7)."""
    nodes = sorted(graph)
    best = None
    for order in permutations(nodes):
        cost = backward_arcs(graph, order)
        if best is None or cost < best:
            best = cost
        if best == 0:
            break
    return best


def cyclic_residue(graph, components):
    """Nodes that cannot be ordered: the cyclic components plus everything reachable from them.
    Computed independently on the condensation, not by running Kahn again."""
    cyclic = [index for index, component in enumerate(components)
              if len(component) > 1 or (component and component[0] in graph[component[0]])]
    condensed = condensation(graph, components)
    residue = set()
    for index in cyclic:
        residue |= reaches(condensed, index)
    return sorted(node for index in residue for node in components[index])


def random_digraph(nodes, edges, seed):
    state, chosen = seed, []
    for _ in range(edges):
        state, r1 = splitmix64(state)
        state, r2 = splitmix64(state)
        source = nodes[r1 % len(nodes)]
        target = nodes[r2 % len(nodes)]
        chosen.append((source, target))
    return build_graph(nodes, chosen), state


def main():
    # 1. acyclic graph: both orders are topological and no cycle is reported
    dag = build_graph(range(6), [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (1, 5)])
    order, leftover = kahn(dag)
    check(leftover == [] and is_topological(dag, order), "Kahn must order a DAG completely")
    dfs_sequence, cycle = dfs_order(dag)
    check(cycle is None and is_topological(dag, dfs_sequence),
          "the DFS order must be topological and report no cycle")
    check(find_cycle(dag) is None, "an acyclic graph must not report a cycle")
    check(tarjan_scc(dag) == [[node] for node in range(6)],
          "an acyclic graph has only singleton components")
    check(backward_arcs(dag, order) == 0, "a topological order of a DAG has no backward arcs")

    # 2. a cycle is reported concretely, and the leftovers match the cyclic components
    cyclic = build_graph(range(6), [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)])
    order, leftover = kahn(cyclic)
    reported = find_cycle(cyclic)
    check(is_cycle(cyclic, reported), f"the reported cycle {reported} must be a real cycle")
    components = tarjan_scc(cyclic)
    check(components == scc_brute(cyclic), "Tarjan must match the mutual-reachability oracle")
    cyclic_nodes = sorted(node for component in components if len(component) > 1
                          for node in component)
    residue = cyclic_residue(cyclic, components)
    check(cyclic_nodes == residue[:len(cyclic_nodes)] or set(cyclic_nodes) <= set(residue),
          "the cyclic components must be part of the residue")
    check(sorted(leftover) == residue,
          f"Kahn's leftovers {sorted(leftover)} must be the residue {residue}")
    check(set(reported) <= set(cyclic_nodes), "the reported cycle must live inside one component")
    component_of = {node: index for index, component in enumerate(components) for node in component}
    check(len({component_of[node] for node in reported[:-1]}) == 1,
          "a reported cycle must be contained in a single component")

    # 3. the condensation is acyclic and orders the components
    condensed = condensation(cyclic, components)
    condensed_order, condensed_left = kahn(condensed)
    check(condensed_left == [] and is_topological(condensed, condensed_order),
          "the condensation of any digraph must be acyclic")
    check(find_cycle(condensed) is None, "no cycle may survive condensation")

    # 4. random digraphs: Tarjan against the brute-force oracle, Kahn's leftovers again
    state = 0x5EED
    for _ in range(25):
        graph, state = random_digraph(list(range(7)), 12, state)
        components = tarjan_scc(graph)
        check(components == scc_brute(graph), "Tarjan must agree with the oracle on random graphs")
        _, leftover = kahn(graph)
        check(sorted(leftover) == cyclic_residue(graph, components),
              "Kahn's leftovers must match the residue computed from the condensation")
        cycle = find_cycle(graph)
        check(cycle is None or is_cycle(graph, cycle), "any reported cycle must be real")
        check((cycle is None) == (not cyclic_nodes),
              "a cycle must be reported exactly when one exists")

    # 5. feedback arc sets: the greedy ordering beats the naive order and never beats the optimum
    for edges in ([(0, 1), (1, 2), (2, 0)],
                  [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)],
                  [(0, 1), (2, 1), (2, 3), (4, 3), (4, 0), (1, 4)]):
        graph = build_graph(range(5), edges)
        naive = sorted(graph)
        greedy = gr_order(graph)
        check(sorted(greedy) == naive, "an ordering must be a permutation of the nodes")
        naive_cost, greedy_cost = backward_arcs(graph, naive), backward_arcs(graph, greedy)
        optimum = min_fas_brute(build_graph(range(4), [(s, t) for s, t in edges if s < 4 and t < 4]))
        check(greedy_cost >= optimum, "the heuristic cannot beat the exact optimum")
        check(greedy_cost <= naive_cost, "the greedy ordering must not be worse than the naive one")
    triangle = build_graph(range(3), [(0, 1), (1, 2), (2, 0)])
    check(min_fas_brute(triangle) == 1, "a directed triangle needs exactly one feedback arc")
    check(backward_arcs(triangle, gr_order(triangle)) == 1,
          "the greedy ordering must find the single feedback arc of a triangle")

    # 6. removing the backward arcs leaves a DAG
    graph = build_graph(range(6), [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (5, 4)])
    ordering = gr_order(graph)
    position = {node: index for index, node in enumerate(ordering)}
    forward = {node: {target for target in graph[node] if position[node] < position[target]}
               for node in graph}
    check(find_cycle(forward) is None, "dropping the backward arcs must leave a DAG")
    check(is_topological(forward, sorted(forward, key=lambda node: position[node])),
          "the greedy order must be topological for the forward arcs")

    # 7. self-loops are cycles of length one and keep the node out of every topological order
    self_loops = build_graph(range(3), [(0, 0), (0, 1), (1, 2)])
    _, leftover = kahn(self_loops)
    check(leftover == [0, 1, 2] and kahn(self_loops)[0] == [],
          "a self-loop blocks its node and everything downstream of it")
    loop = find_cycle(self_loops)
    check(loop == [0, 0] and is_cycle(self_loops, loop), "the self-loop must be reported as [0, 0]")

    # 8. boundary cases: empty graph, single node, duplicate edges
    empty = build_graph([], [])
    check(kahn(empty) == ([], []) and tarjan_scc(empty) == [] and find_cycle(empty) is None,
          "an empty graph orders to nothing and has no cycles")
    single = build_graph([7], [(7, 7)])
    check(kahn(single) == ([], [7]) and is_cycle(single, [7, 7]),
          "a lone self-looping node stays out of the order")

    duplicated = build_graph(range(3), [(0, 1), (0, 1), (1, 2), (1, 2)])
    check(sorted(duplicated[0]) == [1] and kahn(duplicated)[0] == [0, 1, 2],
          "duplicate edges must collapse into one")

    # 9. determinism
    check(kahn(cyclic)[0] == kahn(cyclic)[0] and gr_order(cyclic) == gr_order(cyclic),
          "the orderings and heuristics must be deterministic")
    check(tarjan_scc(cyclic) == tarjan_scc(build_graph(range(6),
          [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)])),
          "components must be reproduced for identical input")

    print(f"746 dag_order={order} cyclic_components={len(components)} "
          f"kahn_leftover={sorted(leftover)}")
    print(f"746 cycle={reported} greedy_backward={backward_arcs(cyclic, gr_order(cyclic))} "
          f"naive_backward={backward_arcs(cyclic, sorted(cyclic))}")
    print("PASS")


if __name__ == "__main__":
    main()
