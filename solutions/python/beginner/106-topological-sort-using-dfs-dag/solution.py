"""Task 106 - Topological sort using DFS (DAG). (Tier: Beginner, CTCI Ch 4)

A vertex is emitted after all of its descendants, so reversing postorder
yields a topological order.  Revisiting a gray vertex means a cycle: there is
no valid order, signalled explicitly with None.
"""

WHITE, GRAY, BLACK = 0, 1, 2


def topo_sort(graph):
    color = {}
    order = []

    def visit(v):
        color[v] = GRAY
        for w in graph.get(v, []):
            c = color.get(w, WHITE)
            if c == GRAY:
                return False
            if c == WHITE and not visit(w):
                return False
        color[v] = BLACK
        order.append(v)
        return True

    for v in graph:
        if color.get(v, WHITE) == WHITE and not visit(v):
            return None
    order.reverse()
    return order


def is_valid_topo(graph, order):
    if order is None or sorted(order) != sorted(graph):
        return False
    index = {v: i for i, v in enumerate(order)}
    return all(index[u] < index[w] for u, ws in graph.items() for w in ws)


def main():
    # classic course prerequisites [course, prereq] -> edge prereq -> course
    prereqs = [[1, 0], [2, 0], [3, 1], [3, 2]]
    graph = {}
    for course, prereq in prereqs:
        graph.setdefault(prereq, []).append(course)
    for course in range(4):
        graph.setdefault(course, [])
    order = topo_sort(graph)
    assert order == [0, 2, 1, 3]
    assert is_valid_topo(graph, order)

    assert topo_sort({0: [1], 1: [2], 2: [0]}) is None    # cycle
    assert topo_sort({0: [0]}) is None                    # self-loop
    assert topo_sort({}) == []
    assert topo_sort({9: []}) == [9]

    forest = {0: [1], 1: [], 2: [3], 3: []}
    order2 = topo_sort(forest)
    assert is_valid_topo(forest, order2) and len(order2) == 4

    chained = {0: [], 1: [0], 2: [1], 3: [2]}
    order3 = topo_sort(chained)
    assert order3 == [3, 2, 1, 0] and is_valid_topo(chained, order3)


if __name__ == "__main__":
    main()
    print("PASS")
