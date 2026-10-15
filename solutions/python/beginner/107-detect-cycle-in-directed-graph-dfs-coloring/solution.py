"""Task 107 - Detect cycle in directed graph (DFS coloring). (Tier: Beginner, CTCI Ch 4)

White = unseen, gray = on the current recursion stack, black = fully done.
A directed cycle exists exactly when some edge points back to a gray vertex.
"""

WHITE, GRAY, BLACK = 0, 1, 2


def has_cycle(graph):
    color = {}

    def visit(v):
        color[v] = GRAY
        for w in graph.get(v, []):
            c = color.get(w, WHITE)
            if c == GRAY:
                return True
            if c == WHITE and visit(w):
                return True
        color[v] = BLACK
        return False

    for v in graph:
        if color.get(v, WHITE) == WHITE and visit(v):
            return True
    return False


def main():
    dag = {0: [1, 2], 1: [3], 2: [3], 3: []}
    assert has_cycle(dag) is False                       # DAG
    assert has_cycle({}) is False                        # empty graph
    assert has_cycle({7: []}) is False                   # single vertex
    assert has_cycle({0: [1], 1: [2], 2: [0]}) is True   # simple cycle
    assert has_cycle({0: [0]}) is True                   # self-loop
    assert has_cycle({0: [1], 1: [], 2: [3], 3: [2]}) is True    # one cyclic component
    assert has_cycle({0: [1], 1: [], 2: [3], 3: []}) is False    # disconnected, acyclic
    assert has_cycle({0: [1], 1: [2], 2: [1]}) is True   # back edge into gray
    # a diamond reaches 3 twice, but never while 3 is gray
    assert has_cycle({0: [1, 2], 1: [3], 2: [3], 3: []}) is False


if __name__ == "__main__":
    main()
    print("PASS")
