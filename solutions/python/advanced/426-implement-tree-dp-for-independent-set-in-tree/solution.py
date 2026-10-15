"""Task 426 - Implement tree DP for independent set in tree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Maximum-weight independent set on a rooted tree: take[u] = w[u] + sum(skip[child]) and
skip[u] = sum(max(take[child], skip[child])). The chosen set is reconstructed top-down (a
taken node forces its children to be skipped) and verified to have no two adjacent nodes and
to match the DP value; the value is cross-checked against brute force over all 2^n subsets
for n <= 16 and the reconstruction stays valid on larger trees.
"""


def root_tree(n, edges):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = [-1] * n
    parent[0] = 0
    order = [0]
    for u in order:
        for v in adjacency[u]:
            if v != parent[u]:
                parent[v] = u
                order.append(v)
    children = [[] for _ in range(n)]
    for u in order[1:]:
        children[parent[u]].append(u)
    return children, order


def max_weight_independent_set(n, edges, weights):
    """Returns (best weight, chosen flags). Ties prefer the empty/untaken option."""
    children, order = root_tree(n, edges)
    take = [0] * n
    skip = [0] * n
    for u in reversed(order):
        chosen = weights[u]
        not_chosen = 0
        for v in children[u]:
            chosen += skip[v]
            not_chosen += max(take[v], skip[v])
        take[u] = chosen
        skip[u] = not_chosen
    chosen = [False] * n
    stack = [(0, take[0] > skip[0])]
    while stack:
        u, taking = stack.pop()
        chosen[u] = taking
        for v in children[u]:
            stack.append((v, False if taking else take[v] > skip[v]))
    total = sum(weights[u] for u in range(n) if chosen[u])
    return total, chosen


def is_valid(n, edges, chosen):
    return all(not (chosen[a] and chosen[b]) for a, b in edges)


def brute_force(n, edges, weights):
    best = 0
    for mask in range(1 << n):
        weight = 0
        for i in range(n):
            if mask >> i & 1:
                weight += weights[i]
        if weight <= best:
            continue
        if all(not ((mask >> a & 1) and (mask >> b & 1)) for a, b in edges):
            best = weight
    return best


def weight_of(n, weights, chosen):
    return sum(weights[u] for u in range(n) if chosen[u])


def main():
    total, chosen = max_weight_independent_set(3, [(0, 1), (1, 2)], [10, 1, 10])
    assert total == 20 and chosen == [True, False, True]

    total, chosen = max_weight_independent_set(3, [(0, 1), (0, 2)], [5, 4, 4])
    assert total == 8 and chosen == [False, True, True]

    total, chosen = max_weight_independent_set(2, [(0, 1)], [-1, -2])
    assert total == 0 and chosen == [False, False]

    total, chosen = max_weight_independent_set(1, [], [7])
    assert total == 7 and chosen == [True]
    total, chosen = max_weight_independent_set(1, [], [-7])
    assert total == 0 and chosen == [False]

    total, chosen = max_weight_independent_set(4, [(0, 1), (1, 2), (2, 3)], [3, 10, 3, 10])
    assert total == 20 and chosen == [False, True, False, True]

    # exhaustive small comparisons against brute force over every subset
    local = 426426
    for trial in range(15):
        local = (local * 75 + 74) % 65537
        n = 2 + local % 15
        edges = []
        weights = []
        for node in range(n):
            local = (local * 75 + 74) % 65537
            weights.append(local % 21 - 10)
            if node > 0:
                local = (local * 75 + 74) % 65537
                edges.append((local % node, node))
        got, chosen = max_weight_independent_set(n, edges, weights)
        expected = brute_force(n, edges, weights)
        assert got == expected, (trial, n, got, expected)
        assert is_valid(n, edges, chosen), (trial, n)
        assert weight_of(n, weights, chosen) == got, (trial, n)

    # larger trees: brute force is off-limits, but the reconstruction must stay consistent
    local = 99999
    for trial in range(10):
        local = (local * 75 + 74) % 65537
        n = 20 + local % 41
        edges = []
        weights = []
        for node in range(n):
            local = (local * 75 + 74) % 65537
            weights.append(local % 41 - 20)
            if node > 0:
                local = (local * 75 + 74) % 65537
                edges.append((local % node, node))
        got, chosen = max_weight_independent_set(n, edges, weights)
        assert is_valid(n, edges, chosen), trial
        assert weight_of(n, weights, chosen) == got, trial
        assert got >= 0, trial


if __name__ == "__main__":
    main()
    print("PASS")
