"""Task 458 - Implement dynamic programming on trees with knapsack per node. (Tier: Advanced, CTCI Ch 4)

Classic tree knapsack: choose at most K vertices of a rooted tree such that whenever a vertex is
chosen its parent is chosen too (a "closure"), maximizing the sum of weights. Per node DP:
dp[u][j] = best weight of a valid choice of exactly j vertices inside subtree(u) that contains
u. Children merge one at a time: a child either contributes 0 vertices (unused) or a valid j >= 1
choice, so dp[u] is merged with dp[child] by (i + j) convolution, j >= 1; the answer is
max(0, max_j dp[root][j]) because the empty choice is allowed (and needed when every weight is
negative).

Hand values re-derived: root(5) with children 3, 4 and 3's child -10 gives 9 at K = 2
({root, 4}), 12 at K = 3 ({root, 3, 4}) and 5 at K = 1; the second tree (10 with children 8, 5,
where 8 has children 7 and 6) gives 25 at K = 3 ({10, 8, 7}); a chain 1, -5, 10 gives 6 at K = 3.
Cross-checked exhaustively against all parent-closed subsets of LCG trees with n <= 9 for every
K in 0..n.
"""

INF_NEG = -10 ** 9


def children_of(n, parent):
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)
    return children


def tree_knapsack(n, parent, weight, k):
    """Best sum of <= k chosen vertices with the parent-before-child closure rule."""
    if k <= 0 or n == 0:
        return 0
    children = children_of(n, parent)

    def dfs(u):
        subtree_size = 1
        dp = [INF_NEG] * (k + 1)
        dp[1] = weight[u]
        for child in children[u]:
            cdp, child_size = dfs(child)
            subtree_size += child_size
            new_dp = dp[:]
            for i in range(1, min(k, subtree_size - child_size) + 1):
                if dp[i] == INF_NEG:
                    continue
                for j in range(1, min(child_size, k - i) + 1):
                    if cdp[j] == INF_NEG:
                        continue
                    cand = dp[i] + cdp[j]
                    if cand > new_dp[i + j]:
                        new_dp[i + j] = cand
            dp = new_dp
        return dp, subtree_size

    dp, _ = dfs(0)
    best = 0
    for j in range(1, k + 1):
        if dp[j] > best:
            best = dp[j]
    return best


def brute_tree_knapsack(n, parent, weight, k):
    """Exhaustive over all parent-closed subsets of size <= k (empty set allowed)."""
    best = 0
    for mask in range(1 << n):
        size = bin(mask).count("1")
        if size > k:
            continue
        ok = True
        for v in range(1, n):
            if (mask >> v) & 1 and not ((mask >> parent[v]) & 1):
                ok = False
                break
        if not ok:
            continue
        total = 0
        for v in range(n):
            if (mask >> v) & 1:
                total += weight[v]
        if total > best:
            best = total
    return best


def lcg_trees(count, seed, min_n=6, max_n=9):
    trees = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        parent = [-1] * n
        weights = []
        seed = (seed * 75 + 74) % 65537
        weights.append(seed % 31 - 15)
        for v in range(1, n):
            seed = (seed * 75 + 74) % 65537
            parent[v] = seed % v
            seed = (seed * 75 + 74) % 65537
            weights.append(seed % 31 - 15)
        trees.append((n, parent, weights))
    return trees


def main():
    n1, p1, w1 = 4, [-1, 0, 0, 1], [5, 3, 4, -10]
    assert tree_knapsack(n1, p1, w1, 1) == 5
    assert tree_knapsack(n1, p1, w1, 2) == 9
    assert tree_knapsack(n1, p1, w1, 3) == 12
    assert tree_knapsack(n1, p1, w1, 4) == 12
    assert tree_knapsack(n1, p1, w1, 0) == 0
    for k in range(5):
        assert tree_knapsack(n1, p1, w1, k) == brute_tree_knapsack(n1, p1, w1, k)

    n2, p2, w2 = 5, [-1, 0, 0, 1, 1], [10, 8, 5, 7, 6]
    assert tree_knapsack(n2, p2, w2, 3) == 25
    for k in range(6):
        assert tree_knapsack(n2, p2, w2, k) == brute_tree_knapsack(n2, p2, w2, k)

    n3, p3, w3 = 2, [-1, 0], [-2, -3]
    assert tree_knapsack(n3, p3, w3, 2) == 0
    assert tree_knapsack(n3, p3, w3, 1) == 0

    n4, p4, w4 = 3, [-1, 0, 1], [1, -5, 10]
    assert tree_knapsack(n4, p4, w4, 1) == 1
    assert tree_knapsack(n4, p4, w4, 2) == 1
    assert tree_knapsack(n4, p4, w4, 3) == 6
    for k in range(4):
        assert tree_knapsack(n4, p4, w4, k) == brute_tree_knapsack(n4, p4, w4, k)

    checked = 0
    comparisons = 0
    expected_comparisons = 0
    for seed in (4581, 4582, 4583):
        for n, parent, weights in lcg_trees(5, seed):
            previous = 0
            for k in range(n + 1):
                got = tree_knapsack(n, parent, weights, k)
                want = brute_tree_knapsack(n, parent, weights, k)
                assert got == want, (n, parent, weights, k, got, want)
                assert got >= previous
                previous = got
                comparisons += 1
            expected_comparisons += n + 1
            checked += 1
    assert checked == 15
    assert comparisons == expected_comparisons


if __name__ == "__main__":
    main()
    print("PASS")
