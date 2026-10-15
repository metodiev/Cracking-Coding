"""Task 397 - Implement Knuth optimization (DP speedup demonstration). (Tier: Intermediate, CTCI Ch 8)

Optimal binary search tree over weighted keys: e[i][j] = min_k e[i][k-1] + e[k+1][j] + W(i,j).
Knuth (1971) proved that for this cost opt[i][j-1] <= opt[i][j] <= opt[i+1][j], so each k-loop
only scans that window. Summed over all ranges the windows are O(n^2) total, turning the
plain O(n^3) DP into O(n^2). Matrix-chain multiplication is the classic NON-example - its
optimal split is not monotone (verified in this repo), which is exactly why the example here
is the search tree, not the chain.
The suite asserts the opt-monotonicity invariant, rebuilds the optimal tree and re-scores it
independently, and cross-checks against the plain O(n^3) DP on random weights.
"""

INF = float("inf")


def obst_plain(weights):
    """weights: list of n positive weights; returns the minimum weighted search cost."""
    n = len(weights)
    if n == 0:
        return 0
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + weights[i - 1]
    e = [[0] * (n + 2) for _ in range(n + 2)]
    for i in range(1, n + 1):
        e[i][i] = weights[i - 1]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            total = prefix[j] - prefix[i - 1]
            best = INF
            for k in range(i, j + 1):
                left = e[i][k - 1] if k - 1 >= i else 0
                right = e[k + 1][j] if k + 1 <= j else 0
                candidate = left + right + total
                if candidate < best:
                    best = candidate
            e[i][j] = best
    return e[1][n]


def obst_knuth(weights):
    """Knuth-optimised DP; returns (cost, opt) with opt[i][j] the best root index."""
    n = len(weights)
    if n == 0:
        return 0, [[0] * (n + 2) for _ in range(n + 2)]
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + weights[i - 1]
    e = [[0] * (n + 2) for _ in range(n + 2)]
    opt = [[0] * (n + 2) for _ in range(n + 2)]
    for i in range(1, n + 1):
        e[i][i] = weights[i - 1]
        opt[i][i] = i
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            total = prefix[j] - prefix[i - 1]
            lo = max(opt[i][j - 1], i)
            hi = min(opt[i + 1][j], j)
            best = INF
            best_k = lo
            for k in range(lo, hi + 1):
                left = e[i][k - 1] if k - 1 >= i else 0
                right = e[k + 1][j] if k + 1 <= j else 0
                candidate = left + right + total
                if candidate < best:
                    best = candidate
                    best_k = k
            e[i][j] = best
            opt[i][j] = best_k
    return e[1][n], opt


def opt_is_monotone(opt, n):
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            if opt[i][j] < opt[i][j - 1] or opt[i][j] > opt[i + 1][j]:
                return False
    return True


def rebuild_cost(weights, opt, i, j, depth=0):
    """Rebuilds the tree from the opt table and re-scores it by walking the structure."""
    if i > j:
        return 0
    root = opt[i][j]
    return (
        weights[root - 1] * (depth + 1)
        + rebuild_cost(weights, opt, i, root - 1, depth + 1)
        + rebuild_cost(weights, opt, root + 1, j, depth + 1)
    )


def rebuild_inorder(opt, i, j):
    if i > j:
        return []
    root = opt[i][j]
    return rebuild_inorder(opt, i, root - 1) + [root] + rebuild_inorder(opt, root + 1, j)


def lcg_weight_sets(count, seed, max_n=40):
    sets = []
    for _ in range(count):
        n = 2 + seed % (max_n - 1)
        weights = []
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            weights.append(1 + seed % 50)
        sets.append(weights)
    return sets


def main():
    assert obst_plain([5]) == 5
    assert obst_knuth([5])[0] == 5
    assert obst_plain([1, 1]) == 3
    assert obst_knuth([1, 1])[0] == 3
    # [3,2,1]: root 3 -> 3*1 + 2*2 + 1*3 = 10, root 2 -> 3*2 + 2*1 + 1*2 = 10
    assert obst_plain([3, 2, 1]) == 10
    assert obst_knuth([3, 2, 1])[0] == 10
    # four equal keys: balanced tree = 1*1 + 2*2 + 1*3 = 8
    assert obst_plain([1, 1, 1, 1]) == 8
    assert obst_knuth([1, 1, 1, 1])[0] == 8

    for weights in lcg_weight_sets(120, seed=2026):
        plain = obst_plain(weights)
        knuth, opt = obst_knuth(weights)
        n = len(weights)
        assert knuth == plain, (weights, plain, knuth)
        assert opt_is_monotone(opt, n), weights
        assert rebuild_cost(weights, opt, 1, n) == knuth, weights
        assert rebuild_inorder(opt, 1, n) == list(range(1, n + 1)), weights

    # the windows scan far less than the n^3/6 candidates of the plain DP
    weights = lcg_weight_sets(1, seed=31415, max_n=40)[0]
    n = len(weights)
    _, opt = obst_knuth(weights)
    scanned = sum(opt[i][j] - opt[i][j - 1] + 1
                  for length in range(2, n + 1)
                  for i in range(1, n - length + 2)
                  for j in [i + length - 1])
    assert scanned < n * n, "Knuth windows stay quadratic"


if __name__ == "__main__":
    main()
    print("PASS")
