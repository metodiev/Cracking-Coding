"""Task 459 - Implement virtual tree building from set of nodes (for queries). (Tier: Advanced, CTCI Ch 4)

A virtual tree (auxiliary tree) of a node set S compresses the original tree to S plus the LCAs
of neighbours in DFS order, so that queries about S run on O(|S|) vertices. Construction: sort S
by entry time tin, keep a stack with the current rightmost virtual root; for the next vertex v
compute l = lca(v, stack top): if l is the top, push v; otherwise pop while the second stack
element is at least as deep as l, add the popped edge, and if the top is not l replace it by l
before pushing v. The virtual tree therefore contains every selected node and every LCA needed,
with edge weight equal to the original distance (dist values), so path sums in it match the
original pairwise distances exactly.

Assertions: every built node is a selected node or an ancestor of one; all selected nodes are
present; edge weights equal dist[child] - dist[parent] > 0; the build has nodes - 1 edges (a
tree); nodes <= 2|S| - 1; and for every pair of selected nodes the virtual-tree path weight
equals the naive tree distance (dist[u] + dist[v] - 2*dist[lca]) on LCG trees and hand cases
(chain S={2,4} -> edge weight 2; star S={1,2} -> root 0 added as LCA; single node -> no edges).
"""


def build_tree_data(n, parent, edge_weight):
    """Returns children, tin, depth, dist, and an ancestor test based on the parent array."""
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)
    tin = [0] * n
    depth = [0] * n
    dist = [0] * n
    order = []
    stack = [0]
    while stack:
        u = stack.pop()
        tin[u] = len(order)
        order.append(u)
        for child in reversed(children[u]):
            depth[child] = depth[u] + 1
            dist[child] = dist[u] + edge_weight[child]
            stack.append(child)
    return children, tin, depth, dist


def lca(parent, depth, u, v):
    while depth[u] > depth[v]:
        u = parent[u]
    while depth[v] > depth[u]:
        v = parent[v]
    while u != v:
        u = parent[u]
        v = parent[v]
    return u


def is_ancestor(parent, candidate, node):
    while node != -1:
        if node == candidate:
            return True
        node = parent[node]
    return False


def naive_distance(parent, depth, dist, u, v):
    return dist[u] + dist[v] - 2 * dist[lca(parent, depth, u, v)]


def build_virtual_tree(parent, depth, dist, tin, selected):
    """Returns (nodes, edges) where edges are (vt_parent, vt_child, weight)."""
    order = sorted(selected, key=lambda v: tin[v])
    if not order:
        return [], []
    stack = [order[0]]
    edges = []
    for v in order[1:]:
        l = lca(parent, depth, v, stack[-1])
        if l == stack[-1]:
            stack.append(v)
            continue
        while len(stack) >= 2 and depth[stack[-2]] >= depth[l]:
            edges.append((stack[-2], stack[-1], dist[stack[-1]] - dist[stack[-2]]))
            stack.pop()
        if stack[-1] != l:
            edges.append((l, stack[-1], dist[stack[-1]] - dist[l]))
            stack[-1] = l
        stack.append(v)
    while len(stack) >= 2:
        edges.append((stack[-2], stack[-1], dist[stack[-1]] - dist[stack[-2]]))
        stack.pop()
    nodes = set(selected)
    for a, b, _ in edges:
        nodes.add(a)
        nodes.add(b)
    return sorted(nodes), edges


def virtual_tree_path_weight(nodes, edges, source, target):
    adj = {v: [] for v in nodes}
    for a, b, w in edges:
        adj[a].append((b, w))
        adj[b].append((a, w))
    seen = {source}
    stack = [(source, 0)]
    while stack:
        node, total = stack.pop()
        if node == target:
            return total
        for nxt, w in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, total + w))
    return None


def lcg_trees(count, seed, min_n=8, max_n=12):
    trees = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        parent = [-1] * n
        edge_weight = [0] * n
        for v in range(1, n):
            seed = (seed * 75 + 74) % 65537
            parent[v] = seed % v
            seed = (seed * 75 + 74) % 65537
            edge_weight[v] = 1 + seed % 10
        trees.append((n, parent, edge_weight))
    return trees


def lcg_selection(n, seed, size):
    seed = (seed * 75 + 74) % 65537
    pool = list(range(n))
    selected = []
    count = min(size, n)
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        idx = seed % len(pool)
        selected.append(pool[idx])
        pool[idx] = pool[-1]
        pool.pop()
        seed = (seed * 75 + 74) % 65537
    return selected


def check_virtual_tree(n, parent, edge_weight, selected, depth, dist, tin, naive):
    nodes, edges = build_virtual_tree(parent, depth, dist, tin, selected)
    assert set(selected) <= set(nodes)
    for v in nodes:
        assert any(is_ancestor(parent, v, s) for s in selected), v
    assert len(edges) == len(nodes) - 1
    assert len(nodes) <= 2 * len(selected) - 1
    for a, b, w in edges:
        assert w == dist[b] - dist[a] and w > 0, (a, b, w)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            u, v = selected[i], selected[j]
            got = virtual_tree_path_weight(nodes, edges, u, v)
            want = naive(u, v)
            assert got == want, (u, v, got, want)
    return nodes, edges


def main():
    # chain 0 - 1 - 2 - 3 - 4 with unit weights
    chain_parent = [-1, 0, 1, 2, 3]
    chain_weight = [0, 1, 1, 1, 1]
    children, tin, depth, dist = build_tree_data(5, chain_parent, chain_weight)
    naive = lambda u, v: naive_distance(chain_parent, depth, dist, u, v)

    nodes, edges = check_virtual_tree(5, chain_parent, chain_weight, [2, 4], depth, dist, tin, naive)
    assert nodes == [2, 4] and edges == [(2, 4, 2)]
    nodes, edges = check_virtual_tree(5, chain_parent, chain_weight, [1, 3], depth, dist, tin, naive)
    assert nodes == [1, 3] and edges == [(1, 3, 2)]
    nodes, edges = check_virtual_tree(5, chain_parent, chain_weight, [0, 3], depth, dist, tin, naive)
    assert nodes == [0, 3] and edges == [(0, 3, 3)]
    nodes, edges = check_virtual_tree(5, chain_parent, chain_weight, [1], depth, dist, tin, naive)
    assert nodes == [1] and edges == []

    # star: root 0 with children 1, 2, 3 (unit edge weights)
    star_parent = [-1, 0, 0, 0]
    star_weight = [0, 1, 1, 1]
    children, tin, depth, dist = build_tree_data(4, star_parent, star_weight)
    naive = lambda u, v: naive_distance(star_parent, depth, dist, u, v)
    nodes, edges = check_virtual_tree(4, star_parent, star_weight, [1, 2], depth, dist, tin, naive)
    assert nodes == [0, 1, 2]
    assert sorted(edges) == [(0, 1, 1), (0, 2, 1)]
    nodes, edges = check_virtual_tree(4, star_parent, star_weight, [1, 2, 3], depth, dist, tin, naive)
    assert nodes == [0, 1, 2, 3]

    # balanced tree, all leaves selected
    bal_parent = [-1, 0, 0, 1, 1, 2, 2]
    bal_weight = [0, 3, 4, 5, 6, 7, 8]
    children, tin, depth, dist = build_tree_data(7, bal_parent, bal_weight)
    naive = lambda u, v: naive_distance(bal_parent, depth, dist, u, v)
    nodes, edges = check_virtual_tree(7, bal_parent, bal_weight, [3, 4, 5, 6], depth, dist, tin, naive)
    assert 0 in nodes and 1 in nodes and 2 in nodes

    checked = 0
    trees = 0
    for seed in (4591, 4592, 4593, 4594):
        for n, parent, edge_weight in lcg_trees(4, seed):
            children, tin, depth, dist = build_tree_data(n, parent, edge_weight)
            naive = lambda u, v: naive_distance(parent, depth, dist, u, v)
            seed2 = seed + n
            for extra in range(2):
                size = 2 + seed2 % 4
                selected = lcg_selection(n, seed2 + extra * 3, size)
                check_virtual_tree(n, parent, edge_weight, selected, depth, dist, tin, naive)
                checked += 1
            trees += 1
    assert trees == 16
    assert checked == 32


if __name__ == "__main__":
    main()
    print("PASS")
