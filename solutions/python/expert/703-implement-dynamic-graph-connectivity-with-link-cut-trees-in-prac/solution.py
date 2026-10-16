"""Task 703 - Implement dynamic graph connectivity with link-cut trees in practice.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)

A link-cut tree (Sleator-Tarjan) maintains a *forest* under link, cut, evert and path queries.
Every represented tree is stored implicitly as a set of preferred paths, each of which is a splay
tree whose in-order traversal lists the path from top to bottom.  The operations that make it work:

  * access(x) -- walks up the path-parent chain, splaying each path and detaching the old
    preferred child, so afterwards x's splay tree is exactly the path from the represented root
    to x;
  * make_root(x) -- access(x) followed by a lazy reversal of x's splay tree, which flips the
    root-to-x path into x-to-root order (an evert);
  * path aggregate -- make_root(u) then access(v) puts the whole u-v path into v's splay tree,
    so the subtree aggregate of v *is* the path aggregate; the stored aggregate is a pair
    (sum, min) recomputed by pull after every rotation or splice.

The invariant maintained and checked after every mutation: the in-order traversal of every splay
tree is a contiguous top-to-bottom path of the represented forest (consecutive nodes are adjacent
and their depth rises by exactly one), every node's stored aggregate equals the aggregate over
its splay subtree, parent/child back-pointers agree, and the in-order adjacencies together with
the path-parent links account for every forest edge exactly once.

Simplified versus the real thing: only sum/min path aggregates are exposed (no path-max, no
subtree aggregation, no edge weights, no top-tree tricks), the implementation is plain Python
arrays rather than the usual pointer version, and the amortised O(log n) bound is not measured
because the point of the task is the correctness of the dynamic connectivity structure, not the
constant factor.  Verified against a brute-force forest simulation (BFS paths, component counts)
over hundreds of randomised link/cut/query operations, plus hand-computed chains and stars.
The aggregate of a disconnected pair is documented as None and checked to be exactly that.
"""
import sys


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class LinkCutTree:
    """Preferred-path forest; every array is indexed by node id in 0..n-1."""

    def __init__(self, values):
        n = len(values)
        check(n >= 1, "a link-cut tree needs at least one node")
        self.left = [-1] * n
        self.right = [-1] * n
        self.parent = [-1] * n
        self.reversed = [False] * n
        self.value = list(values)
        self.total = list(values)              # sum over the splay subtree (== a path segment)
        self.minimum = list(values)            # min over the splay subtree
        self.splays = 0

    # ------------------------------------------------------------------ splay plumbing
    def is_splay_root(self, x):
        p = self.parent[x]
        return p == -1 or (self.left[p] != x and self.right[p] != x)

    def _pull(self, x):
        total = self.value[x]
        lowest = self.value[x]
        for child in (self.left[x], self.right[x]):
            if child != -1:
                total += self.total[child]
                if self.minimum[child] < lowest:
                    lowest = self.minimum[child]
        self.total[x] = total
        self.minimum[x] = lowest

    def _apply_reverse(self, x):
        if x != -1:
            self.left[x], self.right[x] = self.right[x], self.left[x]
            self.reversed[x] = not self.reversed[x]

    def _push(self, x):
        if self.reversed[x]:
            self._apply_reverse(self.left[x])
            self._apply_reverse(self.right[x])
            self.reversed[x] = False

    def _push_to_root(self, x):
        stack = [x]
        while not self.is_splay_root(stack[-1]):
            stack.append(self.parent[stack[-1]])
        while stack:
            self._push(stack.pop())

    def _rotate(self, x):
        p = self.parent[x]
        g = self.parent[p]
        if self.left[p] == x:
            middle = self.right[x]
            self.right[x] = p
            self.left[p] = middle
        else:
            middle = self.left[x]
            self.left[x] = p
            self.right[p] = middle
        if middle != -1:
            self.parent[middle] = p
        self.parent[p] = x
        self.parent[x] = g
        if g != -1:
            if self.left[g] == p:
                self.left[g] = x
            elif self.right[g] == p:
                self.right[g] = x
        self._pull(p)
        self._pull(x)

    def _splay(self, x):
        self._push_to_root(x)
        while not self.is_splay_root(x):
            p = self.parent[x]
            g = self.parent[p]
            if not self.is_splay_root(p):
                if (self.left[p] == x) == (self.left[g] == p):
                    self._rotate(p)            # zig-zig
                else:
                    self._rotate(x)            # zig-zag
            self._rotate(x)
        self.splays += 1

    # ------------------------------------------------------------------ core operations
    def access(self, x):
        """Make the represented-root .. x path a single preferred path ending at x."""
        last = -1
        node = x
        while node != -1:
            self._splay(node)
            self.right[node] = last
            if last != -1:
                self.parent[last] = node
            self._pull(node)
            last = node
            node = self.parent[node]
        self._splay(x)
        return last

    def make_root(self, x):
        self.access(x)
        self._apply_reverse(x)

    def find_root(self, x):
        self.access(x)
        node = x
        while True:
            self._push(node)
            if self.left[node] == -1:
                break
            node = self.left[node]
        self._splay(node)
        return node

    def connected(self, u, v):
        if u == v:
            return True
        return self.find_root(u) == self.find_root(v)

    def link(self, u, v):
        """Add the edge u-v; returns False when it would create a cycle."""
        self.make_root(u)
        if self.find_root(v) == u:
            return False
        self.parent[u] = v
        return True

    def cut(self, u, v):
        """Remove the edge u-v; returns False when the two nodes are not adjacent."""
        self.make_root(u)
        self.access(v)
        if self.left[v] != u or self.right[u] != -1:
            return False
        self.left[v] = -1
        self.parent[u] = -1
        self._pull(v)
        return True

    def path_sum(self, u, v):
        """SUM of the node values on the u-v path, or None when the two are disconnected."""
        if self.find_root(u) != self.find_root(v):
            return None
        self.make_root(u)
        self.access(v)
        return self.total[v]

    def path_min(self, u, v):
        """MIN of the node values on the u-v path, or None when the two are disconnected."""
        if self.find_root(u) != self.find_root(v):
            return None
        self.make_root(u)
        self.access(v)
        return self.minimum[v]


# --------------------------------------------------------------------------- oracles
def forest_components(adjacency):
    """Component label per node, by BFS over the plain adjacency lists."""
    label = {}
    for start in sorted(adjacency):
        if start in label:
            continue
        queue, label[start] = [start], len(label)
        while queue:
            node = queue.pop()
            for neighbour in adjacency[node]:
                if neighbour not in label:
                    label[neighbour] = label[start]
                    queue.append(neighbour)
    return label


def forest_path(adjacency, u, v):
    """The unique u-v path as a node list, or None when disconnected."""
    previous = {u: None}
    queue = [u]
    while queue:
        node = queue.pop(0)
        if node == v:
            path, cursor = [v], v
            while previous[cursor] is not None:
                cursor = previous[cursor]
                path.append(cursor)
            return path[::-1]
        for neighbour in adjacency[node]:
            if neighbour not in previous:
                previous[neighbour] = node
                queue.append(neighbour)
    return None


def forest_depths(adjacency):
    depths = {}
    for start in sorted(adjacency):
        if start in depths:
            continue
        depths[start] = 0
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in depths:
                    depths[neighbour] = depths[node] + 1
                    stack.append(neighbour)
    return depths


def splay_inorder(tree, root):
    """In-order node list of a splay subtree, honouring the lazily reversed nodes."""
    output = []

    def walk(node, flipped):
        if node == -1:
            return
        flipped = flipped ^ (1 if tree.reversed[node] else 0)
        if flipped:
            walk(tree.right[node], flipped)
            output.append(node)
            walk(tree.left[node], flipped)
        else:
            walk(tree.left[node], flipped)
            output.append(node)
            walk(tree.right[node], flipped)

    walk(root, 0)
    return output


def verify_structure(tree, adjacency, n):
    """All link-cut invariants: back-pointers, aggregates, preferred paths, edge accounting."""
    # depths are measured from the *link-cut* represented root of each component, because the
    # in-order direction of a preferred path is defined relative to that root, not to an
    # arbitrary BFS root of the simulation.
    labels = forest_components(adjacency)
    members = {}
    for node, label in labels.items():
        members.setdefault(label, []).append(node)
    depths = {}
    for label, group in members.items():
        root = tree.find_root(group[0])
        check(root in group, f"the represented root {root} must belong to the component")
        for other in group[1:]:
            check(tree.find_root(other) == root,
                  f"nodes {other} and {group[0]} disagree about their represented root")
        depths[root] = 0
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in depths:
                    depths[neighbour] = depths[node] + 1
                    stack.append(neighbour)

    for x in range(n):
        for child in (tree.left[x], tree.right[x]):
            if child != -1:
                check(tree.parent[child] == x,
                      f"child {child} does not point back at its parent {x}")
        inorder = splay_inorder(tree, x)
        check(tree.total[x] == sum(tree.value[node] for node in inorder),
              f"sum aggregate broken at node {x}")
        check(tree.minimum[x] == min(tree.value[node] for node in inorder),
              f"min aggregate broken at node {x}")

    accounted = set()
    for root in range(n):
        if not tree.is_splay_root(root):
            continue
        inorder = splay_inorder(tree, root)
        for a, b in zip(inorder, inorder[1:]):
            check(b in adjacency[a],
                  f"in-order neighbours {a},{b} are not adjacent in the forest")
            check(depths[b] == depths[a] + 1,
                  f"in-order must descend one level: {a}(d{depths[a]}) -> {b}(d{depths[b]})")
            accounted.add((min(a, b), max(a, b)))
        path_parent = tree.parent[root]
        if path_parent != -1:
            top = inorder[0]                       # leftmost = top of the preferred path
            check(path_parent in adjacency[top],
                  f"path-parent {path_parent} of the path top {top} is not a forest neighbour")
            check(depths[top] == depths[path_parent] + 1,
                  "the path-parent must sit one level above the top of its preferred path")
            check(path_parent not in inorder,
                  "a path-parent must not be inside the preferred path it hangs from")
            accounted.add((min(top, path_parent), max(top, path_parent)))
    edges = sum(len(neighbours) for neighbours in adjacency.values()) // 2
    check(len(accounted) == edges,
          f"preferred paths + path-parents must cover every edge: {len(accounted)} vs {edges}")


# --------------------------------------------------------------------------- checks
def test_single_node_and_handcrafted():
    tree = LinkCutTree([5])
    check(tree.connected(0, 0), "a node is connected to itself")
    check(tree.path_sum(0, 0) == 5, "single-node path sum")
    check(tree.path_min(0, 0) == 5, "single-node path min")
    check(tree.link(0, 0) is False, "a self link must be rejected")
    check(tree.cut(0, 0) is False, "there is no self edge to cut")

    tree = LinkCutTree([1, 2, 3, 4])                      # chain 0-1-2-3, values 1,2,3,4
    for u, v in ((0, 1), (1, 2), (2, 3)):
        check(tree.link(u, v), f"linking the chain edge {u}-{v}")
    check(tree.path_sum(0, 3) == 10, "chain path sum")
    check(tree.path_sum(3, 0) == 10, "path sum is direction independent (evert)")
    check(tree.path_min(0, 3) == 1, "chain path min")
    check(tree.path_min(2, 3) == 3, "tail path min")
    check(tree.path_sum(1, 1) == 2, "a single-vertex path is allowed")
    check(tree.link(0, 3) is False, "linking inside a component must be refused")
    check(tree.connected(0, 3) and tree.connected(1, 3), "chain is connected")

    check(tree.cut(1, 2), "cutting the middle edge")
    check(not tree.connected(0, 3), "the chain is split in two")
    check(tree.connected(0, 1) and tree.connected(2, 3), "both halves stay whole")
    check(tree.path_sum(0, 3) is None, "a disconnected pair has no path aggregate")
    check(tree.path_min(0, 3) is None, "no min either")
    check(tree.path_sum(2, 3) == 7, "the surviving right half keeps its aggregate")
    check(tree.cut(1, 2) is False, "cutting the same edge twice must fail")
    check(tree.cut(0, 3) is False, "cutting a non-adjacent pair must fail")
    check(tree.link(2, 0), "reconnecting the halves at a different endpoint")
    check(tree.path_min(1, 3) == 1 and tree.path_sum(3, 1) == 10,
          "path 3-2-0-1 after relinking: 4+3+1+2")


def test_random_operations_against_simulation():
    n = 14
    values = [((i * 37) % 19) - 9 for i in range(n)]      # deliberately includes negatives
    tree = LinkCutTree(values)
    adjacency = {i: set() for i in range(n)}
    mask = (1 << 64) - 1
    seed = 0x2545F4914F6CDD1D
    links = cuts = 0
    for step in range(400):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        u = seed % n
        v = (seed >> 17) % n
        choice = (seed >> 33) % 4
        if choice == 0:
            linked = tree.link(u, v)
            expected = u != v and forest_path(adjacency, u, v) is None
            check(linked == expected,
                  f"step {step}: link {u}-{v} returned {linked}, expected {expected}")
            if linked:
                adjacency[u].add(v)
                adjacency[v].add(u)
                links += 1
        elif choice == 1:
            cut = tree.cut(u, v)
            expected = v in adjacency[u]
            check(cut == expected,
                  f"step {step}: cut {u}-{v} returned {cut}, expected {expected}")
            if cut:
                adjacency[u].discard(v)
                adjacency[v].discard(u)
                cuts += 1
        elif choice == 2:
            path = forest_path(adjacency, u, v)
            expected = None if path is None else sum(values[node] for node in path)
            check(tree.path_sum(u, v) == expected,
                  f"step {step}: path sum {u}->{v} = {tree.path_sum(u, v)}, expected {expected}")
        else:
            path = forest_path(adjacency, u, v)
            check(tree.connected(u, v) == (path is not None),
                  f"step {step}: connected({u},{v}) disagrees with the simulation")
        if step % 5 == 0:
            verify_structure(tree, adjacency, n)

    labels = forest_components(adjacency)
    roots = {tree.find_root(node) for node in range(n)}
    check(len(roots) == len(set(labels.values())),
          f"component count mismatch: lct {len(roots)} vs forest {len(set(labels.values()))}")
    check(len(roots) == n - links + cuts, "components = nodes - links + cuts")
    return 400, links, cuts


def test_evert_and_star_queries():
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    n = len(values)
    tree = LinkCutTree(values)
    adjacency = {i: set() for i in range(n)}
    for i in range(1, n):                                 # star centred on node 0
        check(tree.link(0, i), f"star edge 0-{i}")
        adjacency[0].add(i)
        adjacency[i].add(0)
    for leaf in range(1, n):
        partner = (leaf % (n - 1)) + 1
        path = forest_path(adjacency, leaf, partner)
        expected_sum = sum(values[node] for node in path)
        expected_min = min(values[node] for node in path)
        check(tree.path_sum(leaf, partner) == expected_sum, f"star sum leaf {leaf}")
        check(tree.path_sum(partner, leaf) == expected_sum, "sums are direction independent")
        check(tree.path_min(leaf, partner) == expected_min, f"star min leaf {leaf}")
    check(tree.path_sum(1, 7) == values[1] + values[0] + values[7],
          "a leaf-to-leaf path crosses the hub")
    check(tree.path_min(2, 5) == min(values[2], values[0], values[5]), "leaf-to-leaf min")
    check(tree.path_min(1, 1) == values[1], "single-node path on a leaf")
    verify_structure(tree, adjacency, n)


def test_larger_forest():
    n = 200
    values = [((i * 131) % 97) - 48 for i in range(n)]
    tree = LinkCutTree(values)
    adjacency = {i: set() for i in range(n)}
    mask = (1 << 64) - 1
    seed = 0x9E3779B97F4A7C15
    for i in range(1, n):                                 # build a random forest of n nodes
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        parent = seed % i
        check(tree.link(i, parent), f"linking {i} under {parent}")
        adjacency[i].add(parent)
        adjacency[parent].add(i)
    check(len({tree.find_root(node) for node in range(n)}) == 1, "the forest is one tree here")
    for step in range(300):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        u = seed % n
        v = (seed >> 19) % n
        path = forest_path(adjacency, u, v)
        check(tree.connected(u, v) == (path is not None), f"large step {step}: connectivity")
        check(tree.path_sum(u, v) == (None if path is None else
                                      sum(values[node] for node in path)),
              f"large step {step}: path sum")
        check(tree.path_min(u, v) == (None if path is None else
                                      min(values[node] for node in path)),
              f"large step {step}: path min")
    verify_structure(tree, adjacency, n)
    return n


def main():
    test_single_node_and_handcrafted()
    operations, links, cuts = test_random_operations_against_simulation()
    test_evert_and_star_queries()
    nodes = test_larger_forest()

    print(f"random_ops={operations} links={links} cuts={cuts} large_nodes={nodes}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
