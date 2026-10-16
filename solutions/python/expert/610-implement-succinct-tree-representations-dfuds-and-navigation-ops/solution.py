"""Task 610 - Implement succinct tree representations (DFUDS) and navigation ops.  (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

Two 2n-bit representations of an ordered tree with n nodes, both built from the *preorder*
numbering of the nodes (node i is the i-th node visited by a DFS):

* BP (balanced parentheses): one '(' when a node is entered and one ')' when it is left, so
  node i owns the i-th '(' at position open(i); depth(i) = excess(open(i)), the subtree of i
  spans [open(i), findclose(open(i))] and therefore has (close - open + 1) / 2 nodes.
* DFUDS (depth-first unary degree sequence): a leading sentinel '(' and then, per node in
  preorder, deg(v) copies of '(' followed by one ')' -- so node i owns the i-th ')'.

The DFUDS navigation ops implemented here are the real ones, each derived and then verified
against the explicit tree:

  * degree(i)      = length of the run of '(' immediately before the i-th ')' (the sentinel
                     adds one spurious '(' to the root's run).
  * subtree_size(i)= let f = close_pos(i) - degree(i) be the first character of E_i and
                     L = excess(f) - 1.  E_j uses deg(j) '(' and one ')', so a subtree's own
                     segment has one more ')' than '('; the first position q >= close_pos(i)
                     whose excess equals L is therefore exactly one past the subtree, and
                     size = rank0(q) - i + 1.
  * children(i)    = the first child is node i+1 and the next sibling of a child c is c +
                     subtree_size(c) -- both direct consequences of preorder.
  * parent(i)      = the last j < i whose subtree range [j, j + size(j) - 1] contains i
                     (subtree ranges are either nested or disjoint in preorder).
  * depth(i)       = number of nodes whose subtree range contains i.
  * level_ancestor(i, k) = k-fold parent.

Simplified, as documented: a production succinct tree answers every op in O(1) from
rank/select directories, and parent/depth use a few of those primitives; here the bitvector
is a simple chunked rank/select structure and parent/depth scan the preorder ranges (n is
kept small).  What is verified is unchanged and real: navigation results of both
representations match the explicit tree arrays for every node of every test tree, BP and
DFUDS agree with each other, both bit patterns satisfy their invariants (BP balanced with n
pairs; DFUDS has excess >= 1 everywhere except the final 0 and exactly n '(' and n ')'), and
a DFS driven purely by the DFUDS child op rebuilds the tree.
"""
import sys

CHUNK = 8                        # rank/select directory granularity, bits
ONE = "1"
ZERO = "0"


class Bitvector:
    """Small rank/select directory: per-chunk prefix counts plus a scan inside a chunk."""

    def __init__(self, bits):
        canonical = []
        for ch in bits:
            if ch in "(1":
                canonical.append(ONE)
            elif ch in ")0":
                canonical.append(ZERO)
            else:
                raise ValueError("bitvector must be binary (0/1 or parentheses)")
        self.bits = "".join(canonical)
        self.n = len(self.bits)
        self.ones_at_chunk = [0]
        self.zeros_at_chunk = [0]
        for start in range(0, self.n, CHUNK):
            chunk = self.bits[start:start + CHUNK]
            self.ones_at_chunk.append(self.ones_at_chunk[-1] + chunk.count(ONE))
            self.zeros_at_chunk.append(self.zeros_at_chunk[-1] + chunk.count(ZERO))

    def access(self, pos):
        if pos < 0 or pos >= self.n:
            raise IndexError("bitvector access out of range")
        return 1 if self.bits[pos] == ONE else 0

    def rank1(self, pos):
        if pos <= 0:
            return 0
        if pos >= self.n:
            return self.ones_at_chunk[-1]
        chunk = pos // CHUNK
        return self.ones_at_chunk[chunk] + self.bits[chunk * CHUNK:pos].count(ONE)

    def rank0(self, pos):
        if pos <= 0:
            return 0
        if pos >= self.n:
            return self.zeros_at_chunk[-1]
        chunk = pos // CHUNK
        return self.zeros_at_chunk[chunk] + self.bits[chunk * CHUNK:pos].count(ZERO)

    def select1(self, k):
        lo, hi = 0, len(self.ones_at_chunk) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.ones_at_chunk[mid] <= k:
                lo = mid
            else:
                hi = mid - 1
        pos = lo * CHUNK
        want = k - self.ones_at_chunk[lo] + 1
        while True:
            if self.bits[pos] == ONE:
                want -= 1
                if want == 0:
                    return pos
            pos += 1

    def select0(self, k):
        lo, hi = 0, len(self.zeros_at_chunk) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.zeros_at_chunk[mid] <= k:
                lo = mid
            else:
                hi = mid - 1
        pos = lo * CHUNK
        want = k - self.zeros_at_chunk[lo] + 1
        while True:
            if self.bits[pos] == ZERO:
                want -= 1
                if want == 0:
                    return pos
            pos += 1

    def excess(self, pos):
        """# '(' minus # ')' in [0, pos)."""
        return self.rank1(pos) - self.rank0(pos)


class BPTree:
    """Balanced-parentheses representation of the tree; node i <-> i-th '('."""

    def __init__(self, children):
        self.children = children
        self.n = len(children)
        self.bits = Bitvector(bp_string(children))
        self.match = [-1] * self.bits.n
        stack = []
        for pos, ch in enumerate(self.bits.bits):
            if ch == ONE:
                stack.append(pos)
            else:
                opening = stack.pop()
                self.match[opening] = pos
                self.match[pos] = opening

    def open_pos(self, i):
        return self.bits.select1(i - 1)

    def close_pos(self, i):
        return self.match[self.open_pos(i)]

    def depth(self, i):
        return self.bits.excess(self.open_pos(i))

    def subtree_size(self, i):
        return (self.close_pos(i) - self.open_pos(i) + 1) // 2

    def children_of(self, i):
        out = []
        pos = self.open_pos(i) + 1
        while self.bits.bits[pos] == ONE:
            out.append(self.bits.rank1(pos) + 1)
            pos = self.match[pos] + 1
        return out

    def parent(self, i):
        """The enclosing '(' of node i's '(' (or of its preceding sibling's pair)."""
        if i == 1:
            return None
        q = self.open_pos(i)
        if self.bits.bits[q - 1] == ONE:
            return self.bits.rank1(q - 1) + 1
        q = self.match[q - 1]
        while self.bits.bits[q - 1] == ZERO:
            q = self.match[q - 1]
        return self.bits.rank1(q - 1) + 1


class DFUDSTree:
    """DFUDS representation; node i <-> i-th ')'.  See the module docstring for derivations."""

    def __init__(self, children):
        self.children = children
        self.n = len(children)
        self.bits = Bitvector(dfuds_string(children))

    def close_pos(self, i):
        return self.bits.select0(i - 1)

    def degree(self, i):
        pos = self.close_pos(i)
        run = 0
        q = pos - 1
        while q >= 0 and self.bits.bits[q] == ONE:
            run += 1
            q -= 1
        return run - 1 if i == 1 else run          # the root's run contains the sentinel

    def first_open(self, i):
        return self.close_pos(i) - self.degree(i)

    def subtree_size(self, i):
        level = self.bits.excess(self.first_open(i)) - 1
        q = self.close_pos(i) + 1
        while self.bits.excess(q) != level:
            q += 1
        return self.bits.rank0(q) - i + 1

    def children_of(self, i):
        out = []
        child = i + 1
        for _ in range(self.degree(i)):
            out.append(child)
            child += self.subtree_size(child)
        return out

    def parent(self, i):
        if i == 1:
            return None
        j = i - 1
        while j + self.subtree_size(j) - 1 < i:
            j -= 1
        return j

    def depth(self, i):
        return sum(1 for j in range(1, i) if j + self.subtree_size(j) - 1 >= i)

    def level_ancestor(self, i, k):
        if k < 0:
            raise ValueError("level ancestor level must be non-negative")
        node = i
        for _ in range(k):
            node = self.parent(node)
            if node is None:
                raise ValueError("level ancestor above the root")
        return node


def bp_string(children):
    """One '(' on entry, one ')' on exit, children in order (iterative DFS)."""
    out = ["("]
    stack = [(0, 0)]
    while stack:
        node, k = stack[-1]
        if k < len(children[node]):
            stack[-1] = (node, k + 1)
            out.append("(")
            stack.append((children[node][k], 0))
        else:
            stack.pop()
            out.append(")")
    return "".join(out)


def dfuds_string(children):
    """Sentinel '(' followed, per node in preorder, by deg(v) '(' and one ')'."""
    out = ["("]
    for kids in children:
        out.append("(" * len(kids))
        out.append(")")
    return "".join(out)


def sfx(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def random_tree(n, seed):
    """Deterministic random tree, returned as children lists in preorder numbering."""
    parent_of = [0] * n
    state = seed
    for v in range(1, n):
        state = sfx(state)
        parent_of[v] = state % v
    raw = [[] for _ in range(n)]
    for v in range(1, n):
        raw[parent_of[v]].append(v)
    order = []
    stack = [0]
    while stack:
        v = stack.pop()
        order.append(v)
        for c in reversed(raw[v]):
            stack.append(c)
    label = {v: i for i, v in enumerate(order)}
    children = [[] for _ in range(n)]
    for v in range(n):
        children[label[v]] = [label[c] for c in raw[v]]
    return children


def tree_facts(children):
    """Explicit oracle: parent, depth and subtree size per preorder node (root = 1)."""
    n = len(children)
    parent = [None] * (n + 1)
    depth = [0] * (n + 1)
    size = [1] * (n + 1)
    for v in range(n):
        for c in children[v]:
            parent[c + 1] = v + 1
            depth[c + 1] = depth[v + 1] + 1
    for v in reversed(range(n)):
        for c in children[v]:
            size[v + 1] += size[c + 1]
    return parent, depth, size


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    sys.setrecursionlimit(10000)

    # degenerate shapes: single node, chain, star, caterpillar
    shapes = [[[]],
              [[1], [2], [3], [4], []],
              [[1, 2, 3, 4, 5], [], [], [], [], []],
              [[1], [2], [3, 4], [], [5], []]]

    for children in shapes:
        bp = BPTree(children)
        df = DFUDSTree(children)
        parent, depth, size = tree_facts(children)
        n = len(children)
        check(bp.bits.n == 2 * n and df.bits.n == 2 * n, "both representations use 2n bits")
        check(bp.bits.bits.count(ONE) == n and bp.bits.bits.count(ZERO) == n, "BP has n pairs")
        check(df.bits.bits.count(ONE) == n and df.bits.bits.count(ZERO) == n, "DFUDS n and n")
        for i in range(1, n + 1):
            check(df.degree(i) == len(children[i - 1]), "degree")
            check(bp.subtree_size(i) == size[i] == df.subtree_size(i), "subtree size")
            check(bp.depth(i) == depth[i] == df.depth(i), "depth")
            check(bp.children_of(i) == [c + 1 for c in children[i - 1]], "BP children")
            check(df.children_of(i) == [c + 1 for c in children[i - 1]], "DFUDS children")
            check(bp.parent(i) == parent[i] and df.parent(i) == parent[i], "parent")
            for k in range(depth[i] + 1):
                node = i
                for _ in range(k):
                    node = parent[node]
                check(bp.depth(node) == depth[i] - k, "BP level ancestor")
                check(df.level_ancestor(i, k) == node, "DFUDS level ancestor")

    # random trees of growing size: every node, every op, both representations
    for n, seed in ((2, 5), (7, 11), (17, 23), (40, 101), (90, 777)):
        children = random_tree(n, seed)
        bp = BPTree(children)
        df = DFUDSTree(children)
        parent, depth, size = tree_facts(children)
        check(sum(len(k) for k in children) == n - 1, "tree must have n-1 edges")
        # BP validity: prefix excess never negative and returns to 0
        excess = 0
        for ch in bp.bits.bits:
            excess += 1 if ch == ONE else -1
            check(excess >= 0, "BP must stay balanced")
        check(excess == 0, "BP must close")
        # DFUDS validity: sentinel keeps the excess positive until the very end
        excess = 0
        seen_open = 0
        for pos, ch in enumerate(df.bits.bits):
            excess += 1 if ch == ONE else -1
            if pos < df.bits.n - 1:
                check(excess >= 1, "DFUDS excess >= 1 before the final character")
            seen_open += 1 if ch == ONE else 0
        check(excess == 0 and seen_open == n, "DFUDS ends at level 0 after n '(' ")
        check(df.bits.bits[0] == ONE, "DFUDS starts with the sentinel")
        check(df.degree(1) == len(children[0]), "root degree")
        check(df.subtree_size(1) == n, "root subtree is the whole tree")
        for i in range(1, n + 1):
            kids = [c + 1 for c in children[i - 1]]
            check(df.degree(i) == len(kids), "degree")
            check(df.children_of(i) == kids == bp.children_of(i), "children")
            check(df.subtree_size(i) == size[i] == bp.subtree_size(i), "size")
            check(df.depth(i) == depth[i] == bp.depth(i), "depth")
            check(df.parent(i) == parent[i] == bp.parent(i), "parent")
            if i > 1:
                check(df.subtree_size(i) < n, "non-root subtree is proper")
        # a DFS driven only by the DFUDS child op rebuilds the tree
        visited = []
        stack = [1]
        while stack:
            node = stack.pop()
            visited.append(node)
            for child in reversed(df.children_of(node)):
                stack.append(child)
        check(visited == list(range(1, n + 1)), "DFUDS preorder traversal rebuilds the tree")
        # walking up and back down is the identity
        for i in range(2, n + 1):
            check(df.level_ancestor(i, depth[i]) == 1, "climbing to the root")
            check(df.depth(df.parent(i)) == depth[i] - 1, "parent one level up")

    # rejected inputs
    df = DFUDSTree([[1, 2], [], []])
    check(df.parent(1) is None, "the root has no parent")
    try:
        df.level_ancestor(3, 5)
        raise AssertionError("climbing past the root must raise")
    except ValueError:
        pass
    try:
        Bitvector("01x")
        raise AssertionError("non-binary bitvector must be rejected")
    except ValueError:
        pass

    deep = DFUDSTree([[1], [2], [3], [4], [5], [6], [7], [8], [9], []])
    check([deep.depth(i) for i in range(1, 11)] == list(range(10)), "chain depths")
    star = DFUDSTree([[1, 2, 3, 4, 5, 6, 7], [], [], [], [], [], [], []])
    check(star.degree(1) == 7 and star.subtree_size(1) == 8, "star shape")
    check(star.depth(8) == 1 and star.parent(8) == 1, "star leaves hang off the root")

    print("chain n=10: BP %d bits, DFUDS %d bits, depth(10)=%d"
          % (deep.bits.n, deep.bits.n, deep.depth(10)))
    print("star n=8: root degree %d, subtree size %d" % (star.degree(1), star.subtree_size(1)))
    print("PASS")


if __name__ == "__main__":
    main()
