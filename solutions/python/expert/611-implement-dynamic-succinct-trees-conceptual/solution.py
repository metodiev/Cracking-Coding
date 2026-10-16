"""Task 611 - Implement dynamic succinct trees (conceptual).  (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

A *dynamic* succinct tree keeps the 2n-bit balanced-parenthesis sequence in a bitvector that
supports insertion and deletion and answers every navigation query from that bitvector alone:
no per-node pointers, no parent array, no labels.  Two layers:

DynamicBitvector - the bits live in a list of machine-word chunks (64 bits; bit j of a chunk is
bit j of its string form) plus a parallel list of chunk *lengths*, since leading zeros inside a
chunk carry information that bit_length() would lose.  Two Fenwick trees over the chunk list
hold the lengths and the popcounts, so rank1(pos) is a Fenwick prefix plus one intra-chunk
scan, select1(k) is a Fenwick search plus one intra-chunk scan, and an edit rewrites a single
chunk.  A chunk that overflows 64 bits is split, a chunk that empties is dropped, and a chunk
below 32 bits is merged with a neighbour - the classic dynamic-bits rebalancing.

DynamicBP - node i owns the i-th '(' (open_pos(i) = select1(i-1)).  depth(i) is the excess
there; the matching ')' (first position whose excess returns to the same level) bounds the
subtree; the first child is the '(' right after an open, the next sibling the '(' right after a
close, and the parent the previous '(' sitting one level higher.  add_leaf(parent, k) inserts
"()" at the right slot and remove_leaf(i) deletes a leaf's pair, so the tree allocates nothing.

Conceptual, as documented: a production dynamic succinct tree keeps the bits in a balanced
binary tree of blocks with O(log n) rank/select and O(1)-amortized findclose from a
BP-matching/pioneer directory; here the directory is two Fenwicks over a Python list of chunks
and findclose/parent are documented O(n) excess scans.  What is *not* simplified, and what the
checks pin down: after every single edit the sequence is still balanced, and the dynamic
representation's navigation (degree, children, first child, next sibling, parent, depth,
subtree size) reproduces an explicitly maintained witness tree for every node, through 150
deterministic random edit sequences.  No threads, no sockets, no randomness but a fixed seed.
"""

CHUNK_BITS = 64


class Fenwick:
    """Point add / prefix sum / search over non-negative counts."""

    def __init__(self, size):
        self.n = max(size, 1)
        self.tree = [0] * (self.n + 1)

    def add(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix(self, count):
        total, i = 0, count
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total

    def search(self, target):
        """Smallest index whose element contains the (target + 1)-th unit."""
        idx, bit = 0, 1 << self.n.bit_length()
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.tree[nxt] <= target:
                idx, target = nxt, target - self.tree[nxt]
            bit >>= 1
        return idx


def popcount(value):
    return bin(value).count("1")


class DynamicBitvector:
    def __init__(self, bits=""):
        value = 0
        for offset, ch in enumerate(bits):
            if ch in "(1":
                value |= 1 << offset
            elif ch not in ")0":
                raise ValueError("bits are 0/1 or parentheses")
        self.chunks, self.sizes = [value], [len(bits)]
        self._rebuild()

    def _rebuild(self):
        self.lengths, self.pops = Fenwick(len(self.chunks)), Fenwick(len(self.chunks))
        for i in range(len(self.chunks)):
            self.lengths.add(i, self.sizes[i])
            self.pops.add(i, popcount(self.chunks[i]))

    def __len__(self):
        return self.lengths.prefix(len(self.chunks))

    def to_string(self):
        return "".join("1" if (self.chunks[b] >> o) & 1 else "0"
                       for b in range(len(self.chunks)) for o in range(self.sizes[b]))

    def access(self, pos):
        if pos < 0 or pos >= len(self):
            raise IndexError("access out of range")
        block = self.lengths.search(pos)
        return (self.chunks[block] >> (pos - self.lengths.prefix(block))) & 1

    def rank1(self, pos):
        """Ones in [0, pos): complete-chunk Fenwick prefix plus one intra-chunk mask."""
        if pos <= 0:
            return 0
        if pos >= len(self):
            return self.pops.prefix(len(self.chunks))
        block = self.lengths.search(pos)
        offset = pos - self.lengths.prefix(block)
        return self.pops.prefix(block) + popcount(self.chunks[block] & ((1 << offset) - 1))

    def select1(self, index):
        if index < 0 or index >= self.rank1(len(self)):
            raise IndexError("select1 out of range")
        block = self.pops.search(index)
        remaining = index - self.pops.prefix(block)
        for offset in range(self.sizes[block]):
            if (self.chunks[block] >> offset) & 1:
                if remaining == 0:
                    return self.lengths.prefix(block) + offset
                remaining -= 1
        raise AssertionError("unreachable: select1 fell off the chunk")

    def insert(self, pos, bit):
        total = len(self)
        if pos < 0 or pos > total:
            raise IndexError("insert out of range")
        if bit not in (0, 1):
            raise ValueError("bits are 0 or 1")
        if pos == total:
            block, offset = len(self.chunks) - 1, self.sizes[-1]
        else:
            block = self.lengths.search(pos)
            offset = pos - self.lengths.prefix(block)
        chunk = self.chunks[block]
        low = chunk & ((1 << offset) - 1)
        self.chunks[block] = low | ((chunk >> offset) << (offset + 1)) | (bit << offset)
        self.sizes[block] += 1
        self.lengths.add(block, 1)
        if bit:
            self.pops.add(block, 1)
        if self.sizes[block] > CHUNK_BITS:
            self._split(block)

    def delete(self, pos):
        if pos < 0 or pos >= len(self):
            raise IndexError("delete out of range")
        block = self.lengths.search(pos)
        offset = pos - self.lengths.prefix(block)
        chunk = self.chunks[block]
        self.chunks[block] = (chunk & ((1 << offset) - 1)) | ((chunk >> (offset + 1)) << offset)
        self.sizes[block] -= 1
        self.lengths.add(block, -1)
        if (chunk >> offset) & 1:
            self.pops.add(block, -1)
        if self.sizes[block] == 0 and len(self.chunks) > 1:
            del self.chunks[block], self.sizes[block]
            self._rebuild()
        elif self.sizes[block] < CHUNK_BITS // 2:
            self._merge(block)

    def _split(self, block):
        length, chunk = self.sizes[block], self.chunks[block]
        half = length // 2
        self.chunks[block], self.sizes[block] = chunk & ((1 << half) - 1), half
        self.chunks.insert(block + 1, chunk >> half)
        self.sizes.insert(block + 1, length - half)
        self._rebuild()

    def _merge(self, block):
        """Fuse a short chunk with a neighbour when the pair still fits into one chunk."""
        if len(self.chunks) == 1:
            return
        first = min(block, block + 1 if block + 1 < len(self.chunks) else block - 1)
        second = max(block, block + 1 if block + 1 < len(self.chunks) else block - 1)
        if self.sizes[first] + self.sizes[second] <= CHUNK_BITS:
            self.chunks[first] |= self.chunks[second] << self.sizes[first]
            self.sizes[first] += self.sizes[second]
            del self.chunks[second], self.sizes[second]
            self._rebuild()


class DynamicBP:
    """Balanced-parenthesis tree over a DynamicBitvector; node i <-> i-th '('."""

    def __init__(self, children=None):
        self.bv = DynamicBitvector("" if children is None else bp_string(children))

    def node_count(self):
        return self.bv.rank1(len(self.bv))

    def open_pos(self, i):
        return self.bv.select1(i - 1)

    def excess(self, pos):
        """Nesting level at pos: number of '(' minus ')' in [0, pos)."""
        return 2 * self.bv.rank1(pos) - pos

    def close_pos(self, i):
        opening = self.open_pos(i)
        level, pos = self.excess(opening), self.open_pos(i) + 1
        while self.excess(pos) != level:
            pos += 1
        return pos - 1

    def depth(self, i):
        return self.excess(self.open_pos(i))

    def subtree_size(self, i):
        return (self.close_pos(i) - self.open_pos(i) + 1) // 2

    def first_child(self, i):
        pos = self.open_pos(i) + 1
        return None if self.bv.access(pos) == 0 else self.bv.rank1(pos) + 1

    def next_sibling(self, i):
        pos = self.close_pos(i) + 1
        if pos >= len(self.bv) or self.bv.access(pos) == 0:
            return None
        return self.bv.rank1(pos) + 1

    def children_of(self, i):
        out, child = [], self.first_child(i)
        while child is not None:
            out.append(child)
            child = self.next_sibling(child)
        return out

    def parent(self, i):
        """The '(' one level up that opens the enclosing pair (O(n) scan, see the header)."""
        if i == 1:
            return None
        opening = self.open_pos(i)
        level = self.excess(opening)
        pos = opening - 1
        while pos >= 0:
            if self.bv.access(pos) == 1 and self.excess(pos) == level - 1:
                return self.bv.rank1(pos) + 1
            pos -= 1
        return None

    def add_leaf(self, parent, position):
        """Insert a fresh leaf as the `position`-th child of `parent` (0-based)."""
        if parent < 1 or parent > self.node_count():
            raise ValueError("unknown parent")
        if position < 0 or position > len(self.children_of(parent)):
            raise ValueError("child position out of range")
        if position == 0:
            at = self.open_pos(parent) + 1
        else:
            pos = self.open_pos(parent) + 1
            child = self.bv.rank1(pos) + 1
            for _ in range(position - 1):
                pos = self.close_pos(child) + 1
                child = self.bv.rank1(pos) + 1
            at = self.close_pos(child) + 1
        self.bv.insert(at, 1)                      # the new leaf's '(' ...
        self.bv.insert(at + 1, 0)                  # ... directly followed by its ')'
        return self.bv.rank1(at) + 1

    def remove_leaf(self, i):
        if i < 1 or i > self.node_count():
            raise ValueError("unknown node")
        if self.children_of(i):
            raise ValueError("only leaves can be removed")
        opening = self.open_pos(i)
        self.bv.delete(opening + 1)                # the ')' first, so the '(' keeps its index
        self.bv.delete(opening)


class TreeModel:
    """Explicit witness: node ids in creation order, children lists in sibling order."""

    def __init__(self):
        self.children, self.deleted, self._order = [[]], set(), None

    def preorder(self):
        if self._order is None:
            order, stack = [], [0]
            while stack:
                node = stack.pop()
                order.append(node)
                stack.extend(c for c in reversed(self.children[node]) if c not in self.deleted)
            self._order = order
        return self._order

    def label(self, node_id):
        return self.preorder().index(node_id) + 1

    def id_of(self, label):
        return self.preorder()[label - 1]

    def children_labels(self, label):
        node = self.id_of(label)
        return [self.label(c) for c in self.children[node] if c not in self.deleted]

    def parent_label(self, label):
        node = self.id_of(label)
        for parent, kids in enumerate(self.children):
            if node in kids:
                return self.label(parent)
        return None

    def depth_of(self, label):
        node, depth = self.id_of(label), 0
        while node != 0:
            node = next(p for p, kids in enumerate(self.children) if node in kids)
            depth += 1
        return depth

    def subtree_size_of(self, label):
        size = 1
        for child in self.children[self.id_of(label)]:
            if child not in self.deleted:
                size += self.subtree_size_of(self.label(child))
        return size

    def add_leaf(self, parent_label, position):
        parent = self.id_of(parent_label)
        kids = [c for c in self.children[parent] if c not in self.deleted]
        new_id = len(self.children)
        self.children.append([])
        if position < len(kids):
            self.children[parent].insert(self.children[parent].index(kids[position]), new_id)
        else:
            self.children[parent].append(new_id)
        self._order = None
        return self.label(new_id)

    def remove_leaf(self, label):
        self.deleted.add(self.id_of(label))
        self._order = None


def bp_string(children):
    out, stack = ["("], [(0, 0)]
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


def sfx(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def is_balanced(text):
    level = 0
    for ch in text:
        level += 1 if ch in "(1" else -1
        if level < 0:
            return False
    return level == 0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect(raises, thunk, message):
    try:
        thunk()
    except raises:
        return
    raise AssertionError(message)


def verify_tree(tree, model):
    """Every navigation op of the dynamic representation vs the explicit witness tree."""
    text = tree.bv.to_string()
    check(is_balanced(text), "the sequence must stay balanced")
    check(len(text) == 2 * tree.node_count(), "exactly 2n bits")
    check(tree.node_count() == len(model.preorder()), "node count matches the witness")
    for label in range(1, tree.node_count() + 1):
        kids = model.children_labels(label)
        check(len(tree.children_of(label)) == len(kids), "degree")
        check(tree.children_of(label) == kids, "children")
        check(tree.first_child(label) == (kids[0] if kids else None), "first child")
        check(tree.parent(label) == model.parent_label(label), "parent")
        check(tree.depth(label) == model.depth_of(label), "depth")
        check(tree.subtree_size(label) == model.subtree_size_of(label), "subtree size")
        for index, child in enumerate(kids):
            check(tree.next_sibling(child) == (kids[index + 1] if index + 1 < len(kids)
                                               else None), "next sibling")


def main():
    # ---- dynamic bitvector against a plain list --------------------------
    bits, bv, state = [], DynamicBitvector(), 12345
    check(len(bv) == 0 and bv.to_string() == "", "empty bitvector")
    for step in range(250):
        state = sfx(state)
        if bits and (state >> 8) % 4 == 0:
            victim = state % len(bits)
            bv.delete(victim)
            del bits[victim]
        else:
            pos, bit = state % (len(bits) + 1), (state >> 16) & 1
            bv.insert(pos, bit)
            bits.insert(pos, bit)
        check(bv.to_string() == "".join(str(b) for b in bits),
              "bitvector contents diverged at step %d" % step)
        check(len(bv) == len(bits), "dynamic length")
        running = 0
        for idx in range(len(bits) + 1):
            check(bv.rank1(idx) == running, "rank1 mismatch")
            if idx < len(bits):
                running += bits[idx]
        for index in range(running):
            check(bv.select1(index) == [i for i, b in enumerate(bits) if b][index],
                  "select1 mismatch")
    check(len(bv) > 100, "the random walk must build a sizeable bitvector")

    wide = DynamicBitvector()
    for i in range(170):                       # crosses the 64-bit word boundary twice
        wide.insert(i, 1 if i < 150 else 0)
    check(wide.to_string() == "1" * 150 + "0" * 20, "wide contents")
    check(len(wide.chunks) >= 3 and all(1 <= s <= CHUNK_BITS for s in wide.sizes),
          "chunks split to stay inside one word")
    check(wide.rank1(149) == 149 and wide.rank1(170) == 150, "wide rank")
    for _ in range(150):
        wide.delete(0)
    check(wide.to_string() == "0" * 20, "after deleting the ones only zeros remain")
    check(len(wide.chunks) == 1 and wide.rank1(len(wide)) == 0, "chunks merged back")
    check(DynamicBitvector("0000").to_string() == "0000", "leading zeros survive")
    zeros = DynamicBitvector("0000")
    zeros.delete(1)
    check(zeros.to_string() == "000", "an all-zero chunk keeps its length")

    tiny = DynamicBitvector("10")
    expect(IndexError, lambda: tiny.access(2), "out-of-range access")
    expect(IndexError, lambda: tiny.insert(3, 1), "insert past the end")
    expect(IndexError, lambda: tiny.delete(-1), "negative delete")
    expect(IndexError, lambda: tiny.select1(2), "select past the last one")
    expect(ValueError, lambda: tiny.insert(0, 2), "bit must be 0 or 1")

    # ---- dynamic tree edits against the explicit witness ------------------
    model = TreeModel()
    model.add_leaf(1, 0)
    model.add_leaf(1, 1)
    tree = DynamicBP([[1, 2], [], []])
    check(tree.node_count() == 3 and len(tree.children_of(1)) == 2, "initial tree")
    for step in range(150):
        state = sfx(state)
        n = tree.node_count()
        if n > 1 and (state >> 5) % 3 == 0:
            label = 1 + state % n
            if not model.children_labels(label):
                tree.remove_leaf(label)
                model.remove_leaf(label)
        else:
            parent = 1 + state % n
            position = (state >> 7) % (len(model.children_labels(parent)) + 1)
            tree.add_leaf(parent, position)
            model.add_leaf(parent, position)
        verify_tree(tree, model)

    # error paths on the dynamic tree
    tree = DynamicBP([[1], []])
    expect(ValueError, lambda: tree.remove_leaf(1), "removing a non-leaf")
    expect(ValueError, lambda: tree.add_leaf(1, 5), "child position beyond the degree")
    expect(ValueError, lambda: tree.add_leaf(42, 0), "unknown parent")
    expect(ValueError, lambda: tree.remove_leaf(99), "unknown node")
    tree.add_leaf(1, 0)
    check(tree.node_count() == 3
          and tree.bv.to_string() == bp_string([[1, 2], [], []]).replace("(", "1")
          .replace(")", "0"), "insert as the first child produces the expected sequence")

    print("dynamic bitvector: 250 random edits, final length %d over %d chunk(s)"
          % (len(bv), len(bv.chunks)))
    print("dynamic tree: %d nodes, %d bits, root degree %d"
          % (tree.node_count(), len(tree.bv.to_string()), len(tree.children_of(1))))
    print("PASS")


if __name__ == "__main__":
    main()
