"""Task 436 - Implement rope with concatenation and split in O(log n). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Rope = balanced binary tree over text chunks. Documented thresholds: every leaf holds at most
LEAF_LIMIT = 32 characters (a longer input string is pre-split into such leaves), and after every
concat/split the root height is checked against 2 * ceil(log2(#leaves + 1)) + 1 - if a rope grows
taller than that it is rebuilt from its leaves into a perfectly balanced tree. Each node caches its
character size, height and leaf count, so concat is O(1) plus an amortised rebuild, and split
recurses along one root-to-leaf path (plus an amortised rebuild of the two results). The rebuild
counter is documented and asserted: building a 200-character rope by 200 single-character concats
triggers at least one rebalance. Assertions compare to_string() with a plain-string oracle across
LCG operation scripts including splits at 0 and at the full length, and check leaf, size and height
invariants after every operation.
"""

import math

LEAF_LIMIT = 32


class Node:
    __slots__ = ("text", "left", "right", "size", "height", "leaves")

    def __init__(self, text=None, left=None, right=None):
        self.text = text
        if text is not None:
            self.left = None
            self.right = None
            self.size = len(text)
            self.height = 1
            self.leaves = 1
        else:
            self.left = left
            self.right = right
            self.size = left.size + right.size
            self.height = 1 + max(left.height, right.height)
            self.leaves = left.leaves + right.leaves


def leaves_of(text):
    return [text[i:i + LEAF_LIMIT] for i in range(0, len(text), LEAF_LIMIT)]


def build_balanced(chunks, lo, hi):
    if lo > hi:
        return None
    if lo == hi:
        return Node(text=chunks[lo])
    mid = (lo + hi) // 2
    return Node(left=build_balanced(chunks, lo, mid), right=build_balanced(chunks, mid + 1, hi))


def height_bound(leaf_count):
    return 2 * math.ceil(math.log2(leaf_count + 1)) + 1


def raw_concat(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return Node(left=left, right=right)


def split_root(node, index):
    if node is None:
        return None, None
    if node.text is not None:
        head, tail = node.text[:index], node.text[index:]
        return (Node(text=head) if head else None), (Node(text=tail) if tail else None)
    left_size = node.left.size
    if index < left_size:
        first, second = split_root(node.left, index)
        return first, raw_concat(second, node.right)
    if index == left_size:
        return node.left, node.right
    first, second = split_root(node.right, index - left_size)
    return raw_concat(node.left, first), second


class Rope:
    def __init__(self, text=""):
        chunks = leaves_of(text)
        self.root = build_balanced(chunks, 0, len(chunks) - 1)
        self.rebuilds = 0

    @classmethod
    def _from_root(cls, root, rebuilds):
        rope = cls.__new__(cls)
        rope.root = root
        rope.rebuilds = rebuilds
        return rope

    def size(self):
        return self.root.size if self.root is not None else 0

    def height(self):
        return self.root.height if self.root is not None else 0

    def leaf_count(self):
        return self.root.leaves if self.root is not None else 0

    def to_string(self):
        parts = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.text is not None:
                parts.append(node.text)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return "".join(parts)

    def leaf_texts(self):
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.text is not None:
                out.append(node.text)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def validate(self):
        def walk(node):
            if node is None:
                return 0, 0, 0
            if node.text is not None:
                if len(node.text) > LEAF_LIMIT or len(node.text) == 0:
                    return -1, -1, -1
                return node.size, node.height, node.leaves
            left_size, left_height, left_leaves = walk(node.left)
            right_size, right_height, right_leaves = walk(node.right)
            if left_size < 0 or right_size < 0:
                return -1, -1, -1
            if node.size != left_size + right_size:
                return -1, -1, -1
            if node.height != 1 + max(left_height, right_height):
                return -1, -1, -1
            if node.leaves != left_leaves + right_leaves:
                return -1, -1, -1
            return node.size, node.height, node.leaves

        size, height, leaves = walk(self.root)
        return size == self.size() and height == self.height() and leaves == self.leaf_count()

    def _balanced(self, root):
        if root is None or root.height <= height_bound(root.leaves):
            return root, 0
        chunks = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.text is not None:
                chunks.append(node.text)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return build_balanced(chunks, 0, len(chunks) - 1), 1

    def concat(self, other):
        if self.root is None:
            return Rope._from_root(other.root, self.rebuilds + other.rebuilds)
        if other.root is None:
            return Rope._from_root(self.root, self.rebuilds + other.rebuilds)
        merged = raw_concat(self.root, other.root)
        root, rebuilt = self._balanced(merged)
        return Rope._from_root(root, self.rebuilds + other.rebuilds + rebuilt)

    def split(self, index):
        if not 0 <= index <= self.size():
            raise IndexError("split index out of bounds")
        first, second = split_root(self.root, index)
        first, first_rebuilt = self._balanced(first)
        second, second_rebuilt = self._balanced(second)
        return (Rope._from_root(first, self.rebuilds + first_rebuilt),
                Rope._from_root(second, self.rebuilds + second_rebuilt))


def main():
    rope = Rope("hello world")
    assert rope.to_string() == "hello world"
    assert rope.size() == 11
    assert rope.leaf_count() == 1
    assert rope.height() == 1
    assert rope.validate()

    left, right = rope.split(0)
    assert left.to_string() == "" and left.size() == 0
    assert right.to_string() == "hello world"
    left, right = rope.split(11)
    assert left.to_string() == "hello world"
    assert right.to_string() == "" and right.size() == 0
    left, right = rope.split(5)
    assert left.to_string() == "hello" and right.to_string() == " world"
    assert left.concat(right).to_string() == "hello world"
    assert left.height() <= height_bound(left.leaf_count())

    data = "".join(chr(ord("a") + (i * 7 + i * i) % 26) for i in range(2000))
    rope = Rope(data)
    assert rope.to_string() == data
    assert rope.size() == 2000
    assert rope.leaf_count() == math.ceil(2000 / LEAF_LIMIT)
    assert rope.height() <= height_bound(rope.leaf_count())
    assert rope.validate()
    assert max(len(chunk) for chunk in rope.leaf_texts()) <= LEAF_LIMIT
    left, right = rope.split(999)
    assert left.to_string() + right.to_string() == data
    assert left.concat(right).to_string() == data

    rope = Rope()
    for index in range(200):
        rope = rope.concat(Rope(chr(ord("a") + index % 26)))
    assert rope.size() == 200
    assert rope.rebuilds >= 1, rope.rebuilds
    build_rebuilds = rope.rebuilds
    assert rope.to_string() == "".join(chr(ord("a") + i % 26) for i in range(200))
    assert rope.height() <= height_bound(rope.leaf_count())
    assert rope.validate()

    text = "".join(chr(ord("a") + (i * 13 + 5) % 26) for i in range(64))
    rope = Rope(text)
    oracle = text
    seed = 4360
    for step in range(100):
        seed = (seed * 75 + 74) % 65537
        op = seed % 4
        if op == 0:
            seed = (seed * 75 + 74) % 65537
            index = seed % (len(oracle) + 1)
            first, second = rope.split(index)
            assert first.to_string() == oracle[:index] and second.to_string() == oracle[index:], step
            if seed % 2 == 0:
                rope = first.concat(second)
            else:
                rope = second.concat(first)
                oracle = oracle[index:] + oracle[:index]
        elif op == 1:
            seed = (seed * 75 + 74) % 65537
            index = seed % (len(oracle) + 1)
            rope, _ = rope.split(index)
            oracle = oracle[:index]
        elif op == 2:
            seed = (seed * 75 + 74) % 65537
            index = seed % (len(oracle) + 1)
            _, rope = rope.split(index)
            oracle = oracle[index:]
        else:
            extra = "".join(chr(ord("a") + (step * 3 + i) % 26) for i in range(seed % 20))
            rope = rope.concat(Rope(extra))
            oracle = oracle + extra
        assert rope.to_string() == oracle, step
        assert rope.size() == len(oracle), step
        assert rope.height() <= height_bound(rope.leaf_count()), step
        assert rope.validate(), step
        assert "".join(rope.leaf_texts()) == oracle, step
    assert rope.rebuilds >= 0 and build_rebuilds >= 1
    left, right = rope.split(0)
    assert left.to_string() == "" and right.to_string() == oracle
    left, right = rope.split(rope.size())
    assert right.to_string() == "" and left.to_string() == oracle


if __name__ == "__main__":
    main()
    print("PASS")
