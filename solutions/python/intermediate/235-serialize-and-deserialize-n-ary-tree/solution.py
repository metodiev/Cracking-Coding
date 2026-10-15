"""Task 235 - Serialize and deserialize N-ary tree. (Tier: Intermediate, CTCI Ch 4)

Wire format (identical across the Python, Java, JavaScript and C# solutions): a
recursive preorder where every node is written as "value[childCount]" followed by
the serialized children, e.g. root 1 with children [3, 2, 4] where 3 has children
[5, 6] serializes to "1[3]3[2]5[0]6[0]2[0]4[0]"; the empty tree serializes to "".

Serialization and parsing are both O(n) over the node count (one left-to-right scan).
"""


class NAryNode:
    __slots__ = ("val", "children")

    def __init__(self, val, children=None):
        self.val = val
        self.children = list(children) if children else []


def serialize(root):
    if root is None:
        return ""
    parts = [str(root.val), "[", str(len(root.children)), "]"]
    for child in root.children:
        parts.append(serialize(child))
    return "".join(parts)


def deserialize(data):
    if not data:
        return None
    pos = 0

    def read_node():
        nonlocal pos
        start = pos
        while data[pos] != "[":
            pos += 1
        val = int(data[start:pos])
        pos += 1                                   # skip '['
        start = pos
        while data[pos] != "]":
            pos += 1
        child_count = int(data[start:pos])
        pos += 1                                   # skip ']'
        node = NAryNode(val)
        for _ in range(child_count):
            node.children.append(read_node())
        return node

    return read_node()


def depth(node):
    return 0 if node is None else 1 + max((depth(c) for c in node.children), default=0)


def trees_equal(a, b):
    if a is None or b is None:
        return a is b
    return a.val == b.val and len(a.children) == len(b.children) and \
        all(trees_equal(x, y) for x, y in zip(a.children, b.children))


def main():
    # exact wire format for the documented example
    example = NAryNode(1)
    three, two, four = NAryNode(3), NAryNode(2), NAryNode(4)
    three.children = [NAryNode(5), NAryNode(6)]
    example.children = [three, two, four]
    assert serialize(example) == "1[3]3[2]5[0]6[0]2[0]4[0]"
    assert trees_equal(deserialize("1[3]3[2]5[0]6[0]2[0]4[0]"), example)
    assert serialize(deserialize(serialize(example))) == serialize(example)

    # empty tree
    assert serialize(None) == ""
    assert deserialize("") is None

    # single node
    assert serialize(NAryNode(42)) == "42[0]"
    assert trees_equal(deserialize("42[0]"), NAryNode(42))

    # deep chain of 50 nodes
    chain = NAryNode(50)
    for v in range(49, 0, -1):
        chain = NAryNode(v, [chain])
    assert depth(chain) == 50
    assert serialize(chain).startswith("1[1]2[1]3[1]")
    assert trees_equal(deserialize(serialize(chain)), chain)
    assert depth(deserialize(serialize(chain))) == 50

    # node with many children
    many = NAryNode(1, [NAryNode(v) for v in range(10)])
    expected = "1[10]" + "".join(f"{v}[0]" for v in range(10))
    assert serialize(many) == expected
    restored_many = deserialize(serialize(many))
    assert trees_equal(restored_many, many)
    assert len(restored_many.children) == 10
    assert [c.val for c in restored_many.children] == list(range(10))

    # negative values are handled by the scanner
    assert serialize(NAryNode(-5, [NAryNode(-7)])) == "-5[1]-7[0]"
    assert trees_equal(deserialize("-5[1]-7[0]"), NAryNode(-5, [NAryNode(-7)]))


if __name__ == "__main__":
    main()
    print("PASS")
