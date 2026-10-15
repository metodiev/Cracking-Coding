"""Task 249 - Implement treap (BST + heap) insert/delete. (Tier: Intermediate, CTCI Ch 4)

A treap is a BST on the keys and a max-heap on random priorities: insert bubbles up with
rotations, delete rotates the higher-priority child up until the node becomes a leaf.
Priorities are supplied explicitly here so the whole run is deterministic.
"""

class TreapNode:
    __slots__ = ("key", "pri", "left", "right")

    def __init__(self, key, pri):
        self.key = key
        self.pri = pri
        self.left = None
        self.right = None


def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    return x


def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    return y


def insert(node, key, pri):
    """Returns (new_root, inserted)."""
    if node is None:
        return TreapNode(key, pri), True
    if key == node.key:
        return node, False
    if key < node.key:
        node.left, inserted = insert(node.left, key, pri)
        if node.left.pri > node.pri:
            node = rotate_right(node)
    else:
        node.right, inserted = insert(node.right, key, pri)
        if node.right.pri > node.pri:
            node = rotate_left(node)
    return node, inserted


def erase(node, key):
    """Returns (new_root, removed)."""
    if node is None:
        return None, False
    if key < node.key:
        node.left, removed = erase(node.left, key)
    elif key > node.key:
        node.right, removed = erase(node.right, key)
    else:
        if node.left is None:
            return node.right, True
        if node.right is None:
            return node.left, True
        if node.left.pri > node.right.pri:
            node = rotate_right(node)
            node.right, removed = erase(node.right, key)
        else:
            node = rotate_left(node)
            node.left, removed = erase(node.left, key)
    return node, removed


def build(entries):
    """entries: list of (key, priority) inserted in order."""
    root = None
    for key, pri in entries:
        root, _ = insert(root, key, pri)
    return root


def inorder(node, out=None):
    out = [] if out is None else out
    if node:
        inorder(node.left, out)
        out.append(node.key)
        inorder(node.right, out)
    return out


def preorder(node, out=None):
    out = [] if out is None else out
    if node:
        out.append(node.key)
        preorder(node.left, out)
        preorder(node.right, out)
    return out


def is_heap(node):
    """The treap uses a max-heap on priorities."""
    if node is None:
        return True
    for child in (node.left, node.right):
        if child is not None and child.pri > node.pri:
            return False
    return is_heap(node.left) and is_heap(node.right)


def contains(node, key):
    while node is not None:
        if key == node.key:
            return True
        node = node.left if key < node.key else node.right
    return False


KEYS = [50, 30, 70, 20, 40, 60, 80]
PRIS = [40, 25, 60, 15, 35, 50, 70]


def main():
    root = build(zip(KEYS, PRIS))
    assert root.key == 80 and root.pri == 70                 # highest priority ends at the root
    assert preorder(root) == [80, 70, 60, 50, 40, 30, 20]
    assert inorder(root) == sorted(KEYS)
    assert is_heap(root)

    duplicates = [1, 1]
    _, inserted = insert(build([(1, 5)]), 1, 9)
    assert inserted is False

    root, removed = erase(root, 30)
    assert removed and inorder(root) == [20, 40, 50, 60, 70, 80] and is_heap(root)
    root, removed = erase(root, 80)                          # the root case
    assert removed and root.key == 70 and is_heap(root)
    root, removed = erase(root, 999)
    assert removed is False and inorder(root) == [20, 40, 50, 60, 70]
    assert all(contains(root, key) for key in [20, 40, 50, 60, 70])
    assert not contains(root, 30)

    empty, removed = erase(None, 1)
    assert empty is None and removed is False
    assert inorder(build([])) == []


if __name__ == "__main__":
    main()
    print("PASS")
