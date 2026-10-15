"""Task 248 - Implement red-black tree insert (basic). (Tier: Intermediate, CTCI Ch 4)

CLRS insert + fixup: recolor while the uncle is red, otherwise rotate (zig / zig-zig / zig-zag).
The checker verifies the four invariants: root is black, no red node has a red child, every
root-to-leaf path has the same number of black nodes, and the inorder walk is sorted.
"""

RED = True
BLACK = False


class RBNode:
    __slots__ = ("key", "red", "left", "right", "parent")

    def __init__(self, key, red=RED):
        self.key = key
        self.red = red
        self.left = None
        self.right = None
        self.parent = None


class RedBlackTree:
    def __init__(self):
        self.root = None

    # ---------------- rotations ----------------
    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    # ---------------- insert ----------------
    def insert(self, key):
        parent = None
        node = self.root
        while node is not None:
            parent = node
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return False                       # duplicates ignored

        z = RBNode(key)
        z.parent = parent
        if parent is None:
            self.root = z
        elif key < parent.key:
            parent.left = z
        else:
            parent.right = z
        self._fixup(z)
        return True

    def _fixup(self, z):
        while z.parent is not None and z.parent.red:
            parent = z.parent
            grand = parent.parent
            if parent is grand.left:
                uncle = grand.right
                if uncle is not None and uncle.red:
                    parent.red = BLACK
                    uncle.red = BLACK
                    grand.red = RED
                    z = grand
                else:
                    if z is parent.right:
                        z = parent
                        self._rotate_left(z)
                        parent, grand = z.parent, z.parent.parent
                    parent.red = BLACK
                    grand.red = RED
                    self._rotate_right(grand)
            else:
                uncle = grand.left
                if uncle is not None and uncle.red:
                    parent.red = BLACK
                    uncle.red = BLACK
                    grand.red = RED
                    z = grand
                else:
                    if z is parent.left:
                        z = parent
                        self._rotate_right(z)
                        parent, grand = z.parent, z.parent.parent
                    parent.red = BLACK
                    grand.red = RED
                    self._rotate_left(grand)
        self.root.red = BLACK


# ---------------- invariant checkers ----------------
def inorder(node, out=None):
    out = [] if out is None else out
    if node:
        inorder(node.left, out)
        out.append(node.key)
        inorder(node.right, out)
    return out


def no_red_red(node):
    if node is None:
        return True
    if node.red:
        for child in (node.left, node.right):
            if child is not None and child.red:
                return False
    return no_red_red(node.left) and no_red_red(node.right)


def black_heights(node, measure):
    """Set of black-node counts over all paths from `node` down to the Nones."""
    if node is None:
        return {measure}
    sub = black_heights(node.left, measure) | black_heights(node.right, measure)
    extra = 0 if node.red else 1
    return {value + extra for value in sub}


def main():
    tree = RedBlackTree()
    for key in [10, 20, 30]:                     # classic rotation case
        assert tree.insert(key)
    assert tree.root.key == 20 and not tree.root.red
    assert tree.root.left.key == 10 and tree.root.left.red
    assert tree.root.right.key == 30 and tree.root.right.red

    for order in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [7, 3, 9, 1, 5, 8, 10, 2, 4, 6],
                  [50, 20, 60, 10, 30, 55, 70, 5, 15, 25, 35]):
        tree = RedBlackTree()
        for key in order:
            assert tree.insert(key)
        assert inorder(tree.root) == sorted(order), "BST order"
        assert not tree.root.red, "root is black"
        assert no_red_red(tree.root), "no red-red violation"
        assert len(black_heights(tree.root, 0)) == 1, "uniform black height"

    assert tree.insert(50) is False              # duplicate rejected
    assert inorder(tree.root) == sorted([50, 20, 60, 10, 30, 55, 70, 5, 15, 25, 35])


if __name__ == "__main__":
    main()
    print("PASS")
