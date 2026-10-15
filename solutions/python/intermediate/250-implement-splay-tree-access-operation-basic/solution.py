"""Task 250 - Implement splay tree access operation (basic). (Tier: Intermediate, CTCI Ch 4)

Every access rotates the touched node to the root: zig (parent is root), zig-zig (node and
parent on the same side) or zig-zag. A failed find() does not splay anything (documented).
"""

class SplayNode:
    __slots__ = ("key", "left", "right", "parent")

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class SplayTree:
    def __init__(self):
        self.root = None

    # ---------------- rotations ----------------
    def _replace_child(self, parent, old, new):
        if parent is None:
            self.root = new
        elif parent.left is old:
            parent.left = new
        else:
            parent.right = new
        if new is not None:
            new.parent = parent

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        parent = x.parent
        self._replace_child(parent, x, y)
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        parent = x.parent
        self._replace_child(parent, x, y)
        y.right = x
        x.parent = y

    # ---------------- splay ----------------
    def _splay(self, x):
        while x.parent is not None:
            parent = x.parent
            grand = parent.parent
            if grand is None:                                  # zig
                if x is parent.left:
                    self._rotate_right(parent)
                else:
                    self._rotate_left(parent)
            elif x is parent.left and parent is grand.left:    # zig-zig
                self._rotate_right(grand)
                self._rotate_right(parent)
            elif x is parent.right and parent is grand.right:  # zig-zig (mirror)
                self._rotate_left(grand)
                self._rotate_left(parent)
            elif x is parent.right and parent is grand.left:   # zig-zag
                self._rotate_left(parent)
                self._rotate_right(grand)
            else:                                              # zig-zag (mirror)
                self._rotate_right(parent)
                self._rotate_left(grand)
        self.root = x

    # ---------------- public API ----------------
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
                self._splay(node)
                return False
        fresh = SplayNode(key)
        fresh.parent = parent
        if parent is None:
            self.root = fresh
        elif key < parent.key:
            parent.left = fresh
        else:
            parent.right = fresh
        self._splay(fresh)
        return True

    def access(self, key):
        """Splays key to the root if it exists; returns whether it was found."""
        node = self.root
        while node is not None:
            if key == node.key:
                self._splay(node)
                return True
            node = node.left if key < node.key else node.right
        return False

    def inorder(self, out=None):
        out = [] if out is None else out

        def walk(node):
            if node:
                walk(node.left)
                out.append(node.key)
                walk(node.right)

        walk(self.root)
        return out

    def is_bst(self):
        """The inorder walk must be sorted, and every parent link must be consistent."""
        keys = self.inorder()
        if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
            return False

        def check(node):
            if node is None:
                return True
            for child in (node.left, node.right):
                if child is not None and child.parent is not node:
                    return False
            return check(node.left) and check(node.right)

        return check(self.root)


def main():
    tree = SplayTree()
    for key in [50, 30, 70, 20, 40, 60, 80]:
        assert tree.insert(key)
    assert tree.root.key == 80                       # last inserted is splayed to the root
    assert tree.inorder() == [20, 30, 40, 50, 60, 70, 80]
    assert tree.is_bst()

    assert tree.access(20) and tree.root.key == 20   # zig-zig
    assert tree.inorder() == [20, 30, 40, 50, 60, 70, 80]
    assert tree.is_bst()

    assert tree.access(80) and tree.root.key == 80
    assert tree.is_bst()

    assert tree.access(50) and tree.root.key == 50   # zig-zag territory after the two splays
    assert tree.inorder() == [20, 30, 40, 50, 60, 70, 80]
    assert tree.is_bst()

    assert tree.access(999) is False and tree.root.key == 50   # failed access does not splay
    assert tree.insert(50) is False and tree.root.key == 50    # duplicate is splayed, not inserted
    assert tree.inorder() == [20, 30, 40, 50, 60, 70, 80]

    single = SplayTree()
    assert single.insert(7)
    assert single.access(7) and single.root.key == 7 and single.root.parent is None


if __name__ == "__main__":
    main()
    print("PASS")
