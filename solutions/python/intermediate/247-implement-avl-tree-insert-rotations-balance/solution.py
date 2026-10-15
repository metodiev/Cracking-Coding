"""Task 247 - Implement AVL tree insert (rotations + balance). (Tier: Intermediate, CTCI Ch 4)

Every insert walks back up updating heights; the first node whose balance factor leaves
[-1, 1] is rebalanced with one LL/RR rotation or one LR/RL double rotation.
"""

class AVLNode:
    __slots__ = ("key", "left", "right", "height")

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


def height(node):
    return node.height if node else 0


def update(node):
    node.height = 1 + max(height(node.left), height(node.right))


def balance_factor(node):
    return height(node.left) - height(node.right)


def rotate_right(y):
    x = y.left
    y.left = x.right
    x.right = y
    update(y)
    update(x)
    return x


def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    update(x)
    update(y)
    return y


def insert(node, key):
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        return node                       # duplicates ignored

    update(node)
    bf = balance_factor(node)
    if bf > 1 and key < node.left.key:                       # left-left
        return rotate_right(node)
    if bf < -1 and key > node.right.key:                     # right-right
        return rotate_left(node)
    if bf > 1 and key > node.left.key:                       # left-right
        node.left = rotate_left(node.left)
        return rotate_right(node)
    if bf < -1 and key < node.right.key:                     # right-left
        node.right = rotate_right(node.right)
        return rotate_left(node)
    return node


def build(keys):
    root = None
    for key in keys:
        root = insert(root, key)
    return root


def inorder(node, out=None):
    out = [] if out is None else out
    if node:
        inorder(node.left, out)
        out.append(node.key)
        inorder(node.right, out)
    return out


def is_balanced(node):
    if node is None:
        return True
    if abs(balance_factor(node)) > 1:
        return False
    return is_balanced(node.left) and is_balanced(node.right)


def count(node):
    return 0 if node is None else 1 + count(node.left) + count(node.right)


def main():
    # LL: 10-20-30 -> left rotation about 10
    root = build([10, 20, 30])
    assert (root.key, root.left.key, root.right.key) == (20, 10, 30)
    # RR: 30-20-10 -> right rotation about 30
    root = build([30, 20, 10])
    assert (root.key, root.left.key, root.right.key) == (20, 10, 30)
    # LR: 30-10-20 -> rotate left about 10, then right about 30
    root = build([30, 10, 20])
    assert (root.key, root.left.key, root.right.key) == (20, 10, 30)
    # RL: 10-30-20 -> rotate right about 30, then left about 10
    root = build([10, 30, 20])
    assert (root.key, root.left.key, root.right.key) == (20, 10, 30)

    asc = build(range(1, 8))
    assert inorder(asc) == list(range(1, 8))
    assert is_balanced(asc) and asc.height == 3              # perfect tree of 7 nodes

    ten = build(range(1, 11))
    assert count(ten) == 10 and is_balanced(ten)
    assert ten.height == 4                                   # 3 levels hold at most 7 nodes

    assert count(build([5, 5, 5])) == 1                      # duplicates ignored
    assert inorder(build([50, 20, 60, 10, 30, 55, 70])) == [10, 20, 30, 50, 55, 60, 70]


if __name__ == "__main__":
    main()
    print("PASS")
