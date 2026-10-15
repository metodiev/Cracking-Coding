"""Task 242 - Convert binary tree to doubly linked list (inorder). (Tier: Intermediate, CTCI Ch 4)

In-place: left becomes prev, right becomes next. Returns the list head.
"""

class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    """Level-order builder: None marks a missing child."""
    from collections import deque

    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    q = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            q.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            q.append(node.right)
        i += 1
    return root


def preorder(node, out=None):
    out = [] if out is None else out
    if node:
        out.append(node.val)
        preorder(node.left, out)
        preorder(node.right, out)
    return out


def inorder(node, out=None):
    out = [] if out is None else out
    if node:
        inorder(node.left, out)
        out.append(node.val)
        inorder(node.right, out)
    return out


def to_doubly_linked_list(root):
    """Returns the head of the inorder doubly linked list (or None)."""
    last = [None]
    head = [None]

    def visit(node):
        if node is None:
            return
        visit(node.left)
        node.left = last[0]                 # prev
        if last[0] is not None:
            last[0].right = node            # next of the predecessor
        else:
            head[0] = node                  # first visited node is the head
        last[0] = node
        visit(node.right)

    visit(root)
    return head[0]


def to_values(head):
    out, node = [], head
    while node:
        out.append(node.val)
        node = node.right
    return out


def to_values_reverse(head):
    node = head
    if node is None:
        return []
    while node.right:
        node = node.right
    out = []
    while node:
        out.append(node.val)
        node = node.left
    return out


def main():
    head = to_doubly_linked_list(build_tree([4, 2, 5, 1, 3]))
    assert to_values(head) == [1, 2, 3, 4, 5]
    assert to_values_reverse(head) == [5, 4, 3, 2, 1]        # prev links intact
    assert to_doubly_linked_list(None) is None
    single = to_doubly_linked_list(build_tree([7]))
    assert to_values(single) == [7] and single.left is None and single.right is None


if __name__ == "__main__":
    main()
    print("PASS")
