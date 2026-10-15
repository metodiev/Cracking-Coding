"""Task 054 - Traverse linked list and print nodes. (Tier: Beginner, CTCI Ch 2)"""
class ListNode:
    __slots__ = ("val", "next")

    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt


def from_list(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def traverse(head):
    """Return node values in order (the testable form of 'print each node')."""
    return to_list(head)


def main():
    head = from_list([1, 2, 3])
    assert traverse(head) == [1, 2, 3]
    assert traverse(None) == []


if __name__ == "__main__":
    main()
    print("PASS")
