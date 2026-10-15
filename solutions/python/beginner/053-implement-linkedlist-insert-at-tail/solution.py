"""Task 053 - Implement LinkedList insert at tail. (Tier: Beginner, CTCI Ch 2)"""
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

def insert_tail(head, value):
    """Append at tail in O(n); returns head."""
    node = ListNode(value)
    if head is None:
        return node
    p = head
    while p.next:
        p = p.next
    p.next = node
    return head


def main():
    head = None
    for v in (1, 2, 3):
        head = insert_tail(head, v)
    assert to_list(head) == [1, 2, 3]
    assert to_list(insert_tail(None, 7)) == [7]


if __name__ == "__main__":
    main()
    print("PASS")
