"""Task 052 - Implement simple LinkedList insert at head. (Tier: Beginner, CTCI Ch 2)"""
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

def insert_head(head, value):
    """Insert at head; returns the new head."""
    return ListNode(value, head)


def main():
    head = None
    for v in (1, 2, 3):
        head = insert_head(head, v)     # 1, then 2, then 3 at head
    assert to_list(head) == [3, 2, 1]
    assert to_list(insert_head(None, 9)) == [9]


if __name__ == "__main__":
    main()
    print("PASS")
