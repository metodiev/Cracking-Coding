"""Task 055 - Delete node by value in singly linked list. (Tier: Beginner, CTCI Ch 2)"""
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

def delete_value(head, value):
    """Deletes the first node with the value; returns (possibly new) head."""
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next:
        if prev.next.val == value:
            prev.next = prev.next.next
            break
        prev = prev.next
    return dummy.next


def main():
    assert to_list(delete_value(from_list([1, 2, 3]), 2)) == [1, 3]
    assert to_list(delete_value(from_list([1, 2, 3]), 1)) == [2, 3]   # head
    assert to_list(delete_value(from_list([1]), 1)) == []
    assert to_list(delete_value(from_list([1, 2]), 9)) == [1, 2]      # missing


if __name__ == "__main__":
    main()
    print("PASS")
