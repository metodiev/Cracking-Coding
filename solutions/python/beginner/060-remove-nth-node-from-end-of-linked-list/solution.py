"""Task 060 - Remove nth node from end of linked list. (Tier: Beginner, CTCI Ch 2)"""
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

def remove_nth_from_end(head, n):
    """Two pointers with an n+1 gap, plus a dummy to simplify head removal."""
    dummy = ListNode(0, head)
    slow = fast = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next


def main():
    assert to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
    assert to_list(remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 5)) == [2, 3, 4, 5]  # head removed
    assert to_list(remove_nth_from_end(from_list([1]), 1)) == []


if __name__ == "__main__":
    main()
    print("PASS")
