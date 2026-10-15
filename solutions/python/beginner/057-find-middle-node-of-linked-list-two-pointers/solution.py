"""Task 057 - Find middle node of linked list (two pointers). (Tier: Beginner, CTCI Ch 2)"""
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

def middle(head):
    """Slow/fast pointers; for even length returns the second middle."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def main():
    assert middle(from_list([1, 2, 3, 4, 5])).val == 3
    assert middle(from_list([1, 2, 3, 4])).val == 3
    assert middle(from_list([9])).val == 9
    assert middle(None) is None


if __name__ == "__main__":
    main()
    print("PASS")
