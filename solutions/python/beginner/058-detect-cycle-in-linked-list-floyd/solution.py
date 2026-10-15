"""Task 058 - Detect cycle in linked list (Floyd). (Tier: Beginner, CTCI Ch 2)"""
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

def has_cycle(head):
    """Tortoise & hare: they meet iff a cycle exists."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def main():
    a = from_list([1, 2, 3, 4])
    assert not has_cycle(a)
    a.next.next.next.next = a.next          # 4 -> 2 creates a cycle
    assert has_cycle(a)
    assert not has_cycle(None)


if __name__ == "__main__":
    main()
    print("PASS")
