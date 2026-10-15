"""Task 061 - Reverse singly linked list (iterative). (Tier: Beginner, CTCI Ch 2)"""

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


def reverse(head):
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev


def main():
    assert to_list(reverse(from_list([1, 2, 3]))) == [3, 2, 1]
    assert to_list(reverse(from_list([1]))) == [1]
    assert to_list(reverse(None)) == []


if __name__ == "__main__":
    main()
    print("PASS")
