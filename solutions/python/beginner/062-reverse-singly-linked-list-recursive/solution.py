"""Task 062 - Reverse singly linked list (recursive). (Tier: Beginner, CTCI Ch 2)"""

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


def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


def main():
    assert to_list(reverse_recursive(from_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
    assert to_list(reverse_recursive(from_list([9]))) == [9]
    assert to_list(reverse_recursive(None)) == []


if __name__ == "__main__":
    main()
    print("PASS")
