"""Task 059 - Merge two sorted linked lists. (Tier: Beginner, CTCI Ch 2)"""
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

def merge_sorted(a, b):
    dummy = tail = ListNode(0)
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next


def main():
    assert to_list(merge_sorted(from_list([1, 3, 5]), from_list([2, 4, 6]))) == [1, 2, 3, 4, 5, 6]
    assert to_list(merge_sorted(None, from_list([1]))) == [1]
    assert to_list(merge_sorted(None, None)) == []


if __name__ == "__main__":
    main()
    print("PASS")
