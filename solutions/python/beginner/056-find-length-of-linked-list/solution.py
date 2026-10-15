"""Task 056 - Find length of linked list. (Tier: Beginner, CTCI Ch 2)"""
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

def length(head):
    count = 0
    while head:
        count += 1
        head = head.next
    return count


def main():
    assert length(from_list([1, 2, 3, 4])) == 4
    assert length(None) == 0
    assert length(from_list([7])) == 1


if __name__ == "__main__":
    main()
    print("PASS")
