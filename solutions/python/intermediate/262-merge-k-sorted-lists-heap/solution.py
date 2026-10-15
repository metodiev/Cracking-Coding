"""Task 262 - Merge K sorted lists (heap). (Tier: Intermediate, CTCI Ch 10)

Merge k sorted singly linked lists with a min-heap over the current heads: pop the
smallest head, append it to the output tail and push its successor. Each of the n nodes
enters the heap at most once, so O(n log k) time and O(k) heap space; nodes are reused,
no copies are made. k = 0 (or only empty lists) -> None; k = 1 returns the SAME head
object. Heap entries are (value, list index, node); the list index only breaks value
ties, so equal values are consumed in list order and the merge is deterministic.

merge_oracle flattens + sorts (O(n log n)) and is used only for cross-checks; it must
run before a merge, because merging rewires the list tails.
"""

import heapq


class ListNode:
    __slots__ = ("val", "next")

    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def build_list(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head


def to_array(head):
    out = []
    while head is not None:
        out.append(head.val)
        head = head.next
    return out


def merge_k_lists(lists):
    heap = []
    for idx, node in enumerate(lists):
        if node is not None:
            heap.append((node.val, idx, node))
    heapq.heapify(heap)
    dummy = ListNode(0)
    tail = dummy
    while heap:
        _, idx, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, idx, node.next))
    tail.next = None
    return dummy.next


def merge_oracle(lists):
    """Flat sort reference used only for cross-checks."""
    values = sorted(v for node in lists for v in to_array(node))
    return build_list(values)


def main():
    merged = merge_k_lists([build_list([1, 4, 5]), build_list([1, 3, 4]), build_list([2, 6])])
    assert to_array(merged) == [1, 1, 2, 3, 4, 4, 5, 6]

    assert merge_k_lists([]) is None                       # k = 0 -> empty
    assert merge_k_lists([None, None]) is None             # only empty lists

    single = build_list([1, 2, 3])
    same = merge_k_lists([single])
    assert same is single                                  # k = 1 returns the same list
    assert to_array(same) == [1, 2, 3]

    assert to_array(merge_k_lists([build_list([1]), build_list([2, 3, 4, 5]), build_list([6])])) == [1, 2, 3, 4, 5, 6]
    assert to_array(merge_k_lists([build_list([1, 1]), build_list([1, 1])])) == [1, 1, 1, 1]     # duplicates
    assert to_array(merge_k_lists([None, build_list([1]), None, build_list([])])) == [1]         # empty lists mixed in
    assert to_array(merge_k_lists([build_list([-3, -1]), build_list([-2, 0])])) == [-3, -2, -1, 0]
    assert to_array(merge_k_lists([build_list([5, 5]), build_list([5]), build_list([])])) == [5, 5, 5]

    import random

    rng = random.Random(262)
    for _ in range(300):                            # cross-check vs the flat-sort oracle
        k = rng.randint(0, 5)
        lists = []
        for _ in range(k):
            values = sorted(rng.randint(-6, 6) for _ in range(rng.randint(0, 5)))
            lists.append(build_list(values))
        expected = to_array(merge_oracle(lists))    # oracle first: merging rewires tails
        assert to_array(merge_k_lists(lists)) == expected


if __name__ == "__main__":
    main()
    print("PASS")
