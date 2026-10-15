"""Task 154 - Implement a basic circular linked list. (Tier: Beginner, CTCI Ch 2)

Singly linked circular list: the tail always points back at the head, so a
walk of exactly `size` steps visits every value once and then terminates.
to_list() performs that one-lap walk. append/insert/delete keep the ring
intact, including deleting the head and deleting the last node (which leaves
the list empty).
"""


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value):
        self.value = value
        self.next = None


class CircularLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, value):
        """Append at the tail; the new tail points back at the head."""
        node = _Node(value)
        if self.head is None:
            node.next = node
            self.head = node
        else:
            tail = self.tail()
            tail.next = node
            node.next = self.head
        self.size += 1

    def insert(self, index, value):
        """Insert before position `index` (0 <= index <= size); size appends."""
        if index < 0 or index > self.size:
            raise IndexError("insert index out of range")
        if index == self.size:
            self.append(value)
            return
        node = _Node(value)
        if index == 0:
            tail = self.tail()
            node.next = self.head
            tail.next = node
            self.head = node
        else:
            prev = self.head
            for _ in range(index - 1):
                prev = prev.next
            node.next = prev.next
            prev.next = node
        self.size += 1

    def delete(self, value):
        """Delete the first node holding `value`; returns True if one was removed."""
        if self.head is None:
            return False
        if self.head.value == value:
            if self.size == 1:
                self.head = None
            else:
                tail = self.tail()
                self.head = self.head.next
                tail.next = self.head
            self.size -= 1
            return True
        prev = self.head
        while prev.next is not self.head and prev.next.value != value:
            prev = prev.next
        if prev.next is self.head:
            return False
        prev.next = prev.next.next
        self.size -= 1
        return True

    def tail(self):
        """The node whose next is the head (None for an empty list)."""
        if self.head is None:
            return None
        node = self.head
        while node.next is not self.head:
            node = node.next
        return node

    def to_list(self):
        """One lap: exactly `size` steps starting at the head (terminates)."""
        out = []
        node = self.head
        for _ in range(self.size):
            out.append(node.value)
            node = node.next
        return out


def main():
    lst = CircularLinkedList()
    assert lst.to_list() == [] and lst.head is None and lst.size == 0

    for v in (1, 2, 3):
        lst.append(v)
    assert lst.to_list() == [1, 2, 3] and lst.size == 3
    # wrap-around: the tail and the third node point back at the head
    assert lst.tail().next is lst.head
    assert lst.head.next.next.next is lst.head

    # insert at head / middle / (index == size) tail
    lst.insert(0, 0)
    assert lst.to_list() == [0, 1, 2, 3] and lst.head.value == 0
    lst.insert(2, 99)
    assert lst.to_list() == [0, 1, 99, 2, 3]
    lst.insert(lst.size, 4)
    assert lst.to_list() == [0, 1, 99, 2, 3, 4]
    assert lst.tail().value == 4 and lst.tail().next is lst.head

    try:
        lst.insert(99, 0)
        raise AssertionError("expected IndexError")
    except IndexError:
        pass

    # delete middle / head / tail
    assert lst.delete(99) and lst.to_list() == [0, 1, 2, 3, 4]
    assert lst.delete(0) and lst.head.value == 1
    assert lst.to_list() == [1, 2, 3, 4]
    assert lst.tail().next is lst.head                       # ring still closed
    assert lst.delete(4) and lst.to_list() == [1, 2, 3]
    assert not lst.delete(42) and lst.to_list() == [1, 2, 3]  # missing value

    # drain to empty, then re-append
    assert lst.delete(1) and lst.delete(2) and lst.delete(3)
    assert lst.to_list() == [] and lst.size == 0 and lst.head is None
    assert not lst.delete(1) and lst.tail() is None
    lst.append(7)
    assert lst.to_list() == [7] and lst.head.next is lst.head
    assert lst.delete(7) and lst.to_list() == [] and lst.head is None


if __name__ == "__main__":
    main()
    print("PASS")
