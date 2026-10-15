"""Task 063 - Implement doubly linked list insert/delete. (Tier: Beginner, CTCI Ch 2)"""


class DNode:
    __slots__ = ("val", "prev", "next")

    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, val):
        node = DNode(val)
        if self.tail is None:
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        return node

    def insert_before(self, node, val):
        new = DNode(val)
        new.prev, new.next = node.prev, node
        if node.prev:
            node.prev.next = new
        else:
            self.head = new
        node.prev = new
        return new

    def delete(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.prev = node.next = None

    def to_list(self):
        out, p = [], self.head
        while p:
            out.append(p.val)
            p = p.next
        return out

    def to_list_reversed(self):
        out, p = [], self.tail
        while p:
            out.append(p.val)
            p = p.prev
        return out


def main():
    d = DoublyLinkedList()
    a = d.append(1)
    b = d.append(2)
    c = d.append(3)
    assert d.to_list() == [1, 2, 3]
    assert d.to_list_reversed() == [3, 2, 1]

    d.insert_before(b, 9)                 # middle
    assert d.to_list() == [1, 9, 2, 3]
    d.insert_before(a, 0)                 # before head
    assert d.to_list() == [0, 1, 9, 2, 3]
    assert d.to_list_reversed() == [3, 2, 9, 1, 0]

    d.delete(a)
    assert d.to_list() == [0, 9, 2, 3]
    d.delete(c)                            # delete tail
    assert d.to_list() == [0, 9, 2]
    d.delete(b)                            # b holds value 2
    assert d.to_list() == [0, 9]
    assert d.to_list_reversed() == [9, 0]


if __name__ == "__main__":
    main()
    print("PASS")
