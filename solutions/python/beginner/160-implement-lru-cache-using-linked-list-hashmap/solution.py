"""Task 160 - Implement LRU cache using linked list + hashmap. (Tier: Beginner, CTCI Ch 7)

O(1) get/put with two pieces:
  * a hash map from key to node, and
  * a hand-rolled doubly linked list whose most recently used node sits at the
    front and whose least recently used node sits at the back.

Sentinel head/tail nodes remove every empty-list edge case. No built-in
ordered map (e.g. OrderedDict) is used - the list is implemented from scratch.
Both get() and put() refresh recency; put() on a full cache evicts the LRU
entry, and updating an existing key does not change the cache size.
"""


class _DNode:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._map = {}
        self._head = _DNode()      # most-recent sentinel
        self._tail = _DNode()      # least-recent sentinel
        self._head.next = self._tail
        self._tail.prev = self._head

    def _unlink(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node):
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def get(self, key):
        """Return the value, or -1 if missing; a hit marks the key most recently used."""
        node = self._map.get(key)
        if node is None:
            return -1
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key, value):
        """Insert or update; updates refresh recency; a full cache evicts the LRU entry."""
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            self._unlink(lru)
            del self._map[lru.key]
        node = _DNode(key, value)
        self._map[key] = node
        self._push_front(node)

    def __len__(self):
        return len(self._map)


def main():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)                      # evicts key 2
    assert cache.get(2) == -1
    cache.put(4, 4)                      # evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

    # updating an existing key refreshes recency and does not grow the cache
    c2 = LRUCache(2)
    c2.put(1, 1)
    c2.put(2, 2)
    c2.put(1, 10)                        # key 1 is now most recent
    assert len(c2) == 2
    c2.put(3, 3)                         # evicts key 2, NOT key 1
    assert c2.get(2) == -1
    assert c2.get(1) == 10
    assert c2.get(3) == 3

    # capacity 1 keeps only the newest
    c3 = LRUCache(1)
    c3.put(1, 1)
    c3.put(2, 2)
    assert c3.get(1) == -1 and c3.get(2) == 2

    # get() refreshes recency too
    c4 = LRUCache(2)
    c4.put(1, 1)
    c4.put(2, 2)
    assert c4.get(1) == 1                # key 2 becomes LRU
    c4.put(3, 3)                         # evicts key 2
    assert c4.get(2) == -1 and c4.get(1) == 1 and c4.get(3) == 3

    # a miss returns -1 and does not disturb the recency order
    c5 = LRUCache(2)
    c5.put(1, 1)
    c5.put(2, 2)
    assert c5.get(99) == -1
    c5.put(3, 3)                         # still evicts key 1
    assert c5.get(1) == -1 and c5.get(2) == 2 and c5.get(3) == 3

    try:
        LRUCache(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
