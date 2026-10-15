"""Task 195 - Implement basic producer-consumer using threads (mutex/condition). (Tier: Beginner, CTCI Ch 15)

A bounded buffer guarded by a mutex plus condition variables (threading.Condition):
producers block while the buffer is full and consumers block while it is empty,
capacity being smaller than the item count. Two producers fill disjoint halves of
0..ITEMS-1 and two consumers drain the buffer; the self-test proves no item is lost
or duplicated, the capacity is never exceeded, and every thread terminates.
"""
import threading
from collections import deque

CAPACITY = 7
ITEMS = 1000


class BoundedBuffer:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.max_seen = 0
        self._items = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

    def put(self, item):
        with self._not_full:
            while not self._closed and len(self._items) >= self.capacity:
                self._not_full.wait()
            if self._closed:
                return
            self._items.append(item)
            self.max_seen = max(self.max_seen, len(self._items))
            self._not_empty.notify()

    def take(self):
        with self._not_empty:
            while not self._items and not self._closed:
                self._not_empty.wait()
            if not self._items:
                return None
            item = self._items.popleft()
            self._not_full.notify()
            return item

    def close(self):
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()


def main():
    buffer = BoundedBuffer(CAPACITY)
    consumed = {0: [], 1: []}

    def produce(start, end):
        for item in range(start, end):
            buffer.put(item)

    def consume(tag):
        while True:
            item = buffer.take()
            if item is None:
                break
            consumed[tag].append(item)

    consumers = [threading.Thread(target=consume, args=(tag,), daemon=True) for tag in (0, 1)]
    producers = [
        threading.Thread(target=produce, args=(0, ITEMS // 2), daemon=True),
        threading.Thread(target=produce, args=(ITEMS // 2, ITEMS), daemon=True),
    ]
    for thread in consumers + producers:
        thread.start()
    for thread in producers:
        thread.join(timeout=10)
        assert not thread.is_alive(), "producer did not finish in time"
    buffer.close()
    for thread in consumers:
        thread.join(timeout=10)
        assert not thread.is_alive(), "consumer did not finish in time"

    all_items = consumed[0] + consumed[1]
    assert len(all_items) == ITEMS, "no items lost or duplicated"
    assert sorted(all_items) == list(range(ITEMS)), "every item consumed exactly once"
    assert buffer.max_seen <= CAPACITY, "capacity never exceeded"
    assert buffer.max_seen >= 1, "buffer was actually used"


if __name__ == "__main__":
    main()
    print("PASS")
