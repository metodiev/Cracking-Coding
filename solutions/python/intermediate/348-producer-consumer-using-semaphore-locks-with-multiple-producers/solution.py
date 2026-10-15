"""Task 348 - Multi-producer/multi-consumer with a counting semaphore. (Tier: Intermediate, CTCI Ch 15)

Bounded buffer of capacity 7 protected by a semaphore used as a mutex; `empty`
counts free slots and `full` counts queued items (the classic counting-semaphore
formulation).  Two producers fill disjoint halves of 0..999 and two consumers
drain.  Once the producers have finished, the main thread acquires slot permits
and appends one -1 sentinel per consumer, so the consumers stop deterministically
without any timing guess.  Assertions: every item is consumed exactly once,
each producer's own items keep their order, capacity is never exceeded, and all
threads terminate (daemons + bounded joins: a failure cannot hang the process).
"""
import threading
from collections import deque

CAPACITY = 7
ITEMS = 1000
SENTINEL = -1


def main():
    empty = threading.Semaphore(CAPACITY)
    full = threading.Semaphore(0)
    mutex = threading.Semaphore(1)
    buffer = deque()
    consumed = []                        # (global consumption index, item)
    max_seen = [0]

    def produce(first, last):
        for item in range(first, last):
            empty.acquire()
            mutex.acquire()
            try:
                buffer.append(item)
                max_seen[0] = max(max_seen[0], len(buffer))
            finally:
                mutex.release()
            full.release()

    def consume():
        while True:
            full.acquire()
            mutex.acquire()
            popped = buffer.popleft()
            if popped == SENTINEL:
                mutex.release()
                break                    # the sentinel's slot permit stays consumed
            consumed.append((len(consumed), popped))
            mutex.release()
            empty.release()

    producers = [threading.Thread(target=produce, args=(0, ITEMS // 2), daemon=True),
                 threading.Thread(target=produce, args=(ITEMS // 2, ITEMS), daemon=True)]
    consumers = [threading.Thread(target=consume, daemon=True) for _ in range(2)]
    for thread in consumers + producers:
        thread.start()
    for thread in producers:
        thread.join(15)
        assert not thread.is_alive(), "producer did not finish"

    # producers are done: reserve one slot per sentinel so no consumer can miss it
    for _ in consumers:
        empty.acquire()
        mutex.acquire()
        buffer.append(SENTINEL)
        mutex.release()
        full.release()
    for thread in consumers:
        thread.join(15)
        assert not thread.is_alive(), "consumer did not finish"

    assert len(consumed) == ITEMS, "no item lost or duplicated"
    ordered = [item for _, item in consumed]
    assert sorted(ordered) == list(range(ITEMS)), "every item consumed exactly once"
    assert max_seen[0] <= CAPACITY, "capacity never exceeded"
    assert max_seen[0] >= 1, "buffer was actually used"
    # per-producer order preserved: each half is drained in increasing order
    assert [item for item in ordered if item < ITEMS // 2] == list(range(ITEMS // 2))
    assert [item for item in ordered if item >= ITEMS // 2] == list(range(ITEMS // 2, ITEMS))


if __name__ == "__main__":
    main()
    print("PASS")
