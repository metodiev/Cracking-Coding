"""Task 492 - Implement circular buffer with lock-free design (atomic). (Tier: Advanced, CTCI Ch 15)

Bounded MPMC ring buffer in the Vyukov style: capacity slots each carry a sequence number
(slot i starts at i), head and tail are plain integers advanced only by compare-and-swap, and
every operation retries when its CAS loses the race. A slot is published for enqueuing when
seq == pos and consumable when seq == pos + 1; the claimer overwrites the slot and then stores
seq = pos + 1 (enqueue) or seq = pos + capacity (dequeue), which re-publishes the slot one lap
later. One extra occupancy check (tail - head >= capacity => full) is required because for
capacity 1 the final lap's free and published sequence values coincide; for capacity >= 2 the
slot sequence alone decides. try_enqueue/try_dequeue never block: they report False (full) / None (empty), so a
caller spins outside. CPython offers no hardware CAS, so - exactly as in task 354 - the
compare-and-swap is emulated by a tiny lock around ONLY the compare+store pair (one indivisible
"instruction"), while the retry loops, the sequence protocol and the spin loops are the real
thing exercised by real threads. The tests assert: single-threaded wraparound/full/empty
semantics, exact global FIFO for one producer / one consumer across real threads, exactly-once
delivery plus per-producer FIFO for many producers / consumers, and a deterministic forced
CAS failure that exercises the retry path.
"""
import threading
import time

FULL = False
EMPTY = None


class LockFreeCircularBuffer:
    def __init__(self, capacity, cas_hook=None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.slots = [None] * capacity
        self.seq = list(range(capacity))       # slot i is initially "free at position i"
        self.head = 0                          # position of the next dequeue
        self.tail = 0                          # position of the next enqueue
        self.cas_attempts = 0
        self.cas_failures = 0
        self.enqueue_retries = 0
        self.dequeue_retries = 0
        self._cas_lock = threading.Lock()      # emulates ONE indivisible CAS instruction
        self._cas_hook = cas_hook              # fires once, immediately before a CAS

    def cas(self, holder, name, expected, new_value):
        with self._cas_lock:
            self.cas_attempts += 1
            if getattr(holder, name) == expected:
                setattr(holder, name, new_value)
                return True
            self.cas_failures += 1
            return False

    def _run_hook(self, pos):
        hook, self._cas_hook = self._cas_hook, None
        if hook is not None:
            hook(self, pos)

    def try_enqueue(self, value):
        """Returns True when value was enqueued, False when the queue is full."""
        while True:
            pos = self.tail
            if pos - self.head >= self.capacity:
                return FULL                    # occupancy guard (needed when capacity == 1,
                #                                where free/published seq values coincide)
            slot_index = pos % self.capacity
            slot_seq = self.seq[slot_index]
            if slot_seq == pos:                # the slot is ours to claim
                self._run_hook(pos)
                if self.cas(self, "tail", pos, pos + 1):
                    self.slots[slot_index] = value
                    self.seq[slot_index] = pos + 1     # publish (release store)
                    return True
                # CAS lost: another producer claimed pos, read the new tail and retry.
            elif slot_seq < pos:
                return FULL                    # the previous lap has not caught up: full
            self.enqueue_retries += 1

    def try_dequeue(self, on_claim=None):
        """Returns the oldest value, or EMPTY (None) when the queue is empty. on_claim, when
        given, receives the claimed position right after the head CAS - the self-test uses it
        as a linearization witness; it plays no part in the lock-free protocol."""
        while True:
            pos = self.head
            slot_index = pos % self.capacity
            slot_seq = self.seq[slot_index]
            if slot_seq == pos + 1:            # the slot holds a published value
                if self.cas(self, "head", pos, pos + 1):
                    if on_claim is not None:
                        on_claim(pos)
                    value = self.slots[slot_index]
                    self.seq[slot_index] = pos + self.capacity     # reopen one lap later
                    return value
                # CAS lost: another consumer took pos, read the new head and retry.
            elif slot_seq < pos + 1:
                return EMPTY                   # no value published yet: empty
            self.dequeue_retries += 1


def join_all(threads, timeout=30.0):
    for thread in threads:
        thread.join(timeout)
    assert all(not thread.is_alive() for thread in threads), "worker did not terminate"


def main():
    # Single-threaded semantics: wraparound, full, empty, capacity 1.
    buf = LockFreeCircularBuffer(4)
    for value in (1, 2, 3, 4):
        assert buf.try_enqueue(value) is True
    assert buf.try_enqueue(5) is False                          # full
    assert [buf.try_dequeue() for _ in range(4)] == [1, 2, 3, 4]
    assert buf.try_dequeue() is EMPTY                           # empty
    assert buf.try_enqueue(5) is True and buf.try_enqueue(6) is True
    assert [buf.try_dequeue() for _ in range(2)] == [5, 6]      # positions wrapped twice
    assert buf.try_dequeue() is EMPTY

    single = LockFreeCircularBuffer(1)
    assert single.try_enqueue("a") is True
    assert single.try_enqueue("b") is False
    assert single.try_dequeue() == "a"
    assert single.try_enqueue("b") is True
    assert single.try_dequeue() == "b"

    # Deterministic retry-path demonstration: the hook, running just before our CAS, lets a
    # "concurrent" producer claim the same position; our CAS must then fail once, reread the
    # tail and succeed on the next lap, while the hook's own item stays first in FIFO order.
    def steal_once(buffer, pos):
        assert buffer.try_enqueue("stealer") is True            # claims position `pos` fully
        assert buffer.head == 0 and buffer.tail == pos + 1

    forced = LockFreeCircularBuffer(8, cas_hook=steal_once)
    assert forced.try_enqueue("mine") is True
    assert forced.cas_failures == 1, "the stale CAS must fail exactly once"
    assert forced.enqueue_retries == 1, "exactly one retry lap"
    assert forced.try_dequeue() == "stealer" and forced.try_dequeue() == "mine"
    assert forced.try_dequeue() is EMPTY

    # SPSC under real threads: bounded capacity forces the producer to spin on FULL and the
    # consumer to spin on EMPTY, yet the dequeued stream must be exactly 0, 1, ..., total-1.
    # Spin failures call time.sleep(0) to hand the GIL to the partner thread promptly (a real
    # C/Java/C# spin would simply burn CPU); this is test-side scheduling only.
    total = 10000
    spsc = LockFreeCircularBuffer(8)
    received = []

    def producer():
        for value in range(total):
            while spsc.try_enqueue(value) is FULL:
                time.sleep(0)

    def consumer():
        while len(received) < total:
            value = spsc.try_dequeue()
            if value is not EMPTY:
                received.append(value)
            else:
                time.sleep(0)

    threads = [threading.Thread(target=producer, daemon=True),
               threading.Thread(target=consumer, daemon=True)]
    for thread in threads:
        thread.start()
    join_all(threads)
    assert received == list(range(total)), "SPSC FIFO violated"

    # MPMC accounting: 4 producers x 300 tagged items, 3 consumers, capacity 16. Consumers
    # record the claimed position from the head CAS (the linearization witness), so the merged
    # record must be the dequeue order: positions 0..total-1 exactly once, and each producer's
    # own items in the order that producer enqueued them (its operations are sequential, so
    # its subsequence of the queue order is FIFO). A small test-only lock guards the shared
    # consumed tally used only for termination.
    producers, per_producer, consumers = 4, 300, 3
    mpmc = LockFreeCircularBuffer(16)
    records = [[] for _ in range(consumers)]
    tally = [0]
    tally_lock = threading.Lock()

    def mp_worker(p):
        for i in range(per_producer):
            while mpmc.try_enqueue((p, i)) is FULL:
                time.sleep(0)

    def make_mc_worker(index):
        def mc_worker():
            local = records[index]
            while True:
                with tally_lock:
                    if tally[0] >= producers * per_producer:
                        return
                claimed = []
                value = mpmc.try_dequeue(on_claim=claimed.append)
                if value is EMPTY:
                    time.sleep(0)
                    continue
                local.append((claimed[0], value))
                with tally_lock:
                    tally[0] += 1
        return mc_worker

    threads = [threading.Thread(target=mp_worker, args=(p,), daemon=True) for p in range(producers)]
    threads += [threading.Thread(target=make_mc_worker(index), daemon=True) for index in range(consumers)]
    for thread in threads:
        thread.start()
    join_all(threads)

    merged = sorted(record for per_consumer in records for record in per_consumer)
    assert tally[0] == producers * per_producer, "exactly-once delivery"
    assert [pos for pos, _ in merged] == list(range(producers * per_producer)), \
        "claimed positions must be a permutation of 0..total-1"
    order = {p: [] for p in range(producers)}
    for _, (p, i) in merged:
        order[p].append(i)
    for p in range(producers):
        assert order[p] == list(range(per_producer)), "per-producer FIFO violated"
    assert mpmc.cas_attempts >= producers * per_producer, "every enqueue and dequeue used the CAS path"


if __name__ == "__main__":
    main()
    print("PASS")
