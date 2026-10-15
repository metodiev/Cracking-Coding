"""Task 354 - Implement wait-free counter sketch (atomic operations). (Tier: Intermediate, CTCI Ch 15)

CAS counter sketch: increment() runs a compare-and-swap retry loop instead of +=.
CPython would already serialise += under the GIL, but the exercise builds the explicit
atomic protocol: compare_and_swap() takes a tiny lock that stands in for ONE indivisible
hardware CAS instruction (only the compare+swap pair is atomic, never the whole
increment), and increment() re-reads and retries whenever its CAS finds the value stolen.
Strictly speaking a CAS loop is lock-free, not wait-free: one thread can be overtaken
indefinitely. THREADS x INCREMENTS concurrent increments must yield exactly
THREADS*INCREMENTS, the retry counters are asserted >= 0 (they are legitimately 0 when
the GIL hides every race), and the deterministic after_read hook forces one lost race so
the retry path itself is exercised and checked to return exactly 1.
"""
import threading

THREADS = 8
INCREMENTS = 20000


class CasCounter:
    def __init__(self):
        self._value = 0
        self._cas_lock = threading.Lock()          # emulates the hardware CAS instruction
        self.cas_attempts = 0
        self.cas_failures = 0

    def get(self):
        return self._value

    def compare_and_swap(self, expected, new_value):
        with self._cas_lock:
            self.cas_attempts += 1
            if self._value == expected:
                self._value = new_value
                return True
            self.cas_failures += 1
            return False

    def increment(self, after_read=None):
        """CAS retry loop; after_read(current) runs once between read and CAS so the
        self-test can deterministically simulate a concurrent writer stealing the value."""
        retries = 0
        hook = after_read
        while True:
            current = self._value
            if hook is not None:
                hook(current)
                hook = None
            if self.compare_and_swap(current, current + 1):
                return retries
            retries += 1


def main():
    # Single-threaded baseline: no contention, no retries.
    counter = CasCounter()
    for _ in range(100):
        assert counter.increment() == 0, "uncontended CAS never retries"
    assert counter.get() == 100
    assert counter.cas_attempts == 100 and counter.cas_failures == 0

    # Deterministic retry-path demonstration: a concurrent writer bumps the value after
    # our read, so the first CAS fails (retries == 1) and the loop rereads and succeeds.
    counter = CasCounter()
    assert counter.increment(after_read=lambda current: counter.compare_and_swap(current, current + 1)) == 1, \
        "stale read forces exactly one retry"
    assert counter.get() == 2, "both the stealing write and our retry are counted"

    # Concurrent increments: exactly THREADS * INCREMENTS, no lost updates.
    counter = CasCounter()
    retries_per_thread = [None] * THREADS

    def worker(index):
        retries = 0
        for _ in range(INCREMENTS):
            retries += counter.increment()
        retries_per_thread[index] = retries

    workers = [threading.Thread(target=worker, args=(index,), daemon=True)
               for index in range(THREADS)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(10)
        assert not thread.is_alive(), "increment worker did not finish in time"

    assert counter.get() == THREADS * INCREMENTS, "no increment lost"
    assert all(retries >= 0 for retries in retries_per_thread), "retry counters are non-negative"
    # Exact CAS accounting: every retry corresponds to exactly one failed CAS, so
    # sum(retries) == failures and attempts == increments + failures.
    assert sum(retries_per_thread) == counter.cas_failures, "retries equal recorded CAS failures"
    assert counter.cas_attempts == THREADS * INCREMENTS + counter.cas_failures, \
        "attempts == increments + failures"


if __name__ == "__main__":
    main()
    print("PASS")
