"""Task 359 - Readers-writers fairness implementation and tests. (Tier: Intermediate, CTCI Ch 15)

Readers-writers lock with the documented WRITER-PREFERENCE policy: a reader may enter
only when no writer is active and no writer is waiting (waiting_writers > 0), so a stream
of arriving readers can never starve a writer; a writer enters when no writer is active
and all active readers have left. The self-test has three parts: (1) a scripted
deterministic ordering test using events - reader A holds the lock, writer W registers,
then reader B arrives; after A leaves, W must enter before B, proving the policy; (2) a
mutual-exclusion check on every entry/exit (a writer may never release while readers are
active, readers may never enter next to a writer); (3) a threaded workload (4 readers x
50, 3 writers x 50) asserting that the number of writer entries equals the number of
writer requests (fairness), reader entries equal reader requests, and every join is clean.
"""
import threading

READERS = 4
WRITERS = 3
ITERATIONS = 50


class WriterPreferenceRW:
    """Documented fairness policy: writer preference (FIFO among writer waiters)."""

    def __init__(self):
        self._cond = threading.Condition()
        self.active_readers = 0
        self.active_writer = False
        self.waiting_writers = 0
        self.reader_entries = 0
        self.writer_entries = 0
        self.max_active_readers = 0
        self.violations = 0

    def acquire_read(self):
        with self._cond:
            while self.active_writer or self.waiting_writers > 0:
                self._cond.wait()
            if self.active_writer:
                self.violations += 1                      # unreachable: writer would overlap
            self.active_readers += 1
            self.max_active_readers = max(self.max_active_readers, self.active_readers)
            self.reader_entries += 1

    def release_read(self):
        with self._cond:
            if self.active_readers <= 0 or self.active_writer:
                self.violations += 1                      # a writer overlapped this reader
            self.active_readers -= 1
            if self.active_readers == 0:
                self._cond.notify_all()                   # last reader lets writers proceed

    def acquire_write(self, before_wait=None):
        with self._cond:
            self.waiting_writers += 1
            if before_wait is not None:
                before_wait()                             # hook runs with the intent registered
            while self.active_writer or self.active_readers > 0:
                self._cond.wait()
            if self.active_writer or self.active_readers > 0:
                self.violations += 1                      # unreachable: entry is gated
            self.waiting_writers -= 1
            self.active_writer = True
            self.writer_entries += 1

    def release_write(self):
        with self._cond:
            if not self.active_writer or self.active_readers > 0:
                self.violations += 1                      # readers overlapped this writer
            self.active_writer = False
            self._cond.notify_all()


def main():
    # Part 1: scripted writer-preference ordering test.
    rw = WriterPreferenceRW()
    log = []
    log_lock = threading.Lock()
    reader_a_acquired = threading.Event()
    release_a = threading.Event()
    writer_registered = threading.Event()
    writer_entered = threading.Event()
    release_w = threading.Event()
    reader_b_entered = threading.Event()

    def say(tag):
        with log_lock:
            log.append(tag)

    def reader_a():
        rw.acquire_read()
        say("readA")
        reader_a_acquired.set()
        assert release_a.wait(10), "reader A released in time"
        rw.release_read()

    def writer_w():
        rw.acquire_write(before_wait=writer_registered.set)
        say("writeW")
        writer_entered.set()
        assert release_w.wait(10), "writer W released in time"
        rw.release_write()

    def reader_b():
        rw.acquire_read()
        say("readB")
        reader_b_entered.set()
        rw.release_read()

    threads = [threading.Thread(target=reader_a, daemon=True),
               threading.Thread(target=writer_w, daemon=True),
               threading.Thread(target=reader_b, daemon=True)]
    threads[0].start()
    assert reader_a_acquired.wait(10), "reader A got in first"

    threads[1].start()
    assert writer_registered.wait(10), "writer W registered its request"
    threads[2].start()                                # reader B arrives while W waits

    release_a.set()
    assert writer_entered.wait(10), "writer W entered after reader A left"
    assert not reader_b_entered.is_set(), "new reader waits while a writer is pending"
    release_w.set()
    assert reader_b_entered.wait(10), "reader B entered once the writer was done"

    for thread in threads:
        thread.join(10)
        assert not thread.is_alive(), "scripted thread did not finish in time"
    assert log == ["readA", "writeW", "readB"], "writer-preference order"
    assert rw.writer_entries == 1 and rw.reader_entries == 2, "entries counted"
    assert rw.violations == 0, "scripted scenario kept mutual exclusion"

    # Part 2 + 3: threaded workload with the fairness accounting.
    rw = WriterPreferenceRW()
    state = {"value": 0}                              # guarded by the write lock
    observations = []
    writer_completions = [0] * WRITERS
    reader_completions = [0] * READERS

    def reader_loop(reader_id):
        for _ in range(ITERATIONS):
            rw.acquire_read()
            observations.append(state["value"])
            rw.release_read()
            reader_completions[reader_id] += 1

    def writer_loop(writer_id):
        for _ in range(ITERATIONS):
            rw.acquire_write()
            state["value"] += 1
            rw.release_write()
            writer_completions[writer_id] += 1

    workers = ([threading.Thread(target=reader_loop, args=(i,), daemon=True) for i in range(READERS)]
               + [threading.Thread(target=writer_loop, args=(i,), daemon=True) for i in range(WRITERS)])
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(10)
        assert not thread.is_alive(), "workload thread did not finish in time"

    assert sum(writer_completions) == WRITERS * ITERATIONS, "every writer request was served"
    assert rw.writer_entries == WRITERS * ITERATIONS == 150, \
        "writer entries == writer requests (no starvation)"
    assert rw.reader_entries == READERS * ITERATIONS == 200, "reader entries == reader requests"
    assert state["value"] == 150, "all writer increments applied exactly once"
    assert rw.violations == 0, "mutual exclusion held throughout the workload"
    assert len(observations) == 200, "every read produced an observation"
    assert all(0 <= value <= 150 for value in observations), "readers only saw committed values"
    assert rw.max_active_readers >= 1, "readers actually ran"
    assert rw.active_readers == 0 and not rw.active_writer, "lock left clean"


if __name__ == "__main__":
    main()
    print("PASS")
