"""Task 347 - Read-write lock with writer preference (no writer starvation). (Tier: Intermediate, CTCI Ch 15)

Policy (documented): as soon as a writer is waiting, newly arriving readers
block, and waiting writers take the lock once the current readers/writer drain.
A never-ending stream of readers therefore cannot starve writers.

The scripted demo uses real threads, but every ordering decision is passed
through threading.Event gates and the acquire() calls take an optional on_waiting
hook that fires once the caller has registered as blocked - while still holding
the internal lock, so a release can never slip in between the registration and
the actual wait (no lost wake-up).  Assertions are exact; the only bounded waits
are liveness checks, and threads are daemons so a stuck run fails fast instead of
hanging the process.
"""
import threading


class ReadWriteLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self, on_waiting=None):
        with self._cond:
            if self._writer or self._writers_waiting > 0:
                if on_waiting is not None:
                    on_waiting()            # fired while holding the lock
                while self._writer or self._writers_waiting > 0:
                    self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self, on_waiting=None):
        with self._cond:
            self._writers_waiting += 1
            if self._writer or self._readers > 0:
                if on_waiting is not None:
                    on_waiting()
                while self._writer or self._readers > 0:
                    self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True

    def release_write(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    def state(self):
        with self._cond:
            return self._readers, self._writer, self._writers_waiting


class Auditor:
    """Invariant tracker: readers and writers must never overlap."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active_readers = 0
        self.active_writers = 0
        self.max_readers = 0
        self.violations = 0
        self.grants = []

    def enter_read(self, tag):
        with self._lock:
            if self.active_writers:
                self.violations += 1
            self.active_readers += 1
            self.max_readers = max(self.max_readers, self.active_readers)
            self.grants.append("R:" + tag)

    def exit_read(self):
        with self._lock:
            self.active_readers -= 1

    def enter_write(self, tag):
        with self._lock:
            if self.active_writers or self.active_readers:
                self.violations += 1
            self.active_writers += 1
            self.grants.append("W:" + tag)

    def exit_write(self):
        with self._lock:
            self.active_writers -= 1

    def snapshot(self):
        with self._lock:
            return (self.active_readers, self.active_writers, self.max_readers,
                    self.violations, list(self.grants))


def reader_task(rw, auditor, tag, gate, register_first):
    gate["started"].set()
    hook = gate["registered"].set if register_first else None
    rw.acquire_read(on_waiting=hook)
    auditor.enter_read(tag)
    gate["entered"].set()
    released = gate["release"].wait(15)
    auditor.exit_read()
    rw.release_read()
    if released:
        gate["done"].set()


def writer_task(rw, auditor, tag, gate, register_first):
    gate["started"].set()
    hook = gate["registered"].set if register_first else None
    rw.acquire_write(on_waiting=hook)
    auditor.enter_write(tag)
    gate["entered"].set()
    released = gate["release"].wait(15)
    auditor.exit_write()
    rw.release_write()
    if released:
        gate["done"].set()


def main():
    rw = ReadWriteLock()
    auditor = Auditor()
    tags = ("R1", "R2", "R3", "R4", "W1", "W2")
    gates = {tag: {"started": threading.Event(), "entered": threading.Event(),
                   "registered": threading.Event(), "release": threading.Event(),
                   "done": threading.Event()} for tag in tags}
    threads = []

    def start(tag, register_first):
        target = reader_task if tag.startswith("R") else writer_task
        thread = threading.Thread(target=target,
                                  args=(rw, auditor, tag, gates[tag], register_first),
                                  daemon=True)
        threads.append(thread)
        thread.start()
        assert gates[tag]["started"].wait(15), tag + " never started"

    # two readers hold the lock
    # R1 and R2 are started one after the other (R2 only once R1 is inside): two readers racing for
    # an uncontended lock are legitimately granted in either order, which would make the grant-order
    # assertion below flaky under load. Both still hold the lock at the same time.
    start("R1", False)
    assert gates["R1"]["entered"].wait(15), "R1 holds the lock"
    start("R2", False)
    assert gates["R2"]["entered"].wait(15), "R2 shares the lock"
    assert auditor.snapshot()[:3] == (2, 0, 2)
    assert rw.state() == (2, False, 0)

    # a writer arrives and must register as waiting
    start("W1", True)
    assert gates["W1"]["registered"].wait(15), "W1 registered as waiting"
    assert not gates["W1"]["entered"].is_set()
    assert rw.state() == (2, False, 1)

    # a reader arriving after the writer blocks too: that is the writer preference
    start("R3", True)
    assert gates["R3"]["registered"].wait(15), "R3 registered as waiting"
    assert not gates["R3"]["entered"].is_set()

    # drain the readers; W1 (waiting first) inherits the lock, R3 keeps waiting
    gates["R1"]["release"].set()
    gates["R2"]["release"].set()
    assert gates["R1"]["done"].wait(15) and gates["R2"]["done"].wait(15)
    assert gates["W1"]["entered"].wait(15), "W1 acquired after readers drained"
    assert not gates["R3"]["entered"].is_set()
    assert auditor.snapshot()[:3] == (0, 1, 2)
    assert rw.state() == (0, True, 0)

    gates["W1"]["release"].set()
    assert gates["W1"]["done"].wait(15)
    assert gates["R3"]["entered"].wait(15), "R3 finally granted after the writer"
    assert auditor.snapshot()[:3] == (1, 0, 2)
    gates["R3"]["release"].set()
    assert gates["R3"]["done"].wait(15)

    # phase 2: writer with no readers acquires immediately; a reader behind it waits
    start("W2", False)
    assert gates["W2"]["entered"].wait(15)
    start("R4", True)
    assert gates["R4"]["registered"].wait(15)
    assert not gates["R4"]["entered"].is_set()
    gates["W2"]["release"].set()
    assert gates["W2"]["done"].wait(15)
    assert gates["R4"]["entered"].wait(15)
    gates["R4"]["release"].set()
    assert gates["R4"]["done"].wait(15)

    for thread in threads:
        thread.join(15)
        assert not thread.is_alive(), "thread did not terminate"

    active_readers, active_writers, max_readers, violations, grants = auditor.snapshot()
    assert violations == 0, "readers and writers never overlapped"
    assert max_readers == 2, "two readers shared the lock at peak"
    assert grants == ["R:R1", "R:R2", "W:W1", "R:R3", "W:W2", "R:R4"], grants
    assert active_readers == 0 and active_writers == 0
    assert rw.state() == (0, False, 0)


if __name__ == "__main__":
    main()
    print("PASS")
