"""Task 349 - Reusable barrier synchronization primitive. (Tier: Intermediate, CTCI Ch 15)

Generation-based barrier: participants bump a counter and wait until the last one
arrives; the last participant resets the counter, flips the generation and wakes
everyone - so the same object is immediately reusable for the next round.  A
barrier with one participant is a no-op.  wait(timeout) returns False when the
caller times out before all parties arrived and withdraws its own count, leaving
the barrier usable.

The scripted arena runs four real threads through five rounds and records a
global event sequence; the barrier contract "nobody departs a round before every
participant arrived in that round" is then asserted on that sequence.  Bounded
waits are liveness checks only and threads are daemons, so a stuck run fails fast
instead of hanging.
"""
import threading


class Barrier:
    def __init__(self, parties):
        if parties <= 0:
            raise ValueError("parties must be positive")
        self.parties = parties
        self._count = 0
        self._generation = 0
        self._cond = threading.Condition()

    def wait(self, timeout=None):
        with self._cond:
            generation = self._generation
            self._count += 1
            if self._count == self.parties:
                self._count = 0
                self._generation += 1
                self._cond.notify_all()
                return True
            while generation == self._generation:
                if not self._cond.wait(timeout):
                    self._count -= 1     # withdraw; the barrier stays usable
                    return False
            return True

    def state(self):
        with self._cond:
            return self._count, self._generation


def main():
    records = []
    record_lock = threading.Lock()

    def record(kind, round_index, name):
        with record_lock:
            records.append((len(records), kind, round_index, name))

    def participant(name, barrier, rounds, failures):
        try:
            for round_index in range(rounds):
                record("arrive", round_index, name)
                if not barrier.wait(15):
                    failures.append(name + ": timed out")
                    return
                record("depart", round_index, name)
        except Exception as exc:         # surface worker failures in the main thread
            failures.append("%s: %r" % (name, exc))

    rounds = 5
    barrier = Barrier(4)
    failures = []
    names = ["P0", "P1", "P2", "P3"]
    threads = [threading.Thread(target=participant, args=(name, barrier, rounds, failures),
                                daemon=True) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive(), "participant did not terminate"
    assert failures == []

    assert sum(1 for record_entry in records if record_entry[1] == "arrive") == 4 * rounds
    assert sum(1 for record_entry in records if record_entry[1] == "depart") == 4 * rounds

    # per round: every arrival must happen before every departure
    for round_index in range(rounds):
        arrivals = [seq for seq, kind, r, _ in records if kind == "arrive" and r == round_index]
        departures = [seq for seq, kind, r, _ in records if kind == "depart" and r == round_index]
        assert len(arrivals) == 4 and len(departures) == 4
        assert max(arrivals) < min(departures), "round %d released early" % round_index

    # each participant passed every round, in round order
    for name in names:
        arrivals = [r for _, kind, r, n in records if kind == "arrive" and n == name]
        departures = [r for _, kind, r, n in records if kind == "depart" and n == name]
        assert arrivals == list(range(rounds))
        assert departures == list(range(rounds))

    # the barrier reset itself after every round
    assert barrier.state() == (0, rounds)

    # a single-participant barrier is a no-op and reusable
    solo = Barrier(1)
    for _ in range(3):
        assert solo.wait() is True
    assert solo.state() == (0, 3)

    # timeout path: the caller withdraws cleanly and the barrier stays usable
    pair = Barrier(2)
    assert pair.wait(timeout=0.05) is False
    assert pair.state() == (0, 0)
    results = []
    thread_a = threading.Thread(target=lambda: results.append(pair.wait(15)), daemon=True)
    thread_b = threading.Thread(target=lambda: results.append(pair.wait(15)), daemon=True)
    thread_a.start()
    thread_b.start()
    thread_a.join(15)
    thread_b.join(15)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert results == [True, True]
    assert pair.state() == (0, 1)


if __name__ == "__main__":
    main()
    print("PASS")
