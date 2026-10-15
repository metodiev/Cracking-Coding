"""Task 358 - Dining philosophers solution using ordered locks. (Tier: Intermediate, CTCI Ch 15)

Dining philosophers solved by resource ordering: philosopher i uses forks i and (i+1)%5
but always locks the lower-numbered fork first (philosopher 4 therefore takes fork 0
before fork 4). Because every philosopher acquires its two forks in the same global
order, no circular wait can form, and each fork lock is held by at most one philosopher
at any moment. The self-test runs 5 real threads eating MEALS = 40 times each: every
thread must terminate (joins carry a timeout, so a deadlock/livelock fails the run),
total meals == 5 * MEALS, each philosopher ate MEALS times, each fork was acquired
exactly 2 * MEALS times (once per meal by each of its two neighbours), and the owner
bookkeeping proves the per-fork single-holder invariant held throughout.
"""
import threading

PHILOSOPHERS = 5
MEALS = 40


class ForkTable:
    """Fork locks plus ownership accounting updated under one short-lived lock."""

    def __init__(self, fork_count):
        self.forks = [threading.Lock() for _ in range(fork_count)]
        self.accounting = threading.Lock()
        self.owners = [None] * fork_count
        self.holders = [0] * fork_count
        self.max_holders = [0] * fork_count
        self.acquisitions = [0] * fork_count
        self.meals = [0] * fork_count
        self.violations = 0

    def acquire(self, philosopher, fork):
        self.forks[fork].acquire()
        with self.accounting:
            if self.owners[fork] is not None:
                self.violations += 1                      # would mean the lock failed
            self.owners[fork] = philosopher
            self.holders[fork] += 1
            self.max_holders[fork] = max(self.max_holders[fork], self.holders[fork])
            self.acquisitions[fork] += 1

    def release(self, philosopher, fork):
        with self.accounting:                             # record before unlocking
            if self.owners[fork] != philosopher:
                self.violations += 1
            self.owners[fork] = None
            self.holders[fork] -= 1
        self.forks[fork].release()

    def count_meal(self, philosopher):
        with self.accounting:
            self.meals[philosopher] += 1


def fork_order(philosopher, fork_count):
    """Documented lock ordering: the lower-numbered of the two forks always comes first."""
    left, right = philosopher, (philosopher + 1) % fork_count
    return min(left, right), max(left, right)


def dine(table, philosopher):
    first, second = fork_order(philosopher, len(table.forks))
    for _ in range(MEALS):
        table.acquire(philosopher, first)
        table.acquire(philosopher, second)
        table.count_meal(philosopher)
        table.release(philosopher, second)
        table.release(philosopher, first)


def main():
    # Hand-checked ordering table: only philosopher 4 wraps around, taking fork 0 first.
    assert [fork_order(index, PHILOSOPHERS) for index in range(PHILOSOPHERS)] == \
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], "documented increasing fork order"

    table = ForkTable(PHILOSOPHERS)
    diners = [threading.Thread(target=dine, args=(table, index), daemon=True)
              for index in range(PHILOSOPHERS)]
    for thread in diners:
        thread.start()
    for thread in diners:
        thread.join(10)
        assert not thread.is_alive(), "philosopher did not finish in time (deadlock?)"

    assert sum(table.meals) == PHILOSOPHERS * MEALS, "total meals == 5 * MEALS"
    assert all(count == MEALS for count in table.meals), "every philosopher ate MEALS times"
    assert table.violations == 0, "per-fork single-holder invariant never violated"
    assert all(owner is None for owner in table.owners), "every fork released at the end"
    assert all(count == 0 for count in table.holders), "no residual holders"
    assert all(count == 1 for count in table.max_holders), "every fork was used, one holder at a time"
    assert all(count == 2 * MEALS for count in table.acquisitions), \
        "each fork acquired by exactly its two neighbours"


if __name__ == "__main__":
    main()
    print("PASS")
