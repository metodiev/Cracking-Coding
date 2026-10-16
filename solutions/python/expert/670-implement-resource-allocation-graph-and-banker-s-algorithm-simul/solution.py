"""Task 670 - Implement resource allocation graph and Banker's algorithm simulator.  (Tier: Expert, Beyond — Networking, OS & Real-Time)

Two classic deadlock tools, each with an independent oracle.

`BankersState` implements the Banker's algorithm (Silberschatz): it keeps `available`,
`maximum` and `allocation`, derives `need = maximum - allocation`, and runs the textbook
safety algorithm, returning the *witness* finish order. `request(p, req)` grants only when
the request fits in `need`, fits in `available`, and leaves the state safe; otherwise it
denies and restores the previous state bit for bit. The conservation invariant
`sum_p allocation[p][r] + available[r] == total[r]` is re-checked after every operation,
and a second recursive safety checker (exhaustive search over finish orders instead of the
greedy loop) confirms the verdict on randomly generated states.

`ResourceAllocationGraph` is the single-instance request/assignment graph: one holder and
at most one pending request per resource instance. DFS cycle detection over the bipartite
P/R nodes is cross-checked against the classical reduction oracle (a process is
completable once everything it waits for is free or held by an already-completable
process). For single-instance resources the two must agree exactly: a cycle exists iff at
least one process can never finish. Simplified: one request per call, single-instance
resources only - precisely the case where "cycle => deadlock" is a theorem.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class BankersState:
    """Banker's algorithm state: available vector, maximum claims, allocation."""

    def __init__(self, available, maximum, allocation):
        self.processes = len(maximum)
        self.resources = len(available)
        if len(allocation) != self.processes:
            raise ValueError("allocation must have one row per process")
        for row in maximum + allocation:
            if len(row) != self.resources or any(v < 0 for v in row):
                raise ValueError("every row needs one non-negative entry per resource")
        if any(v < 0 for v in available):
            raise ValueError("available must be non-negative")
        self.available = list(available)
        self.maximum = [list(row) for row in maximum]
        self.allocation = [list(row) for row in allocation]
        if any(self.allocation[p][r] > self.maximum[p][r]
               for p in range(self.processes) for r in range(self.resources)):
            raise ValueError("allocation exceeds a maximum claim")
        self.total = [self.available[r] + sum(self.allocation[p][r] for p in range(self.processes))
                      for r in range(self.resources)]

    def need(self):
        return [[self.maximum[p][r] - self.allocation[p][r] for r in range(self.resources)]
                for p in range(self.processes)]

    def snapshot(self):
        return (tuple(self.available), tuple(tuple(row) for row in self.allocation))

    def totals_hold(self):
        return all(self.available[r] + sum(self.allocation[p][r] for p in range(self.processes))
                   == self.total[r] for r in range(self.resources))

    def is_safe(self):
        """Safety algorithm; returns (safe, witness finish order)."""
        work = list(self.available)
        need = self.need()
        done = [False] * self.processes
        finished = []
        while True:
            picked = None
            for p in range(self.processes):
                if not done[p] and all(need[p][r] <= work[r] for r in range(self.resources)):
                    picked = p
                    break
            if picked is None:
                return len(finished) == self.processes, finished
            done[picked] = True
            finished.append(picked)
            for r in range(self.resources):
                work[r] += self.allocation[picked][r]

    def verify_finish_order(self, order):
        """Witness check: every process in `order` must really fit in the running work."""
        if sorted(order) != list(range(self.processes)):
            return False
        work = list(self.available)
        need = self.need()
        for p in order:
            if any(need[p][r] > work[r] for r in range(self.resources)):
                return False
            for r in range(self.resources):
                work[r] += self.allocation[p][r]
        return True

    def is_safe_oracle(self):
        """Same question, different formulation: exhaustive search over finish orders."""
        def search(work, need, finished):
            if len(finished) == self.processes:
                return True
            for p in range(self.processes):
                if p not in finished and all(need[p][r] <= work[r] for r in range(self.resources)):
                    grown = [work[r] + self.allocation[p][r] for r in range(self.resources)]
                    if search(grown, need, finished | {p}):
                        return True
            return False
        return search(list(self.available), self.need(), frozenset())

    def request(self, p, req):
        """Returns (granted, reason) with reason in {granted, need, available, unsafe}."""
        if not 0 <= p < self.processes:
            raise ValueError("unknown process P%d" % p)
        if len(req) != self.resources or any(v < 0 for v in req):
            raise ValueError("request must be a non-negative vector over all resources")
        need = self.need()
        if any(req[r] > need[p][r] for r in range(self.resources)):
            return False, "need"
        if any(req[r] > self.available[r] for r in range(self.resources)):
            return False, "available"
        for r in range(self.resources):
            self.available[r] -= req[r]
            self.allocation[p][r] += req[r]
        if not self.is_safe()[0]:
            for r in range(self.resources):
                self.available[r] += req[r]
                self.allocation[p][r] -= req[r]
            return False, "unsafe"
        return True, "granted"


class ResourceAllocationGraph:
    """Single-instance RAG: holders[j] holds Rj, requests[j] is Rj's outstanding waiter."""

    def __init__(self, process_count, holders, requests):
        self.process_count = process_count
        self.holders = list(holders)
        self.requests = list(requests)
        if len(self.holders) != len(self.requests):
            raise ValueError("holders and requests must cover the same resources")
        for j, (h, q) in enumerate(zip(self.holders, self.requests)):
            if h is not None and not 0 <= h < process_count:
                raise ValueError("bad holder for R%d" % j)
            if q is not None and not 0 <= q < process_count:
                raise ValueError("bad requester for R%d" % j)
            if h is not None and h == q:
                raise ValueError("R%d cannot be held and requested by the same process" % j)

    def find_cycle(self):
        """DFS with colours over the P/R nodes; the nodes of a cycle, or None."""
        graph = {}
        for j in range(len(self.holders)):
            if self.requests[j] is not None:
                graph.setdefault(("P", self.requests[j]), []).append(("R", j))
            if self.holders[j] is not None:
                graph.setdefault(("R", j), []).append(("P", self.holders[j]))
        colour = {}
        path = []

        def visit(node):
            colour[node] = 1
            path.append(node)
            for nxt in sorted(graph.get(node, ())):
                if colour.get(nxt, 0) == 1:
                    return path[path.index(nxt):]
                if colour.get(nxt, 0) == 0:
                    found = visit(nxt)
                    if found:
                        return found
            path.pop()
            colour[node] = 2
            return None

        nodes = [("P", p) for p in range(self.process_count)]
        nodes += [("R", j) for j in range(len(self.holders))]
        for node in nodes:
            if colour.get(node, 0) == 0:
                found = visit(node)
                if found:
                    return found
        return None

    def has_cycle(self):
        return self.find_cycle() is not None

    def deadlocked(self):
        """Reduction oracle: finish every process whose waits are satisfiable, repeat."""
        done = set()
        changed = True
        while changed:
            changed = False
            for p in range(self.process_count):
                if p in done:
                    continue
                waits = [j for j in range(len(self.holders)) if self.requests[j] == p]
                if all(self.holders[j] is None or self.holders[j] in done for j in waits):
                    done.add(p)
                    changed = True
        return sorted(p for p in range(self.process_count) if p not in done)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_bankers_state(seed, processes, resources):
    """A valid random state: allocation and need are drawn first, maximum = alloc + need."""
    state = seed
    available = []
    allocation = [[0] * resources for _ in range(processes)]
    need = [[0] * resources for _ in range(processes)]
    for r in range(resources):
        state, raw = splitmix64(state)
        for _ in range(raw % (processes + 1)):
            state, raw = splitmix64(state)
            allocation[raw % processes][r] += 1
        state, raw = splitmix64(state)
        available.append(raw % 4)
        for p in range(processes):
            state, raw = splitmix64(state)
            need[p][r] = raw % 4
    maximum = [[allocation[p][r] + need[p][r] for r in range(resources)]
               for p in range(processes)]
    return BankersState(available, maximum, allocation)


def random_rag(seed, processes, resources):
    state = seed
    holders, requests = [], []
    for _ in range(resources):
        state, raw = splitmix64(state)
        holder = None if raw % 3 == 0 else raw % processes
        state, raw = splitmix64(state)
        requester = None if raw % 4 == 0 else raw % processes
        holders.append(holder)
        requests.append(None if requester == holder else requester)
    return ResourceAllocationGraph(processes, holders, requests)


def main():
    # ---- the textbook Banker's example (Silberschatz fig. 7.5) ------------------------
    bank = BankersState([3, 3, 2],
                        [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]],
                        [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]])
    check(bank.need() == [[7, 4, 3], [1, 2, 2], [6, 0, 0], [0, 1, 1], [4, 3, 1]],
          "need = maximum - allocation: %r" % (bank.need(),))
    safe, order = bank.is_safe()
    check(safe and order == [1, 3, 0, 2, 4], "the textbook state is safe, witness %r" % (order,))
    check(bank.verify_finish_order(order) and bank.is_safe_oracle(),
          "greedy witness and exhaustive oracle must both confirm safety")
    check(bank.totals_hold(), "allocation + available must equal the total resources")

    check(bank.request(1, [1, 0, 2]) == (True, "granted"), "P1's textbook request is granted")
    check(bank.available == [2, 3, 0] and bank.allocation[1] == [3, 0, 2],
          "the grant moves exactly the requested amounts")
    check(bank.is_safe()[0] and bank.is_safe_oracle(), "the post-grant state is still safe")
    before = bank.snapshot()
    check(bank.request(4, [3, 3, 0]) == (False, "available"), "P4 asks for more than exists")
    check(bank.request(0, [0, 2, 0]) == (False, "unsafe"),
          "P0's available request would leave an unsafe state")
    check(bank.request(0, [9, 0, 0]) == (False, "need"),
          "a request beyond the remaining need is refused")
    check(bank.snapshot() == before, "every denial must roll back completely")
    check(bank.request(0, [0, 1, 0]) == (True, "granted"), "a smaller P0 request is safe")

    # ---- randomized cross-checks -----------------------------------------------------
    safe_states = 0
    for i in range(160):
        state = random_bankers_state(0xABCDEF + i * 7919, 2 + i % 3, 2 + i % 2)
        greedy, witness = state.is_safe()
        check(greedy == state.is_safe_oracle(), "safety verdicts differ on state %d" % i)
        check(state.verify_finish_order(witness) == greedy, "witness mismatch on state %d" % i)
        check(state.totals_hold(), "constructed totals broken on state %d" % i)
        safe_states += greedy
    check(0 < safe_states < 160, "the sweep must mix safe (%d) and unsafe states" % safe_states)

    granted_count = denied_count = 0
    for i in range(160):
        state = random_bankers_state(0x1357 + i * 104729, 3, 2)
        need = state.need()
        before = state.snapshot()
        granted, reason = state.request(1, [need[1][r] if (i >> r) & 1 else 0 for r in range(2)])
        check(state.totals_hold(), "conservation must hold after every request")
        if granted:
            granted_count += 1
            check(reason == "granted" and state.is_safe_oracle(),
                  "every grant must leave a provably safe state")
        else:
            denied_count += 1
            check(reason in ("need", "available", "unsafe"), "bad denial reason %r" % reason)
            check(state.snapshot() == before, "a denial must leave the state untouched")
    check(granted_count > 0 and denied_count > 0,
          "the sweep must exercise grants (%d) and denials (%d)" % (granted_count, denied_count))

    # ---- resource allocation graph ---------------------------------------------------
    swap = ResourceAllocationGraph(3, [0, 1], [1, 0])
    found = swap.find_cycle()
    check(found is not None and set(found) == {("P", 0), ("P", 1), ("R", 0), ("R", 1)},
          "a two-process swap is a cycle: %r" % (found,))
    check(swap.deadlocked() == [0, 1], "both swapped processes are deadlocked")
    acyclic = ResourceAllocationGraph(3, [0, None], [None, 0])
    check(not acyclic.has_cycle() and acyclic.deadlocked() == [],
          "an acyclic single-instance graph cannot deadlock")
    dangling = ResourceAllocationGraph(4, [0, 1], [1, 2])
    check(not dangling.has_cycle() and dangling.deadlocked() == [],
          "waiting for a free resource is not a deadlock")
    ring = ResourceAllocationGraph(3, [0, 1, 2], [1, 2, 0])
    check(ring.has_cycle() and len(ring.find_cycle()) == 6 and ring.deadlocked() == [0, 1, 2],
          "a three-process ring is a six-node cycle that deadlocks everyone")

    both_true = 0
    for i in range(200):
        graph = random_rag(0x2468 + i * 31337, 3, 3 + i % 3)
        deadlocked = graph.deadlocked() != []
        check(deadlocked == graph.has_cycle(),
              "cycle <=> deadlock failed: holders=%r requests=%r deadlocked=%r cycle=%r"
              % (graph.holders, graph.requests, graph.deadlocked(), graph.find_cycle()))
        if deadlocked:
            both_true += 1
            check(len(graph.find_cycle()) >= 4, "a cycle spans at least P and R nodes")
    check(both_true > 0, "the sweep must contain real deadlocks (%d)" % both_true)

    # ---- invalid input ---------------------------------------------------------------
    bad_calls = [lambda: BankersState([3], [[2]], [[3]]),
                 lambda: BankersState([-1], [[2]], [[1]]),
                 lambda: BankersState([1, 1], [[2]], [[1]]),
                 lambda: bank.request(9, [0, 0, 0]),
                 lambda: bank.request(0, [1]),
                 lambda: bank.request(0, [-1, 0, 0]),
                 lambda: ResourceAllocationGraph(2, [0], [0]),
                 lambda: ResourceAllocationGraph(2, [5], [None])]
    rejected = 0
    for bad in bad_calls:
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == len(bad_calls),
          "every invalid input must raise ValueError (%d of %d)" % (rejected, len(bad_calls)))

    print("bankers: %d/%d random states safe, textbook witness = %r" % (safe_states, 160, order))
    print("rag: %d deadlocked graphs of 200, each matched by cycle detection" % both_true)
    print("PASS")


if __name__ == "__main__":
    main()
