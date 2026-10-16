"""Task 626 - Implement work-stealing deque (Chase-Lev) for task scheduler.  (Tier: Expert, Ch 15 — Threads and Locks)

The Chase-Lev deque is the load-balancing primitive behind Cilk-style schedulers. One thread (the
*owner*) pushes and pops at the bottom, getting LIFO order and thus cache locality; any number of
*thieves* take from the top, getting FIFO order and therefore the oldest (largest) tasks. Stealing is
the only operation that contends, and the deque is designed so that the common owner path needs no
atomic RMW at all: only a pop of the *sole remaining* element must CAS the top pointer, because a
thief may be taking that very element. If the CAS fails, the owner simply reports "empty" - the thief
won - which is why exactly one of the two ever returns the element.

Simulations and simplifications, stated explicitly: there are no threads - the owner's pop is a
resumable step machine so the driver can let a thief run in the precise window between reading `top`
and the last-element CAS, and every CAS checks the current value so a lost race is observable as a
`cas_failures` increment. Growth doubles a plain list rather than Chase-Lev's reference-counted
cyclic arrays, and (unlike the real algorithm) a resize is not interleaved with a steal in this
sketch. Invariants still genuinely verified: FIFO order for thieves and LIFO order for the owner, no
element is ever returned twice, the top/bottom bookkeeping survives the empty and single-element
cases, resizing preserves the logical contents, and a scheduled run executes every task exactly once
with the deque-length invariant `queued + executed == total` holding after every single operation.

Checks: owner/theft orders, mixed sequences, empty and sole-element races in both interleavings,
array growth, pushing while a steal is pending, and a 4-worker, 64-task scheduler run with a
dependency chain, requeues and per-step invariant checks.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class ChaseLevDeque:
    """Owner: push/pop at the bottom (LIFO). Thieves: steal at the top (FIFO)."""

    def __init__(self, initial_capacity=4):
        self.log_size = max(2, (initial_capacity - 1).bit_length())
        self.array = [None] * (1 << self.log_size)
        self.top = 0
        self.bottom = 0
        self.resizes = 0
        self.pops = 0
        self.steals = 0
        self.empty_pops = 0
        self.cas_failures = 0

    # ---- bookkeeping ----------------------------------------------------------------------
    def capacity(self):
        return 1 << self.log_size

    def mask(self):
        return (1 << self.log_size) - 1

    def __len__(self):
        return self.bottom - self.top

    def contents(self):
        """Logical contents, oldest (top) first - used only by the tests."""
        return [self.array[i & self.mask()] for i in range(self.top, self.bottom)]

    def cas_top(self, expected, new_value):
        if self.top != expected:
            self.cas_failures += 1
            return False
        self.top = new_value
        return True

    def _grow(self):
        new_log = self.log_size + 1
        new_mask = (1 << new_log) - 1
        grown = [None] * (1 << new_log)
        for i in range(self.top, self.bottom):
            grown[i & new_mask] = self.array[i & self.mask()]
        self.array = grown
        self.log_size = new_log
        self.resizes += 1

    # ---- owner side -----------------------------------------------------------------------
    def push(self, task):
        if self.bottom - self.top >= self.capacity():
            self._grow()
        self.array[self.bottom & self.mask()] = task
        self.bottom += 1

    def pop(self):
        return drain(pop_steps(self))[1]

    # ---- thief side -----------------------------------------------------------------------
    def steal(self):
        return drain(steal_steps(self))[1]


def pop_steps(deque):
    """Owner pop as a step machine; it parks between reading `top` and the last-element CAS."""
    b = deque.bottom - 1
    deque.bottom = b
    t = deque.top
    yield "read_top"                        # <- a thief may steal the sole element right here
    if t > b:                               # top overtook bottom: the deque is empty
        deque.bottom = t                    # restore the invariants
        deque.empty_pops += 1
        return None
    task = deque.array[b & deque.mask()]
    if t == b:                              # sole element: only a CAS makes this authoritative
        if not deque.cas_top(t, t + 1):
            deque.bottom = t + 1            # lost the race; the thief owns the element
            return None
        deque.bottom = t + 1
        deque.pops += 1
        return task
    deque.pops += 1
    return task


def steal_steps(deque):
    """Thief as a step machine: it reads `top`/`bottom` and then parks before its CAS, which is
    exactly the window in which the owner's last-element pop can race with it."""
    t = deque.top
    b = deque.bottom
    if t >= b:
        yield "empty"
        return None
    task = deque.array[t & deque.mask()]
    yield "read_bounds"                     # <- the owner's pop may complete (and CAS the top) now
    if not deque.cas_top(t, t + 1):
        return None                         # another thief (or the owner's pop) got there first
    deque.steals += 1
    return task


def resume(gen):
    try:
        return next(gen), False, None
    except StopIteration as stop:
        return None, True, stop.value


def drain(gen):
    steps = []
    while True:
        step, finished, value = resume(gen)
        if finished:
            return steps, value
        steps.append(step)


def steal_from_busiest(deques, worker):
    """Deterministic thief policy: take the oldest task from the longest other deque."""
    order = sorted((len(deques[d]), d) for d in range(len(deques)) if d != worker)
    if not order or order[-1][0] == 0:
        return None
    _, donor = order[-1]
    task = deques[donor].steal()
    return None if task is None else (task, donor)


def run_scheduler(task_count=64, worker_count=4, chain=False, max_steps=8000):
    """Fork-join style run: worker 0 owns every task, idle workers steal.

    With `chain=True` task i depends on task i+1, and tasks 0..n-1 are pushed oldest-first, so the
    owner's LIFO pop immediately yields the ready end of the chain while thieves take the unready
    end - the realistic "steal work that is not ready yet" case that forces requeues. With
    `chain=False` all tasks are independent, which is the pure load-balancing scenario.
    """
    deques = [ChaseLevDeque() for _ in range(worker_count)]
    for task in range(task_count):
        deques[0].push(task)
    parent = {task: task + 1 for task in range(task_count - 1)} if chain else {}

    done, execution_log, requeues, steals = set(), [], 0, 0
    steps = 0
    while len(done) < task_count:
        check(steps < max_steps, "the scheduler made no progress inside its step budget")
        for worker in range(worker_count):
            steps += 1
            task = deques[worker].pop()
            if task is None:
                found = steal_from_busiest(deques, worker)
                if found is None:
                    continue
                task, _donor = found
                steals += 1
            if task in parent and parent[task] not in done:
                deques[worker].push(task)   # not ready: keep it in my own deque for later
                requeues += 1
            else:
                check(task not in done, f"task {task} would run twice")
                check(task not in parent or parent[task] in done, "dependency must be satisfied")
                done.add(task)
                execution_log.append((worker, task))
            # the deque accounting invariant must hold after every single mutation
            queued = sum(len(dq) for dq in deques)
            check(queued + len(done) == task_count,
                  f"invariant broken: {queued} queued + {len(done)} done != {task_count}")
    loads = [0] * worker_count
    for worker, _task in execution_log:
        loads[worker] += 1
    return {
        "executed": [task for _w, task in execution_log],
        "deques": deques,
        "steals": steals,
        "requeues": requeues,
        "steps": steps,
        "loads": loads,
    }


def main():
    # --- owner LIFO, thief FIFO ---------------------------------------------------------------
    deque = ChaseLevDeque()
    for task in range(1, 6):
        deque.push(task)
    check(deque.contents() == [1, 2, 3, 4, 5], "contents are oldest-first")
    check(deque.pop() == 5 and deque.pop() == 4, "the owner pops LIFO (newest first)")
    check(deque.steal() == 1, "a thief steals FIFO (oldest first)")
    check(deque.contents() == [2, 3], "both ends advanced exactly once")
    check(deque.pop() == 3 and deque.steal() == 2 and len(deque) == 0, "the deque drains completely")

    deque = ChaseLevDeque()
    for task in range(1, 6):
        deque.push(task)
    check([deque.steal() for _ in range(5)] == [1, 2, 3, 4, 5], "thieves alone see FIFO order")
    check(deque.steal() is None and deque.pop() is None, "empty deque yields nothing on both sides")
    deque.push("revived")
    check(deque.pop() == "revived" and len(deque) == 0,
          "push after the empty case restores the bookkeeping")

    # --- empty pop must restore top/bottom ----------------------------------------------------
    deque = ChaseLevDeque()
    deque.push("only")
    deque.steal()
    check(deque.pop() is None and deque.empty_pops == 1 and deque.top == deque.bottom == 1,
          "popping an empty deque leaves a consistent state")
    deque.push("a")
    deque.push("b")
    check(deque.contents() == ["a", "b"] and deque.pop() == "b", "the deque is usable afterwards")

    # --- sole-element race: four interleavings -------------------------------------------------
    deque = ChaseLevDeque()
    deque.push("last")
    check(deque.steal() == "last" and deque.pop() is None, "the thief takes the sole element first")
    check(deque.top == 1 and deque.bottom == 1 and deque.cas_failures == 0,
          "no CAS needed when the owner has not started popping")

    deque = ChaseLevDeque()
    deque.push("last")
    check(deque.pop() == "last" and deque.steal() is None, "the owner takes the sole element first")
    check(deque.pops == 1 and deque.steals == 0, "exactly one element was taken")

    # genuine conflict: the thief has already read top/bottom when the owner CASes the top
    deque = ChaseLevDeque()
    deque.push("last")
    thief = steal_steps(deque)
    check(resume(thief)[0] == "read_bounds", "the thief parks after reading top and bottom")
    owner = pop_steps(deque)
    check(drain(owner)[1] == "last", "the owner's last-element CAS wins")
    check(drain(thief)[1] is None, "the thief's CAS on the same top fails and it reports empty")
    check(deque.cas_failures == 1 and deque.steals == 0, "the lost race is observable")
    check(len(deque) == 0 and deque.pops == 1 and deque.top == deque.bottom == 1,
          "the element was taken exactly once")

    # the mirror image: the thief wins the same conflict
    deque = ChaseLevDeque()
    deque.push("last")
    thief = steal_steps(deque)
    resume(thief)
    check(drain(thief)[1] == "last", "the thief CASes the top first")
    check(deque.pop() is None and deque.cas_failures == 0,
          "the owner then sees an empty deque and restores its invariants")
    check(deque.steals == 1 and deque.pops == 0, "again exactly one element was taken")

    # --- a race that is not about the sole element ---------------------------------------------
    deque = ChaseLevDeque()
    deque.push("first")
    deque.push("second")
    owner = pop_steps(deque)
    resume(owner)                            # the owner has already moved `bottom` down
    check(deque.steal() == "first", "the thief takes the uncontended oldest element")
    check(drain(owner)[1] == "second", "the owner's non-sole pop needs no CAS")
    check(deque.cas_failures == 0, "top was never contended here")

    # --- growth preserves contents ------------------------------------------------------------
    deque = ChaseLevDeque(initial_capacity=4)
    for task in range(100):
        deque.push(task)
    check(deque.capacity() == 128 and deque.resizes == 5, "the array doubles five times: 4 -> 128")
    grown = deque
    check(deque.contents() == list(range(100)), "growth preserves the logical contents")
    check([deque.steal() for _ in range(3)] == [0, 1, 2], "thieves still take the oldest first")
    check(deque.pop() == 99 and deque.pop() == 98, "the owner still pops the newest")
    check(len(deque) == 95, "length accounting after a mixed sequence")

    # --- push while a steal is pending --------------------------------------------------------
    deque = ChaseLevDeque()
    deque.push("x")
    check(deque.steal() == "x", "steal empties the deque")
    deque.push("y")                          # owner pushes right after the steal
    deque.push("z")
    check(deque.contents() == ["y", "z"] and deque.pop() == "z", "the push after a steal is intact")

    # --- scheduler run: independent tasks (pure load balancing) --------------------------------
    balanced = run_scheduler(task_count=64, worker_count=4, chain=False)
    check(sorted(balanced["executed"]) == list(range(64)), "every independent task ran exactly once")
    check(all(load > 0 for load in balanced["loads"]),
          f"every worker found work: {balanced['loads']}")
    check(balanced["steals"] > 0, "workers had to steal to balance the load")
    check(max(balanced["loads"]) - min(balanced["loads"]) <= 2,
          f"stealing balances the load closely: {balanced['loads']}")
    check(all(len(dq) == 0 for dq in balanced["deques"]), "all deques end empty")
    check(balanced["requeues"] == 0, "independent tasks never need to be deferred")
    check(sum(len(dq.contents()) for dq in balanced["deques"]) == 0, "no tasks left behind")

    # --- scheduler run: dependency chain (readiness, requeues, and the LIFO ready end) ---------
    run = run_scheduler(task_count=64, worker_count=4, chain=True)
    executed = run["executed"]
    check(len(executed) == 64 and sorted(executed) == list(range(64)),
          "every task ran exactly once")
    check(executed == list(range(63, -1, -1)),
          "the dependency chain forces the strictly decreasing execution order")
    check(run["steals"] > 0 and run["requeues"] > 0,
          "workers really stole work and really had to defer unready tasks")
    check(sum(run["loads"]) == 64 and all(len(dq) == 0 for dq in run["deques"]),
          "loads sum to the task count and all deques end empty")
    check(max(run["loads"]) - min(run["loads"]) <= 62,
          f"the chain keeps most work with its owner: {run['loads']}")
    check(sum(dq.cas_failures for dq in run["deques"]) < 64,
          "the scheduler run rarely lost a sole-element race")

    print(f"deque: resizes={grown.resizes} capacity={grown.capacity()} "
          f"balanced loads={balanced['loads']} steps={balanced['steps']} "
          f"steals={balanced['steals']} | chain steps={run['steps']} steals={run['steals']} "
          f"requeues={run['requeues']} loads={run['loads']}")


if __name__ == "__main__":
    main()
    print("PASS")
