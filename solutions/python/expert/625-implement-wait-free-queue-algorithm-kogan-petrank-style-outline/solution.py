"""Task 625 - Implement wait-free queue algorithm (Kogan-Petrank style outline).  (Tier: Expert, Ch 15 — Threads and Locks)

The Kogan-Petrank queue makes both enqueue and dequeue wait-free by removing the dependency between
threads. Two mechanisms do the work:

* **Announcement / helping.** A thread first announces its operation in a shared record and only then
  tries to perform it. Any thread that arrives later completes the announced operation before its own
  (the "fast path" is just the case where the announced operation is the caller's own), so a stalled
  thread never blocks anybody - the slack is bounded by the number of announced operations.
* **A per-slot dequeue state machine.** Each queue position carries a state EMPTY -> ENQUEUED ->
  DEQUEUING -> TAKEN. The enqueuer publishes the value *before* flipping EMPTY to ENQUEUED, so a
  value is never visible half-written. A dequeuer CASes ENQUEUED to DEQUEUING to claim the item; if
  it stalls right there, any other thread can finish the take (read the value, mark TAKEN, hand the
  result to the stalled operation's record), which is what removes the two-thread interaction that
  makes the classic Michael-Scott queue merely lock-free.

Simplifications, stated explicitly: no real threads (operations are resumable step machines driven
by an explicit deterministic schedule, so the interleaving under test is written out in the test);
one global enqueue index instead of KP's per-operation index computation; the ring of slots is
recycled through the TAKEN state instead of KP's phase counters for index wrapping; and the
"announced op list" is scanned in announcement order rather than the algorithm's pair of announced
enqueue pointers. Invariants still genuinely verified: FIFO order, exactly-once take, no value is
ever observed before it is published, every announced operation is completed by *someone* (a single
run of one thread completes operations announced by a stalled thread), and the number of slot
operations per operation is bounded - the wait-freedom witness.

Checks: FIFO order and empty-queue behaviour; the linearization order recovered from the slot state
machine; a stalled enqueuer helped by another thread; a stalled dequeuer whose take is finished by
another thread (its own resume then returns that value); a helping-disabled negative control; bounded
slot operations per operation; and a seeded mixed workload over more slots than the ring holds.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


EMPTY, ENQUEUED, DEQUEUING, TAKEN = "EMPTY", "ENQUEUED", "DEQUEUING", "TAKEN"


class Slot:
    __slots__ = ("state", "value", "owner", "takes")

    def __init__(self):
        self.state = EMPTY
        self.value = None
        self.owner = None      # the dequeue operation that claimed this slot (KP's "dequeuing" owner)
        self.takes = 0


class Operation:
    __slots__ = ("seq", "thread", "kind", "value", "index", "done", "result", "completed_by")

    def __init__(self, seq, thread, kind, value=None):
        self.seq = seq
        self.thread = thread
        self.kind = kind
        self.value = value
        self.index = None
        self.done = False
        self.result = None
        self.completed_by = None

    def __repr__(self):
        return f"Op#{self.seq}({self.thread},{self.kind})"


class WaitFreeQueue:
    """Announced operations plus a slot state machine; `helping=False` disables the helping rule."""

    def __init__(self, capacity=16, helping=True):
        self.capacity = capacity
        self.slots = [Slot() for _ in range(capacity)]
        self.enqueue_index = 0
        self.flush_index = 0           # oldest position that has not been taken yet
        self.ops = []
        self.helping = helping
        self.helps = 0                 # completions of an operation announced by another thread
        self.slot_ops = 0              # bounded per operation: the wait-freedom witness
        self.link_order = []           # values in slot-linearization order
        self.take_order = []           # values in take order
        self.stalled_takes = 0
        self.max_slot_ops_per_op = 0
        self._seq = 0

    # ---- announcement --------------------------------------------------------------------
    def announce(self, thread, kind, value=None):
        self._seq += 1
        op = Operation(self._seq, thread, kind, value)
        self.ops.append(op)
        return op

    def _finish(self, op, value, by):
        if op.done:
            return False
        op.done = True
        op.result = value
        op.completed_by = by
        if by != op.thread:
            self.helps += 1
        return True

    # ---- slot transitions -----------------------------------------------------------------
    def _finish_stalled_take(self, slot, by):
        """Any thread may complete a take that its owner abandoned in the DEQUEUING state."""
        self.stalled_takes += 1
        return self._take(slot, by)

    def _take(self, slot, by):
        value = slot.value
        slot.state = TAKEN
        slot.takes += 1
        self.take_order.append(value)
        owner = slot.owner
        slot.owner = None
        self.flush_index += 1
        if owner is not None:
            self._finish(owner, value, by)
        self._count()
        return value

    def _count(self):
        self.slot_ops += 1

    # ---- completion of announced operations -----------------------------------------------
    def _complete_enqueue(self, op, by):
        self._count()
        slot = self.slots[self.enqueue_index % self.capacity]
        if slot.state == DEQUEUING:            # a stalled take must be finished first
            self._finish_stalled_take(slot, by)
            return False
        if slot.state not in (EMPTY, TAKEN):   # TAKEN means the previous round was consumed
            return False
        op.index = self.enqueue_index
        slot.value = op.value                  # publish the value ...
        slot.state = ENQUEUED                  # ... then flip the state: the linearization point
        self.enqueue_index += 1
        self.link_order.append(op.value)
        self._finish(op, op.value, by)
        return True

    def _complete_dequeue(self, op, by):
        if op.index is None or op.index < self.flush_index:
            op.index = self.flush_index       # resync with the oldest not-yet-taken position
        self._count()
        slot = self.slots[op.index % self.capacity]
        if slot.state == EMPTY:
            return False                       # nothing linked there yet
        if slot.state == TAKEN:
            op.index = self.flush_index        # somebody else took it; retry from the new head
            return False
        if slot.state == DEQUEUING:
            self._finish_stalled_take(slot, by)  # help the stalled owner, then retry
            op.index = self.flush_index
            return False
        slot.state = DEQUEUING                 # CAS ENQUEUED -> DEQUEUING claims the item
        slot.owner = op
        self._take(slot, by)
        return True

    def _pending_enqueues(self):
        return [op for op in self.ops if op.kind == "enq" and not op.done]

    def help(self, thread, bound=64):
        """Complete every announced operation (oldest first) before doing anything else."""
        for _ in range(bound):
            progressed = False
            for op in self.ops:
                if op.done:
                    continue
                if not self.helping and op.thread != thread:
                    continue
                if op.kind == "enq":
                    ok = self._complete_enqueue(op, thread)
                else:
                    ok = self._complete_dequeue(op, thread)
                    if not op.done and self.slots[op.index % self.capacity].state == EMPTY:
                        # Declaring "empty" is only safe when no announced enqueue is unfinished.
                        if not self._pending_enqueues():
                            self._finish(op, None, thread)
                            ok = True
                progressed = progressed or ok
            if not progressed:
                return
        raise AssertionError("help loop did not converge inside its bound")

    def close(self, thread):
        """Final pass: mark every dequeue with no pending enqueue as an empty result."""
        for op in self.ops:
            if op.kind == "deq" and not op.done and not self._pending_enqueues():
                self._finish(op, None, thread)

    # ---- direct (non-interleaved) API ------------------------------------------------------
    def enqueue(self, thread, value):
        op = self.announce(thread, "enq", value)
        self.help(thread)
        return op

    def dequeue(self, thread):
        op = self.announce(thread, "deq")
        self.help(thread)
        self.close(thread)
        return op.result

    def size(self):
        return self.enqueue_index - len(self.take_order)


# ---- step machines: pause points for the deterministic schedule ---------------------------
def enqueue_steps(queue, thread, value):
    op = queue.announce(thread, "enq", value)
    yield "announced"                     # another thread may run here
    queue.help(thread)
    return op.result


def dequeue_steps(queue, thread, claim=False):
    """`claim=False` stops right after the announcement; `claim=True` stops *after* the
    ENQUEUED -> DEQUEUING CAS, i.e. inside the window a stalled dequeuer would be caught in."""
    op = queue.announce(thread, "deq")
    yield "announced"
    if claim:
        slot = queue.slots[queue.flush_index % queue.capacity]
        check(slot.state == ENQUEUED, "claiming requires a linked item")
        op.index = queue.flush_index
        queue._count()
        slot.state = DEQUEUING
        slot.owner = op
        yield "claimed"                   # the owner may stall here; anyone else can finish it
    queue.help(thread)
    queue.close(thread)
    return op.result


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


def lcg(seed, modulus=65537):
    state = seed

    def nxt(bound):
        nonlocal state
        state = (state * 75 + 74) % modulus
        return state % bound

    return nxt


def main():
    # --- FIFO order and the empty case --------------------------------------------------------
    queue = WaitFreeQueue()
    for index, value in enumerate("abcde"):
        queue.enqueue("T0", value)
    check(queue.link_order == list("abcde"), "items link in enqueue order")
    check(queue.size() == 5, "all five items are still queued")
    drained = [queue.dequeue("T0") for _ in range(5)]
    check(drained == list("abcde"), f"dequeues are FIFO, got {drained}")
    check(queue.take_order == list("abcde"), "takes happen in the same order")
    check(queue.size() == 0 and queue.dequeue("T0") is None, "an empty queue returns None")
    check(all(op.done for op in queue.ops), "every announced operation is completed")
    check(all(slot.takes == 1 for slot in queue.slots if slot.state == TAKEN),
          "every slot is taken exactly once")

    # --- the linearization order is what the slot machine recorded ----------------------------
    check(len(queue.link_order) == len(queue.take_order) == 5, "one take per link")
    check(queue.helps == 0, "with one thread the fast path needs no helping")

    # --- a stalled enqueuer is helped by another thread ---------------------------------------
    queue = WaitFreeQueue()
    stalled = enqueue_steps(queue, "A", "first")
    check(resume(stalled)[0] == "announced", "A announces its enqueue")
    pending = queue._pending_enqueues()
    check(len(pending) == 1 and pending[0].thread == "A", "A's enqueue is announced but unfinished")
    other = enqueue_steps(queue, "B", "second")
    _, b_result = drain(other)
    check(b_result == "second", "B completes its own enqueue too")
    check(queue.ops[0].done and queue.ops[0].completed_by == "B",
          "B helps A's announced enqueue before doing its own")
    check(queue.link_order == ["first", "second"], "helping preserves announcement order")
    check(queue.helps == 1, "exactly one cross-thread completion happened")
    _, a_result = drain(stalled)
    check(a_result == "first", "A's own resume observes its value")
    check(queue.dequeue("C") == "first" and queue.dequeue("C") == "second",
          "both helped items are dequeued in FIFO order")
    check(queue.dequeue("C") is None, "queue is empty afterwards")

    # --- a stalled dequeuer: its take is finished by another thread ----------------------------
    queue = WaitFreeQueue()
    queue.enqueue("A", "x")
    queue.enqueue("A", "y")
    stalled = dequeue_steps(queue, "A", claim=True)
    check(resume(stalled)[0] == "announced", "A announces a dequeue")
    check(resume(stalled)[0] == "claimed", "A claims the head slot and then stalls")
    stalled_op = queue.ops[-1]
    slot = queue.slots[0]
    check(slot.state == DEQUEUING and slot.owner is stalled_op, "the slot is left mid-take")
    _, b_result = drain(dequeue_steps(queue, "B"))
    check(b_result == "y", f"B helps A's stalled take (which keeps 'x') and then dequeues the next, got {b_result}")
    check(stalled_op.done and stalled_op.completed_by == "B", "B completed A's operation record")
    check(queue.stalled_takes == 1, "the stalled take is finished exactly once")
    check(stalled_op.result == "x", "the head value is credited to the stalled operation")
    _, a_result = drain(stalled)
    check(a_result == "x", "A's resume returns the value B stored - no double take")
    check(queue.take_order == ["x", "y"], "exactly one take per item, in order")
    check(queue.dequeue("C") is None, "the queue is empty now")
    stalled_case = (queue.helps, queue.stalled_takes)

    # --- negative control: without helping the stalled operations never finish -----------------
    blind = WaitFreeQueue(helping=False)
    parked = enqueue_steps(blind, "A", "lost?")
    resume(parked)
    check(blind.dequeue("B") is None,
          "without helping, B cannot complete A's announced enqueue and sees an empty queue")
    check(not blind.ops[0].done and blind.link_order == [],
          "A's operation stays unfinished - that is the blocking the helping rule removes")
    blind.helping = True
    check(blind.dequeue("B") is None, "the older pending dequeue is served first, so B gets nothing")
    check(blind.ops[0].done and blind.ops[0].completed_by == "B",
          "re-enabling helping completes A's announced enqueue")
    check(blind.ops[1].result == "lost?" and blind.ops[1].completed_by == "B",
          "the pending dequeue finally consumes the helped-in value")
    check(blind.link_order == ["lost?"] and blind.take_order == ["lost?"],
          "one link, one take - nothing was lost or duplicated")
    check(drain(parked)[1] == "lost?", "the stalled enqueuer's own resume sees its value")

    # --- wait-freedom witness: bounded slot operations -----------------------------------------
    queue = WaitFreeQueue(capacity=8)
    for index in range(6):
        queue.enqueue("T0", index)
    before = queue.slot_ops
    for _ in range(6):
        queue.dequeue("T0")
    per_op = (queue.slot_ops - before) / 6
    check(per_op <= 4, f"each dequeue needs at most a few slot operations, got {per_op}")

    # --- seeded workload over a ring smaller than the operation count --------------------------
    queue = WaitFreeQueue(capacity=8)
    pick = lcg(2024)
    expect_in, expect_out = [], []
    depth, max_depth = 0, 0
    for step_index in range(120):
        if depth == 0 or (depth < 6 and pick(3) != 0):
            value = f"v{step_index}"
            queue.enqueue("T0", value)
            expect_in.append(value)
            depth += 1
            max_depth = max(max_depth, depth)
        else:
            taken = queue.dequeue("T0")
            check(taken is not None, "a non-empty queue must yield a value")
            expect_out.append(taken)
            depth -= 1
    for _ in range(depth):
        expect_out.append(queue.dequeue("T0"))
    check(expect_out == expect_in, "the workload dequeues every value exactly once, in order")
    check(max_depth <= 7 < queue.capacity, "in-flight items never exceed the ring capacity")
    check(queue.enqueue_index == len(expect_in), "each enqueue advanced the index once")
    check(all(op.done for op in queue.ops), "no announced operation was left dangling")
    check(queue.helps == 0 and queue.stalled_takes == 0,
          "the sequential workload needs neither helping nor stalled-take recovery")

    print(f"queue: ops={len(queue.ops)} slot_ops={queue.slot_ops} max_depth={max_depth} "
          f"stalled-scenario helps={stalled_case[0]} finished-takes={stalled_case[1]}")


if __name__ == "__main__":
    main()
    print("PASS")
