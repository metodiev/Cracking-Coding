"""Task 621 - Implement scalable concurrent queue (Michael-Scott queue).  (Tier: Expert, CTCI Ch 15 — Threads and Locks)

The Michael-Scott (1996) lock-free queue, advanced by an *explicit schedule* instead of real
threads: one scheduled entry advances one thread by exactly one step of its current operation, a
step ending at every volatile read and every compare-and-swap.  The schedule is plain data, so any
interleaving - including the ones that force the helping path and forced CAS failures - is
reproducible, and there are no threads, sleeps or sockets anywhere.

The algorithm is the real one, not a lock-based shortcut:
  * the queue always holds a dummy node; head.next is the first element and tail points at the last
    node, or at the one before it while an enqueue has linked its node but not yet moved the tail;
  * enqueue links with CAS(tail.next: None -> node) and then CASes tail forward, ignoring failure
    because another thread may already have advanced it;
  * dequeue re-reads head *after* reading tail and head.next and only then CASes head forward, so a
    concurrent enqueue cannot be missed; when head == tail but tail.next is not None, the dequeuer
    *helps* by CASing the tail forward and retries;
  * compare-and-swap is simulated faithfully: it succeeds exactly when the field still holds the
    expected value at the moment the step runs, so failures are produced by the interleaving
    itself instead of being scripted.

Verified: a sequential schedule gives exact FIFO and non-blocking dequeue of an empty queue
returns None; across 40 seeded schedules every enqueued value is claimed at most once, the claim
order is a subsequence of the link order (no reordering, no duplication), the values left in the
list are exactly the linked-but-unclaimed ones in link order, the list is acyclic and quiescent,
and the consumers together observe every value; hand-forced interleavings exercise the lagging
tail, the helping CAS and a CAS failure, all leaving the queue consistent.
"""


class Node:
    __slots__ = ("value", "next")

    def __init__(self, value=None):
        self.value = value
        self.next = None


class MichaelScottQueue:
    def __init__(self):
        dummy = Node(None)
        self.head = dummy
        self.tail = dummy
        self.events = []                 # ("link"/"claim", value), in global step order
        self.cas_attempts = 0
        self.cas_failures = 0
        self.reads = 0
        self.helping = 0
        self.spurious = 0
        self.enqueued = 0
        self.dequeued = 0

    # -- the one indivisible primitive -------------------------------------
    def cas(self, holder, name, expected, new_value):
        self.cas_attempts += 1
        if getattr(holder, name) is expected:
            setattr(holder, name, new_value)
            return True
        self.cas_failures += 1
        return False

    # -- operations as step machines ---------------------------------------
    def enqueue(self, value):
        node = Node(value)
        while True:
            tail = self.tail
            self.reads += 1
            yield "read tail"
            nxt = tail.next
            self.reads += 1
            yield "read next"
            if tail is self.tail:
                if nxt is None:
                    if self.cas(tail, "next", None, node):
                        self.events.append(("link", value))
                        yield "linked"                 # the tail lags behind on purpose here
                        break
                    yield "cas next failed"
                else:
                    if self.cas(self, "tail", tail, nxt):
                        self.helping += 1
                    yield "help tail"
            else:
                self.spurious += 1
        self.cas(self, "tail", tail, node)              # failure is fine: someone helped us
        self.enqueued += 1
        yield "link done"
        return None

    def dequeue(self):
        while True:
            head = self.head
            self.reads += 1
            yield "read head"
            tail = self.tail
            self.reads += 1
            yield "read tail"
            nxt = head.next
            self.reads += 1
            yield "read head.next"
            if head is self.head:
                if head is tail:
                    if nxt is None:
                        yield "empty"
                        return None
                    if self.cas(self, "tail", tail, nxt):
                        self.helping += 1
                    yield "help tail"
                else:
                    value = nxt.value
                    yield "read value"
                    if self.cas(self, "head", head, nxt):
                        self.events.append(("claim", value))
                        self.dequeued += 1
                        yield "claim done"
                        return value
                    yield "cas head failed"
            else:
                self.spurious += 1

    # -- structural integrity ----------------------------------------------
    def values(self):
        out, node, seen = [], self.head.next, set()
        while node is not None:
            if id(node) in seen:
                raise AssertionError("the list has a cycle")
            seen.add(id(node))
            out.append(node.value)
            node = node.next
        return out

    def is_quiescent(self):
        node, guard = self.head, 0
        while node is not self.tail:
            node = node.next
            guard += 1
            if node is None or guard > 10000:
                return False
        return True


class Thread:
    def __init__(self, name, operations, queue):
        self.name = name
        self.queue = queue
        self.operations = list(operations)
        self.index = 0
        self.current = None
        self.results = []
        self.finished = False
        self.steps = 0

    def advance(self):
        if self.finished:
            return "idle"
        if self.current is None:
            if self.index >= len(self.operations):
                self.finished = True
                return "idle"
            kind = self.operations[self.index][0]
            self.current = (self.queue.enqueue(self.operations[self.index][1])
                            if kind == "enq" else self.queue.dequeue())
        self.steps += 1
        try:
            return next(self.current)
        except StopIteration as done:
            self.results.append(done.value)
            self.index += 1
            self.current = None
            if self.index >= len(self.operations):
                self.finished = True
                return "finished"
            return "completed"

    def drain(self, limit=200):
        """Advance this thread until its operations are all done (or the limit is hit)."""
        outcomes = []
        for _ in range(limit):
            if self.finished:
                break
            outcomes.append(self.advance())
        return outcomes


def run(queue, programs, schedule, settle_limit=4000):
    threads = [Thread("t%d" % index, program, queue)
               for index, program in enumerate(programs)]
    for name in schedule:
        index = int(name)
        if index >= len(threads):
            raise IndexError("schedule names an unknown thread: %s" % name)
        threads[index].advance()
    for _ in range(settle_limit):
        if all(thread.finished for thread in threads):
            break
        for thread in threads:
            if not thread.finished:
                thread.advance()
    return threads


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def verify_history(queue):
    """Claims follow link order, never duplicate, and the leftovers are the unclaimed ones."""
    linked = [value for kind, value in queue.events if kind == "link"]
    claimed = [value for kind, value in queue.events if kind == "claim"]
    check(len(linked) == len(set(linked)), "a value was linked twice")
    check(len(claimed) == len(set(claimed)), "a value was dequeued twice")
    walker = 0
    for value in claimed:
        while walker < len(linked) and linked[walker] != value:
            walker += 1
        check(walker < len(linked), "a value was claimed before it was linked: %r" % (value,))
        walker += 1
    check(sorted(queue.values()) == sorted(set(linked) - set(claimed)),
          "the remaining values must be exactly the linked-but-unclaimed ones: %s vs %s"
          % (queue.values(), sorted(set(linked) - set(claimed))))
    check(queue.is_quiescent(),
          "after settling, the tail must be reachable from head with next == None")


def main():
    # ---- sequential FIFO and empty dequeue --------------------------------
    queue = MichaelScottQueue()
    threads = run(queue, [[("enq", value) for value in range(1, 6)]
                          + [("deq",) for _ in range(5)], []], ["0"] * 200)
    check(threads[0].results == [None] * 5 + [1, 2, 3, 4, 5],
          "a sequential schedule must give exact FIFO: %s" % threads[0].results)
    check(queue.values() == [] and queue.is_quiescent(), "the queue is empty and settled")
    check(queue.enqueued == 5 and queue.dequeued == 5, "five of each")
    check(queue.cas_failures == 0 and queue.helping == 0,
          "a single thread never loses a CAS and never has to help")
    verify_history(queue)

    empty = MichaelScottQueue()
    threads = run(empty, [[("deq",), ("deq",), ("enq", 9), ("deq",)], []], ["0"] * 60)
    check(threads[0].results == [None, None, None, 9],
          "dequeueing an empty queue returns None without blocking: %s" % threads[0].results)
    check(empty.dequeued == 1, "only the successful dequeue is counted")

    # ---- interleaved producers and consumers --------------------------------
    programs = [
        [("enq", value) for value in (1, 3, 5, 7)],
        [("enq", value) for value in (2, 4, 6, 8)],
        [("deq",) for _ in range(8)],
        [("deq",) for _ in range(8)],
    ]
    state = 2025
    failures_seen = 0
    helping_seen = 0
    empty_seen = 0
    # the first schedule is hand-crafted so that a dequeuer sees head == tail with a freshly
    # linked node (both threads read the dummy, then the producer links before the consumer
    # reads head.next), which is exactly the case that forces the helping CAS
    crafted = ["0", "0", "2", "2", "0", "2", "2"]
    schedules = [crafted]
    for _ in range(40):
        schedule = []
        for _ in range(140):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            schedule.append(str(state % 4))
        schedules.append(schedule)
    for seed, schedule in enumerate(schedules):
        queue = MichaelScottQueue()
        threads = run(queue, programs, schedule)
        check(all(thread.finished for thread in threads),
              "schedule %d left a thread running" % seed)
        check(queue.enqueued == 8, "every enqueue linked its node (schedule %d)" % seed)
        check(queue.dequeued == 8, "every value was dequeued exactly once (schedule %d)" % seed)
        check(queue.values() == [], "the queue must be empty at the end (schedule %d)" % seed)
        verify_history(queue)
        observed = [value for thread in threads[2:] for value in thread.results
                    if value is not None]
        check(sorted(observed) == list(range(1, 9)),
              "the consumers together must observe every value: %s" % sorted(observed))
        empty_seen += sum(1 for thread in threads[2:] for value in thread.results
                          if value is None)
        if queue.cas_failures:
            failures_seen += 1
        if queue.helping:
            helping_seen += 1
    check(failures_seen > 0, "some schedule must lose a CAS")
    check(helping_seen > 0, "some schedule must need to advance the tail")
    check(empty_seen > 0, "some schedule must observe a momentarily empty queue")

    # ---- forced lagging tail: linked node, tail still on the dummy ----------
    queue = MichaelScottQueue()
    producer = Thread("producer", [("enq", 1)], queue)
    consumer = Thread("consumer", [("deq",)], queue)
    producer.advance()                       # read tail
    producer.advance()                       # read tail.next (None)
    check(producer.advance() == "linked", "the node is linked")
    check(queue.values() == [1], "the node is in the list")
    check(queue.tail is queue.head, "the tail still lags on the dummy node")
    consumer.drain()
    check(consumer.results == [1], "the consumer claimed the value: %s" % consumer.results)
    check(queue.helping >= 1, "the consumer had to help the tail forward: %d" % queue.helping)
    producer.drain()
    check(producer.finished and queue.is_quiescent(),
          "the queue settles even though the producer's tail CAS lost the race")
    check(queue.cas_failures >= 1, "the helper moved the tail first, so the producer's CAS lost")
    verify_history(queue)

    # ---- forced helping: the dequeuer meets head == tail with a fresh node --
    queue = MichaelScottQueue()
    producer = Thread("producer", [("enq", 7)], queue)
    helper = Thread("helper", [("deq",)], queue)
    producer.advance()                       # read tail
    producer.advance()                       # read tail.next (None)
    helper.advance()                         # read head (the dummy)
    helper.advance()                         # read tail (still the dummy)
    producer.advance()                       # link the node; the tail still lags
    outcomes = helper.drain()
    check("help tail" in outcomes, "the dequeuer must help advance the tail: %s" % outcomes)
    check(queue.helping == 1, "the helping CAS is counted exactly once")
    check(helper.results == [7], "the helped dequeue returns the value: %s" % helper.results)
    producer.drain()
    verify_history(queue)

    # ---- forced CAS failure: both threads saw next == None ----------------
    queue = MichaelScottQueue()
    first = Thread("first", [("enq", 10)], queue)
    second = Thread("second", [("enq", 20)], queue)
    first.advance()                          # read tail
    first.advance()                          # read tail.next (None)
    second.advance()                         # read tail (the same dummy)
    second.advance()                         # read tail.next (still None)
    second.advance()                         # second wins the race and links its node
    before = queue.cas_failures
    outcome = first.advance()                # first's CAS on the same next pointer fails
    check(queue.cas_failures == before + 1,
          "the losing CAS must fail (outcome %s)" % outcome)
    check(queue.values() == [20], "only the winner is linked so far")
    first.drain()
    second.drain()
    check(queue.values() == [20, 10],
          "the loser retried and linked behind the winner: %s" % queue.values())
    check(queue.cas_attempts > queue.cas_failures, "most CAS attempts succeed")
    verify_history(queue)

    # ---- rejected schedule entries -------------------------------------------
    raised = False
    try:
        run(MichaelScottQueue(), [[("deq",)]], ["9"])
    except IndexError:
        raised = True
    check(raised, "a schedule naming an unknown thread must raise")

    print("sequential FIFO: 5 values in order, empty dequeue returns None, no CAS failures")
    print("seeded sweep: 40 schedules, %d with CAS failures, %d needing tail help, %d empties"
          % (failures_seen, helping_seen, empty_seen))
    print("forced interleavings: lagging tail, helping CAS and a CAS failure all consistent")
    print("PASS")


if __name__ == "__main__":
    main()
