"""Task 623 - Implement hazard pointers memory reclamation in concurrent structures.  (Tier: Expert, Ch 15 — Threads and Locks)

A lock-free LIFO stack (Michael-Scott style) cannot free a popped node immediately: another thread
may have read the same head pointer just before the pop and still be about to dereference it. Hazard
pointers solve this by publishing, per thread, the node that thread is currently reading. A retiring
thread appends the node to a private retire list; when the list hits a threshold, `scan` reads every
published hazard and frees exactly the retired nodes that nobody protects. Because a thread publishes
its hazard *before* dereferencing a pointer, and clears it *after* it is done, a protected node can
never be freed under a reader's feet.

Simplifications (this is a faithful simulation, not real concurrency): there are no threads - the
schedule is the explicit list of steps the driver takes, and the pop operation is written as a
resumable step machine (a generator) so the driver can pause it at the two points where another
thread could interfere. Freeing is modelled by a `freed` flag plus a dereference checker instead of
real deallocation, because Python cannot observe a dangling pointer. The invariant that is still
genuinely verified: *a node returned to (or being read by) a hazard-protected thread is never freed
while that hazard is published*, and the same interleaving driven against immediate reclamation is
shown to dereference freed memory.

Checks: LIFO order and empty-stack behaviour; retire thresholds; scans free exactly the unprotected
subset (single-thread, multi-thread, permanent-hazard and empty cases); the adversarial interleaving
under both reclamation policies; and a seeded conservative workload asserting node conservation
(created == live + retired + freed) with zero use-after-free.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Node:
    __slots__ = ("key", "nxt", "freed")

    def __init__(self, key, nxt=None):
        self.key = key
        self.nxt = nxt
        self.freed = False

    def __repr__(self):
        return f"Node({self.key})"


class HazardDomain:
    """Hazard-pointer domain: per-thread protection slots plus a retire list and scan."""

    def __init__(self, threads, scan_threshold=1, protect=True):
        self.hazard = {thread: None for thread in threads}
        self.retired = []
        self.freed = []               # keys, in the order they were handed back
        self.scan_threshold = scan_threshold
        self.protect = protect        # False models naive immediate reclamation
        self.scans = 0
        self.use_after_free = 0       # dereferences of already-freed memory

    def set_hazard(self, thread, node):
        self.hazard[thread] = node

    def clear_hazard(self, thread):
        self.hazard[thread] = None

    def protected_ids(self):
        if not self.protect:
            return set()
        return {id(node) for node in self.hazard.values() if node is not None}

    def dereference(self, node, field):
        """Every pointer read goes through here so a use-after-free is observable."""
        if node.freed:
            self.use_after_free += 1
        return getattr(node, field)

    def retire(self, node):
        if not self.protect:          # naive scheme: hand the memory back at once
            node.freed = True
            self.freed.append(node.key)
            return
        self.retired.append(node)
        if len(self.retired) >= self.scan_threshold:
            self.scan()

    def scan(self):
        """Free every retired node that no thread has published as hazardous."""
        self.scans += 1
        protected = self.protected_ids()
        survivors = []
        for node in self.retired:
            if id(node) in protected:
                survivors.append(node)
            else:
                node.freed = True
                self.freed.append(node.key)
        self.retired = survivors
        return list(self.freed)


class LockFreeStack:
    """Singly linked LIFO stack; `head` plays the role of the single atomic pointer."""

    def __init__(self):
        self.head = None
        self.created = 0

    def push(self, key):
        node = Node(key, self.head)
        self.head = node
        self.created += 1
        return node

    def live_keys(self):
        keys, cur = [], self.head
        while cur is not None:
            keys.append(cur.key)
            cur = cur.nxt
        return keys


def pop_steps(thread, stack, domain):
    """Resumable lock-free pop.

    Yields after publishing the hazard and after reading the successor field - the two windows in
    which an interfering thread can pop the same node. Returns the popped key, or None if empty.
    """
    while True:
        head = stack.head
        if head is None:
            return None
        domain.set_hazard(thread, head)
        yield "hazard_set"
        # The successor is read *before* validating the snapshot: that is exactly the window in
        # which an interfering pop can free `head`, and the reason the hazard must be published.
        nxt = domain.dereference(head, "nxt")
        yield "read_next"
        if stack.head is not head:
            domain.clear_hazard(thread)
            continue
        stack.head = nxt
        domain.clear_hazard(thread)
        domain.retire(head)
        yield "retired"
        return head.key


def drain(gen):
    """Run a step machine to completion, returning (steps, return value)."""
    steps = []
    while True:
        try:
            steps.append(next(gen))
        except StopIteration as stop:
            return steps, stop.value


def pop(thread, stack, domain):
    return drain(pop_steps(thread, stack, domain))[1]


def adversarial_run(protect, threshold=1):
    """Reader publishes a hazard on the head; the interfering thread then pops and retires that
    very node before the reader reads its successor field."""
    stack = LockFreeStack()
    for key in ("c", "b", "a"):                # head -> a -> b -> c
        stack.push(key)
    domain = HazardDomain((0, 1), scan_threshold=threshold, protect=protect)
    reader = pop_steps(0, stack, domain)
    watched = stack.head
    paused = []
    while True:
        step = next(reader)
        paused.append(step)
        if step == "hazard_set":
            break
    check(domain.hazard[0] is watched, "reader must publish the head as its hazard")

    _, stolen = drain(pop_steps(1, stack, domain))       # the interferer takes the same node
    starved = stack.head is not watched                  # ... so the reader's snapshot is stale
    freed_during_protection = list(domain.freed)
    protected_alive = not watched.freed

    _, result = drain(reader)
    return {
        "paused": paused,
        "stolen": stolen,
        "result": result,
        "starved": starved,
        "freed_during_protection": freed_during_protection,
        "protected_alive": protected_alive,
        "use_after_free": domain.use_after_free,
        "freed_later": list(domain.freed),
        "watched_freed_at_end": watched.freed,
    }


def lcg(seed, modulus=65537):
    """Deterministic (Park-Miller style) generator so the workload is reproducible."""
    state = seed

    def nxt(bound):
        nonlocal state
        state = (state * 75 + 74) % modulus
        return state % bound

    return nxt


def seeded_workload(steps=300, seed=12345):
    """Interleave pushes and pops from two threads; every pop uses hazard pointers."""
    stack = LockFreeStack()
    domain = HazardDomain((0, 1), scan_threshold=3)
    pick = lcg(seed)
    popped, next_key = [], 0
    for _ in range(steps):
        thread = pick(2)
        if pick(3) == 0 or stack.head is None:
            stack.push(next_key)               # unique keys, so a double pop is visible
            next_key += 1
        else:
            key = pop(thread, stack, domain)
            if key is not None:
                popped.append(key)
    for _ in range(steps):                      # drain everything left, then a final scan
        if stack.head is None:
            break
        key = pop(0, stack, domain)
        if key is not None:
            popped.append(key)
    while domain.retired:
        domain.scan()
    return stack, domain, popped


def main():
    # --- plain stack behaviour -------------------------------------------------------------
    stack = LockFreeStack()
    domain = HazardDomain((0,), scan_threshold=8)
    for key in range(5):
        stack.push(key)
    check(stack.live_keys() == [4, 3, 2, 1, 0], "push order must give LIFO layout")
    drained = [pop(0, stack, domain) for _ in range(5)]
    check(drained == [4, 3, 2, 1, 0], "pop must be LIFO")
    check(stack.head is None and pop(0, stack, domain) is None, "empty pop returns None")
    check(len(domain.retired) == 5 and domain.freed == [], "5 retires do not trip a threshold of 8")
    domain.scan()
    check(domain.retired == [] and domain.scans == 1, "an explicit scan drains the retire list")
    check(sorted(domain.freed) == [0, 1, 2, 3, 4], "each node freed exactly once")

    # --- retire threshold ------------------------------------------------------------------
    stack, domain = LockFreeStack(), HazardDomain((0,), scan_threshold=4)
    for key in range(3):
        stack.push(key)
    for _ in range(3):
        pop(0, stack, domain)
    check(domain.scans == 0 and len(domain.retired) == 3 and domain.freed == [],
          "below the threshold nothing is reclaimed")
    stack.push(99)
    pop(0, stack, domain)                       # fourth retire trips the threshold
    check(domain.scans == 1 and len(domain.retired) == 0, "threshold scan drains the retire list")

    # --- a scan frees exactly the unprotected subset ------------------------------------------
    nodes = [Node(k) for k in range(5)]
    domain = HazardDomain((0, 1, 2), scan_threshold=99)
    for node in nodes:
        domain.retire(node)
    check(domain.scans == 0 and len(domain.retired) == 5, "no scan without a threshold trip")
    domain.set_hazard(0, nodes[1])
    domain.set_hazard(1, nodes[3])
    domain.scan()
    check([n.key for n in domain.retired] == [1, 3], "scan keeps only protected nodes")
    check(sorted(domain.freed) == [0, 2, 4], "scan frees exactly the unprotected nodes")
    check(not nodes[1].freed and not nodes[3].freed, "protected nodes stay alive")

    # permanently published hazard: repeated scans still never free the node
    for _ in range(5):
        domain.scan()
    check([n.key for n in domain.retired] == [1, 3] and domain.freed == [0, 2, 4],
          "a permanently protected node is never reclaimed")
    domain.clear_hazard(1)
    domain.clear_hazard(0)
    check(domain.scan() == [0, 2, 4, 1, 3],
          "nodes are only freed after the last hazard is cleared")
    check(domain.retired == [] and nodes[1].freed and nodes[3].freed, "retire list empties")

    # --- empty / degenerate ------------------------------------------------------------------
    empty = HazardDomain((0, 3), scan_threshold=1)
    check(empty.scan() == [] and empty.retired == [] and empty.scans == 1,
          "scanning an empty retire list is a no-op")

    # --- adversarial interleaving: hazard pointers vs immediate reclamation -------------------
    safe = adversarial_run(protect=True, threshold=1)
    check(safe["paused"] == ["hazard_set"], "reader parks right after publishing the hazard")
    check(safe["stolen"] == "a" and safe["starved"], "interferer steals the protected head")
    check(safe["protected_alive"] and safe["freed_during_protection"] == [],
          "hazard pointer keeps the node alive through the interferer's scan")
    check(safe["use_after_free"] == 0, "no dereference of freed memory")
    check(safe["result"] == "b", "reader retries and pops the new head instead")
    check(safe["freed_later"] == ["a", "b"] and safe["watched_freed_at_end"],
          "the node is reclaimed once the reader cleared its hazard")
    check(safe["stolen"] != safe["result"], "the two threads never return the same node")

    naive = adversarial_run(protect=False, threshold=1)
    check(naive["freed_during_protection"] == ["a"] and not naive["protected_alive"],
          "immediate reclamation frees the node the reader still points at")
    check(naive["use_after_free"] >= 1,
          "naive reclamation is caught dereferencing freed memory - hazards are what prevent this")

    # --- seeded workload: conservation and safety --------------------------------------------
    stack, domain, popped = seeded_workload()
    check(len(popped) == len(set(popped)), "no key is returned twice")
    live = len(stack.live_keys())
    check(stack.created == live + len(domain.retired) + len(domain.freed),
          f"conservation: {stack.created} created == {live} live + {len(domain.retired)} retired "
          f"+ {len(domain.freed)} freed")
    check(len(popped) + live == stack.created, "every node was popped exactly once or is still live")
    check(domain.use_after_free == 0 and domain.retired == [],
          "hazard-protected workload never touches freed memory")
    check(domain.scans > 1 and len(domain.freed) > 0, "the workload really reclaimed nodes")
    check(len(popped) > 50, "workload exercised the pop path")

    print(f"nodes created={stack.created} popped={len(popped)} live={live} freed={len(domain.freed)}")
    print(f"scans={domain.scans} use-after-free(hazard)={domain.use_after_free} "
          f"use-after-free(naive)={naive['use_after_free']}")


if __name__ == "__main__":
    main()
    print("PASS")
