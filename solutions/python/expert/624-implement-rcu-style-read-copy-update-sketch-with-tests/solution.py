"""Task 624 - Implement RCU-style read-copy-update sketch with tests.  (Tier: Expert, Ch 15 — Threads and Locks)

Read-copy-update splits every shared structure into an immutable *version* published through one
atomic pointer. A reader loads that pointer once and then reads the frozen version: it never sees a
mixture of two states, and because a multi-key update is published as a single pointer store, a
whole batch of changes appears atomically. A writer copies the current state, mutates the copy,
publishes it, and only then reclaims the version it replaced - not immediately, but after a *grace
period*: every reader that entered its critical section before the publish must have left it.
Readers that enter after the publish can be ignored, because they can only see the new version;
that asymmetry is what makes RCU reclamation cheap on the read side (no writes to shared state).

This is a sketch, not real concurrency: there are no threads. Readers and writers are resumable step
machines (generators) and a deterministic driver interleaves them, so the exact interleaving under
test is written out explicitly. What is genuinely verified: a reader that paused *inside* its
critical section observes a consistent published state and that state is never freed while it is
inside; a writer parked in synchronize_rcu() reclaims nothing; reclamation happens as soon as the
last pre-existing reader leaves, even if newer readers are active; and a naive variant that frees on
publish is caught reading freed memory (the negative control that shows the grace period matters).

Checks: read-your-writes, torn-read prevention across two single-key publishes and across a batch
publish (compared against a plain-dict replay oracle), old/new reader visibility, the blocking
grace period, the "new readers do not block" property, a reader that never leaves, an empty map,
deletes, and a chained run of five writes reclaimed in one batch while the current version survives.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Version:
    """One immutable published state: a sorted tuple of (key, value) pairs plus its epoch."""

    __slots__ = ("epoch", "data", "freed", "owner")

    def __init__(self, epoch, mapping, owner):
        self.epoch = epoch
        self.data = tuple(sorted(mapping.items()))
        self.freed = False
        self.owner = owner

    def read(self, key):
        if self.freed:                      # observable use-after-free (see RcuMap.use_after_free)
            self.owner.use_after_free += 1
        for candidate, value in self.data:
            if candidate == key:
                return value
        return None

    def read_many(self, keys):
        return {key: self.read(key) for key in keys}

    def state(self):
        return dict(self.data)


class RcuMap:
    """Copy-on-write map; one writer at a time, arbitrarily many lock-free readers."""

    def __init__(self, initial=None, free_on_publish=False):
        self.current = None
        self.versions = []
        self.retired = []                   # replaced versions awaiting a grace period
        self.reclaimed = []                 # epochs of freed versions, in free order
        self.active = {}                    # reader id -> token of its critical section
        self.use_after_free = 0
        self.publishes = 0
        self.epoch = 0
        self.free_on_publish = free_on_publish
        self.current = self._publish(initial or {})

    # ---- writer side ----------------------------------------------------------------------
    def _publish(self, mapping):
        self.epoch += 1
        new = Version(self.epoch, mapping, self)
        old = self.current
        self.current = new                # the single atomic pointer store
        self.versions.append(new)
        self.publishes += 1
        if old is not None:
            if self.free_on_publish:      # naive scheme, used as the negative control
                old.freed = True
                self.reclaimed.append(old.epoch)
            else:
                self.retired.append(old)
        return new

    def publish(self, key, value):
        state = dict(self.current.data)
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
        return self._publish(state)

    def publish_many(self, changes):
        state = dict(self.current.data)
        state.update(changes)
        return self._publish(state)

    # ---- reader side ----------------------------------------------------------------------
    def enter(self, rid):
        token = len(self.versions) * 1000 + len(self.active) + 1
        self.active[rid] = token
        return token

    def leave(self, rid, token):
        if self.active.get(rid) == token:
            del self.active[rid]
            return True
        return False

    # ---- grace period ---------------------------------------------------------------------
    def sync_snapshot(self):
        """What synchronize_rcu() waits for: the readers in flight at the moment of the call."""
        return dict(self.active)

    def elapsed(self, snapshot):
        """A grace period has elapsed once every reader active at snapshot time has left.
        Readers that entered afterwards do not appear in the snapshot, so they cannot delay it."""
        return all(self.active.get(rid) != token for rid, token in snapshot.items())

    def reclaim(self, snapshot):
        """Free every retired version if the grace period has elapsed, else stay blocked."""
        if not self.elapsed(snapshot):
            return None
        freed = []
        for version in self.retired:
            version.freed = True
            self.reclaimed.append(version.epoch)
            freed.append(version.epoch)
        self.retired = []
        return freed


def reader_steps(rcu, rid, key=None, keys=None):
    """Read-side critical section as a step machine.

    `enter` precedes the pointer load - the ordering that makes a writer's grace period provably
    cover this read. The loaded version is then read without touching any shared mutable state.
    """
    token = rcu.enter(rid)
    version = rcu.current                  # single pointer load
    yield "entered"
    observed = version.read(key) if keys is None else version.read_many(keys)
    yield "observed"
    rcu.leave(rid, token)
    return version.epoch, observed


def writer_steps(rcu, changes):
    """Writer: copy-modify-publish, then synchronize_rcu() (a grace period) and reclaim."""
    rcu.publish_many(changes)
    yield "published"
    snapshot = rcu.sync_snapshot()
    yield "synchronizing"
    while not rcu.elapsed(snapshot):
        yield "blocked"                    # parked inside synchronize_rcu()
    rcu.reclaim(snapshot)
    yield "reclaimed"
    return [key for key, _ in rcu.current.data]


def resume(gen):
    """Advance a step machine once; returns (step or None, finished, return_value)."""
    try:
        return next(gen), False, None
    except StopIteration as stop:
        return None, True, stop.value


def drain(gen):
    """Run a step machine to completion, returning (steps, return value)."""
    steps = []
    while True:
        step, finished, value = resume(gen)
        if finished:
            return steps, value
        steps.append(step)


def replay_oracle(writes, initial=None):
    """Independent oracle: the states a plain dict passes through when the writes are applied."""
    state = dict(initial or {})
    states = [dict(state)]
    for changes in writes:
        state = dict(state)
        state.update(changes)
        states.append(dict(state))
    return states


def main():
    # --- read-your-writes and simple visibility ----------------------------------------------
    rcu = RcuMap({"a": 1, "b": 2})
    check(rcu.epoch == 1 and rcu.publishes == 1 and rcu.versions[0].state() == {"a": 1, "b": 2},
          "the initial state is itself a published version")
    _, (_, value) = drain(reader_steps(rcu, "r0", key="a"))
    check(value == 1, "reader sees the current version")
    rcu.publish("a", 5)
    check(rcu.current.read("a") == 5, "the writer reads its own write")
    _, (epoch, value) = drain(reader_steps(rcu, "r1", key="a"))
    check(value == 5 and epoch == 2, "a later reader sees the new version")
    check([v.epoch for v in rcu.retired] == [1], "the replaced version is retired, not freed")
    check(rcu.versions[1].freed is False, "nothing is reclaimed before a grace period")

    # --- torn-read prevention across two separate publishes ----------------------------------
    rcu = RcuMap({"a": 1, "b": 2})
    reader = reader_steps(rcu, "r2", keys=("a", "b"))
    step, finished, _ = resume(reader)
    check(step == "entered" and not finished, "reader parks inside its critical section")
    rcu.publish("a", 99)
    rcu.publish("b", 100)
    _, (epoch, observed) = drain(reader)
    check(observed == {"a": 1, "b": 2} and epoch == 1,
          f"paused reader keeps its consistent snapshot, got {observed}")
    states = [v.state() for v in rcu.versions]
    check(observed in states, "every observation is one of the published states")

    # Negative control: a reader that reloads the live pointer for every key can see a mixture.
    live = RcuMap({"a": 1, "b": 2})
    torn = {"a": live.current.read("a")}    # first key from the old version ...
    live.publish("a", 99)
    live.publish("b", 100)
    torn["b"] = live.current.read("b")      # ... second key from the new one
    check(torn == {"a": 1, "b": 100} and torn not in [v.state() for v in live.versions],
          "reloading the pointer per key can expose a state that was never published")

    # --- atomic batch update ------------------------------------------------------------------
    rcu = RcuMap({"x": 0, "y": 0})
    paused = reader_steps(rcu, "r3", keys=("x", "y"))
    resume(paused)
    rcu.publish_many({"x": 3, "y": 4})      # one pointer store -> both keys or neither
    _, (_, batch) = drain(paused)
    check(batch == {"x": 0, "y": 0}, "batch update is invisible to a reader already inside")
    _, (_, fresh) = drain(reader_steps(rcu, "r4", keys=("x", "y")))
    check(fresh == {"x": 3, "y": 4}, "a reader arriving after the batch sees it in full")

    # --- grace period: the writer stays blocked until pre-existing readers leave --------------
    rcu = RcuMap({"x": 1})
    inside = reader_steps(rcu, "r5", key="x")
    resume(inside)
    writer = writer_steps(rcu, {"x": 2})
    step, finished, _ = resume(writer)
    check(step == "published" and not finished, "writer publishes first")
    step, finished, _ = resume(writer)
    check(step == "synchronizing", "writer then enters synchronize_rcu()")
    for _ in range(3):
        step, finished, _ = resume(writer)
        check(step == "blocked" and not finished, "writer cannot reclaim while r5 is inside")
    check(rcu.retired and rcu.retired[0].epoch == 1 and rcu.reclaimed == [],
          "the retired version survives the parked writer")
    _, (epoch, value) = drain(inside)       # the reader was reading the retired version
    check(value == 1 and epoch == 1, "the reader reads the version it snapshotted")
    check(rcu.use_after_free == 0, "the grace period kept that version alive")
    steps, final = drain(writer)
    check("reclaimed" in steps and final == ["x"], "writer finishes once the grace period elapses")
    check(rcu.reclaimed == [1] and rcu.current.state() == {"x": 2},
          "the replaced version is freed, the current one is not")

    # --- new readers do not extend the grace period -------------------------------------------
    rcu = RcuMap({"k": 1})
    old_reader = reader_steps(rcu, "r6", key="k")
    resume(old_reader)
    rcu.publish("k", 2)
    snapshot = rcu.sync_snapshot()
    check(not rcu.elapsed(snapshot), "the pre-existing reader still blocks reclamation")
    drain(old_reader)
    newcomer = reader_steps(rcu, "r7", key="k")
    resume(newcomer)                        # enters *after* synchronize_rcu() was called
    check(rcu.elapsed(snapshot), "a newer reader cannot delay the grace period")
    check(rcu.reclaim(snapshot) == [1], "reclamation proceeds with a newer reader inside")
    _, (epoch, value) = drain(newcomer)
    check(value == 2 and epoch == 2, "the newer reader was on the new version all along")
    check(rcu.use_after_free == 0, "no reader ever touched the reclaimed version")

    # --- a reader that never leaves parks the writer forever ----------------------------------
    rcu = RcuMap({"z": 0})
    stuck = reader_steps(rcu, "r8", key="z")
    resume(stuck)
    writer = writer_steps(rcu, {"z": 1})
    drain_partial = [resume(writer)[0] for _ in range(4)]
    check(drain_partial[:2] == ["published", "synchronizing"] and set(drain_partial[2:]) == {"blocked"},
          "the writer never gets past the blocked state")
    check(rcu.reclaimed == [] and rcu.retired[0].freed is False,
          "nothing is reclaimed while the reader is stuck")

    # --- chained writes reclaimed in one batch ------------------------------------------------
    rcu = RcuMap({"n": 0})
    long_reader = reader_steps(rcu, "r9", key="n")
    resume(long_reader)
    for step_value in range(1, 6):
        rcu.publish("n", step_value)
    check(rcu.publishes == 6 and len(rcu.versions) == 6, "one version per publish plus the initial")
    check([v.epoch for v in rcu.retired] == [1, 2, 3, 4, 5], "all five replaced versions are retired")
    check([v.epoch for v in rcu.versions] == list(range(1, 7)), "epochs are strictly increasing")
    snapshot = rcu.sync_snapshot()
    check(rcu.reclaim(snapshot) is None, "still blocked by the long reader")
    _, (epoch, value) = drain(long_reader)
    check(value == 0 and epoch == 1, "the long reader saw the very first version")
    check(rcu.reclaim(snapshot) == [1, 2, 3, 4, 5], "all five old versions are freed in one batch")
    chained_versions, chained_reclaimed = len(rcu.versions), list(rcu.reclaimed)
    check(rcu.current.freed is False and rcu.use_after_free == 0,
          "the live version is never freed")
    _, (_, latest) = drain(reader_steps(rcu, "r10", key="n"))
    check(latest == 5, "readers still observe the newest state")

    # --- deletes, empty map, and the replay oracle ---------------------------------------------
    rcu = RcuMap()
    check(rcu.current.state() == {} and rcu.current.read("missing") is None,
          "an empty map has one empty version")
    rcu.publish("only", 1)
    check(rcu.current.state() == {"only": 1}, "insert into an empty map")
    rcu.publish("only", None)
    check(rcu.current.state() == {} and rcu.current.read("only") is None, "publish(value=None) deletes")
    snapshot = rcu.sync_snapshot()
    check(rcu.elapsed(snapshot) and rcu.reclaim(snapshot) == [1, 2],
          "with no readers the grace period is already over")

    writes = [{"a": 1}, {"b": 2}, {"a": 3}]
    rcu = RcuMap({"seed": 0})
    for changes in writes:
        rcu.publish_many(changes)
    expected = replay_oracle(writes, {"seed": 0})
    check([v.state() for v in rcu.versions] == expected,
          "published states match a plain-dict replay of the same writes")

    # --- negative control: free-on-publish is caught reading freed memory ----------------------
    naive = RcuMap({"a": 1}, free_on_publish=True)
    torn = reader_steps(naive, "r11", keys=("a",))
    resume(torn)
    naive.publish("a", 2)
    check(naive.reclaimed == [1] and naive.versions[0].freed, "naive writer frees immediately")
    drain(torn)
    check(naive.use_after_free >= 1,
          "free-on-publish is caught dereferencing a freed version - the grace period prevents this")

    print(f"rcu: chained versions={chained_versions} reclaimed={chained_reclaimed} "
          f"use-after-free(rcu)={rcu.use_after_free} "
          f"use-after-free(naive)={naive.use_after_free}")


if __name__ == "__main__":
    main()
    print("PASS")
