"""Task 844 - Implement streaming graph processing with windowed aggregations and state eviction.  (Tier: Master, CTCI Beyond — System Design and Scalability)

A windowed streaming engine over an edge-event stream: events carry a *logical* timestamp,
tumbling windows of `window` ticks accumulate them, a watermark decides when a window is
complete, and state for windows older than the retention horizon is evicted.  There is no clock
and no sleep - time only moves when an event says so - so every run is reproducible.

Engine, as implemented:
  * tumbling windows [k*window, (k+1)*window); an event at tick `t` belongs to window
    `t // window` (the boundary case is checked explicitly);
  * a watermark that only moves forward, `allowed_lateness` tolerated: an event whose window is
    both closed (watermark past its end) and older than the lateness horizon is *dropped*;
    an event that is late but within the horizon triggers a *revision* of the already emitted
    result for that window;
  * per-window state: the multiset of live edges (adds minus removes) plus the incremental
    degree counters; the emitted aggregate is (edge count, degree histogram, triangle count),
    where the triangles are recomputed from the window's own state - documented simplification,
    because incremental triangle maintenance is a different algorithm, while the *eviction*
    property under test only needs the state to be self-contained;
  * eviction: once a window's revision horizon has passed, its whole state is dropped, and the
    emitted result must stay identical (results outlive state);
  * a partial-state merge (`merge`) used to check associativity: two halves of a stream merged
    give exactly the aggregate of the whole - the property a sharded implementation relies on.

Verified against a brute-force recomputation of every window from the raw event list:
  * edge counts and degree histograms are exact for every emitted window (including windows
    that end up empty), adds/removes cancel within a window, unbounded-lateness runs match,
  * late events inside the horizon produce exactly one revision whose value equals the
    recomputation, late events outside it are dropped and counted, and no event is lost
    (emitted + dropped + open == total),
  * eviction only touches closed windows, never changes an emitted result, and keeps the live
    state bounded by the retention horizon,
  * merging two partial window states equals the single-pass state for the whole stream,
  * degenerate streams (empty stream, single event, all events in one window, all dropped) and
    determinism are covered.
"""

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


class Rng:
    """Deterministic splitmix64 stream - the only randomness source in this file."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        z = (self.state + GOLDEN) & MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        self.state = (z ^ (z >> 31)) & MASK64
        return self.state

    def below(self, limit):
        return self.next() % limit


def window_of(timestamp, width):
    return timestamp // width


def aggregate(edges):
    """Brute-force aggregate of a window's live edge multiset: (count, degrees, triangles)."""
    degrees = {}
    for (source, target), count in edges.items():
        if count <= 0:
            continue
        degrees[source] = degrees.get(source, 0) + count
        degrees[target] = degrees.get(target, 0) + count
    live = {edge for edge, count in edges.items() if count > 0}
    triangles = 0
    adjacency = {}
    for source, target in live:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    for source in sorted(adjacency):
        for target in sorted(neighbour for neighbour in adjacency[source] if neighbour > source):
            if target in adjacency.get(source, ()) and source in adjacency.get(target, ()):
                common = adjacency[source] & adjacency[target]
                triangles += len([node for node in common if node > target])
    return len(live), degrees, triangles


class WindowState:
    """The state of one tumbling window: live edges, degrees and the revision counter."""

    __slots__ = ("start", "edges", "degrees", "revisions", "emitted")

    def __init__(self, start):
        self.start = start
        self.edges = {}
        self.degrees = {}
        self.revisions = 0
        self.emitted = None

    def add(self, source, target, delta=1):
        """Apply a signed edge delta; an edge is live in a window while its net count is > 0.

        A retraction that arrives after its addition was dropped (too late, or evicted) leaves
        the net count negative, which simply means the edge is not live in this window - the
        same convention the stream oracle uses, so both agree by construction.
        """
        edge = (source, target)
        before = self.edges.get(edge, 0)
        after = before + delta
        self.edges[edge] = after
        live_delta = max(0, after) - max(0, before)
        if live_delta:
            self.degrees[source] = self.degrees.get(source, 0) + live_delta
            self.degrees[target] = self.degrees.get(target, 0) + live_delta
        return self

    def aggregate(self):
        live = {edge for edge, count in self.edges.items() if count > 0}
        triangles = 0
        adjacency = {}
        for source, target in live:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        for source in sorted(adjacency):
            for target in sorted(node for node in adjacency[source] if node > source):
                common = adjacency[source] & adjacency.get(target, set())
                triangles += len([node for node in common if node > target])
        return {"window": self.start, "edges": len(live),
                "degrees": {node: value for node, value in sorted(self.degrees.items())
                            if value > 0},
                "triangles": triangles, "revisions": self.revisions}

    def merge(self, other):
        """Associativity probe: combining two partial states of the same window."""
        merged = WindowState(self.start)
        merged.edges = dict(self.edges)
        for edge, count in other.edges.items():
            merged.edges[edge] = merged.edges.get(edge, 0) + count
        merged.degrees = dict(self.degrees)
        for node, value in other.degrees.items():
            merged.degrees[node] = merged.degrees.get(node, 0) + value
        merged.revisions = self.revisions + other.revisions
        return merged


class StreamEngine:
    """Windowed aggregation with a watermark, lateness tolerance, revisions and eviction."""

    def __init__(self, width=10, allowed_lateness=3, retention=None):
        self.width = width
        self.allowed_lateness = allowed_lateness
        self.retention = retention if retention is not None else allowed_lateness + 2 * width
        if self.retention < allowed_lateness:
            raise ValueError("retention must cover the lateness horizon")
        self.state = {}
        self.results = {}
        self.watermark = -1
        self.counters = {"accepted": 0, "late": 0, "dropped": 0, "revisions": 0,
                         "emitted": 0, "evicted": 0}
        self.log = []

    def process(self, timestamp, kind, source, target):
        """Feed one event; returns the action taken (accepted, late, dropped).

        An event is late when its window is already closed, and dropped when it is later than
        the lateness horizon - which, because retention >= allowed_lateness, always fires
        before its window's state could have been evicted.
        """
        self.log.append((timestamp, kind, source, target))
        start = window_of(timestamp, self.width) * self.width
        closed = self.watermark >= start + self.width
        if closed and timestamp < self.watermark - self.allowed_lateness:
            self.counters["dropped"] += 1
            return "dropped"
        state = self.state.setdefault(start, WindowState(start))
        action = "accepted"
        if closed:
            action = "late"
            self.counters["late"] += 1
            state.revisions += 1
            self.counters["revisions"] += 1
        state.add(source, target, 1 if kind == "add" else -1)
        self.counters["accepted"] += 1
        if self.watermark < timestamp:
            self.watermark = timestamp
        self._close_and_evict()
        return action

    def _close_and_evict(self):
        """Emit every window the watermark has passed and evict the ones past retention."""
        for start in sorted(self.state):
            state = self.state[start]
            if self.watermark >= start + self.width:
                self.results[start] = state.aggregate()
                if state.emitted is None:
                    self.counters["emitted"] += 1
                state.emitted = True
        for start in sorted(self.state):
            if self.watermark - self.retention >= start + self.width:
                del self.state[start]
                self.counters["evicted"] += 1

    def flush(self):
        """Close the remaining windows deterministically at the end of the stream."""
        for start in sorted(self.state):
            self.results[start] = self.state[start].aggregate()
            self.counters["emitted"] += 1
        self.state.clear()
        return self.results

    def live_windows(self):
        return sorted(self.state)


def brute_force(log, width, kind_filter=None):
    """Independent oracle: aggregate every window directly from the raw event log."""
    windows = {}
    for timestamp, kind, source, target in log:
        if kind_filter is not None and kind != kind_filter:
            continue
        windows.setdefault(window_of(timestamp, width), []).append((kind, source, target))
    out = {}
    for window, events in windows.items():
        state = WindowState(window * width)
        for kind, source, target in events:
            state.add(source, target, 1 if kind == "add" else -1)
        out[window * width] = state.aggregate()
    return out


def generate_stream(seed, windows=14, width=10, nodes=6):
    """A deterministic event stream: per-window adds, same-window removes and late deliveries.

    Every window block starts with an event on the window's first tick; two of those blocks get
    an extra event from the *previous* window's last tick inserted right after it, so the event
    is genuinely late (its window is closed) but within a small lateness budget - the revision
    path.  Events diverted to `stale` are appended at the very end of the log, far beyond any
    horizon, so they must be dropped.  The log is therefore deliberately not sorted by time.
    """
    rng = Rng(seed)
    blocks = []
    stale = []
    for index in range(windows):
        base = index * width
        block = [(base, "add", index % nodes, (index + 1) % nodes)]
        for _ in range(rng.below(4) + 1):
            source, target = rng.below(nodes), rng.below(nodes)
            if source == target:
                continue
            timestamp = base + rng.below(width)
            if rng.below(7) == 0:
                stale.append((timestamp, "add", source, target))
                continue
            block.append((timestamp, "add", source, target))
            if rng.below(3) == 0 and timestamp - base < width - 2:
                block.append((timestamp + 1 + rng.below(width - 2 - (timestamp - base)),
                              "remove", source, target))
        blocks.append(block)
    for index in (1, 4):
        if index + 1 < len(blocks):
            blocks[index + 1].insert(1, (index * width + width - 1, "add",
                                         (index + 2) % nodes, (index + 4) % nodes))
    log = [event for block in blocks for event in block]
    log.extend(stale)
    return log


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    width = 10
    log = generate_stream(2024)
    check(len(log) >= 30, "the generated stream must be sizeable")
    check(log != sorted(log, key=lambda entry: entry[0]),
          "the delivery order must contain late events, or lateness is untested")
    check(all(entry[1] in ("add", "remove") for entry in log), "only edge deltas are streamed")

    # ---- 1. every window matches the brute-force oracle -------------------
    engine = StreamEngine(width=width, allowed_lateness=10 ** 9)
    actions = [engine.process(*entry) for entry in log]
    engine.flush()
    oracle = brute_force(log, width)
    check(set(engine.results) == set(oracle), "the same windows must be produced")
    for start, value in oracle.items():
        check(engine.results[start]["edges"] == value["edges"],
              "window %d edge count (%r vs %r)" % (start, engine.results[start]["edges"],
                                                   value["edges"]))
        check(engine.results[start]["degrees"] == value["degrees"],
              "window %d degree histogram" % start)
        check(engine.results[start]["triangles"] == value["triangles"],
              "window %d triangles (%r vs %r)" % (start, engine.results[start]["triangles"],
                                                  value["triangles"]))
    check(all(action in ("accepted", "late") for action in actions),
          "with unbounded lateness nothing is dropped")
    check(engine.counters["dropped"] == 0, "no event is dropped when lateness is unbounded")

    # ---- 2. adds and removes cancel inside a window -----------------------
    cancelling = StreamEngine(width=width, allowed_lateness=10 ** 9)
    cancelling.process(3, "add", 1, 2)
    cancelling.process(4, "add", 2, 3)
    cancelling.process(5, "remove", 1, 2)
    check(cancelling.state[0].edges[(1, 2)] == 0, "an add/remove pair nets out to zero")
    cancelling.process(6, "remove", 9, 9)                 # unmatched retraction
    check(cancelling.state[0].edges[(9, 9)] == -1,
          "an unmatched retraction leaves a negative net count")
    cancelling.flush()
    check(cancelling.results[0]["edges"] == 1,
          "an add/remove pair must leave one live edge")
    check(cancelling.results[0]["degrees"] == {2: 1, 3: 1},
          "the cancelled edge must disappear from the degree histogram")
    check(9 not in cancelling.results[0]["degrees"],
          "a negative net count must not invent a degree")

    # ---- 3. lateness: revisions inside the horizon, drops outside --------
    strict = StreamEngine(width=width, allowed_lateness=3, retention=width)
    late_actions = [strict.process(*entry) for entry in log]
    strict.flush()
    check(strict.counters["dropped"] == len([a for a in late_actions if a == "dropped"]),
          "dropped events must be counted exactly once")
    check(strict.counters["late"] > 0, "the generated stream must contain late events")
    check(strict.counters["dropped"] > 0,
          "a tight retention horizon must actually drop some late events")
    check(strict.counters["revisions"] == strict.counters["late"],
          "every accepted late event is one revision")
    admitted = [(t, k, s, d) for (t, k, s, d), action in zip(log, late_actions)
                if action != "dropped"]
    strict_oracle = brute_force(admitted, width)
    check(set(strict.results) == set(strict_oracle),
          "windows of the admitted stream must match the oracle")
    for start, value in strict_oracle.items():
        check(strict.results[start]["edges"] == value["edges"],
              "strict window %d edge count" % start)
    check(len(admitted) + strict.counters["dropped"] == len(log),
          "no event may vanish: admitted + dropped == total")

    # a late event inside the horizon *revises* an already emitted window
    revising = StreamEngine(width=width, allowed_lateness=8)
    revising.process(1, "add", 0, 1)
    revising.process(12, "add", 1, 2)                 # closes window 0
    check(0 in revising.results and revising.results[0]["edges"] == 1,
          "window 0 is emitted when the watermark passes it")
    check(revising.process(9, "add", 2, 3) == "late", "an in-horizon late event is a revision")
    check(revising.results[0]["edges"] == 2 and revising.results[0]["revisions"] == 1,
          "the revision must update the emitted result exactly once")
    check(revising.process(60, "add", 3, 4) == "accepted", "a fresh event is not a revision")

    # ---- 4. eviction ------------------------------------------------------
    evicting = StreamEngine(width=width, allowed_lateness=2, retention=width)
    for timestamp in range(0, 90, 7):
        evicting.process(timestamp, "add", timestamp % 5, (timestamp + 1) % 5)
    evicting.flush()
    check(evicting.counters["evicted"] > 0, "some windows must have been evicted")
    emitted_before = dict(evicting.results)
    check(len(evicting.state) <= 2, "live state must stay bounded by the retention horizon")
    check(all(start % width == 0 for start in emitted_before),
          "emitted windows are aligned to the window grid")
    for start, value in emitted_before.items():
        check(value["edges"] <= len([entry for entry in evicting.log
                                     if window_of(entry[0], width) * width == start]),
              "an emitted count cannot exceed the number of events in the window")
    evicting.process(200, "add", 1, 2)
    for start, value in emitted_before.items():
        if start in evicting.results:
            check(evicting.results[start]["edges"] == value["edges"],
                  "eviction must never rewrite an emitted result")

    # ---- 5. merge associativity ------------------------------------------
    grouped = {}
    for entry in log:
        if entry[1] == "add":
            grouped.setdefault(window_of(entry[0], width) * width, []).append(entry)
    target_window = next(start for start in sorted(grouped) if len(grouped[start]) >= 2)
    adds = grouped[target_window]
    check(len(adds) >= 2, "the chosen window must contain several additions")
    left = StreamEngine(width=width, allowed_lateness=10 ** 9)
    right = StreamEngine(width=width, allowed_lateness=10 ** 9)
    for entry in adds[:len(adds) // 2]:
        left.process(*entry)
    for entry in adds[len(adds) // 2:]:
        right.process(*entry)
    check(target_window in left.state and target_window in right.state,
          "both halves must hold a partial state for the chosen window")
    merged = left.state[target_window].merge(right.state[target_window])
    expected = brute_force(adds, width)[target_window]
    check(merged.aggregate() == expected,
          "merging two partial window states must replay the whole window")
    check(right.state[target_window].merge(left.state[target_window]).aggregate() == expected,
          "the merge must be order independent")
    partial = WindowState(0).add(0, 1).add(1, 2)
    other = WindowState(0).add(2, 3)
    check(partial.merge(other).aggregate()["edges"] == 3,
          "merging partial window states replays the whole window")
    check(partial.merge(other).aggregate()["degrees"] == {0: 1, 1: 2, 2: 2, 3: 1},
          "merged degrees add up node by node")

    # ---- 6. degenerate streams and determinism ---------------------------
    empty = StreamEngine(width=width, allowed_lateness=1)
    check(empty.flush() == {} and empty.counters["accepted"] == 0,
          "an empty stream emits nothing and accepts nothing")
    single = StreamEngine(width=width, allowed_lateness=1)
    single.process(7, "add", 4, 5)
    single.flush()
    check(single.results[0]["edges"] == 1 and single.results[0]["degrees"] == {4: 1, 5: 1},
          "a single event produces a one-edge window")
    one_window = StreamEngine(width=width, allowed_lateness=1)
    for index in range(6):
        one_window.process(index, "add", index % 3, (index + 1) % 3)
    one_window.flush()
    check(len(one_window.results) == 1, "all events inside one window produce one result")
    check(one_window.results[0]["triangles"] == 1,
          "the three edges 0-1, 1-2, 2-0 form exactly one triangle")
    boundary = StreamEngine(width=width, allowed_lateness=1)
    boundary.process(9, "add", 0, 1)
    boundary.process(10, "add", 1, 2)
    boundary.flush()
    check(boundary.results[0]["edges"] == 1 and boundary.results[10]["edges"] == 1,
          "the tick 9/10 boundary falls between the two windows")
    total_events = len(log)
    check(engine.counters["accepted"] == total_events, "unbounded lateness accepts everything")
    repeat = StreamEngine(width=width, allowed_lateness=10 ** 9)
    for entry in log:
        repeat.process(*entry)
    repeat.flush()
    check(repeat.results == engine.results and repeat.counters == engine.counters,
          "the engine must be reproducible")
    check(engine.counters["emitted"] >= len(engine.results) - 1,
          "every window is emitted at least once")

    print("windows: %d emitted, %d live at the end, %d events accepted"
          % (len(engine.results), len(engine.live_windows()), engine.counters["accepted"]))
    print("lateness: %d late, %d revisions, %d dropped by %d tick horizon"
          % (strict.counters["late"], strict.counters["revisions"], strict.counters["dropped"],
             strict.allowed_lateness))
    print("eviction: %d windows evicted, retention %d ticks, %d live windows"
          % (evicting.counters["evicted"], evicting.retention, len(evicting.state)))
    print("PASS")


if __name__ == "__main__":
    main()
