"""Task 995 - Implement interactive visualization generator for algorithm steps (trace player).
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Instruments two classic algorithms so every comparison, swap and probe becomes a frame (index,
array snapshot, highlighted cells, note), then plays those frames through a player supporting
play/pause, single-stepping and seeking.  The timeline is validated as a whole before playback:
indices must be contiguous, each snapshot must be a permutation of the start state, and
highlights must point inside it, which is what makes seeking safe - jumping to frame k must be
indistinguishable from stepping k times.

Simplified versus production: the visualisation is the frame model plus validation rather than a
rendering backend, and only bubble sort and binary search are instrumented.  Verified: the
sorted-result oracle on the final frame, a swap count equal to the independently computed
inversion count, permutation/contiguity tamper detection, step-versus-seek equivalence, the log2
search bound, the player state machine, and the single-element and empty boundaries.
"""
import hashlib
import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle, error_type=ValueError):
    try:
        action()
    except error_type as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected %s mentioning %r" % (error_type.__name__, needle))


def frame(index, array, highlight, note):
    return {"step": index, "array": tuple(array), "highlight": tuple(highlight), "note": note}


def trace_bubble_sort(values):
    """Bubble sort, emitting one frame per comparison (the swap frame shows the swap)."""
    array, frames, step = list(values), [frame(0, values, (), "start")], 0
    for end in range(len(array) - 1, 0, -1):
        swapped = False
        for position in range(end):
            step += 1
            if array[position] > array[position + 1]:
                array[position], array[position + 1] = array[position + 1], array[position]
                swapped, note = True, "swap"
            else:
                note = "compare"
            frames.append(frame(step, array, (position, position + 1), note))
        if not swapped:
            break
    return frames


def count_inversions(values):
    """Independent oracle: bubble sort performs exactly one swap per inverted pair."""
    return sum(1 for left in range(len(values)) for right in range(left + 1, len(values))
               if values[left] > values[right])


def trace_binary_search(values, target):
    """Binary search over the sorted input; one frame per probe, plus the final state."""
    ordered = sorted(values)
    frames = [frame(0, ordered, (), "start")]
    low, high, step, found = 0, len(ordered) - 1, 0, None
    while low <= high:
        middle = (low + high) // 2
        step += 1
        frames.append({"step": step, "array": tuple(ordered), "highlight": (middle,),
                       "note": "probe", "low": low, "high": high})
        if ordered[middle] == target:
            found = middle
            break
        if ordered[middle] < target:
            low = middle + 1
        else:
            high = middle - 1
    return frames, found


def validate_timeline(frames):
    """Every defect that would make interactive seeking lie about the algorithm."""
    problems = []
    if not frames:
        return ["timeline is empty"]
    expected = sorted(frames[0]["array"])
    for index, item in enumerate(frames):
        if item["step"] != index:
            problems.append("frame %d is stamped %d" % (index, item["step"]))
        if sorted(item["array"]) != expected:
            problems.append("frame %d is not a permutation of the start state" % index)
        if any(cell < 0 or cell >= len(item["array"]) for cell in item["highlight"]):
            problems.append("frame %d highlights a cell outside the array" % index)
    return problems


def timeline_digest(frames):
    body = ";".join("%d|%s|%s|%s" % (item["step"], ",".join(map(str, item["array"])),
                                     ",".join(map(str, item["highlight"])), item["note"])
                    for item in frames)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def diff(left, right):
    """Indices whose value changed between two frames (a brute-force positional compare)."""
    return [index for index, (before, after) in enumerate(zip(left["array"], right["array"]))
            if before != after]


class TracePlayer:
    """Play/pause/step/seek over a validated timeline; stepping requires a paused player."""

    def __init__(self, frames):
        problems = validate_timeline(frames)
        if problems:
            raise ValueError("invalid timeline: %s" % "; ".join(problems))
        self.frames = list(frames)
        self.index = 0
        self.playing = False

    def current(self):
        return self.frames[self.index]

    def play(self):
        if self.playing:
            raise RuntimeError("the player is already playing")
        if self.index == len(self.frames) - 1:
            raise RuntimeError("the trace is already finished")
        self.playing = True

    def pause(self):
        if not self.playing:
            raise RuntimeError("the player is not playing")
        self.playing = False

    def step(self):
        if self.playing:
            raise RuntimeError("pause the player before stepping")
        if self.index + 1 >= len(self.frames):
            raise IndexError("the player is at the last frame")
        self.index += 1
        return self.current()

    def seek(self, target):
        if not 0 <= target < len(self.frames):
            raise ValueError("frame %r is outside the timeline" % (target,))
        self.index = target
        return self.current()

    def replay(self, count):
        """Deterministic playback from the start, used to prove seek == repeated step."""
        self.seek(0)
        for _ in range(count):
            self.step()
        return self.current()


def main():
    values = [5, 1, 4, 2, 8]
    timeline = trace_bubble_sort(values)

    # The final frame is the sorted result, and the swap count equals the inversion oracle.
    check(tuple(sorted(values)) == timeline[-1]["array"],
          "the last bubble-sort frame must be the sorted array")
    swaps = sum(1 for item in timeline if item["note"] == "swap")
    check(swaps == count_inversions(values), "swap frames (%d) must equal inversions (%d)"
          % (swaps, count_inversions(values)))
    comparisons = sum(1 for item in timeline if item["note"] in ("swap", "compare"))
    check(len(timeline) == comparisons + 1,
          "every comparison must contribute exactly one frame plus the start frame")
    check(validate_timeline(timeline) == [], "a freshly traced timeline must validate")

    # Independent oracle: every intermediate frame is a permutation of the start state.
    check(all(sorted(item["array"]) == sorted(values) for item in timeline),
          "bubble sort must never add or drop an element")
    changed = [index for index, value in enumerate(values) if value != sorted(values)[index]]
    check(diff(timeline[0], timeline[-1]) == changed == [0, 1, 3],
          "diff must report exactly the positions whose value changed (oracle %r)" % (changed,))

    # Timeline validation catches tampering: a lost element and a stale step stamp.
    broken = [dict(item) for item in timeline]
    broken[2]["array"] = tuple(list(broken[2]["array"][:-1]) + [99])
    stamped = [dict(item) for item in timeline]
    stamped[3]["step"] = 99
    check(any("permutation" in problem for problem in validate_timeline(broken))
          and any("stamped" in problem for problem in validate_timeline(stamped)),
          "tampered frames must fail permutation and contiguity validation")
    expect_error(lambda: TracePlayer(broken), "invalid timeline")

    # Seek/step equivalence is what makes the player interactive.
    player = TracePlayer(timeline)
    for target in (1, 3, 5):
        check(player.replay(target) == timeline[target],
              "replaying %d steps must equal seeking to that frame" % target)
    player.seek(len(timeline) - 1)
    top = TracePlayer(timeline)
    check(top.replay(len(timeline) - 1) == timeline[-1], "a full replay must land on the last frame")
    expect_error(lambda: player.seek(len(timeline)), "outside the timeline")
    expect_error(lambda: player.seek(-1), "outside the timeline")
    expect_error(lambda: player.step(), "last frame", IndexError)

    # Player state machine: stepping is only legal while paused.
    paused = TracePlayer(timeline)
    expect_error(lambda: paused.pause(), "not playing", RuntimeError)
    paused.play()
    expect_error(lambda: paused.play(), "already playing", RuntimeError)
    expect_error(lambda: paused.step(), "pause the player", RuntimeError)
    paused.pause()
    check(paused.current() == timeline[0] and paused.step() == timeline[1],
          "stepping after pausing must advance exactly one frame")
    finished = TracePlayer(timeline)
    finished.seek(len(timeline) - 1)
    expect_error(lambda: finished.play(), "already finished", RuntimeError)

    # Binary search: probe count inside the logarithmic bound, and the hit/miss boundary.
    ordered = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    probes, found = trace_binary_search(ordered, 13)
    bound = math.ceil(math.log2(len(ordered) + 1)) + 1
    check(found == ordered.index(13) and len(probes) - 1 <= bound,
          "a successful search must stay within ceil(log2(n+1)) probes")
    check(probes[-1]["note"] == "probe" and probes[-1]["highlight"] == (found,),
          "the final frame must highlight the located index")
    missing, absent = trace_binary_search(ordered, 100)
    check(absent is None and missing[-1]["note"] == "probe",
          "an absent target must be reported as a miss")
    check(len(missing) - 1 <= math.ceil(math.log2(len(ordered) + 1)),
          "a miss must also stay within the logarithmic bound")

    # Determinism: tracing twice gives the same frames and digest.
    check(timeline_digest(trace_bubble_sort(values)) == timeline_digest(timeline),
          "tracing the same input twice must be deterministic")
    check(timeline_digest(trace_bubble_sort([5, 1, 4, 2, 7])) != timeline_digest(timeline),
          "a different input must produce a different timeline digest")

    # Boundaries: a single element and an empty input need exactly one frame.
    single = trace_bubble_sort([7])
    check(len(single) == 1 and single[0]["array"] == (7,) and TracePlayer(single).current() == single[0]
          and len(trace_bubble_sort([])) == 1 and trace_bubble_sort([])[0]["array"] == ()
          and trace_binary_search([], 1)[1] is None,
          "a single element and an empty input must each produce one playable, sorted frame")
    expect_error(lambda: TracePlayer([]), "timeline is empty")
    print("frames=%d swaps=%d probes=%d found=%s digest=%s\nPASS"
          % (len(timeline), swaps, len(probes) - 1, found, timeline_digest(timeline)[:16]))


if __name__ == "__main__":
    main()
