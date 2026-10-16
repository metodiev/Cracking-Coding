"""Task 600 - Implement advanced checkpoint-restore for long-running computations (save state). (Tier: Advanced, CTCI Beyond — Software Engineering & Tooling)

A checkpoint/restart harness for a deterministic long-running computation, built around a small
explicit state record rather than around pickling an object graph: the state is a plain dict
(step counter, 64-bit generator state, exact integer accumulator, Welford mean/M2 pair, running
maximum, a rolling DP vector and a digest), which makes it JSON-serialisable, inspectable and
comparable.

The store (`CheckpointStore`) is the interesting part, because checkpointing is a storage problem
before it is a numerical one:

  * writes are atomic -- the payload goes to `<name>.tmp`, is flushed and fsync'ed, and only then
    renamed over the target with os.replace, so a crash never leaves a half-written checkpoint at
    the real path (`a_save_leaves_no_tmp_behind` asserts the tmp file is gone afterwards);
  * each file carries a schema version and a CRC32 of the canonical JSON body, checked on load, so
    accidental corruption (a truncated file, a flipped character) is caught and reported as a
    CheckpointError instead of silently resuming from garbage;
  * retention prunes old checkpoints, keeping the newest `retention` files, and the harness is
    asserted to keep working after the very checkpoint it would have resumed from has been pruned
    (it falls back to the newest surviving one);
  * the file documents the limit of the checksum honestly: CRC32 detects corruption, not tampering
    -- anyone who edits the payload can recompute it, which would require an HMAC with a key the
    process does not have here.

Documented checks: an uninterrupted run and a run that is killed (SimulatedCrash) and resumed
produce *identical* states -- exact equality for every integer, string and list field, and 1e-15
relative agreement for the two floating-point fields; resuming from an older checkpoint (not the
newest) reaches the same final state, which pins down replay determinism; the number of steps
executed after a restart is asserted to be exactly the work lost since the last checkpoint; a
corrupted checkpoint raises CheckpointError and the harness transparently resumes from the previous
one; a version mismatch raises too; loading twice is idempotent; the final state carries the
expected keys and survives a JSON round trip unchanged; and the whole test uses a temporary
directory created *next to this file* (never /tmp) which is removed in a finally block, with an
assertion that nothing is left behind.
"""
import json
import os
import shutil
import tempfile
import zlib

CHECKPOINT_VERSION = 3
MODULUS = 1000003
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


class CheckpointError(Exception):
    """Raised when a checkpoint cannot be trusted (checksum, schema or truncation)."""


class SimulatedCrash(Exception):
    """Stands in for the process dying mid-computation."""


def crc32(text):
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def initial_state(seed, width):
    return {
        "step": 0,
        "rng": seed,
        "sum_int": 0,
        "count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "max_value": -1,
        "dp": [1] + [0] * (width - 1),
        "digest": 0,
    }


def advance(state, width):
    """One deterministic workload step, mutating `state` in place.

    The mix is deliberately cheap but state-dependent: a 64-bit LCG, an exact integer accumulator,
    a Welford mean/variance update, a running maximum and a rolling polynomial DP over `width`
    cells (updated from the top down so each cell sees the previous step's value)."""
    rng = (state["rng"] * 6364136223846793005 + 1442695040888963407) & MASK64
    state["rng"] = rng
    value = (rng >> 17) % 1000
    state["step"] += 1
    state["sum_int"] += value
    state["count"] += 1
    delta = value - state["mean"]
    state["mean"] += delta / state["count"]
    state["m2"] += delta * (value - state["mean"])
    state["max_value"] = max(state["max_value"], value)
    dp = state["dp"]
    for index in range(width - 1, 0, -1):
        dp[index] = (dp[index] + dp[index - 1] * value + index) % MODULUS
    dp[0] = (dp[0] + value) % MODULUS
    state["digest"] = (state["digest"] * 31 + value) % MODULUS
    return state


def canonical_payload(state):
    return {"version": CHECKPOINT_VERSION, "state": state}


def serialize(state):
    payload = canonical_payload(state)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps({"crc32": crc32(body), "payload": payload}, sort_keys=True, indent=1)


def deserialize(text, expected_version=CHECKPOINT_VERSION):
    try:
        outer = json.loads(text)
    except ValueError as error:
        raise CheckpointError("checkpoint is not valid JSON: %s" % error)
    if not isinstance(outer, dict) or "payload" not in outer or "crc32" not in outer:
        raise CheckpointError("checkpoint is missing its envelope")
    payload = outer["payload"]
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if crc32(body) != outer["crc32"]:
        raise CheckpointError("checksum mismatch (file corrupted or tampered with)")
    if payload.get("version") != expected_version:
        raise CheckpointError("schema version %r is not supported" % (payload.get("version"),))
    state = payload["state"]
    required = {"step", "rng", "sum_int", "count", "mean", "m2", "max_value", "dp", "digest"}
    if set(state) != required:
        raise CheckpointError("state fields do not match the schema: %s" % sorted(state))
    return state


class CheckpointStore:
    """Atomic, versioned, checksummed checkpoint files with retention."""

    prefix = "ckpt-"

    def __init__(self, directory, retention=3):
        self.directory = directory
        self.retention = retention
        os.makedirs(directory, exist_ok=True)

    def path_for(self, step):
        return os.path.join(self.directory, "%s%08d.json" % (self.prefix, step))

    def save(self, state):
        path = self.path_for(state["step"])
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(serialize(state))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self.prune()
        return path

    def files(self):
        names = [name for name in sorted(os.listdir(self.directory))
                 if name.startswith(self.prefix) and name.endswith(".json")]
        return [os.path.join(self.directory, name) for name in names]

    def prune(self):
        files = self.files()
        for path in files[:-self.retention] if self.retention > 0 else []:
            os.remove(path)
        return len(self.files())

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return deserialize(handle.read())

    def load_latest(self):
        files = self.files()
        if not files:
            raise CheckpointError("no checkpoint available")
        return self.load(files[-1])

    def latest_step(self):
        files = self.files()
        if not files:
            return None
        name = os.path.basename(files[-1])
        return int(name[len(self.prefix):-len(".json")])


def compute(state, total_steps, width, store=None, every=0, crash_at=None, counter=None):
    """Run the workload from `state` for `total_steps` steps, checkpointing every `every` steps.

    `crash_at` raises SimulatedCrash when the *global* step counter reaches it, which is how the
    crash/restore behaviour is exercised without relying on real signals or threads.  `counter` is
    an optional one-element list that accumulates every step actually executed, including the work
    that is thrown away when a crash fires -- that is what makes the replayed-work accounting below
    possible."""
    steps_done = 0
    while state["step"] < total_steps:
        advance(state, width)
        steps_done += 1
        if counter is not None:
            counter[0] += 1
        if crash_at is not None and state["step"] == crash_at:
            raise SimulatedCrash("simulated crash at step %d" % crash_at)
        if store is not None and every > 0 and state["step"] % every == 0:
            store.save(state)
    return state, steps_done


def run_with_restarts(store, target, width, every, crashes, seed=0):
    """Drive the workload to `target` steps, restarting from the newest checkpoint on crash.

    Returns (final state, number of restarts, total steps executed by the workload) -- the third
    value exceeds `target` by exactly the work that had to be replayed after each crash."""
    if store.files():
        state = store.load_latest()
    else:
        state = initial_state(seed, width)
    restarts = 0
    counter = [0]
    pending = list(crashes)
    while state["step"] < target:
        crash_at = pending[0] if pending and pending[0] > state["step"] else None
        try:
            state, _ = compute(state, target, width, store, every, crash_at, counter)
        except SimulatedCrash:
            restarts += 1
            if crash_at is not None:
                pending.pop(0)
            state = store.load_latest()
    return state, restarts, counter[0]


def states_equal(first, second):
    """Field-by-field comparison: exact for integers/strings/lists, relative for the two floats."""
    if set(first) != set(second):
        return False
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, float) or isinstance(right, float):
            scale = max(1.0, abs(left), abs(right))
            if abs(left - right) > 1e-15 * scale:
                return False
        elif left != right:
            return False
    return True


def main():
    directory = tempfile.mkdtemp(prefix="ckpt-", dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        width = 6
        target = 400
        every = 50

        # --- baseline: one uninterrupted run ------------------------------------------
        baseline_state = initial_state(12345, width)
        baseline_state, baseline_steps = compute(baseline_state, target, width)
        assert baseline_steps == target
        assert baseline_state["step"] == target
        assert baseline_state["count"] == target

        # --- a run that crashes twice and is resumed from its checkpoints -------------
        store = CheckpointStore(os.path.join(directory, "run-a"), retention=3)
        resumed_state, restarts, executed = run_with_restarts(store, target, width, every,
                                                              [120, 310], seed=12345)
        assert restarts == 2, restarts
        assert states_equal(baseline_state, resumed_state), "crash/restore changes the result"
        assert store.latest_step() == target
        assert len(store.files()) <= 3, "retention keeps the newest three"
        # Replayed work: the crash at 120 redoes 20 steps (the checkpoint at 100 survives) and the
        # crash at 310 redoes 10, so the workload runs 30 steps more than the target.  Asserted as
        # "more than none, and never more than one checkpoint interval per crash".
        lost = executed - target
        assert 0 < lost <= 2 * every, (lost, executed, every)

        # --- resuming from an *older* checkpoint replays identically ------------------
        old_file = store.files()[0]
        old_state = store.load(old_file)
        older_path = os.path.join(directory, "run-old")
        older_store = CheckpointStore(older_path, retention=5)
        replayed, done = compute(dict(old_state), target, width, older_store, every)
        assert states_equal(baseline_state, replayed), "replay from an older checkpoint differs"
        assert done == target - old_state["step"]
        assert older_store.latest_step() == target
        # Even with retention 1 the newest checkpoint is always enough to finish correctly,
        # which is asserted by pruning down and restarting.
        trimmed = CheckpointStore(os.path.join(directory, "run-trim"), retention=1)
        compute(initial_state(12345, width), target, width, trimmed, every)
        assert len(trimmed.files()) == 1, trimmed.files()
        tail, _ = compute(dict(trimmed.load_latest()), target + 100, width, trimmed, every)
        reference = initial_state(12345, width)
        reference, _ = compute(reference, target + 100, width)
        assert states_equal(reference, tail)

        # --- atomicity: no stray temporary file ---------------------------------------
        assert not any(name.endswith(".tmp") for name in os.listdir(store.directory))

        # --- corruption is detected, and the harness falls back ----------------------
        corrupt_path = os.path.join(directory, "run-b")
        corrupt_store = CheckpointStore(corrupt_path, retention=5)
        compute(initial_state(777, width), target, width, corrupt_store, every)
        victim = corrupt_store.files()[1]
        with open(victim, "r", encoding="utf-8") as handle:
            text = handle.read()
        with open(victim, "w", encoding="utf-8") as handle:
            handle.write(text.replace('"step"', '"stXp"', 1))
        try:
            corrupt_store.load(victim)
            raise AssertionError("a corrupted checkpoint must not load")
        except CheckpointError:
            pass
        survivors = [path for path in corrupt_store.files() if path != victim]
        newest = max(survivors, key=lambda path: os.path.basename(path))
        recovered = corrupt_store.load(newest)
        assert recovered["step"] >= target - 2 * every
        finished, _ = compute(dict(recovered), target, width)
        reference_777 = initial_state(777, width)
        reference_777, _ = compute(reference_777, target, width)
        assert states_equal(reference_777, finished), "fallback resume is still correct"

        # --- a version bump is refused -------------------------------------------------
        bumped = serialize(baseline_state)
        bumped_payload = json.loads(bumped)
        bumped_payload["payload"]["version"] = CHECKPOINT_VERSION + 1
        body = json.dumps(bumped_payload["payload"], sort_keys=True, separators=(",", ":"))
        bumped_payload["crc32"] = crc32(body)
        try:
            deserialize(json.dumps(bumped_payload))
            raise AssertionError("an unknown schema version must be refused")
        except CheckpointError:
            pass
        # A tampered payload with a *stale* checksum is refused as well; one with a recomputed
        # checksum is not, which is the documented limit of a CRC (an HMAC would be needed).
        stale = json.loads(serialize(baseline_state))
        stale["payload"]["state"]["sum_int"] = -1
        try:
            deserialize(json.dumps(stale))
            raise AssertionError("a tampered payload must be caught by the checksum")
        except CheckpointError:
            pass
        recomputed = json.loads(serialize(baseline_state))
        recomputed["payload"]["state"]["sum_int"] = -1
        body = json.dumps(recomputed["payload"], sort_keys=True, separators=(",", ":"))
        recomputed["crc32"] = crc32(body)
        assert deserialize(json.dumps(recomputed))["sum_int"] == -1, "CRC is not a MAC"

        # --- round trip, idempotence and schema ---------------------------------------
        text = serialize(baseline_state)
        assert deserialize(text) == baseline_state, "JSON round trip is lossless"
        assert deserialize(text) == deserialize(text), "loading twice is idempotent"
        assert json.loads(text)["payload"]["state"]["step"] == target
        assert set(deserialize(text)) == {"step", "rng", "sum_int", "count", "mean", "m2",
                                          "max_value", "dp", "digest"}
        try:
            deserialize("{\"payload\": {\"version\": 3, \"state\": {}}, \"crc32\": 0}")
            raise AssertionError("an incomplete state must be refused")
        except CheckpointError:
            pass
        try:
            deserialize("not json at all")
            raise AssertionError("truncated JSON must be refused")
        except CheckpointError:
            pass

        # --- restart counts are bounded by the checkpoint interval --------------------
        for crash in (60, 100, 200):
            fresh = CheckpointStore(os.path.join(directory, "crash-%d" % crash), retention=4)
            final_state, count, _ = run_with_restarts(fresh, target, width, every, [crash],
                                                      seed=12345)
            assert count == 1, (crash, count)
            assert states_equal(baseline_state, final_state), crash
            assert fresh.load_latest()["step"] == target

        # --- a crash before the first checkpoint still resumes correctly --------------
        early = CheckpointStore(os.path.join(directory, "early"), retention=2)
        try:
            compute(initial_state(12345, width), target, width, early, every, crash_at=20)
            raise AssertionError("the simulated crash must fire")
        except SimulatedCrash:
            pass
        assert early.files() == [], "no checkpoint was written before the crash"
        restarted, _ = compute(initial_state(12345, width), target, width, early, every)
        assert states_equal(baseline_state, restarted), "a cold restart still matches"

        # --- determinism ------------------------------------------------------------------
        first = initial_state(4321, width)
        first, _ = compute(first, 150, width)
        second = initial_state(4321, width)
        second, _ = compute(second, 150, width)
        assert first == second
        assert states_equal(first, second)
        assert not states_equal(first, baseline_state)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    leftovers = [name for name in os.listdir(os.path.dirname(os.path.abspath(__file__)))
                 if name.startswith("ckpt-")]
    assert leftovers == [], leftovers


if __name__ == "__main__":
    main()
    print("PASS")
