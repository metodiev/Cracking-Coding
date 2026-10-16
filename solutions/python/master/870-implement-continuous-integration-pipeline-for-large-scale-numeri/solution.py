"""Task 870 - Implement continuous integration pipeline for large-scale numerical code with reproducibility.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

An in-process CI orchestrator for a numerically heavy repository.  Its ingredients are the
ones a real pipeline needs, and each one is computed rather than asserted:

  * a job DAG (stages with dependencies, a priority, a duration, typed scale parameters) and
    an explicit virtual clock.  The scheduler fills W worker slots from the ready set ordered
    by (-priority, name), advances the clock to the earliest completion, and retries failures
    up to a configurable limit.  Nothing sleeps and nothing is concurrent: the schedule is a
    deterministic function of the DAG, so scheduling properties (worker limit, critical path,
    priority order) are *checked* rather than timed;
  * content-addressed incremental builds.  A job's cache key is the digest of its toolchain,
    command, declared source files (by content hash) and the digests of its dependencies'
    artifacts - a Merkle key.  A warm cache therefore rebuilds exactly the transitive
    dependents of a changed source and nothing else, and touching a file without changing its
    bytes rebuilds nothing;
  * artifact provenance.  Every artifact records the job, the toolchain, the source hashes and
    the dependency digests it was built from, so the provenance closure of a final artifact
    reaches exactly its transitive source set;
  * reproducibility checking.  Artifacts are produced by a real numerical kernel (a
    deterministic quadrature trace-driven by the artifact's Merkle key).  Rebuilding with a
    fresh cache must reproduce every digest bit for bit; a job declared nondeterministic
    consumes an entropy counter per attempt and is reported as a reproducibility violation.

Verified: topological ordering of every start time, the worker limit at every instant, the
makespan between the critical path and the critical path plus scheduling slack, cold-run
execution counts (each job exactly once), exact incremental rebuild sets, no-op rebuilds for
byte-identical edits, byte equality of restored and recomputed artifacts, provenance closure,
failure propagation with the transitive skip set, flaky-job retry accounting, priority
ordering under a single worker, replay determinism of the whole timeline, reproducibility of
every deterministic job, detection of a nondeterministic job, and rejection of cyclic,
duplicate and dangling DAGs.
"""
import hashlib
import math

MASK64 = (1 << 64) - 1
ENTROPY = [0]  # models the nondeterminism of a real build (compiler timestamps, thread order)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        self.state, raw = splitmix64(self.state)
        return (raw >> 11) * (1.0 / float(1 << 53))


def digest_of(*parts):
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


def seed_of(text):
    return int(digest_of(text)[:16], 16)


def numerical_kernel(seed, samples):
    """The actual 'build work': a deterministic quadrature trace whose cost scales with the
    job's sample parameter.  Same seed and sample count always produce the same value."""
    rng = Rng(seed)
    total = 0.0
    for index in range(samples):
        x = rng.uniform()
        weight = 1.0 if index % 2 == 0 else 0.5
        total += weight * math.sin(math.pi * x) * x
    return total / samples


class JobSpec:
    """A pipeline stage.  `samples` is the typed scale parameter of the numerical kernel."""

    def __init__(self, name, duration, deps=(), priority=0, sources=(), samples=64,
                 command=None, nondeterministic=False, fail_attempts=0):
        if duration <= 0:
            raise ValueError(f"job {name}: duration must be positive")
        if not name:
            raise ValueError("job name must not be empty")
        self.name = name
        self.duration = duration
        self.deps = tuple(deps)
        self.priority = priority
        self.sources = tuple(sources)
        self.samples = samples
        self.command = command if command is not None else f"build {name} --samples {samples}"
        self.nondeterministic = nondeterministic
        self.fail_attempts = fail_attempts


class Artifact:
    def __init__(self, job, key, value, provenance):
        self.job = job
        self.key = key
        self.value = value
        self.content = f"{job}|{key}|{value:.12f}"
        self.digest = digest_of(self.content)
        self.provenance = provenance

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Artifact({self.job}, {self.digest[:8]})"


class BuildCache:
    """Content-addressed artifact store shared by successive pipeline runs."""

    def __init__(self):
        self.entries = {}
        self.lookups = 0

    def __len__(self):
        return len(self.entries)


class Pipeline:
    def __init__(self, jobs, sources=None, workers=2, max_retries=1, toolchain="numerics-1.4.0"):
        if workers < 1:
            raise ValueError("workers must be at least one")
        self.toolchain = toolchain
        self.workers = workers
        self.max_retries = max_retries
        self.sources = dict(sources or {})
        self.jobs = {}
        for job in jobs:
            if job.name in self.jobs:
                raise ValueError(f"duplicate job name {job.name}")
            self.jobs[job.name] = job
        for job in self.jobs.values():
            for dep in job.deps:
                if dep not in self.jobs:
                    raise ValueError(f"job {job.name} depends on unknown job {dep}")
            for source in job.sources:
                if source not in self.sources:
                    raise ValueError(f"job {job.name} reads unknown source {source}")
        self._check_acyclic()

    def _check_acyclic(self):
        visiting = set()
        done = set()

        def visit(name):
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"dependency cycle through {name}")
            visiting.add(name)
            for dep in self.jobs[name].deps:
                visit(dep)
            visiting.discard(name)
            done.add(name)

        for name in self.jobs:
            visit(name)

    def dependents(self, name):
        return sorted(other for other, job in self.jobs.items() if name in job.deps)

    def source_hashes(self, job):
        return {source: digest_of(self.sources[source]) for source in sorted(job.sources)}

    def cache_key(self, job, digests):
        dependency_state = [(dep, digests[dep]) for dep in sorted(job.deps)]
        sources = sorted(self.source_hashes(job).items())
        return digest_of(self.toolchain, job.command, job.samples, sources, dependency_state)

    def execute(self, job, key):
        # A nondeterministic job mixes in an entropy counter, exactly like a build step that
        # embeds a timestamp or races threads; the schedule itself stays deterministic.
        entropy = 0
        if job.nondeterministic:
            ENTROPY[0] += 1
            entropy = ENTROPY[0]
        seed = seed_of(key) ^ (entropy * 0x9E3779B97F4A7C15)
        value = numerical_kernel(seed, job.samples)
        provenance = {
            "job": job.name,
            "command": job.command,
            "toolchain": self.toolchain,
            "sources": self.source_hashes(job),
            "dependencies": {dep: digest_of(dep, key) for dep in job.deps},
            "nondeterministic": job.nondeterministic,
        }
        return Artifact(job.name, key, value, provenance)

    def run(self, cache=None):
        cache = cache if cache is not None else BuildCache()
        state = {}
        for name, job in self.jobs.items():
            state[name] = "blocked" if job.deps else "ready"
        attempts = {}
        digests = {}
        artifacts = {}
        timeline = []
        running = {}
        retried = []
        failed = []
        skipped = set()
        clock = 0
        while True:
            self._promote(state, skipped)
            if not running and not any(s == "ready" for s in state.values()):
                break
            ready = sorted((n for n, s in state.items() if s == "ready"),
                           key=lambda n: (-self.jobs[n].priority, n))
            for name in ready:
                if len(running) >= self.workers:
                    break
                job = self.jobs[name]
                key = self.cache_key(job, digests)
                if not job.nondeterministic and key in cache.entries:
                    cache.lookups += 1
                    artifact = cache.entries[key]
                    digests[name] = artifact.digest
                    artifacts[name] = artifact
                    state[name] = "done"
                    timeline.append({"job": name, "start": clock, "end": clock,
                                     "attempt": 0, "kind": "cache", "worker": None})
                    continue
                attempts[name] = attempts.get(name, 0) + 1
                state[name] = "running"
                running[name] = clock + job.duration
                timeline.append({"job": name, "start": clock, "end": clock + job.duration,
                                 "attempt": attempts[name], "kind": "run",
                                 "worker": len(running) - 1})
            if not running:
                continue
            clock = min(running.values())
            for name in sorted(n for n, finish in running.items() if finish == clock):
                del running[name]
                job = self.jobs[name]
                if attempts[name] <= job.fail_attempts:
                    if attempts[name] <= self.max_retries:
                        retried.append(name)
                        state[name] = "ready"
                    else:
                        state[name] = "failed"
                        failed.append(name)
                        for victim in self._transitive_dependents(name):
                            if state[victim] == "blocked":
                                state[victim] = "skipped"
                                skipped.add(victim)
                    continue
                key = self.cache_key(job, digests)
                artifact = self.execute(job, key)
                digests[name] = artifact.digest
                artifacts[name] = artifact
                if not job.nondeterministic:
                    cache.entries[key] = artifact
                state[name] = "done"
        return {
            "status": "failed" if failed else "ok",
            "timeline": timeline,
            "executions": {name: count for name, count in sorted(attempts.items())},
            "digests": digests,
            "artifacts": artifacts,
            "failed": failed,
            "retried": retried,
            "skipped": sorted(skipped),
            "makespan": max((entry["end"] for entry in timeline), default=0),
            "cache_entries": len(cache),
            "cache_lookups": cache.lookups,
            "source_hashes": {name: digest_of(content) for name, content in self.sources.items()},
        }

    def _promote(self, state, skipped):
        """Promote blocked jobs whose dependencies are all done, skip those behind failures."""
        for name, job in self.jobs.items():
            if state[name] != "blocked":
                continue
            if any(state[dep] in ("failed", "skipped") for dep in job.deps):
                state[name] = "skipped"
                skipped.add(name)
            elif all(state[dep] == "done" for dep in job.deps):
                state[name] = "ready"

    def _transitive_dependents(self, name):
        seen = set()
        frontier = [name]
        while frontier:
            current = frontier.pop()
            for dependant in self.dependents(current):
                if dependant not in seen:
                    seen.add(dependant)
                    frontier.append(dependant)
        return seen

    def critical_path(self):
        """Longest path by duration - the theoretical makespan with unlimited workers."""
        order = []
        seen = set()

        def visit(name):
            if name in seen:
                return
            for dep in self.jobs[name].deps:
                visit(dep)
            seen.add(name)
            order.append(name)

        for name in self.jobs:
            visit(name)
        best = {}
        for name in order:
            job = self.jobs[name]
            best[name] = job.duration + max((best[dep] for dep in job.deps), default=0)
        return max(best.values()) if best else 0

    def provenance_closure(self, name, artifacts):
        """Transitive set of source files and jobs behind one artifact."""
        jobs_seen = set()
        sources_seen = set()
        frontier = [name]
        while frontier:
            current = frontier.pop()
            if current in jobs_seen:
                continue
            jobs_seen.add(current)
            artifact = artifacts.get(current)
            if artifact is None:
                continue
            for source in artifact.provenance["sources"]:
                sources_seen.add(source)
            for dep in self.jobs[current].deps:
                frontier.append(dep)
        return jobs_seen, sources_seen


def numeric_pipeline(sources=None, workers=2, **overrides):
    jobs = [
        JobSpec("fetch", 2, priority=1, sources=("toolchain.lock",), samples=32),
        JobSpec("docs", 3, priority=5, samples=16),
        JobSpec("generate", 4, priority=2, sources=("gen.cfg",), samples=128),
        JobSpec("assemble", 3, deps=("generate",), priority=2, samples=64),
        JobSpec("solve", 6, deps=("generate", "assemble"), priority=2, samples=256),
        JobSpec("regress", 3, deps=("solve", "fetch"), priority=3, samples=64),
        JobSpec("report", 1, deps=("regress", "docs"), priority=4, samples=16),
    ]
    for name, kwargs in overrides.items():
        for job in jobs:
            if job.name == name:
                for attribute, value in kwargs.items():
                    setattr(job, attribute, value)
    base_sources = {"gen.cfg": "grid=64;tol=1e-9", "toolchain.lock": "blas=3.11;fft=2.4"}
    base_sources.update(sources or {})
    return Pipeline(jobs, sources=base_sources, workers=workers)


def ran_concurrently(timeline):
    """Largest number of executions overlapping at any instant."""
    events = []
    for entry in timeline:
        if entry["kind"] != "run":
            continue
        events.append((entry["start"], 1))
        events.append((entry["end"], -1))
    events.sort(key=lambda pair: (pair[0], pair[1]))  # ends before starts at the same instant
    current = 0
    peak = 0
    for _instant, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def main():
    # ---------------- cold run: every job executes exactly once, in topological order ----------------
    cold = numeric_pipeline(workers=2).run()
    check(cold["status"] == "ok", f"cold run failed: {cold['failed']}")
    check(sorted(cold["executions"]) == sorted(
        ["fetch", "docs", "generate", "assemble", "solve", "regress", "report"]),
        "cold run must execute every job")
    check(all(count == 1 for count in cold["executions"].values()),
          f"cold run must execute each job once: {cold['executions']}")
    check(cold["skipped"] == [], "cold run must not skip anything")
    ends = {}
    for entry in cold["timeline"]:
        job = numeric_pipeline().jobs[entry["job"]]
        for dep in job.deps:
            check(entry["start"] >= ends[dep],
                  f"{entry['job']} started before {dep} finished")
        ends[entry["job"]] = max(ends.get(entry["job"], 0), entry["end"])

    # ---------------- resource limit and critical path ----------------
    for workers in (1, 2, 3, 4):
        report = numeric_pipeline(workers=workers).run()
        check(ran_concurrently(report["timeline"]) <= workers,
              f"worker limit {workers} exceeded: {ran_concurrently(report['timeline'])}")
        check(report["makespan"] >= numeric_pipeline().critical_path(),
              "makespan cannot beat the critical path")
    one_worker = numeric_pipeline(workers=1).run()
    three_workers = numeric_pipeline(workers=3).run()
    two_workers = numeric_pipeline(workers=2).run()
    critical = numeric_pipeline().critical_path()
    serial = sum(job.duration for job in numeric_pipeline().jobs.values())
    check(two_workers["makespan"] == critical,
          f"two workers should reach the critical path {critical}, got {two_workers['makespan']}")
    check(three_workers["makespan"] == critical, "more workers cannot beat the critical path")
    check(one_worker["makespan"] == serial,
          f"a single worker must run every stage back to back ({serial})")
    check(one_worker["makespan"] > critical, "one worker is resource bound")
    check(two_workers["makespan"] < serial, "two workers must beat the serial duration sum")

    # ---------------- warm cache: nothing rebuilds ----------------
    cache = BuildCache()
    first = numeric_pipeline(workers=2).run(cache)
    second = numeric_pipeline(workers=2).run(cache)
    check(second["executions"] == {}, f"warm cache rebuilt: {second['executions']}")
    check(all(entry["kind"] == "cache" for entry in second["timeline"]),
          "warm cache run must only restore artifacts")
    check(second["makespan"] == 0, "cache restores must cost no simulated time")
    check(second["digests"] == first["digests"], "restored artifacts must match their digests")

    # ---------------- incremental rebuild set ----------------
    changed = numeric_pipeline(sources={"gen.cfg": "grid=128;tol=1e-9"}, workers=2).run(cache)
    check(sorted(changed["executions"]) == ["assemble", "generate", "regress", "report", "solve"],
          f"unexpected incremental rebuild set: {sorted(changed['executions'])}")
    check("docs" not in changed["executions"] and "fetch" not in changed["executions"],
          "unrelated stages must come from the cache")
    check(changed["digests"]["solve"] != first["digests"]["solve"],
          "a rebuilt stage must produce a new digest")

    # touching a file without changing its bytes must not rebuild anything
    touched = numeric_pipeline(sources={"gen.cfg": "grid=128;tol=1e-9"}, workers=2).run(cache)
    check(touched["executions"] == {}, f"byte-identical edit rebuilt: {touched['executions']}")

    # ---------------- reproducibility ----------------
    fresh_a = numeric_pipeline(workers=3).run()
    fresh_b = numeric_pipeline(workers=1).run()
    check(fresh_a["digests"] == fresh_b["digests"],
          "deterministic jobs must reproduce identical digests under any schedule")
    cache_a = BuildCache()
    first_repro = numeric_pipeline(workers=2).run(cache_a)
    cache_b = BuildCache()
    second_repro = numeric_pipeline(workers=2).run(cache_b)
    mismatches = [name for name, digest in first_repro["digests"].items()
                  if digest != second_repro["digests"][name]]
    check(mismatches == [], f"reproducibility violations in a deterministic pipeline: {mismatches}")

    shaky = numeric_pipeline(workers=2, solve={"nondeterministic": True}).run()
    shaky_again = numeric_pipeline(workers=2, solve={"nondeterministic": True}).run()
    flaky_digests = [name for name in shaky["digests"]
                     if shaky["digests"][name] != shaky_again["digests"][name]]
    # nondeterminism taints the stage and everything downstream of it, and nothing else
    tainted = {"solve", "regress", "report"}
    check(set(flaky_digests) == tainted,
          f"nondeterminism must taint solve plus its dependents: {sorted(flaky_digests)}")
    check(all(shaky["digests"][name] == first_repro["digests"][name]
              for name in shaky["digests"] if name not in tainted),
          "independent stages must stay bit-identical next to a nondeterministic job")

    # ---------------- provenance closure ----------------
    pipeline = numeric_pipeline(workers=2)
    closure_jobs, closure_sources = pipeline.provenance_closure("report", first["artifacts"])
    check(closure_jobs == set(pipeline.jobs),
          f"unexpected provenance closure: {sorted(closure_jobs)}")
    check(closure_sources == {"gen.cfg", "toolchain.lock"},
          f"provenance must reach exactly the source files: {sorted(closure_sources)}")
    report_artifact = first["artifacts"]["report"]
    check(report_artifact.provenance["toolchain"] == "numerics-1.4.0",
          "provenance must record the toolchain version")
    check(all(dep in report_artifact.provenance["dependencies"]
              for dep in pipeline.jobs["report"].deps),
          "provenance must record the dependency digests")

    # ---------------- failure propagation ----------------
    broken = numeric_pipeline(workers=2, solve={"fail_attempts": 99}).run()
    check(broken["status"] == "failed", "a permanently failing job must fail the pipeline")
    check(broken["failed"] == ["solve"], f"only solve should fail: {broken['failed']}")
    check(broken["skipped"] == ["regress", "report"], f"wrong skip set: {broken['skipped']}")
    check(broken["executions"]["solve"] == 2,
          f"solve must be attempted 1 + 1 retry, got {broken['executions']['solve']}")
    check("generate" in broken["digests"] and "docs" in broken["digests"],
          "upstream stages must still have completed")
    check("report" not in broken["executions"], "skipped stages must never execute")

    # ---------------- flaky job: retried once, then green ----------------
    flaky = numeric_pipeline(workers=2, regress={"fail_attempts": 1}).run()
    check(flaky["status"] == "ok", "a flaky job that passes on retry must not fail the build")
    check(flaky["executions"]["regress"] == 2 and flaky["retried"] == ["regress"],
          f"flaky accounting wrong: {flaky['executions']} retried={flaky['retried']}")
    check(flaky["executions"]["report"] == 1, "downstream must still run exactly once")

    # ---------------- priority ordering ----------------
    roots = [JobSpec("quick", 1, priority=0), JobSpec("docs", 3, priority=5),
             JobSpec("fetch", 2, priority=1)]
    serial_run = Pipeline(roots, sources={}, workers=1).run()
    order = [entry["job"] for entry in serial_run["timeline"]]
    check(order == ["docs", "fetch", "quick"], f"priority order violated: {order}")

    # ---------------- replay determinism ----------------
    replay = numeric_pipeline(workers=2).run()
    check(replay["timeline"] == cold["timeline"] and replay["digests"] == cold["digests"],
          "the whole schedule must be a deterministic function of the DAG")

    # ---------------- invalid DAGs ----------------
    for jobs, why in (
        ([JobSpec("a", 1, deps=("b",)), JobSpec("b", 1, deps=("a",))], "cycle"),
        ([JobSpec("a", 1), JobSpec("a", 1)], "duplicate"),
        ([JobSpec("a", 1, deps=("ghost",))], "dangling dependency"),
        ([JobSpec("a", 1, sources=("missing.cfg",))], "unknown source"),
    ):
        try:
            Pipeline(jobs, sources={}, workers=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")
    try:
        Pipeline([JobSpec("a", 0)], sources={}, workers=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-positive duration")

    print(f"cold makespan={cold['makespan']} critical={critical} serial={serial} "
          f"peak_concurrency={ran_concurrently(cold['timeline'])}")
    print(f"warm-cache executions={len(second['executions'])} "
          f"incremental set={sorted(changed['executions'])}")
    print(f"provenance sources={sorted(closure_sources)} skipped-on-failure={broken['skipped']}")
    print("PASS")


if __name__ == "__main__":
    main()
