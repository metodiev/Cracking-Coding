"""Task 954 - Implement CI pipeline that runs on multiple JDKs and OSes with resource limits.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Pipeline model: a matrix expansion turns the (os, jdk) axes plus include/exclude entries into job
definitions, and a discrete-event simulator schedules them tick by tick against the runner pool's
cpu, memory and parallelism limits.  A job that demands more than the whole pool can never start and
is rejected up front; a job that fails permanently can cancel the queued remainder under fail-fast;
a flaky job is requeued while attempts remain.  After every start the simulator records the pool's
peak usage, and the whole schedule is re-audited by an independent pass over the timeline.

Simplified: ticks stand in for minutes, jobs are closed-form (cpu/memory/duration) rather than real
toolchain runs, and nothing is executed.  Verified: matrix expansion and exclusions, the resource
invariant (independently recomputed from the timeline), the serialised and parallel makespans against
hand-computed values, makespan lower bounds, fail-fast cancellation, retry accounting, rejected jobs
and the empty/single-job boundaries.
"""
OSES = ("ubuntu-latest", "windows-latest", "macos-latest")
JDKS = (17, 21)
POOL = {"cpu": 4, "memory_mb": 8192, "max_parallel": 3}
# (cpu, memory_mb, duration) per OS; JDK 21 costs one extra tick on every platform.
DEMAND = {"ubuntu-latest": (1, 2048, 3),
          "windows-latest": (2, 3072, 5),
          "macos-latest": (1, 2560, 4)}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def job(name, os_name, jdk, cpu=None, memory_mb=None, duration=None, **extra):
    base_cpu, base_memory, base_duration = DEMAND[os_name]
    record = {"id": name, "os": os_name, "jdk": jdk,
              "cpu": base_cpu if cpu is None else cpu,
              "memory_mb": base_memory if memory_mb is None else memory_mb,
              "duration": (base_duration if duration is None else duration) + (jdk - 17) // 4}
    record.update(extra)
    return record


def expand_matrix(oses=OSES, jdks=JDKS, exclude=(), include=()):
    """Axes -> jobs, minus exclusions, plus extra entries; always sorted by id."""
    jobs = [job(f"{os_name}-jdk{jdk}", os_name, jdk) for os_name in oses for jdk in jdks]
    jobs = [item for item in jobs if (item["os"], item["jdk"]) not in set(exclude)]
    for entry in include:
        jobs.append(job(entry["id"], entry["os"], entry["jdk"],
                        entry.get("cpu"), entry.get("memory_mb"), entry.get("duration")))
    return sorted(jobs, key=lambda item: item["id"])


def reject_unschedulable(jobs, limits):
    """Jobs whose own demand exceeds the whole pool can never run - reject them up front."""
    accepted, rejected = [], []
    for item in jobs:
        if item["cpu"] > limits["cpu"] or item["memory_mb"] > limits["memory_mb"]:
            rejected.append((item["id"], f"needs {item['cpu']}cpu/{item['memory_mb']}MB"))
        else:
            accepted.append(item)
    return accepted, rejected


def simulate(jobs, limits, fail_fast=False):
    """Tick-by-tick scheduler; returns the timeline, makespan, peak usage and cancellations."""
    queue = sorted((dict(item, attempt=0) for item in jobs), key=lambda item: item["id"])
    running, timeline, cancelled, failed = [], {}, [], []
    peak = {"cpu": 0, "memory_mb": 0, "parallel": 0}
    tick = 0
    while queue or running:
        for row in [row for row in running if row["end"] <= tick]:
            running.remove(row)
            timeline[row["job"]["id"]] = (row["start"], row["end"])
            flaky = row["attempt"] < row["job"].get("fails_until", 0)
            if flaky and row["attempt"] + 1 < row["job"].get("attempts", 1):
                queue.append(dict(row["job"], attempt=row["attempt"] + 1))
            elif flaky:
                failed.append(row["job"]["id"])
        if fail_fast and failed:
            cancelled.extend(item["id"] for item in queue)
            queue.clear()
        for item in list(queue):
            if len(running) >= limits["max_parallel"]:
                break
            cpu = sum(row["job"]["cpu"] for row in running) + item["cpu"]
            memory = sum(row["job"]["memory_mb"] for row in running) + item["memory_mb"]
            if cpu > limits["cpu"] or memory > limits["memory_mb"]:
                continue
            queue.remove(item)
            running.append({"job": item, "start": tick, "end": tick + item["duration"],
                            "attempt": item["attempt"]})
            peak = {"cpu": max(peak["cpu"], cpu), "memory_mb": max(peak["memory_mb"], memory),
                    "parallel": max(peak["parallel"], len(running))}
        tick += 1
        if tick > 500:
            raise RuntimeError("simulation did not converge")
    return {"timeline": timeline, "makespan": max((end for _, end in timeline.values()), default=0),
            "cancelled": sorted(cancelled), "failed": sorted(failed), "peak": peak}


def audit_timeline(timeline, jobs, limits):
    """Independent oracle: recompute pool usage per tick straight from the timeline."""
    demand = {item["id"]: item for item in jobs}
    makespan = max((end for _, end in timeline.values()), default=0)
    peak = {"cpu": 0, "memory_mb": 0, "parallel": 0}
    for tick in range(makespan):
        active = [name for name, (start, end) in timeline.items() if start <= tick < end]
        peak["cpu"] = max(peak["cpu"], sum(demand[name]["cpu"] for name in active))
        peak["memory_mb"] = max(peak["memory_mb"],
                                sum(demand[name]["memory_mb"] for name in active))
        peak["parallel"] = max(peak["parallel"], len(active))
    over = (peak["cpu"] > limits["cpu"] or peak["memory_mb"] > limits["memory_mb"]
            or peak["parallel"] > limits["max_parallel"])
    return peak, over


def main():
    # 1) matrix expansion: 3 x 2 axes, exclusions and includes, deterministic ordering.
    full = expand_matrix()
    reduced = expand_matrix(exclude=(("windows-latest", 21), ("macos-latest", 17)),
                            include=({"id": "extra-static", "os": "ubuntu-latest", "jdk": 17},))
    check(len(full) == 6 and [item["id"] for item in full] == sorted(item["id"] for item in full)
          and {item["id"] for item in full} == {f"{os}-jdk{jdk}" for os in OSES for jdk in JDKS}
          and len(reduced) == 5 and all(item["id"] != "windows-latest-jdk21" for item in reduced)
          and any(item["id"] == "extra-static" for item in reduced),
          f"matrix expansion wrong: {[item['id'] for item in reduced]}")

    # 2) resource invariant: the simulator's peak usage matches an independent re-audit of the
    #    timeline and never exceeds the pool.
    heavy = expand_matrix()
    run = simulate(heavy, POOL)
    audited, over = audit_timeline(run["timeline"], heavy, POOL)
    check(not over and audited == run["peak"]
          and run["peak"]["parallel"] <= POOL["max_parallel"]
          and run["peak"]["cpu"] <= POOL["cpu"]
          and run["peak"]["memory_mb"] <= POOL["memory_mb"] and len(run["timeline"]) == 6,
          f"resource limit broken: simulated {run['peak']} vs audited {audited}")

    # 3) serialised makespan: with max_parallel=1 the value is the hand-computed sum of durations.
    serial = {"cpu": 4, "memory_mb": 4096, "max_parallel": 1}
    serial_run = simulate(heavy, serial)
    expected = sum(item["duration"] for item in heavy)
    check(serial_run["makespan"] == expected == 27 and serial_run["peak"]["parallel"] == 1,
          f"serial makespan {serial_run['makespan']} != hand-computed {expected}")

    # 4) lower bounds and parallelism benefit: makespan >= max duration, >= work/cpu, and the wide
    #    pool beats the serial one.
    work = sum(item["cpu"] * item["duration"] for item in heavy)
    check(run["makespan"] >= max(item["duration"] for item in heavy)
          and run["makespan"] >= -(-work // POOL["cpu"]) and run["makespan"] < serial_run["makespan"],
          f"makespan {run['makespan']} must dominate its lower bounds and beat serial")

    # 5) fail-fast: a permanently failing job cancels the queued remainder, running jobs finish.
    fail_jobs = [job("a-fails", "ubuntu-latest", 17), job("b-ok", "ubuntu-latest", 17),
                 job("c-queued", "ubuntu-latest", 21), job("d-queued", "macos-latest", 17)]
    cancelled_run = simulate(matrix_sorted(fail_jobs, fail_until="a-fails"), POOL, fail_fast=True)
    check(cancelled_run["failed"] == ["a-fails"] and cancelled_run["cancelled"] == ["d-queued"]
          and sorted(cancelled_run["timeline"]) == ["a-fails", "b-ok", "c-queued"],
          f"fail-fast did not cancel the tail: {cancelled_run['cancelled']}")

    # 6) retries: a flaky job that fails its first attempt is requeued and passes on the second.
    retried = simulate(matrix_sorted(fail_jobs, fail_until="a-fails", attempts=2), POOL)
    check(retried["failed"] == [] and retried["timeline"]["a-fails"] == (3, 6)
          and retried["makespan"] == 7
          and sorted(retried["timeline"]) == ["a-fails", "b-ok", "c-queued", "d-queued"],
          f"the flaky job must be retried, not failed: {retried['timeline'].get('a-fails')}")

    # 7) rejected jobs: a demand larger than the pool is refused with a reason and never scheduled.
    oversized = job("huge", "windows-latest", 21, cpu=8, memory_mb=8192)
    accepted, rejected = reject_unschedulable(expand_matrix() + [oversized], POOL)
    rejected_run = simulate(accepted, POOL)
    check([name for name, _ in rejected] == ["huge"] and "needs 8cpu/8192MB" in rejected[0][1]
          and "huge" not in rejected_run["timeline"] and len(accepted) == 6,
          f"oversized jobs must be rejected: {rejected}")

    # 8) boundaries: an empty pipeline finishes in zero ticks; one job costs its duration.
    empty_run = simulate([], POOL)
    single = [job("solo", "macos-latest", 21)]
    single_run = simulate(single, POOL)
    check(empty_run["makespan"] == 0 and empty_run["timeline"] == {}
          and audit_timeline({}, [], POOL) == ({"cpu": 0, "memory_mb": 0, "parallel": 0}, False)
          and single_run["makespan"] == 5 and "solo" not in empty_run["timeline"],
          f"empty/single boundary wrong: {empty_run['makespan']} / {single_run['makespan']}")

    # 9) determinism: identical runs, and a permuted job list yields the same schedule.
    permuted = simulate(list(reversed(heavy)), POOL)
    check(simulate(heavy, POOL) == run and permuted == run,
          "the schedule must be deterministic and independent of input order")

    print(f"matrix={len(heavy)} serial_makespan={serial_run['makespan']} "
          f"parallel_makespan={run['makespan']} work={work} peak={run['peak']}")
    print(f"cancelled={cancelled_run['cancelled']} retried={sorted(retried['timeline'])} "
          f"rejected={[name for name, _ in rejected]}")
    print("PASS")


def matrix_sorted(jobs, fail_until=None, attempts=1):
    """Helper: stamp failure/attempt policy onto a list of jobs, sorted by id."""
    stamped = [dict(item, fails_until=(1 if item["id"] == fail_until else 0), attempts=attempts)
               for item in jobs]
    return sorted(stamped, key=lambda item: item["id"])


if __name__ == "__main__":
    main()
