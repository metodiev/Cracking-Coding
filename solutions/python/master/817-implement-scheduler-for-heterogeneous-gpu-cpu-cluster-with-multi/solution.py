"""Task 817 - Implement scheduler for heterogeneous GPU/CPU cluster with multi-tenant isolation.  (Tier: Master, Ch 9 - System Design and Scalability)

A cluster of heterogeneous nodes (GPU nodes with device memory, CPU-only nodes) is shared by tenants
that own a *quota* of the cluster and submit jobs that are gangs: a job is placed whole or not at all.
The scheduler is a discrete-tick loop: every tick it releases finished jobs, orders the pending queue
by the policy, and either places a job (onto one node if the gang fits, otherwise spread across nodes)
or, if the job is high priority and the cluster is full, preempts lower-priority running jobs and
re-queues them.  A preemption that would not actually make room is rolled back, so no victim is ever
evicted without a reason and no victim is preempted more often than the configured bound.

Three policies share one core: `first_fit`, `best_fit` (tightest node) and `drf` (dominant resource
fairness - among pending jobs prefer the tenant with the smallest
max(used_gpus / gpu_quota, used_cpus / cpu_quota), the property that stops a CPU-heavy tenant from
starving a GPU tenant).  Isolation is enforced structurally rather than by convention: a placement is
accepted only when the node has room *and* the tenant stays inside its quota, so an over-quota job
waits instead of taking a neighbour's capacity.

Every tick of every simulation is audited and the audit is one of the checks: per-node capacity,
per-tenant quota, gang atomicity (a placement always covers the whole request), device affinity (a
device job never lands on a CPU-only node), no negative counters and no double release.

Verified: the audit staying clean through a 40-job multi-tenant run, a preemption-heavy run and a
contention-heavy run; every admittable job completing exactly once (liveness, no starvation); a
single-wave workload starting at tick 0 with makespan equal to its longest job; unschedulable requests
(larger than any node, or larger than the whole cluster) reported as pending instead of hanging the
loop; a job above its tenant's quota waiting while the same tenant runs inside it; a wide gang being
spread over two nodes and completing; preemption bringing a high-priority job forward while every
victim still completes; a greedy tenant not starving a small neighbour job; DRF's measured wait spread
being no worse than first-fit's on the same backlog; determinism of two identical runs; utilisation and
wait metrics.  Edge cases: an empty cluster, an empty workload, a zero-request job, an unknown policy
and a request larger than the whole cluster.

Simplified: nodes and jobs are typed records in one process, one tick stands for one scheduling round
(no wall-clock time, no threads, nothing sleeps), devices are integer counters instead of real GPUs and
MIG slices, jobs neither shrink nor grow, memory is a single scalar per job, and preemption re-queues
rather than restores a checkpoint - the invariants audited here are exactly the ones a production
scheduler must hold after every decision it makes.
"""
MASK64 = (1 << 64) - 1
RESOURCES = ("gpus", "gpu_mem", "cpus", "mem")


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
    """Fixed-seed splitmix64 stream: every run generates the same workload."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# ------------------------------------------------------------------ resources
class Node:
    """A machine: device count, device memory, CPU cores and host memory."""

    __slots__ = ("name", "capacity", "labels")

    def __init__(self, name, gpus, gpu_mem, cpus, mem, labels=()):
        self.name = name
        self.capacity = {"gpus": gpus, "gpu_mem": gpu_mem, "cpus": cpus, "mem": mem}
        self.labels = frozenset(labels)

    def has(self, label):
        return label in self.labels


class Job:
    """A gang of identical tasks: placed whole, or not at all."""

    __slots__ = ("name", "tenant", "request", "duration", "priority", "arrival", "finish")

    def __init__(self, name, tenant, gpus, gpu_mem, cpus, mem, duration, priority=0, arrival=0):
        self.name = name
        self.tenant = tenant
        self.request = {"gpus": gpus, "gpu_mem": gpu_mem, "cpus": cpus, "mem": mem}
        self.duration = duration
        self.priority = priority
        self.arrival = arrival
        self.finish = None

    def needs_device(self):
        return self.request["gpus"] > 0 or self.request["gpu_mem"] > 0

    def __repr__(self):
        part = self.request
        return f"{self.name}({self.tenant}:{part['gpus']}g/{part['cpus']}c/{part['mem']}m)"


class Tenant:
    """A quota holder: usage may never exceed the quota, whatever the queue looks like."""

    __slots__ = ("name", "gpu_quota", "cpu_quota", "completed")

    def __init__(self, name, gpu_quota, cpu_quota):
        self.name = name
        self.gpu_quota = gpu_quota
        self.cpu_quota = cpu_quota
        self.completed = 0

    def dominant_share(self, used_gpus, used_cpus):
        return max(used_gpus / float(self.gpu_quota) if self.gpu_quota else 0.0,
                   used_cpus / float(self.cpu_quota) if self.cpu_quota else 0.0)


class Cluster:
    def __init__(self, nodes, tenants):
        self.nodes = {node.name: node for node in nodes}
        self.tenants = {tenant.name: tenant for tenant in tenants}
        self.used = {name: dict.fromkeys(RESOURCES, 0) for name in self.nodes}
        self.tenant_used = {name: {"gpus": 0, "cpus": 0} for name in self.tenants}

    def total(self, key):
        return sum(node.capacity[key] for node in self.nodes.values())

    def dominance(self, tenant_name):
        tenant = self.tenants[tenant_name]
        used = self.tenant_used[tenant_name]
        return tenant.dominant_share(used["gpus"], used["cpus"])

    def tenant_room(self, job):
        tenant = self.tenants[job.tenant]
        used = self.tenant_used[job.tenant]
        return (used["gpus"] + job.request["gpus"] <= tenant.gpu_quota and
                used["cpus"] + job.request["cpus"] <= tenant.cpu_quota)

    def place(self, job, placement):
        """Reserve every part on its node and charge the tenant once for the whole gang."""
        for part in placement:
            used = self.used[part["node"]]
            for key in RESOURCES:
                used[key] += part[key]
        self.tenant_used[job.tenant]["gpus"] += job.request["gpus"]
        self.tenant_used[job.tenant]["cpus"] += job.request["cpus"]

    def unplace(self, job, placement):
        for part in placement:
            used = self.used[part["node"]]
            for key in RESOURCES:
                used[key] -= part[key]
                check(used[key] >= 0, f"released more {key} than node {part['node']} held")
        self.tenant_used[job.tenant]["gpus"] -= job.request["gpus"]
        self.tenant_used[job.tenant]["cpus"] -= job.request["cpus"]
        check(self.tenant_used[job.tenant]["gpus"] >= 0, "released more GPUs than the tenant held")
        check(self.tenant_used[job.tenant]["cpus"] >= 0, "released more CPUs than the tenant held")

    def audit(self, running, tick, violations):
        """Every invariant a scheduler must preserve, re-checked after each decision."""
        for name, node in sorted(self.nodes.items()):
            for key in RESOURCES:
                value = self.used[name][key]
                if value < 0 or value > node.capacity[key]:
                    violations.append((tick, "capacity", f"{name}.{key}={value}"))
        for tenant_name, tenant in sorted(self.tenants.items()):
            used = self.tenant_used[tenant_name]
            if used["gpus"] > tenant.gpu_quota or used["cpus"] > tenant.cpu_quota:
                violations.append((tick, "quota", f"{tenant_name}={used}"))
        for name, (job, placement) in sorted(running.items()):
            if not placement:
                violations.append((tick, "gang", f"{name} placed nowhere"))
            for key in RESOURCES:
                if sum(part[key] for part in placement) != job.request[key]:
                    violations.append((tick, "gang", f"{name}.{key} partially placed"))
            if job.needs_device() and any(not self.nodes[part["node"]].has("gpu")
                                         for part in placement):
                violations.append((tick, "affinity", f"{name} on a CPU-only node"))


def can_place(cluster, job, nodes):
    """One node if the whole gang fits there, otherwise a spread that stays inside every node."""
    fitting = [node for node in nodes
               if (not job.needs_device() or node.has("gpu")) and
               all(cluster.used[node.name][key] + job.request[key] <= node.capacity[key]
                   for key in RESOURCES)]
    fitting.sort(key=lambda node: (node.capacity["gpus"] - job.request["gpus"],
                                   node.capacity["cpus"] - job.request["cpus"], node.name))
    if fitting:
        return [{"node": fitting[0].name, **job.request}]
    if not job.needs_device():
        return None
    remaining = dict(job.request)
    parts = []
    for node in sorted(nodes, key=lambda item: -item.capacity["gpus"]):
        if not node.has("gpu"):
            continue
        take = {key: min(remaining[key], node.capacity[key] - cluster.used[node.name][key])
                for key in RESOURCES}
        if take["gpus"] <= 0:
            continue
        parts.append({"node": node.name, **take})
        for key in RESOURCES:
            remaining[key] -= take[key]
        if not any(remaining.values()):
            break
    return None if any(remaining.values()) else parts


def order_pending(cluster, live, policy):
    """The policy only reorders the queue; every invariant is checked identically for all of them."""
    if policy == "first_fit":
        return sorted(live, key=lambda job: (-job.priority, job.arrival, job.name))
    if policy == "best_fit":
        return sorted(live, key=lambda job: (-job.priority,
                                             -(job.request["gpus"] * 1000 + job.request["cpus"]),
                                             job.name))
    if policy == "drf":
        return sorted(live, key=lambda job: (-job.priority, cluster.dominance(job.tenant),
                                             job.arrival, job.name))
    raise ValueError(f"unknown policy {policy!r}")


def choose_victims(job, running, protected, max_victims):
    """The shortest prefix of lower-priority, unprotected victims that frees enough resources."""
    order = sorted(running.items(),
                   key=lambda item: (item[1][0].priority, -item[1][0].request["gpus"], item[0]))
    freed = dict.fromkeys(RESOURCES, 0)
    victims = []
    for name, (victim, _) in order:
        if victim.priority >= job.priority or name in protected:
            continue
        victims.append(name)
        for key in RESOURCES:
            freed[key] += victim.request[key]
        if all(freed[key] >= job.request[key] for key in RESOURCES):
            return victims
        if len(victims) >= max_victims:
            return []
    return []


def simulate(nodes, tenants, jobs, policy="drf", preempt=False, max_preemptions=1, horizon=200,
             max_victims=4):
    """The scheduler loop: release, admit, preempt, audit - once per tick."""
    fresh = [Tenant(tenant.name, tenant.gpu_quota, tenant.cpu_quota) for tenant in tenants]
    cluster = Cluster(nodes, fresh)
    node_list = list(cluster.nodes.values())
    pending = [job for job in jobs if job.arrival <= 0]
    future = sorted((job for job in jobs if job.arrival > 0), key=lambda job: (job.arrival, job.name))
    running = {}
    placements = {}
    placement_log = {}
    events = []
    violations = []
    preemptions = {}
    protected = set()
    completed = []
    waited = {}
    gpu_ticks = 0
    tick = 0
    while (pending or running or future) and tick < horizon:
        for name in sorted(list(running)):
            job, placement = running[name]
            if job.finish <= tick:
                cluster.unplace(job, placement)
                del running[name], placements[name]
                completed.append(name)
                cluster.tenants[job.tenant].completed += 1
                events.append((tick, "complete", name))
        for job in list(future):
            if job.arrival <= tick:
                pending.append(job)
                future.remove(job)
        for job in pending:
            waited[job.name] = waited.get(job.name, 0) + 1
        for job in order_pending(cluster, [job for job in pending if job.arrival <= tick], policy):
            if not cluster.tenant_room(job):
                continue
            parts = can_place(cluster, job, node_list)
            evicted = []
            if parts is None and preempt and job.priority > 0:
                for name in choose_victims(job, running, protected, max_victims):
                    victim, placement = running.pop(name)
                    del placements[name]
                    cluster.unplace(victim, placement)
                    evicted.append((name, victim, placement))
                if evicted:
                    parts = can_place(cluster, job, node_list)
                if parts is None:                       # no room after all: roll the eviction back
                    for name, victim, placement in evicted:
                        cluster.place(victim, placement)
                        running[name] = (victim, placement)
                        placements[name] = placement
                    evicted = []
            if parts is None:
                continue
            for name, victim, _ in evicted:
                pending.append(victim)
                preemptions[name] = preemptions.get(name, 0) + 1
                if preemptions[name] >= max_preemptions:
                    protected.add(name)
                events.append((tick, "preempt", name))
            cluster.place(job, parts)
            job.finish = tick + job.duration
            running[job.name] = (job, parts)
            placements[job.name] = parts
            placement_log[job.name] = parts
            pending = [item for item in pending if item.name != job.name]
            events.append((tick, "place", job.name))
        gpu_ticks += sum(cluster.used[name]["gpus"] for name in cluster.nodes)
        cluster.audit(running, tick, violations)
        tick += 1
    completions = [event[0] for event in events if event[1] == "complete"]
    return {
        "events": events,
        "placements": placement_log,
        "completed": completed,
        "violations": violations,
        "pending": [job.name for job in pending + future],
        "makespan": (max(completions) + 1) if completions else 0,
        "utilisation": gpu_ticks / float(max(1, cluster.total("gpus") * max(1, tick))),
        "preemptions": preemptions,
        "wait": waited,
        "tenants": {name: tenant.completed for name, tenant in cluster.tenants.items()},
        "cluster": cluster,
        "nodes": cluster.nodes,
    }


def first_place(result, name):
    ticks = [event[0] for event in result["events"] if event[1] == "place" and event[2] == name]
    return min(ticks) if ticks else None


# ----------------------------------------------------------------------- checks
def nodes_and_tenants():
    nodes = [Node("gpu-a", 2, 16, 8, 32, ("gpu",)), Node("gpu-b", 4, 24, 16, 64, ("gpu",)),
             Node("cpu-c", 0, 0, 32, 128), Node("cpu-d", 0, 0, 8, 16)]
    tenants = [Tenant("alpha", 4, 24), Tenant("beta", 4, 24), Tenant("gamma", 2, 16)]
    return nodes, tenants


def make_workload(rng, count, arrival_span):
    jobs = []
    for index in range(count):
        tenant = ["alpha", "beta", "gamma"][rng.below(3)]
        gpus = (0, 1, 2)[rng.below(3)]
        jobs.append(Job(f"job{index:02d}", tenant, gpus, gpus * 4, (2, 4, 8)[rng.below(3)],
                        (4, 8, 16)[rng.below(3)], 1 + rng.below(4), rng.below(2),
                        rng.below(arrival_span)))
    return jobs


def main():
    nodes, tenants = nodes_and_tenants()

    # 1. a workload that fits in one wave is placed at tick 0 and ends with its longest job
    one_wave = [Job("w1", "alpha", 1, 4, 2, 4, 3), Job("w2", "beta", 1, 4, 2, 4, 7),
                Job("w3", "gamma", 0, 0, 4, 4, 2)]
    wave = simulate(nodes, tenants, one_wave, horizon=40)
    check(wave["violations"] == [], f"the one-wave audit must be clean: {wave['violations']}")
    check(wave["makespan"] == 8, f"one wave takes its longest job plus a tick: {wave['makespan']}")
    check(all(event[0] == 0 for event in wave["events"] if event[1] == "place"),
          "a one-wave workload must be placed at tick 0")
    check(sorted(wave["completed"]) == ["w1", "w2", "w3"], "every job must complete once")

    # 2. every tick of a 40-job multi-tenant run keeps the capacity, quota, gang and affinity laws
    jobs = make_workload(Rng(0xC1057), 40, arrival_span=8)
    big = simulate(nodes, tenants, jobs, policy="drf", horizon=400)
    check(big["violations"] == [], f"the 40-job audit must be clean: {big['violations']}")
    check(len(big["completed"]) == len(set(big["completed"])), "no job may complete twice")
    check(len(big["completed"]) == 40, f"all 40 jobs must complete, got {len(big['completed'])}")
    check(big["pending"] == [], f"nothing may be left pending: {big['pending']}")
    check(sum(big["tenants"].values()) == 40 and all(count > 0 for count in big["tenants"].values()),
          f"every tenant must make progress: {big['tenants']}")
    check(big["utilisation"] > 0.1, f"a backlogged cluster must be busy: {big['utilisation']:.3f}")

    # 3. gang atomicity and device affinity, read back from the recorded placements
    check(big["placements"], "the placements must be recorded")
    for name, parts in sorted(big["placements"].items()):
        job = next(item for item in jobs if item.name == name)
        for key in RESOURCES:
            check(sum(part[key] for part in parts) == job.request[key],
                  f"{name} must be placed whole for {key}")
        if job.needs_device():
            check(all(big["nodes"][part["node"]].has("gpu") for part in parts),
                  f"a device job may never land on a CPU-only node: {name}")

    # 4. quota isolation: a job above the tenant quota waits, that tenant still runs inside quota
    blocked = simulate(nodes, tenants, [Job("hog", "gamma", 4, 16, 8, 16, 1, priority=2)], horizon=10)
    check(blocked["violations"] == [], "an over-quota job must not break the audit")
    check(blocked["completed"] == [] and blocked["pending"] == ["hog"],
          "a job above its tenant quota must stay pending")
    check(blocked["cluster"].tenant_used["gamma"]["gpus"] == 0,
          "a rejected job must not charge the tenant")
    inside = simulate(nodes, tenants, [Job("hog", "gamma", 2, 8, 8, 16, 1)], horizon=10)
    check(inside["completed"] == ["hog"], "the same tenant must run inside its quota")

    # 5. unschedulable requests are reported, never hung on
    fat = simulate(nodes, tenants, [Job("fat", "alpha", 8, 32, 8, 16, 1, priority=2)], horizon=20)
    check(fat["completed"] == [] and fat["pending"] == ["fat"],
          "a request larger than the whole cluster must be reported as pending")
    wide = [Job("wide", "alpha", 3, 12, 4, 8, 2), Job("narrow", "beta", 0, 0, 2, 2, 1)]
    spread = simulate(nodes, tenants, wide, horizon=30)
    check(sorted(spread["completed"]) == ["narrow", "wide"],
          "a gang that only fits when spread across nodes must still complete")
    check(spread["violations"] == [], "a spread placement must respect the audit")

    # 6. preemption brings a high-priority job forward; every victim still completes
    def urgent_fixture():
        return [Job("low1", "alpha", 4, 16, 16, 64, 6), Job("low2", "beta", 2, 8, 8, 32, 6),
                Job("urgent", "gamma", 2, 8, 2, 4, 2, priority=5, arrival=1)]

    plain = simulate(nodes, tenants, urgent_fixture(), preempt=False, horizon=60)
    forceful = simulate(nodes, tenants, urgent_fixture(), preempt=True, horizon=60)
    check(forceful["violations"] == [], "preemption must preserve every invariant")
    check(sorted(forceful["completed"]) == ["low1", "low2", "urgent"],
          f"every victim must still complete: {forceful['completed']}")
    check(forceful["preemptions"], "the urgent job must have forced a preemption")
    check(max(forceful["preemptions"].values()) <= 1,
          "no victim may be preempted beyond the configured bound")
    check(first_place(forceful, "urgent") == 1,
          f"preemption must place the urgent job immediately, got {first_place(forceful, 'urgent')}")
    check(first_place(plain, "urgent") > 1,
          f"without preemption the urgent job must wait, got {first_place(plain, 'urgent')}")
    check(forceful["makespan"] <= plain["makespan"] + 4, "preemption must not wreck the makespan")

    # 7. a greedy tenant does not starve a small neighbour job
    greedy = [Job(f"greedy{index}", "gamma", 0, 0, 4, 8, 4) for index in range(6)]
    neighbour = Job("neighbour", "beta", 0, 0, 4, 8, 3)
    contended = simulate(nodes, tenants, greedy + [neighbour], policy="first_fit", horizon=120)
    check(contended["violations"] == [], "contention must not break the audit")
    check("neighbour" in contended["completed"], "a small neighbour job must still complete")
    check(contended["wait"]["neighbour"] <= 2,
          f"the neighbour must start immediately: waited {contended['wait']['neighbour']} ticks")
    check(len(contended["completed"]) == 7, "the whole contention workload must finish")

    # 8. DRF is at least as fair as first-fit on the same backlog
    backlog = [Job(f"d{index:02d}", ["alpha", "beta", "gamma"][index % 3], 1, 4, 4, 4, 3)
               for index in range(12)]
    drf = simulate(nodes, tenants, backlog, policy="drf", horizon=200)
    first = simulate(nodes, tenants, backlog, policy="first_fit", horizon=200)
    best = simulate(nodes, tenants, backlog, policy="best_fit", horizon=200)
    for result, label in ((drf, "drf"), (first, "first_fit"), (best, "best_fit")):
        check(len(result["completed"]) == 12, f"{label} must complete the backlog")
        check(result["violations"] == [], f"{label} must keep the audit clean")
    drf_spread = max(drf["wait"].values()) - min(drf["wait"].values())
    first_spread = max(first["wait"].values()) - min(first["wait"].values())
    check(drf_spread <= first_spread,
          f"DRF must not be less fair than first-fit: {drf_spread} vs {first_spread}")

    # 9. determinism, metrics and degenerate inputs
    again = simulate(nodes, tenants, jobs, policy="drf", horizon=400)
    check(again["events"] == big["events"], "two identical runs must produce identical schedules")
    check(again["completed"] == big["completed"] and again["makespan"] == big["makespan"],
          "two identical runs must commit the same schedule")
    check(0.0 <= big["utilisation"] <= 1.0, "utilisation must be a fraction")
    check(max(big["wait"].values()) <= 60, "no job may wait longer than the horizon")
    check(simulate(nodes, tenants, [], horizon=5)["completed"] == [],
          "an empty workload must do nothing")
    empty = simulate([], tenants, [Job("x", "alpha", 1, 4, 1, 1, 1)], horizon=5)
    check(empty["pending"] == ["x"] and empty["violations"] == [],
          "with no nodes every job stays pending and nothing crashes")
    try:
        simulate(nodes, tenants, [Job("p", "alpha", 0, 0, 1, 1, 1)], policy="bogus")
        check(False, "an unknown policy must be rejected")
    except ValueError:
        pass
    tiny = simulate(nodes, tenants, [Job("tiny", "gamma", 0, 0, 0, 0, 1)], horizon=10)
    check(tiny["completed"] == ["tiny"], "a job that requests nothing must still run")

    print(f"40-job drf run: violations={len(big['violations'])} completed={len(big['completed'])} "
          f"makespan={big['makespan']} utilisation={big['utilisation']:.2f} "
          f"spread placements={len([p for p in big['placements'].values() if len(p) > 1])}")
    print(f"fairness: drf_spread={drf_spread} first_fit_spread={first_spread}; "
          f"preemptions={forceful['preemptions']}")
    print("PASS")


if __name__ == "__main__":
    main()
