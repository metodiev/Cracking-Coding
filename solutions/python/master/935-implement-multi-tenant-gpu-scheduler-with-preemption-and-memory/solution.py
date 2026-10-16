"""Task 935 - Implement multi-tenant GPU scheduler with preemption and memory reclamation.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A deterministic, in-process multi-tenant GPU scheduler.  Every device owns a first-fit block
allocator with coalescing free lists, so a job occupies [offset, offset + mem) and releasing a job
merges its block with both neighbours - the memory-reclamation path.  Tenants submit jobs (device,
memory, priority, ticks, preemptible) at fixed ticks; the scheduler admits what fits (highest
priority, then oldest submission, then job id) and a request that does not fit evicts the cheapest
strictly lower-priority preemptible victims until it fits; victims are requeued with their budget.

Simplified versus a real driver: device sharing is a tick counter, preemption is instantaneous, jobs
never migrate and an oversized job is rejected once.  Verified: no over-commit and used bytes equal
the running jobs' bytes (re-audited every tick), minimal strictly-lower-priority victim sets,
coalescing plus the fragmentation boundary, liveness, exact-fit admission, the preemptive run
against an independent non-preemptive reference run, isolation and determinism.
"""
from collections import Counter


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class BlockAllocator:
    """First-fit block allocator with coalescing free lists."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.free = [(0, capacity)]
        self.live = {}

    def used(self):
        return sum(size for _, size in self.live.values())

    def alloc(self, key, size):
        if size <= 0 or size > self.capacity:
            return None
        for index, (offset, block) in enumerate(self.free):
            if block >= size:
                self.free[index:index + 1] = ([] if block == size
                                              else [(offset + size, block - size)])
                self.live[key] = (offset, size)
                return offset
        return None

    def release(self, key):
        """Free a block and merge it with both neighbours (the reclamation path)."""
        self.free.append(self.live.pop(key))
        self.free.sort()
        merged = []
        for offset, size in self.free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        self.free = merged


class Job:
    __slots__ = ("jid", "tenant", "device", "mem", "priority", "ticks", "submit",
                 "preemptible", "remaining", "offset", "finished", "preemptions")

    def __init__(self, spec):
        (self.jid, self.tenant, self.device, self.mem, self.priority, self.ticks,
         self.submit) = spec
        self.preemptible = True
        self.remaining = self.ticks
        self.offset = -1
        self.finished = -1
        self.preemptions = 0


class Scheduler:
    """Priority scheduler; one tick runs every running job for one time slice."""

    def __init__(self, capacities, preemption=True):
        self.capacity = dict(capacities)
        self.alloc = {name: BlockAllocator(size) for name, size in capacities.items()}
        self.running = {name: [] for name in capacities}
        self.preemption = preemption
        self.now = 0
        self.pending = []
        self.rejected = []
        self.evictions = []
        self.violations = []

    def held(self, device):
        return sum(job.mem for job in self.running[device])

    def submit(self, job, tick):
        if job.device not in self.capacity or job.mem > self.capacity[job.device]:
            self.rejected.append((job.jid, job.mem, self.capacity.get(job.device, 0)))
            return False
        job.submit = tick
        job.remaining = job.ticks
        self.pending.append(job)
        return True

    def _admit(self, job):
        """Admit if a block fits, else evict the minimal strictly lower-priority victim prefix."""
        device = job.device
        alloc, running = self.alloc[device], self.running[device]
        offset = alloc.alloc(job.jid, job.mem)
        if offset is None and self.preemption:
            victims = sorted(
                (run for run in running
                 if run.preemptible and run.priority < job.priority and run.remaining > 0),
                key=lambda run: (run.priority, -run.mem, -run.remaining, run.jid))
            freed = 0
            for count, victim in enumerate(victims, start=1):
                freed += victim.mem
                if self.held(device) - freed + job.mem > self.capacity[device]:
                    continue
                for evicted in victims[:count]:
                    alloc.release(evicted.jid)
                    running.remove(evicted)
                    evicted.preemptions += 1
                    evicted.offset = -1
                    self.pending.append(evicted)
                    self.evictions.append((self.now, evicted.jid, evicted.priority,
                                           job.jid, job.priority))
                offset = alloc.alloc(job.jid, job.mem)
                break
        if offset is None:
            return False
        job.offset = offset
        running.append(job)
        return True

    def step(self):
        for job in sorted(self.pending, key=lambda item: (-item.priority, item.submit, item.jid)):
            if job in self.pending and self._admit(job):
                self.pending.remove(job)
        for device, running in self.running.items():
            for job in list(running):
                job.remaining -= 1
                if job.remaining <= 0:
                    self.alloc[device].release(job.jid)
                    running.remove(job)
                    job.finished = self.now
        self._audit()
        self.now += 1

    def _audit(self):
        """Re-check the accounting invariant and that the live blocks are disjoint and in range."""
        for device, running in self.running.items():
            held, used, capacity = (self.held(device), self.alloc[device].used(),
                                    self.capacity[device])
            spans = sorted((job.offset, job.mem) for job in running)
            if held != used or held > capacity or any(offset < 0 for offset, _ in spans):
                self.violations.append((self.now, device, held, used))
            for (offset, size), (next_offset, _) in zip(spans, spans[1:]):
                if offset + size > next_offset:
                    self.violations.append((self.now, device, offset, next_offset))
            if spans and spans[-1][0] + spans[-1][1] > capacity:
                self.violations.append((self.now, device, spans[-1]))


JOB_SPECS = (
    # jid, tenant, device, mem, priority, ticks, submit tick
    ("j1", "tenant-1", "gpu-a", 45, 1, 12, 0),
    ("j2", "tenant-2", "gpu-a", 45, 1, 12, 0),
    ("j3", "tenant-3", "gpu-a", 80, 5, 3, 2),
    ("j4", "tenant-4", "gpu-a", 150, 9, 1, 1),
    ("j5", "tenant-2", "gpu-b", 60, 2, 2, 0),
    ("j6", "tenant-1", "gpu-a", 20, 3, 4, 8),
)


def simulate(preemption=True, horizon=40):
    scheduler = Scheduler({"gpu-a": 100, "gpu-b": 60}, preemption=preemption)
    submitted = {}
    for tick in range(horizon):
        for spec in JOB_SPECS:
            if spec[6] == tick:
                job = Job(spec)
                submitted[job.jid] = job
                scheduler.submit(job, tick)
        scheduler.step()
    return scheduler, submitted


def main():
    pre, pre_jobs = simulate(preemption=True)
    plain, plain_jobs = simulate(preemption=False)  # independent non-preemptive reference
    completed = {jid for jid, job in pre_jobs.items() if job.finished >= 0}

    # safety and progress: the audit never saw an over-commit and nothing stays queued.
    check(pre.violations == [] and pre.pending == [],
          f"unsafe state: violations={pre.violations[:3]}")
    # boundary: the 150 MiB job is refused on the 100 MiB device, every other job completes.
    check([entry[0] for entry in pre.rejected] == ["j4"]
          and [entry[0] for entry in plain.rejected] == ["j4"]
          and completed == {spec[0] for spec in JOB_SPECS} - {"j4"},
          f"rejected={pre.rejected} completed={sorted(completed)}")
    # preemption policy: minimal victim prefix, strictly lower priority, first eviction at tick 2.
    check(tuple(entry[1] for entry in pre.evictions if entry[3] == "j3") == ("j1", "j2")
          and tuple(entry[1] for entry in pre.evictions if entry[3] == "j6") == ("j1",)
          and pre.evictions[0][:2] == (2, "j1")
          and all(vip < rip for _, _, vip, _, rip in pre.evictions),
          f"victim policy broken: {pre.evictions}")
    # preemptive vs reference run: the high-priority tenant wins, its victim pays twice.
    check((pre_jobs["j3"].finished, plain_jobs["j3"].finished) == (4, 14)
          and pre_jobs["j1"].finished > plain_jobs["j1"].finished
          and pre_jobs["j1"].preemptions == 2 and plain_jobs["j1"].preemptions == 0,
          "preemption must beat the reference run for j3 and delay its 45 MiB victims")
    # isolation plus the exact-fit boundary: the 60 MiB job on the 60 MiB device is untouched.
    check((pre_jobs["j5"].offset, pre_jobs["j5"].finished, pre_jobs["j5"].preemptions) == (0, 1, 0)
          == (plain_jobs["j5"].offset, plain_jobs["j5"].finished, plain_jobs["j5"].preemptions),
          "an exactly fitting job on a busy sibling device must be unaffected")
    # reclamation: each release merges with the neighbouring holes, ending in one full block.
    alloc = BlockAllocator(100)
    offsets = (alloc.alloc("a", 30), alloc.alloc("b", 20), alloc.alloc("c", 50))
    alloc.release("b")
    hole = list(alloc.free)
    alloc.release("c")
    joined = list(alloc.free)
    alloc.release("a")
    check(offsets == (0, 30, 50) and hole == [(30, 20)] and joined == [(30, 70)]
          and alloc.free == [(0, 100)] and alloc.used() == 0,
          f"reclamation broken: {offsets} {hole} {joined} {alloc.free} {alloc.used()}")
    # fragmentation boundary: total free memory does not make a contiguous block.
    frag = BlockAllocator(100)
    frag.alloc("a", 40)
    frag.alloc("b", 40)
    frag.release("a")
    check(frag.alloc("c", 60) is None and frag.used() == 40
          and frag.alloc("d", 0) is None and frag.alloc("e", 101) is None
          and frag.alloc("f", 40) == 0,
          f"fragmentation boundary broken: free={frag.free} used={frag.used()}")
    # determinism, then per-tenant accounting against the workload specification.
    again, again_jobs = simulate(preemption=True)
    check([(job.jid, job.finished, job.preemptions) for job in pre_jobs.values()]
          == [(job.jid, job.finished, job.preemptions) for job in again_jobs.values()]
          and again.evictions == pre.evictions,
          "two runs of the same workload must be identical")
    done = Counter(job.tenant for job in pre_jobs.values() if job.finished >= 0)
    expected = Counter(tenant for jid, tenant, *_ in JOB_SPECS if jid != "j4")
    check(done == expected == Counter({"tenant-1": 2, "tenant-2": 2, "tenant-3": 1}),
          f"per-tenant completion accounting: {done}")

    print(f"jobs={len(pre_jobs)} evictions={len(pre.evictions)} rejected={len(pre.rejected)}; "
          f"j3 finished preemptive={pre_jobs['j3'].finished} "
          f"non-preemptive={plain_jobs['j3'].finished}; j1 preemptive={pre_jobs['j1'].finished} "
          f"non-preemptive={plain_jobs['j1'].finished}")
    print("PASS")


if __name__ == "__main__":
    main()
