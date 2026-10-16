// Task 935 - Implement multi-tenant GPU scheduler with preemption and memory reclamation.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A discrete-time simulator of one GPU node shared by several tenants. Each job declares a
// footprint (MiB), a priority, a service time and whether it may be preempted. Each tick the
// node (a) retires jobs whose service finished, reclaiming their memory, then (b) admits waiting
// jobs in (priority desc, arrival asc, id asc) order, evicting strictly lower-priority
// preemptible residents - lowest priority first - until both node capacity and the tenant's
// memory quota fit the newcomer. An evicted job keeps its progress and returns to the waiting
// queue (checkpointed preemption), so "reclamation" frees GPU memory immediately without
// discarding work already done.
//
// The invariant battery re-derives residency from the job lists instead of trusting the
// incrementally maintained counters: resident MiB equals the sum of resident footprints,
// per-tenant usage stays inside its quota, residency never exceeds capacity, and no job is at
// once waiting, resident and completed. The independent oracle is a conservation law: because
// preemption preserves progress, executed ticks must equal the sum of all service times, and
// reclaimed memory must equal the sum of the footprints evicted.
// Simplified: one node, integer ticks, no interconnect/thermal/power model and quotas are given
// rather than negotiated - the placement and preemption policy is the delivered artefact.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class GpuJob
{
    internal readonly int Id;
    internal readonly string Tenant;
    internal readonly int MemoryMiB;
    internal readonly int Priority;
    internal readonly int ServiceTicks;
    internal readonly bool Preemptible;
    internal int Remaining;
    internal int Arrival;
    internal int DueTick;
    internal int Preemptions;
    internal int Allocations;
    internal bool Resident;
    internal bool Completed;
    internal bool Rejected;

    internal GpuJob(int id, string tenant, int memoryMiB, int priority, int serviceTicks, bool preemptible)
    {
        Id = id;
        Tenant = tenant;
        MemoryMiB = memoryMiB;
        Priority = priority;
        ServiceTicks = serviceTicks;
        Preemptible = preemptible;
        Remaining = serviceTicks;
    }
}

internal sealed class GpuScheduler
{
    internal readonly int CapacityMiB;
    private readonly Dictionary<string, int> quotas;
    private readonly List<GpuJob> waiting = new List<GpuJob>();
    private readonly List<GpuJob> resident = new List<GpuJob>();
    internal readonly List<GpuJob> Completed = new List<GpuJob>();
    internal readonly List<GpuJob> Rejected = new List<GpuJob>();
    internal int Clock;
    internal long ExecutedTicks;
    internal long MemoryReclaimed;
    internal int PreemptionEvents;
    internal int ResidentMiB;
    private readonly Dictionary<string, int> residentByTenant = new Dictionary<string, int>();

    internal GpuScheduler(int capacityMiB, Dictionary<string, int> tenantQuotas)
    {
        CapacityMiB = capacityMiB;
        quotas = tenantQuotas;
    }

    internal IReadOnlyList<GpuJob> Waiting => waiting;
    internal IReadOnlyList<GpuJob> Residents => resident;
    internal int TenantUsage(string tenant) => residentByTenant.TryGetValue(tenant, out int used) ? used : 0;
    private int QuotaOf(string tenant) => quotas.TryGetValue(tenant, out int q) ? q : int.MaxValue;

    internal void Submit(GpuJob job)
    {
        job.Arrival = Clock;
        waiting.Add(job);
    }

    private void Place(GpuJob job)
    {
        resident.Add(job);
        job.Resident = true;
        job.Allocations++;
        ResidentMiB += job.MemoryMiB;
        residentByTenant[job.Tenant] = TenantUsage(job.Tenant) + job.MemoryMiB;
    }

    private void Evict(GpuJob job)
    {
        resident.Remove(job);
        job.Resident = false;
        ResidentMiB -= job.MemoryMiB;
        residentByTenant[job.Tenant] = TenantUsage(job.Tenant) - job.MemoryMiB;
    }

    private bool FitsEver(GpuJob job) => job.MemoryMiB <= CapacityMiB && job.MemoryMiB <= QuotaOf(job.Tenant);

    /** Lowest-priority-first eviction set, or null when the newcomer cannot be made to fit. */
    private List<GpuJob> SelectVictims(GpuJob job)
    {
        int needTotal = Math.Max(0, ResidentMiB + job.MemoryMiB - CapacityMiB);
        int needTenant = Math.Max(0, TenantUsage(job.Tenant) + job.MemoryMiB - QuotaOf(job.Tenant));
        var candidates = resident.Where(r => r.Preemptible && r.Priority < job.Priority)
                                 .OrderBy(r => r.Priority).ThenByDescending(r => r.Id).ToList();
        var victims = new List<GpuJob>();
        int freedTotal = 0, freedTenant = 0;
        foreach (GpuJob c in candidates)
        {
            if (freedTotal >= needTotal && freedTenant >= needTenant) break;
            victims.Add(c);
            freedTotal += c.MemoryMiB;
            if (c.Tenant == job.Tenant) freedTenant += c.MemoryMiB;
        }
        return freedTotal >= needTotal && freedTenant >= needTenant ? victims : null;
    }

    internal void Tick()
    {
        Clock++;
        for (int i = resident.Count - 1; i >= 0; i--)
        {
            GpuJob job = resident[i];
            job.Remaining--;
            ExecutedTicks++;
            if (job.Remaining == 0)
            {
                job.Completed = true;
                Completed.Add(job);
                Evict(job);
            }
        }

        foreach (GpuJob job in waiting.OrderByDescending(j => j.Priority).ThenBy(j => j.Arrival).ThenBy(j => j.Id).ToList())
        {
            if (!FitsEver(job))
            {
                waiting.Remove(job);
                job.Rejected = true;
                Rejected.Add(job);
                continue;
            }
            if (ResidentMiB + job.MemoryMiB > CapacityMiB
                || TenantUsage(job.Tenant) + job.MemoryMiB > QuotaOf(job.Tenant))
            {
                List<GpuJob> victims = SelectVictims(job);
                if (victims == null) continue;  // cannot be made to fit yet, keep waiting
                foreach (GpuJob victim in victims)
                {
                    Evict(victim);
                    victim.Preemptions++;
                    PreemptionEvents++;
                    MemoryReclaimed += victim.MemoryMiB;
                    waiting.Add(victim);  // progress survives, the allocation does not
                }
            }
            Place(job);
            waiting.Remove(job);
        }
    }

    /** Re-derives every residency fact from the job lists and compares it with the counters. */
    internal bool Invariants(out string detail)
    {
        detail = "ok";
        int sum = resident.Sum(r => r.MemoryMiB);
        if (sum != ResidentMiB) { detail = "resident MiB counter " + ResidentMiB + " != " + sum; return false; }
        if (sum > CapacityMiB) { detail = "capacity exceeded: " + sum + " > " + CapacityMiB; return false; }
        foreach (string tenant in resident.Select(r => r.Tenant).Distinct())
        {
            int used = resident.Where(r => r.Tenant == tenant).Sum(r => r.MemoryMiB);
            if (used != TenantUsage(tenant)) { detail = "tenant " + tenant + " counter mismatch"; return false; }
            if (used > QuotaOf(tenant)) { detail = "tenant " + tenant + " quota exceeded: " + used; return false; }
        }
        foreach (GpuJob job in resident)
        {
            if (!job.Resident || job.Completed || job.Rejected) { detail = "job " + job.Id + " resident but not runnable"; return false; }
            if (waiting.Contains(job)) { detail = "job " + job.Id + " both resident and waiting"; return false; }
        }
        foreach (GpuJob job in waiting)
        {
            if (job.Resident || job.Completed || job.Rejected) { detail = "job " + job.Id + " waiting but not pending"; return false; }
        }
        if (resident.Select(r => r.Id).Distinct().Count() != resident.Count) { detail = "duplicate residency"; return false; }
        if (Completed.Select(c => c.Id).Distinct().Count() != Completed.Count) { detail = "a job completed twice"; return false; }
        return true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Dictionary<string, int> Quotas() => new Dictionary<string, int> { { "A", 6144 }, { "B", 6144 } };

    /** Deterministic 24-tick arrival script; every call yields fresh, un-run job objects. */
    private static List<GpuJob> GenerateJobs()
    {
        var jobs = new List<GpuJob>();
        string[] tenants = { "A", "B" };
        ulong state = 0x5EED0935UL;
        int id = 0;
        for (int tick = 1; tick <= 24; tick++)
        {
            int arrivals = 1 + (int)(SplitMix64.Next(ref state) % 2);
            for (int a = 0; a < arrivals; a++)
            {
                ulong r = SplitMix64.Next(ref state);
                var job = new GpuJob(id++, tenants[(int)(r % 2)],
                    1024 * (1 + (int)((r >> 8) % 4)),
                    1 + (int)((r >> 16) % 3),
                    1 + (int)((r >> 24) % 5),
                    ((r >> 32) & 1UL) == 1UL);
                job.DueTick = tick;
                jobs.Add(job);
            }
        }
        return jobs;
    }

    private static GpuScheduler DriveScripted(List<GpuJob> jobs, int ticks, out bool invariantsHeld, out string why)
    {
        var scheduler = new GpuScheduler(8192, Quotas());
        invariantsHeld = true;
        why = "ok";
        for (int tick = 1; tick <= ticks; tick++)
        {
            foreach (GpuJob job in jobs) if (job.DueTick == tick) scheduler.Submit(job);
            scheduler.Tick();
            if (!scheduler.Invariants(out why)) { invariantsHeld = false; break; }
        }
        return scheduler;
    }

    public static void Main()
    {
        // 1-2. Invariants after every tick, then liveness and exactly-once completion.
        List<GpuJob> submitted = GenerateJobs();
        GpuScheduler scheduler = DriveScripted(submitted, 80, out bool clean, out string why);
        Check(clean, "residency invariants hold after every one of the 80 ticks (" + why + ")");
        Check(submitted.All(j => j.Completed) && scheduler.Completed.Count == submitted.Count
              && scheduler.Completed.Select(j => j.Id).Distinct().Count() == submitted.Count,
            "every submitted job completes exactly once (" + scheduler.Completed.Count + "/" + submitted.Count + ")");

        // 3-4. Conservation oracle: preemption preserves progress, so no service time is lost.
        long service = submitted.Sum(j => (long)j.ServiceTicks);
        Check(scheduler.ExecutedTicks == service,
            "executed ticks " + scheduler.ExecutedTicks + " equal total service time " + service);
        long expectedReclaimed = submitted.Sum(j => (long)j.Preemptions * j.MemoryMiB);
        Check(scheduler.MemoryReclaimed == expectedReclaimed
              && scheduler.PreemptionEvents == submitted.Sum(j => j.Preemptions),
            "reclaimed memory " + scheduler.MemoryReclaimed + " MiB equals preemptions x footprint");
        Check(scheduler.ResidentMiB == 0 && scheduler.Residents.Count == 0, "the node is fully drained at the end");
        Console.WriteLine("  workload: " + submitted.Count + " jobs, " + scheduler.PreemptionEvents
            + " preemptions, " + scheduler.MemoryReclaimed + " MiB reclaimed, " + scheduler.ExecutedTicks + " GPU ticks");

        // 5. Scripted preemption: a high-priority arrival displaces the running low-priority job.
        var node = new GpuScheduler(4096, new Dictionary<string, int> { { "A", 4096 }, { "B", 4096 } });
        var low = new GpuJob(0, "A", 4096, 1, 6, true);
        var high = new GpuJob(1, "B", 4096, 3, 2, true);
        node.Submit(low);
        node.Tick();
        Check(low.Resident && low.Remaining == 6, "the low-priority job is admitted and holds the whole node");
        node.Submit(high);
        node.Tick();
        Check(high.Resident && !low.Resident && low.Preemptions == 1 && node.MemoryReclaimed == 4096,
            "the arrival preempts it and exactly one eviction reclaims the whole footprint");
        node.Tick();
        node.Tick();
        Check(high.Completed && low.Resident && low.Remaining == 5,
            "the preemptor finishes, then the evicted job resumes with its progress intact");

        // 6. Negative check: a non-preemptible job is never evicted; the newcomer waits instead.
        var strict = new GpuScheduler(2048, new Dictionary<string, int>());
        var pinned = new GpuJob(0, "A", 2048, 1, 5, false);
        var urgent = new GpuJob(1, "B", 1024, 9, 1, true);
        strict.Submit(pinned);
        strict.Tick();
        strict.Submit(urgent);
        strict.Tick();
        Check(pinned.Preemptions == 0 && pinned.Resident, "a non-preemptible job is never evicted");
        Check(!urgent.Resident && strict.Waiting.Contains(urgent), "the higher-priority job waits for the pinned slice");
        for (int i = 0; i < 5; i++) strict.Tick();
        Check(urgent.Completed, "it is admitted as soon as the pinned job releases the node");

        // 7. Victim choice: the lowest-priority eligible resident goes first.
        var choice = new GpuScheduler(6144, new Dictionary<string, int>());
        var lowPri = new GpuJob(0, "A", 2048, 1, 9, true);
        var midPri = new GpuJob(1, "B", 2048, 2, 9, true);
        var big = new GpuJob(2, "C", 4096, 5, 1, true);
        choice.Submit(lowPri);
        choice.Submit(midPri);
        choice.Tick();
        Check(lowPri.Resident && midPri.Resident, "both background jobs are resident before the arrival");
        choice.Submit(big);
        choice.Tick();
        Check(big.Resident && !lowPri.Resident && midPri.Resident,
            "eviction picks the lowest-priority resident, not the oldest one");

        // 8-9. Boundaries: oversized jobs and a zero-capacity node reject instead of thrashing.
        var tiny = new GpuScheduler(1024, new Dictionary<string, int>());
        var tooBig = new GpuJob(0, "A", 2048, 9, 1, true);
        tiny.Submit(tooBig);
        tiny.Tick();
        Check(tooBig.Rejected && !tooBig.Resident && tiny.Rejected.Count == 1,
            "a job larger than the node is rejected permanently");
        var empty = new GpuScheduler(0, new Dictionary<string, int>());
        var anyJob = new GpuJob(0, "A", 1, 1, 1, true);
        empty.Submit(anyJob);
        for (int i = 0; i < 3; i++) empty.Tick();
        Check(anyJob.Rejected && empty.Completed.Count == 0 && empty.ExecutedTicks == 0,
            "a zero-capacity node runs nothing and rejects the job");

        // 10. Determinism: the same workload replays to the same completion order and statistics.
        List<GpuJob> submittedAgain = GenerateJobs();
        GpuScheduler again = DriveScripted(submittedAgain, 80, out bool _, out string _);
        Check(scheduler.Completed.Select(j => j.Id).SequenceEqual(again.Completed.Select(j => j.Id))
              && again.ExecutedTicks == scheduler.ExecutedTicks && again.MemoryReclaimed == scheduler.MemoryReclaimed
              && submitted.Count == submittedAgain.Count,
            "two runs of the same scripted workload agree bit for bit");

        Console.WriteLine("PASS");
    }
}
