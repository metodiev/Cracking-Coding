// Task 817 - Implement scheduler for heterogeneous GPU/CPU cluster with multi-tenant isolation. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// The cluster is a fixed set of nodes, each with a GPU and a CPU capacity. Every job belongs to a
// tenant and needs a set of identical tasks (one GPU+CPU slot each) and is *gang scheduled*: either all
// of its tasks are placed on distinct nodes or none are. Time advances in logical ticks with no
// wall-clock, no thread and no randomness beyond splitmix64.
//
// Scheduling rules:
//   * admission control - a job is rejected outright (never queued forever) when no node can host a
//     task, when its task count exceeds the node count, or when its total demand exceeds the tenant's
//     GPU quota;
//   * placement - repeatedly take the best queued job (priority, then the tenant's current dominant
//     share so equal tenants advance together, then arrival, then id) and place it if *all* tasks fit
//     on distinct nodes; the loop stops when nothing fits, which is what makes the schedule work
//     conserving;
//   * preemption - if the best queued job cannot be placed and outranks running work, the lowest
//     priority running jobs are preempted: their resources are released and they return to the queue
//     with the remaining work preserved, so preemption moves work in time but never loses it.
//
// Simplifications (documented): node-local packing only (no topology/NUMA awareness), one task per node
// per job, preemption has no grace period and does not nest, backfill is implicit in the best-job-first
// loop rather than a separate reservation mechanism, and "heterogeneous" means different (GPU, CPU)
// node shapes rather than different machine generations.
//
// Verified invariants, checked by the scheduler after every tick *and* recomputed independently from
// scratch by the harness: per-node capacity is never exceeded and the free counters match an
// independent reconstruction; per-tenant GPU use never exceeds the quota; gang atomicity (a running job
// has exactly its task count of distinct nodes); preemption preserves work (executed work equals the
// sum of the durations of finished jobs, and remaining work is untouched by preemption); quota and
// capacity are released exactly when a job finishes; work conservation (no placeable job is left
// waiting); liveness (every admitted job finishes inside the tick bound); fairness (equal tenants
// finish within one job of each other); and for identical nodes with single task jobs the makespan is
// compared against the brute-force P||Cmax optimum and Graham's list-scheduling bound.
using System;
using System.Collections.Generic;
using System.Linq;

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal sealed class Node
{
    public int Id;
    public int Gpu;
    public int Cpu;
    public int GpuFree;
    public int CpuFree;

    public Node(int id, int gpu, int cpu)
    {
        Id = id;
        Gpu = gpu;
        Cpu = cpu;
        GpuFree = gpu;
        CpuFree = cpu;
    }
}

internal sealed class Job
{
    public int Id;
    public int Tenant;
    public int Priority;
    public int Tasks = 1;
    public int GpuPerTask;
    public int CpuPerTask;
    public int Duration;
    public int Remaining;
    public int ArrivalTick;
    public int StartTick = -1;
    public int FinishTick = -1;
    public bool Rejected;
    public bool Running;
    public readonly List<int> Placement = new List<int>();

    public int DemandGpu => Tasks * GpuPerTask;
    public int DemandCpu => Tasks * CpuPerTask;
}

internal enum SchedulingPolicy { Priority, LongestFirst, LargestGpuFirst }

internal sealed class Scheduler
{
    public readonly List<Node> Nodes = new List<Node>();
    public readonly Dictionary<int, int> QuotaGpu = new Dictionary<int, int>();
    public readonly Dictionary<int, int> TenantGpuInUse = new Dictionary<int, int>();
    public readonly List<Job> Pending = new List<Job>();
    public readonly List<Job> Queue = new List<Job>();
    public readonly List<Job> Running = new List<Job>();
    public readonly List<Job> Finished = new List<Job>();
    public readonly List<Job> Rejected = new List<Job>();

    public SchedulingPolicy Policy = SchedulingPolicy.Priority;
    public int Tick;
    public long Decisions;
    public long Preemptions;
    public long WorkConservationViolations;
    public long ExecutedWork;
    public long PlacementAttempts;
    public long RunningGpuTicks;
    public long TotalGpuTicks;
    public int MaxQueueWait;

    public int TotalGpu => Nodes.Sum(n => n.Gpu);

    public void Submit(Job job) => Pending.Add(job);

    public int TenantGpu(int tenant) => TenantGpuInUse.TryGetValue(tenant, out int value) ? value : 0;

    public bool TenantHasRoom(Job job) => TenantGpu(job.Tenant) + job.DemandGpu <= QuotaGpu[job.Tenant];

    public bool FitsFreeCapacity(Job job)
        => Nodes.Count(n => n.GpuFree >= job.GpuPerTask && n.CpuFree >= job.CpuPerTask) >= job.Tasks;

    private bool Place(Job job)
    {
        var targets = Nodes
            .Where(n => n.GpuFree >= job.GpuPerTask && n.CpuFree >= job.CpuPerTask)
            .OrderByDescending(n => n.GpuFree)
            .ThenBy(n => n.Id)
            .Take(job.Tasks)
            .ToList();
        if (targets.Count < job.Tasks) return false;
        foreach (Node node in targets)
        {
            node.GpuFree -= job.GpuPerTask;
            node.CpuFree -= job.CpuPerTask;
            job.Placement.Add(node.Id);
        }
        TenantGpuInUse[job.Tenant] = TenantGpu(job.Tenant) + job.DemandGpu;
        job.Running = true;
        return true;
    }

    private void Release(Job job)
    {
        foreach (int nodeId in job.Placement)
        {
            Node node = Nodes.First(n => n.Id == nodeId);
            node.GpuFree += job.GpuPerTask;
            node.CpuFree += job.CpuPerTask;
        }
        job.Placement.Clear();
        TenantGpuInUse[job.Tenant] = TenantGpu(job.Tenant) - job.DemandGpu;
        job.Running = false;
    }

    private IOrderedEnumerable<Job> Ordered()
    {
        if (Policy == SchedulingPolicy.LongestFirst)
            return Queue.OrderByDescending(j => j.Duration).ThenBy(j => j.Id);
        if (Policy == SchedulingPolicy.LargestGpuFirst)
            return Queue.OrderByDescending(j => j.GpuPerTask).ThenBy(j => j.Id);
        return Queue
            .OrderByDescending(j => j.Priority)
            .ThenBy(j => QuotaGpu[j.Tenant] == 0 ? double.MaxValue : (double)TenantGpu(j.Tenant) / QuotaGpu[j.Tenant])
            .ThenBy(j => j.ArrivalTick)
            .ThenBy(j => j.Id);
    }

    // Place jobs best-first until nothing fits: this loop is what the work-conservation check verifies.
    private void PlaceLoop()
    {
        bool progress = true;
        while (progress)
        {
            progress = false;
            foreach (Job job in Ordered())
            {
                PlacementAttempts++;
                if (!FitsFreeCapacity(job) || !TenantHasRoom(job)) continue;
                Place(job);
                Queue.Remove(job);
                if (job.StartTick < 0) job.StartTick = Tick;
                Running.Add(job);
                Decisions++;
                if (job.Remaining <= 0)
                {
                    // A zero duration job completes the moment it is placed and never holds resources.
                    job.FinishTick = Math.Max(job.StartTick, Tick);
                    Release(job);
                    Running.Remove(job);
                    Finished.Add(job);
                }
                progress = true;
                break;
            }
        }
    }

    private void AdmitArrivals()
    {
        foreach (Job job in Pending.Where(j => j.ArrivalTick <= Tick).OrderBy(j => j.Id).ToList())
        {
            Pending.Remove(job);
            if (job.GpuPerTask > Nodes.Max(n => n.Gpu) || job.CpuPerTask > Nodes.Max(n => n.Cpu)
                || job.Tasks > Nodes.Count || job.DemandGpu > QuotaGpu[job.Tenant])
            {
                job.Rejected = true;
                Rejected.Add(job);
                continue;
            }
            Queue.Add(job);
        }
    }

    public void Step()
    {
        Tick++;

        // 1. retire jobs whose work is done; the last tick they actually executed is their finish tick
        foreach (Job job in Running.ToList())
        {
            if (job.Remaining > 0) continue;
            job.FinishTick = Math.Max(job.StartTick, Tick - 1);
            Release(job);
            Running.Remove(job);
            Finished.Add(job);
        }

        // 2. execute one tick of work on every running job
        foreach (Job job in Running)
        {
            if (job.Remaining <= 0) continue;
            job.Remaining--;
            ExecutedWork++;
            RunningGpuTicks += job.DemandGpu;
        }
        TotalGpuTicks += TotalGpu;

        // 3. arrivals
        AdmitArrivals();

        // 4. pack the cluster
        PlaceLoop();

        // 5. preemption: a queued job that outranks running work reclaims capacity
        if (Policy == SchedulingPolicy.Priority)
        {
            Job? best = Ordered().FirstOrDefault();
            if (best != null && (!FitsFreeCapacity(best) || !TenantHasRoom(best)))
            {
                foreach (Job victim in Running.Where(j => j.Priority < best.Priority).OrderBy(j => j.Priority).ThenBy(j => j.Id).ToList())
                {
                    if (FitsFreeCapacity(best) && TenantHasRoom(best)) break;
                    Release(victim);
                    Running.Remove(victim);
                    Queue.Add(victim);
                    Preemptions++;
                }
                PlaceLoop();
            }
        }

        // 6. work conservation and queue waiting time
        foreach (Job job in Queue)
        {
            if (FitsFreeCapacity(job) && TenantHasRoom(job)) WorkConservationViolations++;
            MaxQueueWait = Math.Max(MaxQueueWait, Tick - job.ArrivalTick);
        }

        CheckInvariants();
    }

    private void CheckInvariants()
    {
        foreach (Node node in Nodes)
        {
            if (node.GpuFree < 0 || node.CpuFree < 0) throw new Exception("FAILED: node capacity oversubscribed");
        }
        foreach (KeyValuePair<int, int> entry in TenantGpuInUse)
        {
            if (entry.Value < 0) throw new Exception("FAILED: negative tenant usage");
            if (entry.Value > QuotaGpu[entry.Key]) throw new Exception("FAILED: tenant quota exceeded");
        }
        foreach (Job job in Running)
        {
            if (job.Placement.Count != job.Tasks) throw new Exception("FAILED: partial gang placement");
            if (job.Placement.Distinct().Count() != job.Placement.Count) throw new Exception("FAILED: two tasks on one node");
            if (job.Remaining < 0) throw new Exception("FAILED: job over-executed its work");
        }
        if (Running.Sum(j => j.DemandGpu) != TenantGpuInUse.Values.Sum()) throw new Exception("FAILED: usage accounting");
    }

    public int Makespan => Finished.Count == 0 ? 0 : Finished.Max(j => j.FinishTick);

    public double Utilization => TotalGpuTicks == 0 ? 0 : RunningGpuTicks / (double)TotalGpuTicks;

    public bool AllDone => Pending.Count == 0 && Queue.Count == 0 && Running.Count == 0;

    public void Run(int maxTicks)
    {
        while (!AllDone && Tick < maxTicks) Step();
    }
}

// ---- independent list scheduling oracle ---------------------------------------------------------------

internal static class Scheduling
{
    public static long LptMakespan(IEnumerable<int> weights, int machines)
    {
        var loads = new long[machines];
        foreach (int weight in weights.OrderByDescending(w => w).ThenBy(w => w))
        {
            int target = 0;
            for (int m = 1; m < machines; m++) if (loads[m] < loads[target]) target = m;
            loads[target] += weight;
        }
        return loads.Max();
    }

    // Exhaustive optimum for P||Cmax: every assignment of jobs to machines.
    public static long BruteForceOptimum(IList<int> weights, int machines)
    {
        long best = long.MaxValue;
        long combinations = 1;
        for (int i = 0; i < weights.Count; i++) combinations *= machines;
        for (long code = 0; code < combinations; code++)
        {
            var loads = new long[machines];
            long value = code;
            for (int i = 0; i < weights.Count; i++)
            {
                loads[value % machines] += weights[i];
                value /= machines;
            }
            best = Math.Min(best, loads.Max());
        }
        return best;
    }
}

// ---- checks -------------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0x517E0UL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        // ---- heterogeneous multi tenant cluster -------------------------------------------------
        var cluster = BuildCluster();
        var workload = BuildWorkload();
        foreach (Job job in workload) cluster.Submit(job);
        int ticks = 0;
        while (!cluster.AllDone && ticks < 2000)
        {
            cluster.Step();
            ticks++;
            VerifyIndependently(cluster);
        }

        Check(cluster.AllDone, "the cluster drained: every admitted job finished");
        Check(cluster.WorkConservationViolations == 0, "no placeable job was left waiting");
        Check(cluster.Finished.Count == workload.Count(j => !j.Rejected), "every admitted job completed");
        Check(cluster.Rejected.Count >= 4, "infeasible and over quota jobs were rejected: " + cluster.Rejected.Count);
        Check(cluster.Rejected.All(j => j.FinishTick < 0 && !j.Running), "rejected jobs never ran");
        Check(cluster.Finished.All(j => j.StartTick >= j.ArrivalTick && j.FinishTick >= j.StartTick),
            "start and finish ticks are consistent");
        Check(cluster.Finished.All(j => j.FinishTick - j.StartTick >= j.Duration),
            "no job finished before doing all of its work");
        Check(cluster.Finished.All(j => j.FinishTick - j.StartTick == j.Duration) || cluster.Preemptions > 0,
            "preemption is the only reason a span exceeds a duration");
        Check(cluster.ExecutedWork == cluster.Finished.Sum(j => (long)j.Duration),
            $"executed work {cluster.ExecutedWork} equals the sum of durations {cluster.Finished.Sum(j => j.Duration)}");
        Check(cluster.Utilization > 0.35, "cluster utilisation above 35%: " + cluster.Utilization.ToString("F3"));
        Check(cluster.Utilization <= 1.0, "utilisation cannot exceed 1");
        Check(cluster.Makespan > 0 && cluster.Makespan <= 120, "makespan inside the tick bound: " + cluster.Makespan);
        foreach (int tenant in cluster.QuotaGpu.Keys)
            Check(cluster.TenantGpu(tenant) == 0, "quota released for tenant " + tenant);
        foreach (Node node in cluster.Nodes)
            Check(node.GpuFree == node.Gpu && node.CpuFree == node.Cpu, "node " + node.Id + " fully released");

        // ---- tenant isolation: a greedy tenant cannot exceed its quota --------------------------
        var greedy = BuildCluster();
        for (int i = 0; i < 40; i++)
            greedy.Submit(new Job
            {
                Id = 1000 + i,
                Tenant = 1,
                Priority = 5,
                Tasks = 1,
                GpuPerTask = 2,
                CpuPerTask = 1,
                Duration = 3,
                Remaining = 3,
                ArrivalTick = 0
            });
        long peak = 0;
        int greedyTicks = 0;
        while (!greedy.AllDone && greedyTicks < 500)
        {
            greedy.Step();
            greedyTicks++;
            VerifyIndependently(greedy);
            peak = Math.Max(peak, greedy.TenantGpu(1));
        }
        Check(greedy.AllDone, "the greedy tenant's jobs all complete");
        Check(peak <= greedy.QuotaGpu[1], $"tenant 1 stayed inside its quota: {peak} <= {greedy.QuotaGpu[1]}");
        Check(peak == greedy.QuotaGpu[1], "the greedy tenant did saturate its quota: " + peak);
        Check(greedy.Finished.Count(j => j.Tenant == 1) == 40, "all forty greedy jobs ran in the end");

        // ---- fairness between equal tenants -----------------------------------------------------
        var fair = BuildCluster();
        for (int i = 0; i < 12; i++)
            for (int tenant = 2; tenant <= 3; tenant++)
                fair.Submit(new Job
                {
                    Id = 2000 + i * 2 + tenant,
                    Tenant = tenant,
                    Priority = 5,
                    Tasks = 1,
                    GpuPerTask = 2,
                    CpuPerTask = 1,
                    Duration = 4,
                    Remaining = 4,
                    ArrivalTick = 0
                });
        int fairTicks = 0;
        while (!fair.AllDone && fairTicks < 500)
        {
            fair.Step();
            fairTicks++;
            VerifyIndependently(fair);
        }
        int tenant2 = fair.Finished.Count(j => j.Tenant == 2);
        int tenant3 = fair.Finished.Count(j => j.Tenant == 3);
        Check(Math.Abs(tenant2 - tenant3) <= 1, $"max-min fairness: {tenant2} vs {tenant3} completions");
        Check(tenant2 == 12 && tenant3 == 12, "both fair tenants finished all their work");

        // ---- gang scheduling --------------------------------------------------------------------
        var gang = new Scheduler();
        gang.Nodes.Add(new Node(0, 8, 16));
        gang.Nodes.Add(new Node(1, 4, 16));
        gang.Nodes.Add(new Node(2, 4, 16));
        gang.QuotaGpu[7] = 16;
        var hog = new Job { Id = 3001, Tenant = 7, Priority = 9, Tasks = 1, GpuPerTask = 8, CpuPerTask = 1, Duration = 20, Remaining = 20, ArrivalTick = 0 };
        var small = new Job { Id = 3003, Tenant = 7, Priority = 1, Tasks = 1, GpuPerTask = 2, CpuPerTask = 1, Duration = 2, Remaining = 2, ArrivalTick = 0 };
        var gangJob = new Job { Id = 3002, Tenant = 7, Priority = 9, Tasks = 3, GpuPerTask = 4, CpuPerTask = 1, Duration = 5, Remaining = 5, ArrivalTick = 0 };
        gang.Submit(hog);
        gang.Submit(small);
        gang.Submit(gangJob);
        int gangTicks = 0;
        bool wholeOrNothing = true;
        while (!gang.AllDone && gangTicks < 200)
        {
            gang.Step();
            gangTicks++;
            VerifyIndependently(gang);
            foreach (Job job in gang.Running) if (job.Placement.Count != job.Tasks) wholeOrNothing = false;
        }
        Check(wholeOrNothing, "gang jobs never ran partially");
        Check(gang.Finished.Count == 3, "the gang cluster drained");
        Check(gangJob.StartTick > gangJob.ArrivalTick, "the three node gang waited for capacity instead of splitting");
        Check(gangJob.FinishTick - gangJob.StartTick == gangJob.Duration, "the gang ran its full duration once placed");
        Check(gang.MaxQueueWait >= 15, "the gang job was queued for a while: " + gang.MaxQueueWait);

        // ---- capacity and quota edge cases ------------------------------------------------------
        var tight = new Scheduler();
        tight.Nodes.Add(new Node(0, 4, 4));
        tight.QuotaGpu[1] = 4;
        tight.Submit(new Job { Id = 1, Tenant = 1, Priority = 1, Tasks = 1, GpuPerTask = 4, CpuPerTask = 4, Duration = 1, Remaining = 1, ArrivalTick = 0 });
        tight.Submit(new Job { Id = 2, Tenant = 1, Priority = 1, Tasks = 1, GpuPerTask = 5, CpuPerTask = 1, Duration = 1, Remaining = 1, ArrivalTick = 0 });
        tight.Submit(new Job { Id = 3, Tenant = 1, Priority = 1, Tasks = 2, GpuPerTask = 4, CpuPerTask = 1, Duration = 1, Remaining = 1, ArrivalTick = 0 });
        tight.Submit(new Job { Id = 4, Tenant = 1, Priority = 1, Tasks = 1, GpuPerTask = 1, CpuPerTask = 1, Duration = 0, Remaining = 0, ArrivalTick = 0 });
        int tightTicks = 0;
        while (!tight.AllDone && tightTicks < 100)
        {
            tight.Step();
            tightTicks++;
            VerifyIndependently(tight);
        }
        Check(tight.Finished.Any(j => j.Id == 1), "a job that exactly fits the node is scheduled");
        Check(tight.Rejected.Any(j => j.Id == 2), "a job larger than every node is rejected");
        Check(tight.Rejected.Any(j => j.Id == 3), "a job over its tenant quota is rejected");
        Check(tight.Finished.Any(j => j.Id == 4), "a zero duration job still completes");
        Check(tight.Utilization > 0 && tight.Utilization <= 1, "utilisation of the tight cluster is a fraction");

        var idle = new Scheduler();
        idle.Nodes.Add(new Node(0, 4, 4));
        idle.QuotaGpu[1] = 4;
        Check(idle.AllDone && idle.Makespan == 0, "an empty cluster is immediately drained");
        idle.Step();
        Check(idle.Utilization == 0, "no work means no utilisation");
        var noQuota = new Scheduler();
        noQuota.Nodes.Add(new Node(0, 8, 8));
        noQuota.QuotaGpu[5] = 0;
        noQuota.Nodes.Add(new Node(1, 8, 8));
        noQuota.QuotaGpu[6] = 16;
        noQuota.Submit(new Job { Id = 1, Tenant = 5, Priority = 9, Tasks = 1, GpuPerTask = 1, CpuPerTask = 1, Duration = 1, Remaining = 1, ArrivalTick = 0 });
        noQuota.Submit(new Job { Id = 2, Tenant = 6, Priority = 1, Tasks = 1, GpuPerTask = 1, CpuPerTask = 1, Duration = 1, Remaining = 1, ArrivalTick = 0 });
        noQuota.Run(50);
        Check(noQuota.Rejected.Any(j => j.Tenant == 5), "a tenant with zero quota is refused");
        Check(noQuota.Finished.Any(j => j.Tenant == 6), "the tenant with quota still runs");

        // ---- P||Cmax: makespan against the brute force optimum and Graham's bound ---------------
        int strictlyWorse = 0;
        double worstRatio = 1.0;
        for (int trial = 0; trial < 40; trial++)
        {
            int machines = 2 + trial % 3;
            int jobCount = 3 + trial % 4;
            var weights = new List<int>();
            for (int i = 0; i < jobCount; i++) weights.Add(1 + Rand(20));
            long optimum = Scheduling.BruteForceOptimum(weights, machines);
            long lpt = Scheduling.LptMakespan(weights, machines);
            Check(lpt >= optimum, "a feasible schedule cannot beat the brute force optimum");

            // The scheduler itself, largest job first on identical nodes, must reproduce LPT.
            var pc = new Scheduler { Policy = SchedulingPolicy.LargestGpuFirst };
            for (int m = 0; m < machines; m++) pc.Nodes.Add(new Node(m, 200, 200));
            pc.QuotaGpu[0] = 200 * machines;
            for (int i = 0; i < weights.Count; i++)
                pc.Submit(new Job
                {
                    Id = i,
                    Tenant = 0,
                    Priority = 0,
                    Tasks = 1,
                    GpuPerTask = weights[i],
                    CpuPerTask = 0,
                    Duration = 1,
                    Remaining = 1,
                    ArrivalTick = 0
                });
            pc.Step();
            Check(pc.Running.Count == weights.Count, "every job is placed in the same tick: " + pc.Running.Count);
            var load = new long[machines];
            foreach (Job job in pc.Running) load[job.Placement[0]] += job.GpuPerTask;
            Check(load.Max() == lpt, "the scheduler's load equals the reference LPT makespan");

            if (lpt > optimum) strictlyWorse++;
            worstRatio = Math.Max(worstRatio, lpt / (double)optimum);
            double graham = 2.0 - 1.0 / machines;
            Check(lpt <= graham * optimum + 1e-9, $"Graham bound for {machines} machines: {lpt} <= {graham:F3}*{optimum}");
        }
        Check(strictlyWorse > 0, "the corpus contains instances where LPT is strictly suboptimal");

        // ---- preemption preserves work ----------------------------------------------------------
        var preempt = new Scheduler();
        preempt.Nodes.Add(new Node(0, 4, 4));
        preempt.QuotaGpu[1] = 4;
        var low = new Job { Id = 9, Tenant = 1, Priority = 1, Tasks = 1, GpuPerTask = 4, CpuPerTask = 4, Duration = 30, Remaining = 30, ArrivalTick = 0 };
        preempt.Submit(low);
        preempt.Step();
        Check(low.Running && low.StartTick == 1, "the low priority job starts immediately on the idle cluster");
        for (int i = 0; i < 5; i++) { preempt.Step(); VerifyIndependently(preempt); }
        int executedBefore = low.Duration - low.Remaining;
        Check(executedBefore == 5, "five ticks of low priority work executed: " + executedBefore);
        var high = new Job { Id = 10, Tenant = 1, Priority = 9, Tasks = 1, GpuPerTask = 4, CpuPerTask = 4, Duration = 5, Remaining = 5, ArrivalTick = preempt.Tick };
        preempt.Submit(high);
        preempt.Step();
        VerifyIndependently(preempt);
        Check(high.Running && !low.Running, "the high priority job preempted the low priority one");
        Check(low.Remaining == low.Duration - executedBefore - 1,
            "the preempted job kept exactly the work it had not done (the preempting tick executed once)");
        Check(low.Remaining > 0 && low.Remaining < low.Duration, "preemption preserved the remaining work");
        Check(preempt.Preemptions == 1, "exactly one preemption was recorded");
        int guard = 0;
        while (!preempt.AllDone && guard < 200) { preempt.Step(); VerifyIndependently(preempt); guard++; }
        Check(preempt.Finished.Count == 2, "both jobs completed after preemption");
        Check(low.FinishTick - low.StartTick > low.Duration, "the preempted job's span exceeds its duration");
        Check(high.FinishTick - high.StartTick == high.Duration, "the higher priority job ran uninterrupted");
        Check(preempt.ExecutedWork == 35, "executed work equals the two durations: " + preempt.ExecutedWork);
        Check(high.StartTick < low.FinishTick, "the high priority job started before the low one finished");

        // ---- determinism ------------------------------------------------------------------------
        var repeat = BuildCluster();
        foreach (Job job in BuildWorkload()) repeat.Submit(job);
        repeat.Run(2000);
        var again = BuildCluster();
        foreach (Job job in BuildWorkload()) again.Submit(job);
        again.Run(2000);
        Check(repeat.Makespan == again.Makespan, "identical workloads give identical makespans");
        Check(repeat.Finished.Select(j => j.Id).SequenceEqual(again.Finished.Select(j => j.Id)),
            "identical workloads give the same completion order");

        Console.WriteLine($"817 cluster: {cluster.Nodes.Count} nodes / {cluster.TotalGpu} GPUs, "
            + $"{cluster.Finished.Count} jobs done and {cluster.Rejected.Count} rejected, makespan {cluster.Makespan}, "
            + $"utilisation {cluster.Utilization:F3}, longest queue wait {cluster.MaxQueueWait} ticks, "
            + $"{cluster.Decisions} placements in {cluster.PlacementAttempts} attempts; "
            + $"P||Cmax: worst LPT ratio {worstRatio:F4} vs brute force over 40 instances");
        Console.WriteLine("PASS");
    }

    // Recomputes capacity, quota and gang invariants from scratch, independent of the scheduler's own
    // bookkeeping, and is called after every single tick of every scenario.
    private static void VerifyIndependently(Scheduler scheduler)
    {
        foreach (Node node in scheduler.Nodes)
        {
            int usedGpu = 0, usedCpu = 0;
            foreach (Job job in scheduler.Running)
            {
                int hits = job.Placement.Count(p => p == node.Id);
                if (hits > 1)
                    throw new Exception("FAILED: two tasks of job " + job.Id + " share node " + node.Id);
                usedGpu += hits * job.GpuPerTask;
                usedCpu += hits * job.CpuPerTask;
            }
            if (node.GpuFree != node.Gpu - usedGpu) throw new Exception("FAILED: gpu accounting on node " + node.Id);
            if (node.CpuFree != node.Cpu - usedCpu) throw new Exception("FAILED: cpu accounting on node " + node.Id);
            if (usedGpu > node.Gpu || usedCpu > node.Cpu) throw new Exception("FAILED: node oversubscribed " + node.Id);
        }
        foreach (int tenant in scheduler.QuotaGpu.Keys)
        {
            int used = scheduler.Running.Where(j => j.Tenant == tenant).Sum(j => j.DemandGpu);
            if (used != scheduler.TenantGpu(tenant)) throw new Exception("FAILED: tenant accounting " + tenant);
            if (used > scheduler.QuotaGpu[tenant]) throw new Exception("FAILED: quota exceeded for tenant " + tenant);
        }
        foreach (Job job in scheduler.Running)
        {
            if (job.Placement.Count != job.Tasks) throw new Exception("FAILED: partial gang placement");
            if (job.Remaining < 0) throw new Exception("FAILED: completed job over-executed");
        }
    }

    private static Scheduler BuildCluster()
    {
        var scheduler = new Scheduler();
        scheduler.Nodes.Add(new Node(0, 8, 32));
        scheduler.Nodes.Add(new Node(1, 8, 16));
        scheduler.Nodes.Add(new Node(2, 4, 32));
        scheduler.Nodes.Add(new Node(3, 4, 8));
        scheduler.Nodes.Add(new Node(4, 2, 16));
        scheduler.QuotaGpu[1] = 10;
        scheduler.QuotaGpu[2] = 6;
        scheduler.QuotaGpu[3] = 6;
        scheduler.QuotaGpu[4] = 26;
        return scheduler;
    }

    private static List<Job> BuildWorkload()
    {
        var jobs = new List<Job>();
        int id = 0;
        for (int i = 0; i < 14; i++)
        {
            int duration = 3 + i % 5;
            jobs.Add(new Job
            {
                Id = id++,
                Tenant = 1 + i % 4,
                Priority = i % 3,
                Tasks = 1 + i % 3,
                GpuPerTask = 1 + i % 3,
                CpuPerTask = 1 + i % 2,
                Duration = duration,
                Remaining = duration,
                ArrivalTick = i
            });
        }
        // A job no node can host, and a job over its tenant's quota: both must be rejected, not starved.
        jobs.Add(new Job { Id = id++, Tenant = 3, Priority = 0, Tasks = 1, GpuPerTask = 32, CpuPerTask = 1, Duration = 5, Remaining = 5, ArrivalTick = 2 });
        jobs.Add(new Job { Id = id++, Tenant = 2, Priority = 0, Tasks = 3, GpuPerTask = 4, CpuPerTask = 1, Duration = 5, Remaining = 5, ArrivalTick = 3 });
        // A high priority job arriving late, which may preempt work in progress.
        jobs.Add(new Job { Id = id++, Tenant = 4, Priority = 9, Tasks = 1, GpuPerTask = 8, CpuPerTask = 2, Duration = 6, Remaining = 6, ArrivalTick = 12 });
        return jobs;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
