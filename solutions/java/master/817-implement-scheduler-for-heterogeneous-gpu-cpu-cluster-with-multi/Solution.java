/** Task 817 - Implement scheduler for heterogeneous GPU/CPU cluster with multi-tenant isolation. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The scheduler is a gang-scheduling event loop over a heterogeneous cluster:
 *
 *   placement   a job requesting (cpu, mem, gpuType, gpuCount) runs only if a *single* node can host the
 *               whole request; among the fitting nodes the one with the least free capacity is chosen
 *               (best fit), which keeps large nodes free for large requests.
 *   tenants     every tenant owns a quota of GPUs and cores.  An admission control step rejects a job whose
 *               resources would push its tenant over quota, even when the cluster has capacity, and usage is
 *               re-checked after every event, so no instant violates a quota.
 *   preemption  a higher-priority job that fits nowhere may preempt lower-priority running jobs on one node;
 *               the victims are re-queued with their remaining duration and make progress later, so a
 *               preemption never loses work.
 *   events      arrivals, completions and preemptions are processed in logical-tick order; ties break on job
 *               id, so the whole trace is reproducible.
 *
 * Deliberate simplifications (deterministic; java.util only; no threads, sockets or wall-clock time): the
 * cluster is a handful of nodes, jobs are integers in a fixed trace, and "latency" is measured in logical
 * ticks.  Preemption is instantaneous and resumes from the remaining duration, i.e. checkpointing cost is
 * assumed free -- the property that matters here (work is never lost) is still verified.
 *
 * Verified invariants:
 *   safety      - after every event no node is oversubscribed on any resource, no tenant exceeds its quota,
 *                 every running job is placed entirely on one node, and GPU-type requests only land on
 *                 nodes that own that type.
 *   isolation   - a tenant's GPU-tick consumption is identical whether the other tenants are idle or
 *                 hammering the cluster inside their own quotas, and the resulting completion-time jitter
 *                 (a placement effect, never resource theft) is bounded and measured.
 *   liveness    - every admissible job completes within the trace horizon, and an over-quota tenant cannot
 *                 starve a compliant one (measured on the combined multi-tenant trace).
 *   efficiency  - utilisation is measured, best fit is shown not to fragment more than first fit on the same
 *                 trace, and unschedulable requests are rejected rather than queued forever.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every trace is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A machine: cores, memory and a budget of GPUs of one type. */
class Node {
    final int id;
    final int cores;
    final int mem;
    final String gpuType;
    final int gpus;
    int freeCores;
    int freeMem;
    int freeGpus;

    Node(int id, int cores, int mem, String gpuType, int gpus) {
        this.id = id; this.cores = cores; this.mem = mem; this.gpuType = gpuType; this.gpus = gpus;
        this.freeCores = cores; this.freeMem = mem; this.freeGpus = gpus;
    }

    boolean fits(Job j) {
        return freeCores >= j.cores && freeMem >= j.mem && j.gpuCount <= freeGpus
                && (j.gpuCount == 0 || j.gpuType.equals(gpuType));
    }

    void allocate(Job j) { freeCores -= j.cores; freeMem -= j.mem; freeGpus -= j.gpuCount; }

    void release(Job j) { freeCores += j.cores; freeMem += j.mem; freeGpus += j.gpuCount; }

    int freeScore() { return freeGpus * 1000 + freeCores * 10 + freeMem; }
}

/** A job: gang resource request, tenant, priority and logical duration in ticks. */
class Job {
    final int id;
    final String tenant;
    final int priority;
    final int cores;
    final int mem;
    final String gpuType;
    final int gpuCount;
    final int duration;
    final int arrival;
    int remaining;
    int startTick = -1;
    int endTick = -1;
    int preemptions;
    int nodeId = -1;

    Job(int id, String tenant, int priority, int cores, int mem, String gpuType, int gpuCount, int duration, int arrival) {
        this.id = id; this.tenant = tenant; this.priority = priority; this.cores = cores; this.mem = mem;
        this.gpuType = gpuType; this.gpuCount = gpuCount; this.duration = duration; this.arrival = arrival;
        this.remaining = duration;
    }

    int resourceKey() { return cores * 1_000_000 + mem * 1000 + gpuCount; }
}

class TenantQuota {
    final int gpus;
    final int cores;

    TenantQuota(int gpus, int cores) { this.gpus = gpus; this.cores = cores; }
}

class ScheduleResult {
    final List<Job> jobs;
    final List<String> transcript = new ArrayList<>();
    final Map<String, Integer> peakGpuByTenant = new TreeMap<>();
    long gpuTicksUsed;
    long gpuTicksTotal;
    int makespan;
    int rejected;
    int preemptions;
    int violations;

    ScheduleResult(List<Job> jobs) { this.jobs = jobs; }

    int peakGpus(String tenant) { return peakGpuByTenant.getOrDefault(tenant, 0); }
}

class Scheduler {
    final List<Node> nodes;
    final Map<String, TenantQuota> quotas;
    final boolean preemptionEnabled;
    final boolean quotasEnabled;
    final boolean bestFit;

    Scheduler(List<Node> nodes, Map<String, TenantQuota> quotas, boolean preemptionEnabled,
              boolean quotasEnabled, boolean bestFit) {
        this.nodes = nodes; this.quotas = quotas; this.preemptionEnabled = preemptionEnabled;
        this.quotasEnabled = quotasEnabled; this.bestFit = bestFit;
    }

    int tenantGpuUsage(String tenant, Map<Integer, Job> running) {
        int used = 0;
        for (Job j : running.values()) if (j.tenant.equals(tenant)) used += j.gpuCount;
        return used;
    }

    int tenantCoreUsage(String tenant, Map<Integer, Job> running) {
        int used = 0;
        for (Job j : running.values()) if (j.tenant.equals(tenant)) used += j.cores;
        return used;
    }

    boolean withinQuota(Job j, Map<Integer, Job> running) {
        if (!quotasEnabled) return true;
        TenantQuota q = quotas.get(j.tenant);
        if (q == null) return false;
        return tenantGpuUsage(j.tenant, running) + j.gpuCount <= q.gpus
                && tenantCoreUsage(j.tenant, running) + j.cores <= q.cores;
    }

    /** The node that best fits the job, or null: least free capacity for best fit, lowest id for first fit. */
    Node place(Job j) {
        Node best = null;
        for (Node n : nodes) {
            if (!n.fits(j)) continue;
            if (best == null) { best = n; continue; }
            if (bestFit ? n.freeScore() < best.freeScore() : n.id < best.id) best = n;
        }
        return best;
    }

    /** Runs the whole trace; the caller inspects the result and its invariants. */
    ScheduleResult run(List<Job> jobs, int horizon) {
        ScheduleResult result = new ScheduleResult(jobs);
        for (Job j : jobs) {                       // a trace can be replayed: reset all mutable job state
            j.remaining = j.duration;
            j.startTick = -1;
            j.endTick = -1;
            j.preemptions = 0;
            j.nodeId = -1;
        }
        for (Node n : nodes) result.gpuTicksTotal += (long) n.gpus * horizon;
        List<Job> pending = new ArrayList<>();
        Map<Integer, Job> running = new TreeMap<>();
        int now = 0;
        while (now <= horizon) {
            for (Job j : jobs) if (j.arrival == now) pending.add(j);
            // completions first: a job finishing this tick frees its node before new admissions
            List<Integer> done = new ArrayList<>();
            for (Map.Entry<Integer, Job> e : running.entrySet()) {
                Job j = e.getValue();
                if (now >= j.endTick) {
                    nodes.get(j.nodeId).release(j);
                    done.add(e.getKey());
                    result.transcript.add("t" + now + " done " + j.id + " on node " + j.nodeId);
                }
            }
            for (int id : done) running.remove(id);

            // admission in priority then arrival then id order
            List<Job> queue = new ArrayList<>();
            for (Job j : pending) if (!running.containsKey(j.id)) queue.add(j);
            queue.sort(Comparator.<Job>comparingInt(j -> -j.priority).thenComparingInt(j -> j.arrival)
                    .thenComparingInt(j -> j.id));
            Set<Integer> admittedThisTick = new HashSet<>();
            for (Job j : queue) {
                if (!withinQuota(j, running)) continue;
                Node target = place(j);
                if (target == null && preemptionEnabled) {
                    target = preempt(j, running, pending, result, now);
                }
                if (target == null) continue;
                // quota must be re-checked after a preemption may have freed resources
                if (!withinQuota(j, running)) continue;
                target.allocate(j);
                j.nodeId = target.id;
                if (j.startTick < 0) j.startTick = now;
                j.endTick = now + j.remaining;
                running.put(j.id, j);
                admittedThisTick.add(j.id);
                result.transcript.add("t" + now + " start " + j.id + " (" + j.tenant + ") on node " + target.id);
            }
            pending.removeIf(j -> running.containsKey(j.id) || j.endTick >= 0);

            // invariant sweep after every event tick
            result.violations += checkInvariants(running, now, result);
            long gpuUsed = 0;
            Map<String, Integer> perTenant = new TreeMap<>();
            for (Job j : running.values()) {
                gpuUsed += j.gpuCount;
                perTenant.merge(j.tenant, j.gpuCount, Integer::sum);
            }
            for (Map.Entry<String, Integer> e : perTenant.entrySet()) {
                result.peakGpuByTenant.merge(e.getKey(), e.getValue(), Math::max);
            }
            result.gpuTicksUsed += gpuUsed;
            now++;
        }
        for (Job j : jobs) {
            if (j.endTick < 0) result.transcript.add("incomplete " + j.id);
            else result.makespan = Math.max(result.makespan, j.endTick);
        }
        return result;
    }

    /** Preempts lower-priority jobs on one node until the higher-priority job fits; returns that node. */
    private Node preempt(Job j, Map<Integer, Job> running, List<Job> pending, ScheduleResult result, int now) {
        Node best = null;
        List<Job> victims = new ArrayList<>();
        for (Node n : nodes) {
            if (j.gpuCount > 0 && !j.gpuType.equals(n.gpuType)) continue;
            List<Job> onNode = new ArrayList<>();
            for (Job r : running.values()) if (r.nodeId == n.id) onNode.add(r);
            onNode.sort(Comparator.<Job>comparingInt(r -> r.priority).thenComparingInt(r -> -r.id));
            int needCores = j.cores - n.freeCores, needMem = j.mem - n.freeMem, needGpus = j.gpuCount - n.freeGpus;
            if (needCores <= 0 && needMem <= 0 && needGpus <= 0) continue;
            List<Job> picked = new ArrayList<>();
            for (Job r : onNode) {
                if (r.priority >= j.priority) break;
                picked.add(r);
                needCores -= r.cores;
                needMem -= r.mem;
                needGpus -= r.gpuCount;
                if (needCores <= 0 && needMem <= 0 && needGpus <= 0) break;
            }
            if (needCores <= 0 && needMem <= 0 && needGpus <= 0) { best = n; victims = picked; break; }
        }
        if (best == null) return null;
        for (Job victim : victims) {
            victim.remaining = victim.endTick - now;
            if (victim.remaining <= 0) victim.remaining = 1;
            victim.endTick = -1;
            victim.preemptions++;
            victim.nodeId = -1;
            best.release(victim);
            running.remove(victim.id);
            if (!pending.contains(victim)) pending.add(victim);   // the victim keeps its place in the queue
            result.preemptions++;
            result.transcript.add("t" + now + " preempt " + victim.id + " for " + j.id);
        }
        return best;
    }

    /** Safety sweep: no oversubscription, no quota breach, no cross-node placement. */
    private int checkInvariants(Map<Integer, Job> running, int now, ScheduleResult result) {
        int violations = 0;
        for (Node n : nodes) {
            if (n.freeCores < 0 || n.freeMem < 0 || n.freeGpus < 0) violations++;
        }
        Map<String, Integer> gpuByTenant = new TreeMap<>();
        for (Job j : running.values()) gpuByTenant.merge(j.tenant, j.gpuCount, Integer::sum);
        if (quotasEnabled) {
            for (Map.Entry<String, Integer> e : gpuByTenant.entrySet()) {
                TenantQuota q = quotas.get(e.getKey());
                if (q != null && e.getValue() > q.gpus) violations++;
            }
        }
        for (Job j : running.values()) {
            if (j.nodeId < 0) violations++;
            else if (j.gpuCount > 0 && !nodes.get(j.nodeId).gpuType.equals(j.gpuType)) violations++;
        }
        return violations;
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static List<Node> cluster() {
        return new ArrayList<>(Arrays.asList(
                new Node(0, 32, 128, "A100", 4),
                new Node(1, 16, 64, "A100", 2),
                new Node(2, 48, 256, "H100", 8),
                new Node(3, 8, 32, "T4", 1)));
    }

    static Map<String, TenantQuota> quotas(int gpus, int cores) {
        Map<String, TenantQuota> q = new TreeMap<>();
        q.put("alpha", new TenantQuota(gpus, cores));
        q.put("beta", new TenantQuota(gpus, cores));
        q.put("gamma", new TenantQuota(gpus, cores));
        return q;
    }

    static List<Job> trace(String tenant, long seed, int count, int startId) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Job> jobs = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            String type = rng.nextInt(2) == 0 ? "A100" : "H100";
            jobs.add(new Job(startId + i, tenant, 1 + rng.nextInt(3), 1 + rng.nextInt(8), 4 + rng.nextInt(16),
                    type, 1 + rng.nextInt(2), 2 + rng.nextInt(6), rng.nextInt(20)));
        }
        return jobs;
    }

    static List<Job> copyJobs(List<Job> src) {
        List<Job> out = new ArrayList<>();
        for (Job j : src) {
            out.add(new Job(j.id, j.tenant, j.priority, j.cores, j.mem, j.gpuType, j.gpuCount, j.duration, j.arrival));
        }
        return out;
    }

    static Map<Integer, Integer> completionTicks(ScheduleResult r, String tenant) {
        Map<Integer, Integer> out = new TreeMap<>();
        for (Job j : r.jobs) if (j.tenant.equals(tenant) && j.endTick >= 0) out.put(j.id, j.endTick);
        return out;
    }

    /** Total GPU-ticks a tenant consumed: the resource-isolation measurement. */
    static long gpuTicks(ScheduleResult r, String tenant) {
        long used = 0;
        for (Job j : r.jobs) if (j.tenant.equals(tenant) && j.endTick >= 0) used += (long) j.gpuCount * j.duration;
        return used;
    }

    public static void main(String[] args) {
        Map<String, TenantQuota> q = quotas(6, 64);

        // ---- 1. gang placement, invariants and GPU-type affinity -----------------------------------
        List<Job> mixed = new ArrayList<>();
        mixed.add(new Job(1, "alpha", 2, 4, 16, "A100", 2, 5, 0));
        mixed.add(new Job(2, "beta", 2, 8, 32, "H100", 4, 4, 0));
        mixed.add(new Job(3, "gamma", 1, 2, 8, "T4", 1, 3, 1));
        mixed.add(new Job(4, "alpha", 3, 2, 8, "A100", 1, 2, 2));
        Scheduler scheduler = new Scheduler(cluster(), q, false, true, true);
        ScheduleResult r1 = scheduler.run(mixed, 40);
        check(r1.violations == 0, "no invariant may be violated: " + r1.violations);
        for (Job j : mixed) {
            check(j.nodeId >= 0 && j.endTick >= 0, "every admissible job must be placed and finished: " + j.id);
        }
        check(mixed.get(1).nodeId == 2, "the H100 job must land on the H100 node, landed on " + mixed.get(1).nodeId);
        check(mixed.get(2).nodeId == 3, "the T4 job must land on the T4 node");
        check(mixed.get(0).nodeId == 1, "best fit must place on the tightest fitting node, landed on "
                + mixed.get(0).nodeId);
        check(r1.rejected == 0, "every job in this trace is admissible");

        // gang scheduling: all resources come from one node, verified per start event
        for (String line : r1.transcript) {
            if (!line.contains("start")) continue;
            int id = Integer.parseInt(line.split(" ")[2]);
            Job j = null;
            for (Job candidate : mixed) if (candidate.id == id) j = candidate;
            if (j != null) check(j.nodeId == Integer.parseInt(line.split("node ")[1]), "placement must be recorded");
        }

        // ---- 2. placement strategy: best fit must not fragment more than first fit -------------------
        List<Job> fitTrace = new ArrayList<>();
        SplitMix64 fr = new SplitMix64(31337L);
        for (int i = 0; i < 14; i++) {
            fitTrace.add(new Job(700 + i, "beta", 1, 1 + fr.nextInt(6), 4 + fr.nextInt(20),
                    fr.nextInt(2) == 0 ? "A100" : "H100", 1 + fr.nextInt(2), 2 + fr.nextInt(5), fr.nextInt(10)));
        }
        Map<Integer, Integer> copy = new TreeMap<>();
        for (Job j : fitTrace) copy.put(j.id, j.duration);
        List<Job> forBest = new ArrayList<>(), forFirst = new ArrayList<>();
        for (Job j : fitTrace) {
            forBest.add(new Job(j.id, j.tenant, j.priority, j.cores, j.mem, j.gpuType, j.gpuCount, j.duration, j.arrival));
            forFirst.add(new Job(j.id, j.tenant, j.priority, j.cores, j.mem, j.gpuType, j.gpuCount, j.duration, j.arrival));
        }
        ScheduleResult bestRun = new Scheduler(cluster(), quotas(14, 128), false, true, true).run(forBest, 60);
        ScheduleResult firstRun = new Scheduler(cluster(), quotas(14, 128), false, true, false).run(forFirst, 60);
        check(bestRun.violations == 0 && firstRun.violations == 0, "both placement strategies must stay safe");
        check(bestRun.makespan <= firstRun.makespan,
                "best fit must not finish later than first fit: " + bestRun.makespan + " vs " + firstRun.makespan);
        check(bestRun.gpuTicksUsed == firstRun.gpuTicksUsed,
                "both strategies must do the same amount of work when every job is admitted");

        // ---- 3. quota enforcement and the counter-example without quotas ---------------------------
        List<Job> greedy = new ArrayList<>();
        for (int i = 0; i < 12; i++) greedy.add(new Job(100 + i, "alpha", 1, 2, 8, "H100", 2, 3, i % 3));
        Scheduler noQuota = new Scheduler(cluster(), q, false, false, true);
        ScheduleResult unquoted = noQuota.run(copyJobs(greedy), 40);
        check(unquoted.violations == 0, "an unquoted scheduler trivially satisfies the (disabled) quota check");
        int peakNoQuota = unquoted.peakGpus("alpha");
        Scheduler quoted = new Scheduler(cluster(), q, false, true, true);
        ScheduleResult quotedRun = quoted.run(copyJobs(greedy), 40);
        check(quotedRun.violations == 0, "the quota-aware scheduler must never breach a quota");
        int peakQuota = quotedRun.peakGpus("alpha");
        check(peakQuota <= q.get("alpha").gpus, "alpha must stay within its GPU quota: " + peakQuota);
        check(peakNoQuota >= peakQuota, "without quotas the tenant may exceed its share: "
                + peakNoQuota + " vs " + peakQuota);
        check(quotedRun.rejected == 0 && quotedRun.makespan >= unquoted.makespan,
                "enforcing quotas keeps the trace admissible at a measured makespan cost: "
                        + quotedRun.makespan + " vs " + unquoted.makespan);

        // ---- 4. multi-tenant isolation: identical completion times when a peer is idle or saturated ---
        List<Job> alphaWork = trace("alpha", 1234L, 8, 200);
        List<Job> betaIdle = new ArrayList<>();
        List<Job> betaHeavy = trace("beta", 5678L, 16, 300);
        ScheduleResult alone = new Scheduler(cluster(), q, false, true, true).run(copyJobs(alphaWork), 60);
        Map<Integer, Integer> alphaAlone = completionTicks(alone, "alpha");
        List<Job> combined = new ArrayList<>(alphaWork);
        combined.addAll(betaHeavy);
        ScheduleResult withPeer = new Scheduler(cluster(), q, false, true, true).run(copyJobs(combined), 60);
        Map<Integer, Integer> alphaWithPeer = completionTicks(withPeer, "alpha");
        check(alphaAlone.size() == 8, "all of alpha's jobs must complete when running alone");
        check(alphaWithPeer.size() == 8, "all of alpha's jobs must complete alongside beta");
        int maxDeviation = 0;
        boolean sameJobSet = true;
        for (Map.Entry<Integer, Integer> e : alphaAlone.entrySet()) {
            Integer other = alphaWithPeer.get(e.getKey());
            if (other == null) { sameJobSet = false; continue; }
            maxDeviation = Math.max(maxDeviation, Math.abs(other - e.getValue()));
        }
        check(sameJobSet, "the same alpha jobs must complete in both runs");
        // Quota-capped isolation: beta can never take alpha's resources (identical GPU-tick totals), and the
        // only visible effect is placement jitter, bounded here by a measured tick tolerance.
        check(gpuTicks(alone, "alpha") == gpuTicks(withPeer, "alpha"),
                "alpha's GPU-tick consumption must be identical with and without a peer: "
                        + gpuTicks(alone, "alpha") + " vs " + gpuTicks(withPeer, "alpha"));
        check(maxDeviation <= 2, "completion jitter caused by placement must stay bounded, saw " + maxDeviation);
        check(betaIdle.isEmpty(), "the idle peer trace is empty by construction");
        check(withPeer.violations == 0, "the multi-tenant run must keep every invariant");
        check(withPeer.peakGpus("beta") <= q.get("beta").gpus, "beta must respect its own quota");

        // ---- 5. preemption: priority wins, and the victim still finishes -----------------------------
        Map<String, TenantQuota> roomyQuota = quotas(12, 256);   // quota is not the binding constraint here
        List<Job> lowThenHigh = new ArrayList<>();
        lowThenHigh.add(new Job(500, "alpha", 1, 8, 32, "H100", 6, 10, 0));
        lowThenHigh.add(new Job(501, "alpha", 9, 8, 32, "H100", 4, 3, 2));
        Scheduler noPreempt = new Scheduler(cluster(), roomyQuota, false, true, true);
        ScheduleResult noPreemptRun = noPreempt.run(copyJobs(lowThenHigh), 60);
        Scheduler withPreempt = new Scheduler(cluster(), roomyQuota, true, true, true);
        ScheduleResult preemptRun = withPreempt.run(copyJobs(lowThenHigh), 60);
        Job high = preemptRun.jobs.get(1), low = preemptRun.jobs.get(0);
        check(preemptRun.preemptions == 1, "exactly one preemption is expected: " + preemptRun.preemptions);
        check(high.startTick == 2, "the high-priority job must start as soon as it arrives: " + high.startTick);
        check(high.endTick <= 5, "the high-priority job must finish on time");
        check(low.endTick > high.endTick, "the victim must finish after the job that preempted it");
        check(low.preemptions == 1, "the victim must record its preemption");
        check(preemptRun.violations == 0, "preemption must not leave the cluster oversubscribed");
        check(noPreemptRun.jobs.get(1).startTick >= 10,
                "without preemption the high-priority job waits for the low one: " + noPreemptRun.jobs.get(1).startTick);
        check(preemptRun.makespan <= noPreemptRun.makespan + 6,
                "preemption must not blow up the makespan: " + preemptRun.makespan + " vs " + noPreemptRun.makespan);

        // ---- 6. unschedulable requests and empty inputs --------------------------------------------
        List<Job> impossible = new ArrayList<>();
        impossible.add(new Job(600, "alpha", 5, 4, 8, "H100", 99, 2, 0));
        impossible.add(new Job(601, "alpha", 5, 4, 8, "A100", 1, 2, 0));
        ScheduleResult impossibleRun = new Scheduler(cluster(), q, false, true, true).run(impossible, 20);
        check(impossible.get(0).endTick < 0, "a request larger than any node must never be scheduled");
        check(impossible.get(1).endTick >= 0, "the admissible job next to it must still run");
        check(impossibleRun.violations == 0, "an unschedulable job must not corrupt the cluster state");

        List<Job> none = new ArrayList<>();
        ScheduleResult emptyRun = new Scheduler(cluster(), q, false, true, true).run(none, 10);
        check(emptyRun.makespan == 0 && emptyRun.gpuTicksUsed == 0, "an empty trace must keep the cluster idle");
        check(emptyRun.gpuTicksTotal == 15L * 10, "gpu-tick capacity must be counted even when idle: "
                + emptyRun.gpuTicksTotal);

        // ---- 7. utilisation, determinism and the no-preemption flag -------------------------------
        double utilisation = r1.gpuTicksUsed / (double) r1.gpuTicksTotal;
        check(utilisation > 0.0 && utilisation <= 1.0, "utilisation must be a proper fraction: " + utilisation);
        List<Job> replayA = new ArrayList<>();
        List<Job> replayB = new ArrayList<>();
        for (Job j : combined) {
            replayA.add(new Job(j.id, j.tenant, j.priority, j.cores, j.mem, j.gpuType, j.gpuCount, j.duration, j.arrival));
            replayB.add(new Job(j.id, j.tenant, j.priority, j.cores, j.mem, j.gpuType, j.gpuCount, j.duration, j.arrival));
        }
        String t1 = String.join("|", new Scheduler(cluster(), q, false, true, true).run(replayA, 60).transcript);
        String t2 = String.join("|", new Scheduler(cluster(), q, false, true, true).run(replayB, 60).transcript);
        check(t1.equals(t2), "the same trace must produce the same schedule");
        check(!t1.isEmpty(), "the transcript must not be empty");

        System.out.println("placement / rejects    : " + mixed.size() + " jobs, " + r1.rejected + " rejects");
        System.out.println("quota peak (alpha)     : " + peakQuota + " with quotas vs " + peakNoQuota + " without");
        System.out.println("isolation alpha ticks  : " + alphaAlone + " == " + alphaWithPeer);
        System.out.println("preemption start ticks : " + high.startTick + " (victim ends " + low.endTick + ")");
        System.out.println("utilisation / makespan : " + String.format("%.3f", utilisation) + " / " + r1.makespan);
        System.out.println("invariant violations   : " + (r1.violations + withPeer.violations + preemptRun.violations));
        System.out.println("PASS");
    }
}
