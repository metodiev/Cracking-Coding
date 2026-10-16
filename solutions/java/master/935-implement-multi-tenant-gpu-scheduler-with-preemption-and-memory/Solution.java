/** Task 935 - Implement multi-tenant GPU scheduler with preemption and memory reclamation.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  Tick-driven simulation of a multi-tenant GPU cluster. Each GPU owns a fixed memory budget and a job
 *  needs a slice on exactly one GPU. Admission is priority ordered and WFQ-fair inside a priority class
 *  (least recently served tenant first). A job that fits nowhere preempts the lowest-priority running
 *  job that frees enough memory on one GPU; the victim is charged RESTART_COST ticks, re-queued, and the
 *  reclaimed slice is handed to the preemptor in the same tick, so reclamation is observable.
 *  Simplified: one GPU per job, no migration, no compute oversubscription, constant checkpoint cost.
 *  Verified: per-GPU memory never exceeds capacity and equals the sum of its running jobs; preemption only
 *  evicts a strictly lower-priority job and always reuses the freed slice on that GPU; no tenant starves;
 *  equal-priority tenants alternate without preemption; oversized jobs are rejected instead of deadlocking
 *  the queue; the makespan obeys an area lower bound and a work upper bound.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final int CAPACITY = 8;      // memory units per GPU
    static final int RESTART_COST = 2;  // ticks lost by a preempted job
    static final int MAX_TICKS = 4000;

    static final class Job {
        final int id, tenant, priority, mem, duration, arrival;
        int remaining, startTick = -1, endTick = -1, evictions;
        Job(int id, int tenant, int priority, int mem, int duration, int arrival) {
            this.id = id; this.tenant = tenant; this.priority = priority;
            this.mem = mem; this.duration = duration; this.arrival = arrival;
            this.remaining = duration;
        }
    }

    static final class Gpu {
        final int id;
        int used;
        final List<Job> running = new ArrayList<>();
        Gpu(int id) { this.id = id; }
    }

    /** One observed preemption: who was evicted, who took the slice, how much was reclaimed. */
    static final class Preemption {
        final int tick, victimId, victimPriority, preemptorId, preemptorPriority, gpuId, freed, usedAfter, preemptorMem;
        int preemptorGpu = -1;
        Preemption(int tick, Job victim, Job preemptor, Gpu gpu) {
            this.tick = tick; this.victimId = victim.id; this.victimPriority = victim.priority;
            this.preemptorId = preemptor.id; this.preemptorPriority = preemptor.priority;
            this.gpuId = gpu.id; this.freed = victim.mem; this.usedAfter = gpu.used;
            this.preemptorMem = preemptor.mem;
        }
    }

    static final class Sim {
        final List<Job> jobs = new ArrayList<>();
        final List<Gpu> gpus = new ArrayList<>();
        final List<Job> waiting = new ArrayList<>();
        final List<Job> finished = new ArrayList<>();
        final List<Job> rejected = new ArrayList<>();
        final List<Preemption> preemptions = new ArrayList<>();
        final Map<Integer, Integer> lastServed = new TreeMap<>();
        final Map<Integer, Integer> completedPerTenant = new TreeMap<>();
        int tick, violations, peakUsed, makespan;
        long trace = 0xcbf29ce484222325L;

        Sim(int gpuCount) { for (int i = 0; i < gpuCount; i++) gpus.add(new Gpu(i)); }

        void run() {
            while (makespan < MAX_TICKS) {
                tick = makespan;
                for (Job j : jobs) if (j.arrival == tick) {
                    if (j.mem > CAPACITY) rejected.add(j);   // can never fit: reject, never wedge the queue
                    else waiting.add(j);
                }                                schedule();
                if (waiting.isEmpty() && idle()) return;
                advance();
                makespan++;
            }
        }

        /** Place everything that fits; preempt when a strictly higher priority job is blocked. */
        void schedule() {
            boolean progress = true;
            while (progress) {
                progress = false;
                Job best = null;
                for (Job j : waiting) if (best == null || beats(j, best)) best = j;
                if (best == null) return;
                Gpu g = fitGpu(best);
                if (g != null) { place(best, g, null); progress = true; }
                else if (preemptFor(best)) progress = true;
            }
        }

        /** Priority class first, then WFQ fairness, then arrival order, then id. */
        boolean beats(Job a, Job b) {
            if (a.priority != b.priority) return a.priority > b.priority;
            int la = lastServed.containsKey(a.tenant) ? lastServed.get(a.tenant) : Integer.MIN_VALUE;
            int lb = lastServed.containsKey(b.tenant) ? lastServed.get(b.tenant) : Integer.MIN_VALUE;
            if (la != lb) return la < lb;
            if (a.arrival != b.arrival) return a.arrival < b.arrival;
            return a.id < b.id;
        }

        Gpu fitGpu(Job j) {
            Gpu pick = null;
            for (Gpu g : gpus) if (g.used + j.mem <= CAPACITY && (pick == null || g.used < pick.used)) pick = g;
            return pick;
        }

        /** Reclaim a slice on one GPU and immediately hand it to the blocked job. */
        boolean preemptFor(Job best) {
            Gpu target = null;
            Job victim = null;
            for (Gpu g : gpus) {
                Job v = victimOn(g, best);
                if (v != null && (target == null || g.used < target.used)) { target = g; victim = v; }
            }
            if (victim == null) return false;
            target.running.remove(victim);
            target.used -= victim.mem;
            victim.evictions++;
            victim.remaining += RESTART_COST;
            waiting.add(victim);
            Preemption p = new Preemption(tick, victim, best, target);
            preemptions.add(p);
            place(best, target, p);
            return true;
        }

        Job victimOn(Gpu g, Job best) {
            Job v = null;
            for (Job r : g.running) {
                if (r.priority >= best.priority) continue;
                if (g.used - r.mem + best.mem > CAPACITY) continue;
                if (v == null || r.priority < v.priority || (r.priority == v.priority && r.mem > v.mem)
                        || (r.priority == v.priority && r.mem == v.mem && r.id < v.id)) v = r;
            }
            return v;
        }

        void place(Job j, Gpu g, Preemption p) {
            waiting.remove(j);
            g.used += j.mem;
            g.running.add(j);
            if (j.startTick < 0) j.startTick = tick;
            lastServed.put(j.tenant, tick);
            if (p != null) p.preemptorGpu = g.id;
            verify();
        }

        void advance() {
            for (Gpu g : gpus) {
                for (Iterator<Job> it = g.running.iterator(); it.hasNext(); ) {
                    Job j = it.next();
                    if (--j.remaining > 0) continue;
                    it.remove();
                    g.used -= j.mem;
                    j.endTick = tick;
                    finished.add(j);
                    completedPerTenant.merge(j.tenant, 1, Integer::sum);
                }
            }
            verify();
        }

        boolean idle() {
            for (Gpu g : gpus) if (!g.running.isEmpty()) return false;
            return true;
        }

        /** Memory-safety invariant: every GPU's accounting agrees with its running jobs. */
        void verify() {
            int total = 0;
            for (Gpu g : gpus) {
                int sum = 0;
                for (Job j : g.running) sum += j.mem;
                if (sum != g.used || g.used > CAPACITY) violations++;
                total += g.used;
            }
            peakUsed = Math.max(peakUsed, total);
            trace = trace * 1000003L + tick * 131 + total * 17 + waiting.size() * 7 + finished.size();
        }
    }

    /** Low-priority training jobs, a medium serving job, then interactive bursts that preempt. */
    static Sim fixture() {
        Sim s = new Sim(4);
        int[][] spec = {{0, 0, 1, 8, 60, 0}, {1, 0, 1, 8, 60, 0}, {2, 0, 1, 8, 40, 0}, {3, 1, 5, 4, 30, 0},
                        {4, 2, 9, 8, 6, 12}, {5, 2, 9, 8, 6, 14}, {6, 2, 8, 4, 10, 20}};
        for (int[] j : spec) s.jobs.add(new Job(j[0], j[1], j[2], j[3], j[4], j[5]));
        return s;
    }

    /** Two equal-priority tenants on one GPU: WFQ must alternate them without preemption. */
    static Sim fairnessFixture() {
        Sim s = new Sim(1);
        for (int i = 0; i < 4; i++) {
            s.jobs.add(new Job(2 * i, 3, 5, CAPACITY, 3, 0));
            s.jobs.add(new Job(2 * i + 1, 4, 5, CAPACITY, 3, 0));
        }
        return s;
    }

    public static void main(String[] args) {
        Sim cluster = fixture();
        cluster.run();

        check(cluster.violations == 0, "per-GPU memory must never exceed capacity: " + cluster.violations + " violations");
        check(cluster.finished.size() == cluster.jobs.size() && cluster.completedPerTenant.size() == 3,
                "every admitted job must finish with no starving tenant: " + cluster.finished.size() + "/"
                        + cluster.jobs.size() + " tenants " + cluster.completedPerTenant);
        check(cluster.preemptions.size() >= 2, "the fixture must exercise preemption, saw " + cluster.preemptions.size());

        int badPriority = 0, badReuse = 0, badBookkeeping = 0;
        Map<Integer, Integer> evictions = new TreeMap<>();
        for (Preemption p : cluster.preemptions) {
            if (p.victimPriority >= p.preemptorPriority) badPriority++;
            if (p.preemptorGpu != p.gpuId || p.usedAfter + p.preemptorMem > CAPACITY) badReuse++;
            evictions.merge(p.victimId, 1, Integer::sum);
        }
        for (Job j : cluster.jobs) if (j.evictions != evictions.getOrDefault(j.id, 0) || j.endTick < 0) badBookkeeping++;
        check(badPriority == 0, badPriority + " preemptions evicted a job of equal or higher priority");
        check(badReuse == 0, badReuse + " preemptions did not reuse the reclaimed slice on the same GPU");
        check(badBookkeeping == 0, "eviction accounting or resumption is inconsistent for " + badBookkeeping + " jobs");

        Sim fair = fairnessFixture();
        fair.run();
        boolean alternates = true;
        for (int i = 1; i < fair.finished.size(); i++)
            alternates &= fair.finished.get(i).tenant != fair.finished.get(i - 1).tenant;
        check(fair.preemptions.isEmpty() && alternates && fair.finished.size() == 8,
                "equal-priority tenants must alternate with zero preemptions, got "
                        + fair.finished.stream().map(j -> j.tenant).toList());

        Sim empty = new Sim(2);
        empty.run();
        check(empty.makespan == 0 && empty.preemptions.isEmpty() && empty.peakUsed == 0,
                "an empty workload must finish in zero ticks and touch no memory");

        Sim oversized = new Sim(1);
        oversized.jobs.add(new Job(0, 0, 9, CAPACITY + 4, 5, 0));
        oversized.jobs.add(new Job(1, 1, 1, CAPACITY, 3, 0));
        oversized.run();
        check(oversized.rejected.size() == 1 && oversized.rejected.get(0).startTick < 0
                        && oversized.finished.size() == 1,
                "a job larger than any GPU must be rejected without blocking the queue");

        Sim exact = new Sim(1);
        exact.jobs.add(new Job(0, 0, 1, CAPACITY, 5, 0));
        exact.jobs.add(new Job(1, 0, 7, CAPACITY, 1, 1));
        exact.run();
        check(exact.preemptions.size() == 1 && exact.preemptions.get(0).victimId == 0 && exact.finished.get(0).id == 1,
                "a strictly higher-priority job must reclaim the slice of an exactly filled GPU");

        long area = 0, work = 0;
        int maxArrival = 0;
        for (Job j : cluster.jobs) {
            area += (long) j.mem * j.duration;
            work += (long) j.duration + (long) j.evictions * RESTART_COST;
            maxArrival = Math.max(maxArrival, j.arrival);
        }
        long lower = (area + (long) cluster.gpus.size() * CAPACITY - 1) / ((long) cluster.gpus.size() * CAPACITY);
        for (Job j : cluster.jobs) lower = Math.max(lower, (long) j.arrival + j.duration);
        check(cluster.makespan >= lower && cluster.makespan <= maxArrival + work,
                "makespan " + cluster.makespan + " outside [" + lower + ", " + (maxArrival + work) + "]");

        Sim replay = fixture();
        replay.run();
        check(replay.trace == cluster.trace && replay.makespan == cluster.makespan
                        && replay.finished.size() == cluster.finished.size()
                        && replay.preemptions.size() == cluster.preemptions.size(),
                "the schedule must be reproducible for the same workload");

        System.out.println("GPU scheduler: " + cluster.jobs.size() + " jobs on " + cluster.gpus.size() + " GPUs, "
                + cluster.preemptions.size() + " preemptions, peak memory " + cluster.peakUsed + "/"
                + (cluster.gpus.size() * CAPACITY) + ", makespan " + cluster.makespan + " (area bound " + lower + ")");
        System.out.println("per-tenant completions " + cluster.completedPerTenant + ", preemption victims "
                + evictions.keySet() + ", fairness order " + fair.finished.stream().map(j -> j.tenant).toList());
        System.out.println("PASS");
    }
}
