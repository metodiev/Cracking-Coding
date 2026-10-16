/** Task 669 - Implement earliest deadline first scheduler simulator and feasibility check. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * A periodic task set (C_i, D_i, T_i) is released synchronously (all first jobs at time 0) on one
 * preemptive processor. The simulator runs tick by tick; every tick it first releases the jobs whose
 * arrival time has come, fails a job whose absolute deadline has been reached unfinished, then runs
 * one unit of the pending job with the earliest absolute deadline (deadline-monotonic tie-break by
 * task index). Setting |rateMonotonic| switches the priority rule to shortest-period-first so the two
 * schedulers can be compared on the same task sets.
 *
 * Feasibility is decided analytically with the demand-bound function
 *   dbf(t) = sum_i C_i * max(0, floor((t - D_i)/T_i) + 1) <= t   for every 1 <= t <= hyperperiod,
 * which is exact for preemptive uniprocessor EDF with D_i <= T_i. Because the demand is computed with
 * integer arithmetic, "U <= 1" becomes the exact test sum_i C_i * (H/T_i) <= H, avoiding any floating
 * point tie-breaking, and dbf(t) is the integer analogue of the utilisation test for constrained
 * deadlines.
 *
 * Verified: utilisation values, feasible and overloaded sets, the classic counterexample where U <= 1
 * but a constrained deadline still misses (C=1,D=1,T=2 with C=2,D=2,T=5), the dbf verdict against the
 * simulated one on 120 deterministic random task sets, the worst case bound L = n(2^(1/n) - 1) for
 * rate-monotonic, hyperperiod periodicity of the produced schedule, per-task execution accounting, and
 * an independent re-derivation of the EDF choice at every tick (the schedule is replayed by a separate
 * checker that rebuilds the pending job set from scratch and compares deadlines).
 */
import java.util.*;

/** One periodic task: worst-case execution |c|, relative deadline |d| and period |t|. */
final class PeriodicTask {
    final int id, c, d, t;

    PeriodicTask(int id, int c, int d, int t) { this.id = id; this.c = c; this.d = d; this.t = t; }

    String name() { return "T" + id; }

    public String toString() { return name() + "(C=" + c + ",D=" + d + ",T=" + t + ")"; }
}

/** Shared analytical helpers: demand bound, utilisation and the hyperperiod. */
class Feasibility {
    static long gcd(long a, long b) { return b == 0 ? a : gcd(b, a % b); }

    static long hyperperiod(List<PeriodicTask> tasks) {
        long h = 1;
        for (PeriodicTask x : tasks) h = h / gcd(h, x.t) * x.t;
        return h;
    }

    static double utilization(List<PeriodicTask> tasks) {
        double u = 0;
        for (PeriodicTask x : tasks) u += (double) x.c / x.t;
        return u;
    }

    /** Demand of all jobs with absolute deadline <= t, evaluated with exact integer arithmetic. */
    static int demand(List<PeriodicTask> tasks, int t) {
        int sum = 0;
        for (PeriodicTask x : tasks) {
            int jobs = Math.floorDiv(t - x.d, x.t) + 1;
            if (jobs > 0) sum += x.c * jobs;
        }
        return sum;
    }

    /** Exact equivalent of U <= 1: the demand over one full hyperperiod never exceeds the capacity. */
    static boolean utilizationAtMostOne(List<PeriodicTask> tasks, long horizon) {
        int capacity = 0;
        for (PeriodicTask x : tasks) capacity += x.c * (int) (horizon / x.t);
        return capacity <= horizon;
    }

    static int dbfViolation(List<PeriodicTask> tasks, int horizon) {
        for (int t = 1; t <= horizon; t++) if (demand(tasks, t) > t) return t;
        return -1;
    }

    static boolean dbfFeasible(List<PeriodicTask> tasks, int horizon) {
        return dbfViolation(tasks, horizon) < 0;
    }

    /** Sufficient utilisation bound of Liu & Layland for rate-monotonic scheduling. */
    static double rateMonotonicBound(int n) { return n * (Math.pow(2.0, 1.0 / n) - 1.0); }
}

/** Tick-driven uniprocessor simulator; |rateMonotonic| selects fixed-priority RM instead of EDF. */
final class Scheduler {
    final List<PeriodicTask> tasks;
    final boolean rateMonotonic;
    final List<String> schedule = new ArrayList<>();
    int[] executed, nextRelease, remaining, deadline, releaseOf, maxResponse;
    boolean missed;
    String missedJob = "none";
    int missedAt = -1, idleTicks, executedTotal;

    Scheduler(List<PeriodicTask> tasks, boolean rateMonotonic) {
        this.tasks = tasks;
        this.rateMonotonic = rateMonotonic;
        int n = tasks.size();
        executed = new int[n];
        nextRelease = new int[n];
        remaining = new int[n];
        deadline = new int[n];
        releaseOf = new int[n];
        maxResponse = new int[n];
    }

    void run(int horizon) {
        for (int tick = 0; tick < horizon && !missed; tick++) {
            for (int i = 0; i < tasks.size(); i++) {
                PeriodicTask x = tasks.get(i);
                if (tick == nextRelease[i]) {
                    if (remaining[i] > 0) {                  // previous job never finished
                        missed = true; missedJob = x.name(); missedAt = tick;
                    }
                    remaining[i] = x.c;
                    deadline[i] = tick + x.d;
                    releaseOf[i] = tick;
                    nextRelease[i] += x.t;
                }
            }
            for (int i = 0; i < tasks.size(); i++)
                if (remaining[i] > 0 && deadline[i] <= tick) {
                    missed = true; missedJob = tasks.get(i).name(); missedAt = tick;
                }
            if (missed) return;
            int pick = -1;
            for (int i = 0; i < tasks.size(); i++) {
                if (remaining[i] <= 0) continue;
                if (pick < 0 || higherPriority(i, pick)) pick = i;
            }
            if (pick < 0) {
                idleTicks++;
                schedule.add("-");
            } else {
                remaining[pick]--;
                executed[pick]++;
                executedTotal++;
                schedule.add(tasks.get(pick).name());
                if (remaining[pick] == 0)
                    maxResponse[pick] = Math.max(maxResponse[pick], tick + 1 - releaseOf[pick]);
            }
        }
    }

    boolean higherPriority(int a, int b) {
        if (rateMonotonic) {
            int ta = tasks.get(a).t, tb = tasks.get(b).t;
            return ta != tb ? ta < tb : a < b;
        }
        int da = deadline[a], db = deadline[b];
        return da != db ? da < db : a < b;
    }

    String scheduleText() { return String.join(",", schedule); }
}

public class Solution {
    static long state = 0x5EED1234L;

    static long nextLong() {                        // splitmix64, fixed seed
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static List<PeriodicTask> tasks(int... triples) {
        List<PeriodicTask> list = new ArrayList<>();
        for (int i = 0; i * 3 < triples.length; i++)
            list.add(new PeriodicTask(i, triples[3 * i], triples[3 * i + 1], triples[3 * i + 2]));
        return list;
    }

    /**
     * Independent replay: rebuilds the job state from the task set alone and checks that every tick
     * of a recorded schedule ran the EDF-best pending job, then compares the resulting execution
     * counters with the simulator's.
     */
    static void verifyEdfSchedule(Scheduler sim, int horizon) {
        List<PeriodicTask> tasks = sim.tasks;
        int n = tasks.size();
        int[] nextRelease = new int[n], remaining = new int[n], deadline = new int[n];
        int[] counted = new int[n];
        for (int tick = 0; tick < horizon; tick++) {
            String entry = sim.schedule.get(tick);
            for (int i = 0; i < n; i++) {
                PeriodicTask x = tasks.get(i);
                if (tick == nextRelease[i]) {
                    remaining[i] = x.c;
                    deadline[i] = tick + x.d;
                    nextRelease[i] += x.t;
                }
            }
            int best = -1;
            for (int i = 0; i < n; i++) {
                if (remaining[i] <= 0) continue;
                if (best < 0 || deadline[i] < deadline[best]
                        || (deadline[i] == deadline[best] && i < best)) best = i;
            }
            if (best < 0) {
                check(entry.equals("-"), "replay agrees the processor idles at " + tick);
            } else {
                check(entry.equals(tasks.get(best).name()), "replay picks " + tasks.get(best).name()
                        + " at tick " + tick + " but the schedule ran " + entry);
                remaining[best]--;
                counted[best]++;
            }
        }
        for (int i = 0; i < n; i++)
            check(counted[i] == sim.executed[i], "replay counts task " + i + ": " + counted[i]
                    + " vs " + sim.executed[i]);
    }

    static List<PeriodicTask> randomTaskSet(int n, int maxC, int maxT) {
        List<PeriodicTask> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int t = 2 + (int) (nextLong() >>> 41) % (maxT - 1);
            int c = 1 + (int) (nextLong() >>> 39) % Math.min(maxC, t);
            int d = c + (int) (nextLong() >>> 43) % Math.max(1, t - c + 1);
            if (d > t) d = t;
            list.add(new PeriodicTask(i, c, d, t));
        }
        return list;
    }

    public static void main(String[] args) {
        List<PeriodicTask> feasible = tasks(1, 4, 4, 2, 5, 5);
        check(Math.abs(Feasibility.utilization(feasible) - 0.65) < 1e-9, "utilisation of (1,4,4),(2,5,5) is 0.65");
        long h1 = Feasibility.hyperperiod(feasible);
        check(h1 == 20, "hyperperiod of periods 4 and 5 is 20, got " + h1);
        check(Feasibility.dbfFeasible(feasible, (int) h1) && Feasibility.demand(feasible, (int) h1) == 13,
                "demand over the hyperperiod equals the processor capacity used (13 of 20)");

        Scheduler edf = new Scheduler(feasible, false);
        Scheduler rm = new Scheduler(feasible, true);
        edf.run((int) h1);
        rm.run((int) h1);
        check(!edf.missed && !rm.missed, "both EDF and RM schedule the implicit-deadline set");
        check(edf.executedTotal == 13 && edf.idleTicks == 7 && edf.schedule.size() == 20,
                "one unit per tick: " + edf.executedTotal + " busy, " + edf.idleTicks + " idle");
        verifyEdfSchedule(edf, (int) h1);

        Scheduler repeat = new Scheduler(feasible, false);
        repeat.run(2 * (int) h1);
        List<String> first = repeat.schedule.subList(0, (int) h1);
        List<String> second = repeat.schedule.subList((int) h1, 2 * (int) h1);
        check(first.equals(second), "the schedule repeats every hyperperiod");

        List<PeriodicTask> constrained = tasks(1, 1, 2, 2, 2, 5);
        long h2 = Feasibility.hyperperiod(constrained);
        check(Feasibility.utilizationAtMostOne(constrained, h2),
                "U <= 1 holds exactly (9 of 10 units) for the constrained-deadline set");
        check(Feasibility.dbfViolation(constrained, (int) h2) == 2,
                "dbf catches the infeasibility at t=2, got " + Feasibility.dbfViolation(constrained, (int) h2));
        Scheduler badEdf = new Scheduler(constrained, false);
        badEdf.run((int) h2);
        check(badEdf.missed && badEdf.missedAt == 2 && badEdf.missedJob.equals("T1"),
                "EDF misses T1's first deadline at t=2, got " + badEdf.missedJob + "@" + badEdf.missedAt);
        check(Math.abs(Feasibility.utilization(constrained) - 0.9) < 1e-9,
                "the counterexample still has U = 0.9, so U <= 1 is not sufficient");

        List<PeriodicTask> overloaded = tasks(3, 3, 3, 2, 2, 3);
        check(!Feasibility.utilizationAtMostOne(overloaded, 3) && !Feasibility.dbfFeasible(overloaded, 3),
                "utilisation above one is caught by the exact test");
        Scheduler over = new Scheduler(overloaded, false);
        over.run(9);
        check(over.missed, "EDF cannot meet deadlines when the demand exceeds capacity");

        List<PeriodicTask> impossible = tasks(5, 3, 8, 1, 8, 8);
        check(!Feasibility.dbfFeasible(impossible, 8) && Feasibility.demand(impossible, 3) > 3,
                "a task with C > D is rejected by the demand bound function");

        List<PeriodicTask> empty = tasks();
        Scheduler idle = new Scheduler(empty, false);
        idle.run(10);
        check(!idle.missed && idle.executedTotal == 0 && idle.idleTicks == 10 && idle.schedule.size() == 10,
                "an empty task set leaves the processor idle");
        List<PeriodicTask> full = tasks(3, 3, 3);
        Scheduler busy = new Scheduler(full, false);
        busy.run(9);
        check(!busy.missed && busy.idleTicks == 0 && busy.executed[0] == 9 && busy.maxResponse[0] == 3,
                "C = D = T keeps the processor saturated with response time C");
        check(Feasibility.rateMonotonicBound(1) == 1.0
                        && Math.abs(Feasibility.rateMonotonicBound(2) - 0.8284) < 1e-3
                        && Math.abs(Feasibility.rateMonotonicBound(3) - 0.7798) < 1e-3,
                "Liu & Layland bound for n = 1, 2, 3");

        int agree = 0, mismatches = 0, rmSufficient = 0, rmWins = 0, edfBetter = 0, edfWorse = 0;
        for (int trial = 0; trial < 120; trial++) {
            List<PeriodicTask> set = randomTaskSet(1 + (int) (nextLong() >>> 44) % 3, 5, 10);
            long h = Feasibility.hyperperiod(set);
            boolean analytic = Feasibility.dbfFeasible(set, (int) h);
            Scheduler s = new Scheduler(set, false);
            s.run(2 * (int) h);
            if (analytic == !s.missed) agree++; else mismatches++;
            Scheduler r = new Scheduler(set, true);
            r.run(2 * (int) h);
            if (r.missed && !s.missed) edfBetter++;
            if (!r.missed && s.missed) edfWorse++;
            boolean implicitDeadlines = true;
            for (PeriodicTask x : set) if (x.d != x.t) implicitDeadlines = false;
            if (!analytic || !implicitDeadlines) continue;              // LL bound assumes D = T
            if (Feasibility.utilization(set) <= Feasibility.rateMonotonicBound(set.size()) + 1e-12) {
                rmSufficient++;
                if (!r.missed) rmWins++;
            }
        }
        check(mismatches == 0, "demand-bound verdict matches the simulation on all 120 random sets ("
                + agree + " agreed, " + mismatches + " mismatched)");
        check(rmWins == rmSufficient && rmSufficient > 0,
                "RM met every implicit-deadline set inside its sufficient bound (" + rmWins + "/" + rmSufficient + ")");
        check(edfWorse == 0 && edfBetter > 0,
                "EDF never loses to fixed-priority RM and beats it on " + edfBetter + " sets");

        List<PeriodicTask> cross = tasks(1, 2, 3, 2, 4, 6, 1, 3, 6);
        long h3 = Feasibility.hyperperiod(cross);
        Scheduler e3 = new Scheduler(cross, false), r3 = new Scheduler(cross, true);
        e3.run((int) h3);
        r3.run((int) h3);
        check(!e3.missed, "EDF meets the constrained-deadline set the demand bound accepts");
        check(r3.missed, "fixed-priority RM fails the same set (it is only sufficient, not optimal)");
        verifyEdfSchedule(e3, (int) h3);
        int perTask = 0;
        for (int i = 0; i < cross.size(); i++)
            perTask += cross.get(i).c * ((int) h3 / cross.get(i).t);
        check(perTask == e3.executedTotal, "per-task execution sums to " + perTask + " over the hyperperiod");
        Scheduler again = new Scheduler(cross, false);
        again.run((int) h3);
        check(again.scheduleText().equals(e3.scheduleText()), "the scheduler is deterministic");

        System.out.println("feasible set       : U=0.65, " + edf.executedTotal + " busy ticks, RM also meets deadlines");
        System.out.println("constrained set    : U=0.90 but dbf fails at t=2, EDF misses T1 at t=2");
        System.out.println("random cross-check : " + agree + " task sets, dbf verdict == simulation verdict");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
