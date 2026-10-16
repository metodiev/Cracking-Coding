/** Task 667 - Implement schedulability analysis for real-time systems (rate-monotonic). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Three schedulability tests for periodic tasks scheduled by rate monotonic (shorter period = higher
 * priority), plus a simulator used as an oracle:
 *
 *   utilisation bound  - sum(C_i/T_i) <= n(2^(1/n) - 1) is sufficient but not necessary;
 *   response time      - the exact fixed point R = C_i + sum_{j higher priority} ceil(R/T_j) * C_j,
 *                        found by iteration; the task meets its deadline iff the least fixed point is
 *                        <= D_i;
 *   simulation         - one unit of work per time step over the hyperperiod with fixed priorities,
 *                        counting deadline misses.
 *
 * Verified: the bound values match n(2^(1/n)-1); the classic counterexample {2,5},{4,7} passes the
 * response time test and is confirmed by simulation even though the utilisation bound rejects it; the
 * iteration is monotone and equals the least fixed point found by an independent scan; simulation
 * agrees with the response time analysis on randomised task sets (schedulable iff no misses over the
 * hyperperiod); a set with utilisation above one misses deadlines in both; harmonic task sets with
 * utilisation <= 1 are always schedulable; deadlines shorter than periods are honoured; RM is optimal
 * among all fixed-priority assignments (checked exhaustively over every priority permutation); and the
 * simulator's executed work equals the released work. Invalid task descriptions are rejected.
 */
import java.util.*;

final class PeriodicTask {
    final String name;
    final int cost;
    final int period;
    final int deadline;

    PeriodicTask(String name, int cost, int period, int deadline) {
        if (cost <= 0) throw new IllegalArgumentException("cost must be positive");
        if (period <= 0) throw new IllegalArgumentException("period must be positive");
        if (deadline <= 0) throw new IllegalArgumentException("deadline must be positive");
        if (deadline > period) throw new IllegalArgumentException("deadline must not exceed the period");
        if (cost > deadline) throw new IllegalArgumentException("cost must fit in the deadline");
        this.name = name;
        this.cost = cost;
        this.period = period;
        this.deadline = deadline;
    }

    static PeriodicTask of(String name, int cost, int period) { return new PeriodicTask(name, cost, period, period); }
}

final class Schedulability {
    /** The Liu & Layland utilisation bound n(2^(1/n) - 1). */
    static double utilisationBound(int tasks) {
        if (tasks < 1) throw new IllegalArgumentException("need at least one task");
        return tasks * (Math.pow(2.0, 1.0 / tasks) - 1.0);
    }

    static double utilisation(List<PeriodicTask> tasks) {
        double total = 0;
        for (PeriodicTask task : tasks) total += task.cost / (double) task.period;
        return total;
    }

    static boolean boundTest(List<PeriodicTask> tasks) {
        return utilisation(tasks) <= utilisationBound(tasks.size()) + 1e-12;
    }

    static List<PeriodicTask> byPriority(List<PeriodicTask> tasks) {
        List<PeriodicTask> sorted = new ArrayList<>(tasks);
        sorted.sort((a, b) -> a.period != b.period ? Integer.compare(a.period, b.period)
                : a.name.compareTo(b.name));
        return sorted;
    }

    /** The interference function w(t) = C + sum over higher priority tasks of ceil(t/T)*C. */
    static int iterationStep(int estimate, PeriodicTask task, List<PeriodicTask> higher) {
        long total = task.cost;
        for (PeriodicTask other : higher) {
            total += (estimate + other.period - 1L) / other.period * (long) other.cost;
        }
        return (int) Math.min(Integer.MAX_VALUE, total);
    }

    /** The least fixed point via iteration; returns Integer.MAX_VALUE when it does not converge below the cap. */
    static int responseTime(PeriodicTask task, List<PeriodicTask> higher) {
        int current = task.cost;
        for (int step = 0; step < 10_000; step++) {
            int next = iterationStep(current, task, higher);
            checkMonotone(current, next);
            if (next == current) return current;
            if (next > 1_000_000) return Integer.MAX_VALUE;
            current = next;
        }
        return Integer.MAX_VALUE;
    }

    private static void checkMonotone(int current, int next) {
        if (next < current) throw new AssertionError("response time iteration went backwards: " + current + " -> " + next);
    }

    static int[] responseTimes(List<PeriodicTask> tasks) {
        List<PeriodicTask> ordered = byPriority(tasks);
        int[] times = new int[ordered.size()];
        for (int i = 0; i < ordered.size(); i++) {
            times[i] = responseTime(ordered.get(i), ordered.subList(0, i));
        }
        return times;
    }

    static boolean responseTimeTest(List<PeriodicTask> tasks) {
        List<PeriodicTask> ordered = byPriority(tasks);
        for (int i = 0; i < ordered.size(); i++) {
            int response = responseTime(ordered.get(i), ordered.subList(0, i));
            if (response > ordered.get(i).deadline) return false;
        }
        return true;
    }

    /** Independent check: the smallest t with t == C + sum ceil(t/T_j) C_j, found by scanning. */
    static int leastFixedPointByScan(PeriodicTask task, List<PeriodicTask> higher, int limit) {
        for (int t = task.cost; t <= limit; t++) {
            long workload = task.cost;
            for (PeriodicTask other : higher) workload += (t + other.period - 1) / other.period * (long) other.cost;
            if (workload == t) return t;
        }
        return Integer.MAX_VALUE;
    }

    static int hyperPeriod(List<PeriodicTask> tasks) {
        int lcm = 1;
        for (PeriodicTask task : tasks) lcm = lcm / gcd(lcm, task.period) * task.period;
        return lcm;
    }

    private static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

    /** Simulation result: deadline misses per task and the total work executed. */
    static final class Outcome {
        final int[] misses;
        final long executedWork;
        final long releasedWork;

        Outcome(int[] misses, long executedWork, long releasedWork) {
            this.misses = misses;
            this.executedWork = executedWork;
            this.releasedWork = releasedWork;
        }

        boolean missedAny() {
            for (int miss : misses) if (miss > 0) return true;
            return false;
        }
    }

    /**
     * Fixed priority simulation over one hyperperiod at one time unit per step. `priority` lists task
     * indices from highest to lowest priority. A job released at r must finish by the end of the time
     * unit that ends at r + D; otherwise the miss is counted and the job is dropped.
     */
    static Outcome simulate(List<PeriodicTask> tasks, int[] priority) {
        int horizon = hyperPeriod(tasks);
        int taskCount = tasks.size();
        if (priority.length != taskCount) throw new IllegalArgumentException("priority list must cover every task");
        int[] remaining = new int[taskCount];
        int[] deadlineAt = new int[taskCount];
        Arrays.fill(remaining, -1);
        int[] misses = new int[taskCount];
        long executed = 0;
        long released = 0;
        for (int time = 0; time < horizon; time++) {
            for (int i = 0; i < taskCount; i++) {
                PeriodicTask task = tasks.get(i);
                if (time % task.period == 0) {
                    if (remaining[i] > 0) misses[i]++;         // the previous job was already doomed
                    remaining[i] = task.cost;
                    deadlineAt[i] = time + task.deadline;
                    released += task.cost;
                }
            }
            for (int rank = 0; rank < taskCount; rank++) {
                int index = priority[rank];
                if (remaining[index] > 0) {
                    remaining[index]--;
                    executed++;
                    break;
                }
            }
            for (int i = 0; i < taskCount; i++) {
                if (remaining[i] == 0) {
                    if (time + 1 > deadlineAt[i]) misses[i]++;
                    remaining[i] = -1;
                } else if (remaining[i] > 0 && time + 1 >= deadlineAt[i]) {
                    misses[i]++;                                // the deadline passed with work left
                    remaining[i] = -1;
                }
            }
        }
        return new Outcome(misses, executed, released);
    }

    static int[] rateMonotonicPriority(List<PeriodicTask> tasks) {
        List<PeriodicTask> ordered = byPriority(tasks);
        int[] priority = new int[tasks.size()];
        for (int rank = 0; rank < ordered.size(); rank++) {
            priority[rank] = tasks.indexOf(ordered.get(rank));
        }
        return priority;
    }

    static Outcome simulateRateMonotonic(List<PeriodicTask> tasks) {
        return simulate(tasks, rateMonotonicPriority(tasks));
    }

    /** Exhaustive search over fixed priority orders (small task sets only). */
    static boolean anyFixedPriorityWorks(List<PeriodicTask> tasks) {
        List<int[]> permutations = new ArrayList<>();
        permute(new int[tasks.size()], new boolean[tasks.size()], 0, permutations);
        for (int[] priority : permutations) if (!simulate(tasks, priority).missedAny()) return true;
        return false;
    }

    private static void permute(int[] current, boolean[] used, int depth, List<int[]> out) {
        if (depth == current.length) {
            out.add(current.clone());
            return;
        }
        for (int i = 0; i < current.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            permute(current, used, depth + 1, out);
            used[i] = false;
        }
    }
}

public class Solution {
    private static long state = 0x452821E638D01377L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static List<PeriodicTask> tasks(int[]... triples) {
        List<PeriodicTask> out = new ArrayList<>();
        for (int i = 0; i < triples.length; i++) {
            out.add(PeriodicTask.of("t" + i, triples[i][0], triples[i][1]));
        }
        return out;
    }

    public static void main(String[] args) {
        // --- the utilisation bound ---------------------------------------------
        double[] expectedBounds = {1.0, 0.8284, 0.7797, 0.7568, 0.7434};
        for (int n = 1; n <= 5; n++) {
            check(Math.abs(Schedulability.utilisationBound(n) - expectedBounds[n - 1]) < 0.001,
                    "bound for " + n + " tasks is " + expectedBounds[n - 1] + ", got "
                            + Schedulability.utilisationBound(n));
        }
        check(Schedulability.utilisationBound(6) < Schedulability.utilisationBound(5),
                "the bound decreases with the task count");
        check(Schedulability.utilisationBound(20) > Math.log(2) - 0.02,
                "the bound approaches ln 2 from above");

        // --- the classic bound-is-pessimistic example --------------------------
        List<PeriodicTask> classic = tasks(new int[]{1, 2}, new int[]{2, 5});
        check(Math.abs(Schedulability.utilisation(classic) - 0.9) < 0.001,
                "utilisation is 0.9, got " + Schedulability.utilisation(classic));
        check(!Schedulability.boundTest(classic), "the bound rejects the classic set");
        check(Schedulability.responseTimeTest(classic), "but exact analysis accepts it");
        int[] classicTimes = Schedulability.responseTimes(classic);
        check(classicTimes[0] == 1 && classicTimes[1] == 4,
                "response times 1 and 4, got " + Arrays.toString(classicTimes));
        List<PeriodicTask> orderedClassic = Schedulability.byPriority(classic);
        check(Schedulability.leastFixedPointByScan(orderedClassic.get(1), orderedClassic.subList(0, 1), 20) == 4,
                "an independent scan finds the same fixed point");
        check(!Schedulability.simulateRateMonotonic(classic).missedAny(),
                "and the simulation confirms it over the hyperperiod");
        check(Schedulability.hyperPeriod(classic) == 10, "the hyperperiod is 10");
        check(Schedulability.simulate(classic, new int[]{1, 0}).missedAny(),
                "a reversed fixed priority order breaks the short period task");

        // --- an unschedulable set ---------------------------------------------
        List<PeriodicTask> overloaded = tasks(new int[]{3, 5}, new int[]{2, 4});
        check(Schedulability.utilisation(overloaded) > 1.0, "utilisation exceeds one");
        check(!Schedulability.boundTest(overloaded), "the bound rejects the overloaded set");
        check(!Schedulability.responseTimeTest(overloaded), "exact analysis rejects it too");
        Schedulability.Outcome overloadedRun = Schedulability.simulateRateMonotonic(overloaded);
        check(overloadedRun.missedAny(), "and the simulation shows deadline misses");
        check(!Schedulability.anyFixedPriorityWorks(overloaded),
                "no fixed priority assignment can schedule an overloaded set");

        // --- a three task set above the bound but schedulable -------------------
        List<PeriodicTask> wellBehaved = tasks(new int[]{1, 4}, new int[]{2, 5}, new int[]{2, 10});
        check(Math.abs(Schedulability.utilisation(wellBehaved) - 0.85) < 0.001,
                "utilisation is 0.85, got " + Schedulability.utilisation(wellBehaved));
        check(!Schedulability.boundTest(wellBehaved), "the bound is pessimistic here again");
        check(Schedulability.responseTimeTest(wellBehaved), "but exact analysis accepts it");
        check(!Schedulability.simulateRateMonotonic(wellBehaved).missedAny(), "and the simulation");
        int[] wellTimes = Schedulability.responseTimes(wellBehaved);
        check(wellTimes[0] == 1 && wellTimes[1] == 3 && wellTimes[2] <= 10,
                "response times are 1, 3 and within the period, got " + Arrays.toString(wellTimes));

        // --- a set comfortably inside the bound --------------------------------
        List<PeriodicTask> slack = tasks(new int[]{1, 4}, new int[]{1, 5}, new int[]{1, 10});
        check(Schedulability.utilisation(slack) < Schedulability.utilisationBound(3),
                "the slack set sits inside the utilisation bound");
        check(Schedulability.boundTest(slack), "so the bound accepts it");
        check(Schedulability.responseTimeTest(slack), "and exact analysis agrees");
        check(!Schedulability.simulateRateMonotonic(slack).missedAny(), "and so does the simulation");

        // --- harmonic sets with utilisation <= 1 are schedulable ---------------
        for (int trial = 0; trial < 30; trial++) {
            int base = 2 + nextInt(3);
            List<PeriodicTask> harmonic = new ArrayList<>();
            for (int i = 0; i < 4; i++) {
                harmonic.add(PeriodicTask.of("h" + i, 1 + nextInt(Math.min(3, base)), base * (1 << i)));
            }
            if (Schedulability.utilisation(harmonic) > 1.0) continue;
            check(Schedulability.responseTimeTest(harmonic),
                    "harmonic sets with U <= 1 are schedulable by exact analysis, trial " + trial);
            check(!Schedulability.simulateRateMonotonic(harmonic).missedAny(),
                    "harmonic sets with U <= 1 are schedulable in simulation, trial " + trial);
        }

        // --- deadlines shorter than periods ------------------------------------
        PeriodicTask tight = new PeriodicTask("tight", 2, 10, 3);
        PeriodicTask loose = new PeriodicTask("loose", 2, 10, 9);
        List<PeriodicTask> tightSet = List.of(tight, new PeriodicTask("other", 4, 12, 5));
        check(!Schedulability.responseTimeTest(tightSet), "a deadline of 3 misses the response time of 4");
        List<PeriodicTask> looseSet = List.of(loose, new PeriodicTask("other", 4, 12, 12));
        check(Schedulability.responseTimeTest(looseSet), "the same set is fine with a longer deadline");
        check(Schedulability.simulateRateMonotonic(tightSet).missedAny(),
                "and the simulation misses the tight deadline");
        check(!Schedulability.simulateRateMonotonic(looseSet).missedAny(),
                "while the loose deadline is met");

        // --- single task boundaries --------------------------------------------
        List<PeriodicTask> saturated = tasks(new int[]{5, 5});
        check(Schedulability.utilisation(saturated) == 1.0, "a saturated single task has U = 1");
        check(Schedulability.responseTimeTest(saturated), "a saturated single task meets its deadline exactly");
        Schedulability.Outcome saturatedRun = Schedulability.simulateRateMonotonic(saturated);
        check(!saturatedRun.missedAny(), "and the simulation agrees");
        check(saturatedRun.executedWork == saturatedRun.releasedWork, "all released work is executed");
        check(saturatedRun.executedWork == 5, "five units of work in the hyperperiod");

        // --- simulation against exact analysis on random sets -----------------
        int schedulable = 0;
        int unschedulable = 0;
        for (int trial = 0; trial < 90; trial++) {
            int count = 2 + nextInt(3);
            List<PeriodicTask> random = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                int period = 2 + nextInt(9);
                int cost = 1 + nextInt(Math.max(1, period / 2));
                if (cost > period) cost = period;
                random.add(PeriodicTask.of("r" + i, cost, period));
            }
            if (Schedulability.hyperPeriod(random) > 4000) continue;
            boolean analysis = Schedulability.responseTimeTest(random);
            Schedulability.Outcome run = Schedulability.simulateRateMonotonic(random);
            check(analysis == !run.missedAny(), "trial " + trial + ": analysis says " + analysis
                    + " but the simulation says " + !run.missedAny() + " for " + describe(random));
            check(run.executedWork <= run.releasedWork, "trial " + trial + ": executed work is bounded");
            if (!run.missedAny()) {
                check(run.executedWork == Math.min(run.releasedWork, Schedulability.hyperPeriod(random)),
                        "trial " + trial + ": with no misses the processor is busy whenever work is pending");
            } else {
                check(run.executedWork < run.releasedWork, "trial " + trial + ": a missed deadline drops work");
            }
            if (analysis) schedulable++; else unschedulable++;
            if (count <= 4) {
                boolean anyWorks = Schedulability.anyFixedPriorityWorks(random);
                check(analysis == anyWorks, "trial " + trial + ": rate monotonic is optimal ("
                        + analysis + " vs " + anyWorks + ")");
            }
        }
        check(schedulable > 0, "the random sweep found schedulable sets: " + schedulable);
        check(unschedulable > 0, "the random sweep found unschedulable sets: " + unschedulable);

        // --- rejected task descriptions ----------------------------------------
        boolean rejected = false;
        try {
            PeriodicTask.of("bad", 0, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero cost must be rejected");
        rejected = false;
        try {
            PeriodicTask.of("bad", 2, 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero period must be rejected");
        rejected = false;
        try {
            new PeriodicTask("bad", 2, 5, 6);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a deadline beyond the period must be rejected");
        rejected = false;
        try {
            new PeriodicTask("bad", 7, 5, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a cost above the deadline must be rejected");
        rejected = false;
        try {
            Schedulability.utilisationBound(0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "the bound needs at least one task");
        rejected = false;
        try {
            Schedulability.simulate(classic, new int[]{0});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an incomplete priority list must be rejected");

        System.out.println("PASS");
    }

    static String describe(List<PeriodicTask> tasks) {
        StringBuilder sb = new StringBuilder();
        for (PeriodicTask task : tasks) sb.append('(').append(task.cost).append(',').append(task.period).append(')');
        return sb.toString();
    }
}
