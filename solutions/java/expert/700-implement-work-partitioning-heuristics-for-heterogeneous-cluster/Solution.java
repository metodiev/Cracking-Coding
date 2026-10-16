/** Task 700 - Implement work partitioning heuristics for heterogeneous clusters. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * Jobs with integer sizes and machines with different throughputs (size units per second). The
 * schedule is an assignment job -> machine, its makespan is max over machines of
 * (assigned size / machine speed), and four partitioning heuristics are compared:
 *
 *   LIST             jobs in arrival order, each appended to whichever machine currently has the
 *                    smallest relative load (load / speed), ties going to the faster machine;
 *   LPT              the same rule but processing the largest job first - the classical
 *                    longest-processing-time-first heuristic;
 *   ROUND-ROBIN      one job per machine in turn, which distributes *jobs*, not work;
 *   SPEED-PROPORTIONAL  each job goes to the machine with the largest remaining share of its
 *                    target load total * speed_i / sum(speed), so a machine twice as fast
 *                    receives twice the work.
 *
 * Two lower bounds bracket any schedule: the biggest job on the fastest machine, and the total
 * work divided by the total speed. The brute-force optimum over all m^n assignments is computed
 * for small instances, which is what makes the classical guarantees checkable: LPT within
 * 4/3 - 1/(3m) of the optimum on identical machines, list scheduling within 2 - 1/m.
 *
 * Verified: every job is scheduled exactly once, both lower bounds always hold, the classical
 * ratios against the true optimum, the speedup bound makespan >= total/sum(speed), the
 * proportional-load balance of LPT on a heterogeneous cluster, and LPT's large win over
 * round-robin when jobs are small and machines are not equally fast.
 */
import java.util.*;

public class Solution {
    private static long prngState = 0xAB1E7L;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    /** Makespan of an assignment: the slowest machine decides. */
    static double makespan(int[] machineOfJob, int[] jobSize, double[] speed) {
        double[] load = new double[speed.length];
        for (int job = 0; job < jobSize.length; job++) load[machineOfJob[job]] += jobSize[job];
        double worst = 0;
        for (int machine = 0; machine < speed.length; machine++) {
            worst = Math.max(worst, load[machine] / speed[machine]);
        }
        return worst;
    }

    static double[] loads(int[] machineOfJob, int[] jobSize, int machines) {
        double[] load = new double[machines];
        for (int job = 0; job < jobSize.length; job++) load[machineOfJob[job]] += jobSize[job];
        return load;
    }

    /** Assigns each job to the machine with the smallest relative load; ties go to the faster one. */
    static int assignRelative(int job, double[] load, double[] speed) {
        int best = 0;
        double bestRatio = Double.MAX_VALUE;
        for (int machine = 0; machine < speed.length; machine++) {
            double ratio = load[machine] / speed[machine];
            if (ratio < bestRatio - 1e-12
                    || (ratio < bestRatio + 1e-12 && speed[machine] > speed[best])) {
                bestRatio = ratio;
                best = machine;
            }
        }
        return best;
    }

    static int[] listOrder(int[] jobSize, double[] speed) {
        int[] assignment = new int[jobSize.length];
        double[] load = new double[speed.length];
        for (int job = 0; job < jobSize.length; job++) {
            int machine = assignRelative(job, load, speed);
            assignment[job] = machine;
            load[machine] += jobSize[job];
        }
        return assignment;
    }

    static int[] lpt(int[] jobSize, double[] speed) {
        Integer[] order = new Integer[jobSize.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Integer.compare(jobSize[b], jobSize[a]));
        int[] assignment = new int[jobSize.length];
        double[] load = new double[speed.length];
        for (int job : order) {
            int machine = assignRelative(job, load, speed);
            assignment[job] = machine;
            load[machine] += jobSize[job];
        }
        return assignment;
    }

    static int[] roundRobin(int[] jobSize, double[] speed) {
        int[] assignment = new int[jobSize.length];
        for (int job = 0; job < jobSize.length; job++) assignment[job] = job % speed.length;
        return assignment;
    }

    /** Keeps each machine close to its share total * speed_i / sum(speed). */
    static int[] speedProportional(int[] jobSize, double[] speed) {
        double total = 0;
        double totalSpeed = 0;
        for (int size : jobSize) total += size;
        for (double value : speed) totalSpeed += value;
        Integer[] order = new Integer[jobSize.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Integer.compare(jobSize[b], jobSize[a]));
        int[] assignment = new int[jobSize.length];
        double[] load = new double[speed.length];
        for (int job : order) {
            int best = 0;
            double bestHeadroom = -Double.MAX_VALUE;
            for (int machine = 0; machine < speed.length; machine++) {
                double target = total * speed[machine] / totalSpeed;
                double headroom = target - load[machine];
                if (headroom > bestHeadroom + 1e-12) {
                    bestHeadroom = headroom;
                    best = machine;
                }
            }
            assignment[job] = best;
            load[best] += jobSize[job];
        }
        return assignment;
    }

    /** Exhaustive optimum over all m^n assignments (small instances only). */
    static double optimal(int[] jobSize, double[] speed) {
        int machines = speed.length;
        int[] assignment = new int[jobSize.length];
        return enumerate(jobSize, speed, assignment, 0, Integer.MAX_VALUE);
    }

    private static double enumerate(int[] jobSize, double[] speed, int[] assignment, int job,
                                    double best) {
        if (job == jobSize.length) return Math.min(best, makespan(assignment, jobSize, speed));
        for (int machine = 0; machine < speed.length; machine++) {
            assignment[job] = machine;
            double value = enumerate(jobSize, speed, assignment, job + 1, best);
            if (value < best) best = value;
        }
        return best;
    }

    static double lowerBoundBiggestJob(int[] jobSize, double[] speed) {
        double fastest = 0;
        for (double value : speed) fastest = Math.max(fastest, value);
        double worst = 0;
        for (int size : jobSize) worst = Math.max(worst, size / fastest);
        return worst;
    }

    static double lowerBoundTotal(int[] jobSize, double[] speed) {
        double total = 0;
        double totalSpeed = 0;
        for (int size : jobSize) total += size;
        for (double value : speed) totalSpeed += value;
        return totalSpeed == 0 ? 0 : total / totalSpeed;
    }

    static boolean scheduledEveryJob(int[] assignment, int[] jobSize, double[] speed) {
        double assigned = 0;
        for (int job = 0; job < jobSize.length; job++) {
            if (assignment[job] < 0 || assignment[job] >= speed.length) return false;
            assigned += jobSize[job];
        }
        double[] load = loads(assignment, jobSize, speed.length);
        double summed = 0;
        for (double value : load) summed += value;
        return Math.abs(summed - assigned) < 1e-9;
    }

    static int[] randomJobs(int count, int maxSize) {
        int[] sizes = new int[count];
        for (int i = 0; i < count; i++) sizes[i] = 1 + nextInt(maxSize);
        return sizes;
    }

    public static void main(String[] args) {
        // 1. Manual instances: the fastest machine is preferred on ties, the total is preserved.
        double[] equal = {1, 1, 1};
        int[] jobs = {6, 5, 4, 3, 2, 1};
        int[] lptAssignment = lpt(jobs, equal);
        double lptMakespan = makespan(lptAssignment, jobs, equal);
        check(scheduledEveryJob(lptAssignment, jobs, equal), "LPT schedules every job exactly once");
        check(Math.abs(lptMakespan - 7) < 1e-9, "hand-checked LPT makespan is 7: " + lptMakespan);
        check(Math.abs(optimal(jobs, equal) - 7) < 1e-9,
                "the brute-force optimum is 7 for this instance: " + optimal(jobs, equal));
        check(Math.abs(makespan(listOrder(jobs, equal), jobs, equal) - 7) < 1e-9,
                "list scheduling also lands on 7 here");
        check(Math.abs(makespan(roundRobin(jobs, equal), jobs, equal) - 9) < 1e-9,
                "round-robin is worse on the same jobs: " + makespan(roundRobin(jobs, equal), jobs, equal));

        double[] mixed = {1, 2, 4};
        int[] single = {100};
        check(makespan(lpt(single, mixed), single, mixed) == 25.0,
                "a single job goes to the fastest machine: " + makespan(lpt(single, mixed), single, mixed));
        check(Math.abs(makespan(lpt(single, mixed), single, mixed)
                - lowerBoundBiggestJob(single, mixed)) < 1e-9,
                "and the makespan equals the biggest-job lower bound");

        // 2. Random instances: lower bounds hold for every heuristic, every time.
        int violations = 0;
        int instances = 0;
        for (int trial = 0; trial < 30; trial++) {
            double[] speeds = new double[]{1 + nextInt(3), 1 + nextInt(3), 1 + nextInt(4)};
            int[] sizes = randomJobs(8, 20);
            instances++;
            double big = lowerBoundBiggestJob(sizes, speeds);
            double total = lowerBoundTotal(sizes, speeds);
            for (int[] assignment : new int[][]{listOrder(sizes, speeds), lpt(sizes, speeds),
                    roundRobin(sizes, speeds), speedProportional(sizes, speeds)}) {
                double value = makespan(assignment, sizes, speeds);
                if (value < big - 1e-9 || value < total - 1e-9) violations++;
                if (!scheduledEveryJob(assignment, sizes, speeds)) violations++;
            }
        }
        check(violations == 0, "30 heterogeneous instances respect both lower bounds: " + instances);

        // 3. The classical guarantees against the brute-force optimum on identical machines.
        double worstLptRatio = 0;
        double worstListRatio = 0;
        int machineCount = 3;
        for (int trial = 0; trial < 25; trial++) {
            double[] speeds = new double[machineCount];
            Arrays.fill(speeds, 1);
            int[] sizes = randomJobs(8, 25);
            double optimum = optimal(sizes, speeds);
            worstLptRatio = Math.max(worstLptRatio, makespan(lpt(sizes, speeds), sizes, speeds) / optimum);
            worstListRatio = Math.max(worstListRatio,
                    makespan(listOrder(sizes, speeds), sizes, speeds) / optimum);
        }
        check(worstLptRatio <= 4.0 / 3.0 - 1.0 / (3.0 * machineCount) + 1e-9,
                "LPT stays within 4/3 - 1/(3m): " + worstLptRatio);
        check(worstListRatio <= 2.0 - 1.0 / machineCount + 1e-9,
                "list scheduling stays within 2 - 1/m: " + worstListRatio);
        check(optimal(new int[]{1, 1, 1, 1, 1, 1}, new double[]{1, 1, 1}) == 2.0,
                "six unit jobs on three machines need exactly two units");

        // 4. Speedup: no schedule can beat the total-work lower bound.
        double[] speeds = {1, 3, 5};
        int[] sizes = randomJobs(40, 12);
        double total = 0;
        for (int size : sizes) total += size;
        double totalSpeed = 11;
        double best = makespan(lpt(sizes, speeds), sizes, speeds);
        double speedup = total / best;
        check(best >= total / totalSpeed - 1e-9, "makespan bounded by total work over total speed");
        check(speedup <= totalSpeed + 1e-9, "speedup never exceeds the sum of the speeds: "
                + speedup + " vs " + totalSpeed);
        check(speedup / totalSpeed <= 1 + 1e-9, "efficiency stays below one");

        // 5. Heterogeneous cluster with many equal jobs: proportional partitioning beats chunks.
        double[] cluster = {1, 2, 4};
        int[] manyEqual = new int[300];
        Arrays.fill(manyEqual, 1);
        double proportional = makespan(speedProportional(manyEqual, cluster), manyEqual, cluster);
        double lptValue = makespan(lpt(manyEqual, cluster), manyEqual, cluster);
        double chunked = makespan(roundRobin(manyEqual, cluster), manyEqual, cluster);
        check(chunked > 2 * proportional,
                "round-robin wastes the fast machines (2.3x): " + chunked + " vs " + proportional);
        check(Math.abs(proportional - 300.0 / 7.0) < 0.5,
                "the proportional split approaches total speed: " + proportional);
        check(lptValue <= proportional + 1e-9, "LPT is as good as the proportional target");
        double[] proportionalLoad = loads(speedProportional(manyEqual, cluster), manyEqual, 3);
        check(Math.abs(proportionalLoad[2] / 4 - proportionalLoad[0] / 1) < 0.5,
                "the fast machine receives four times the work of the slow one: "
                        + proportionalLoad[2] + " vs " + proportionalLoad[0]);

        // 6. Relative-load balance of LPT on a heterogeneous cluster with small jobs.
        int[] smallJobs = randomJobs(200, 3);
        double[] balance = loads(lpt(smallJobs, cluster), smallJobs, 3);
        double slowest = Double.MAX_VALUE;
        double fastest = 0;
        for (int machine = 0; machine < 3; machine++) {
            double relative = balance[machine] / cluster[machine];
            slowest = Math.min(slowest, relative);
            fastest = Math.max(fastest, relative);
        }
        check(fastest - slowest <= 3.0 + 1e-9,
                "relative loads stay within one maximum job of each other: " + (fastest - slowest));

        // 7. Degenerate inputs.
        check(makespan(new int[0], new int[0], equal) == 0.0, "no jobs means no makespan");
        check(makespan(new int[]{0}, new int[]{7}, new double[]{2}) == 3.5,
                "one machine processes everything serially");
        int[] zeroJobs = new int[5];
        check(makespan(lpt(zeroJobs, equal), zeroJobs, equal) == 0.0, "zero-sized jobs finish instantly");
        int[] oneJob = {9};
        double[] oneMachine = {3};
        check(makespan(lpt(oneJob, oneMachine), oneJob, oneMachine) == 3.0,
                "a single machine and job: 9/3");
        check(optimal(new int[]{1}, new double[]{1, 5}) == 0.2, "the optimum sends one job to speed 5");

        // 8. Determinism and accounting.
        prngState = 0x777L;
        int[] firstRun = randomJobs(10, 30);
        prngState = 0x777L;
        int[] secondRun = randomJobs(10, 30);
        check(Arrays.equals(firstRun, secondRun), "the instance generator is deterministic");
        int[] repeatA = lpt(firstRun, cluster);
        int[] repeatB = lpt(firstRun, cluster);
        check(Arrays.equals(repeatA, repeatB) && makespan(repeatA, firstRun, cluster)
                == makespan(repeatB, firstRun, cluster), "schedules are reproducible");
        check(scheduledEveryJob(repeatA, firstRun, cluster), "accounting holds on the re-run");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
