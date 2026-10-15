/** Task 481 - Implement optimistic checkpointing/restart simulation. (Tier: Advanced, CTCI Ch 14)
 *
 * Model (documented semantics):
 * - A job applies one effect per step, step ids 0..n-1. Effects are idempotent: replaying a step
 *   never duplicates effect id `step` (apply() is a set insert).
 * - Optimistic checkpointing: after every `interval` completed steps (progress % interval == 0)
 *   the job snapshots (progress, effects) into a single checkpoint slot, overwriting the old one.
 * - A crash when "about to start step k" means steps 0..k-1 executed and the checkpoint slot is
 *   the last one taken among them. Recovery rolls back to that slot: effects of steps that were
 *   never checkpointed are lost (optimistic, uncommitted), while everything at or before the last
 *   checkpoint survives: no committed effect is ever lost.
 * - Recovered progress is exactly (k / interval) * interval, so the amount of redone work is
 *   k - recovered < interval steps (the documented redo bound).
 */
import java.util.*;

class OptimisticJob {
    final int n;
    final int interval;
    int progress;
    final List<Integer> effects = new ArrayList<>();
    int checkpointProgress;
    List<Integer> checkpointEffects = new ArrayList<>();
    int checkpointCount;
    int executions;

    OptimisticJob(int n, int interval) {
        this.n = n;
        this.interval = interval;
    }

    void apply(int step) {
        executions++;
        if (!effects.contains(step)) effects.add(step);
    }

    /** Execute steps 0..k-1, then crash while trying to start step k. */
    void runUntil(int k) {
        while (progress < k) {
            int step = progress;
            apply(step);
            progress = step + 1;
            if (progress % interval == 0) {
                checkpointProgress = progress;
                checkpointEffects = new ArrayList<>(effects);
                checkpointCount++;
            }
        }
    }

    void recover() {
        progress = checkpointProgress;
        effects.clear();
        effects.addAll(checkpointEffects);
    }

    void finish() {
        runUntil(n);
    }

    int redone() {
        return executions - effects.size();
    }
}

public class Solution {
    private static int seed = 481;

    private static int[] crashPoints(int count, int limit) {
        int[] points = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            points[i] = seed % limit;
        }
        return points;
    }

    private static OptimisticJob verifyCrash(int n, int interval, int k) {
        OptimisticJob job = new OptimisticJob(n, interval);
        job.runUntil(k);
        check(job.progress == k, "progress at crash " + k);
        job.recover();
        int expected = (k / interval) * interval;
        check(job.progress == expected, "recovered progress " + k);
        check(job.progress <= k, "recovered <= crash " + k);
        check(k - job.progress < interval, "redo bound " + k);
        check(effectsSorted(job).equals(rangeList(expected)), "committed effects survive " + k);
        job.finish();
        check(job.progress == n, "finish progress");
        check(effectsSorted(job).equals(rangeList(n)), "final effect set");
        check(job.redone() == k - expected, "redone count " + k);
        return job;
    }

    private static List<Integer> effectsSorted(OptimisticJob job) {
        List<Integer> sorted = new ArrayList<>(job.effects);
        Collections.sort(sorted);
        return sorted;
    }

    private static List<Integer> rangeList(int n) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) list.add(i);
        return list;
    }

    public static void main(String[] args) {
        OptimisticJob job = new OptimisticJob(25, 4);
        job.runUntil(7);
        check(job.progress == 7 && effectsSorted(job).equals(rangeList(7)), "crash state");
        job.recover();
        check(job.progress == 4, "first recovery");
        check(effectsSorted(job).equals(rangeList(4)), "first recovered effects");
        job.runUntil(16);
        job.recover();
        check(job.progress == 16, "second recovery");
        check(effectsSorted(job).equals(rangeList(16)), "second recovered effects");
        job.runUntil(23);
        job.recover();
        check(job.progress == 20, "third recovery");
        check(effectsSorted(job).equals(rangeList(20)), "third recovered effects");
        job.finish();
        check(job.progress == 25, "final progress");
        check(effectsSorted(job).equals(rangeList(25)), "final effects");
        check(job.executions == 7 + 12 + 7 + 5, "execution count");
        check(job.redone() == (7 - 4) + (23 - 20), "total redone");
        check(job.checkpointCount == 6, "checkpoint count");
        check(job.checkpointProgress == 24 && job.checkpointEffects.size() == 24, "last checkpoint");

        for (int k : new int[]{0, 3, 7, 8, 16, 23}) {
            OptimisticJob verified = verifyCrash(25, 4, k);
            check(verified.progress == 25, "scripted finish " + k);
        }
        int[][] expectations = {{3, 0}, {7, 4}, {8, 8}, {16, 16}, {23, 20}};
        for (int[] pair : expectations) {
            check((pair[0] / 4) * 4 == pair[1], "boundary expectation " + pair[0]);
        }

        int checked = 0;
        for (int k : crashPoints(20, 26)) {
            int expected = (k / 4) * 4;
            OptimisticJob verified = verifyCrash(25, 4, k);
            check(verified.progress == 25 && expected == (k / 4) * 4, "lcg crash " + k);
            checked++;
        }
        check(checked == 20, "lcg crash count");

        for (int interval : new int[]{1, 3, 5}) {
            for (int k = 0; k <= 20; k++) {
                OptimisticJob verified = verifyCrash(20, interval, k);
                check(verified.redone() == k - (k / interval) * interval, "redo " + interval + " " + k);
                check(verified.redone() < interval, "redo bound " + interval);
            }
        }
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
