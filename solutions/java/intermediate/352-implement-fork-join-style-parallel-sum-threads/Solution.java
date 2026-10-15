/** Task 352 - Implement fork-join style parallel sum (threads). (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;

public class Solution {
    /** Leaf ranges hold at most THRESHOLD elements; larger ranges split in half. */
    static final int THRESHOLD = 256;

    static class Stats {
        private int leaves;
        private int spawned;

        synchronized void countLeaf() { leaves++; }
        synchronized void countSpawn() { spawned++; }
        synchronized int leaves() { return leaves; }
        synchronized int spawned() { return spawned; }
    }

    /** sumRange returns the sum and records leaf/spawn accounting in stats. */
    static long sumRange(long[] values, int lo, int hi, Stats stats, long timeoutMillis) {
        if (hi - lo <= THRESHOLD) {                   // leaf task: sum without splitting
            stats.countLeaf();
            long sum = 0;
            for (int i = lo; i < hi; i++) sum += values[i];
            return sum;
        }

        int mid = (lo + hi) / 2;
        long[] forked = new long[1];
        stats.countSpawn();
        Thread thread = new Thread(() -> { forked[0] = sumRange(values, lo, mid, stats, timeoutMillis); });
        thread.start();
        long right = sumRange(values, mid, hi, stats, timeoutMillis);   // one half inline
        join(thread, timeoutMillis);                                    // join the fork
        return forked[0] + right;
    }

    static void join(Thread thread, long timeoutMillis) {
        try {
            thread.join(timeoutMillis);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
        check(!thread.isAlive(), "forked worker did not finish in time");
    }

    /** Independent (iterative) oracle for the number of leaves the split policy creates. */
    static int countLeaves(int size) {
        int leaves = 0;
        Deque<int[]> pending = new ArrayDeque<>();
        pending.push(new int[]{0, size});
        while (!pending.isEmpty()) {
            int[] range = pending.pop();
            if (range[1] - range[0] <= THRESHOLD) {
                leaves++;
            } else {
                int mid = (range[0] + range[1]) / 2;
                pending.push(new int[]{range[0], mid});
                pending.push(new int[]{mid, range[1]});
            }
        }
        return leaves;
    }

    static long[] lcgArray(int size) {
        long[] values = new long[size];
        long seed = 24680;
        for (int i = 0; i < size; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed;
        }
        return values;
    }

    static long sequentialSum(long[] values) {
        long total = 0;
        for (long value : values) total += value;
        return total;
    }

    public static void main(String[] args) {
        Stats emptyStats = new Stats();
        check(sumRange(new long[0], 0, 0, emptyStats, 10000) == 0, "empty array sums to 0");
        check(emptyStats.leaves() == 1 && emptyStats.spawned() == 0, "empty range is a single leaf");

        int[] sizes = {1, 7, THRESHOLD, THRESHOLD + 1, 2 * THRESHOLD, 3 * THRESHOLD,
                       1024, 2048, 4096, 8192};
        for (int size : sizes) {
            long[] values = lcgArray(size);
            Stats stats = new Stats();
            long total = sumRange(values, 0, size, stats, 10000);
            check(total == sequentialSum(values), "parallel sum mismatch at size " + size);
            check(stats.leaves() == countLeaves(size), "leaf count at size " + size);
            check(stats.spawned() == stats.leaves() - 1, "binary fork tree at size " + size);
        }

        // Documented threshold accounting for power-of-two sizes: leaves == n / THRESHOLD.
        int[] powers = {512, 1024, 2048, 4096, 8192};
        for (int size : powers) {
            Stats stats = new Stats();
            long[] values = lcgArray(size);
            check(sumRange(values, 0, size, stats, 10000) == sequentialSum(values),
                  "power-of-two sum at " + size);
            check(stats.leaves() == size / THRESHOLD, "power-of-two leaves at " + size);
            check(stats.spawned() == size / THRESHOLD - 1, "power-of-two threads at " + size);
        }

        // Splitting really happens above the threshold, and 8192 splits into exactly 31 forks.
        Stats atThreshold = new Stats();
        sumRange(lcgArray(THRESHOLD), 0, THRESHOLD, atThreshold, 10000);
        check(atThreshold.spawned() == 0, "at threshold: no fork");
        Stats justAbove = new Stats();
        sumRange(lcgArray(THRESHOLD + 1), 0, THRESHOLD + 1, justAbove, 10000);
        check(justAbove.spawned() == 1, "just above: one fork");
        Stats big = new Stats();
        sumRange(lcgArray(8192), 0, 8192, big, 10000);
        check(big.spawned() == 31, "8192 -> 32 leaves, 31 forks");

        // LCG-driven sizes across the whole range, each cross-checked against the sequential sum.
        long seed = 9753;
        for (int trial = 0; trial < 20; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = (int) (seed % 3000);
            long[] values = lcgArray(size);
            Stats stats = new Stats();
            long total = sumRange(values, 0, size, stats, 10000);
            check(total == sequentialSum(values), "LCG size " + size);
            check(stats.leaves() == countLeaves(size), "LCG leaves at " + size);
            check(stats.spawned() == stats.leaves() - 1, "LCG fork tree at " + size);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
