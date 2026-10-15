/** Task 416 - Implement bitset DP optimizations for subset problems. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Subset-sum reachability with a bitset: bit s of `mask` is set iff some subset of the
 * processed values sums to s. Adding value v shifts the whole reachable set by v:
 *     mask |= mask << v        (masked to the [0, limit] window after every step)
 * Java's int/long cannot address bit 4000 or 20000, so the bitset is a BigInteger —
 * `BigInteger.equals/bitCount/bitLength` act as the word-array operations. The classic
 * boolean DP is kept as the oracle and the reachable sets must match exactly.
 */
import java.math.BigInteger;
import java.util.*;

public class Solution {
    static BigInteger reachableMask(int[] values, int limit) {
        BigInteger mask = BigInteger.ONE;               // the empty subset sums to 0
        BigInteger window = BigInteger.ONE.shiftLeft(limit + 1).subtract(BigInteger.ONE);
        for (int v : values) {
            if (v <= limit) mask = mask.or(mask.shiftLeft(v)).and(window);
        }
        return mask.and(window);
    }

    static boolean canMake(int[] values, int target, int limit) {
        if (target < 0 || target > limit) return false;
        return reachableMask(values, limit).testBit(target);
    }

    static boolean canMake(int[] values, int target) {
        return canMake(values, target, target);
    }

    static boolean canPartition(int[] values) {
        long total = 0;
        for (int v : values) total += v;
        if (total % 2 != 0) return false;
        return canMake(values, (int) (total / 2));
    }

    /** Oracle: plain boolean DP over sums 0..limit. */
    static boolean[] classicDp(int[] values, int limit) {
        boolean[] reach = new boolean[limit + 1];
        reach[0] = true;
        for (int v : values) {
            if (v > limit) continue;
            for (int s = limit; s >= v; s--) {
                if (reach[s - v]) reach[s] = true;
            }
        }
        return reach;
    }

    private static int seed = 416;

    private static int[] lcgValues(int count, int lo, int hi) {
        int[] values = new int[count];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            values[k] = lo + seed % (hi - lo + 1);
        }
        return values;
    }

    public static void main(String[] args) {
        int[] base = {3, 34, 4, 12, 5, 2};
        // classic subset-sum hand cases
        check(canMake(base, 9), "base 9");            // 3 + 4 + 2 (or 5 + 4)
        check(!canMake(base, 30), "base 30");         // sums jump from 26 to 34+
        check(canMake(base, 34), "base 34");          // the single element 34
        check(canMake(base, 0), "base 0");            // the empty subset
        check(canMake(base, 60), "base 60");          // everything
        check(!canMake(base, 59), "base 59");
        check(canMake(base, 50), "base 50");          // 34 + 12 + 4

        // hand-derived reachable set of {2, 3, 4, 5, 12, 34}: {2,3,4,5} reaches
        // {0, 2..12, 14} (1 and 13 stay empty), adding 12 fills {0, 2..12, 14..24, 26}
        // (13 and 25 stay empty), and adding 34 mirrors that set 34 higher:
        // {0, 2..12, 14..24, 26, 34, 36..46, 48..58, 60} — 48 values.
        BigInteger mask = reachableMask(base, 60);
        check(mask.bitCount() == 48, "hand count 48");
        for (int s : new int[]{0, 2, 12, 14, 24, 26, 34, 36, 46, 48, 58, 60}) {
            check(mask.testBit(s), "hand present " + s);
        }
        for (int s : new int[]{1, 13, 25, 35, 47, 59}) {
            check(!mask.testBit(s), "hand absent " + s);
        }
        check(!canMake(base, 25), "base 25");

        // equal-partition hand cases
        check(canPartition(new int[]{1, 5, 11, 5}), "partition 1 5 11 5");   // 11 vs 1 + 5 + 5
        check(!canPartition(new int[]{1, 5, 3}), "partition odd total");
        check(canPartition(new int[]{2, 2, 2, 2}), "partition four 2s");

        // oracle agreement on hand and LCG sets
        int[][] handSets = {{3, 34, 4, 12, 5, 2}, {1, 5, 11, 5}, {2, 2, 2, 2}, {7}, {}};
        int[] handLimits = {60, 22, 8, 7, 5};
        for (int h = 0; h < handSets.length; h++) {
            BigInteger m = reachableMask(handSets[h], handLimits[h]);
            boolean[] oracle = classicDp(handSets[h], handLimits[h]);
            for (int s = 0; s <= handLimits[h]; s++) {
                check(m.testBit(s) == oracle[s], "hand oracle set " + h + " sum " + s);
            }
        }

        for (int round = 0; round < 8; round++) {
            int n = 10 + seed % 31;
            int[] values = lcgValues(n, 1, 60);
            int sum = 0;
            for (int v : values) sum += v;
            int limit = Math.min(sum, 1500);
            BigInteger m = reachableMask(values, limit);
            boolean[] oracle = classicDp(values, limit);
            for (int s = 0; s <= limit; s++) {
                check(m.testBit(s) == oracle[s], "lcg " + round + " sum " + s);
            }
            check(m.bitLength() <= limit + 1, "lcg mask length " + round);
            int running = 0;
            for (int v : values) {
                running += v;
                if (running <= limit) check(m.testBit(running), "prefix sum " + round);
            }
            check(!canMake(values, limit + 1, limit), "truncated target " + round);
        }

        // limits beyond 63 bits: a single 4000-value jump needs bit 4000
        BigInteger big = reachableMask(new int[]{4000}, 5000);
        check(big.testBit(4000) && !big.testBit(4001), "bit 4000");
        check(big.bitLength() == 4001, "bit length 4001");
        check(canMake(new int[]{4000}, 4000) && !canMake(new int[]{4000}, 3999), "single jump");

        // n = 40 stress: no 32/64-bit integer could hold these masks
        seed = 65537 / 2;
        for (int round = 0; round < 3; round++) {
            int[] values = lcgValues(40, 1, 1000);
            int limit = 20000;
            BigInteger m = reachableMask(values, limit);
            boolean[] oracle = classicDp(values, limit);
            int reached = 0;
            for (int s = 0; s <= limit; s++) {
                boolean bit = m.testBit(s);
                check(bit == oracle[s], "stress " + round + " sum " + s);
                if (bit) reached++;
            }
            int subsetSum = 0;
            for (int k = 0; k < 20; k++) subsetSum += values[k];
            if (subsetSum <= limit) check(m.testBit(subsetSum), "stress subset sum");
            check(m.bitLength() <= limit + 1, "stress length");
            int running = 0;
            for (int k = 0; k < 5; k++) {
                running += values[k];
                check(m.testBit(running), "stress prefix");
            }
            check(reached >= 1, "stress reached");
        }

        // partition on n = 40 inputs (both parities exercised)
        int[] values40 = new int[40];
        for (int i = 0; i < 40; i++) values40[i] = i + 1;         // total 820, even
        check(canPartition(values40), "partition 1..40");          // 1..n reaches every sum
        int[] values41 = new int[41];
        for (int i = 0; i < 41; i++) values41[i] = i + 1;         // total 861, odd
        check(!canPartition(values41), "partition odd 1..41");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
