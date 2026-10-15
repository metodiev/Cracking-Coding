/** Task 219 - Implement meet-in-the-middle for subset problems (n ~ 40). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static long[] subsetSums(int[] values) {
        long[] sums = new long[1 << values.length];
        int size = 1;
        for (int x : values) {
            for (int i = 0; i < size; i++) sums[size + i] = sums[i] + x;
            size <<= 1;
        }
        return sums;
    }

    static class MeetInTheMiddle {
        final long[] left, right;

        MeetInTheMiddle(int[] nums) {
            int mid = nums.length / 2;
            left = subsetSums(Arrays.copyOfRange(nums, 0, mid));
            right = subsetSums(Arrays.copyOfRange(nums, mid, nums.length));
            Arrays.sort(right);
        }

        boolean exists(long target) {
            for (long s : left) {
                int i = lowerBound(right, target - s);
                if (i < right.length && right[i] == target - s) return true;
            }
            return false;
        }

        long closest(long target) {
            Long best = null;
            for (long s : left) {
                int i = lowerBound(right, target - s);
                for (int j : new int[]{i - 1, i}) {
                    if (j >= 0 && j < right.length) {
                        long candidate = s + right[j];
                        if (best == null || Math.abs(candidate - target) < Math.abs(best - target)) best = candidate;
                    }
                }
            }
            return best == null ? 0 : best;
        }

        private static int lowerBound(long[] a, long value) {
            int lo = 0, hi = a.length;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (a[mid] < value) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }
    }

    static boolean bruteExists(int[] nums, long target) {
        for (long s : subsetSums(nums)) if (s == target) return true;
        return false;
    }

    /** Independent oracle: subset-sum via a BigInteger bitset (brute force is 2^40 here). */
    static boolean subsetExistsBitset(int[] nums, long target) {
        if (target < 0) return false;
        java.math.BigInteger bits = java.math.BigInteger.ONE;
        for (int x : nums) bits = bits.or(bits.shiftLeft(x));
        return bits.testBit((int) target);
    }

    public static void main(String[] args) {
        Random rng = new Random(4);

        int[] small = new int[16];
        for (int i = 0; i < small.length; i++) small[i] = 1 + rng.nextInt(99);
        MeetInTheMiddle mitmSmall = new MeetInTheMiddle(small);
        for (int t = 0; t < 200; t++) {
            long target = rng.nextInt(600);
            check(mitmSmall.exists(target) == bruteExists(small, target), "small cross-check " + target);
        }

        int[] big = new int[40];
        long total = 0;
        for (int i = 0; i < big.length; i++) { big[i] = 1 + rng.nextInt(9); total += big[i]; }
        MeetInTheMiddle mitmBig = new MeetInTheMiddle(big);
        check(mitmBig.exists(total) == subsetExistsBitset(big, total), "full set sums to total");
        check(!mitmBig.exists(total + 1), "impossible target");
        check(mitmBig.exists(0) == subsetExistsBitset(big, 0), "empty subset");
        for (int t = 0; t < 3; t++) {
            long target = rng.nextInt((int) total + 1);
            check(mitmBig.exists(target) == subsetExistsBitset(big, target), "big cross-check " + target);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
