/** Task 118 - Longest increasing subarray (contiguous). (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    /** Returns {length, start}; start = -1 for an empty array. */
    static int[] longestIncreasingSubarray(int[] a) {
        if (a.length == 0) return new int[]{0, -1};
        int bestLen = 1, bestStart = 0, curLen = 1, curStart = 0;
        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[i - 1]) {
                curLen++;
            } else {
                curLen = 1;
                curStart = i;
            }
            if (curLen > bestLen) {
                bestLen = curLen;
                bestStart = curStart;
            }
        }
        return new int[]{bestLen, bestStart};
    }

    public static void main(String[] args) {
        check(longestIncreasingSubarray(new int[]{1, 2, 3, 1, 2})[0] == 3, "run at start");
        check(longestIncreasingSubarray(new int[]{5, 4, 3})[0] == 1, "decreasing");
        int[] r = longestIncreasingSubarray(new int[]{1, 3, 2, 4, 5, 6});
        check(r[0] == 4 && r[1] == 2, "later run");
        check(longestIncreasingSubarray(new int[]{})[1] == -1, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
