/** Task 120 - Implement LIS using patience sorting idea (O(n log n) outline). (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /** LIS length via the tails array + binary search. O(n log n). */
    static int lisLength(int[] a) {
        int[] tails = new int[a.length];
        int size = 0;
        for (int x : a) {
            int lo = 0, hi = size;
            while (lo < hi) {                       // first index with tails[i] >= x
                int mid = (lo + hi) / 2;
                if (tails[mid] < x) lo = mid + 1;
                else hi = mid;
            }
            tails[lo] = x;
            if (lo == size) size++;
        }
        return size;
    }

    /** Length + one concrete LIS via parent tracking. */
    static int[] lisReconstructed(int[] a) {
        if (a.length == 0) return new int[0];
        int[] tails = new int[a.length], tailIdx = new int[a.length], parent = new int[a.length];
        Arrays.fill(parent, -1);
        int size = 0;
        for (int i = 0; i < a.length; i++) {
            int lo = 0, hi = size;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (tails[mid] < a[i]) lo = mid + 1;
                else hi = mid;
            }
            tails[lo] = a[i];
            tailIdx[lo] = i;
            parent[i] = lo > 0 ? tailIdx[lo - 1] : -1;
            if (lo == size) size++;
        }
        int[] seq = new int[size];
        int node = tailIdx[size - 1];
        for (int k = size - 1; k >= 0; k--) {
            seq[k] = a[node];
            node = parent[node];
        }
        return seq;
    }

    static int naiveLis(int[] a) {
        int[] dp = new int[a.length];
        int best = 0;
        for (int i = 0; i < a.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) if (a[j] < a[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
            best = Math.max(best, dp[i]);
        }
        return best;
    }

    public static void main(String[] args) {
        int[] a = {10, 9, 2, 5, 3, 7, 101, 18};
        check(lisLength(a) == 4, "classic sample");
        check(lisLength(new int[]{}) == 0, "empty");
        check(lisLength(new int[]{3, 2, 1}) == 1, "decreasing");
        check(Arrays.equals(lisReconstructed(a), new int[]{2, 3, 7, 18}), "reconstruction");

        int[][] cases = {{1, 3, 2}, {5, 1, 2, 3, 4}, {2, 2, 2}, {-1, -2, -3}};
        for (int[] c : cases) check(lisLength(c) == naiveLis(c), "agrees with naive on " + Arrays.toString(c));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
