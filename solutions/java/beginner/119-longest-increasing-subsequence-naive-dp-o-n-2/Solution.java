/** Task 119 - Longest increasing subsequence (naive DP O(n^2)). (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static int lisLength(int[] a) {
        if (a.length == 0) return 0;
        int[] dp = new int[a.length];
        Arrays.fill(dp, 1);
        int best = 1;
        for (int i = 1; i < a.length; i++) {
            for (int j = 0; j < i; j++)
                if (a[j] < a[i] && dp[j] + 1 > dp[i]) dp[i] = dp[j] + 1;
            best = Math.max(best, dp[i]);
        }
        return best;
    }

    static List<Integer> lisWithSequence(int[] a) {
        if (a.length == 0) return List.of();
        int[] dp = new int[a.length], parent = new int[a.length];
        Arrays.fill(dp, 1);
        Arrays.fill(parent, -1);
        int best = 0;
        for (int i = 1; i < a.length; i++) {
            for (int j = 0; j < i; j++)
                if (a[j] < a[i] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;
                }
            if (dp[i] >= dp[best]) best = i;   // >= keeps the LAST index
        }
        LinkedList<Integer> seq = new LinkedList<>();
        for (int node = best; node != -1; node = parent[node]) seq.addFirst(a[node]);
        return seq;
    }

    public static void main(String[] args) {
        int[] a = {10, 9, 2, 5, 3, 7, 101, 18};
        check(lisLength(a) == 4, "classic sample");
        check(lisLength(new int[]{}) == 0, "empty");
        check(lisLength(new int[]{3, 2, 1}) == 1, "decreasing");
        check(lisLength(new int[]{1, 2, 3, 4}) == 4, "increasing");
        check(lisWithSequence(a).toString().equals("[2, 5, 7, 18]"), "reconstruction");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
