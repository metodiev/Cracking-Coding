/** Task 331 - Weighted job scheduling with reconstruction. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /** Returns {best total weight, chosen indices in increasing order} for (start, end, weight) jobs. */
    static int[] weightedSchedule(int[][] jobs) {
        int n = jobs.length;
        if (n == 0) return new int[]{0};
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> jobs[a][1] != jobs[b][1]
                ? Integer.compare(jobs[a][1], jobs[b][1]) : Integer.compare(jobs[a][0], jobs[b][0]));
        int[] ends = new int[n];
        for (int i = 0; i < n; i++) ends[i] = jobs[order[i]][1];

        // pred[i] = last job (in sorted order) finishing at or before job i starts.
        int[] pred = new int[n];
        for (int i = 0; i < n; i++) {
            int lo = 0, hi = i;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (ends[mid] <= jobs[order[i]][0]) lo = mid + 1;
                else hi = mid;
            }
            pred[i] = lo - 1;
        }

        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int take = jobs[order[i - 1]][2] + dp[pred[i - 1] + 1];
            dp[i] = Math.max(dp[i - 1], take);
        }

        List<Integer> chosen = new ArrayList<>();
        int i = n;
        while (i > 0) {
            if (dp[i] > dp[i - 1]) {
                chosen.add(order[i - 1]);
                i = pred[i - 1] + 1;
            } else {
                i--;
            }
        }
        Collections.reverse(chosen);
        int[] result = new int[chosen.size() + 1];
        result[0] = dp[n];
        for (int k = 0; k < chosen.size(); k++) result[k + 1] = chosen.get(k);
        return result;
    }

    /** Exhaustive best weight over all subsets (n <= 12). */
    static int bruteSchedule(int[][] jobs) {
        int n = jobs.length;
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int weight = 0;
            List<Integer> picked = new ArrayList<>();
            boolean valid = true;
            for (int i = 0; i < n && valid; i++) {
                if (((mask >> i) & 1) == 0) continue;
                for (int j : picked) {
                    if (overlaps(jobs[i], jobs[j])) { valid = false; break; }
                }
                if (valid) { picked.add(i); weight += jobs[i][2]; }
            }
            if (valid) best = Math.max(best, weight);
        }
        return best;
    }

    static boolean overlaps(int[] a, int[] b) {
        return a[0] < b[1] && b[0] < a[1];
    }

    private static int seed = 9091;

    private static int[][] lcgJobs(int count) {
        int[][] jobs = new int[count][3];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            int start = seed % 30;
            seed = (seed * 75 + 74) % 65537;
            int length = 1 + seed % 8;
            seed = (seed * 75 + 74) % 65537;
            jobs[i] = new int[]{start, start + length, 1 + seed % 20};
        }
        return jobs;
    }

    private static String join(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    private static boolean isNonOverlapping(int[][] jobs, int[] indices) {
        for (int i = 0; i < indices.length; i++)
            for (int j = i + 1; j < indices.length; j++)
                if (overlaps(jobs[indices[i]], jobs[indices[j]])) return false;
        return true;
    }

    public static void main(String[] args) {
        // Hand-checked: (1,3,w5) + (4,6,w5) + (6,7,w4) = 14.
        int[][] hand = {{1, 3, 5}, {2, 5, 6}, {4, 6, 5}, {6, 7, 4}};
        int[] result = weightedSchedule(hand);
        check(result[0] == 14 && result.length == 4 && join(Arrays.copyOfRange(result, 1, 4)).equals("0,2,3"),
                "hand value and set");
        int handSum = 0;
        for (int i = 1; i < result.length; i++) handSum += hand[result[i]][2];
        check(handSum == result[0] && isNonOverlapping(hand, Arrays.copyOfRange(result, 1, result.length)),
                "hand feasibility");
        check(result[0] == bruteSchedule(hand), "hand matches brute force");

        check(join(weightedSchedule(new int[][]{} )).equals("0"), "empty");
        check(join(weightedSchedule(new int[][]{{5, 9, 7}})).equals("7,0"), "single job");

        int[][] touch = {{0, 5, 3}, {5, 10, 4}};
        check(join(weightedSchedule(touch)).equals("7,0,1"), "touching intervals chain");

        int[][] heavy = {{0, 10, 100}, {0, 4, 3}, {6, 10, 3}};
        check(join(weightedSchedule(heavy)).equals("100,0"), "one heavy job wins");

        for (int trial = 0; trial < 200; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 12;
            int[][] jobs = lcgJobs(n);
            int[] schedule = weightedSchedule(jobs);
            check(schedule[0] == bruteSchedule(jobs), "random value " + trial);
            int[] chosen = Arrays.copyOfRange(schedule, 1, schedule.length);
            check(isNonOverlapping(jobs, chosen), "random feasibility " + trial);
            int sum = 0;
            for (int index : chosen) sum += jobs[index][2];
            check(sum == schedule[0], "random weight sum " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
