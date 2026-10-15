/** Task 330 - Weighted interval scheduling (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

public class Solution {
    /**
     * Jobs are {start, finish, weight}; compatible when one finishes at or before the
     * next starts (finish <= start). Returns {best total weight, chosen jobs...}.
     */
    static int[][] weightedIntervalScheduling(int[][] jobs) {
        if (jobs.length == 0) return new int[][]{{0}};
        int[][] ordered = jobs.clone();
        Arrays.sort(ordered, (u, v) -> u[1] != v[1] ? Integer.compare(u[1], v[1]) : Integer.compare(u[0], v[0]));
        int[] finishes = new int[ordered.length];
        for (int i = 0; i < ordered.length; i++) finishes[i] = ordered[i][1];
        int[] best = new int[ordered.length + 1];
        boolean[] choose = new boolean[ordered.length + 1];
        for (int i = 1; i <= ordered.length; i++) {
            int without = best[i - 1];
            int withJob = ordered[i - 1][2] + best[lastCompatible(finishes, ordered[i - 1][0], i - 1) + 1];
            if (withJob > without) {
                best[i] = withJob;
                choose[i] = true;
            } else {
                best[i] = without;
            }
        }
        List<int[]> chosen = new ArrayList<>();
        int i = ordered.length;
        while (i > 0) {
            if (choose[i]) {
                chosen.add(ordered[i - 1]);
                i = lastCompatible(finishes, ordered[i - 1][0], i - 1) + 1;
            } else {
                i--;
            }
        }
        Collections.reverse(chosen);
        int[][] result = new int[chosen.size() + 1][];
        result[0] = new int[]{best[ordered.length]};
        for (int k = 0; k < chosen.size(); k++) result[k + 1] = chosen.get(k);
        return result;
    }

    /** Largest index j < i with finishes[j] <= start; -1 when none exists. */
    static int lastCompatible(int[] finishes, int start, int i) {
        int lo = 0, hi = i - 1, best = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (finishes[mid] <= start) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return best;
    }

    /** Exhaustive subset enumeration; the independent oracle for small inputs. */
    static int bruteForce(int[][] jobs) {
        int n = jobs.length;
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            List<int[]> chosen = new ArrayList<>();
            int total = 0;
            for (int i = 0; i < n; i++) {
                if ((mask >> i & 1) == 1) {
                    chosen.add(jobs[i]);
                    total += jobs[i][2];
                }
            }
            if (total <= best) continue;
            chosen.sort((u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
            boolean compatible = true;
            for (int k = 0; k + 1 < chosen.size(); k++) {
                if (chosen.get(k)[1] > chosen.get(k + 1)[0]) compatible = false;
            }
            if (compatible) best = total;
        }
        return best;
    }

    static int[][] lcgJobs(int count, Lcg lcg) {
        int[][] jobs = new int[count][];
        for (int i = 0; i < count; i++) {
            int start = lcg.next(30);
            int finish = start + 1 + lcg.next(8);
            jobs[i] = new int[]{start, finish, 1 + lcg.next(20)};
        }
        return jobs;
    }

    public static void main(String[] args) {
        // the job set from the task description: (1,3)+(4,6)+(6,7) = 5+5+4 = 14, not 12
        int[][] briefJobs = {{1, 3, 5}, {2, 5, 6}, {4, 6, 5}, {6, 7, 4}};
        int[][] result = weightedIntervalScheduling(briefJobs);
        check(result[0][0] == 14, "brief job set is 14, not 12: 5+5+4");
        check(Arrays.deepEquals(Arrays.copyOfRange(result, 1, result.length),
                new int[][]{{1, 3, 5}, {4, 6, 5}, {6, 7, 4}}), "brief job set chosen jobs");
        check(bruteForce(briefJobs) == 14, "brief job set brute force");

        // hand-solved example: (1,3,5)+(3,5,4)+(5,7,3) = 12; the heavy decoys lose
        int[][] hand = {{1, 3, 5}, {3, 5, 4}, {5, 7, 3}, {1, 7, 10}, {2, 3, 2}};
        result = weightedIntervalScheduling(hand);
        check(result[0][0] == 12, "hand example totals 12");
        check(Arrays.deepEquals(Arrays.copyOfRange(result, 1, result.length),
                new int[][]{{1, 3, 5}, {3, 5, 4}, {5, 7, 3}}), "hand example chosen jobs");
        check(bruteForce(hand) == 12, "hand example brute force");

        result = weightedIntervalScheduling(new int[0][]);
        check(result[0][0] == 0 && result.length == 1, "no jobs");
        result = weightedIntervalScheduling(new int[][]{{4, 9, 7}});
        check(result[0][0] == 7 && Arrays.equals(result[1], new int[]{4, 9, 7}), "single job");

        Lcg lcg = new Lcg(57721);
        for (int trial = 0; trial < 60; trial++) {
            int count = 4 + lcg.next(6);
            int[][] jobs = lcgJobs(count, lcg);
            result = weightedIntervalScheduling(jobs);
            check(result[0][0] == bruteForce(jobs), "random brute force cross-check " + trial);
            int total = 0;
            int[][] chosen = Arrays.copyOfRange(result, 1, result.length);
            for (int[] job : chosen) total += job[2];
            check(total == result[0][0], "chosen weights match total " + trial);
            for (int k = 0; k + 1 < chosen.length; k++) {
                check(chosen[k][1] <= chosen[k + 1][0], "chosen jobs are compatible " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
