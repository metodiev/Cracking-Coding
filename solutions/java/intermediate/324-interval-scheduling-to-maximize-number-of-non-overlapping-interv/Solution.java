/** Task 324 - Interval scheduling to maximize number of non-overlapping intervals (greedy). (Tier: Intermediate, CTCI Ch 1) */
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
    /** Greedy earliest finish time; touching endpoints count as overlapping (start > lastEnd). */
    static int maxNonOverlapping(int[][] intervals) {
        int[][] ordered = intervals.clone();
        Arrays.sort(ordered, (u, v) -> u[1] != v[1] ? Integer.compare(u[1], v[1]) : Integer.compare(u[0], v[0]));
        int count = 0;
        int lastEnd = Integer.MIN_VALUE;
        boolean hasLast = false;
        for (int[] interval : ordered) {
            if (!hasLast || interval[0] > lastEnd) {
                count++;
                lastEnd = interval[1];
                hasLast = true;
            }
        }
        return count;
    }

    /** Independent DP: best[i] = max(best[i-1], 1 + best[p(i)]) with p by binary search. */
    static int maxNonOverlappingDp(int[][] intervals) {
        if (intervals.length == 0) return 0;
        int[][] ordered = intervals.clone();
        Arrays.sort(ordered, (u, v) -> u[1] != v[1] ? Integer.compare(u[1], v[1]) : Integer.compare(u[0], v[0]));
        int[] finishes = new int[ordered.length];
        for (int i = 0; i < ordered.length; i++) finishes[i] = ordered[i][1];
        int[] best = new int[ordered.length + 1];
        for (int i = 1; i <= ordered.length; i++) {
            int start = ordered[i - 1][0];
            int lo = 0, hi = i - 1;                 // last j < i with finishes[j-1] < start
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (finishes[mid - 1] < start) lo = mid;
                else hi = mid - 1;
            }
            int j = (lo > 0 && finishes[lo - 1] < start) ? lo : 0;
            best[i] = Math.max(best[i - 1], 1 + best[j]);
        }
        return best[ordered.length];
    }

    static int[][] lcgIntervals(int count, Lcg lcg) {
        int[][] intervals = new int[count][];
        for (int i = 0; i < count; i++) {
            int start = lcg.next(20);
            intervals[i] = new int[]{start, start + 1 + lcg.next(6)};
        }
        return intervals;
    }

    public static void main(String[] args) {
        // classic example: [1,2] and [2,3] touch, so they conflict; best is 2
        check(maxNonOverlapping(new int[][]{{1, 2}, {2, 3}, {3, 4}, {1, 3}}) == 2, "classic greedy");
        check(maxNonOverlappingDp(new int[][]{{1, 2}, {2, 3}, {3, 4}, {1, 3}}) == 2, "classic dp");

        // chain with gaps: all four fit; the wide decoy is not worth taking
        int[][] chain = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {1, 8}};
        check(maxNonOverlapping(chain) == 4, "chain greedy");
        check(maxNonOverlappingDp(chain) == 4, "chain dp");
        check(maxNonOverlapping(new int[][]{{1, 2}, {3, 4}, {5, 6}, {7, 8}}) == 4, "plain chain");

        check(maxNonOverlapping(new int[0][]) == 0, "empty");
        check(maxNonOverlapping(new int[][]{{1, 5}}) == 1, "single");
        check(maxNonOverlapping(new int[][]{{1, 3}, {2, 4}, {3, 5}, {4, 6}}) == 2, "overlap chain");
        check(maxNonOverlapping(new int[][]{{1, 2}, {1, 2}, {1, 2}}) == 1, "duplicates");
        check(maxNonOverlapping(new int[][]{{1, 4}, {4, 4}}) == 1, "touching conflicts");

        Lcg lcg = new Lcg(909);
        for (int trial = 0; trial < 400; trial++) {
            int count = 1 + lcg.next(9);
            int[][] intervals = lcgIntervals(count, lcg);
            check(maxNonOverlapping(intervals) == maxNonOverlappingDp(intervals),
                    "random dp cross-check " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
