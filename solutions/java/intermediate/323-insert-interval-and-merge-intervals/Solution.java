/** Task 323 - Insert interval and merge intervals. (Tier: Intermediate, CTCI Ch 1) */
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
    /** Sort by start and merge overlapping or touching intervals (start <= current end). */
    static int[][] mergeIntervals(int[][] intervals) {
        if (intervals.length == 0) return new int[0][];
        int[][] ordered = new int[intervals.length][];
        for (int i = 0; i < intervals.length; i++) ordered[i] = intervals[i].clone();
        Arrays.sort(ordered, (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        List<int[]> merged = new ArrayList<>();
        merged.add(ordered[0]);
        for (int i = 1; i < ordered.length; i++) {
            int[] last = merged.get(merged.size() - 1);
            if (ordered[i][0] <= last[1]) {          // touching counts: <=
                last[1] = Math.max(last[1], ordered[i][1]);
            } else {
                merged.add(ordered[i]);
            }
        }
        return merged.toArray(new int[0][]);
    }

    /** Insert into a sorted non-overlapping list, merging overlapping/touching parts. */
    static int[][] insertInterval(int[][] intervals, int[] newInterval) {
        int start = newInterval[0], end = newInterval[1];
        List<int[]> result = new ArrayList<>();
        int i = 0;
        while (i < intervals.length && intervals[i][1] < start) {
            result.add(intervals[i].clone());
            i++;
        }
        while (i < intervals.length && intervals[i][0] <= end) {
            start = Math.min(start, intervals[i][0]);
            end = Math.max(end, intervals[i][1]);
            i++;
        }
        result.add(new int[]{start, end});
        while (i < intervals.length) {
            result.add(intervals[i].clone());
            i++;
        }
        return result.toArray(new int[0][]);
    }

    /** Independent oracle: endpoint events; starts sort before ends at the same x. */
    static int[][] mergeIntervalsSweep(int[][] intervals) {
        int[][] events = new int[intervals.length * 2][];
        int k = 0;
        for (int[] interval : intervals) {
            events[k++] = new int[]{interval[0], 0};
            events[k++] = new int[]{interval[1], 1};
        }
        Arrays.sort(events, (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        List<int[]> merged = new ArrayList<>();
        int depth = 0;
        for (int[] event : events) {
            if (event[1] == 0) {
                if (depth == 0) merged.add(new int[]{event[0], 0});
                depth++;
            } else {
                depth--;
                if (depth == 0) merged.get(merged.size() - 1)[1] = event[0];
            }
        }
        return merged.toArray(new int[0][]);
    }

    static int[][] lcgIntervals(int count, Lcg lcg) {
        int[][] intervals = new int[count][];
        for (int i = 0; i < count; i++) {
            int start = lcg.next(25);
            intervals[i] = new int[]{start, start + 1 + lcg.next(8)};
        }
        return intervals;
    }

    static int[][] lcgDisjointSorted(int count, Lcg lcg) {
        int[][] intervals = new int[count][];
        int cursor = 0;
        for (int i = 0; i < count; i++) {
            cursor += 1 + lcg.next(4);
            int end = cursor + 1 + lcg.next(5);
            intervals[i] = new int[]{cursor, end};
            cursor = end + 1;
        }
        return intervals;
    }

    public static void main(String[] args) {
        check(Arrays.deepEquals(mergeIntervals(new int[][]{{1, 3}, {2, 6}, {8, 10}, {15, 18}}),
                new int[][]{{1, 6}, {8, 10}, {15, 18}}), "classic merge");
        check(Arrays.deepEquals(mergeIntervals(new int[][]{{1, 2}, {2, 3}}),
                new int[][]{{1, 3}}), "touching merge policy");
        check(Arrays.deepEquals(mergeIntervals(new int[][]{{1, 4}, {4, 5}, {7, 9}}),
                new int[][]{{1, 5}, {7, 9}}), "chain merge");
        check(Arrays.deepEquals(mergeIntervals(new int[][]{{1, 10}, {2, 3}, {4, 5}}),
                new int[][]{{1, 10}}), "swallowed intervals");
        check(Arrays.deepEquals(mergeIntervals(new int[][]{{5, 5}}), new int[][]{{5, 5}}), "point interval");
        check(mergeIntervals(new int[0][]).length == 0, "empty merge");
        check(Arrays.deepEquals(mergeIntervalsSweep(new int[][]{{1, 2}, {2, 3}}),
                new int[][]{{1, 3}}), "sweep touching merge");

        int[][] base = {{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}};
        check(Arrays.deepEquals(insertInterval(base, new int[]{9, 10}),
                new int[][]{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}), "insert inside");
        check(Arrays.deepEquals(insertInterval(base, new int[]{0, 0}),
                new int[][]{{0, 0}, {1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}), "insert before");
        check(Arrays.deepEquals(insertInterval(base, new int[]{4, 4}),
                new int[][]{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}), "insert swallowed");
        check(Arrays.deepEquals(insertInterval(new int[][]{{1, 2}, {5, 7}}, new int[]{2, 5}),
                new int[][]{{1, 7}}), "insert touches both sides");
        check(Arrays.deepEquals(insertInterval(new int[][]{{3, 4}}, new int[]{1, 2}),
                new int[][]{{1, 2}, {3, 4}}), "insert with gap");
        check(Arrays.deepEquals(insertInterval(new int[0][], new int[]{4, 6}),
                new int[][]{{4, 6}}), "insert into empty");

        Lcg lcg = new Lcg(20077);
        for (int trial = 0; trial < 200; trial++) {
            int[][] intervals = lcgIntervals(6, lcg);
            check(Arrays.deepEquals(mergeIntervals(intervals), mergeIntervalsSweep(intervals)),
                    "random sweep cross-check " + trial);
        }

        Lcg insertLcg = new Lcg(4242);
        for (int trial = 0; trial < 200; trial++) {
            int[][] baseSorted = lcgDisjointSorted(5, insertLcg);
            int newStart = insertLcg.next(30);
            int[] newInterval = new int[]{newStart, newStart + 1 + insertLcg.next(8)};
            int[][] withNew = new int[baseSorted.length + 1][];
            System.arraycopy(baseSorted, 0, withNew, 0, baseSorted.length);
            withNew[baseSorted.length] = newInterval;
            check(Arrays.deepEquals(insertInterval(baseSorted, newInterval), mergeIntervals(withNew)),
                    "insert equals merge of the union " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
