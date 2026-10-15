/** Task 326 - Meeting rooms required (interval partitioning). (Tier: Intermediate, CTCI Ch 1) */
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
    /** Maximum overlap via endpoint events; an end (-1) sorts before a start (+1). */
    static int minMeetingRooms(int[][] intervals) {
        int[][] events = new int[intervals.length * 2][];
        int k = 0;
        for (int[] interval : intervals) {
            events[k++] = new int[]{interval[0], 1};
            events[k++] = new int[]{interval[1], -1};
        }
        Arrays.sort(events, (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        int rooms = 0, best = 0;
        for (int[] event : events) {
            rooms += event[1];
            best = Math.max(best, rooms);
        }
        return best;
    }

    /** Independent sweep with a min-heap of finish times. */
    static int minRoomsHeap(int[][] intervals) {
        int[][] ordered = intervals.clone();
        Arrays.sort(ordered, (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        for (int[] interval : ordered) {
            if (!heap.isEmpty() && heap.peek() <= interval[0]) heap.poll();
            heap.add(interval[1]);
        }
        return heap.size();
    }

    /** Independent oracle for small coordinates: count active meetings at each minute. */
    static int minRoomsPerMinute(int[][] intervals) {
        if (intervals.length == 0) return 0;
        int lo = Integer.MAX_VALUE, hi = Integer.MIN_VALUE;
        for (int[] interval : intervals) {
            lo = Math.min(lo, interval[0]);
            hi = Math.max(hi, interval[1]);
        }
        int best = 0;
        for (int minute = lo; minute <= hi; minute++) {
            int active = 0;
            for (int[] interval : intervals) {
                if (interval[0] <= minute && minute < interval[1]) active++;
            }
            best = Math.max(best, active);
        }
        return best;
    }

    static int[][] lcgIntervals(int count, Lcg lcg) {
        int[][] intervals = new int[count][];
        for (int i = 0; i < count; i++) {
            int start = lcg.next(30);
            intervals[i] = new int[]{start, start + 1 + lcg.next(10)};
        }
        return intervals;
    }

    public static void main(String[] args) {
        check(minMeetingRooms(new int[][]{{0, 30}, {5, 10}, {15, 20}}) == 2, "classic rooms 2");
        check(minMeetingRooms(new int[][]{{7, 10}, {2, 4}}) == 1, "classic rooms 1");
        check(minRoomsHeap(new int[][]{{0, 30}, {5, 10}, {15, 20}}) == 2, "heap rooms 2");
        check(minRoomsHeap(new int[][]{{7, 10}, {2, 4}}) == 1, "heap rooms 1");

        check(minMeetingRooms(new int[][]{{1, 2}, {2, 3}, {3, 4}}) == 1, "back-to-back shares a room");
        check(minMeetingRooms(new int[][]{{1, 5}, {2, 6}, {3, 7}}) == 3, "triple overlap");
        check(minMeetingRooms(new int[0][]) == 0, "no meetings");
        check(minRoomsHeap(new int[0][]) == 0, "no meetings heap");

        Lcg lcg = new Lcg(2718);
        for (int trial = 0; trial < 300; trial++) {
            int count = 1 + lcg.next(10);
            int[][] intervals = lcgIntervals(count, lcg);
            int expected = minRoomsPerMinute(intervals);
            check(minMeetingRooms(intervals) == expected, "random event cross-check " + trial);
            check(minRoomsHeap(intervals) == expected, "random heap cross-check " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
