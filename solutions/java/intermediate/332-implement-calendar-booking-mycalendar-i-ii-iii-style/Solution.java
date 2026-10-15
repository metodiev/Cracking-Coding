/** Task 332 - Implement calendar booking (MyCalendar I/II/III style). (Tier: Intermediate, CTCI Ch 7) */
import java.util.*;

class MyCalendar {
    final List<int[]> accepted = new ArrayList<>();

    /** Accepts a booking iff it overlaps no accepted booking. */
    boolean book(int start, int end) {
        for (int[] other : accepted) if (MyCalendar.overlaps(start, end, other)) return false;
        accepted.add(new int[]{start, end});
        return true;
    }

    static boolean overlaps(int start, int end, int[] other) {
        return start < other[1] && other[0] < end;
    }

    static boolean overlaps(int[] a, int[] b) {
        return a[0] < b[1] && b[0] < a[1];
    }
}

class MyCalendarTwo {
    final List<int[]> singles = new ArrayList<>();
    final List<int[]> pairs = new ArrayList<>();

    /** Allows double booking: rejects a booking that would create a triple overlap. */
    boolean book(int start, int end) {
        for (int[] other : pairs) if (MyCalendar.overlaps(start, end, other)) return false;
        for (int[] other : singles) {
            if (MyCalendar.overlaps(start, end, other)) {
                pairs.add(new int[]{Math.max(start, other[0]), Math.min(end, other[1])});
            }
        }
        singles.add(new int[]{start, end});
        return true;
    }

    int maxOverlap() {
        return Solution.sweepPeak(singles);
    }
}

class MyCalendarK {
    private final int k;
    private final TreeMap<Integer, Integer> delta = new TreeMap<>();

    MyCalendarK(int k) {
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        this.k = k;
    }

    /** Accepts a booking iff no instant ends up covered by more than k bookings. */
    boolean book(int start, int end) {
        delta.merge(start, 1, Integer::sum);
        delta.merge(end, -1, Integer::sum);
        if (peak() > k) {
            delta.merge(start, -1, Integer::sum);
            delta.merge(end, 1, Integer::sum);
            return false;
        }
        return true;
    }

    int maxOverlap() {
        return peak();
    }

    private int peak() {
        int current = 0, best = 0;
        for (int change : delta.values()) {
            current += change;
            best = Math.max(best, current);
        }
        return best;
    }
}

public class Solution {
    /** Maximum number of intervals covering any single instant (independent sweep oracle). */
    static int sweepPeak(List<int[]> intervals) {
        TreeMap<Integer, Integer> events = new TreeMap<>();
        for (int[] interval : intervals) {
            events.merge(interval[0], 1, Integer::sum);
            events.merge(interval[1], -1, Integer::sum);
        }
        int current = 0, peak = 0;
        for (int change : events.values()) {
            current += change;
            peak = Math.max(peak, current);
        }
        return peak;
    }

    private static int seed = 1717;

    private static List<int[]> lcgBookings(int count, int span) {
        List<int[]> bookings = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            int start = seed % span;
            seed = (seed * 75 + 74) % 65537;
            bookings.add(new int[]{start, start + 1 + seed % 9});
        }
        return bookings;
    }

    private static String join(List<int[]> intervals) {
        StringBuilder builder = new StringBuilder();
        for (int[] interval : intervals) builder.append(Arrays.toString(interval));
        return builder.toString();
    }

    public static void main(String[] args) {
        MyCalendar cal = new MyCalendar();
        check(cal.book(10, 20), "I accept (10,20)");
        check(!cal.book(15, 25), "I reject overlap");
        check(cal.book(20, 30), "I accept touching (20,30)");
        check(cal.book(5, 10), "I accept touching (5,10)");
        check(!cal.book(19, 20), "I reject (19,20)");
        check(!cal.book(10, 15), "I reject (10,15)");
        check(join(cal.accepted).equals("[10, 20][20, 30][5, 10]"), "I accepted list");

        MyCalendarTwo two = new MyCalendarTwo();
        int[][] classic = {{10, 20}, {50, 60}, {10, 40}, {5, 15}, {5, 10}, {25, 55}};
        boolean[] expected2 = {true, true, true, false, true, true};
        for (int i = 0; i < classic.length; i++) {
            check(two.book(classic[i][0], classic[i][1]) == expected2[i], "II step " + i);
        }
        check(two.maxOverlap() == 2, "II peak two");
        check(!two.book(30, 45), "II reject triple at (30,45)");
        check(two.book(60, 70), "II accept (60,70)");

        MyCalendarK three = new MyCalendarK(3);
        for (int i = 0; i < classic.length; i++) {
            check(three.book(classic[i][0], classic[i][1]), "III k=3 step " + i);
        }
        check(three.maxOverlap() == 3, "III peak three");
        check(!three.book(12, 13), "III k=3 rejects fourth cover");

        MyCalendarK twoK = new MyCalendarK(2);
        boolean[] expectedK2 = {true, true, true, false, true, true};
        for (int i = 0; i < classic.length; i++) {
            check(twoK.book(classic[i][0], classic[i][1]) == expectedK2[i], "III k=2 step " + i);
        }
        check(twoK.maxOverlap() == 2, "III k=2 peak");
        check(!twoK.book(12, 13), "III k=2 rejects (12,13)");

        check(sweepPeak(new ArrayList<>()) == 0, "oracle empty");
        check(sweepPeak(Collections.singletonList(new int[]{1, 2})) == 1, "oracle one");
        check(sweepPeak(Arrays.asList(new int[]{1, 5}, new int[]{2, 6}, new int[]{3, 7})) == 3,
                "oracle triple");
        check(sweepPeak(Arrays.asList(new int[]{1, 2}, new int[]{2, 3})) == 1, "oracle touching");

        for (int k = 1; k <= 3; k++) {
            seed = 1717;
            List<int[]> bookings = lcgBookings(240, 40);
            MyCalendarK calendarK = new MyCalendarK(k);
            MyCalendarTwo twoForComparison = k == 2 ? new MyCalendarTwo() : null;
            List<int[]> oracle = new ArrayList<>();
            for (int[] booking : bookings) {
                List<int[]> candidate = new ArrayList<>(oracle);
                candidate.add(booking);
                boolean expected = sweepPeak(candidate) <= k;
                boolean actual = calendarK.book(booking[0], booking[1]);
                check(actual == expected, "random k=" + k + " " + Arrays.toString(booking));
                if (actual) oracle.add(booking);
                if (twoForComparison != null) {
                    check(twoForComparison.book(booking[0], booking[1]) == actual,
                            "II vs k=2 " + Arrays.toString(booking));
                }
                check(calendarK.maxOverlap() == sweepPeak(oracle) && calendarK.maxOverlap() <= k,
                        "peak invariant " + Arrays.toString(booking));
            }
            check(oracle.size() <= bookings.size(), "bounded accepts");
        }

        seed = 2626;
        List<int[]> bookings = lcgBookings(200, 40);
        MyCalendar one = new MyCalendar();
        MyCalendarK oneK = new MyCalendarK(1);
        List<int[]> oracle = new ArrayList<>();
        for (int[] booking : bookings) {
            List<int[]> candidate = new ArrayList<>(oracle);
            candidate.add(booking);
            boolean expected = sweepPeak(candidate) <= 1;
            check(one.book(booking[0], booking[1]) == expected, "I random " + Arrays.toString(booking));
            check(oneK.book(booking[0], booking[1]) == expected, "I via k=1 " + Arrays.toString(booking));
            if (expected) oracle.add(booking);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
