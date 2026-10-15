/** Task 327 - Car pooling capacity check (difference array / sweep). (Tier: Intermediate, CTCI Ch 1) */
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
    /** Difference-array sweep; true when the load never exceeds the capacity. */
    static boolean carPooling(int[][] trips, int capacity) {
        TreeMap<Integer, Integer> delta = new TreeMap<>();
        for (int[] trip : trips) {
            delta.merge(trip[1], trip[0], Integer::sum);   // pick up at start
            delta.merge(trip[2], -trip[0], Integer::sum);  // drop off at end
        }
        int load = 0;
        for (int change : delta.values()) {
            load += change;
            if (load > capacity) return false;
        }
        return true;
    }

    /** Independent oracle for small coordinates: materialise the load at every stop. */
    static boolean carPoolingBruteforce(int[][] trips, int capacity) {
        if (trips.length == 0) return capacity >= 0;
        int lo = Integer.MAX_VALUE, hi = Integer.MIN_VALUE;
        for (int[] trip : trips) {
            lo = Math.min(lo, trip[1]);
            hi = Math.max(hi, trip[2]);
        }
        int[] loads = new int[hi - lo];
        for (int[] trip : trips) {
            for (int stop = trip[1]; stop < trip[2]; stop++) loads[stop - lo] += trip[0];
        }
        int peak = 0;
        for (int load : loads) peak = Math.max(peak, load);
        return peak <= capacity;
    }

    static int[][] lcgTrips(int count, Lcg lcg) {
        int[][] trips = new int[count][];
        for (int i = 0; i < count; i++) {
            int passengers = 1 + lcg.next(6);
            int start = lcg.next(25);
            trips[i] = new int[]{passengers, start, start + 1 + lcg.next(8)};
        }
        return trips;
    }

    public static void main(String[] args) {
        check(!carPooling(new int[][]{{2, 1, 5}, {3, 3, 7}}, 4), "capacity 4 is too small");
        check(carPooling(new int[][]{{2, 1, 5}, {3, 3, 7}}, 5), "capacity 5 fits");
        check(carPooling(new int[][]{{2, 1, 5}, {3, 5, 7}}, 3), "drop before pick-up at stop 5");
        check(!carPooling(new int[][]{{2, 1, 5}, {3, 5, 7}}, 2), "capacity 2 is too small");
        check(carPooling(new int[0][], 0), "no trips fit zero capacity");
        check(!carPooling(new int[][]{{1, 0, 1}}, 0), "single passenger over capacity");

        check(!carPoolingBruteforce(new int[][]{{2, 1, 5}, {3, 3, 7}}, 4), "bruteforce false");
        check(carPoolingBruteforce(new int[][]{{2, 1, 5}, {3, 5, 7}}, 3), "bruteforce true");

        Lcg lcg = new Lcg(1618);
        for (int trial = 0; trial < 300; trial++) {
            int count = 1 + lcg.next(8);
            int[][] trips = lcgTrips(count, lcg);
            int capacity = lcg.next(21);
            boolean expected = carPoolingBruteforce(trips, capacity);
            check(carPooling(trips, capacity) == expected, "random bruteforce cross-check " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
