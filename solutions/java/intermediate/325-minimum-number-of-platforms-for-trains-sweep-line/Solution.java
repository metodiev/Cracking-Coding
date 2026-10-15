/** Task 325 - Minimum number of platforms for trains (sweep-line). (Tier: Intermediate, CTCI Beyond — Software Engineering & Tooling) */
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
    /** Two-pointer sweep; a train leaving at t frees its platform before any arrival at t. */
    static int minPlatforms(int[] arrivals, int[] departures) {
        int[] arr = arrivals.clone();
        int[] dep = departures.clone();
        Arrays.sort(arr);
        Arrays.sort(dep);
        int i = 0, j = 0, needed = 0, best = 0;
        while (i < arr.length) {
            if (arr[i] < dep[j]) {
                needed++;
                best = Math.max(best, needed);
                i++;
            } else {
                needed--;
                j++;
            }
        }
        return best;
    }

    /** Independent oracle: count active trains at every minute of the schedule. */
    static int minPlatformsPerMinute(int[] arrivals, int[] departures) {
        if (arrivals.length == 0) return 0;
        int lo = Integer.MAX_VALUE, hi = Integer.MIN_VALUE;
        for (int arrival : arrivals) lo = Math.min(lo, arrival);
        for (int departure : departures) hi = Math.max(hi, departure);
        int best = 0;
        for (int minute = lo; minute <= hi; minute++) {
            int active = 0;
            for (int k = 0; k < arrivals.length; k++) {
                if (arrivals[k] <= minute && minute < departures[k]) active++;
            }
            best = Math.max(best, active);
        }
        return best;
    }

    static int[][] lcgSchedules(int count, Lcg lcg) {
        int[] arrivals = new int[count];
        int[] departures = new int[count];
        for (int i = 0; i < count; i++) {
            int arrival = lcg.next(60);
            arrivals[i] = arrival;
            departures[i] = arrival + 1 + lcg.next(20);
        }
        return new int[][]{arrivals, departures};
    }

    public static void main(String[] args) {
        int[] arrivals = {900, 940, 950, 1100, 1500, 1800};
        int[] departures = {910, 1200, 1120, 1130, 1900, 2000};
        check(minPlatforms(arrivals, departures) == 3, "classic trains need 3 platforms");
        check(minPlatformsPerMinute(arrivals, departures) == 3, "classic per-minute oracle");

        check(minPlatforms(new int[]{1, 2}, new int[]{2, 3}) == 1, "release before acquire");
        check(minPlatformsPerMinute(new int[]{1, 2}, new int[]{2, 3}) == 1, "release per-minute");
        check(minPlatforms(new int[]{1, 1, 1}, new int[]{2, 2, 2}) == 3, "three simultaneous trains");
        check(minPlatforms(new int[]{5}, new int[]{9}) == 1, "single train");
        check(minPlatforms(new int[0], new int[0]) == 0, "no trains");

        Lcg lcg = new Lcg(31415);
        for (int trial = 0; trial < 300; trial++) {
            int count = 1 + lcg.next(12);
            int[][] schedule = lcgSchedules(count, lcg);
            int expected = minPlatformsPerMinute(schedule[0], schedule[1]);
            check(minPlatforms(schedule[0], schedule[1]) == expected, "random per-minute cross-check " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
