/** Task 333 - Design hit counter (sliding time window). (Tier: Intermediate, CTCI Ch 7) */
import java.util.*;

class HitCounter {
    private final List<Integer> times = new ArrayList<>();
    static final int WINDOW = 300;

    /** Records a hit; timestamps must be non-decreasing. */
    void hit(int timestamp) {
        if (!times.isEmpty()) {
            if (timestamp < times.get(times.size() - 1)) {
                throw new IllegalArgumentException("timestamps must not go backwards");
            }
        }
        times.add(timestamp);
    }

    /** Number of hits h with timestamp - 299 <= h <= timestamp. */
    int getHits(int timestamp) {
        return firstGreater(timestamp) - firstAtLeast(timestamp - (WINDOW - 1));
    }

    private int firstAtLeast(int timestamp) {
        int lo = 0, hi = times.size();
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (times.get(mid) < timestamp) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private int firstGreater(int timestamp) {
        int lo = 0, hi = times.size();
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (times.get(mid) <= timestamp) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public class Solution {
    /** Oracle: linear scan of every recorded timestamp. */
    static int naiveHits(List<Integer> times, int timestamp) {
        int count = 0;
        for (int time : times) {
            if (timestamp - (HitCounter.WINDOW - 1) <= time && time <= timestamp) count++;
        }
        return count;
    }

    private static int seed = 2027;

    public static void main(String[] args) {
        // Hand-checked: hits at 1, 2, 3 -> get(4) = 3; windows slide away one at a time.
        HitCounter counter = new HitCounter();
        for (int t : new int[]{1, 2, 3}) counter.hit(t);
        check(counter.getHits(1) == 1, "get(1)");
        check(counter.getHits(2) == 2, "get(2)");
        check(counter.getHits(3) == 3, "get(3)");
        check(counter.getHits(4) == 3, "get(4) sees all three");
        check(counter.getHits(299) == 3, "get(299)");
        check(counter.getHits(300) == 3, "get(300): 300-299 = 1 still counts hit 1");
        check(counter.getHits(301) == 2, "get(301): hit 1 left the window");
        check(counter.getHits(302) == 1, "get(302)");
        check(counter.getHits(303) == 0, "get(303)");

        // A later hit at 300 joins the window: get(300) = 4, get(301) = 3, get(302) = 2.
        counter.hit(300);
        check(counter.getHits(300) == 4, "get(300) after hit(300)");
        check(counter.getHits(301) == 3, "get(301) after hit(300)");
        check(counter.getHits(302) == 2, "get(302) after hit(300)");
        check(counter.getHits(599) == 1, "get(599)");
        check(counter.getHits(600) == 0, "get(600)");

        // LeetCode-flavoured sequence: hits at 1, 2, 3 and (again) at 300.
        HitCounter leet = new HitCounter();
        for (int t : new int[]{1, 2, 3, 300}) leet.hit(t);
        check(leet.getHits(3) == 3, "leet get(3)");
        check(leet.getHits(300) == 4, "leet get(300)");
        check(leet.getHits(301) == 3, "leet get(301)");
        check(leet.getHits(2) == 2, "query before newest hit");
        check(leet.getHits(0) == 0, "query before everything");

        // Same-timestamp bursts.
        HitCounter burst = new HitCounter();
        for (int i = 0; i < 5; i++) burst.hit(7);
        check(burst.getHits(7) == 5, "burst get(7)");
        check(burst.getHits(306) == 5, "burst get(306)");
        check(burst.getHits(307) == 0, "burst get(307)");

        // LCG event stream against the naive scan.
        HitCounter stream = new HitCounter();
        List<Integer> history = new ArrayList<>();
        int now = 0;
        for (int step = 0; step < 4000; step++) {
            seed = (seed * 75 + 74) % 65537;
            now += seed % 5;
            seed = (seed * 75 + 74) % 65537;
            if (seed % 4 == 0) {
                seed = (seed * 75 + 74) % 65537;
                int timestamp = now + seed % 3;
                stream.hit(timestamp);
                history.add(timestamp);
                now = timestamp;
            } else {
                check(stream.getHits(now) == naiveHits(history, now), "random get " + step);
            }
            if (step % 97 == 0) {
                check(stream.getHits(now) == naiveHits(history, now), "periodic get " + step);
            }
        }
        check(stream.getHits(now) == naiveHits(history, now), "final get");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
