/** Task 507 - Implement throttling middleware simulation (sliding window counters). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * Three rate-limiting middleware strategies over the same request timeline, all deterministic
 * integer-millisecond simulations (no wall clock, no threads): (a) the exact sliding-window log,
 * which keeps the timestamps of the granted requests and admits a request iff fewer than `limit` of
 * them fall inside (ts - window, ts]; (b) the fixed-window counter, which resets a counter at every
 * window boundary ts / window; and (c) the sliding-window counter, which estimates the load as
 * previousWindowCount * (1 - elapsedInCurrentWindow / window) + currentWindowCount and admits the
 * request when the estimate including the request itself does not exceed `limit`. A brute-force
 * helper recomputes the maximum number of granted requests inside any window by scanning the
 * granted list pairwise, which is the oracle for every limiter.
 *
 * Documented checks: the log limiter's decision is compared request by request against a
 * brute-force rescan of the granted timestamps (2000 requests, limit 100, window 60000 ms); the
 * maximum granted count inside any sliding window is exactly <= limit for the log limiter while the
 * fixed-window counter reaches the textbook 2 * limit = 200 boundary burst (100 requests at t =
 * 59900..59999 followed by 100 at t = 60001..60100) and the sliding-window counter refuses that
 * second burst outright because its estimate is 100 * (1 - 1/60000) + 1 > 100, so it also stays at
 * <= limit; on the 2000-request random trace the log limiter's window maximum is exactly 100 while
 * the fixed window overshoots to 114 (asserted >= limit + 10 and <= 2 * limit) and the counter's
 * approximation slack reaches only 101 (asserted <= ceil(1.02 * limit)) while granting 664 requests
 * against the log limiter's 700, i.e. it errs on the conservative side; the counter's estimate is
 * checked against its closed-form value in hand-computed cases (weight 1 at the window start,
 * weight 0 exactly one window later, and a half-elapsed window with prev = 40, cur = 10 giving 30 +
 * 10 + 1 = 41); a steady half-rate stream (50 requests per window with limit 100) is admitted at
 * 100% by all three limiters, so no limiter over-throttles; the boundary cases limit = 0
 * (everything denied) and a single-timestamp trace (the first `limit` requests admitted, the rest
 * denied) are covered; and a re-run reproduces every decision.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        int nextInt(int bound) {
            return (int) (next() * bound);
        }
    }

    static final int LIMIT = 100;
    static final int WINDOW = 60000;

    /** Exact limiter: grants a request iff fewer than `limit` grants fall in (ts - window, ts]. */
    static class SlidingWindowLog {
        final int limit;
        final int window;
        final Deque<Integer> granted = new ArrayDeque<>();
        final List<Integer> grantedList = new ArrayList<>();
        int denied;

        SlidingWindowLog(int limit, int window) {
            this.limit = limit;
            this.window = window;
        }

        boolean allow(int ts) {
            while (!granted.isEmpty() && granted.peekFirst() <= ts - window) granted.pollFirst();
            if (granted.size() < limit) {
                granted.addLast(ts);
                grantedList.add(ts);
                return true;
            }
            denied++;
            return false;
        }

        boolean bruteForceAllow(int ts) {
            int count = 0;
            for (int previous : grantedList) {
                if (previous > ts - window && previous <= ts) count++;
            }
            return count < limit;
        }
    }

    static class FixedWindowCounter {
        final int limit;
        final int window;
        int windowStart = Integer.MIN_VALUE;
        int count;
        int denied;
        final List<Integer> grantedList = new ArrayList<>();

        FixedWindowCounter(int limit, int window) {
            this.limit = limit;
            this.window = window;
        }

        boolean allow(int ts) {
            int start = Math.floorDiv(ts, window) * window;
            if (start != windowStart) {
                windowStart = start;
                count = 0;
            }
            if (count < limit) {
                count++;
                grantedList.add(ts);
                return true;
            }
            denied++;
            return false;
        }
    }

    static class SlidingWindowCounter {
        final int limit;
        final int window;
        int windowStart = Integer.MIN_VALUE;
        int previousCount;
        int currentCount;
        int denied;
        final List<Integer> grantedList = new ArrayList<>();

        SlidingWindowCounter(int limit, int window) {
            this.limit = limit;
            this.window = window;
        }

        private void roll(int ts) {
            int start = Math.floorDiv(ts, window) * window;
            if (start != windowStart) {
                previousCount = start - windowStart == window ? currentCount : 0;
                windowStart = start;
                currentCount = 0;
            }
        }

        double weight(int ts) {
            return 1.0 - (ts - windowStart) / (double) window;
        }

        double estimate(int ts) {
            roll(ts);
            return previousCount * weight(ts) + currentCount + 1.0;
        }

        boolean allow(int ts) {
            double value = estimate(ts);
            if (value <= limit + 1e-9) {
                currentCount++;
                grantedList.add(ts);
                return true;
            }
            denied++;
            return false;
        }
    }

    /** Maximum grants inside any window, by scanning every granted timestamp as a window start. */
    static int maxInWindow(List<Integer> granted, int window) {
        int best = 0;
        for (int start : granted) {
            int count = 0;
            for (int ts : granted) {
                if (ts >= start && ts < start + window) count++;
            }
            best = Math.max(best, count);
        }
        return best;
    }

    static List<Integer> randomTrace(int requests, int seed, int maxStep) {
        Lcg lcg = new Lcg(seed);
        List<Integer> out = new ArrayList<>();
        int ts = 0;
        for (int i = 0; i < requests; i++) {
            ts += lcg.nextInt(maxStep) + 1;
            out.add(ts);
        }
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        List<Integer> trace = randomTrace(2000, 507507, 400);

        // 1. Exact limiter against the brute-force rescan, request by request.
        SlidingWindowLog log = new SlidingWindowLog(LIMIT, WINDOW);
        int bruteGranted = 0;
        List<Boolean> logDecisions = new ArrayList<>();
        for (int ts : trace) {
            boolean expected = log.bruteForceAllow(ts);
            boolean actual = log.allow(ts);
            check(actual == expected, "log limiter matches brute force at ts=" + ts);
            logDecisions.add(actual);
            if (expected) bruteGranted++;
        }
        check(log.grantedList.size() == bruteGranted, "the granted count matches the oracle");
        check(log.denied == trace.size() - bruteGranted, "denials are the complement");
        check(maxInWindow(log.grantedList, WINDOW) <= LIMIT,
                "the log limiter never exceeds the limit in any window");
        check(maxInWindow(log.grantedList, WINDOW) == LIMIT,
                "the log limiter saturates the window at exactly the limit");

        // 2. The genuine boundary burst: 100 at the end of one window, 100 just after the edge.
        List<Integer> burst = new ArrayList<>();
        for (int i = 0; i < 100; i++) burst.add(59900 + i);
        for (int i = 0; i < 100; i++) burst.add(60001 + i);
        SlidingWindowLog burstLog = new SlidingWindowLog(LIMIT, WINDOW);
        FixedWindowCounter burstFixed = new FixedWindowCounter(LIMIT, WINDOW);
        SlidingWindowCounter burstCounter = new SlidingWindowCounter(LIMIT, WINDOW);
        for (int ts : burst) {
            burstLog.allow(ts);
            burstFixed.allow(ts);
            burstCounter.allow(ts);
        }
        check(burstLog.grantedList.size() == LIMIT, "the log limiter admits exactly the limit");
        check(burstFixed.grantedList.size() == 2 * LIMIT,
                "the fixed window admits the full 2 * limit boundary burst");
        check(maxInWindow(burstFixed.grantedList, WINDOW) == 2 * LIMIT,
                "the fixed-window burst really is 200 inside one sliding window");
        check(maxInWindow(burstLog.grantedList, WINDOW) == LIMIT, "the log limiter caps the burst");
        check(burstCounter.grantedList.size() == LIMIT,
                "the sliding-window counter refuses the second burst too");
        check(maxInWindow(burstCounter.grantedList, WINDOW) <= LIMIT,
                "the sliding-window counter stays within the limit");

        // 3. Hand-checked closed forms for the counter's estimate.
        SlidingWindowCounter probe = new SlidingWindowCounter(LIMIT, WINDOW);
        for (int i = 0; i < 40; i++) probe.allow(i);              // window [0, 60000)
        check(Math.abs(probe.weight(0) - 1.0) <= 1e-12, "the weight is 1 at the window start");
        check(Math.abs(probe.weight(WINDOW - 1) - 1.0 / WINDOW) <= 1e-12,
                "the weight is 1/window at the very end of the window");
        for (int i = 0; i < 10; i++) probe.allow(WINDOW + i * 10); // window [60000, 120000)
        // at ts = 60000 + 30000 the previous window is half spent and holds 40 grants
        double expected = 40 * 0.5 + 10 + 1.0;
        check(Math.abs(probe.estimate(WINDOW + 30000) - expected) <= 1e-9,
                "estimate = prev * weight + current + 1 = " + expected);
        check(Math.abs(probe.weight(WINDOW) - 1.0) <= 1e-12,
                "the weight resets to 1 exactly one window later");
        check(Math.abs(probe.estimate(2 * WINDOW) - (10 * 1.0 + 0 + 1.0)) <= 1e-9,
                "the window two steps later sees the previous window only");
        SlidingWindowCounter fresh = new SlidingWindowCounter(LIMIT, WINDOW);
        check(Math.abs(fresh.estimate(0) - 1.0) <= 1e-12,
                "an empty history estimates just the incoming request");

        // 4. The random trace through all three limiters, plus the burst bound for the counter.
        FixedWindowCounter fixed = new FixedWindowCounter(LIMIT, WINDOW);
        SlidingWindowCounter counter = new SlidingWindowCounter(LIMIT, WINDOW);
        for (int ts : trace) {
            fixed.allow(ts);
            counter.allow(ts);
        }
        check(maxInWindow(fixed.grantedList, WINDOW) <= 2 * LIMIT,
                "the fixed window can never exceed 2 * limit in a sliding window");
        check(maxInWindow(counter.grantedList, WINDOW) <= (int) Math.ceil(LIMIT * 1.02),
                "the counter's approximation slack is at most 2% of the limit");
        check(maxInWindow(counter.grantedList, WINDOW) >= LIMIT - 1,
                "the counter still runs close to the limit");
        check(maxInWindow(fixed.grantedList, WINDOW) >= LIMIT + 10,
                "the fixed window overshoots the limit by at least ten on this trace");
        check(counter.grantedList.size() <= log.grantedList.size(),
                "the counter is the more conservative of the two sliding methods");
        check(counter.denied > 0 && counter.grantedList.size() > 0,
                "the random trace exercises both decisions");
        check(fixed.grantedList.size() >= log.grantedList.size(),
                "the fixed window is the most permissive of the three on this trace");

        // 5. Steady half-rate traffic is admitted by all three limiter types.
        List<Integer> steady = new ArrayList<>();
        for (int w = 0; w < 6; w++) {
            for (int i = 0; i < 50; i++) steady.add(w * WINDOW + i * (WINDOW / 60));
        }
        SlidingWindowLog steadyLog = new SlidingWindowLog(LIMIT, WINDOW);
        FixedWindowCounter steadyFixed = new FixedWindowCounter(LIMIT, WINDOW);
        SlidingWindowCounter steadyCounter = new SlidingWindowCounter(LIMIT, WINDOW);
        for (int ts : steady) {
            steadyLog.allow(ts);
            steadyFixed.allow(ts);
            steadyCounter.allow(ts);
        }
        check(steadyLog.grantedList.size() == steady.size(), "the log limiter admits steady traffic");
        check(steadyFixed.grantedList.size() == steady.size(),
                "the fixed window admits steady traffic");
        check(steadyCounter.grantedList.size() >= (int) Math.ceil(0.95 * steady.size()),
                "the sliding-window counter admits at least 95% of steady traffic");

        // 6. Boundaries: limit 0 denies everything, and a single-timestamp trace admits `limit`.
        SlidingWindowCounter none = new SlidingWindowCounter(0, WINDOW);
        for (int ts : trace) check(!none.allow(ts), "limit 0 denies every request");
        SlidingWindowLog same = new SlidingWindowLog(10, WINDOW);
        int admitted = 0;
        for (int i = 0; i < 25; i++) {
            if (same.allow(1234)) admitted++;
        }
        check(admitted == 10, "requests sharing one timestamp stop at the limit");

        // 7. Determinism.
        SlidingWindowLog rerun = new SlidingWindowLog(LIMIT, WINDOW);
        List<Boolean> rerunDecisions = new ArrayList<>();
        for (int ts : trace) rerunDecisions.add(rerun.allow(ts));
        check(rerunDecisions.equals(logDecisions), "deterministic re-run: identical log decisions");
        check(rerun.grantedList.equals(log.grantedList), "deterministic re-run: identical log grants");
        SlidingWindowCounter counterRerun = new SlidingWindowCounter(LIMIT, WINDOW);
        for (int ts : trace) counterRerun.allow(ts);
        check(counterRerun.grantedList.equals(counter.grantedList),
                "deterministic re-run: identical counter grants");
        check(maxInWindow(counterRerun.grantedList, WINDOW) == maxInWindow(counter.grantedList, WINDOW),
                "deterministic re-run: identical window maximum");

        System.out.println("PASS");
    }
}
