/** Task 162 - Implement rate limiter (token bucket simple). (Tier: Beginner, CTCI Ch 7) */
public class Solution {
    static class TokenBucket {
        private final double capacity, rate;
        private double tokens;
        private Long lastMs = null;

        TokenBucket(double capacity, double refillPerSecond) {
            if (capacity <= 0 || refillPerSecond <= 0)
                throw new IllegalArgumentException("capacity and refill rate must be positive");
            this.capacity = capacity;
            this.rate = refillPerSecond;
            this.tokens = capacity;
        }

        private void refill(long nowMs) {
            if (lastMs == null) {
                lastMs = nowMs;
                return;
            }
            long elapsed = Math.max(0, nowMs - lastMs);
            tokens = Math.min(capacity, tokens + elapsed / 1000.0 * rate);
            lastMs = nowMs;
        }

        boolean allow(long nowMs, double cost) {
            refill(nowMs);
            if (tokens >= cost) {
                tokens -= cost;
                return true;
            }
            return false;
        }
    }

    public static void main(String[] args) {
        TokenBucket bucket = new TokenBucket(3, 1.0);
        check(bucket.allow(0, 1) && bucket.allow(0, 1) && bucket.allow(0, 1), "three initial tokens");
        check(!bucket.allow(0, 1), "drained");
        check(bucket.allow(1000, 1), "one second refills one token");
        check(!bucket.allow(1000, 2), "only one token present");
        check(bucket.allow(3000, 2), "two more seconds refill two tokens");
        check(bucket.allow(100_000, 1), "idle clamps to capacity");
        check(!bucket.allow(100_000, 4), "cannot exceed capacity");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
