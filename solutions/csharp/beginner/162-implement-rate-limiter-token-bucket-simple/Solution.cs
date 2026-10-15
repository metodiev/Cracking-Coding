// Task 162 - Implement rate limiter (token bucket simple). (Tier: Beginner, CTCI Ch 7)
using System;

public static class Solution
{
    private class TokenBucket
    {
        private readonly double capacity, rate;
        private double tokens;
        private long? lastMs = null;

        public TokenBucket(double capacity, double refillPerSecond)
        {
            if (capacity <= 0 || refillPerSecond <= 0)
                throw new ArgumentException("capacity and refill rate must be positive");
            this.capacity = capacity;
            this.rate = refillPerSecond;
            this.tokens = capacity;
        }

        private void Refill(long nowMs)
        {
            if (lastMs == null)
            {
                lastMs = nowMs;
                return;
            }
            long elapsed = Math.Max(0, nowMs - lastMs.Value);
            tokens = Math.Min(capacity, tokens + elapsed / 1000.0 * rate);
            lastMs = nowMs;
        }

        public bool Allow(long nowMs, double cost)
        {
            Refill(nowMs);
            if (tokens >= cost)
            {
                tokens -= cost;
                return true;
            }
            return false;
        }
    }

    public static void Main()
    {
        var bucket = new TokenBucket(3, 1.0);
        Check(bucket.Allow(0, 1) && bucket.Allow(0, 1) && bucket.Allow(0, 1), "three initial tokens");
        Check(!bucket.Allow(0, 1), "drained");
        Check(bucket.Allow(1000, 1), "one second refills one token");
        Check(!bucket.Allow(1000, 2), "only one token present");
        Check(bucket.Allow(3000, 2), "two more seconds refill two tokens");
        Check(bucket.Allow(100000, 1), "idle clamps to capacity");
        Check(!bucket.Allow(100000, 4), "cannot exceed capacity");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
