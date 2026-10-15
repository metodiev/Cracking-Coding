// Task 194 - Implement caching of expensive function results with TTL. (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class TTLCache<T>
    {
        private class Entry
        {
            public T Value { get; }
            public long ExpiresAtMs { get; }

            public Entry(T value, long expiresAtMs)
            {
                Value = value;
                ExpiresAtMs = expiresAtMs;
            }
        }

        private readonly Dictionary<string, Entry> entries = new Dictionary<string, Entry>();

        public int Hits { get; private set; }
        public int Misses { get; private set; }
        public int Evictions { get; private set; }

        public void Put(string key, T value, long nowMs, long ttlMs)
        {
            if (ttlMs <= 0) throw new ArgumentException("ttl_ms must be positive");
            entries[key] = new Entry(value, nowMs + ttlMs);
        }

        // C# has no nullable return for unconstrained T? (it would be default(T), not null),
        // so the lookup returns Found/Value explicitly.
        public (bool Found, T Value) Get(string key, long nowMs)
        {
            if (!entries.TryGetValue(key, out var entry))
            {
                Misses++;
                return (false, default!);
            }
            if (nowMs >= entry.ExpiresAtMs)                 // exactly at TTL -> expired
            {
                entries.Remove(key);
                Evictions++;
                Misses++;
                return (false, default!);
            }
            Hits++;
            return (true, entry.Value);
        }

        public T Memoize(Func<string, T> fn, string key, long nowMs, long ttlMs)
        {
            var (found, cached) = Get(key, nowMs);
            if (found) return cached;
            T value = fn(key);
            Put(key, value, nowMs, ttlMs);
            return value;
        }

        public int Size()
        {
            return entries.Count;
        }
    }

    public static void Main()
    {
        var cache = new TTLCache<int>();
        cache.Put("a", 10, 1000, 100);                   // valid while now < 1100
        Check(Is(cache.Get("a", 1099), 10), "valid before TTL boundary");
        Check(!cache.Get("a", 1100).Found, "exactly at TTL -> expired");
        Check(cache.Hits == 1 && cache.Misses == 1 && cache.Evictions == 1, "stats after boundary access");
        Check(cache.Size() == 0, "expired entry removed lazily");

        cache.Put("b", 20, 0, 50);
        Check(Is(cache.Get("b", 25), 20), "fresh entry");
        cache.Put("b", 21, 30, 50);                      // refresh -> expires at 80
        Check(Is(cache.Get("b", 79), 21), "refreshed value");
        Check(!cache.Get("b", 80).Found, "refresh moved the expiry");

        var missing = new TTLCache<int>();
        Check(!missing.Get("nope", 0).Found, "missing key");
        Check(missing.Hits == 0 && missing.Misses == 1, "missing key counted as a miss");
        CheckThrows(() => missing.Put("x", 1, 0, 0), "non-positive ttl rejected");

        var calls = new List<string>();
        Func<string, int> slow = n =>
        {
            calls.Add(n);
            return 49;
        };
        var memo = new TTLCache<int>();
        Check(memo.Memoize(slow, "7", 0, 100) == 49, "first call computes");
        Check(memo.Memoize(slow, "7", 99, 100) == 49, "second call hits the cache");
        Check(calls.Count == 1, "expensive function called once");
        Check(memo.Memoize(slow, "7", 100, 100) == 49, "expired value recomputed");
        Check(calls.Count == 2, "expensive function called again after expiry");
        Check(memo.Hits == 1 && memo.Misses == 2 && memo.Evictions == 1, "memoize stats");
        Console.WriteLine("PASS");
    }

    private static bool Is((bool Found, int Value) result, int expected)
    {
        return result.Found && result.Value == expected;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected error - " + label);
    }
}
