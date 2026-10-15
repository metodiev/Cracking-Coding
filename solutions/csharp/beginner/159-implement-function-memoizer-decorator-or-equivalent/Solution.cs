// Task 159 - Implement function memoizer decorator (or equivalent). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

/// <summary>
/// Memoizer for single-argument int functions, distinct from the task 128 wrapper:
/// records hits / misses, exposes the cache size, and supports an optional capacity
/// with FIFO eviction - when full, the OLDEST inserted key leaves first and a hit
/// never changes insertion order (FIFO, deliberately not LRU; see task 160).
/// </summary>
public class Memoizer
{
    private readonly Func<int, int> fn;
    private readonly int? capacity;
    private readonly Dictionary<int, int> cache = new Dictionary<int, int>();
    private readonly Queue<int> order = new Queue<int>();

    public int Hits;
    public int Misses;

    public Memoizer(Func<int, int> fn, int? capacity = null)
    {
        if (capacity != null && capacity <= 0) throw new ArgumentException("capacity must be >= 1");
        this.fn = fn;
        this.capacity = capacity;
    }

    public int Call(int n)
    {
        if (cache.TryGetValue(n, out int cached))
        {
            Hits++;
            return cached;
        }
        Misses++;
        int value = fn(n);
        if (capacity != null && cache.Count >= capacity)
        {
            int evicted = order.Dequeue();     // FIFO: oldest insertion goes first
            cache.Remove(evicted);
        }
        cache[n] = value;
        order.Enqueue(n);
        return value;
    }

    public int Size => cache.Count;

    /// <summary>True iff n is currently cached (used to assert eviction).</summary>
    public bool Cached(int n) => cache.ContainsKey(n);
}

public static class Solution
{
    /// <summary>Steps for n to reach 1; CollatzSteps(27) == 111.</summary>
    public static int CollatzSteps(int n)
    {
        if (n < 1) throw new ArgumentException("n must be >= 1");
        int steps = 0;
        while (n != 1)
        {
            n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
            steps++;
        }
        return steps;
    }

    public static void Main()
    {
        // 1) no capacity: everything stays cached
        var memo = new Memoizer(CollatzSteps);
        Check(memo.Call(27) == 111, "collatz 27");
        Check(memo.Call(6) == 8 && memo.Call(7) == 16, "more values");
        Check(memo.Hits == 0 && memo.Misses == 3 && memo.Size == 3, "stats after three misses");
        Check(memo.Call(27) == 111 && memo.Call(6) == 8, "cached repeats");
        Check(memo.Hits == 2 && memo.Misses == 3, "two hits");

        // 2) capacity 3 with FIFO eviction
        var small = new Memoizer(CollatzSteps, 3);
        small.Call(27);
        small.Call(6);
        small.Call(7);
        Check(small.Size == 3 && small.Hits == 0 && small.Misses == 3, "filled to capacity");
        Check(small.Call(27) == 111 && small.Cached(27), "hit keeps FIFO order");
        Check(small.Call(9) == 19, "collatz 9");
        Check(small.Size == 3 && !small.Cached(27) && small.Cached(9), "oldest evicted");
        Check(small.Call(6) == 8, "6 still cached");
        Check(small.Hits == 2 && small.Misses == 4, "small stats");

        // 3) capacity 1 behaves like a one-slot cache
        var tiny = new Memoizer(n => n * n, 1);
        Check(tiny.Call(3) == 9 && tiny.Call(4) == 16, "capacity 1");
        Check(tiny.Size == 1 && !tiny.Cached(3) && tiny.Cached(4), "single slot");
        Check(tiny.Call(4) == 16 && tiny.Hits == 1 && tiny.Misses == 2, "tiny stats");

        // capacity must be positive
        bool threw = false;
        try { new Memoizer(n => n, 0); } catch (ArgumentException) { threw = true; }
        Check(threw, "capacity 0 throws");

        // stats are per-instance, not shared
        var other = new Memoizer(n => n + 1);
        Check(other.Call(1) == 2 && other.Hits == 0 && other.Misses == 1, "other instance");
        Check(memo.Hits == 2 && memo.Misses == 3, "original stats untouched");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
