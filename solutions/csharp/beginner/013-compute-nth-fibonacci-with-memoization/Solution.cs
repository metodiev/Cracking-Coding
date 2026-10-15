// Task 013 - Compute nth Fibonacci with memoization. O(n). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static long FibMemo(int n)
    {
        var cache = new Dictionary<int, long> { [0] = 0, [1] = 1 };
        return FibMemo(n, cache);
    }

    private static long FibMemo(int n, Dictionary<int, long> cache)
    {
        if (cache.TryGetValue(n, out var hit)) return hit;
        var value = FibMemo(n - 1, cache) + FibMemo(n - 2, cache);
        cache[n] = value;
        return value;
    }

    public static void Main()
    {
        Check(FibMemo(50) == 12586269025L, "fib(50)");
        Check(FibMemo(0) == 0 && FibMemo(1) == 1, "bases");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
