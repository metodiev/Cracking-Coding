// Task 128 - Implement basic memoization wrapper. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static readonly Dictionary<int, long> FibCache = new Dictionary<int, long>();
    private static int fibCalls = 0;

    public static long Fib(int n)
    {
        if (FibCache.TryGetValue(n, out var hit)) return hit;
        fibCalls++;
        long value = n <= 1 ? n : Fib(n - 1) + Fib(n - 2);
        FibCache[n] = value;
        return value;
    }

    /// <summary>Generic memoizer keyed by the argument string.</summary>
    public static Func<T, R> Memoize<T, R>(Func<T, R> fn, Dictionary<T, R> cache) where T : notnull
    {
        return arg =>
        {
            if (!cache.TryGetValue(arg, out var hit))
            {
                hit = fn(arg);
                cache[arg] = hit;
            }
            return hit;
        };
    }

    public static void Main()
    {
        Check(Fib(30) == 832040L, "fib(30)");
        Check(fibCalls == 31, "distinct subproblems only (0..30), got " + fibCalls);
        Check(FibCache[10] == 55L, "cache inspectable");

        var squares = new Dictionary<int, long>();
        var memoSquare = Memoize<int, long>(x => (long)x * x, squares);
        Check(memoSquare(9) == 81L, "square");
        Check(memoSquare(9) == 81L && squares.Count == 1, "wrapped result cached once");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
