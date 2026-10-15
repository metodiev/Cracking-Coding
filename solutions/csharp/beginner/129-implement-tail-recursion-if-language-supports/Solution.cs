// Task 129 - Implement tail recursion (if language supports). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Numerics;

public static class Solution
{
    /// <summary>Tail-recursive shape (.NET has no guaranteed TCO; small n only).</summary>
    public static long FactorialTail(int n, long acc)
    {
        return n <= 1 ? acc : FactorialTail(n - 1, n * acc);
    }

    private static object FactStep(int remaining, BigInteger acc)
    {
        if (remaining <= 1) return acc;
        int next = remaining - 1;
        var nextAcc = acc * remaining;
        return (Func<object>)(() => FactStep(next, nextAcc));
    }

    /// <summary>Stack-safe: constant stack depth regardless of n.</summary>
    public static object Trampoline(Func<object> start)
    {
        object result = start;
        while (result is Func<object> thunk) result = thunk();
        return result;
    }

    public static BigInteger FactorialTrampolined(int n)
    {
        return (BigInteger)Trampoline(() => FactStep(n, BigInteger.One));
    }

    public static void Main()
    {
        Check(FactorialTail(5, 1) == 120, "tail 5");
        Check(FactorialTail(20, 1) == 2432902008176640000L, "tail 20");

        var expected = BigInteger.One;
        for (int i = 2; i <= 5000; i++) expected *= i;
        Check(FactorialTrampolined(5000) == expected, "trampoline matches 5000!");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
