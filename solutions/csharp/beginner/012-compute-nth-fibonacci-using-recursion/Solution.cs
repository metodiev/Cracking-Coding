// Task 012 - Compute nth Fibonacci using recursion. O(2^n), intentionally naive. (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static long Fib(int n)
    {
        return n <= 1 ? n : Fib(n - 1) + Fib(n - 2);
    }

    public static void Main()
    {
        Check(Fib(10) == 55, "fib(10)");
        Check(Fib(20) == 6765, "fib(20)");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
