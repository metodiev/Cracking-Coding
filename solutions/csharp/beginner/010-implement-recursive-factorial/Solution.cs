// Task 010 - Implement recursive factorial. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Numerics;

public static class Solution
{
    public static BigInteger FactorialRecursive(int n)
    {
        if (n < 0) throw new ArgumentException("n must be >= 0");
        return n <= 1 ? BigInteger.One : n * FactorialRecursive(n - 1);
    }

    public static void Main()
    {
        Check(FactorialRecursive(0) == 1, "0");
        Check(FactorialRecursive(5) == 120, "5");
        Check(FactorialRecursive(20) == BigInteger.Parse("2432902008176640000"), "20");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
