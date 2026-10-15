// Task 009 - Implement iterative factorial. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Numerics;

public static class Solution
{
    public static BigInteger FactorialIterative(int n)
    {
        if (n < 0) throw new ArgumentException("n must be >= 0");
        var result = BigInteger.One;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }

    public static void Main()
    {
        Check(FactorialIterative(0) == 1, "0");
        Check(FactorialIterative(5) == 120, "5");
        Check(FactorialIterative(20) == BigInteger.Parse("2432902008176640000"), "20");
        Check(FactorialIterative(25) == BigInteger.Parse("15511210043330985984000000"), "25 (BigInteger)");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
