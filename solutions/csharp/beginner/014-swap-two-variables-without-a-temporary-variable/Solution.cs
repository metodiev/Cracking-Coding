// Task 014 - Swap two variables without a temporary variable. (Tier: Beginner, CTCI Ch 5)
using System;

public static class Solution
{
    public static int[] SwapArithmetic(int a, int b)
    {
        a = a + b;
        b = a - b;
        a = a - b;
        return new[] { a, b };
    }

    public static int[] SwapXor(int a, int b)
    {
        a ^= b;
        b ^= a;
        a ^= b;
        return new[] { a, b };
    }

    public static void Main()
    {
        Check(string.Join(",", SwapArithmetic(5, 3)) == "3,5", "arithmetic");
        Check(string.Join(",", SwapXor(5, 3)) == "3,5", "xor");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
