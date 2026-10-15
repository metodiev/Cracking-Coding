// Task 038 - Implement binary exponentiation (fast power). (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static long PowerFast(long baseV, int exp)
    {
        long result = 1;
        while (exp > 0)
        {
            if ((exp & 1) == 1) result *= baseV;
            baseV *= baseV;
            exp >>= 1;
        }
        return result;
    }

    public static void Main()
    {
        Check(PowerFast(3, 13) == 1594323L, "3^13");
        Check(PowerFast(2, 10) == 1024L, "2^10");
        Check(PowerFast(7, 0) == 1L, "7^0");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
