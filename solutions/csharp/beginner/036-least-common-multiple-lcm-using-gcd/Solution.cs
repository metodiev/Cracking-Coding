// Task 036 - Least common multiple (LCM) using GCD. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static long Lcm(long a, long b)
    {
        if (a == 0 || b == 0) return 0;
        long g = Gcd(Math.Abs(a), Math.Abs(b));
        return Math.Abs(a / g * b);
    }

    private static long Gcd(long a, long b)
    {
        while (b != 0) { long t = b; b = a % b; a = t; }
        return a;
    }

    public static void Main()
    {
        Check(Lcm(4, 6) == 12, "4/6");
        Check(Lcm(21, 6) == 42, "21/6");
        Check(Lcm(0, 5) == 0, "zero");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
