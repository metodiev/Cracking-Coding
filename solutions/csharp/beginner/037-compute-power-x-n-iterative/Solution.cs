// Task 037 - Compute power(x, n) iterative. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static double Power(double x, int n)
    {
        int exp = Math.Abs(n);
        double result = 1.0;
        for (int i = 0; i < exp; i++) result *= x;
        return n >= 0 ? result : 1.0 / result;
    }

    public static void Main()
    {
        Check(Power(2, 10) == 1024.0, "2^10");
        Check(Power(2, -2) == 0.25, "2^-2");
        Check(Power(5, 0) == 1.0, "x^0");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
