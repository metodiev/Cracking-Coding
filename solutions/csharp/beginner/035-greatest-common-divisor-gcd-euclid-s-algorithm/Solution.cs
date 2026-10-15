// Task 035 - Greatest common divisor (GCD) Euclid's algorithm. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int Gcd(int a, int b)
    {
        a = Math.Abs(a); b = Math.Abs(b);
        while (b != 0) { int t = b; b = a % b; a = t; }
        return a;
    }

    public static void Main()
    {
        Check(Gcd(48, 18) == 6, "48/18");
        Check(Gcd(17, 5) == 1, "17/5");
        Check(Gcd(0, 9) == 9, "0/9");
        Check(Gcd(-24, 36) == 12, "negatives");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
