// Task 033 - List primes up to N using trial division. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static List<int> PrimesUpto(int n)
    {
        var outList = new List<int>();
        for (int x = 2; x <= n; x++) if (IsPrime(x)) outList.Add(x);
        return outList;
    }

    private static bool IsPrime(int n)
    {
        for (long i = 2; i * i <= n; i++) if (n % i == 0) return false;
        return true;
    }

    public static void Main()
    {
        Check(PrimesUpto(1).Count == 0, "<=1");
        Check(string.Join(",", PrimesUpto(10)) == "2,3,5,7", "10");
        Check(string.Join(",", PrimesUpto(20)) == "2,3,5,7,11,13,17,19", "20");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
