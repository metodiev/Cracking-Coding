// Task 011 - Generate first N Fibonacci numbers (iterative). (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static long[] FibSequence(int n)
    {
        var seq = new long[n];
        long a = 0, b = 1;
        for (int i = 0; i < n; i++)
        {
            seq[i] = a;
            long next = a + b;
            a = b;
            b = next;
        }
        return seq;
    }

    public static void Main()
    {
        Check(FibSequence(0).Length == 0, "n=0");
        Check(string.Join(",", FibSequence(8)) == "0,1,1,2,3,5,8,13", "n=8");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
