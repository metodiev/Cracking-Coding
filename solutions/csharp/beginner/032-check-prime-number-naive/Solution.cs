// Task 032 - Check prime number naive. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static bool IsPrime(int n)
    {
        if (n < 2) return false;
        for (long i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    public static void Main()
    {
        Check(!IsPrime(0) && !IsPrime(1), "below 2");
        Check(IsPrime(2) && IsPrime(97), "primes");
        Check(!IsPrime(100), "100");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
