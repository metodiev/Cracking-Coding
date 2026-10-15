// Task 034 - Sieve of Eratosthenes to list primes up to N. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static List<int> Sieve(int n)
    {
        var outList = new List<int>();
        if (n < 2) return outList;
        var isPrime = new bool[n + 1];
        for (int i = 0; i <= n; i++) isPrime[i] = true;
        isPrime[0] = isPrime[1] = false;
        for (long p = 2; p * p <= n; p++)
            if (isPrime[p])
                for (long m = p * p; m <= n; m += p) isPrime[m] = false;
        for (int x = 2; x <= n; x++) if (isPrime[x]) outList.Add(x);
        return outList;
    }

    public static void Main()
    {
        Check(Sieve(1).Count == 0, "1");
        Check(string.Join(",", Sieve(30)) == "2,3,5,7,11,13,17,19,23,29", "30");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
