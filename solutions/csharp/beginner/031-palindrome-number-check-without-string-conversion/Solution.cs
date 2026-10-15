// Task 031 - Palindrome number check (without string conversion). (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static bool IsPalindromeNumber(int n)
    {
        if (n < 0) return false;
        long reversed = 0;
        int original = n;
        while (n != 0) { reversed = reversed * 10 + n % 10; n /= 10; }
        return original == reversed;
    }

    public static void Main()
    {
        Check(IsPalindromeNumber(121), "121");
        Check(!IsPalindromeNumber(-121), "-121");
        Check(IsPalindromeNumber(0), "0");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
