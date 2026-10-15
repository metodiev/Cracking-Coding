// Task 044 - Trim leading and trailing spaces from string. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static string Trim(string s)
    {
        int lo = 0, hi = s.Length;
        while (lo < hi && s[lo] == ' ') lo++;
        while (hi > lo && s[hi - 1] == ' ') hi--;
        return s.Substring(lo, hi - lo);
    }

    public static void Main()
    {
        Check(Trim("  hello  ") == "hello", "both sides");
        Check(Trim("   ") == "", "all spaces");
        Check(Trim("  a b ") == "a b", "inner space kept");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
