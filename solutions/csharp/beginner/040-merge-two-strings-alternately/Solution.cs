// Task 040 - Merge two strings alternately. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static string MergeAlternately(string a, string b)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < Math.Max(a.Length, b.Length); i++)
        {
            if (i < a.Length) sb.Append(a[i]);
            if (i < b.Length) sb.Append(b[i]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        Check(MergeAlternately("abc", "123") == "a1b2c3", "equal length");
        Check(MergeAlternately("ab", "1234") == "a1b234", "b longer");
        Check(MergeAlternately("", "xyz") == "xyz", "a empty");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
