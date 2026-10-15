// Task 046 - Implement substring search (naive). (Tier: Beginner, CTCI Ch 10)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static int SubstringSearch(string text, string pattern)
    {
        int n = text.Length, m = pattern.Length;
        if (m == 0) return 0;
        for (int i = 0; i + m <= n; i++)
        {
            int j = 0;
            while (j < m && text[i + j] == pattern[j]) j++;
            if (j == m) return i;
        }
        return -1;
    }

    public static void Main()
    {
        Check(SubstringSearch("hello world", "world") == 6, "found");
        Check(SubstringSearch("aaaa", "aa") == 0, "first occurrence");
        Check(SubstringSearch("abc", "xyz") == -1, "missing");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
