// Task 047 - Find first non-repeating character in a string. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static char? FirstNonRepeating(string s)
    {
        var counts = new int[128];
        foreach (var c in s) counts[c]++;
        foreach (var c in s) if (counts[c] == 1) return c;
        return null;
    }

    public static void Main()
    {
        Check(FirstNonRepeating("aabbcde") == 'c', "c");
        Check(FirstNonRepeating("aabb") == null, "none");
        Check(FirstNonRepeating("x") == 'x', "single");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
