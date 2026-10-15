// Task 015 - Check if two strings are anagrams. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static bool AreAnagrams(string a, string b)
    {
        if (a.Length != b.Length) return false;
        var count = new int[128];
        for (int i = 0; i < a.Length; i++)
        {
            count[a[i]]++;
            count[b[i]]--;
        }
        foreach (var c in count) if (c != 0) return false;
        return true;
    }

    public static void Main()
    {
        Check(AreAnagrams("listen", "silent"), "listen/silent");
        Check(!AreAnagrams("hello", "world"), "hello/world");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
