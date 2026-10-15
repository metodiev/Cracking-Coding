// Task 043 - Count words in a string. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static int CountWords(string s)
    {
        int count = 0;
        bool inWord = false;
        foreach (var c in s)
        {
            if (char.IsWhiteSpace(c)) inWord = false;
            else if (!inWord) { inWord = true; count++; }
        }
        return count;
    }

    public static void Main()
    {
        Check(CountWords("the sky is blue") == 4, "sentence");
        Check(CountWords("   ") == 0, "spaces only");
        Check(CountWords("one") == 1, "single");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
