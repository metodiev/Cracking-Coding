// Task 001 - Reverse a string. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static string ReverseString(string s)
    {
        var chars = s.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    public static void Main()
    {
        Check(ReverseString("hello") == "olleh", "hello");
        Check(ReverseString("") == "", "empty");
        Check(ReverseString("ab") == "ba", "even");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
