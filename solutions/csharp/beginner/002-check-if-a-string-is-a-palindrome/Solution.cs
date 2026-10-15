// Task 002 - Check if a string is a palindrome. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static bool IsPalindrome(string s)
    {
        for (int i = 0, j = s.Length - 1; i < j; i++, j--)
            if (s[i] != s[j]) return false;
        return true;
    }

    public static void Main()
    {
        Check(IsPalindrome("racecar"), "racecar");
        Check(IsPalindrome(""), "empty");
        Check(!IsPalindrome("hello"), "hello");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
