// Task 094 - Find longest common prefix in array of strings. (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static string LongestCommonPrefix(string[] strs)
    {
        if (strs.Length == 0) return "";
        string first = strs[0];
        for (int i = 0; i < first.Length; i++)
        {
            char ch = first[i];
            for (int k = 1; k < strs.Length; k++)
            {
                string s = strs[k];
                if (i >= s.Length || s[i] != ch) return first.Substring(0, i);
            }
        }
        return first;
    }

    public static void Main()
    {
        Check(LongestCommonPrefix(new[] { "flower", "flow", "flight" }) == "fl", "fl");
        Check(LongestCommonPrefix(new[] { "dog", "racecar", "car" }) == "", "none");
        Check(LongestCommonPrefix(new[] { "apple" }) == "apple", "single");
        Check(LongestCommonPrefix(new string[0]) == "", "empty array");
        Check(LongestCommonPrefix(new[] { "" }) == "", "empty string entry");
        Check(LongestCommonPrefix(new[] { "", "a" }) == "", "empty first");
        Check(LongestCommonPrefix(new[] { "a", "ab" }) == "a", "prefix is whole first");
        Check(LongestCommonPrefix(new[] { "abc", "abc", "abc" }) == "abc", "identical");
        Check(LongestCommonPrefix(new[] { "prefix", "pref" }) == "pref", "shorter second");
        Check(LongestCommonPrefix(new[] { "ab", "ac" }) == "a", "shared single char");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
