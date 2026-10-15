// Task 170 - Simple palindrome partitioning backtracking (list partitions). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static bool IsPalindrome(string s)
    {
        for (int i = 0, j = s.Length - 1; i < j; i++, j--)
            if (s[i] != s[j]) return false;
        return true;
    }

    public static List<List<string>> PalindromePartitions(string s)
    {
        var outList = new List<List<string>>();
        Backtrack(s, 0, new List<string>(), outList);
        return outList;
    }

    private static void Backtrack(string s, int start, List<string> current, List<List<string>> outList)
    {
        if (start == s.Length)
        {
            outList.Add(new List<string>(current));
            return;
        }
        for (int end = start + 1; end <= s.Length; end++)
        {
            var piece = s.Substring(start, end - start);
            if (IsPalindrome(piece))
            {
                current.Add(piece);
                Backtrack(s, end, current, outList);
                current.RemoveAt(current.Count - 1);
            }
        }
    }

    public static void Main()
    {
        var aab = PalindromePartitions("aab");
        Check(aab.Count == 2 && string.Join(",", aab[0]) == "a,a,b" && string.Join(",", aab[1]) == "aa,b", "aab");
        Check(PalindromePartitions("a").Count == 1, "single");
        Check(PalindromePartitions("").Count == 1 && PalindromePartitions("")[0].Count == 0, "empty -> one empty partition");
        Check(PalindromePartitions("aaa").Count == 4, "aaa has 4 partitions");
        Check(PalindromePartitions("ab").Count == 1, "ab");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
