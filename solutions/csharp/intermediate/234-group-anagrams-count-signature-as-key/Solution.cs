// Task 234 - Group anagrams (count signature as key). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    /// <summary>
    /// Key each word by its 26-entry count signature (how many 'a's, 'b's, ... it
    /// contains) instead of the sorted word: O(n * L) time, no sorting. Inputs are
    /// assumed lowercase. Ordering is identical to task 233 (first-seen key order,
    /// input order within a group), so both approaches return equal output.
    /// </summary>
    public static List<List<string>> GroupAnagramsSignature(string[] words)
    {
        var groups = new Dictionary<string, List<string>>();
        var keyOrder = new List<string>();
        foreach (string word in words)
        {
            var counts = new int[26];
            foreach (char ch in word) counts[ch - 'a']++;
            string key = string.Join(",", counts);
            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<string>();
                groups[key] = list;
                keyOrder.Add(key);
            }
            list.Add(word);
        }
        var result = new List<List<string>>();
        foreach (string key in keyOrder) result.Add(groups[key]);
        return result;
    }

    /// <summary>Task 233's sorted-key variant, kept here only for the agreement checks.</summary>
    public static List<List<string>> GroupAnagramsSorted(string[] words)
    {
        var groups = new Dictionary<string, List<string>>();
        var keyOrder = new List<string>();
        foreach (string word in words)
        {
            var chars = word.ToCharArray();
            Array.Sort(chars);
            string key = new string(chars);
            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<string>();
                groups[key] = list;
                keyOrder.Add(key);
            }
            list.Add(word);
        }
        var result = new List<List<string>>();
        foreach (string key in keyOrder) result.Add(groups[key]);
        return result;
    }

    public static void Main()
    {
        string[][] cases =
        {
            new[] { "eat", "tea", "tan", "ate", "nat", "bat" },
            new[] { "listen", "silent", "enlist", "google", "banana", "banana" },
            new[] { "", "a", "", "a", "ab", "ba" },
            new[] { "abc", "abc" },
            new[] { "z" },
            new string[0],
        };
        foreach (string[] words in cases)           // both approaches must agree
            Check(SameGroups(GroupAnagramsSignature(words), GroupAnagramsSorted(words)), "approaches agree");

        CheckGroups(GroupAnagramsSignature(new[] { "eat", "tea", "tan", "ate", "nat", "bat" }),
            new[] { new[] { "eat", "tea", "ate" }, new[] { "tan", "nat" }, new[] { "bat" } },
            "classic input");
        Check(GroupAnagramsSignature(new string[0]).Count == 0, "empty input");
        CheckGroups(GroupAnagramsSignature(new[] { "" }), new[] { new[] { "" } }, "empty word");
        CheckGroups(GroupAnagramsSignature(new[] { "a", "b", "a" }),
            new[] { new[] { "a", "a" }, new[] { "b" } }, "repeated words split by key");
        CheckGroups(GroupAnagramsSignature(new[] { "ab", "ba", "abc", "cab", "bca" }),
            new[] { new[] { "ab", "ba" }, new[] { "abc", "cab", "bca" } }, "different lengths");

        var rng = new Random(234);
        var sb = new StringBuilder();
        for (int test = 0; test < 200; test++)      // random agreement check
        {
            int count = rng.Next(11);
            var words = new string[count];
            for (int i = 0; i < count; i++)
            {
                sb.Clear();
                int len = rng.Next(7);
                for (int j = 0; j < len; j++) sb.Append((char) ('a' + rng.Next(3)));
                words[i] = sb.ToString();
            }
            Check(SameGroups(GroupAnagramsSignature(words), GroupAnagramsSorted(words)), "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static bool SameGroups(List<List<string>> a, List<List<string>> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Count != b[i].Count) return false;
            for (int j = 0; j < a[i].Count; j++)
                if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    private static void CheckGroups(List<List<string>> actual, string[][] expected, string label)
    {
        bool ok = actual.Count == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++)
        {
            ok = actual[i].Count == expected[i].Length;
            for (int j = 0; ok && j < expected[i].Length; j++)
                ok = actual[i][j] == expected[i][j];
        }
        Check(ok, label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
