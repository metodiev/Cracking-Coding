// Task 233 - Group anagrams (hashmap with sorted key). (Tier: Intermediate, CTCI Ch 10)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>
    /// Key each word by its characters sorted ("eat"/"tea"/"ate" -> "aet") and collect
    /// the words in a dictionary: O(n * L log L) for n words of length L.
    /// Deterministic order: groups appear in first-seen key order, words keep input order.
    /// </summary>
    public static List<List<string>> GroupAnagrams(string[] words)
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
        CheckGroups(GroupAnagrams(new[] { "eat", "tea", "tan", "ate", "nat", "bat" }),
            new[] { new[] { "eat", "tea", "ate" }, new[] { "tan", "nat" }, new[] { "bat" } },
            "classic input");
        Check(GroupAnagrams(new string[0]).Count == 0, "empty input");
        CheckGroups(GroupAnagrams(new[] { "" }), new[] { new[] { "" } }, "single empty word");
        CheckGroups(GroupAnagrams(new[] { "", "" }), new[] { new[] { "", "" } }, "two empty words");
        CheckGroups(GroupAnagrams(new[] { "a" }), new[] { new[] { "a" } }, "single word");
        CheckGroups(GroupAnagrams(new[] { "ab", "ba", "abc", "cab", "bca" }),
            new[] { new[] { "ab", "ba" }, new[] { "abc", "cab", "bca" } }, "different lengths");
        CheckGroups(GroupAnagrams(new[] { "a", "b", "a" }),
            new[] { new[] { "a", "a" }, new[] { "b" } }, "repeated words split by key");
        CheckGroups(GroupAnagrams(new[] { "abc", "abc" }),
            new[] { new[] { "abc", "abc" } }, "same word twice");
        CheckGroups(GroupAnagrams(new[] { "listen", "silent", "enlist", "google" }),
            new[] { new[] { "listen", "silent", "enlist" }, new[] { "google" } }, "listen/silent/enlist");
        Console.WriteLine("PASS");
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
