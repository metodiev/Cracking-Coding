// Task 319 - Sliding window with variable constraints (two pointers, advanced). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class Solution
{
    /// <summary>Length of the longest substring with at most k distinct characters (0 if k &lt;= 0).</summary>
    public static int LongestAtMostKDistinct(string s, int k)
    {
        if (k <= 0) return 0;
        var counts = new Dictionary<char, int>();
        int left = 0, best = 0;
        for (int right = 0; right < s.Length; right++)
        {
            char ch = s[right];
            counts[ch] = counts.TryGetValue(ch, out int seen) ? seen + 1 : 1;
            while (counts.Count > k)
            {
                char removed = s[left++];
                if (--counts[removed] == 0) counts.Remove(removed);
            }
            best = Math.Max(best, right - left + 1);
        }
        return best;
    }

    /// <summary>Length of the longest substring obtainable by replacing at most k characters.</summary>
    public static int CharacterReplacement(string s, int k)
    {
        var counts = new Dictionary<char, int>();
        int left = 0, best = 0, maxCount = 0;
        for (int right = 0; right < s.Length; right++)
        {
            char ch = s[right];
            counts[ch] = counts.TryGetValue(ch, out int seen) ? seen + 1 : 1;
            maxCount = Math.Max(maxCount, counts[ch]);
            while (right - left + 1 - maxCount > k)
            {
                char removed = s[left++];
                counts[removed]--;
                maxCount = counts.Values.Max();             // recomputed after every shrink
            }
            best = Math.Max(best, right - left + 1);
        }
        return best;
    }

    /// <summary>Try every window; the independent oracle for the distinct-characters variant.</summary>
    public static int BruteAtMostKDistinct(string s, int k)
    {
        int best = 0;
        for (int i = 0; i < s.Length; i++)
        {
            var seen = new HashSet<char>();
            for (int j = i; j < s.Length; j++)
            {
                seen.Add(s[j]);
                if (seen.Count > k) break;
                best = Math.Max(best, j - i + 1);
            }
        }
        return best;
    }

    /// <summary>Try every window and count its maximal frequency; the independent oracle.</summary>
    public static int BruteCharacterReplacement(string s, int k)
    {
        int best = 0;
        for (int i = 0; i < s.Length; i++)
        {
            var counts = new Dictionary<char, int>();
            for (int j = i; j < s.Length; j++)
            {
                counts[s[j]] = counts.TryGetValue(s[j], out int seen) ? seen + 1 : 1;
                if (j - i + 1 - counts.Values.Max() <= k) best = Math.Max(best, j - i + 1);
            }
        }
        return best;
    }

    private static int seed = 3190;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        Check(LongestAtMostKDistinct("eceba", 2) == 3, "eceba k=2");              // "ece"
        Check(LongestAtMostKDistinct("aa", 1) == 2, "aa k=1");
        Check(LongestAtMostKDistinct("abcadcacacaca", 3) == 11, "13 chars k=3");  // "cadcacacaca"
        Check(LongestAtMostKDistinct("abcadcacacaca", 2) == 8, "13 chars k=2");   // "cacacaca"
        Check(LongestAtMostKDistinct("", 2) == 0, "empty");
        Check(LongestAtMostKDistinct("abc", 0) == 0, "k=0");
        Check(LongestAtMostKDistinct("abc", 5) == 3, "k large");
        Check(LongestAtMostKDistinct("abaccc", 1) == 3, "ccc");

        Check(CharacterReplacement("ABAB", 2) == 4, "ABAB k=2");
        Check(CharacterReplacement("AABABBA", 1) == 4, "AABABBA k=1");
        Check(CharacterReplacement("AAAA", 2) == 4, "AAAA");
        Check(CharacterReplacement("ABCDE", 0) == 1, "ABCDE k=0");
        Check(CharacterReplacement("", 0) == 0, "empty");
        Check(CharacterReplacement("A", 0) == 1, "single");
        Check(CharacterReplacement("ABAB", 0) == 1, "ABAB k=0");

        var samples = new List<string> { "eceba", "aa", "abcadcacacaca" };
        for (int i = 0; i < 30; i++) samples.Add(LcgString(20, "abc"));
        for (int i = 0; i < 15; i++) samples.Add(LcgString(14, "ab"));

        foreach (string s in samples)
        {
            for (int k = 0; k < 4; k++)
            {
                Check(LongestAtMostKDistinct(s, k) == BruteAtMostKDistinct(s, k), "distinct " + s + " " + k);
                Check(CharacterReplacement(s, k) == BruteCharacterReplacement(s, k), "replace " + s + " " + k);
            }
            Check(LongestAtMostKDistinct(s, 4) == s.Length, "all distinct " + s);
            for (int k = 0; k < 3; k++)                                             // monotone in k
            {
                Check(LongestAtMostKDistinct(s, k) <= LongestAtMostKDistinct(s, k + 1), "monotone distinct");
                Check(CharacterReplacement(s, k) <= CharacterReplacement(s, k + 1), "monotone replace");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
