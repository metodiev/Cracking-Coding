// Task 232 - Minimum window substring covering all chars of pattern. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    /// <summary>
    /// Sliding window with a need/missing counter: grow with `right`; when the window
    /// stops missing characters, shrink from the left while still valid, keeping the
    /// smallest window seen. O(|s| + |t|) time. Leftmost window wins ties; "" if none.
    /// </summary>
    public static string MinWindow(string s, string t)
    {
        if (s.Length == 0 || t.Length == 0 || t.Length > s.Length) return "";
        var need = new Dictionary<char, int>();
        foreach (char c in t)
        {
            need.TryGetValue(c, out int v);
            need[c] = v + 1;
        }
        int missing = t.Length;
        int left = 0, bestStart = 0, bestLen = int.MaxValue;
        for (int right = 0; right < s.Length; right++)
        {
            char ch = s[right];
            need.TryGetValue(ch, out int have);
            if (have > 0) missing--;
            need[ch] = have - 1;
            while (missing == 0)
            {
                if (right - left + 1 < bestLen)   // strict: ties keep the leftmost window
                {
                    bestStart = left;
                    bestLen = right - left + 1;
                }
                char lc = s[left];
                need[lc]++;
                if (need[lc] > 0) missing++;
                left++;
            }
        }
        return bestLen == int.MaxValue ? "" : s.Substring(bestStart, bestLen);
    }

    /// <summary>O(n^2 * k) reference used only for cross-checks; same leftmost tie-break.</summary>
    public static string BruteMinWindow(string s, string t)
    {
        if (s.Length == 0 || t.Length == 0 || t.Length > s.Length) return "";
        var want = new int[128];
        foreach (char c in t) want[c]++;
        string best = "";
        for (int left = 0; left < s.Length; left++)
        {
            for (int right = left + t.Length; right <= s.Length; right++)
            {
                var have = new int[128];
                for (int i = left; i < right; i++) have[s[i]]++;
                bool ok = true;
                for (int c = 0; c < 128 && ok; c++) if (have[c] < want[c]) ok = false;
                if (ok)
                {
                    string window = s.Substring(left, right - left);
                    if (best.Length == 0 || window.Length < best.Length) best = window;
                    break;
                }
            }
        }
        return best;
    }

    public static void Main()
    {
        Check(MinWindow("ADOBECODEBANC", "ABC") == "BANC", "classic");
        Check(MinWindow("cabwefgewcwaefgcf", "cae") == "cwae", "second classic");
        Check(MinWindow("a", "a") == "a", "single char");
        Check(MinWindow("a", "aa") == "", "pattern longer than s");
        Check(MinWindow("abc", "abcd") == "", "pattern longer than s again");
        Check(MinWindow("ab", "cd") == "", "no window");
        Check(MinWindow("aa", "aa") == "aa", "duplicates in t");
        Check(MinWindow("abaab", "aa") == "aa", "duplicates in t again");
        Check(MinWindow("bba", "ab") == "ba", "shorter right side");
        Check(MinWindow("aabc", "aabc") == "aabc", "whole string");
        Check(MinWindow("abc", "") == "", "empty pattern");
        Check(MinWindow("", "a") == "", "empty s");

        var rng = new Random(232);
        var sb = new StringBuilder();
        for (int test = 0; test < 200; test++)     // cross-check vs brute force
        {
            int sLen = rng.Next(15);
            sb.Clear();
            for (int i = 0; i < sLen; i++) sb.Append(rng.Next(2) == 0 ? 'a' : 'b');
            string s = sb.ToString();
            int tLen = rng.Next(5);
            sb.Clear();
            for (int i = 0; i < tLen; i++) sb.Append(rng.Next(2) == 0 ? 'a' : 'b');
            string t = sb.ToString();
            Check(MinWindow(s, t) == BruteMinWindow(s, t), "matches brute force");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
