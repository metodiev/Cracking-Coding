// Task 186 - Rabin-Karp rolling hash string search. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private const long DefaultBase = 256;
    private const long DefaultMod = 1000000007L;

    private static long PowMod(long baseValue, long exponent, long mod)
    {
        long result = 1;
        for (long i = 0; i < exponent; i++) result = result * baseValue % mod;
        return result;
    }

    /// <summary>
    /// Polynomial window hash h = c0 * base^(m-1) + ... + c(m-1) (mod mod). Sliding one step
    /// drops the leaving character's weighted term and appends the new one in O(1). Equal
    /// hashes can also mean a collision, so every hash hit is confirmed with a direct
    /// substring comparison: a false positive never becomes a reported match.
    /// </summary>
    private static List<int> RabinKarp(string text, string pattern)
    {
        return RabinKarp(text, pattern, DefaultBase, DefaultMod);
    }

    private static List<int> RabinKarp(string text, string pattern, long baseValue, long mod)
    {
        var matches = new List<int>();
        if (pattern.Length == 0) return matches;
        int m = pattern.Length;
        int n = text.Length;
        if (m > n) return matches;

        long highest = PowMod(baseValue, m - 1, mod);   // weight of a window's first character
        long patternHash = 0;
        long windowHash = 0;
        for (int i = 0; i < m; i++)
        {
            patternHash = (patternHash * baseValue + pattern[i]) % mod;
            windowHash = (windowHash * baseValue + text[i]) % mod;
        }

        if (windowHash == patternHash && text.Substring(0, m) == pattern) matches.Add(0);
        for (int start = 1; start + m <= n; start++)
        {
            windowHash = (windowHash - text[start - 1] * highest) % mod;
            if (windowHash < 0) windowHash += mod;
            windowHash = (windowHash * baseValue + text[start + m - 1]) % mod;
            if (windowHash == patternHash && text.Substring(start, m) == pattern) matches.Add(start);
        }
        return matches;
    }

    /// <summary>Reference scan used by the self-check.</summary>
    private static List<int> RabinKarpNaive(string text, string pattern)
    {
        var matches = new List<int>();
        if (pattern.Length == 0) return matches;
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (text.Substring(i, pattern.Length) == pattern) matches.Add(i);
        }
        return matches;
    }

    private static string Repeat(string unit, int times)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < times; i++) sb.Append(unit);
        return sb.ToString();
    }

    private static string RangeStart(int start, int stop, int step)
    {
        var parts = new List<string>();
        for (int i = start; i < stop; i += step) parts.Add(i.ToString());
        return string.Join(", ", parts);
    }

    private static bool Same(List<int> actual, params int[] expected)
    {
        if (actual.Count != expected.Length) return false;
        for (int i = 0; i < actual.Count; i++)
        {
            if (actual[i] != expected[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        Check(Same(RabinKarp("abracadabra", "abra"), 0, 7), "abracadabra");
        Check(Same(RabinKarp("aaaaa", "aa"), 0, 1, 2, 3), "aaaaa / aa");
        Check(Same(RabinKarp("abcabcabc", "abc"), 0, 3, 6), "repeated block");
        Check(Same(RabinKarp("abcd", "bcd"), 1), "suffix");
        Check(Same(RabinKarp("aaaa", "b")), "no match");
        Check(Same(RabinKarp("ab", "abc")), "pattern longer than text");
        Check(Same(RabinKarp("", "a")), "empty text");
        Check(Same(RabinKarp("", "")), "empty text and pattern");
        Check(Same(RabinKarp("anything", "")), "empty pattern");
        Check(Same(RabinKarp("a", "a"), 0), "single char");

        // repetition-heavy text
        string heavy = Repeat("a", 1000) + "b";
        Check(Join(RabinKarp(heavy, Repeat("a", 20))) == RangeStart(0, 981, 1), "many a-windows");
        Check(Same(RabinKarp(heavy, "ab"), 999), "single ab crossing");
        Check(Join(RabinKarp(Repeat("ab", 300), "abab")) == RangeStart(0, 597, 2), "overlapping abab");

        // agreement with the naive scan on pseudo-random strings
        var rng = new Random(17);
        for (int t = 0; t < 200; t++)
        {
            string text = RandomText(rng, rng.Next(41), 2);
            string pattern = RandomText(rng, 1 + rng.Next(5), 2);
            Check(Join(RabinKarp(text, pattern)) == Join(RabinKarpNaive(text, pattern)), "cross-check " + text + " / " + pattern);
        }

        // a tiny modulus forces collisions; the substring verification must filter them out
        for (int t = 0; t < 200; t++)
        {
            string text = RandomText(rng, rng.Next(41), 3);
            string pattern = RandomText(rng, 1 + rng.Next(4), 3);
            Check(Join(RabinKarp(text, pattern, 256, 101)) == Join(RabinKarpNaive(text, pattern)),
                    "tiny modulus " + text + " / " + pattern);
        }
        string collidingText = RandomText(rng, 60, 6);
        var seen = new Dictionary<long, string>();
        bool collided = false;
        for (int start = 0; start + 3 <= collidingText.Length; start++)
        {
            string window = collidingText.Substring(start, 3);
            long h = 0;
            for (int i = 0; i < 3; i++) h = (h * 256 + window[i]) % 101;
            if (seen.ContainsKey(h) && seen[h] != window) collided = true;
            if (!seen.ContainsKey(h)) seen[h] = window;
        }
        Check(collided, "expected the tiny modulus to produce a real hash collision");
        Console.WriteLine("PASS");
    }

    private static string Join(List<int> values)
    {
        return string.Join(", ", values);
    }

    private static string RandomText(Random rng, int length, int alphabet)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.Append((char)('a' + rng.Next(alphabet)));
        return sb.ToString();
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
