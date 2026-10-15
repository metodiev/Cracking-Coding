// Task 315 - Rolling hash with two moduli to reduce collisions. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class DoubleRollingHash
{
    public const long Mod1 = 1_000_000_007L;
    public const long Mod2 = 998_244_353L;
    public const long Base = 131L;

    private readonly long[] h1;
    private readonly long[] h2;
    private readonly long[] p1;
    private readonly long[] p2;

    public DoubleRollingHash(string s)
    {
        int n = s.Length;
        h1 = new long[n + 1];
        h2 = new long[n + 1];
        p1 = new long[n + 1];
        p2 = new long[n + 1];
        p1[0] = 1;
        p2[0] = 1;
        for (int i = 0; i < n; i++)
        {
            long code = s[i];
            h1[i + 1] = (h1[i] * Base + code) % Mod1;
            h2[i + 1] = (h2[i] * Base + code) % Mod2;
            p1[i + 1] = p1[i] * Base % Mod1;
            p2[i + 1] = p2[i] * Base % Mod2;
        }
    }

    /// <summary>Hash of s[lo..hi) as the pair (mod Mod1, mod Mod2).</summary>
    public long[] Hash(int lo, int hi)
    {
        long a = (h1[hi] - h1[lo] * p1[hi - lo] % Mod1 + Mod1) % Mod1;
        long b = (h2[hi] - h2[lo] * p2[hi - lo] % Mod2 + Mod2) % Mod2;
        return new[] { a, b };
    }

    public bool Equal(int lo1, int hi1, int lo2, int hi2)
    {
        long[] left = Hash(lo1, hi1);
        long[] right = Hash(lo2, hi2);
        return left[0] == right[0] && left[1] == right[1];
    }
}

public static class Solution
{
    /// <summary>Rolling hash built from scratch, no prefix arrays; used as a cross-check.</summary>
    public static long[] DirectHash(string s)
    {
        long h1 = 0, h2 = 0;
        foreach (char ch in s)
        {
            h1 = (h1 * DoubleRollingHash.Base + ch) % DoubleRollingHash.Mod1;
            h2 = (h2 * DoubleRollingHash.Base + ch) % DoubleRollingHash.Mod2;
        }
        return new[] { h1, h2 };
    }

    public const long ModWeak = 65537L;
    public const long BaseWeak = 65536L;        // == -1 (mod 65537), a deliberately bad base

    /// <summary>Single-modulus hash with a base that is -1 modulo the modulus (alternating sum).</summary>
    public static long WeakHash(string s)
    {
        long h = 0;
        foreach (char ch in s) h = (h * BaseWeak + ch) % ModWeak;
        return h;
    }

    private static int seed = 3150;

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
        Check((DoubleRollingHash.Mod1 - 1) % 2 == 0 && (DoubleRollingHash.Mod2 - 1) % 2 == 0, "bases");
        Check(DoubleRollingHash.Mod1 != DoubleRollingHash.Mod2, "distinct moduli");

        string s = "banana";
        var rolling = new DoubleRollingHash(s);
        Check(string.Join(",", rolling.Hash(1, 3)) == string.Join(",", DirectHash("an")), "an");
        Check(string.Join(",", rolling.Hash(0, 2)) == string.Join(",", DirectHash("ba")), "ba");
        Check(rolling.Equal(1, 3, 3, 5), "an == an");
        Check(!rolling.Equal(0, 2, 3, 5), "ba != an");
        Check(rolling.Equal(1, 6, 1, 6), "self equal");

        // Deliberate collision of a weak single-modulus hash, separated by the double hash
        Check(WeakHash("ab") == WeakHash("bc") && WeakHash("ab") == 1, "weak collision");  // 97*(-1)+98 == 98*(-1)+99
        Check(WeakHash("ab") != WeakHash("ba"), "weak not universal");
        var ab = new DoubleRollingHash("ab");
        var bc = new DoubleRollingHash("bc");
        Check(!(ab.Hash(0, 2)[0] == bc.Hash(0, 2)[0] && ab.Hash(0, 2)[1] == bc.Hash(0, 2)[1]), "double separates");
        Check(!(DirectHash("ab")[0] == DirectHash("bc")[0] && DirectHash("ab")[1] == DirectHash("bc")[1]), "double separates 2");
        Check(DirectHash("ab")[0] != DirectHash("bc")[0], "single strong separator");

        // O(1) equality queries must agree with naive slicing, systematically and randomly
        var texts = new List<string> { "a", "banana", "mississippi", "aaaaaaaa" };
        for (int i = 0; i < 20; i++) texts.Add(LcgString(12, "abca"));
        for (int i = 0; i < 10; i++) texts.Add(LcgString(16, "abcd"));

        long checkedCount = 0;
        foreach (string text in texts)
        {
            var hashes = new DoubleRollingHash(text);
            int n = text.Length;
            for (int length = 1; length <= n; length++)
            {
                for (int lo1 = 0; lo1 + length <= n; lo1++)
                {
                    for (int lo2 = lo1; lo2 + length <= n; lo2++)
                    {
                        bool expected = string.CompareOrdinal(text.Substring(lo1, length), text.Substring(lo2, length)) == 0;
                        Check(hashes.Equal(lo1, lo1 + length, lo2, lo2 + length) == expected, "equal " + text);
                        checkedCount++;
                    }
                }
            }
            for (int lo = 0; lo <= n; lo++)
            {
                for (int hi = lo; hi <= n; hi++)
                {
                    Check(string.Join(",", hashes.Hash(lo, hi)) == string.Join(",", DirectHash(text.Substring(lo, hi - lo))),
                        "hash " + text);
                }
            }
        }
        Check(checkedCount > 1000, "coverage");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
