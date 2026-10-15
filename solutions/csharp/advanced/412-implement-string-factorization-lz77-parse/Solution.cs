// Task 412 - Implement string factorization (LZ77 parse). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Greedy LZ77 factorization. Every token is a triple (offset, length, next):
//   * offset = distance back to the match source (0 for a pure literal),
//   * length = matched characters copied from that earlier position,
//   * next   = the literal character that follows the match (-1 when the match
//              consumes the end of the string).
// Rules (documented, deterministic):
//   * a match source j must satisfy max(0, i - window) <= j < i (window = 8),
//   * the greedy step takes the LONGEST match, ties broken by the SMALLEST offset,
//   * matches shorter than min_len = 3 are not worth a token, so a literal is emitted,
//   * matches may overlap the source region (s[j+L] is read from the original string).
// The brute-force oracle recomputes the same token list with explicit enumeration, and
// a DP computes a lower bound on the token count of any parse.
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private const int MinLen = 3;
    private const int Window = 8;

    public static int[][] Lz77Parse(string s, int window, int minLen)
    {
        var tokens = new List<int[]>();
        int n = s.Length;
        int i = 0;
        while (i < n)
        {
            int bestLen = 0, bestOff = 0;
            for (int j = Math.Max(0, i - window); j < i; j++)
            {
                int L = 0;
                while (i + L < n && s[j + L] == s[i + L]) L++;
                if (L > bestLen) { bestLen = L; bestOff = i - j; }
            }
            if (bestLen >= minLen)
            {
                bool atEnd = i + bestLen == n;
                tokens.Add(new[] { bestOff, bestLen, atEnd ? -1 : s[i + bestLen] });
                i += bestLen + (atEnd ? 0 : 1);
            }
            else
            {
                tokens.Add(new[] { 0, 0, (int)s[i] });
                i += 1;
            }
        }
        return tokens.ToArray();
    }

    /// <summary>Replays the tokens; overlapping copies read characters emitted moments ago.</summary>
    public static string Lz77Decode(int[][] tokens)
    {
        var outSb = new StringBuilder();
        foreach (var t in tokens)
        {
            if (t[1] > 0)
            {
                int start = outSb.Length - t[0];
                for (int k = 0; k < t[1]; k++) outSb.Append(outSb[start + k]);
            }
            if (t[2] != -1) outSb.Append((char)t[2]);
        }
        return outSb.ToString();
    }

    /// <summary>Same greedy parse written with explicit candidate enumeration over all lengths.</summary>
    public static int[][] Lz77Bruteforce(string s, int window, int minLen)
    {
        var tokens = new List<int[]>();
        int n = s.Length;
        int i = 0;
        while (i < n)
        {
            int bestLen = 0, bestOff = 0;
            for (int length = n - i; length >= 1 && bestLen == 0; length--)
            {
                for (int j = Math.Max(0, i - window); j < i; j++)
                {
                    if (string.CompareOrdinal(s, j, s, i, length) == 0)
                    {
                        bestLen = length;
                        bestOff = i - j;
                        break;
                    }
                }
            }
            if (bestLen >= minLen)
            {
                bool atEnd = i + bestLen == n;
                tokens.Add(new[] { bestOff, bestLen, atEnd ? -1 : s[i + bestLen] });
                i += bestLen + (atEnd ? 0 : 1);
            }
            else
            {
                tokens.Add(new[] { 0, 0, (int)s[i] });
                i += 1;
            }
        }
        return tokens.ToArray();
    }

    /// <summary>DP: fewest tokens any parse of s can use; greedy can only be >= this.</summary>
    public static int OptimalParseLen(string s, int window, int minLen)
    {
        int n = s.Length;
        var dp = new int[n + 1];
        for (int i = n - 1; i >= 0; i--)
        {
            int best = 1 + dp[i + 1];
            for (int j = Math.Max(0, i - window); j < i; j++)
            {
                int L = 0;
                while (i + L < n && s[j + L] == s[i + L]) L++;
                for (int length = minLen; length <= L; length++)
                {
                    int cand = i + length == n ? 1 : 1 + dp[i + length + 1];
                    if (cand < best) best = cand;
                }
            }
            dp[i] = best;
        }
        return dp[0];
    }

    private static int seed = 777;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new StringBuilder();
        for (int k = 0; k < length; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    private static bool SameTokens(int[][] a, int[][] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1] || a[i][2] != b[i][2]) return false;
        }
        return true;
    }

    public static void Main()
    {
        // exact greedy factorizations demanded by the task
        Check(SameTokens(Lz77Parse("abcabcabc", 8, 3), new[]
        {
            new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' }, new[] { 0, 0, (int)'c' }, new[] { 3, 6, -1 }
        }), "abcabcabc");
        Check(SameTokens(Lz77Parse("aaaaaaaa", 8, 3),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 1, 7, -1 } }), "aaaaaaaa");

        // round trips and hand-built corner cases
        Check(Lz77Decode(Lz77Parse("abcabcabc", 8, 3)) == "abcabcabc", "rt abc");
        Check(Lz77Decode(Lz77Parse("aaaaaaaa", 8, 3)) == "aaaaaaaa", "rt aaa");
        Check(Lz77Parse("", 8, 3).Length == 0, "empty parse");
        Check(Lz77Decode(Array.Empty<int[]>()).Length == 0, "empty decode");
        Check(SameTokens(Lz77Parse("ab", 8, 3),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' } }), "too short to match");
        Check(SameTokens(Lz77Parse("abcabcx", 8, 3),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' }, new[] { 0, 0, (int)'c' }, new[] { 3, 3, (int)'x' } }),
            "match then next char");
        Check(SameTokens(Lz77Parse("aabaaba", 8, 3),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' }, new[] { 3, 4, -1 } }),
            "consume tail");

        // window bounds the match source: distance-3 source needs window >= 3
        Check(SameTokens(Lz77Parse("aabaab", 3, 3),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' }, new[] { 3, 3, -1 } }),
            "window 3");
        Check(SameTokens(Lz77Parse("aabaab", 2, 3), new[]
        {
            new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' },
            new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' }
        }), "window 2");

        // min length boundary: length-2 match is rejected, length-3 accepted
        Check(SameTokens(Lz77Parse("aaab", 8, 3),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'a' }, new[] { 0, 0, (int)'b' } }),
            "min len 3");
        Check(SameTokens(Lz77Parse("aaab", 8, 2),
            new[] { new[] { 0, 0, (int)'a' }, new[] { 1, 2, (int)'b' } }), "min len 2");

        // optimal parses of the two hand strings
        Check(OptimalParseLen("abcabcabc", 8, 3) == 4, "opt abc");
        Check(OptimalParseLen("aaaaaaaa", 8, 3) == 2, "opt aaa");
        Check(OptimalParseLen("aaab", 8, 3) == 4, "opt aaab");
        Check(OptimalParseLen("aabaab", 3, 3) == 4, "opt aabaab");

        for (int round = 0; round < 8; round++)
        {
            int length = 60 + seed % 100;
            string alphabet = seed % 2 == 0 ? "ab" : "abc";
            string text = LcgString(length, alphabet);
            int window = 4 + seed % 29;
            int[][] greedy = Lz77Parse(text, window, 3);
            int[][] slow = Lz77Bruteforce(text, window, 3);
            Check(SameTokens(greedy, slow), "oracle equal round " + round);
            Check(Lz77Decode(greedy) == text, "round trip round " + round);
            foreach (var t in greedy)
            {
                Check(t[0] <= window && (t[1] == 0 || t[1] >= 3), "invariants round " + round);
            }
            int optimal = OptimalParseLen(text, window, 3);
            Check(optimal <= greedy.Length && greedy.Length <= text.Length, "bound round " + round);
        }

        // greedy re-derivation: at every token boundary the token must be the longest
        // match, ties resolved to the smallest offset (or a literal when it is shorter)
        string str = "abababababab";
        int[][] tokens = Lz77Parse(str, 8, 3);
        int pos = 0;
        foreach (var t in tokens)
        {
            int bestLen = 0, bestJ = -1;
            for (int j = Math.Max(0, pos - 8); j < pos; j++)
            {
                int L = 0;
                while (pos + L < str.Length && str[j + L] == str[pos + L]) L++;
                if (L > bestLen) { bestLen = L; bestJ = j; }
            }
            if (t[1] > 0)
            {
                Check(t[1] == bestLen && t[0] == pos - bestJ, "rederive match at " + pos);
            }
            else
            {
                Check(bestLen < 3, "rederive literal at " + pos);
            }
            pos += t[1] + (t[2] == -1 ? 0 : 1);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
