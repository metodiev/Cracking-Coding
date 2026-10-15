// Task 410 - FM-index: basic concept and small demo. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Structure for text s: build t = s + "$" with a unique smallest sentinel, store
//   * the Burrows-Wheeler last column L of t,
//   * the C table: C[c] = number of characters of t strictly smaller than c,
//   * rank samples: for every character, the cumulative count in L at each block boundary
//     (block size configurable); occ(c, i) = sample + direct scan of the residual (< block).
//
// Backward search walks the pattern from the right: starting from the full row range [0, n+1),
// for each character c set lo = C[c] + occ(c, lo) and hi = C[c] + occ(c, hi). The interval is
// the contiguous range of rows whose rotations start with the pattern suffix processed so far,
// so the final width counts the matches; an empty interval means no match.
//
// LF-mapping (the inverse of the sorted first column F): LF[i] = C[L[i]] + rank of L[i] among
// equal characters before i. Following LF moves from a row to the rotation starting one
// position earlier, so walking LF from a match row until row 0 (the "$..." rotation) gives the
// position: after `steps` LF applications the match starts at (n + steps) mod (n + 1) in t.
//
// Checks: hand values for banana ("ana" -> 2, "na" -> 2, "a" -> 3, locate positions [1,3]
// etc.), occ() against direct counts, LF is one cycle, LCG texts/patterns agree with a naive
// scan for both count and locate, and several block sizes give identical answers.
using System;
using System.Collections.Generic;
using System.Text;

internal static class FmBwt
{
    public const char Sentinel = '$';

    public static string BwtLast(string s)
    {
        int n = s.Length;
        if (n == 0) return "";
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            for (int k = 0; k < n; k++)
            {
                char ca = s[(a + k) % n];
                char cb = s[(b + k) % n];
                if (ca != cb) return ca < cb ? -1 : 1;
            }
            return 0;
        });
        var last = new StringBuilder();
        for (int i = 0; i < n; i++) last.Append(s[(order[i] - 1 + n) % n]);
        return last.ToString();
    }

    public static int[] LfMapping(string last)
    {
        int n = last.Length;
        var lf = new int[n];
        var stamps = new SortedDictionary<char, int>();
        int total = 0;
        foreach (char ch in SortedUnique(last))
        {
            stamps[ch] = total;
            int count = 0;
            foreach (char c in last) if (c == ch) count++;
            total += count;
        }
        var seen = new Dictionary<char, int>();
        for (int i = 0; i < n; i++)
        {
            char ch = last[i];
            seen.TryGetValue(ch, out int rank);
            lf[i] = stamps[ch] + rank;
            seen[ch] = rank + 1;
        }
        return lf;
    }

    public static List<char> SortedUnique(string s)
    {
        var set = new SortedSet<char>();
        foreach (char ch in s) set.Add(ch);
        return new List<char>(set);
    }
}

internal class FmIndex
{
    public readonly string Text;
    public readonly string T;
    public readonly string Last;
    public readonly int Block;
    public readonly Dictionary<char, int> CTable = new Dictionary<char, int>();
    public readonly Dictionary<char, int[]> Samples = new Dictionary<char, int[]>();
    public readonly int[] Lf;

    public FmIndex(string text) : this(text, 4)
    {
    }

    public FmIndex(string text, int block)
    {
        Text = text;
        Block = block;
        T = text + FmBwt.Sentinel;
        Last = FmBwt.BwtLast(T);
        foreach (char ch in FmBwt.SortedUnique(T))
        {
            int smaller = 0;
            foreach (char x in T) if (x < ch) smaller++;
            CTable[ch] = smaller;
        }
        foreach (char ch in FmBwt.SortedUnique(T))
        {
            var row = new int[T.Length / block + 1];
            int count = 0;
            for (int pos = 0; pos <= T.Length; pos++)
            {
                if (pos % block == 0) row[pos / block] = count;
                if (pos < T.Length && Last[pos] == ch) count++;
            }
            Samples[ch] = row;
        }
        Lf = FmBwt.LfMapping(Last);
    }

    public int Occ(char ch, int i)
    {
        int[] row = Samples[ch];
        int start = (i / Block) * Block;
        int count = row[i / Block];
        for (int pos = start; pos < i; pos++)
        {
            if (Last[pos] == ch) count++;
        }
        return count;
    }

    public (int Lo, int Hi) BackwardSearch(string pattern)
    {
        int lo = 0, hi = T.Length;
        for (int i = pattern.Length - 1; i >= 0; i--)
        {
            char ch = pattern[i];
            if (!CTable.TryGetValue(ch, out int c)) return (lo, lo);
            lo = c + Occ(ch, lo);
            hi = c + Occ(ch, hi);
            if (lo >= hi) return (lo, hi);
        }
        return (lo, hi);
    }

    public int Count(string pattern)
    {
        if (pattern.Length == 0) return 0;             // documented: empty pattern is not searched
        var (lo, hi) = BackwardSearch(pattern);
        return hi - lo;
    }

    public int[] Locate(string pattern)
    {
        if (pattern.Length == 0) return Array.Empty<int>();
        var (lo, hi) = BackwardSearch(pattern);
        var positions = new List<int>();
        for (int row = lo; row < hi; row++)
        {
            int steps = 0;
            int current = row;
            while (current != 0)
            {
                current = Lf[current];
                steps++;
            }
            positions.Add((Text.Length + steps) % T.Length);
        }
        int[] outArray = positions.ToArray();
        Array.Sort(outArray);
        return outArray;
    }
}

public static class Solution
{
    private static int seed = 2718;

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

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NaiveCount(string text, string pattern)
    {
        int count = 0;
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) count++;
        }
        return count;
    }

    private static int[] NaiveLocate(string text, string pattern)
    {
        var positions = new List<int>();
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) positions.Add(i);
        }
        return positions.ToArray();
    }

    private static void CheckAgainstNaive(string text, IEnumerable<string> patterns)
    {
        var index = new FmIndex(text);
        foreach (string pattern in patterns)
        {
            Check(index.Count(pattern) == NaiveCount(text, pattern), "count " + text + "/" + pattern);
            Check(string.Join(",", index.Locate(pattern)) == string.Join(",", NaiveLocate(text, pattern)),
                "locate " + text + "/" + pattern);
        }
    }

    public static void Main()
    {
        var index = new FmIndex("banana");
        Check(index.T == "banana$", "sentinel text");
        Check(index.Last == "annb$aa", "bwt");
        Check(index.CTable['$'] == 0 && index.CTable['a'] == 1 && index.CTable['b'] == 4 && index.CTable['n'] == 5,
            "C table");
        foreach (char ch in new[] { '$', 'a', 'b', 'n' })
        {
            for (int i = 0; i <= index.T.Length; i++)
            {
                int direct = 0;
                for (int k = 0; k < i; k++) if (index.Last[k] == ch) direct++;
                Check(index.Occ(ch, i) == direct, "occ " + ch + "/" + i);
            }
        }

        Check(index.Count("ana") == 2 && index.Count("na") == 2 && index.Count("a") == 3, "counts");
        Check(index.Count("banana") == 1 && index.Count("nana") == 1 && index.Count("anana") == 1, "counts 2");
        Check(index.Count("bananas") == 0 && index.Count("x") == 0 && index.Count("") == 0, "counts 3");
        Check(string.Join(",", index.Locate("ana")) == "1,3", "locate ana");
        Check(string.Join(",", index.Locate("na")) == "2,4", "locate na");
        Check(string.Join(",", index.Locate("a")) == "1,3,5", "locate a");
        Check(string.Join(",", index.Locate("banana")) == "0", "locate banana");
        Check(string.Join(",", index.Locate("nana")) == "2", "locate nana");
        Check(string.Join(",", index.Locate("anana")) == "1", "locate anana");
        Check(index.Locate("bananas").Length == 0 && index.Locate("x").Length == 0 && index.Locate("").Length == 0,
            "locate empty results");
        var sortedLf = (int[])index.Lf.Clone();
        Array.Sort(sortedLf);
        for (int i = 0; i < sortedLf.Length; i++) Check(sortedLf[i] == i, "lf permutation " + i);
        int row = 0;
        for (int step = 0; step < index.T.Length; step++) row = index.Lf[row];
        Check(row == 0, "lf single cycle");

        var reference = new FmIndex("banana", 1);
        foreach (int block in new[] { 1, 2, 3, 5, 8, 16 })
        {
            var variant = new FmIndex("banana", block);
            foreach (string pattern in new[] { "a", "an", "ana", "anan", "n", "na", "banana", "zz" })
            {
                Check(variant.Count(pattern) == reference.Count(pattern), "block count " + block);
                Check(string.Join(",", variant.Locate(pattern)) == string.Join(",", reference.Locate(pattern)),
                    "block locate " + block);
            }
            foreach (char ch in new[] { '$', 'a', 'b', 'n' })
            {
                for (int i = 0; i <= variant.T.Length; i++)
                {
                    Check(variant.Occ(ch, i) == reference.Occ(ch, i), "block occ " + block + "/" + ch + "/" + i);
                }
            }
        }

        CheckAgainstNaive("mississippi", new[] { "i", "is", "issi", "ssi", "ppi", "sipp", "mississippix", "q" });
        CheckAgainstNaive("", new[] { "a", "x" });
        CheckAgainstNaive("a", new[] { "a", "aa", "b" });
        CheckAgainstNaive("aaaa", new[] { "a", "aa", "aaa", "aaaa", "aaaaa", "b" });

        for (int trial = 0; trial < 30; trial++)
        {
            string text = LcgString(30, "abc");
            var patterns = new List<string>();
            for (int i = 0; i < text.Length; i++)
            {
                for (int j = i + 1; j <= Math.Min(text.Length, i + 4); j++) patterns.Add(text.Substring(i, j - i));
            }
            patterns.AddRange(new[] { text, text.Substring(0, text.Length - 1), "d" + text.Substring(0, 5), "zz", "q" });
            CheckAgainstNaive(text, patterns);
        }

        string medium = LcgString(300, "abc");
        var mediumIndex = new FmIndex(medium, 7);
        foreach (string pattern in new[] { "abc", "cba", "cab", "abcabc", "zzz", medium.Substring(150, 10), medium })
        {
            Check(mediumIndex.Count(pattern) == NaiveCount(medium, pattern), "medium count " + pattern);
            Check(string.Join(",", mediumIndex.Locate(pattern)) == string.Join(",", NaiveLocate(medium, pattern)),
                "medium locate " + pattern);
        }
        Console.WriteLine("PASS");
    }
}
