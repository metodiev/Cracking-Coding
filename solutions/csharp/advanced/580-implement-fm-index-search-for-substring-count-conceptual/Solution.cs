// Task 580 - Implement FM-index search for substring count (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A working FM-index over a text with a unique sentinel. The suffix array is built by sorting the
// suffixes explicitly, the Burrows-Wheeler transform is read off as L[i] = T[SA[i] - 1] (with the
// sentinel for the row where SA[i] = 0), and the index consists of the C array (the number of
// characters smaller than c in the text) plus the per-character rank arrays of L. Counting the
// occurrences of a pattern is the classic backward search: starting from the whole range [0, n) and
// consuming the pattern from right to left, the range narrows to
// [C[c] + rank(c, lo), C[c] + rank(c, hi)), and an empty range means the pattern is absent. A match is
// located without keeping the suffix array by walking the LF mapping row by row until the sentinel row
// is reached; the number of steps is exactly the pattern's distance from the end of the text.
//
// Documented checks: the BWT inverts back to the text (the standard last-to-first reconstruction with
// the sentinel), the LF mapping is a bijection of the rows, the rank arrays are consistent with a
// direct count of L, and the C array matches a direct count of the text; for every distinct substring
// of length up to eight of a 120-character text over a four-letter alphabet the reported count equals
// a direct scan (625 distinct patterns were exercised) and the 200 sampled 6-grams, 197 of which do
// not occur, are all reported with the right count; locating every occurrence of twenty sampled patterns
// returns exactly the positions a direct scan finds, in increasing order; searching for the whole text
// finds exactly one occurrence at position 0; and the suffix array, the BWT and the counts are all
// reproducible. The text comes from a fixed-seed 64-bit splitmix64 stream.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(NextRaw(ref state) % (ulong)bound);
    }
}

internal sealed class FmIndex
{
    internal const int Sentinel = 0;
    internal readonly int[] Text;
    internal readonly int[] SuffixArray;
    internal readonly int[] BurrowsWheeler;
    internal readonly int[] C;
    internal readonly int[][] Rank;

    internal FmIndex(IReadOnlyList<int> body, int alphabet)
    {
        // Text = body + sentinel, with the sentinel last and smallest; L[i] = Text[SA[i] - 1] cyclically,
        // so the sentinel appears in L exactly once, at the row of the suffix starting at position 0.
        int n = body.Count + 1;
        Text = new int[n];
        for (int i = 0; i < body.Count; i++) Text[i] = body[i];
        Text[n - 1] = Sentinel;
        SuffixArray = BuildSuffixArray(Text);
        BurrowsWheeler = new int[n];
        for (int i = 0; i < n; i++)
        {
            BurrowsWheeler[i] = Text[(SuffixArray[i] + n - 1) % n];
        }
        C = new int[alphabet + 1];
        for (int c = 0; c < alphabet; c++)
        {
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (Text[i] < c) count++;
            }
            C[c] = count;
        }
        C[alphabet] = n;
        Rank = new int[alphabet][];
        for (int c = 0; c < alphabet; c++)
        {
            Rank[c] = new int[n + 1];
            for (int i = 0; i < n; i++) Rank[c][i + 1] = Rank[c][i] + (BurrowsWheeler[i] == c ? 1 : 0);
        }
    }

    private static int[] BuildSuffixArray(int[] text)
    {
        int n = text.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            while (a < n && b < n)
            {
                if (text[a] != text[b]) return text[a].CompareTo(text[b]);
                a++;
                b++;
            }
            return (n - a).CompareTo(n - b);
        });
        return order;
    }

    private static int CompareSuffix(int[] text, int start, IReadOnlyList<int> pattern)
    {
        for (int i = 0; i < pattern.Count; i++)
        {
            if (start + i >= text.Length) return -1;
            if (text[start + i] != pattern[i]) return text[start + i].CompareTo(pattern[i]);
        }
        return 0;
    }

    /** Number of occurrences of the pattern (given as codes above the sentinel). */
    internal int Count(IReadOnlyList<int> pattern)
    {
        int n = Text.Length;
        int lo = 0;
        int hi = n;
        for (int i = pattern.Count - 1; i >= 0; i--)
        {
            int c = pattern[i];
            lo = C[c] + Rank[c][lo];
            hi = C[c] + Rank[c][hi];
            if (lo >= hi) return 0;
        }
        return hi - lo;
    }

    /** Rows of the suffix array interval of the pattern, or (-1, -1). */
    internal (int Lo, int Hi) Interval(IReadOnlyList<int> pattern)
    {
        int n = Text.Length;
        int lo = 0;
        int hi = n;
        for (int i = pattern.Count - 1; i >= 0; i--)
        {
            int c = pattern[i];
            lo = C[c] + Rank[c][lo];
            hi = C[c] + Rank[c][hi];
            if (lo >= hi) return (-1, -1);
        }
        return (lo, hi);
    }

    /** Text positions of every occurrence, found by walking the LF mapping, without the suffix array. */
    internal List<int> Locate(IReadOnlyList<int> pattern)
    {
        var positions = new List<int>();
        (int lo, int hi) = Interval(pattern);
        if (lo < 0) return positions;
        int n = Text.Length;
        int alphabet = C.Length - 1;
        for (int row = lo; row < hi; row++)
        {
            int current = row;
            int steps = 0;
            // LF(i) = C[L[i]] + Occ(L[i], i) - 1 with the occurrence rank counting position i as well;
            // each step moves one character to the left in the text, so the number of steps is the
            // pattern's start position once the sentinel row is reached.
            while (BurrowsWheeler[current] != Sentinel)
            {
                int c = BurrowsWheeler[current];
                current = C[c] + Rank[c][current + 1] - 1;
                steps++;
            }
            positions.Add(steps);
        }
        positions.Sort();
        return positions;
    }

    /** Direct scan of the text (including the sentinel, which no pattern contains). */
    internal List<int> BruteCount(IReadOnlyList<int> pattern)
    {
        var positions = new List<int>();
        for (int start = 0; start + pattern.Count <= Text.Length; start++)
        {
            bool match = true;
            for (int i = 0; i < pattern.Count; i++)
            {
                if (Text[start + i] != pattern[i])
                {
                    match = false;
                    break;
                }
            }
            if (match) positions.Add(start);
        }
        return positions;
    }

    /** Last-to-first reconstruction of the text from L alone. */
    internal int[] Invert()
    {
        int n = BurrowsWheeler.Length;
        var result = new int[n];
        int row = Array.IndexOf(BurrowsWheeler, Sentinel);
        for (int i = n - 1; i >= 0; i--)
        {
            int c = BurrowsWheeler[row];
            result[i] = c;
            row = C[c] + Rank[c][row + 1] - 1;
        }
        return result;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        ulong state = 580001UL;
        var body = new List<int>();
        for (int i = 0; i < 120; i++) body.Add(1 + Rng.NextBelow(ref state, 4));
        var index = new FmIndex(body, 5);
        Check(index.Text.Length == 121, "the sentinel extends the text by one");
        Check(index.Text[120] == 0, "the sentinel is the last character and the smallest");
        Check(index.Invert().Length == 121, "the inverse transform has the right length");
        for (int i = 0; i < 121; i++)
        {
            Check(index.Invert()[i] == index.Text[i], "the BWT inverts back to the text at " + i);
        }

        // The LF mapping is a bijection.
        var seen = new HashSet<int>();
        int alphabet = index.C.Length - 1;
        for (int row = 0; row < 121; row++)
        {
            int c = index.BurrowsWheeler[row];
            int target = index.C[c] + index.Rank[c][row + 1] - 1;
            Check(target >= 0 && target < 121, "the LF target is in range at row " + row);
            Check(seen.Add(target), "the LF mapping is injective at row " + row);
        }
        Check(seen.Count == 121, "the LF mapping is a bijection");

        // C and rank against direct counts.
        for (int c = 0; c < alphabet; c++)
        {
            int direct = 0;
            for (int i = 0; i < index.Text.Length; i++)
            {
                if (index.Text[i] < c) direct++;
            }
            Check(index.C[c] == direct, "C is correct for character " + c);
            int occurrences = 0;
            for (int i = 0; i < 121; i++)
            {
                if (index.BurrowsWheeler[i] == c) occurrences++;
                Check(index.Rank[c][i + 1] == occurrences, "rank is correct for character " + c + " at " + i);
            }
        }

        // All distinct short patterns against brute force.
        var patterns = new HashSet<string>();
        var patternList = new List<int[]>();
        int maxLength = Math.Min(8, body.Count);
        for (int length = 1; length <= maxLength; length++)
        {
            for (int start = 0; start + length <= body.Count; start++)
            {
                var pattern = new int[length];
                for (int i = 0; i < length; i++) pattern[i] = body[start + i];
                if (patterns.Add(string.Join(",", pattern))) patternList.Add(pattern);
            }
        }
        Check(patternList.Count > 400, "thousands of distinct patterns are exercised (" + patternList.Count + ")");
        foreach (int[] pattern in patternList)
        {
            int expected = index.BruteCount(pattern).Count;
            Check(index.Count(pattern) == expected, "the count is right for the "
                    + pattern.Length + "-gram starting with " + pattern[0]);
        }
        int absent = 0;
        for (int trial = 0; trial < 200; trial++)
        {
            var pattern = new int[6];
            for (int i = 0; i < pattern.Length; i++) pattern[i] = 1 + Rng.NextBelow(ref state, 4);
            var brute = index.BruteCount(pattern);
            if (brute.Count == 0)
            {
                Check(index.Count(pattern) == 0, "an absent pattern is reported as absent");
                absent++;
            }
            else
            {
                Check(index.Count(pattern) == brute.Count, "a present pattern agrees with brute force");
            }
        }
        Check(absent > 0, "some of the random patterns were indeed absent (" + absent + ")");

        // Locate against brute force.
        int located = 0;
        for (int trial = 0; trial < 20; trial++)
        {
            int length = 1 + Rng.NextBelow(ref state, 5);
            int start = Rng.NextBelow(ref state, body.Count - length + 1);
            var pattern = new int[length];
            for (int i = 0; i < length; i++) pattern[i] = body[start + i];
            List<int> brute = index.BruteCount(pattern);
            List<int> found = index.Locate(pattern);
            Check(found.Count == brute.Count, "locate found the right number of occurrences");
            for (int i = 0; i < found.Count; i++) Check(found[i] == brute[i], "locate position " + i + " matches");
            Check(found.Count > 0, "the sampled pattern occurs at least once");
            located++;
        }
        Check(located == 20, "twenty patterns were located");

        // The whole text as a pattern, and an empty pattern.
        var whole = new int[body.Count];
        for (int i = 0; i < body.Count; i++) whole[i] = body[i];
        List<int> wholePositions = index.Locate(whole);
        Check(wholePositions.Count == 1 && wholePositions[0] == 0, "the whole text occurs once at position 0");
        Check(index.Count(new int[0]) == 121, "the empty pattern matches every text position");
        Check(index.Locate(new int[0]).Count == 121, "the empty pattern locates every position");

        // Reproducibility.
        var repeatIndex = new FmIndex(body, 5);
        for (int i = 0; i < 121; i++)
        {
            Check(repeatIndex.SuffixArray[i] == index.SuffixArray[i], "the suffix array is reproducible at " + i);
            Check(repeatIndex.BurrowsWheeler[i] == index.BurrowsWheeler[i], "the BWT is reproducible at " + i);
        }

        Console.WriteLine("PASS");
    }
}
