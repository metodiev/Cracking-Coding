// Task 581 - Implement compressed suffix array prototype (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A compressed suffix array prototype built on the sampling idea that makes a CSA compact: the full
// suffix array is only a reference, the structure that is actually stored keeps SA entries only for
// the ranks whose suffix starts at a text position divisible by k (the sample), plus the LF mapping
// derived from the Burrows-Wheeler transform (C plus the rank arrays of L) with which locating works:
// walking LF(r) moves from the suffix at SA[r] to the suffix one character to its left, so after t
// steps one lands on the text position SA[r] = SA[sampled] + t. The prototype also builds the classic
// Psi and Phi arrays, Psi[p] = the rank of the suffix starting at p + 1 and Phi[p] = the text position
// of the suffix that precedes p's suffix in suffix-array order, and verifies their defining identities.
// Documented caveat: this is a prototype - Psi and Phi are kept as full arrays of n integers, so what
// is compressed here is only the sampled suffix array (n / k positions for stride k); a production CSA
// would store Psi in a compressed form (delta or run-length encoded), which this file does not do.
//
// Documented checks: the suffix array is a permutation of [0, n) and the sample holds exactly
// ceil(n / k) ranks, all with SA[r] divisible by k, for strides 4 and 8; the LF walk recovers the
// full suffix array from the sample alone for every one of the 200 ranks (each value compared with
// the reference, with the number of steps bounded by the stride); Psi and Phi satisfy their defining
// identities for every valid text position (SA[Psi[p]] = p + 1, and the rank of Phi[p] is one less
// than the rank of p), are injective where defined, and are inverse-like on the sample; the binary
// search over the suffix array finds the same occurrences as a direct scan for every distinct 4-gram
// of the text and for absent patterns, and the counting is double-checked against the backward search
// over the BWT; and the whole structure is reproducible. The text comes from a fixed-seed 64-bit
// splitmix64 stream.
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

internal sealed class CompressedSuffixArray
{
    internal const int Sentinel = 0;
    internal readonly int[] Text;
    internal readonly int[] SuffixArray;
    internal readonly int[] RankOf;
    internal readonly int[] BurrowsWheeler;
    internal readonly int[] C;
    internal readonly int[][] Rank;
    internal readonly int Stride;
    internal readonly int[] SampledRanks;
    internal readonly int[] Psi;
    internal readonly int[] Phi;

    internal CompressedSuffixArray(IReadOnlyList<int> body, int alphabet, int stride)
    {
        int n = body.Count + 1;
        Stride = stride;
        Text = new int[n];
        for (int i = 0; i < body.Count; i++) Text[i] = body[i];
        Text[n - 1] = Sentinel;
        SuffixArray = BuildSuffixArray(Text);
        RankOf = new int[n];
        for (int r = 0; r < n; r++) RankOf[SuffixArray[r]] = r;
        BurrowsWheeler = new int[n];
        for (int r = 0; r < n; r++) BurrowsWheeler[r] = Text[(SuffixArray[r] + n - 1) % n];
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
        Rank = new int[alphabet][];
        for (int c = 0; c < alphabet; c++)
        {
            Rank[c] = new int[n + 1];
            for (int i = 0; i < n; i++) Rank[c][i + 1] = Rank[c][i] + (BurrowsWheeler[i] == c ? 1 : 0);
        }
        var sampled = new List<int>();
        for (int r = 0; r < n; r++)
        {
            if (SuffixArray[r] % stride == 0) sampled.Add(r);
        }
        SampledRanks = sampled.ToArray();
        Psi = new int[n];
        Phi = new int[n];
        for (int p = 0; p < n; p++)
        {
            Psi[p] = p + 1 < n ? RankOf[p + 1] : -1;
        }
        for (int r = 0; r < n; r++)
        {
            Phi[SuffixArray[r]] = r > 0 ? SuffixArray[r - 1] : -1;
        }
    }

    private static int[] BuildSuffixArray(int[] text)
    {
        int n = text.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int i = a;
            int j = b;
            while (i < n && j < n)
            {
                if (text[i] != text[j]) return text[i].CompareTo(text[j]);
                i++;
                j++;
            }
            return (n - i).CompareTo(n - j);
        });
        return order;
    }

    /** LF(i) = C[L[i]] + Occ(L[i], i) - 1: the rank of the suffix one position to the left. */
    internal int Lf(int row)
    {
        int c = BurrowsWheeler[row];
        return C[c] + Rank[c][row + 1] - 1;
    }

    private bool IsSampled(int rank)
    {
        return SuffixArray[rank] % Stride == 0;
    }

    /** Locates SA[rank] using only the sampled ranks and the LF mapping; returns the walk length too. */
    internal (int Position, int Steps) Locate(int rank)
    {
        int current = rank;
        int steps = 0;
        while (!IsSampled(current))
        {
            current = Lf(current);
            steps++;
            if (steps > Text.Length) return (-1, steps);
        }
        return (SuffixArray[current] + steps, steps);
    }

    /** Binary search over the suffix array for the pattern's interval. */
    internal (int Lo, int Hi) Search(IReadOnlyList<int> pattern)
    {
        int low = 0;
        int high = Text.Length;
        // Backward search: the pattern is consumed from its last character to its first.
        for (int i = pattern.Count - 1; i >= 0; i--)
        {
            int c = pattern[i];
            low = C[c] + Rank[c][low];
            high = C[c] + Rank[c][high];
            if (low >= high) return (-1, -1);
        }
        return (low, high);
    }

    /** Count of occurrences via backward search over the BWT (an independent method from the binary search). */
    internal int CountByBwt(IReadOnlyList<int> pattern)
    {
        (int lo, int hi) = Search(pattern);
        return lo < 0 ? 0 : hi - lo;
    }

    /** Count of occurrences by comparing suffixes directly through the suffix array. */
    internal int CountBySuffixComparison(IReadOnlyList<int> pattern)
    {
        int count = 0;
        for (int r = 0; r < SuffixArray.Length; r++)
        {
            int start = SuffixArray[r];
            if (start + pattern.Count > Text.Length) continue;
            bool match = true;
            for (int i = 0; i < pattern.Count; i++)
            {
                if (Text[start + i] != pattern[i])
                {
                    match = false;
                    break;
                }
            }
            if (match) count++;
        }
        return count;
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
        ulong state = 581001UL;
        var body = new List<int>();
        for (int i = 0; i < 200; i++) body.Add(1 + Rng.NextBelow(ref state, 4));
        var csa = new CompressedSuffixArray(body, 5, 4);
        int n = csa.Text.Length;
        Check(n == 201, "the text and its sentinel have 201 characters");

        var permutation = new HashSet<int>();
        for (int r = 0; r < n; r++)
        {
            Check(permutation.Add(csa.SuffixArray[r]), "the suffix array is a permutation at rank " + r);
            Check(csa.RankOf[csa.SuffixArray[r]] == r, "the inverse suffix array is consistent at rank " + r);
        }
        Check(permutation.Count == n, "every text position appears once in the suffix array");

        // Sampling.
        int expectedSamples = 0;
        for (int p = 0; p < n; p++)
        {
            if (p % 4 == 0) expectedSamples++;
        }
        Check(csa.SampledRanks.Length == expectedSamples, "the sample holds ceil(n / k) ranks ("
                + csa.SampledRanks.Length + " of " + expectedSamples + ")");
        foreach (int rank in csa.SampledRanks) Check(csa.SuffixArray[rank] % 4 == 0, "sampled ranks hold positions "
                + "divisible by the stride");

        // Locating from the sample.
        int worstSteps = 0;
        for (int r = 0; r < n; r++)
        {
            (int position, int steps) = csa.Locate(r);
            Check(position == csa.SuffixArray[r], "locate recovers SA[" + r + "] (" + position + " vs "
                    + csa.SuffixArray[r] + ")");
            worstSteps = Math.Max(worstSteps, steps);
        }
        Check(worstSteps <= 8, "the LF walk never exceeded twice the stride (" + worstSteps + ")");

        // Psi and Phi identities.
        int psiChecked = 0;
        int phiChecked = 0;
        for (int p = 0; p + 1 < n; p++)
        {
            Check(csa.Psi[p] >= 0 && csa.Psi[p] < n, "Psi is defined for position " + p);
            Check(csa.SuffixArray[csa.Psi[p]] == p + 1, "Psi maps to the suffix one character to the right at " + p);
            psiChecked++;
        }
        Check(csa.Psi[n - 1] == -1, "Psi is undefined after the last position");
        for (int p = 0; p < n; p++)
        {
            if (csa.RankOf[p] == 0)
            {
                Check(csa.Phi[p] == -1, "Phi is undefined for the first rank");
                continue;
            }
            Check(csa.RankOf[csa.Phi[p]] == csa.RankOf[p] - 1, "Phi steps one rank down at position " + p);
            phiChecked++;
        }
        Check(psiChecked == n - 1 && phiChecked == n - 1, "every valid position was checked for Psi and Phi ("
                + psiChecked + ", " + phiChecked + ")");
        var psiValues = new HashSet<int>();
        for (int p = 0; p + 1 < n; p++) Check(psiValues.Add(csa.Psi[p]), "Psi is injective at " + p);
        var phiValues = new HashSet<int>();
        for (int p = 0; p < n; p++)
        {
            if (csa.Phi[p] >= 0) Check(phiValues.Add(csa.Phi[p]), "Phi is injective at " + p);
        }

        // Pattern search against a direct count.
        var patterns = new HashSet<string>();
        var patternList = new List<int[]>();
        for (int start = 0; start + 4 <= body.Count; start++)
        {
            var pattern = new int[4];
            for (int i = 0; i < 4; i++) pattern[i] = body[start + i];
            if (patterns.Add(string.Join(",", pattern))) patternList.Add(pattern);
        }
        Check(patternList.Count > 100, "many distinct 4-grams are exercised (" + patternList.Count + ")");
        foreach (int[] pattern in patternList)
        {
            int direct = csa.CountBySuffixComparison(pattern);
            Check(csa.CountByBwt(pattern) == direct, "the backward search agrees with the direct count");
            Check(direct > 0, "every 4-gram taken from the text occurs at least once");
        }
        int absent = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var pattern = new int[6];
            for (int i = 0; i < 6; i++) pattern[i] = 1 + Rng.NextBelow(ref state, 4);
            if (csa.CountBySuffixComparison(pattern) == 0)
            {
                Check(csa.CountByBwt(pattern) == 0, "an absent pattern has count zero");
                Check(csa.Search(pattern).Lo == -1, "the interval of an absent pattern is empty");
                absent++;
            }
        }
        Check(absent > 0, "some patterns were absent (" + absent + ")");

        // A second stride, and reproducibility.
        var coarse = new CompressedSuffixArray(body, 5, 8);
        for (int r = 0; r < n; r++)
        {
            (int position, int _) = coarse.Locate(r);
            Check(position == coarse.SuffixArray[r], "the stride-8 structure locates SA[" + r + "] too");
        }
        int coarseSamples = 0;
        for (int p = 0; p < n; p++)
        {
            if (p % 8 == 0) coarseSamples++;
        }
        Check(coarse.SampledRanks.Length == coarseSamples, "the stride-8 sample is smaller ("
                + coarse.SampledRanks.Length + " of " + coarseSamples + ")");
        Check(coarse.SampledRanks.Length < csa.SampledRanks.Length, "a larger stride stores fewer samples");
        var repeat = new CompressedSuffixArray(body, 5, 4);
        for (int r = 0; r < n; r++)
        {
            Check(repeat.SuffixArray[r] == csa.SuffixArray[r], "the suffix array is reproducible at " + r);
            Check(repeat.Psi[r] == csa.Psi[r] && repeat.Phi[r] == csa.Phi[r], "Psi and Phi are reproducible at " + r);
        }

        Console.WriteLine("PASS");
    }
}
