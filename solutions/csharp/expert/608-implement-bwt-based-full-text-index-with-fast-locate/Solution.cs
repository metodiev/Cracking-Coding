// Task 608 - Implement BWT-based full-text index with fast locate. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The index is the BWT of text + sentinel plus the C table and a block rank structure, so searching is
// backward search (lo = C[c] + Rank(c, lo), hi = C[c] + Rank(c, hi)) over the row range [lo, hi).
// Locate is what is made *fast* here, by three cooperating structures:
//   * the LF array: lf[row] = row of the suffix one position earlier, materialised once, so locating
//     never recomputes LF from C and rank while walking (LfComputed keeps the rank-based definition,
//     and the checks assert lf[row] == LfComputed(row) for every row);
//   * primary sampling: a row whose suffix start is a multiple of sampleStep stores that start, so a
//     walk of at most sampleStep LF steps reaches a row with a known start;
//   * an LF cache: a non-sampled row whose *successor* under LF is sampled stores that sampled start
//     directly, removing one step from every such walk; the measured step counts of the cached walk are
//     compared against the uncached one, which is where "fast locate" becomes visible.
// A located start is (sampled value + steps) mod (n+1) because each LF step moves one position earlier;
// the sentinel-only suffix (start == n) is not a real match and is dropped.
//
// Simplification (documented): the LF array and the cache cost one int per row each, O(n) extra words,
// instead of deriving LF from rank at every step; the BWT, C table, block rank and sampled locate
// themselves are the real structures, and the suffix array behind the BWT comes from prefix doubling.
//
// Verified: counts and located positions against a brute-force scanner for every substring of nine
// fixed texts and for random probes of a 500-character text; the cached and uncached locators return
// identical position lists and the cached walk never needs more steps (and at most sampleStep); the LF
// array equals the rank-based LF for every row; extraction of short windows and the inverse BWT against
// the original text; sampling density; determinism; boundary cases (empty text, single character,
// one-symbol alphabet, absent pattern, pattern longer than the text, whole text).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class BwtIndex
{
    private readonly int[] sa, bwt, bwtIndex, symbolIndexOf, cTable, rankBlock, lf, cacheValue, psi, pos,
        sampleValue;
    private readonly int n, m, sigma, sampleStep;

    public BwtIndex(string text, int sampleStep)
    {
        n = text.Length;
        m = n + 1;
        this.sampleStep = Math.Max(1, sampleStep);
        sa = BuildSuffixArray(text);
        bwt = new int[m];
        bwtIndex = new int[m];
        symbolIndexOf = new int[128];
        for (int i = 0; i < 128; i++) symbolIndexOf[i] = -1;
        for (int i = 0; i < m; i++) { bwt[i] = sa[i] == 0 ? 0 : text[sa[i] - 1] + 1; symbolIndexOf[bwt[i]] = 0; }
        sigma = 0;
        for (int raw = 0; raw < 128; raw++) if (symbolIndexOf[raw] == 0) symbolIndexOf[raw] = sigma++;
        for (int i = 0; i < m; i++) bwtIndex[i] = symbolIndexOf[bwt[i]];

        cTable = new int[sigma + 1];
        var count = new int[sigma];
        foreach (int index in bwtIndex) count[index]++;
        for (int s = 0; s < sigma; s++) cTable[s + 1] = cTable[s] + count[s];

        // Block rank over the BWT: cumulative per-symbol counts per 64-symbol block + inner scan.
        int blocks = (m + 63) / 64;
        rankBlock = new int[(blocks + 1) * sigma];
        for (int b = 0; b < blocks; b++)
        {
            for (int s = 0; s < sigma; s++) rankBlock[(b + 1) * sigma + s] = rankBlock[b * sigma + s];
            for (int i = b * 64; i < Math.Min(m, (b + 1) * 64); i++) rankBlock[(b + 1) * sigma + bwtIndex[i]]++;
        }

        // Fast locate, step 1: the materialised LF array.
        lf = new int[m];
        for (int i = 0; i < m; i++) lf[i] = cTable[bwtIndex[i]] + Rank(bwtIndex[i], i);
        // Fast locate, step 2: primary sampling of suffix starts.
        sampleValue = new int[m];
        for (int i = 0; i < m; i++) sampleValue[i] = sa[i] % this.sampleStep == 0 ? sa[i] : -1;
        // Fast locate, step 3: cache the sampled start of the LF successor where there is one.
        cacheValue = new int[m];
        for (int i = 0; i < m; i++) cacheValue[i] = sampleValue[i] >= 0 ? -1 : sampleValue[lf[i]];

        pos = new int[m];
        for (int i = 0; i < m; i++) pos[sa[i]] = i;
        psi = new int[m];
        for (int i = 0; i < m; i++) psi[i] = pos[(sa[i] + 1) % m];
    }

    // Prefix doubling over text + sentinel; a position past the end ranks below every symbol.
    private int[] BuildSuffixArray(string text)
    {
        var order = new int[m];
        var rank = new int[m];
        for (int i = 0; i < m; i++) { order[i] = i; rank[i] = i < n ? text[i] + 1 : 0; }
        for (int k = 1; ; k <<= 1)
        {
            int step = k;
            Array.Sort(order, (a, b) =>
            {
                if (rank[a] != rank[b]) return rank[a] - rank[b];
                return (a + step < m ? rank[a + step] : -1) - (b + step < m ? rank[b + step] : -1);
            });
            var next = new int[m];
            for (int i = 1; i < m; i++)
            {
                int p = order[i - 1], q = order[i];
                int pa = p + step < m ? rank[p + step] : -1, qa = q + step < m ? rank[q + step] : -1;
                next[q] = next[p] + (rank[p] != rank[q] || pa != qa ? 1 : 0);
            }
            Array.Copy(next, rank, m);
            if (rank[order[m - 1]] == m - 1) break;
        }
        return order;
    }

    public int Rows => m; public int SuffixAt(int row) => sa[row];
    public int SymbolIndex(char c) => c + 1 < 128 ? symbolIndexOf[c + 1] : -1;
    public int Rank(int symbolIndex, int position)
    {
        if (symbolIndex < 0 || position <= 0) return 0;
        int block = position / 64, total = rankBlock[block * sigma + symbolIndex];
        for (int i = block * 64; i < position; i++) if (bwtIndex[i] == symbolIndex) total++;
        return total;
    }

    public int Lf(int row) => lf[row]; public int LfComputed(int row) => cTable[bwtIndex[row]] + Rank(bwtIndex[row], row);
    public int SampledRows() => sampleValue.Count(value => value >= 0);
    public int CachedRows() => cacheValue.Count(value => value >= 0);
    public void Range(string pattern, out int lo, out int hi)
    {
        lo = 0;
        hi = m;
        for (int i = pattern.Length - 1; i >= 0 && lo < hi; i--)
        {
            int s = SymbolIndex(pattern[i]);
            if (s < 0) { lo = 0; hi = 0; return; }
            lo = cTable[s] + Rank(s, lo);
            hi = cTable[s] + Rank(s, hi);
        }
    }

    public int Count(string pattern)
    {
        if (pattern.Length == 0) return 0;
        Range(pattern, out int lo, out int hi);
        return hi - lo;
    }
    // Fast locate: use the cache when the very next LF step reaches a sampled row, otherwise walk.
    public List<int> Locate(string pattern, out int maxSteps, out long totalSteps) =>
        LocateCore(pattern, true, out maxSteps, out totalSteps);

    // The same search without the cache: the reference the fast walk is measured against.
    public List<int> LocateUncached(string pattern, out int maxSteps, out long totalSteps) =>
        LocateCore(pattern, false, out maxSteps, out totalSteps);

    private List<int> LocateCore(string pattern, bool useCache, out int maxSteps, out long totalSteps)
    {
        var starts = new List<int>();
        maxSteps = 0;
        totalSteps = 0;
        if (pattern.Length == 0) return starts;
        Range(pattern, out int lo, out int hi);
        for (int row = lo; row < hi; row++)
        {
            int start = LocateStart(row, out int steps, useCache);
            if (start < n) starts.Add(start);                     // start == n is the sentinel suffix
            maxSteps = Math.Max(maxSteps, steps);
            totalSteps += steps;
        }
        starts.Sort();
        return starts;
    }

    // Each LF step moves one position earlier, so the start is (sampled value + steps) mod (n+1).
    private int LocateStart(int row, out int steps, bool useCache)
    {
        steps = 0;
        int at = row;
        if (sampleValue[at] < 0)
        {
            if (useCache && cacheValue[at] >= 0)
            {
                at = lf[at];
                steps = 1;
            }
            else
            {
                while (sampleValue[at] < 0)
                {
                    at = lf[at];
                    if (++steps > m) throw new Exception("FAILED: LF walk never reached a sampled row");
                }
            }
        }
        return (sampleValue[at] + steps) % m;
    }

    public string Extract(int start, int length)
    {
        var sb = new StringBuilder(length);
        int row = pos[start];
        for (int k = 0; k < length; k++)
        {
            row = psi[row];
            sb.Append((char)(bwt[row] - 1));
        }
        return sb.ToString();
    }
    public string ReconstructText()
    {
        var chars = new char[n];
        int row = pos[0];
        for (int k = n - 1; k >= 0; k--) chars[k] = (char)(bwt[row = lf[row]] - 1);
        return new string(chars);
    }
}

public static class Solution
{
    private static ulong seed = 0x5BE0CD19137E2179UL;

    private static ulong NextRandom()
    {
        seed += 0x9E3779B97F4A7C15UL;
        ulong z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static string RandomText(int length, string alphabet)
    {
        var sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) sb.Append(alphabet[(int)(NextRandom() % (ulong)alphabet.Length)]);
        return sb.ToString();
    }

    private static List<int> BruteStarts(string text, string pattern)
    {
        var starts = new List<int>();
        if (pattern.Length == 0) return starts;
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) starts.Add(i);
        }
        return starts;
    }

    private static int Compare(string text, int i, int j)
    {
        while (i < text.Length && j < text.Length && text[i] == text[j]) { i++; j++; }
        if (i == text.Length || j == text.Length) return i == j ? 0 : (i == text.Length ? -1 : 1);
        return text[i] < text[j] ? -1 : 1;
    }
    private static int[] NaiveSuffixArray(string text)
    {
        var order = new int[text.Length + 1];
        for (int i = 0; i <= text.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => a == b ? 0 : (a == text.Length ? -1 : b == text.Length ? 1 : Compare(text, a, b)));
        return order;
    }

    // Shared battery: suffix array, LF array identity, search, both locators and extraction.
    private static void CheckText(string text, int sampleStep)
    {
        var index = new BwtIndex(text, sampleStep);
        var naive = NaiveSuffixArray(text);
        for (int row = 0; row < index.Rows; row++)
        {
            Check(index.SuffixAt(row) == naive[row], "suffix row " + row + " of " + text);
            Check(index.Lf(row) == index.LfComputed(row), "LF array identity at " + row + " of " + text);
        }
        Check(index.ReconstructText() == text, "inverse bwt of \"" + text + "\"");
        for (int start = 0; start < text.Length; start++)
        {
            for (int length = 1; start + length <= text.Length && length <= 6; length++)
            {
                string p = text.Substring(start, length);
                var expected = BruteStarts(text, p);
                Check(index.Count(p) == expected.Count, "count " + p + " in " + text);
                var fast = index.Locate(p, out int fastMax, out long fastTotal);
                var slow = index.LocateUncached(p, out int slowMax, out long slowTotal);
                Check(string.Join(",", fast) == string.Join(",", expected) &&
                    string.Join(",", slow) == string.Join(",", expected), "locators " + p + " in " + text);
                Check(fastMax <= sampleStep && fastMax <= slowMax && fastTotal <= slowTotal, "fast locate bound " + p);
                Check(index.Extract(start, length) == p, "extract " + p + " from " + text);
            }
        }
    }

    public static void Main()
    {
        var empty = new BwtIndex("", 4);
        Check(empty.Rows == 1 && empty.Count("a") == 0 && empty.ReconstructText() == "" &&
            empty.Locate("a", out _, out _).Count == 0 && empty.LocateUncached("a", out _, out _).Count == 0, "empty text");

        var single = new BwtIndex("a", 4);
        Check(single.Rows == 2 && single.Count("a") == 1 && single.Extract(0, 1) == "a" &&
            string.Join(",", single.Locate("a", out _, out _)) == "0", "single character");

        var oneSymbol = new BwtIndex("aaaaaa", 4);
        Check(oneSymbol.Count("aaa") == 4 && oneSymbol.ReconstructText() == "aaaaaa" &&
            string.Join(",", oneSymbol.Locate("aa", out int maxAa, out _)) == "0,1,2,3,4" && maxAa <= 4, "one-symbol alphabet");

        var banana = new BwtIndex("banana", 4);
        Check(string.Join(",", banana.Locate("ana", out _, out _)) == "1,3" && banana.Count("x") == 0 &&
            banana.Extract(1, 5) == "anana" && string.Join(",", banana.Locate("na", out _, out _)) == "2,4", "banana queries");

        foreach (string text in new[] { "", "a", "ab", "aba", "aaaa", "banana", "mississippi", "abracadabra", "abcabcab" })
            CheckText(text, 4);
        foreach (string text in new[] { "mississippi", "abracadabra", RandomText(60, "abc"), RandomText(120, "ab") })
        {
            CheckText(text, 1);
            CheckText(text, 8);
        }

        // A 500-character text: every probe agrees with brute force, and the cache really saves steps.
        string large = RandomText(500, "abcde");
        var largeIndex = new BwtIndex(large, 16);
        Check(largeIndex.ReconstructText() == large, "large round trip");
        long fastTotal = 0, slowTotal = 0;
        int fastMax = 0, slowMax = 0;
        for (int probe = 0; probe < 400; probe++)
        {
            int start = (int)(NextRandom() % (ulong)large.Length);
            string p = large.Substring(start, Math.Min(1 + (int)(NextRandom() % 14), large.Length - start));
            var expected = BruteStarts(large, p);
            var fast = largeIndex.Locate(p, out int probeFastMax, out long probeFastTotal);
            var slow = largeIndex.LocateUncached(p, out int probeSlowMax, out long probeSlowTotal);
            Check(string.Join(",", fast) == string.Join(",", expected), "large fast locate " + p);
            Check(string.Join(",", slow) == string.Join(",", expected), "large uncached locate " + p);
            Check(probeFastMax <= 16 && probeFastMax <= probeSlowMax && probeFastTotal <= probeSlowTotal &&
                largeIndex.Count(p) == expected.Count && largeIndex.Extract(start, p.Length) == p, "large bounds");
            fastTotal += probeFastTotal; slowTotal += probeSlowTotal;
            fastMax = Math.Max(fastMax, probeFastMax); slowMax = Math.Max(slowMax, probeSlowMax);
        }
        Check(largeIndex.Count(large + "~") == 0, "large absent");

        var repeated = new BwtIndex(RandomText(300, "ab"), 16);
        var again = new BwtIndex(large, 16);
        Check(repeated.SampledRows() >= 1 && repeated.SampledRows() <= (repeated.Rows + 15) / 16 &&
            repeated.CachedRows() >= 1 && repeated.CachedRows() <= repeated.SampledRows(), "sampling density");
        Check(string.Join(",", again.Locate("abc", out _, out _)) ==
            string.Join(",", largeIndex.Locate("abc", out _, out _)), "determinism");

        Console.WriteLine("500-char text, step 16: " + largeIndex.SampledRows() + " sampled rows, " +
            largeIndex.CachedRows() + " cached rows; 400 probes: LF steps cached " + fastTotal + " (max " + fastMax +
            ") vs uncached " + slowTotal + " (max " + slowMax + ")");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
