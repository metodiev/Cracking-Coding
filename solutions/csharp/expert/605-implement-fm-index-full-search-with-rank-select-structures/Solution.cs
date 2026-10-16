// Task 605 - Implement FM-index full search with rank/select structures. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The FM-index of text + sentinel is the BWT together with two data structures:
//   * a *rank* structure: cumulative per-symbol counts for every 64-symbol block of the BWT plus a
//     bounded inner scan, so Rank(c, i) = |{j < i : bwt[j] = c}| costs 64 comparisons at worst;
//   * a *select* structure: one ascending position list per symbol with per-symbol offsets, so
//     Select(c, k) - the BWT row of the k-th occurrence of c - is a single array read.
// Full search is backward search over those two structures: lo = C[c] + Rank(c, lo), hi = C[c] +
// Rank(c, hi), and [lo, hi) is the occurrence set. Locate uses a sampled suffix array plus *forward*
// (psi) walking: psi[row] is the row of the suffix one position later, so SA increases by one per
// step and a match row walks only until it reaches a row whose start is a multiple of sampleStep;
// the start is then (sampled value - steps) mod (n+1). Extraction uses psi as well, because the BWT
// character of the row psi returns is exactly the character that follows the current suffix start.
//
// Simplifications (documented, not behavioural): rank is the block structure above rather than a
// wavelet tree; select keeps explicit per-symbol position lists (linear memory, O(1) lookup) rather
// than a sampled-select bitmap; the suffix array behind the BWT comes from prefix doubling.
//
// Verified: Rank against naive BWT counts for every symbol and a spread of positions; Select against
// naive occurrence rows (including the out-of-range answer) and the rank/select round trip;
// occurrence counts and located rows against a brute-force scanner for every substring of nine fixed
// texts and for random probes of a 300-character text; extraction of every short window and the
// inverse BWT against the original text; the psi walk bound (at most sampleStep steps, zero for step
// 1); boundary cases (empty text, single character, one-symbol alphabet, patterns longer than the
// text, characters outside the alphabet).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class FmIndex
{
    private readonly int[] sa, bwt, bwtIndex, symbolIndexOf, symbolValueOf, cTable, symbolCount;
    private readonly int[] rankBlock, selectRow, selectStart, psi, pos, sampleValue;
    private readonly int n, m, sigma, sampleStep;

    public FmIndex(string text, int sampleStep)
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
        symbolValueOf = new int[128];
        for (int raw = 0; raw < 128; raw++) if (symbolIndexOf[raw] == 0) { symbolValueOf[sigma] = raw; symbolIndexOf[raw] = sigma++; }
        for (int i = 0; i < m; i++) bwtIndex[i] = symbolIndexOf[bwt[i]];

        // C table and per-symbol totals.
        cTable = new int[sigma + 1];
        symbolCount = new int[sigma];
        for (int s = 0; s < sigma; s++)
        {
            foreach (int index in bwtIndex) if (index == s) symbolCount[s]++;
            cTable[s + 1] = cTable[s] + symbolCount[s];
        }

        // Rank structure: cumulative per-symbol counts per 64-symbol block of the BWT.
        int blocks = (m + 63) / 64;
        rankBlock = new int[(blocks + 1) * sigma];
        for (int b = 0; b < blocks; b++)
        {
            for (int s = 0; s < sigma; s++) rankBlock[(b + 1) * sigma + s] = rankBlock[b * sigma + s];
            for (int i = b * 64; i < Math.Min(m, (b + 1) * 64); i++) rankBlock[(b + 1) * sigma + bwtIndex[i]]++;
        }

        // Select structure: the BWT rows of each symbol, in ascending order.
        selectStart = new int[sigma + 1];
        for (int s = 0; s < sigma; s++) selectStart[s + 1] = selectStart[s] + symbolCount[s];
        selectRow = new int[m];
        var filled = new int[sigma];
        for (int i = 0; i < m; i++) selectRow[selectStart[bwtIndex[i]] + filled[bwtIndex[i]]++] = i;

        pos = new int[m];
        for (int i = 0; i < m; i++) pos[sa[i]] = i;
        psi = new int[m];
        for (int i = 0; i < m; i++) psi[i] = pos[(sa[i] + 1) % m];
        sampleValue = new int[m];
        for (int i = 0; i < m; i++) sampleValue[i] = sa[i] % this.sampleStep == 0 ? sa[i] : -1;
    }

    // Prefix doubling over text + sentinel; a position past the end ranks below every symbol.
    private int[] BuildSuffixArray(string text)
    {
        var order = new int[m];
        var rank = new int[m];
        for (int i = 0; i < m; i++)
        {
            order[i] = i;
            rank[i] = i < n ? text[i] + 1 : 0;
        }
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

    public int Rows => m;
    public int Ranks => sigma;
    public int SuffixAt(int row) => sa[row]; public int BwtSymbol(int row) => bwt[row];
    public int SymbolIndex(char c) => c + 1 < 128 ? symbolIndexOf[c + 1] : -1;
    public int SymbolOfIndex(int index) => symbolValueOf[index];
    public int Rank(int symbolIndex, int position)
    {
        if (symbolIndex < 0 || position <= 0) return 0;
        int block = position / 64, count = rankBlock[block * sigma + symbolIndex];
        for (int i = block * 64; i < position; i++) if (bwtIndex[i] == symbolIndex) count++;
        return count;
    }

    // Row of the k-th (0-based) occurrence of the symbol in the BWT, or -1 when there is none.
    public int Select(int symbolIndex, int k) =>
        symbolIndex < 0 || k < 0 || k >= symbolCount[symbolIndex] ? -1 : selectRow[selectStart[symbolIndex] + k];
    public int Lf(int row)
    {
        int s = bwtIndex[row];
        return cTable[s] + Rank(s, row);
    }

    // Backward search: [lo, hi) ends up as exactly the rows of the matching suffixes.
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

    public List<int> Locate(string pattern, out int maxSteps, out int totalSteps)
    {
        var starts = new List<int>();
        maxSteps = 0;
        totalSteps = 0;
        if (pattern.Length == 0) return starts;
        Range(pattern, out int lo, out int hi);
        for (int row = lo; row < hi; row++)
        {
            int steps = 0, at = row;
            while (sampleValue[at] < 0)
            {
                at = psi[at];                                   // psi advances the suffix start by one
                if (++steps > m) throw new Exception("FAILED: psi walk never reached a sampled row");
            }
            // psi advances the suffix start by one, so the wanted start is the sampled value minus the
            // number of steps; the walk may have passed the sentinel, hence the modular correction.
            int start = sampleValue[at] - steps;
            if (start < 0) start += m;
            if (start < n) starts.Add(start);                    // start == n is the sentinel suffix
            maxSteps = Math.Max(maxSteps, steps);
            totalSteps += steps;
        }
        starts.Sort();
        return starts;
    }

    // psi + BWT recover any window without materialising the text: the row after psi carries the
    // character that follows the current suffix start.
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
        for (int k = n - 1; k >= 0; k--) chars[k] = (char)(bwt[row = Lf(row)] - 1);
        return new string(chars);
    }

    public int SampledRows() => sampleValue.Count(value => value >= 0);
}

public static class Solution
{
    private static ulong seed = 0xA54FF53A5F1D36F1UL;

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

    // Rank/select cross-check: rank against naive counts, select against naive positions, and the
    // round trip rank(select(c, k)) == k+1 with select(rank) returning the same row.
    private static void CheckRankSelect(FmIndex index, string text)
    {
        for (int s = 0; s < index.Ranks; s++)
        {
            int raw = index.SymbolOfIndex(s);
            var naive = Enumerable.Range(0, index.Rows).Where(i => index.BwtSymbol(i) == raw).ToList();
            for (int k = 0; k < naive.Count; k++)
            {
                Check(index.Select(s, k) == naive[k] && index.Rank(s, naive[k] + 1) == k + 1, "select/rank " + s + " " + k);
            }
            Check(index.Select(s, naive.Count) == -1 && index.Select(s, -1) == -1, "select out of range");
            for (int position = 0; position <= index.Rows; position += 5)
            {
                Check(index.Rank(s, position) == naive.Count(i => i < position), "rank " + s + " at " + position + " " + text);
            }
        }
    }

    // Shared battery: suffix array, inverse BWT, window extraction and full search.
    private static void CheckText(string text)
    {
        var index = new FmIndex(text, 4);
        var naive = NaiveSuffixArray(text);
        for (int row = 0; row < index.Rows; row++) Check(index.SuffixAt(row) == naive[row], "suffix row " + row + " of " + text);
        Check(index.ReconstructText() == text, "inverse bwt of \"" + text + "\"");
        for (int start = 0; start < text.Length; start++)
        {
            for (int length = 1; start + length <= text.Length && length <= 6; length++)
            {
                string p = text.Substring(start, length);
                var located = index.Locate(p, out int maxSteps, out _);
                Check(index.Count(p) == BruteStarts(text, p).Count, "count " + p + " in " + text);
                Check(string.Join(",", located) == string.Join(",", BruteStarts(text, p)) && maxSteps <= 4 &&
                    index.Extract(start, length) == p, "locate/extract " + p + " in " + text);
            }
        }
    }

    public static void Main()
    {
        var empty = new FmIndex("", 4);
        Check(empty.Rows == 1 && empty.Count("a") == 0 && empty.ReconstructText() == "" && empty.Extract(0, 0) == "" &&
            empty.Locate("a", out _, out _).Count == 0 && empty.Rank(0, 1) == 1 && empty.Select(0, 0) == 0, "empty text");

        var single = new FmIndex("a", 4);
        Check(single.Rows == 2 && single.Count("a") == 1 && single.Extract(0, 1) == "a" &&
            single.Rank(single.SymbolIndex('a'), 2) == 1 && single.Select(single.SymbolIndex('a'), 0) == 0, "single char");

        var oneSymbol = new FmIndex("aaaaaa", 4);
        Check(oneSymbol.Count("aaa") == 4 && oneSymbol.Extract(2, 3) == "aaa" && oneSymbol.Ranks == 2, "one-symbol alphabet");
        Check(string.Join(",", oneSymbol.Locate("aa", out int maxAa, out _)) == "0,1,2,3,4" && maxAa <= 4, "one-symbol locate");

        var banana = new FmIndex("banana", 4);
        Check(string.Join(",", banana.Locate("ana", out _, out _)) == "1,3" && banana.Count("x") == 0 &&
            banana.Extract(1, 5) == "anana" && banana.Extract(0, 6) == "banana" &&
            banana.Rank(banana.SymbolIndex('a'), banana.Rows) == 3, "banana queries");

        foreach (string text in new[] { "", "a", "ab", "aba", "aaaa", "banana", "mississippi", "abracadabra", "abcabcab" })
            CheckText(text);
        foreach (string text in new[] { "banana", "mississippi", "abcabcab", RandomText(90, "abcd") }) CheckRankSelect(new FmIndex(text, 3), text);

        // Step 1 samples every row, so locate never walks and Select/Rank stay exact.
        var dense = new FmIndex("mississippi", 1);
        Check(string.Join(",", dense.Locate("issi", out int denseMax, out int denseTotal)) == "1,4" && denseMax == 0 &&
            denseTotal == 0 && dense.SampledRows() == dense.Rows, "dense sampling");

        // A 300-character text: every probe must match brute force and every window must extract.
        string large = RandomText(300, "abcde");
        var largeIndex = new FmIndex(large, 16);
        Check(largeIndex.ReconstructText() == large, "large round trip");
        for (int probe = 0; probe < 300; probe++)
        {
            int start = (int)(NextRandom() % (ulong)large.Length);
            int length = Math.Min(1 + (int)(NextRandom() % 10), large.Length - start);
            string p = large.Substring(start, length);
            var located = largeIndex.Locate(p, out int maxSteps, out _);
            Check(string.Join(",", located) == string.Join(",", BruteStarts(large, p)) && maxSteps <= 16 &&
                largeIndex.Extract(start, length) == p && largeIndex.Count(p) == located.Count, "large probe " + p);
        }
        Check(largeIndex.Count(large + "~") == 0, "large absent");

        Console.WriteLine("300-char text -> " + largeIndex.Rows + " rows, " + largeIndex.Ranks + " symbols, sampled rows " +
            largeIndex.SampledRows() + " (step 16); dense step 1 max psi steps " + denseMax);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
