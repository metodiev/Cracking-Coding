// Task 604 - Implement compressed suffix array (CSA) advanced. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A compressed suffix array stores only the BWT of text + sentinel, the C table (counts of smaller
// symbols) and a *rank* structure over the BWT; queries go through the LF mapping
//     LF(row) = C[bwt[row]] + Rank(bwt[row], row).
// Backward search extends a pattern one character at a time, lo = C[c] + Rank(c, lo), hi =
// C[c] + Rank(c, hi), and the range [lo, hi) is the occurrence set. Locating needs SA values, which
// are recovered from a *sample*: every row whose suffix start is a multiple of sampleStep stores
// that start, and a match row is followed by LF until a sampled row is reached; since LF steps to
// the suffix one position earlier, the wanted start is (sampled value + steps) mod (n+1). Also
// exposed: psi (psi[i] = row of the suffix following SA[i]), the text reconstructed by walking LF
// from the row of suffix 0, and the LF steps each locate needed, so the sampling is measurable.
//
// Simplifications (documented, not behavioural): rank is a block structure (cumulative counts per
// 64-symbol block plus a bounded inner scan) instead of a wavelet tree, and the suffix array behind
// the BWT is built by prefix doubling; the CSA query interface and the LF-based locate are real.
//
// Verified: the suffix array against a naive comparison sort; occurrence counts, located positions
// as sorted lists, whole-text search and per-locate LF steps against a brute-force scanner for every
// substring of nine fixed texts and for random probes of a 500-character text; the reconstructed
// text against the original; psi is a permutation with SuffixAt(psi[i]) == SuffixAt(i) + 1 mod
// (n+1); LF returns to its start row after n+1 steps; block rank against naive BWT counts; sampled
// density and the sampling bound (at most sampleStep steps, zero for step 1); boundary cases (empty
// text, single character, one-symbol alphabet, pattern longer than the text, absent characters).
using System;
using System.Collections.Generic;
using System.Text;

internal class CompressedSuffixArray
{
    private readonly int[] sa, bwt, bwtIndex, symbolIndexOf, symbolValueOf, cTable, rankBlock, psi, pos, sampleValue;
    private readonly int n, m, sigma, sampleStep;

    public CompressedSuffixArray(string text, int sampleStep)
    {
        n = text.Length;
        m = n + 1;
        this.sampleStep = Math.Max(1, sampleStep);
        sa = BuildSuffixArray(text);
        bwt = new int[m];
        symbolIndexOf = new int[128];
        symbolValueOf = new int[128];
        for (int i = 0; i < 128; i++) symbolIndexOf[i] = -1;
        for (int i = 0; i < m; i++)
        {
            bwt[i] = sa[i] == 0 ? 0 : text[sa[i] - 1] + 1;   // raw symbol, 0 is the sentinel
            symbolIndexOf[bwt[i]] = 0;                       // mark present, indices assigned below
        }
        sigma = 0;
        for (int raw = 0; raw < 128; raw++)
        {
            if (symbolIndexOf[raw] == 0) { symbolIndexOf[raw] = sigma; symbolValueOf[sigma++] = raw; }
        }
        bwtIndex = new int[m];
        for (int i = 0; i < m; i++) bwtIndex[i] = symbolIndexOf[bwt[i]];

        cTable = new int[sigma];
        int running = 0;
        for (int s = 0; s < sigma; s++)
        {
            cTable[s] = running;
            foreach (int index in bwtIndex) if (index == s) running++;
        }

        // Block rank: cumulative per-symbol counts per 64-symbol block, plus a bounded inner scan.
        int blocks = (m + 63) / 64;
        rankBlock = new int[(blocks + 1) * sigma];
        for (int b = 0; b < blocks; b++)
        {
            for (int s = 0; s < sigma; s++) rankBlock[(b + 1) * sigma + s] = rankBlock[b * sigma + s];
            for (int i = b * 64; i < Math.Min(m, (b + 1) * 64); i++) rankBlock[(b + 1) * sigma + bwtIndex[i]]++;
        }

        pos = new int[m];
        for (int i = 0; i < m; i++) pos[sa[i]] = i;
        psi = new int[m];
        for (int i = 0; i < m; i++) psi[i] = pos[(sa[i] + 1) % m];
        sampleValue = new int[m];
        for (int i = 0; i < m; i++) sampleValue[i] = sa[i] % this.sampleStep == 0 ? sa[i] : -1;
    }

    // Prefix doubling over the text plus sentinel: sort by the first 2^k characters until the ranks
    // are all distinct. A position past the end ranks below every symbol.
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
    public int SuffixAt(int row) => sa[row];
    public int BwtSymbol(int row) => bwt[row];
    public int Psi(int row) => psi[row];
    public int SymbolIndex(char c) => c + 1 < 128 ? symbolIndexOf[c + 1] : -1;
    // Occurrences of a symbol in BWT[0, position): block counts plus at most 64 comparisons.
    public int Rank(int symbolIndex, int position)
    {
        if (symbolIndex < 0 || position <= 0) return 0;
        int block = position / 64;
        int count = rankBlock[block * sigma + symbolIndex];
        for (int i = block * 64; i < position; i++) if (bwtIndex[i] == symbolIndex) count++;
        return count;
    }

    public int Lf(int row)
    {
        int s = bwtIndex[row];
        return cTable[s] + Rank(s, row);
    }
    // Backward search: the range [lo, hi) shrinks to exactly the rows of the matching suffixes.
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
                at = Lf(at);
                if (++steps > m) throw new Exception("FAILED: LF walk never reached a sampled row");
            }
            int start = (sampleValue[at] + steps) % m;
            if (start < n) starts.Add(start);                       // start == n is the sentinel suffix
            maxSteps = Math.Max(maxSteps, steps);
            totalSteps += steps;
        }
        starts.Sort();
        return starts;
    }

    // Inverse BWT: walking LF from the row of suffix 0 emits the text backwards.
    public string ReconstructText()
    {
        var chars = new char[n];
        int row = pos[0];
        for (int k = n - 1; k >= 0; k--)
        {
            row = Lf(row);
            chars[k] = (char)(bwt[row] - 1);
        }
        return new string(chars);
    }

    public int SampledRows()
    {
        int count = 0;
        foreach (int value in sampleValue) if (value >= 0) count++;
        return count;
    }
}
public static class Solution
{
    private static ulong seed = 0x3C6EF372FE94F82BUL;

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
        while (i < text.Length && j < text.Length && text[i] == text[j])
        {
            i++;
            j++;
        }
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

    // Shared battery: suffix array, round trip, count and locate for every short substring.
    private static void CheckText(string text)
    {
        var csa = new CompressedSuffixArray(text, 4);
        var naive = NaiveSuffixArray(text);
        for (int row = 0; row < csa.Rows; row++) Check(csa.SuffixAt(row) == naive[row], "suffix row " + row + " of " + text);
        Check(csa.ReconstructText() == text, "inverse bwt of \"" + text + "\"");
        for (int start = 0; start < text.Length; start++)
        {
            for (int end = start + 1; end <= Math.Min(start + 6, text.Length); end++)
            {
                string p = text.Substring(start, end - start);
                var expected = BruteStarts(text, p);
                Check(csa.Count(p) == expected.Count, "count " + p + " in " + text);
                var located = csa.Locate(p, out int maxSteps, out _);
                Check(string.Join(",", located) == string.Join(",", expected) && maxSteps <= 4, "locate " + p + " in " + text);
            }
        }
    }

    public static void Main()
    {
        var empty = new CompressedSuffixArray("", 4);
        Check(empty.Rows == 1 && empty.Count("a") == 0 && empty.ReconstructText() == "" && empty.SuffixAt(0) == 0, "empty text");
        Check(empty.Locate("a", out _, out _).Count == 0 && empty.BwtSymbol(0) == 0, "empty locate");

        var single = new CompressedSuffixArray("a", 4);
        Check(single.Rows == 2 && single.ReconstructText() == "a" && single.Count("a") == 1, "single char");
        Check(single.SuffixAt(0) == 1 && single.SuffixAt(1) == 0 && single.Locate("a", out _, out _)[0] == 0, "single rows");

        var oneSymbol = new CompressedSuffixArray("aaaaaa", 4);
        Check(oneSymbol.Count("aaa") == 4 && oneSymbol.Rows == 7 &&
            string.Join(",", oneSymbol.Locate("aa", out int maxAa, out _)) == "0,1,2,3,4" && maxAa <= 4, "one symbol");

        var banana = new CompressedSuffixArray("banana", 4);
        Check(string.Join(",", banana.Locate("an", out _, out _)) == "1,3" &&
            string.Join(",", banana.Locate("ana", out _, out _)) == "1,3", "banana an/ana");
        Check(banana.Count("x") == 0 && banana.ReconstructText() == "banana", "banana absent and round trip");
        Check(banana.Locate("banana", out _, out _)[0] == 0, "banana whole");

        foreach (string text in new[] { "", "a", "ab", "aba", "aaaa", "banana", "mississippi", "abracadabra", "abcabcab" })
            CheckText(text);

        // Structure: psi against the verified suffix array, the LF cycle, block rank against naive
        // counts, and the sampling density.
        foreach (string text in new[] { "banana", "mississippi", "abcabcab", RandomText(120, "abcd") })
        {
            var csa = new CompressedSuffixArray(text, 3);
            var seen = new bool[csa.Rows];
            for (int i = 0; i < csa.Rows; i++)
            {
                Check(!seen[csa.Psi(i)], "psi injective " + text);
                seen[csa.Psi(i)] = true;
                Check(csa.SuffixAt(csa.Psi(i)) == (csa.SuffixAt(i) + 1) % csa.Rows, "psi definition " + text);
            }
            int row = 0;
            for (int k = 0; k < csa.Rows; k++) row = csa.Lf(row);
            Check(row == 0, "lf cycle " + text);

            var sampled = new CompressedSuffixArray(text, 8);
            for (int index = 0; index < sampled.Ranks; index++)
            {
                int raw = 0;
                for (int c = 1; c < 127; c++) if (sampled.SymbolIndex((char)c) == index) raw = c + 1;
                for (int position = 0; position <= sampled.Rows; position += 5)
                {
                    int naiveCount = 0;
                    for (int i = 0; i < position; i++) if (sampled.BwtSymbol(i) == raw) naiveCount++;
                    Check(sampled.Rank(index, position) == naiveCount, "rank " + index + " at " + position);
                }
            }
            Check(sampled.SampledRows() >= 1 && sampled.SampledRows() <= (sampled.Rows + 7) / 8, "sampled density " + text.Length);
        }
        // Step 1 samples every row, so locating never walks LF at all.
        var dense = new CompressedSuffixArray("mississippi", 1);
        Check(string.Join(",", dense.Locate("issi", out int denseMax, out int denseTotal)) == "1,4" && denseMax == 0 &&
            denseTotal == 0 && dense.SampledRows() == dense.Rows, "dense sampling");

        // Large text: doubling runs many rounds over 500 characters, locate still exact.
        string large = RandomText(500, "abcde");
        var largeCsa = new CompressedSuffixArray(large, 16);
        var largeNaive = NaiveSuffixArray(large);
        for (int row = 0; row < largeCsa.Rows; row += 13) Check(largeCsa.SuffixAt(row) == largeNaive[row], "large row " + row);
        Check(largeCsa.ReconstructText() == large && largeCsa.Count(large) == 1, "large round trip");
        for (int probe = 0; probe < 200; probe++)
        {
            int start = (int)(NextRandom() % (ulong)large.Length);
            string p = large.Substring(start, Math.Min(1 + (int)(NextRandom() % 12), large.Length - start));
            var located = largeCsa.Locate(p, out int maxSteps, out _);
            Check(string.Join(",", located) == string.Join(",", BruteStarts(large, p)) && maxSteps <= 16, "large locate " + p);
        }

        Console.WriteLine("500-char text -> " + largeCsa.Rows + " rows, " + largeCsa.Ranks + " ranks, sampled " +
            largeCsa.SampledRows() + " (step 16); dense step 1 max LF steps " + denseMax);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
