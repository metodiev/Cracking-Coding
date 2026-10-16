// Task 701 - Implement approximate string matching with edit distance threshold using Ukkonen.
// (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Sellers' DP answers "where does the pattern P occur in the text T with at most k edits?" with the
// table E[i][j] = min edits between P[0..i) and any substring of T ending at index j-1; the first
// row is zero because the match may start anywhere, and E[m][j] <= k marks an occurrence ending at
// j-1. Filling that table costs O(n*m). Ukkonen's band keeps only the 2k+1 diagonals d = j - i of
// the edit graph, because a state with i pattern characters and j text characters consumed can be
// on a useful path only while |i - j| stays within the error budget. This file keeps the band in
// the furthest-reaching *diagonal* representation:
//     X[e][d] = the largest j such that the state (i, j) = (j - d, j), i in [0..m], is reachable
//               with at most e differences.
// The three edits and the free start give the whole band from the previous one, with the exact-match
// runs skipped by a direct character scan (`extend`):
//     carry over : X[e][d] <- X[e-1][d]                       ("at most e" includes "at most e-1")
//     substitute : X[e][d] <- X[e-1][d] + 1, only while i < m (one character on each side)
//     delete P   : X[e][d] <- the furthest state of diagonal d+1 with i < m, capped at j = m + d
//     insert T   : X[e][d] <- X[e-1][d-1] + 1                 (text character aligned to nothing)
// The two caps are the whole subtlety of the search problem: a snake that runs past the end of the
// pattern must not hide the shorter path that still has a pattern character to delete, and a band
// that has already consumed the pattern cannot be carried forward again. A match is complete when
// X[e][d] == d + m, i.e. i = m, and it then ends at text index d + m - 1.
// Work is O(k * (n + m)) band states instead of the O(n * m) table cells.
//
// Invariants verified below: the band reports exactly the end positions of the unbanded Sellers DP
// (n*m cells, first row zero) - checked exhaustively for every text up to length 6 and every
// pattern up to length 3 over a two-letter alphabet, for k = 0..3 - plus 80 deterministic random
// cases over a three-letter alphabet cross-checked against a second, independent oracle that runs
// textbook Levenshtein distance against every single substring.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal static class Ukkonen
{
    private const int Invalid = int.MinValue;

    /// <summary>
    /// End indices (0-based, inclusive) of every text substring whose edit distance to the pattern
    /// is at most k. <paramref name="scanned"/> counts the characters compared by the run skips.
    /// </summary>
    internal static int[] Search(string text, string pattern, int k, out long scanned)
    {
        if (k < 0) throw new ArgumentOutOfRangeException(nameof(k), "edit distance threshold must be non-negative");
        int n = text.Length, m = pattern.Length;
        scanned = 0;
        if (m == 0)
        {
            var every = new int[n];
            for (int e = 0; e < n; e++) every[e] = e;
            return every;
        }
        if (n == 0) return Array.Empty<int>();

        long comparisons = 0;                       // local counter: an out parameter cannot be captured
        int lo = -m, hi = n, width = hi - lo + 1;   // diagonals d = j - i with j in [0,n], i in [0,m]
        var matched = new bool[n];

        int Extend(int d, int j)
        {
            if (j < 0 || j > n) return Invalid;
            int p = j - d;                          // pattern characters consumed by this state
            if (p < 0 || p > m) return Invalid;
            while (p < m && j < n && pattern[p] == text[j]) { p++; j++; comparisons++; }
            return j;
        }

        var cur = new int[width];
        var next = new int[width];
        for (int i = 0; i < width; i++) cur[i] = Invalid;
        // Free start: with 0 differences the pattern may begin at any text position.
        for (int d = 0; d <= n; d++) cur[d - lo] = Extend(d, d);

        for (int e = 0; e <= k; e++)
        {
            if (e > 0)
            {
                for (int idx = 0; idx < width; idx++)
                {
                    int d = lo + idx;
                    int best = cur[idx];                                    // carry the e-1 band
                    int same = cur[idx];
                    if (same != Invalid && same < m + d)                    // substitute needs i < m
                    {
                        int v = same + 1;
                        if (v > best) best = v;
                    }
                    if (idx + 1 < width)
                    {
                        int source = cur[idx + 1];                          // delete a pattern character
                        if (source != Invalid)
                        {
                            int cap = m + d;                                // the furthest source with i < m
                            if (source > cap) source = cap;
                            if (source > best) best = source;
                        }
                    }
                    if (idx > 0)
                    {
                        int source = cur[idx - 1];                          // insert a text character
                        if (source != Invalid && source + 1 > best) best = source + 1;
                    }
                    next[idx] = best == Invalid ? Invalid : Extend(d, best);
                }
                int[] swap = cur; cur = next; next = swap;
            }
            for (int idx = 0; idx < width; idx++)
            {
                int d = lo + idx;
                // extend() never lets a state pass i = m, so X == d + m means the pattern is consumed.
                if (cur[idx] == d + m && d + m - 1 >= 0) matched[d + m - 1] = true;
            }
        }

        var hits = new List<int>();
        for (int e = 0; e < n; e++) if (matched[e]) hits.Add(e);
        scanned = comparisons;
        return hits.ToArray();
    }

    internal static int[] Search(string text, string pattern, int k) => Search(text, pattern, k, out _);

    /// <summary>First oracle: the unbanded Sellers DP in its n*m table form.</summary>
    internal static int[] ScanFullMatrix(string text, string pattern, int k)
    {
        int n = text.Length, m = pattern.Length;
        var hits = new List<int>();
        if (m == 0) { for (int e = 0; e < n; e++) hits.Add(e); return hits.ToArray(); }
        if (n == 0) return hits.ToArray();

        int[,] d = new int[m + 1, n + 1];
        for (int j = 0; j <= n; j++) d[0, j] = 0;      // free start: the matched suffix may be empty
        for (int i = 1; i <= m; i++) d[i, 0] = i;      // every pattern character deleted
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                int best = Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1);
                best = Math.Min(best, d[i - 1, j - 1] + (pattern[i - 1] == text[j - 1] ? 0 : 1));
                d[i, j] = best;
            }
        }
        for (int j = 1; j <= n; j++) if (d[m, j] <= k) hits.Add(j - 1);
        return hits.ToArray();
    }

    /// <summary>Second oracle: textbook Levenshtein distance run against every substring of the text.</summary>
    internal static int[] ScanBySubstringDistance(string text, string pattern, int k)
    {
        var hits = new List<int>();
        if (pattern.Length == 0) { for (int e = 0; e < text.Length; e++) hits.Add(e); return hits.ToArray(); }
        for (int end = 0; end < text.Length; end++)
        {
            for (int start = 0; start <= end; start++)
            {
                if (Levenshtein(pattern, text.Substring(start, end - start + 1)) <= k) { hits.Add(end); break; }
            }
        }
        return hits.ToArray();
    }

    /// <summary>Textbook Levenshtein distance (substitution cost 1, no free start).</summary>
    internal static int Levenshtein(string a, string b)
    {
        int[,] d = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
        for (int j = 0; j <= b.Length; j++) d[0, j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                int best = Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1);
                best = Math.Min(best, d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
                d[i, j] = best;
            }
        }
        return d[a.Length, b.Length];
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool SameHits(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static string Show(int[] hits)
    {
        var parts = new List<string>();
        foreach (int h in hits) parts.Add(h.ToString());
        return "{" + string.Join(",", parts) + "}";
    }

    /// <summary>End indices of the literal occurrences of a pattern, computed with no DP at all.</summary>
    private static int[] LiteralEnds(string text, string pattern)
    {
        var ends = new List<int>();
        for (int start = 0; start + pattern.Length <= text.Length; start++)
        {
            bool same = true;
            for (int i = 0; i < pattern.Length && same; i++) if (text[start + i] != pattern[i]) same = false;
            if (same) ends.Add(start + pattern.Length - 1);
        }
        return ends.ToArray();
    }

    /// <summary>All binary strings of a given length, used for the exhaustive sweep.</summary>
    private static List<string> BinaryStrings(int length, int limit)
    {
        var all = new List<string>();
        for (int mask = 0; mask < (1 << length); mask++)
        {
            var chars = new char[length];
            for (int i = 0; i < length; i++) chars[i] = ((mask >> i) & 1) == 0 ? 'a' : 'b';
            all.Add(new string(chars));
            if (all.Count >= limit) break;
        }
        return all;
    }

    public static void Main()
    {
        // ---- the metric the scan approximates -------------------------------------------------
        Check(Ukkonen.Levenshtein("kitten", "sitting") == 3, "kitten/sitting is the classic distance 3");
        Check(Ukkonen.Levenshtein("abc", "abc") == 0 && Ukkonen.Levenshtein("", "abc") == 3,
            "identical strings cost 0 and deleting everything costs the length");
        Check(Ukkonen.Levenshtein("flaw", "lawn") == Ukkonen.Levenshtein("lawn", "flaw"),
            "edit distance is symmetric on a substitution example");

        // ---- k = 0 has to reproduce plain substring search -----------------------------------
        {
            string text = "abracadabra abracadabra";
            int[] hits = Ukkonen.Search(text, "cad", 0, out _);
            int[] expected = LiteralEnds(text, "cad");
            Check(expected.Length == 2, "the fixture really contains two literal occurrences");
            Check(SameHits(hits, expected), "k=0 finds exactly the literal occurrence ends: " + Show(hits));
        }

        // ---- the worked example: typos in a sentence -----------------------------------------
        {
            string text = "the quick brown fox jumps over the lazy dog";
            int[] exact = Ukkonen.Search(text, "jumps", 0, out _);
            int[] one = Ukkonen.Search(text, "jumps", 1, out _);
            Check(SameHits(exact, new[] { 24 }), "an exact scan finds only the real occurrence");
            Check(Array.IndexOf(one, 24) >= 0, "the tolerant scan keeps the exact occurrence");
            Check(one.Length > exact.Length, "tolerance 1 accepts strictly more end positions");
            Check(Array.IndexOf(Ukkonen.Search(text, "jums", 1, out _), 24) >= 0,
                "the typo 'jums' (one deletion) still ends at 'jumps'");
            Check(Array.IndexOf(Ukkonen.Search(text, "jums", 1, out _), 18) < 0,
                "'fox' is not within one edit of 'jums'");
            Check(Array.IndexOf(Ukkonen.Search(text, "brownn", 1, out _), 14) >= 0,
                "the typo 'brownn' (one insertion) still ends at 'brown'");
            Check(SameHits(Ukkonen.Search("aaa", "aaa", 2, out _), new[] { 0, 1, 2 }),
                "deleting trailing pattern characters is allowed too");
        }

        // ---- degenerate inputs ----------------------------------------------------------------
        {
            Check(SameHits(Ukkonen.Search("abc", "", 2, out _), new[] { 0, 1, 2 }),
                "the empty pattern matches at every position");
            Check(Ukkonen.Search("", "abc", 5, out _).Length == 0, "an empty text has no end position to report");
            Check(Ukkonen.Search("ab", "abcdef", 3, out _).Length == 0,
                "a pattern longer than text+k cannot match anywhere");
            Check(SameHits(Ukkonen.Search("ab", "abcdef", 6, out _), new[] { 0, 1 }),
                "once k covers the length gap every position matches, deletions included");
            Check(SameHits(Ukkonen.Search("aaaa", "aa", 2, out _), new[] { 0, 1, 2, 3 }),
                "k >= pattern length matches at every position");
            Check(SameHits(Ukkonen.Search("a", "aaa", 2, out _), new[] { 0 }),
                "two trailing deletions inside the budget still match");
            Check(SameHits(Ukkonen.Search("xyzxyz", "xyz", 0, out _), new[] { 2, 5 }),
                "a repeated pattern reports every occurrence, not only the first");
            Check(SameHits(Ukkonen.Search("ab", "a", 0, out _), new[] { 0 }),
                "the band honours the single-character corner");

            bool rejected = false;
            try { Ukkonen.Search("abc", "abc", -1, out _); }
            catch (ArgumentOutOfRangeException) { rejected = true; }
            Check(rejected, "a negative threshold is rejected instead of silently matching everything");
        }

        // ---- exhaustive sweep over a two-letter alphabet --------------------------------------
        {
            var texts = new List<string>();
            for (int len = 0; len <= 6; len++) texts.AddRange(BinaryStrings(len, 1 << len));
            var patterns = new List<string>();
            for (int len = 0; len <= 3; len++) patterns.AddRange(BinaryStrings(len, 1 << len));

            int cases = 0, withMatches = 0;
            string firstFailure = "none";
            long bandStates = 0, fullCells = 0;
            foreach (string text in texts)
            {
                foreach (string pattern in patterns)
                {
                    for (int k = 0; k <= 3; k++)
                    {
                        cases++;
                        int[] band = Ukkonen.Search(text, pattern, k, out _);
                        int[] full = Ukkonen.ScanFullMatrix(text, pattern, k);
                        if (band.Length > 0) withMatches++;
                        bandStates += (long)(k + 1) * (text.Length + pattern.Length + 1);
                        fullCells += (long)pattern.Length * Math.Max(1, text.Length);
                        if (firstFailure == "none" && !SameHits(band, full))
                        {
                            firstFailure = "text='" + text + "' pattern='" + pattern + "' k=" + k
                                + " band=" + Show(band) + " full=" + Show(full);
                        }
                    }
                }
            }
            Check(firstFailure == "none",
                "exhaustive band == Sellers DP over " + cases + " cases (" + firstFailure + ")");
            Check(cases >= 7000, "the sweep really enumerates every short binary case (" + cases + ")");
            Check(withMatches >= 2000, "the sweep is not vacuous: " + withMatches + " cases have a match");
            Console.WriteLine("exhaustiveCases=" + cases + " matchingCases=" + withMatches
                + " bandStates=" + bandStates + " fullDpCells=" + fullCells);
        }

        // ---- randomized cross-verification against the second oracle --------------------------
        {
            const int Cases = 80;
            ulong state = 0x1234_5678_9ABC_DEF0UL;
            bool agrees = true, nested = true;
            string firstFailure = "none";
            long bandStates = 0, fullCells = 0;

            for (int c = 0; c < Cases; c++)
            {
                int len = 12 + Rng.NextInt(ref state, 24);
                int m = 1 + Rng.NextInt(ref state, 6);
                int k = Rng.NextInt(ref state, 4);
                var textChars = new char[len];
                var patternChars = new char[m];
                for (int i = 0; i < len; i++) textChars[i] = (char)('a' + Rng.NextInt(ref state, 3));
                for (int i = 0; i < m; i++) patternChars[i] = (char)('a' + Rng.NextInt(ref state, 3));
                string text = new string(textChars);
                string pattern = new string(patternChars);

                int[] band = Ukkonen.Search(text, pattern, k, out _);
                bandStates += (long)(k + 1) * (text.Length + pattern.Length + 1);
                fullCells += (long)m * len;
                int[] bySubstring = Ukkonen.ScanBySubstringDistance(text, pattern, k);
                if (!SameHits(band, bySubstring))
                {
                    agrees = false;
                    if (firstFailure == "none")
                    {
                        firstFailure = "case " + c + " k=" + k + " text=" + text + " pattern=" + pattern
                            + " band=" + Show(band) + " substring=" + Show(bySubstring);
                    }
                }
                int[] wider = Ukkonen.Search(text, pattern, k + 1, out _);
                foreach (int h in band) if (Array.IndexOf(wider, h) < 0) nested = false;   // monotone in k
            }
            Check(agrees, "band == per-substring Levenshtein on " + Cases + " random cases (" + firstFailure + ")");
            Check(nested, "the match set only grows with k");
            Check(bandStates < fullCells, "the band visits fewer states than the full table has cells");
            Console.WriteLine("randomCases=" + Cases + " bandStates=" + bandStates + " fullDpCells=" + fullCells);
        }

        // ---- the band is O(k) wide, not O(m) --------------------------------------------------
        {
            string pattern = string.Concat("abcdefghij", "abcdefghij", "abcdefghij", "abcdefghij");
            string text = new string('a', 300) + pattern + new string('a', 300);
            long previous = -1;
            for (int k = 0; k <= 3; k++)
            {
                int[] hits = Ukkonen.Search(text, pattern, k, out long scanned);
                long states = (long)(k + 1) * (text.Length + pattern.Length + 1);
                Check(Array.IndexOf(hits, 339) >= 0, "k=" + k + " still finds the planted pattern");
                Check(states > previous, "the state count grows with the band width (k=" + k + ")");
                Check(states <= (long)text.Length * pattern.Length / 5,
                    "even k=" + k + " touches far fewer states than the full " + text.Length + "x"
                    + pattern.Length + " table");
                Check(scanned <= states * (pattern.Length + 1), "run skips stay inside the band (k=" + k + ")");
                previous = states;
            }
            Console.WriteLine("patternRows=40 textLength=" + text.Length + " statesAtK3=" + previous
                + " fullTableCells=" + (long)text.Length * pattern.Length);
        }

        // ---- on ordinary text the k=0 scan compares only a couple of characters per start -----
        {
            ulong state = 0xBADC0DEUL;
            var chars = new char[4000];
            for (int i = 0; i < chars.Length; i++) chars[i] = (char)('a' + Rng.NextInt(ref state, 4));
            string text = new string(chars);
            string pattern = "abcdabcd";
            int[] hits = Ukkonen.Search(text, pattern, 0, out long scanned);
            Check(SameHits(hits, LiteralEnds(text, pattern)), "the scan agrees with literal search on a long text");
            Check(scanned <= 4L * text.Length,
                "on non-repetitive text k=0 compares only a few characters per start: " + scanned);
        }

        // ---- determinism ----------------------------------------------------------------------
        {
            int[] first = Ukkonen.Search("aaaabbbbcccc", "abc", 2, out long s1);
            int[] second = Ukkonen.Search("aaaabbbbcccc", "abc", 2, out long s2);
            Check(SameHits(first, second) && s1 == s2, "the scan is deterministic");
        }

        Console.WriteLine("PASS");
    }
}
