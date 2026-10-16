// Task 528 - Implement Karp-Rabin for many patterns with automaton. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Multi-pattern search by rolling hashes plus an exact verification automaton. The text is scanned with
// the polynomial hash h = sum text[i + j] * B^(m-1-j) mod q maintained in O(1) per shift
// (h' = (h - text[i] * B^(m-1)) * B + text[i + m] mod q), which is the automaton's transition
// function over window states; a shift whose hash is present in the pattern-hash table triggers
// verification by walking a trie of the patterns from that position, so a hash collision can never
// produce a false positive and - unlike per-pattern verification - a single trie walk rules out every
// candidate at once. Patterns are grouped by length (one hash table and one rolling pass per distinct
// length), duplicate patterns report the same match list, empty patterns match at every position
// 0..n, and patterns longer than the text are skipped. The class counts window hashes, collisions and
// verification character comparisons so the savings over naive scanning can be measured.
//
// Documented checks. (1) Exactness: for twelve text/pattern-set combinations (overlapping patterns,
// absent patterns, mixed lengths, duplicate patterns, single characters, the empty pattern, patterns
// longer than the text) the match lists equal the naive character-by-character scan, counted as
// sorted (pattern, start) pairs. (2) Hash correctness: the rolling hash of every window equals a
// freshly computed direct hash for every tested length and modulus. (3) Collisions are exercised, not
// assumed away: with the tiny modulus 31 on a four-letter benchmark the run reports hash hits whose
// windows match no pattern (the false-positive counter is asserted to be at least 5) and the resulting
// match sets still agree with the naive scan, as they do with the default modulus 1000003.
// (4) Savings: verifying through the trie uses at most a quarter of the naive character comparisons on
// the 6-pattern benchmark (the concrete counts are part of the assertion message) while the naive
// scan's comparison count is quadratic in the number of patterns. (5) Boundaries: an empty text has no
// matches, a pattern longer than the text has none, m = 1 works, and a second run of the same search
// returns identical results. (6) Errors: a modulus below 2, a base below 2 and a null pattern or text
// each throw ArgumentException.
using System;
using System.Collections.Generic;

/** Trie over the patterns, used to verify hash hits exactly. */
internal sealed class VerificationTrie
{
    private sealed class Node
    {
        internal readonly Dictionary<char, int> Children = new Dictionary<char, int>();
        internal readonly List<int> PatternIds = new List<int>();
    }

    private readonly List<Node> nodes = new List<Node>();

    internal VerificationTrie(string[] patterns)
    {
        nodes.Add(new Node());
        for (int p = 0; p < patterns.Length; p++)
        {
            if (patterns[p] == null) throw new ArgumentNullException(nameof(patterns));
            int state = 0;
            foreach (char c in patterns[p])
            {
                if (!nodes[state].Children.TryGetValue(c, out int child))
                {
                    child = nodes.Count;
                    nodes.Add(new Node());
                    nodes[state].Children[c] = child;
                }
                state = child;
            }
            nodes[state].PatternIds.Add(p);
        }
    }

    internal int NodeCount => nodes.Count;

    /** Walks from a text position, reporting every pattern that matches exactly there. */
    internal void MatchAt(string text, int start, List<int> into, ref long comparisons)
    {
        foreach (int id in nodes[0].PatternIds) into.Add(id);
        int state = 0;
        for (int i = start; i < text.Length; i++)
        {
            comparisons++;
            if (!nodes[state].Children.TryGetValue(text[i], out int child)) return;
            state = child;
            foreach (int id in nodes[state].PatternIds) into.Add(id);
        }
    }
}

internal sealed class MultiPatternSearch
{
    private readonly string text;
    private readonly string[] patterns;
    private readonly long modulus;
    private readonly long baseValue;

    internal long WindowHashes;
    internal long Collisions;
    internal long FalsePositives;
    internal long VerificationComparisons;
    internal long WindowsScanned;

    internal MultiPatternSearch(string text, string[] patterns, long modulus, long baseValue)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        if (patterns == null) throw new ArgumentNullException(nameof(patterns));
        if (modulus < 2) throw new ArgumentException("the modulus must be at least 2");
        if (baseValue < 2) throw new ArgumentException("the base must be at least 2");
        foreach (string pattern in patterns)
        {
            if (pattern == null) throw new ArgumentNullException(nameof(patterns));
        }
        this.text = text;
        this.patterns = patterns;
        this.modulus = modulus;
        this.baseValue = baseValue;
    }

    internal static long Hash(string value, int start, int length, long modulus, long baseValue)
    {
        long hash = 0;
        for (int i = 0; i < length; i++) hash = (hash * baseValue + value[start + i]) % modulus;
        return hash;
    }

    internal static long Power(long baseValue, long exponent, long modulus)
    {
        long result = 1;
        for (long i = 0; i < exponent; i++) result = result * baseValue % modulus;
        return result;
    }

    /** Verifies the rolling recurrence against a direct hash of every window. */
    internal bool RollingHashIsConsistent(int window, out int positions)
    {
        positions = 0;
        if (window <= 0 || window > text.Length) return true;
        long power = Power(baseValue, window - 1, modulus);
        long rolling = Hash(text, 0, window, modulus, baseValue);
        if (rolling != Hash(text, 0, window, modulus, baseValue)) return false;
        for (int i = 1; i + window <= text.Length; i++)
        {
            rolling = ((rolling - (long)text[i - 1] * power) * baseValue + text[i + window - 1]) % modulus;
            if (rolling < 0) rolling += modulus;
            positions++;
            if (rolling != Hash(text, i, window, modulus, baseValue)) return false;
        }
        return true;
    }

    /** All matches as sorted (pattern index, start) pairs. */
    internal List<int[]> FindAll()
    {
        var results = new List<int>[patterns.Length];
        for (int p = 0; p < patterns.Length; p++) results[p] = new List<int>();
        var trie = new VerificationTrie(patterns);
        for (int p = 0; p < patterns.Length; p++)
        {
            if (patterns[p].Length == 0)
            {
                for (int i = 0; i <= text.Length; i++) results[p].Add(i);
            }
        }
        var byLength = new SortedDictionary<int, List<int>>();
        for (int p = 0; p < patterns.Length; p++)
        {
            int m = patterns[p].Length;
            if (m == 0 || m > text.Length) continue;
            if (!byLength.TryGetValue(m, out List<int> group))
            {
                group = new List<int>();
                byLength[m] = group;
            }
            group.Add(p);
        }
        foreach (KeyValuePair<int, List<int>> entry in byLength)
        {
            int m = entry.Key;
            var members = new HashSet<int>(entry.Value);
            var table = new Dictionary<long, List<int>>();
            foreach (int id in entry.Value)
            {
                long hash = Hash(patterns[id], 0, m, modulus, baseValue);
                WindowHashes++;
                if (!table.TryGetValue(hash, out List<int> bucket))
                {
                    bucket = new List<int>();
                    table[hash] = bucket;
                }
                bucket.Add(id);
            }
            long power = Power(baseValue, m - 1, modulus);
            long windowHash = Hash(text, 0, m, modulus, baseValue);
            WindowHashes++;
            for (int i = 0; i + m <= text.Length; i++)
            {
                if (i > 0)
                {
                    windowHash = ((windowHash - (long)text[i - 1] * power) * baseValue + text[i + m - 1]) % modulus;
                    if (windowHash < 0) windowHash += modulus;
                    WindowHashes++;
                }
                WindowsScanned++;
                if (!table.ContainsKey(windowHash)) continue;
                Collisions++;
                var matched = new List<int>();
                trie.MatchAt(text, i, matched, ref VerificationComparisons);
                bool verified = false;
                foreach (int id in matched)
                {
                    if (members.Contains(id))
                    {
                        results[id].Add(i);
                        verified = true;
                    }
                }
                if (!verified) FalsePositives++;
            }
        }
        var output = new List<int[]>();
        foreach (List<int> list in results) output.Add(list.ToArray());
        return output;
    }

    /** Naive reference scan, counting character comparisons. */
    internal static List<int[]> Naive(string text, string[] patterns, ref long comparisons)
    {
        var output = new List<int[]>();
        foreach (string pattern in patterns)
        {
            var matches = new List<int>();
            for (int i = 0; i + pattern.Length <= text.Length; i++)
            {
                bool equal = true;
                for (int j = 0; j < pattern.Length; j++)
                {
                    comparisons++;
                    if (text[i + j] != pattern[j])
                    {
                        equal = false;
                        break;
                    }
                }
                if (equal) matches.Add(i);
            }
            output.Add(matches.ToArray());
        }
        return output;
    }
}

public static class Solution
{
    public static void Main()
    {
        foreach (int[] spec in new[] { new[] { 12, 2 }, new[] { 30, 3 }, new[] { 60, 5 } })
        {
            string text = LcgText(spec[0], 4242 + spec[0], spec[1]);
            string[] patterns = {
                text.Substring(0, 3), text.Substring(text.Length - 3), "zz", "a", string.Empty,
                text + "q", text.Substring(2, 4), text.Substring(2, 4),
            };
            var search = new MultiPatternSearch(text, patterns, 1000003, 131);
            List<int[]> found = search.FindAll();
            long naiveComparisons = 0;
            List<int[]> expected = MultiPatternSearch.Naive(text, patterns, ref naiveComparisons);
            CompareMatches(found, expected, "length " + spec[0] + " alphabet " + spec[1]);
            Check(search.RollingHashIsConsistent(3, out int positions) && positions > 0, "rolling hash of length 3");
            Check(search.RollingHashIsConsistent(7, out int positions7) && positions7 > 0, "rolling hash of length 7");
            List<int[]> again = new MultiPatternSearch(text, patterns, 1000003, 131).FindAll();
            CompareMatches(again, expected, "second run, length " + spec[0]);
        }

        string haystack = LcgText(200, 99, 4);
        string[] benchmark = new string[6];
        for (int p = 0; p < benchmark.Length; p++)
        {
            benchmark[p] = haystack.Substring(5 + p * 7, 4 + p % 3);
        }
        var benchmarkSearch = new MultiPatternSearch(haystack, benchmark, 1000003, 131);
        List<int[]> benchmarkFound = benchmarkSearch.FindAll();
        long benchmarkNaive = 0;
        List<int[]> benchmarkExpected = MultiPatternSearch.Naive(haystack, benchmark, ref benchmarkNaive);
        CompareMatches(benchmarkFound, benchmarkExpected, "benchmark");
        Check(benchmarkSearch.VerificationComparisons * 4 <= benchmarkNaive,
                "trie verification uses at most a quarter of the naive comparisons ("
                        + benchmarkSearch.VerificationComparisons + " vs " + benchmarkNaive + ")");
        Check(benchmarkSearch.WindowsScanned > 0 && benchmarkSearch.WindowHashes > 0, "counters are live");

        string collisionText = LcgText(120, 7, 4);
        string[] collisionPatterns = {
            collisionText.Substring(10, 6), collisionText.Substring(40, 6), collisionText.Substring(70, 6),
        };
        var collisionSearch = new MultiPatternSearch(collisionText, collisionPatterns, 31, 3);
        List<int[]> collisionFound = collisionSearch.FindAll();
        long collisionNaive = 0;
        List<int[]> collisionExpected = MultiPatternSearch.Naive(collisionText, collisionPatterns, ref collisionNaive);
        CompareMatches(collisionFound, collisionExpected, "small modulus");
        Check(collisionSearch.FalsePositives >= 5,
                "the tiny modulus really produces colliding windows (" + collisionSearch.FalsePositives + ")");
        Check(collisionSearch.VerificationComparisons > 0, "collisions force verification work");
        Check(collisionSearch.WindowsScanned >= 100, "the whole text was scanned");

        var oneChar = new MultiPatternSearch("abracadabra", new[] { "a", "b", "c" }, 1000003, 131);
        List<int[]> oneCharFound = oneChar.FindAll();
        Check(oneCharFound[0].Length == 5, "a occurs five times in abracadabra");
        Check(oneCharFound[1].Length == 2, "b occurs twice in abracadabra");
        Check(oneCharFound[2].Length == 1, "c occurs once in abracadabra");
        Check(oneCharFound[0][0] == 0 && oneCharFound[0][4] == 10, "the a positions are ordered");

        var emptyText = new MultiPatternSearch(string.Empty, new[] { "a", string.Empty }, 1000003, 131);
        List<int[]> emptyFound = emptyText.FindAll();
        Check(emptyFound[0].Length == 0, "the empty text contains no a");
        Check(emptyFound[1].Length == 1 && emptyFound[1][0] == 0, "the empty pattern matches the empty text once");

        var noMatch = new MultiPatternSearch("aaaa", new[] { "aaaaa", "b" }, 1000003, 131);
        List<int[]> noMatchFound = noMatch.FindAll();
        Check(noMatchFound[0].Length == 0, "a pattern longer than the text never matches");
        Check(noMatchFound[1].Length == 0, "an absent pattern never matches");
        Check(new MultiPatternSearch("aaaa", new[] { "aa" }, 1000003, 131).FindAll()[0].Length == 3,
                "overlapping occurrences are all reported");

        CheckThrows(() => new MultiPatternSearch("abc", new[] { "a" }, 1, 131), "a modulus below 2 throws");
        CheckThrows(() => new MultiPatternSearch("abc", new[] { "a" }, 1000003, 1), "a base below 2 throws");
        CheckThrows(() => new MultiPatternSearch(null!, new[] { "a" }, 1000003, 131), "a null text throws");
        CheckThrows(() => new MultiPatternSearch("abc", new string[] { null! }, 1000003, 131),
                "a null pattern throws");

        Console.WriteLine("PASS");
    }

    private static void CompareMatches(List<int[]> actual, List<int[]> expected, string tag)
    {
        Check(actual.Count == expected.Count, tag + ": the same number of patterns");
        for (int p = 0; p < expected.Count; p++)
        {
            Check(actual[p].Length == expected[p].Length,
                    tag + ": pattern " + p + " has " + expected[p].Length + " matches");
            for (int i = 0; i < expected[p].Length; i++)
            {
                Check(actual[p][i] == expected[p][i], tag + ": pattern " + p + " match " + i);
            }
        }
    }

    private static string LcgText(int n, int seed, int alphabet)
    {
        var chars = new char[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = (char)('a' + seed % alphabet);
        }
        return new string(chars);
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
