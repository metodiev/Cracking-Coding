// Task 526 - Implement suffix automaton with endpos count (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A suffix automaton is the minimal deterministic automaton recognising exactly the factors of a text.
// Each state v represents the substrings whose endpos set (the positions at which the substring ends)
// is the same set, and those substrings are exactly the lengths in [len(link(v)) + 1, len(v)]; the
// automaton is built by feeding the characters left to right, creating one state per position plus
// clone states whenever an existing transition has to be split (the standard two-case link update).
// The endpos cardinality |endpos(v)| is obtained by setting count = 1 on every non-clone state (it
// ends a prefix), 0 on clones, and then adding every state's count into its suffix-link parent in
// decreasing len order - the endpos sets of a parent are exactly the union of its children's sets in
// the suffix-link tree.
//
// Documented checks. (1) Occurrence counts: for ten texts (including the empty string, "a", "aa",
// "abab", "banana", "aab", "zzzabc", an LCG string over a three-letter alphabet, the Fibonacci word
// and the Thue-Morse word) the automaton's |endpos| for every one of the O(n^2) substrings equals the
// count obtained by naive substring counting, and the count of a non-factor is 0. (2) Distinct
// substrings: sum over states of len(v) - len(link(v)) equals the size of the naive set of distinct
// substrings, and the same sum restricted to states with |endpos| >= k counts the distinct substrings
// occurring at least k times, checked against brute force for k = 1, 2, 3. (3) Size bounds: the number
// of states never exceeds 2n - 1 and the number of transitions never exceeds 3n - 4 (n >= 3); the
// family "a" followed by n - 1 b's attains exactly 2n - 1 states for n = 6, 8, 12. (4) Longest
// repeated substring and the number of distinct substrings occurring at least twice equal the naive
// values. (5) Structure, verified on short texts by materialising every endpos set through the link
// tree: each state's set size equals the propagated count, and the set equals the naive match
// positions of the state's representative substring; the representative of link(v) is a proper suffix
// of the representative of v. (6) Hand-checked counts in "banana": "a" 3, "n" 2, "ana" 2, "ban" 1,
// "banana" 1. (7) Boundaries: the empty text has one state, no transitions and no non-empty substring;
// Occurrences(null) throws ArgumentNullException.
using System;
using System.Collections.Generic;

internal sealed class SuffixAutomaton
{
    private readonly Dictionary<char, int>[] next;
    private readonly List<int> link = new List<int>();
    private readonly List<int> length = new List<int>();
    private readonly List<int> occurrence = new List<int>();
    private readonly List<int> creationEnd = new List<int>();
    private readonly string text;
    private int last;
    private int size;

    internal SuffixAutomaton(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        this.text = text;
        next = new Dictionary<char, int>[2 * text.Length + 2];
        next[0] = new Dictionary<char, int>();
        link.Add(-1);
        length.Add(0);
        occurrence.Add(0);
        creationEnd.Add(-1);
        size = 1;
        for (int i = 0; i < text.Length; i++) Extend(text[i], i);
        Propagate();
    }

    internal int TextLength => text.Length;
    internal int StateCount => size;
    internal int TransitionCount
    {
        get
        {
            int total = 0;
            for (int v = 0; v < size; v++) total += next[v].Count;
            return total;
        }
    }

    private void Extend(char c, int position)
    {
        int cur = size++;
        next[cur] = new Dictionary<char, int>();
        link.Add(0);
        length.Add(length[last] + 1);
        occurrence.Add(1);
        creationEnd.Add(position);
        int p = last;
        while (p != -1 && !next[p].ContainsKey(c))
        {
            next[p][c] = cur;
            p = link[p];
        }
        if (p == -1)
        {
            link[cur] = 0;
        }
        else
        {
            int q = next[p][c];
            if (length[p] + 1 == length[q])
            {
                link[cur] = q;
            }
            else
            {
                int clone = size++;
                next[clone] = new Dictionary<char, int>(next[q]);
                link.Add(link[q]);
                length.Add(length[p] + 1);
                occurrence.Add(0);
                creationEnd.Add(creationEnd[q]);
                while (p != -1 && next[p].TryGetValue(c, out int target) && target == q)
                {
                    next[p][c] = clone;
                    p = link[p];
                }
                link[q] = clone;
                link[cur] = clone;
            }
        }
        last = cur;
    }

    /** Adds every state's occurrence count into its suffix-link parent, longest states first. */
    private void Propagate()
    {
        int[] order = OrderByLengthDescending();
        foreach (int v in order)
        {
            if (v == 0) continue;
            occurrence[link[v]] += occurrence[v];
        }
    }

    private int[] OrderByLengthDescending()
    {
        int maxLength = 0;
        for (int v = 0; v < size; v++) maxLength = Math.Max(maxLength, length[v]);
        var bucket = new int[maxLength + 2];
        for (int v = 0; v < size; v++) bucket[length[v]]++;
        for (int i = 1; i < bucket.Length; i++) bucket[i] += bucket[i - 1];
        var order = new int[size];
        for (int v = size - 1; v >= 0; v--) order[--bucket[length[v]]] = v;
        Array.Reverse(order);
        return order;
    }

    internal int LengthOf(int state) => length[state];
    internal int LinkOf(int state) => link[state];
    internal long OccurrencesOf(int state) => occurrence[state];

    /** Walks the pattern through the automaton; returns the accepting state or -1. */
    internal int Walk(string pattern)
    {
        if (pattern == null) throw new ArgumentNullException(nameof(pattern));
        int state = 0;
        foreach (char c in pattern)
        {
            if (!next[state].TryGetValue(c, out int target)) return -1;
            state = target;
        }
        return state;
    }

    /** |endpos| of the pattern, i.e. how often it occurs as a factor (0 if it does not). */
    internal long Occurrences(string pattern)
    {
        int state = Walk(pattern);
        return state < 0 ? 0L : occurrence[state];
    }

    internal bool Contains(string pattern)
    {
        return Walk(pattern) >= 0;
    }

    /** Number of distinct substrings, optionally restricted to those occurring at least minOccurrences times. */
    internal long DistinctSubstrings(long minOccurrences)
    {
        long total = 0;
        for (int v = 1; v < size; v++)
        {
            if (occurrence[v] >= minOccurrences) total += length[v] - length[link[v]];
        }
        return total;
    }

    /** Representative substring of a state: its longest string, taken from the first occurrence. */
    internal string Representative(int state)
    {
        int end = creationEnd[state];
        return text.Substring(end - length[state] + 1, length[state]);
    }

    internal sealed class Repeated
    {
        internal string Substring = string.Empty;
        internal int Count;
    }

    internal Repeated LongestRepeated()
    {
        var best = new Repeated();
        for (int v = 1; v < size; v++)
        {
            if (occurrence[v] >= 2 && length[v] > best.Substring.Length)
            {
                best.Substring = Representative(v);
                best.Count = (int)occurrence[v];
            }
        }
        return best;
    }

    /** Materialises all endpos sets by merging the suffix-link tree bottom-up. */
    internal int[][] EndPositions()
    {
        var sets = new List<int>[size];
        for (int v = 0; v < size; v++)
        {
            sets[v] = new List<int>();
            if (creationEnd[v] >= 0) sets[v].Add(creationEnd[v]);
        }
        int[] order = OrderByLengthDescending();
        foreach (int v in order)
        {
            if (v == 0) continue;
            sets[link[v]].AddRange(sets[v]);
        }
        var result = new int[size][];
        for (int v = 0; v < size; v++)
        {
            var sorted = new SortedSet<int>(sets[v]);
            var array = new int[sorted.Count];
            sorted.CopyTo(array);
            result[v] = array;
        }
        return result;
    }
}

public static class Solution
{
    public static void Main()
    {
        string[] texts = {
            string.Empty, "a", "aa", "ab", "abab", "banana", "aab", "zzzabc",
            LcgText(40, 2024, 3), FibonacciWord(24), ThueMorse(31),
        };
        foreach (string text in texts)
        {
            var automaton = new SuffixAutomaton(text);
            var counts = BruteForceCounts(text);
            string tag = "text of length " + text.Length;
            Check(automaton.DistinctSubstrings(1) == counts.Count,
                    tag + ": distinct substrings " + automaton.DistinctSubstrings(1) + " vs " + counts.Count);
            foreach (KeyValuePair<string, int> entry in counts)
            {
                Check(automaton.Occurrences(entry.Key) == entry.Value,
                        tag + ": |endpos| of " + entry.Key + " is " + entry.Value);
            }
            for (int start = 0; start < text.Length; start++)
            {
                for (int len = 1; len <= text.Length - start; len++)
                {
                    string substring = text.Substring(start, len);
                    Check(automaton.Contains(substring), tag + ": the automaton recognises " + substring);
                }
            }
            Check(automaton.Occurrences("qq") == 0, tag + ": a non-factor has |endpos| = 0");
            if (text.Length >= 2) Check(automaton.StateCount <= 2 * text.Length - 1, tag + ": states <= 2n - 1");
            if (text.Length >= 3)
            {
                Check(automaton.TransitionCount <= 3 * text.Length - 4, tag + ": transitions <= 3n - 4");
            }
            for (long k = 1; k <= 3; k++)
            {
                Check(automaton.DistinctSubstrings(k) == CountDistinctWithAtLeast(counts, k),
                        tag + ": distinct substrings occurring at least " + k + " times");
            }
            SuffixAutomaton.Repeated repeated = automaton.LongestRepeated();
            string expected = LongestRepeatedSubstring(counts);
            Check(repeated.Substring.Length == expected.Length,
                    tag + ": longest repeated substring length " + expected.Length);
            if (expected.Length > 0)
            {
                Check(counts[repeated.Substring] == repeated.Count, tag + ": the reported count is the real one");
            }
            if (text.Length <= 16) CheckStructure(automaton, text, tag);
        }

        var banana = new SuffixAutomaton("banana");
        Check(banana.Occurrences("a") == 3, "banana: a occurs 3 times");
        Check(banana.Occurrences("n") == 2, "banana: n occurs 2 times");
        Check(banana.Occurrences("ana") == 2, "banana: ana occurs 2 times");
        Check(banana.Occurrences("ban") == 1, "banana: ban occurs 1 time");
        Check(banana.Occurrences("banana") == 1, "banana: the whole text occurs once");
        Check(banana.DistinctSubstrings(1) == 15, "banana has 15 distinct substrings");
        SuffixAutomaton.Repeated bananaRepeated = banana.LongestRepeated();
        Check(bananaRepeated.Substring == "ana" && bananaRepeated.Count == 2, "banana: longest repeated is ana");
        Check(banana.StateCount == 10 && banana.TransitionCount == 11, "banana: 10 states and 11 transitions");

        foreach (int n in new[] { 6, 8, 12 })
        {
            var maximal = new SuffixAutomaton(MaxStateText(n));
            Check(maximal.StateCount == 2 * n - 1, "a b^(n-1) has exactly 2n - 1 states for n = " + n);
        }

        var emptyText = new SuffixAutomaton(string.Empty);
        Check(emptyText.StateCount == 1, "the empty text has only the root");
        Check(emptyText.TransitionCount == 0, "the empty text has no transitions");
        Check(emptyText.DistinctSubstrings(1) == 0, "the empty text has no substrings");
        Check(emptyText.Occurrences("a") == 0, "no factor in the empty text");

        foreach (int seed in new[] { 5, 17, 29, 41 })
        {
            string text = LcgText(60, seed, 2);
            var automaton = new SuffixAutomaton(text);
            var counts = BruteForceCounts(text);
            Check(automaton.DistinctSubstrings(1) == counts.Count, "binary LCG text, seed " + seed);
            Check(automaton.StateCount <= 2 * text.Length - 1, "binary LCG text, seed " + seed + ": states");
            foreach (KeyValuePair<string, int> entry in counts)
            {
                Check(automaton.Occurrences(entry.Key) == entry.Value,
                        "binary LCG text, seed " + seed + ": count of " + entry.Key);
            }
        }

        var throwsAutomaton = new SuffixAutomaton("abc");
        CheckThrows(() => throwsAutomaton.Occurrences(null!), "a null pattern throws");
        CheckThrows(() => throwsAutomaton.Walk(null!), "a null walk throws");
        CheckThrows(() => new SuffixAutomaton(null!), "a null text throws");

        Console.WriteLine("PASS");
    }

    private static void CheckStructure(SuffixAutomaton automaton, string text, string tag)
    {
        int[][] endPositions = automaton.EndPositions();
        Check(endPositions.Length == automaton.StateCount, tag + ": one endpos set per state");
        for (int v = 1; v < automaton.StateCount; v++)
        {
            Check(endPositions[v].Length == automaton.OccurrencesOf(v),
                    tag + ": |endpos| equals the propagated count for state " + v);
            string representative = automaton.Representative(v);
            int[] matches = MatchPositions(text, representative);
            Check(matches.Length == endPositions[v].Length,
                    tag + ": endpos size of " + representative + " is " + matches.Length);
            for (int i = 0; i < matches.Length; i++)
            {
                Check(matches[i] == endPositions[v][i],
                        tag + ": endpos entry " + i + " of " + representative);
            }
            int parent = automaton.LinkOf(v);
            if (parent == 0) continue;
            string parentRepresentative = automaton.Representative(parent);
            Check(automaton.LengthOf(parent) < automaton.LengthOf(v),
                    tag + ": the link shortens the state " + v);
            Check(EndsWith(representative, parentRepresentative),
                    tag + ": the link representative is a suffix of " + representative);
        }
    }

    private static bool EndsWith(string text, string suffix)
    {
        return suffix.Length <= text.Length && text.Substring(text.Length - suffix.Length) == suffix;
    }

    private static int[] MatchPositions(string text, string pattern)
    {
        var positions = new List<int>();
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) positions.Add(i + pattern.Length - 1);
        }
        return positions.ToArray();
    }

    private static Dictionary<string, int> BruteForceCounts(string text)
    {
        var counts = new Dictionary<string, int>();
        for (int start = 0; start < text.Length; start++)
        {
            for (int end = start + 1; end <= text.Length; end++)
            {
                string substring = text.Substring(start, end - start);
                counts.TryGetValue(substring, out int current);
                counts[substring] = current + 1;
            }
        }
        return counts;
    }

    private static long CountDistinctWithAtLeast(Dictionary<string, int> counts, long minimum)
    {
        long total = 0;
        foreach (KeyValuePair<string, int> entry in counts)
        {
            if (entry.Value >= minimum) total++;
        }
        return total;
    }

    private static string LongestRepeatedSubstring(Dictionary<string, int> counts)
    {
        string best = string.Empty;
        foreach (KeyValuePair<string, int> entry in counts)
        {
            if (entry.Value >= 2 && entry.Key.Length > best.Length) best = entry.Key;
        }
        return best;
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

    private static string FibonacciWord(int length)
    {
        string a = "a";
        string b = "ab";
        while (b.Length < length)
        {
            string next = b + a;
            a = b;
            b = next;
        }
        return b.Substring(0, length);
    }

    private static string ThueMorse(int length)
    {
        string word = "a";
        while (word.Length < length)
        {
            var flipped = new char[word.Length];
            for (int i = 0; i < word.Length; i++) flipped[i] = word[i] == 'a' ? 'b' : 'a';
            word = word + new string(flipped);
        }
        return word.Substring(0, length);
    }

    private static string MaxStateText(int n)
    {
        return "a" + new string('b', n - 1);
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
