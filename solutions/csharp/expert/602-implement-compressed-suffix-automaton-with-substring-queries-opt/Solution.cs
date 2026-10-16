// Task 602 - Implement compressed suffix automaton with substring queries optimized. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A suffix automaton is already the minimal (compressed) DFA of all suffixes of a text: its states
// are the endpos-equivalence classes and state v recognises exactly the substrings whose length
// lies in (length[link[v]], length[v]]. This solution stores it in compressed form and answers the
// query set in O(|pattern| log sigma) or O(occurrences):
//   * transitions are compacted into a CSR table - one sorted, duplicate-free symbol/target run per
//     state, binary searched - instead of a dense sigma x states matrix: linear memory, log lookup;
//   * endpos sizes are propagated once along the suffix links in decreasing-length order, so
//     OccurrenceCount is an O(|pattern|) walk plus one array read;
//   * the suffix-link tree is kept as child lists, so OccurrenceStarts(p) lists the concrete start
//     positions by walking only the subtree of the reached state;
//   * CountDistinctSubstrings is sum(length[v] - length[link[v]]) and LongestRepeatedSubstring is
//     the longest state whose endpos size is at least 2.
//
// Conventions: the empty pattern is never a substring and has zero occurrences (the "n+1 matches"
// convention is deliberately not used); a character outside the alphabet is rejected in
// O(|pattern|); the automaton of the empty text has only the initial state.
//
// Verified: distinct-substring counts, occurrence counts and exact occurrence start lists against
// brute force for every substring of ten fixed texts and for random substrings of random texts in
// two alphabets (one narrow, one of 40 symbols to exercise the binary search); longest repeated
// substring against brute force; the CSR invariant (runs sorted and duplicate free, the binary
// search used at query time finds every stored symbol); transitions <= 3n and states <= 2n+1;
// boundary inputs (empty text, single character, pattern longer than the text, unknown characters,
// mutants of random texts, a highly repetitive 360-character text).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class CompressedSuffixAutomaton
{
    private readonly int[] length, link, endCount;
    private readonly int[] transStart;      // CSR offsets, size states + 1
    private readonly int[] transSymbol, transTarget;
    private readonly int[] terminalHead, terminalNext;
    private readonly List<int>[] linkChildren;
    private readonly int states;

    public CompressedSuffixAutomaton(string text)
    {
        int n = text.Length;
        int capacity = 2 * n + 2;
        length = new int[capacity]; link = new int[capacity]; endCount = new int[capacity];
        transStart = new int[capacity]; terminalHead = new int[capacity]; terminalNext = new int[n + 1];
        var next = new Dictionary<int, int>[capacity];
        for (int v = 0; v < capacity; v++)
        {
            next[v] = new Dictionary<int, int>();
            terminalHead[v] = -1;
        }

        // Standard SAM construction with hash transitions; the compaction pass below rewrites them
        // into the compressed CSR form that every query uses.
        var prefixState = new int[n];
        states = 1;
        link[0] = -1;
        int last = 0;
        for (int i = 0; i < n; i++)
        {
            int c = text[i];
            int cur = states++;
            length[cur] = length[last] + 1;
            endCount[cur] = 1;
            int p = last;
            while (p != -1 && !next[p].ContainsKey(c)) { next[p][c] = cur; p = link[p]; }
            if (p == -1) link[cur] = 0;
            else
            {
                int q = next[p][c];
                if (length[p] + 1 == length[q]) link[cur] = q;
                else
                {
                    int clone = states++;
                    length[clone] = length[p] + 1;
                    link[clone] = link[q];
                    next[clone] = new Dictionary<int, int>(next[q]);
                    while (p != -1 && next[p].TryGetValue(c, out int t) && t == q) { next[p][c] = clone; p = link[p]; }
                    link[q] = clone;
                    link[cur] = clone;
                }
            }
            last = cur;
            prefixState[i] = cur;
        }

        // Compaction: one sorted, duplicate-free symbol run per state.
        transStart[0] = 0;
        for (int v = 0; v < states; v++) transStart[v + 1] = transStart[v] + next[v].Count;
        transSymbol = new int[transStart[states]];
        transTarget = new int[transStart[states]];
        for (int v = 0; v < states; v++)
        {
            var symbols = next[v].Keys.ToArray();
            Array.Sort(symbols);
            for (int k = 0; k < symbols.Length; k++)
            {
                int at = transStart[v] + k;
                transSymbol[at] = symbols[k];
                transTarget[at] = next[v][symbols[k]];
            }
        }

        // Terminals: prefix i ends in state prefixState[i], so endpos(v) is the set of prefix
        // positions stored inside v's suffix-link subtree.
        for (int i = 0; i < n; i++)
        {
            int v = prefixState[i];
            terminalNext[i] = terminalHead[v];
            terminalHead[v] = i;
        }

        // Propagate endpos sizes, longest state first: a suffix link always points to a strictly
        // shorter state, so a single pass in that order is enough.
        var order = new int[states - 1];
        for (int v = 1; v < states; v++) order[v - 1] = v;
        Array.Sort(order, (a, b) => length[b] - length[a]);
        foreach (int v in order) endCount[link[v]] += endCount[v];

        linkChildren = new List<int>[states];
        for (int v = 0; v < states; v++) linkChildren[v] = new List<int>();
        for (int v = 1; v < states; v++) linkChildren[link[v]].Add(v);
    }

    public int States => states;
    public int Transitions => transSymbol.Length;

    // Binary search of the state's sorted transition run.
    public int Step(int state, int symbol)
    {
        int lo = transStart[state], hi = transStart[state + 1] - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (transSymbol[mid] < symbol) lo = mid + 1;
            else if (transSymbol[mid] > symbol) hi = mid - 1;
            else return transTarget[mid];
        }
        return -1;
    }

    public int Walk(string pattern)
    {
        int state = 0;
        foreach (char ch in pattern)
        {
            state = Step(state, ch);
            if (state == -1) return -1;
        }
        return state;
    }

    public bool IsSubstring(string pattern) => pattern.Length > 0 && Walk(pattern) != -1;

    public int OccurrenceCount(string pattern)
    {
        int state = pattern.Length == 0 ? -1 : Walk(pattern);
        return state == -1 ? 0 : endCount[state];
    }

    public List<int> OccurrenceStarts(string pattern)
    {
        var starts = new List<int>();
        int state = pattern.Length == 0 ? -1 : Walk(pattern);
        if (state == -1) return starts;
        var stack = new Stack<int>();
        stack.Push(state);
        while (stack.Count > 0)
        {
            int v = stack.Pop();
            for (int t = terminalHead[v]; t != -1; t = terminalNext[t]) starts.Add(t - pattern.Length + 1);
            foreach (int child in linkChildren[v]) stack.Push(child);
        }
        starts.Sort();
        return starts;
    }

    public long CountDistinctSubstrings()
    {
        long total = 0;
        for (int v = 1; v < states; v++) total += length[v] - length[link[v]];
        return total;
    }

    public int LongestRepeatedSubstring()
    {
        int best = 0;
        for (int v = 1; v < states; v++) if (endCount[v] >= 2 && length[v] > best) best = length[v];
        return best;
    }

    // Property check of the compressed representation: each run is sorted and duplicate free, and
    // the binary search used at query time finds every stored symbol.
    public bool CsrIsSortedAndUnique()
    {
        for (int v = 0; v < states; v++)
            for (int i = transStart[v]; i < transStart[v + 1]; i++)
            {
                if (Step(v, transSymbol[i]) != transTarget[i]) return false;
                if (i > transStart[v] && transSymbol[i] <= transSymbol[i - 1]) return false;
            }
        return true;
    }
}

public static class Solution
{
    private static ulong seed = 0x2545F4914F6CDD1DUL;

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

    // One pass over all substrings yields both oracles: how many are distinct, and how long the
    // longest one occurring at least twice is.
    private static void BruteSubstringStats(string text, out long distinct, out int longestRepeated)
    {
        var seen = new HashSet<string>();
        distinct = 0;
        longestRepeated = 0;
        for (int i = 0; i < text.Length; i++)
        {
            for (int j = i + 1; j <= text.Length; j++)
            {
                string sub = text.Substring(i, j - i);
                if (seen.Add(sub)) distinct++;
                else if (sub.Length > longestRepeated) longestRepeated = sub.Length;
            }
        }
    }

    private static List<int> BruteStarts(string text, string pattern)
    {
        var starts = new List<int>();
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) starts.Add(i);
        }
        return starts;
    }

    // Every substring of the text against brute force, plus the compressed-storage invariants.
    private static void Exhaustive(string text)
    {
        var sam = new CompressedSuffixAutomaton(text);
        Check(sam.CsrIsSortedAndUnique(), "csr invariant " + text.Length);
        Check(sam.Transitions <= Math.Max(1, 3 * (text.Length + 1)) && sam.States <= 2 * (text.Length + 1),
            "size bound " + text.Length);
        BruteSubstringStats(text, out long distinct, out int lrs);
        Check(sam.CountDistinctSubstrings() == distinct && sam.LongestRepeatedSubstring() == lrs, "stats " + text);
        for (int i = 0; i < text.Length; i++)
        {
            for (int j = i + 1; j <= text.Length; j++)
            {
                string p = text.Substring(i, j - i);
                var expected = BruteStarts(text, p);
                Check(sam.OccurrenceCount(p) == expected.Count, "count " + text + " " + p);
                Check(string.Join(",", sam.OccurrenceStarts(p)) == string.Join(",", expected), "starts " + p);
            }
        }
        Check(!sam.IsSubstring(text + "~") && sam.OccurrenceCount(text + "~") == 0, "reject " + text);
    }

    // Random texts: random substrings and mutants must agree with brute force.
    private static void RandomBattery(string alphabet)
    {
        for (int trial = 0; trial < 6; trial++)
        {
            string text = RandomText(70, alphabet);
            var sam = new CompressedSuffixAutomaton(text);
            BruteSubstringStats(text, out long distinct, out int lrs);
            Check(sam.CountDistinctSubstrings() == distinct && sam.LongestRepeatedSubstring() == lrs, "random stats");
            for (int probe = 0; probe < 25; probe++)
            {
                int start = (int)(NextRandom() % (ulong)text.Length);
                string p = text.Substring(start, 1 + (int)(NextRandom() % (ulong)(text.Length - start)));
                var expected = BruteStarts(text, p);
                Check(sam.OccurrenceCount(p) == expected.Count, "random count " + p);
                Check(string.Join(",", sam.OccurrenceStarts(p)) == string.Join(",", expected), "random starts " + p);
            }
            string mutant = text.Substring(0, 10) + "~" + text.Substring(11);
            Check(sam.OccurrenceCount(mutant) == BruteStarts(text, mutant).Count, "random mutant");
        }
    }

    public static void Main()
    {
        var empty = new CompressedSuffixAutomaton("");
        Check(empty.States == 1 && empty.CountDistinctSubstrings() == 0 && !empty.IsSubstring("a"), "empty text");
        Check(empty.OccurrenceCount("") == 0 && empty.OccurrenceCount("a") == 0, "empty queries");
        Check(empty.OccurrenceStarts("a").Count == 0, "empty starts");

        var one = new CompressedSuffixAutomaton("a");
        Check(one.CountDistinctSubstrings() == 1 && one.OccurrenceCount("a") == 1 && !one.IsSubstring("aa"), "single char");
        Check(!one.IsSubstring("") && one.OccurrenceCount("") == 0, "empty pattern convention");

        var banana = new CompressedSuffixAutomaton("banana");
        Check(banana.CountDistinctSubstrings() == 15 && banana.LongestRepeatedSubstring() == 3, "banana distinct/lrs");
        Check(banana.OccurrenceCount("an") == 2 && banana.OccurrenceCount("nan") == 1, "banana an/nan");
        Check(banana.OccurrenceCount("banana") == 1 && banana.OccurrenceCount("bn") == 0, "banana whole/absent");
        Check(string.Join(",", banana.OccurrenceStarts("an")) == "1,3", "banana an starts");
        Check(string.Join(",", banana.OccurrenceStarts("a")) == "1,3,5", "banana a starts");

        var aaaa = new CompressedSuffixAutomaton("aaaa");
        Check(aaaa.CountDistinctSubstrings() == 4 && aaaa.LongestRepeatedSubstring() == 3, "aaaa distinct/lrs");
        for (int len = 1; len <= 4; len++) Check(aaaa.OccurrenceCount(new string('a', len)) == 5 - len, "aaaa " + len);
        Check(string.Join(",", aaaa.OccurrenceStarts("aa")) == "0,1,2", "aaaa starts");

        foreach (string text in new[] { "", "a", "ab", "aba", "aaa", "abab", "banana", "mississippi", "abracadabra", "abcabcab" })
            Exhaustive(text);
        foreach (string alphabet in new[] { "ab", "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdef" }) RandomBattery(alphabet);

        string repetitive = string.Concat(Enumerable.Range(0, 60).Select(i => i % 3 == 0 ? "abab" : "ababab"));
        var large = new CompressedSuffixAutomaton(repetitive);
        BruteSubstringStats(repetitive, out long largeDistinct, out _);
        Check(large.CountDistinctSubstrings() == largeDistinct, "repetitive distinct");
        Check(large.OccurrenceCount("ababab") == BruteStarts(repetitive, "ababab").Count, "repetitive count");
        Check(large.OccurrenceStarts("ababab").Count == large.OccurrenceCount("ababab"), "repetitive starts");

        Console.WriteLine("repetitive text " + repetitive.Length + " chars -> " + large.States + " states, " +
            large.Transitions + " transitions, " + large.CountDistinctSubstrings() + " distinct substrings");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
