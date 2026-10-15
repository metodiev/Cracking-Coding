// Task 431 - Implement suffix automaton for distinct substring queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Query implemented: the distinct-substring count of the automaton's own string, i.e. the sum
// of len[state] - len[link[state]] over all non-initial states ("abab" -> 7, "aaaa" -> 4).
// Two further suffix-automaton queries back the accept/reject checks: IsSubstring(t) walks the
// transition function from the initial state, and Occurrences(t) reads the end-position count of
// the reached state (terminal counts are propagated along suffix links in decreasing-length
// order). Patterns passed to Occurrences are non-empty; every answer is cross-checked against a
// brute-force set of all substrings on LCG-generated strings.
using System;
using System.Collections.Generic;

internal class SuffixAutomaton
{
    private readonly int[] length;
    private readonly int[] link;
    private readonly int[] ends;
    private readonly List<Dictionary<char, int>> next;
    private int states;

    public SuffixAutomaton(string text)
    {
        int capacity = 2 * text.Length + 1;
        length = new int[capacity];
        link = new int[capacity];
        ends = new int[capacity];
        next = new List<Dictionary<char, int>>(capacity);
        for (int i = 0; i < capacity; i++) next.Add(new Dictionary<char, int>());
        states = 1;
        link[0] = -1;
        int last = 0;
        foreach (char ch in text)
        {
            int cur = states++;
            length[cur] = length[last] + 1;
            link[cur] = -1;
            ends[cur] = 1;
            int p = last;
            while (p != -1 && !next[p].ContainsKey(ch))
            {
                next[p][ch] = cur;
                p = link[p];
            }
            if (p == -1)
            {
                link[cur] = 0;
            }
            else
            {
                int q = next[p][ch];
                if (length[p] + 1 == length[q])
                {
                    link[cur] = q;
                }
                else
                {
                    int clone = states++;
                    length[clone] = length[p] + 1;
                    link[clone] = link[q];
                    next[clone] = new Dictionary<char, int>(next[q]);
                    ends[clone] = 0;
                    while (p != -1 && next[p].TryGetValue(ch, out int t) && t == q)
                    {
                        next[p][ch] = clone;
                        p = link[p];
                    }
                    link[q] = clone;
                    link[cur] = clone;
                }
            }
            last = cur;
        }
        var order = new int[states - 1];
        for (int i = 1; i < states; i++) order[i - 1] = i;
        Array.Sort(order, (a, b) => length[b] - length[a]);
        foreach (int state in order) ends[link[state]] += ends[state];
    }

    public long DistinctSubstrings()
    {
        long total = 0;
        for (int v = 1; v < states; v++) total += length[v] - length[link[v]];
        return total;
    }

    public int Walk(string pattern)
    {
        int state = 0;
        foreach (char ch in pattern)
        {
            if (!next[state].TryGetValue(ch, out int nxt)) return -1;
            state = nxt;
        }
        return state;
    }

    public bool IsSubstring(string pattern) => Walk(pattern) != -1;

    public int Occurrences(string pattern)
    {
        int state = Walk(pattern);
        return state == -1 ? 0 : ends[state];
    }
}

public static class Solution
{
    private static long BruteDistinctSubstrings(string text)
    {
        var found = new HashSet<string>();
        for (int i = 0; i < text.Length; i++)
        {
            for (int j = i + 1; j <= text.Length; j++) found.Add(text.Substring(i, j - i));
        }
        return found.Count;
    }

    private static int BruteOccurrences(string text, string pattern)
    {
        int count = 0;
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (string.CompareOrdinal(text, i, pattern, 0, pattern.Length) == 0) count++;
        }
        return count;
    }

    private static List<string> LcgStrings(int count, int seed, int maxLen, string alphabet)
    {
        var strings = new List<string>();
        for (int s = 0; s < count; s++)
        {
            int length = 1 + seed % maxLen;
            var chars = new char[length];
            for (int i = 0; i < length; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                chars[i] = alphabet[seed % alphabet.Length];
            }
            strings.Add(new string(chars));
        }
        return strings;
    }

    public static void Main()
    {
        Check(new SuffixAutomaton("").DistinctSubstrings() == 0, "empty");
        Check(new SuffixAutomaton("a").DistinctSubstrings() == 1, "single char");
        Check(new SuffixAutomaton("abab").DistinctSubstrings() == 7, "abab");
        Check(new SuffixAutomaton("aaaa").DistinctSubstrings() == 4, "aaaa");
        Check(new SuffixAutomaton("abc").DistinctSubstrings() == 6, "abc");
        Check(new SuffixAutomaton("ababa").DistinctSubstrings() == 9, "ababa");

        var banana = new SuffixAutomaton("banana");
        Check(banana.DistinctSubstrings() == 15, "banana distinct");
        Check(banana.IsSubstring("ban") && banana.IsSubstring("ana") && banana.IsSubstring("nan"), "banana accept");
        Check(!banana.IsSubstring("bn") && !banana.IsSubstring("bananas"), "banana reject");
        Check(banana.Occurrences("an") == 2 && banana.Occurrences("ana") == 2, "banana occurrences");
        Check(banana.Occurrences("x") == 0, "banana absent");

        var abab = new SuffixAutomaton("abab");
        Check(abab.Occurrences("ab") == 2 && abab.Occurrences("ba") == 1, "abab occurrences");
        Check(abab.Occurrences("abab") == 1, "abab whole");

        var aaaa = new SuffixAutomaton("aaaa");
        Check(aaaa.Occurrences("a") == 4 && aaaa.Occurrences("aa") == 3, "aaaa short");
        Check(aaaa.Occurrences("aaa") == 2 && aaaa.Occurrences("aaaa") == 1, "aaaa long");

        int rejected = 0;
        int accepted = 0;
        foreach (string text in LcgStrings(40, 431, 40, "abc"))
        {
            var auto = new SuffixAutomaton(text);
            Check(auto.DistinctSubstrings() == BruteDistinctSubstrings(text), "random distinct " + text);
            Check(auto.IsSubstring(text), "whole accepted");
            Check(auto.Occurrences(text) == 1, "whole once");
            for (int start = 0; start < text.Length; start++)
            {
                for (int end = start + 1; end <= Math.Min(start + 6, text.Length); end++)
                {
                    string pattern = text.Substring(start, end - start);
                    Check(auto.IsSubstring(pattern), "window accepted");
                    Check(auto.Occurrences(pattern) == BruteOccurrences(text, pattern), "window occurrences");
                    accepted++;
                }
            }
            var mutations = new[] { "d", "bd", "abz", text + "d" };
            foreach (string mutation in mutations)
            {
                if (auto.IsSubstring(mutation))
                {
                    Check(auto.Occurrences(mutation) == BruteOccurrences(text, mutation), "mutation occurrences");
                }
                else
                {
                    Check(BruteOccurrences(text, mutation) == 0, "mutation rejected");
                    rejected++;
                }
            }
        }
        Check(rejected > 0 && accepted > 0, "both verdicts exercised");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
