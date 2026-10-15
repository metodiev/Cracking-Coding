// Task 404 - Implement a suffix automaton and count different substrings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A suffix automaton is the minimal DFA recognizing all substrings of s. Each state stands for
// an equivalence class of substrings sharing the same end-position set, with `Length` the
// longest member of the class and `Link` the state for the longest proper suffix outside it.
// Every substring in the class is a suffix of the state's longest string with length in
// (Length(Link), Length], so each class contributes Length - Length(Link) distinct substrings
// and the total is their sum over all states except the initial one.
//
// Anchors: "abab" -> 7, "aaaa" -> 4, "abc" -> 6. Cross-checks: the count matches a brute-force
// substring set, every substring is accepted by walking transitions from the initial state,
// generated non-substrings are rejected, and OccurrenceCount (end-position propagation along
// suffix links in decreasing-length order) matches a naive scan.
using System;
using System.Collections.Generic;
using System.Text;

internal class SuffixAutomaton
{
    private readonly List<Dictionary<char, int>> next = new List<Dictionary<char, int>>();
    private readonly List<int> link = new List<int>();
    private readonly List<int> length = new List<int>();
    private readonly List<bool> isClone = new List<bool>();
    private int last;

    public SuffixAutomaton()
    {
        next.Add(new Dictionary<char, int>());
        link.Add(-1);
        length.Add(0);
        isClone.Add(false);
    }

    public void Extend(char ch)
    {
        int cur = length.Count;
        next.Add(new Dictionary<char, int>());
        link.Add(0);
        length.Add(length[last] + 1);
        isClone.Add(false);
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
                int clone = length.Count;
                next.Add(new Dictionary<char, int>(next[q]));
                link.Add(link[q]);
                length.Add(length[p] + 1);
                isClone.Add(true);
                while (p != -1 && next[p].TryGetValue(ch, out int target) && target == q)
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

    public int StateFor(string t)
    {
        int state = 0;
        foreach (char ch in t)
        {
            if (!next[state].TryGetValue(ch, out int nxt)) return -1;
            state = nxt;
        }
        return state;
    }

    public bool Contains(string t) => StateFor(t) != -1;

    public long DistinctSubstrings()
    {
        long total = 0;
        for (int v = 1; v < length.Count; v++) total += length[v] - length[link[v]];
        return total;
    }

    public int[] EndposSizes()
    {
        int states = length.Count;
        var order = new List<int>();
        for (int v = 1; v < states; v++) order.Add(v);
        order.Sort((a, b) => length[b].CompareTo(length[a]));
        var count = new int[states];
        for (int v = 1; v < states; v++) count[v] = isClone[v] ? 0 : 1;
        foreach (int v in order) count[link[v]] += count[v];
        return count;
    }

    public int OccurrenceCount(string t)
    {
        int state = StateFor(t);
        return state == -1 ? 0 : EndposSizes()[state];
    }

    public static SuffixAutomaton Build(string s)
    {
        var automaton = new SuffixAutomaton();
        foreach (char ch in s) automaton.Extend(ch);
        return automaton;
    }
}

internal static class SubstringTools
{
    public static HashSet<string> AllSubstrings(string s)
    {
        var seen = new HashSet<string>();
        for (int i = 0; i < s.Length; i++)
            for (int j = i + 1; j <= s.Length; j++)
                seen.Add(s.Substring(i, j - i));
        return seen;
    }

    public static int CountOccurrences(string s, string t)
    {
        if (t.Length == 0) return 0;
        int count = 0;
        for (int i = 0; i + t.Length <= s.Length; i++)
        {
            if (string.CompareOrdinal(s, i, t, 0, t.Length) == 0) count++;
        }
        return count;
    }

    public static int CompareSuffixes(string s, int i, int j)
    {
        int n = s.Length;
        while (i < n && j < n)
        {
            if (s[i] != s[j]) return s[i] < s[j] ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;
    }

    public static int[] SuffixArrayNaive(string s)
    {
        int n = s.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => CompareSuffixes(s, a, b));
        return idx;
    }

    public static int[] Kasai(string s, int[] sa)
    {
        int n = s.Length;
        var lcp = new int[n];
        var rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int h = 0;
        for (int i = 0; i < n; i++)
        {
            int r = rank[i];
            if (r == 0)
            {
                h = 0;
                continue;
            }
            int j = sa[r - 1];
            while (i + h < n && j + h < n && s[i + h] == s[j + h]) h++;
            lcp[r] = h;
            if (h > 0) h--;
        }
        return lcp;
    }

    public static int Sum(int[] values)
    {
        int total = 0;
        foreach (int value in values) total += value;
        return total;
    }
}

public static class Solution
{
    private static int seed = 2718;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        var abab = SuffixAutomaton.Build("abab");
        Check(abab.DistinctSubstrings() == 7, "abab count");
        Check(SubstringTools.AllSubstrings("abab").Count == 7, "abab brute");

        Check(SuffixAutomaton.Build("aaaa").DistinctSubstrings() == 4, "aaaa count");
        Check(SuffixAutomaton.Build("").DistinctSubstrings() == 0, "empty");
        Check(SuffixAutomaton.Build("a").DistinctSubstrings() == 1, "single");
        Check(SuffixAutomaton.Build("abc").DistinctSubstrings() == 6, "abc");
        Check(SuffixAutomaton.Build("banana").DistinctSubstrings() == 15, "banana");
        Check(abab.Contains(""), "empty accepted");
        Check(!abab.Contains("baba"), "baba rejected");
        Check(abab.StateFor("abab") != -1 && abab.StateFor("ababa") == -1, "abab state");

        foreach (string s in new[] { "", "a", "aa", "ab", "banana", "mississippi", "abababab", "aabbaabb" })
        {
            var automaton = SuffixAutomaton.Build(s);
            var substrings = SubstringTools.AllSubstrings(s);
            Check(automaton.DistinctSubstrings() == substrings.Count, "distinct " + s);
            var lcp = SubstringTools.Kasai(s, SubstringTools.SuffixArrayNaive(s));
            Check(automaton.DistinctSubstrings() == s.Length * (s.Length + 1) / 2 - SubstringTools.Sum(lcp),
                "identity " + s);
        }

        var ababSubstrings = SubstringTools.AllSubstrings("abab");
        foreach (string t in ababSubstrings) Check(abab.Contains(t), "accept " + t);
        foreach (string t in new[] { "c", "bb", "abb", "baa", "ababa", "babab", "xyz" })
        {
            Check(!abab.Contains(t), "reject " + t);
        }
        foreach (string a in ababSubstrings)
        {
            foreach (string b in ababSubstrings)
            {
                string joined = a + b;
                Check(abab.Contains(joined) == ababSubstrings.Contains(joined), "join " + joined);
            }
        }

        foreach (string s in new[] { "banana", "abab", "aaaa", "mississippi", "abababab" })
        {
            var automaton = SuffixAutomaton.Build(s);
            foreach (string t in SubstringTools.AllSubstrings(s))
            {
                Check(automaton.OccurrenceCount(t) == SubstringTools.CountOccurrences(s, t), "occ " + s + "/" + t);
            }
            Check(automaton.OccurrenceCount("xx") == 0, "absent " + s);
        }
        Check(SuffixAutomaton.Build("banana").OccurrenceCount("ana") == 2, "ana");
        Check(SuffixAutomaton.Build("banana").OccurrenceCount("a") == 3, "a");
        Check(SuffixAutomaton.Build("mississippi").OccurrenceCount("issi") == 2, "issi");

        foreach (string alphabet in new[] { "ab", "abc" })
        {
            for (int trial = 0; trial < 30; trial++)
            {
                string s = LcgString(20, alphabet);
                var automaton = SuffixAutomaton.Build(s);
                var substrings = SubstringTools.AllSubstrings(s);
                Check(automaton.DistinctSubstrings() == substrings.Count, "lcg distinct " + s);
                foreach (string t in substrings)
                {
                    Check(automaton.Contains(t), "lcg accept " + t);
                    Check(automaton.OccurrenceCount(t) == SubstringTools.CountOccurrences(s, t), "lcg occ " + t);
                }
                foreach (string t in new[]
                {
                    alphabet[0].ToString() + alphabet[alphabet.Length - 1] + alphabet[0], "zzz", "q", s + alphabet[0],
                })
                {
                    if (!substrings.Contains(t)) Check(!automaton.Contains(t), "lcg reject " + t);
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
