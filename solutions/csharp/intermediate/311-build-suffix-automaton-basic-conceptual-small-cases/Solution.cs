// Task 311 - Suffix automaton (basic conceptual, small cases). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class SuffixAutomaton
{
    public readonly List<int> Length = new List<int> { 0 };              // len[v]
    public readonly List<int> Link = new List<int> { -1 };               // suffix link; -1 marks the initial state
    public readonly List<Dictionary<char, int>> Transitions = new List<Dictionary<char, int>> { new Dictionary<char, int>() };
    public int Last;

    public void Extend(char ch)
    {
        int cur = Length.Count;
        Length.Add(Length[Last] + 1);
        Link.Add(0);
        Transitions.Add(new Dictionary<char, int>());
        int p = Last;
        while (p != -1 && !Transitions[p].ContainsKey(ch))
        {
            Transitions[p][ch] = cur;
            p = Link[p];
        }
        if (p == -1)
        {
            Link[cur] = 0;
        }
        else
        {
            int q = Transitions[p][ch];
            if (Length[p] + 1 == Length[q])
            {
                Link[cur] = q;
            }
            else
            {
                int clone = Length.Count;
                Length.Add(Length[p] + 1);
                Link.Add(Link[q]);
                Transitions.Add(new Dictionary<char, int>(Transitions[q]));
                while (p != -1 && Transitions[p].TryGetValue(ch, out int mapped) && mapped == q)
                {
                    Transitions[p][ch] = clone;
                    p = Link[p];
                }
                Link[q] = clone;
                Link[cur] = clone;
            }
        }
        Last = cur;
    }

    /// <summary>DFA walk from the initial state; accepts "" vacuously (the empty substring).</summary>
    public bool Accepts(string word)
    {
        int state = 0;
        foreach (char ch in word)
        {
            if (!Transitions[state].TryGetValue(ch, out int next)) return false;
            state = next;
        }
        return true;
    }

    public int DistinctSubstrings()
    {
        int total = 0;
        for (int v = 1; v < Length.Count; v++) total += Length[v] - Length[Link[v]];
        return total;
    }
}

public static class Solution
{
    // internal because SuffixAutomaton is an internal helper type (C# accessibility rule)
    internal static SuffixAutomaton BuildSuffixAutomaton(string s)
    {
        var automaton = new SuffixAutomaton();
        foreach (char ch in s) automaton.Extend(ch);
        return automaton;
    }

    /// <summary>The independent oracle: every substring of s, collected in a set.</summary>
    public static HashSet<string> BruteSubstrings(string s)
    {
        var outSet = new HashSet<string>();
        for (int i = 0; i < s.Length; i++)
            for (int j = i + 1; j <= s.Length; j++) outSet.Add(s.Substring(i, j - i));
        return outSet;
    }

    public static List<string> AllWords(string alphabet, int maxLength)
    {
        var words = new List<string>();
        var frontier = new List<string> { "" };
        for (int depth = 0; depth < maxLength; depth++)
        {
            var next = new List<string>();
            foreach (string word in frontier)
                foreach (char ch in alphabet) next.Add(word + ch);
            words.AddRange(next);
            frontier = next;
        }
        return words;
    }

    private static void Verify(string s, string alphabet)
    {
        var automaton = BuildSuffixAutomaton(s);
        var substrings = BruteSubstrings(s);
        // 1) the len/link formula agrees with the brute-force substring set
        Check(automaton.DistinctSubstrings() == substrings.Count, "count " + s);
        // 2) every substring of s is accepted by walking the automaton
        foreach (string word in substrings) Check(automaton.Accepts(word), "accept " + s + " " + word);
        // 3) every short word over the alphabet is accepted iff it really is a substring
        //    (the empty word is accepted vacuously; the brute set only holds non-empty ones)
        foreach (string word in AllWords(alphabet, 4))
        {
            if (word.Length == 0) Check(automaton.Accepts(word), "empty " + s);
            else Check(automaton.Accepts(word) == substrings.Contains(word), "member " + s + " " + word);
        }
        // 4) the state count stays within the textbook 2n bound
        Check(automaton.Length.Count <= 2 * s.Length + 1, "state bound " + s);
    }

    private static int seed = 3110;

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
        // "abab" has exactly the 7 distinct substrings a, b, ab, ba, aba, bab, abab
        var abab = BuildSuffixAutomaton("abab");
        Check(BruteSubstrings("abab").SetEquals(new[] { "a", "b", "ab", "ba", "aba", "bab", "abab" }), "abab set");
        Check(abab.DistinctSubstrings() == 7, "abab formula");

        // "aaaa" has 4: a, aa, aaa, aaaa
        var aaaa = BuildSuffixAutomaton("aaaa");
        Check(BruteSubstrings("aaaa").Count == 4, "aaaa brute");
        Check(aaaa.DistinctSubstrings() == 4, "aaaa formula");

        var banana = BuildSuffixAutomaton("banana");
        Check(banana.DistinctSubstrings() == 15 && BruteSubstrings("banana").Count == 15, "banana");

        var empty = BuildSuffixAutomaton("");
        Check(empty.DistinctSubstrings() == 0, "empty count");
        Check(empty.Accepts(""), "empty word accepted");

        Check(!abab.Accepts("bb") && !abab.Accepts("baa") && !abab.Accepts("baab"), "rejections");
        Check(abab.Accepts("bab"), "bab accepted");

        string[] fixedTexts = { "a", "ab", "abab", "banana", "mississippi" };
        string[] fixedAlphabet = { "ab", "ab", "ab", "ban", "mips" };
        for (int i = 0; i < fixedTexts.Length; i++) Verify(fixedTexts[i], fixedAlphabet[i]);
        for (int i = 0; i < 12; i++) Verify(LcgString(10, "ab"), "ab");
        for (int i = 0; i < 8; i++) Verify(LcgString(9, "abc"), "abc");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
