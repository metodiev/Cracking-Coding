// Task 489 - Implement suffix-prefix automaton based string counting (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Documented clean task: given a finite set of strings S, count the distinct strings x that are both
// a non-empty prefix of some s in S and a non-empty suffix of some s in S, i.e. |P ∩ Q| where
// P = {s[:i] : s in S, 1 <= i <= |s|} and Q = {s[j:] : s in S, 0 <= j < |s|}.
//
// Automaton approach: build a trie over all prefixes of S (every trie node corresponds to a member
// of P, and the trie accepts exactly P - it is the "prefix automaton" of S). For each suffix
// x = s[j:] of every s in S, feed x to the automaton from the root in one walk: x is accepted iff
// x in P, so the walk decides membership x in P ∩ Q. A set of the accepted x counts each string
// once. The walk costs O(|x|) per suffix, O(sum |s|^2) overall - documented as fine for the small
// sets used here.
//
// The result is asserted against a brute-force set intersection on small LCG-generated sets
// (30 sets of 5 strings over {a, b}, lengths 1..12) and on hand cases; the random-set count total is
// pinned at 330 and the larger {a,b,c} case at 11. Hand case: {ababa, baba} -> 5
// (a, ba, aba, baba, ababa).
using System;
using System.Collections.Generic;

internal class TrieNode
{
    public readonly Dictionary<char, TrieNode> Children = new Dictionary<char, TrieNode>();
}

internal class Trie
{
    public readonly TrieNode Root = new TrieNode();
    public int Size;

    public void Insert(string word)
    {
        TrieNode node = Root;
        foreach (char ch in word)
        {
            TrieNode next;
            if (!node.Children.TryGetValue(ch, out next))
            {
                next = new TrieNode();
                node.Children[ch] = next;
                Size++;
            }
            node = next;
        }
    }

    public bool Accepts(string word)
    {
        TrieNode node = Root;
        foreach (char ch in word)
        {
            if (!node.Children.TryGetValue(ch, out node)) return false;
        }
        return true;
    }
}

public static class Solution
{
    private static Trie BuildPrefixAutomaton(string[] strings)
    {
        var trie = new Trie();
        foreach (string s in strings)
        {
            for (int i = 1; i <= s.Length; i++) trie.Insert(s.Substring(0, i));
        }
        return trie;
    }

    private static int[] CountPrefixSuffixIntersection(string[] strings, out HashSet<string> matchedOut)
    {
        Trie trie = BuildPrefixAutomaton(strings);
        var matched = new HashSet<string>();
        int walked = 0;
        foreach (string s in strings)
        {
            for (int j = 0; j < s.Length; j++)
            {
                string x = s.Substring(j);
                walked++;
                if (trie.Accepts(x)) matched.Add(x);
            }
        }
        matchedOut = matched;
        return new[] { matched.Count, walked, trie.Size };
    }

    private static HashSet<string> BrutePrefixSuffixIntersection(string[] strings)
    {
        var prefixes = new HashSet<string>();
        var suffixes = new HashSet<string>();
        foreach (string s in strings)
        {
            for (int i = 1; i <= s.Length; i++) prefixes.Add(s.Substring(0, i));
            for (int j = 0; j < s.Length; j++) suffixes.Add(s.Substring(j));
        }
        prefixes.IntersectWith(suffixes);
        return prefixes;
    }

    private static string[] LcgStrings(int count, int seed, int maxLength, string alphabet)
    {
        var strings = new string[count];
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            int length = 1 + seed % maxLength;
            var chars = new char[length];
            for (int i = 0; i < length; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                chars[i] = alphabet[seed % alphabet.Length];
            }
            strings[k] = new string(chars);
        }
        return strings;
    }

    private static int TotalSuffixes(string[] strings)
    {
        int total = 0;
        foreach (string s in strings) total += s.Length;
        return total;
    }

    private static int PrefixCount(string[] strings)
    {
        var prefixes = new HashSet<string>();
        foreach (string s in strings)
        {
            for (int i = 1; i <= s.Length; i++) prefixes.Add(s.Substring(0, i));
        }
        return prefixes.Count;
    }

    private static bool SameSet(HashSet<string> set, params string[] expected)
    {
        if (set.Count != expected.Length) return false;
        foreach (string value in expected) if (!set.Contains(value)) return false;
        return true;
    }

    public static void Main()
    {
        HashSet<string> matched;
        int[] result = CountPrefixSuffixIntersection(new[] { "ababa", "baba" }, out matched);
        Check(result[0] == 5, "ababa hand " + result[0]);
        Check(SameSet(BrutePrefixSuffixIntersection(new[] { "ababa", "baba" }), "a", "ba", "aba", "baba", "ababa"), "ababa brute");
        result = CountPrefixSuffixIntersection(new[] { "abc" }, out matched);
        Check(result[0] == 1, "abc hand");
        Check(SameSet(BrutePrefixSuffixIntersection(new[] { "abc" }), "abc"), "abc brute");
        result = CountPrefixSuffixIntersection(new[] { "a" }, out matched);
        Check(result[0] == 1, "single char");
        result = CountPrefixSuffixIntersection(new[] { "ab", "ba" }, out matched);
        Check(result[0] == 4, "ab ba hand");
        Check(SameSet(BrutePrefixSuffixIntersection(new[] { "ab", "ba" }), "a", "b", "ab", "ba"), "ab ba brute");
        result = CountPrefixSuffixIntersection(new[] { "aaaa" }, out matched);
        Check(result[0] == 4, "aaaa hand");

        var automatonCases = new[]
        {
            new[] { "ababa", "baba" },
            new[] { "abc", "cba", "bac" }
        };
        foreach (string[] strings in automatonCases)
        {
            Trie trie = BuildPrefixAutomaton(strings);
            Check(trie.Size == PrefixCount(strings), "automaton size");
            var prefixes = new HashSet<string>();
            foreach (string s in strings)
            {
                for (int i = 1; i <= s.Length; i++) prefixes.Add(s.Substring(0, i));
            }
            foreach (string x in prefixes) Check(trie.Accepts(x), "accepts " + x);
            Check(!trie.Accepts("zzz"), "rejects zzz");
        }

        int total = 0;
        for (int k = 0; k < 30; k++)
        {
            string[] strings = LcgStrings(5, 489 + k, 12, "ab");
            result = CountPrefixSuffixIntersection(strings, out matched);
            HashSet<string> expected = BrutePrefixSuffixIntersection(strings);
            Check(result[0] == expected.Count, "random set " + k + " " + result[0] + " " + expected.Count);
            Check(result[1] == TotalSuffixes(strings), "walked " + k);
            Check(result[2] == PrefixCount(strings), "nodes " + k);
            total += result[0];
        }
        Check(total == 330, "random total " + total);

        string[] large = LcgStrings(6, 777, 14, "abc");
        result = CountPrefixSuffixIntersection(large, out matched);
        HashSet<string> largeExpected = BrutePrefixSuffixIntersection(large);
        Check(result[0] == largeExpected.Count && result[0] == 11, "large case " + result[0]);
        Check(result[1] == TotalSuffixes(large), "large walked");
        Check(result[2] == PrefixCount(large), "large nodes");

        int[] duplicate = CountPrefixSuffixIntersection(new[] { "abc", "abc", "abc" }, out matched);
        int[] single = CountPrefixSuffixIntersection(new[] { "abc" }, out matched);
        Check(duplicate[0] == single[0], "duplicates collapse");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
