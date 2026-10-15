// Task 340 - Implement palindrome tree (Eertree) overview & small test. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

public class Eertree
{
    // Two roots: node 0 has length -1 (imaginary), node 1 has length 0 (empty). Every other
    // node is a distinct palindromic substring with a suffix link to its longest proper
    // palindromic suffix. Counts start as "how often this node was the longest palindromic
    // suffix" and are then propagated along suffix links, longest node first.
    private readonly string _text;
    private readonly List<int> _length = new List<int> { -1, 0 };
    private readonly List<int> _link = new List<int> { 0, 0 };
    private readonly List<Dictionary<char, int>> _next =
        new List<Dictionary<char, int>> { new Dictionary<char, int>(), new Dictionary<char, int>() };
    private readonly List<int> _occurrences = new List<int> { 0, 0 };
    private readonly List<string?> _palindrome = new List<string?> { null, "" };
    private int _last = 1;

    public Eertree(string text)
    {
        _text = text;
        for (int position = 0; position < text.Length; position++)
        {
            Extend(position, text[position]);
        }
        var nodes = new List<int>();
        for (int node = 2; node < _length.Count; node++) nodes.Add(node);
        nodes.Sort((a, b) => _length[b].CompareTo(_length[a]));
        foreach (int node in nodes)
        {
            _occurrences[_link[node]] += _occurrences[node];
        }
    }

    private void Extend(int position, char ch)
    {
        int current = _last;
        while (true)
        {
            int index = position - 1 - _length[current];
            if (index >= 0 && _text[index] == ch) break;
            current = _link[current];
        }

        if (_next[current].TryGetValue(ch, out int existing))
        {
            _last = existing;
            _occurrences[_last] += 1;
            return;
        }

        int node = _length.Count;
        _length.Add(_length[current] + 2);
        _next.Add(new Dictionary<char, int>());
        _palindrome.Add(_length[node] == 1
            ? ch.ToString() : ch + _palindrome[current] + ch);
        if (_length[node] == 1)
        {
            _link.Add(1);
        }
        else
        {
            int candidate = _link[current];
            while (true)
            {
                int index = position - 1 - _length[candidate];
                if (index >= 0 && _text[index] == ch) break;
                candidate = _link[candidate];
            }
            _link.Add(_next[candidate][ch]);
        }
        _next[current][ch] = node;
        _occurrences.Add(1);
        _last = node;
    }

    public int DistinctCount()
    {
        return _length.Count - 2;
    }

    /// <summary>Map palindrome -> full occurrence count, for every distinct palindromic substring.</summary>
    public SortedDictionary<string, int> Counts()
    {
        var result = new SortedDictionary<string, int>();
        for (int node = 2; node < _length.Count; node++)
        {
            result[_palindrome[node]!] = _occurrences[node];
        }
        return result;
    }
}

public static class Solution
{
    private static int seed = 1717;

    public static HashSet<string> BruteDistinct(string text)
    {
        var found = new HashSet<string>();
        for (int start = 0; start < text.Length; start++)
        {
            for (int end = start + 1; end <= text.Length; end++)
            {
                string piece = text.Substring(start, end - start);
                char[] characters = piece.ToCharArray();
                Array.Reverse(characters);
                if (piece == new string(characters)) found.Add(piece);
            }
        }
        return found;
    }

    public static SortedDictionary<string, int> BruteCounts(string text)
    {
        var counts = new SortedDictionary<string, int>();
        foreach (string piece in BruteDistinct(text))
        {
            int total = 0;
            for (int start = 0; start + piece.Length <= text.Length; start++)
            {
                if (text.Substring(start, piece.Length) == piece) total++;
            }
            counts[piece] = total;
        }
        return counts;
    }

    private static string Render(SortedDictionary<string, int> counts)
    {
        var parts = new List<string>();
        foreach (KeyValuePair<string, int> entry in counts)
        {
            parts.Add(entry.Key + "=" + entry.Value + ";");
        }
        return string.Join("", parts);
    }

    public static void Main()
    {
        // Hand-checked: "abba" has distinct palindromes {a, b, bb, abba}.
        var tree = new Eertree("abba");
        Check(tree.DistinctCount() == 4, "abba distinct count");
        Check(Render(tree.Counts()) == "a=2;abba=1;b=2;bb=1;", "abba counts");
        Check(Render(tree.Counts()) == Render(BruteCounts("abba")), "abba matches brute force");
        Check(tree.DistinctCount() == BruteDistinct("abba").Count, "abba distinct vs brute");

        // "aaaa": four distinct palindromes with counts 4, 3, 2, 1.
        var four = new Eertree("aaaa");
        Check(four.DistinctCount() == 4, "aaaa distinct count");
        Check(Render(four.Counts()) == "a=4;aa=3;aaa=2;aaaa=1;", "aaaa counts");
        Check(Render(four.Counts()) == Render(BruteCounts("aaaa")), "aaaa matches brute force");

        // Edge cases.
        Check(new Eertree("").DistinctCount() == 0, "empty string");
        Check(new Eertree("").Counts().Count == 0, "empty counts");
        var single = new Eertree("x");
        Check(single.DistinctCount() == 1 && Render(single.Counts()) == "x=1;", "single char");
        Check(new Eertree("abcd").DistinctCount() == 4, "no long palindromes");
        Check(Render(new Eertree("abcd").Counts()) == "a=1;b=1;c=1;d=1;", "abcd counts");
        Check(Render(new Eertree("aba").Counts()) == "a=2;aba=1;b=1;", "aba counts");
        Check(Render(new Eertree("abccba").Counts()) == "a=2;abccba=1;b=2;bccb=1;c=2;cc=1;",
            "abccba counts");

        // LCG strings: distinct palindromes and every occurrence count must match brute force.
        for (int trial = 0; trial < 60; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int size = seed % 40;
            seed = (seed * 75 + 74) % 65537;
            string alphabet = seed % 2 == 0 ? "ab" : "abc";
            var builder = new System.Text.StringBuilder();
            for (int i = 0; i < size; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                builder.Append(alphabet[seed % alphabet.Length]);
            }
            string text = builder.ToString();
            var random = new Eertree(text);
            var expected = BruteCounts(text);
            Check(Render(random.Counts()) == Render(expected), "random counts " + trial + " " + text);
            Check(random.DistinctCount() == expected.Count, "random distinct " + trial + " " + text);
            foreach (KeyValuePair<string, int> entry in random.Counts())
            {
                string piece = entry.Key;
                char[] characters = piece.ToCharArray();
                Array.Reverse(characters);
                Check(piece == new string(characters), "palindrome check");
                Check(entry.Value >= 1 && piece.Length <= text.Length, "count bounds");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
