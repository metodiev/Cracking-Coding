// Task 314 - Aho-Corasick multi-pattern matching (outline). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal class AhoCorasick
{
    private readonly List<Dictionary<char, int>> gotoMap = new List<Dictionary<char, int>> { new Dictionary<char, int>() };
    private readonly List<int> fail = new List<int> { 0 };
    private readonly List<HashSet<string>> output = new List<HashSet<string>> { new HashSet<string>() };
    private readonly List<string> patterns = new List<string>();

    /// <summary>Patterns are assumed non-empty.</summary>
    public AhoCorasick(IEnumerable<string> patterns)
    {
        this.patterns.AddRange(patterns);
        foreach (string pattern in this.patterns)
        {
            int node = 0;
            foreach (char ch in pattern)
            {
                if (!gotoMap[node].TryGetValue(ch, out int next))
                {
                    next = gotoMap.Count;
                    gotoMap[node][ch] = next;
                    gotoMap.Add(new Dictionary<char, int>());
                    fail.Add(0);
                    output.Add(new HashSet<string>());
                }
                node = next;
            }
            output[node].Add(pattern);
        }
        // BFS over the trie: depth-1 states already fail to the root
        var queue = new Queue<int>(gotoMap[0].Values);
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            output[node].UnionWith(output[fail[node]]);
            foreach (var entry in gotoMap[node])
            {
                char ch = entry.Key;
                int child = entry.Value;
                int f = fail[node];
                while (f != 0 && !gotoMap[f].ContainsKey(ch)) f = fail[f];
                fail[child] = gotoMap[f].TryGetValue(ch, out int target) ? target : 0;
                queue.Enqueue(child);
            }
        }
    }

    /// <summary>pattern -> ascending list of start indices (overlapping matches included).</summary>
    public List<KeyValuePair<string, List<int>>> Search(string text)
    {
        var found = new Dictionary<string, List<int>>();
        foreach (string pattern in patterns) found[pattern] = new List<int>();
        int node = 0;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            while (node != 0 && !gotoMap[node].ContainsKey(ch)) node = fail[node];
            node = gotoMap[node].TryGetValue(ch, out int next) ? next : 0;
            foreach (string pattern in output[node]) found[pattern].Add(i - pattern.Length + 1);
        }
        return patterns.Select(p => new KeyValuePair<string, List<int>>(p, found[p])).ToList();
    }
}

public static class Solution
{
    /// <summary>Independent oracle: scan each pattern on its own.</summary>
    public static List<KeyValuePair<string, List<int>>> NaiveSearchAll(string text, IEnumerable<string> patterns)
    {
        var results = new List<KeyValuePair<string, List<int>>>();
        foreach (string pattern in patterns)
        {
            var hits = new List<int>();
            for (int i = 0; i + pattern.Length <= text.Length; i++)
            {
                if (string.CompareOrdinal(text.Substring(i, pattern.Length), pattern) == 0) hits.Add(i);
            }
            results.Add(new KeyValuePair<string, List<int>>(pattern, hits));
        }
        return results;
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

    public static string Format(List<KeyValuePair<string, List<int>>> result)
    {
        return string.Join(";", result.Select(entry => entry.Key + "=[" + string.Join(",", entry.Value) + "]"));
    }

    private static int seed = 3140;

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
        // Classic example: hand-checked against "ushers" (u0 s1 h2 e3 r4 s5)
        var classic = new List<string> { "he", "she", "his", "hers" };
        var automaton = new AhoCorasick(classic);
        Check(Format(automaton.Search("ushers")) == "he=[2];she=[1];his=[];hers=[2]", "ushers");
        Check(Format(automaton.Search("ushers")) == Format(NaiveSearchAll("ushers", classic)), "ushers oracle");

        // Overlapping matches are all reported
        Check(Format(new AhoCorasick(new[] { "aa", "aaa" }).Search("aaaa")) == "aa=[0,1,2];aaa=[0,1]", "overlaps");
        var single = new AhoCorasick(new[] { "a", "ab" });
        Check(Format(single.Search("ab")) == "a=[0];ab=[0]", "single");
        Check(Format(single.Search("")) == "a=[];ab=[]", "empty text");

        foreach (string alphabet in new[] { "ab", "abc" })
        {
            var patternSets = new List<List<string>>
            {
                AllWords(alphabet, 3),
                Enumerable.Range(1, 3).Select(k => new string(alphabet[0], k)).ToList(),
                AllWords(alphabet, 2).Where(word => word.Length == 2).ToList(),
            };
            var texts = new List<string> { "", alphabet, string.Concat(Enumerable.Repeat(alphabet, 4)), new string(alphabet[0], 6) };
            for (int i = 0; i < 20; i++) texts.Add(LcgString(18, alphabet));

            foreach (var patterns in patternSets)
            {
                var ac = new AhoCorasick(patterns);
                foreach (string text in texts)
                {
                    var result = ac.Search(text);
                    Check(Format(result) == Format(NaiveSearchAll(text, patterns)), "cross-check " + text);
                    foreach (var entry in result)
                    {
                        foreach (int start in entry.Value)               // every report is a real match
                        {
                            Check(string.CompareOrdinal(text.Substring(start, entry.Key.Length), entry.Key) == 0,
                                "report " + text + " " + entry.Key);
                        }
                    }
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
