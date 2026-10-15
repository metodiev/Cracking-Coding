// Task 269 - Word ladder all shortest sequences (BFS + backtracking). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>
    /// All shortest transformation sequences (each step changes one letter; every
    /// intermediate word must be in the word list). BFS over the wildcard bucket index
    /// assigns minimum distances; the DFS then walks the LAYERED DAG, following only
    /// edges u -> v with dist[v] == dist[u] + 1. That keeps every shortest path exactly
    /// once (dist strictly increases, so no word repeats within a path) and prunes all
    /// non-shortest continuations. Returned sequences are sorted element-wise
    /// lexicographically so hash-iteration order never leaks into the result.
    /// Empty when end is not in the list, the lengths differ, or end is unreachable;
    /// [[begin]] when begin == end (and end is in the list).
    /// </summary>
    public static List<List<string>> FindLadders(string begin, string end, IList<string> wordList)
    {
        var words = new HashSet<string>(wordList);
        var result = new List<List<string>>();
        if (!words.Contains(end) || begin.Length != end.Length) return result;
        if (begin == end)
        {
            result.Add(new List<string> { begin });
            return result;
        }
        int length = begin.Length;
        var buckets = new Dictionary<string, List<string>>();
        foreach (var word in words)
        {
            if (word.Length != length) continue;
            for (int i = 0; i < length; i++)
            {
                string pattern = word.Substring(0, i) + "*" + word.Substring(i + 1);
                if (!buckets.TryGetValue(pattern, out var list))
                {
                    list = new List<string>();
                    buckets[pattern] = list;
                }
                list.Add(word);
            }
        }

        var dist = new Dictionary<string, int> { [begin] = 0 };   // BFS: minimum distance per word
        var queue = new Queue<string>();
        queue.Enqueue(begin);
        while (queue.Count > 0)
        {
            string word = queue.Dequeue();
            for (int i = 0; i < length; i++)
            {
                if (!buckets.TryGetValue(word.Substring(0, i) + "*" + word.Substring(i + 1), out var neighbours)) continue;
                foreach (var nxt in neighbours)
                    if (!dist.ContainsKey(nxt))
                    {
                        dist[nxt] = dist[word] + 1;
                        queue.Enqueue(nxt);
                    }
            }
        }
        if (!dist.ContainsKey(end)) return result;

        int target = dist[end];
        var path = new List<string> { begin };
        Dfs(begin, end, target, length, buckets, dist, path, result);
        SortPaths(result);
        return result;
    }

    private static void Dfs(string word, string end, int target, int length,
                            Dictionary<string, List<string>> buckets, Dictionary<string, int> dist,
                            List<string> path, List<List<string>> result)
    {
        if (word == end)
        {
            result.Add(new List<string>(path));
            return;
        }
        if (dist[word] >= target) return;
        for (int i = 0; i < length; i++)
        {
            if (!buckets.TryGetValue(word.Substring(0, i) + "*" + word.Substring(i + 1), out var neighbours)) continue;
            foreach (var nxt in neighbours)
                if (dist.TryGetValue(nxt, out int d) && d == dist[word] + 1)
                {
                    path.Add(nxt);
                    Dfs(nxt, end, target, length, buckets, dist, path, result);
                    path.RemoveAt(path.Count - 1);
                }
        }
    }

    /// <summary>Element-wise lexicographic order (deterministic output).</summary>
    private static void SortPaths(List<List<string>> paths)
    {
        paths.Sort((p, q) =>
        {
            int common = Math.Min(p.Count, q.Count);
            for (int i = 0; i < common; i++)
            {
                int c = string.CompareOrdinal(p[i], q[i]);
                if (c != 0) return c;
            }
            return p.Count.CompareTo(q.Count);
        });
    }

    /// <summary>Enumerate all simple paths, keep the shortest; cross-check only.</summary>
    public static List<List<string>> FindLaddersBrute(string begin, string end, IList<string> wordList)
    {
        var words = new HashSet<string>(wordList);
        var found = new List<List<string>>();
        if (!words.Contains(end) || begin.Length != end.Length) return found;
        if (begin == end)
        {
            found.Add(new List<string> { begin });
            return found;
        }
        DfsBrute(begin, end, words, new List<string> { begin }, new HashSet<string> { begin }, found);
        if (found.Count == 0) return found;
        int best = int.MaxValue;
        foreach (var p in found) best = Math.Min(best, p.Count);
        var result = new List<List<string>>();
        foreach (var p in found) if (p.Count == best) result.Add(p);
        SortPaths(result);
        return result;
    }

    private static void DfsBrute(string word, string end, HashSet<string> words,
                                 List<string> path, HashSet<string> seen, List<List<string>> found)
    {
        if (word == end)
        {
            found.Add(new List<string>(path));
            return;
        }
        if (path.Count > words.Count + 1) return;       // simple paths cannot exceed this
        foreach (var nxt in words)
        {
            if (seen.Contains(nxt) || nxt.Length != word.Length) continue;
            int diff = 0;
            for (int i = 0; i < word.Length && diff <= 1; i++) if (word[i] != nxt[i]) diff++;
            if (diff == 1)
            {
                seen.Add(nxt);
                path.Add(nxt);
                DfsBrute(nxt, end, words, path, seen, found);
                path.RemoveAt(path.Count - 1);
                seen.Remove(nxt);
            }
        }
    }

    public static void Main()
    {
        CheckPaths(FindLadders("hit", "cog", new[] { "hot", "dot", "dog", "lot", "log", "cog" }),
            new[] { new[] { "hit", "hot", "dot", "dog", "cog" }, new[] { "hit", "hot", "lot", "log", "cog" } },
            "classic two sequences");
        CheckPaths(FindLadders("red", "tax", new[] { "ted", "tex", "red", "tax", "tad", "den", "rex", "pee" }),
            new[] { new[] { "red", "rex", "tex", "tax" }, new[] { "red", "ted", "tad", "tax" }, new[] { "red", "ted", "tex", "tax" } },
            "red -> tax three sequences");
        CheckPaths(FindLadders("ab", "ba", new[] { "bb", "aa", "ba" }),
            new[] { new[] { "ab", "aa", "ba" }, new[] { "ab", "bb", "ba" } }, "two short paths");
        CheckPaths(FindLadders("hit", "cog", new[] { "hot", "dot", "dog", "lot", "log" }),
            new string[0][], "unreachable");
        CheckPaths(FindLadders("hot", "hot", new[] { "hot" }), new[] { new[] { "hot" } }, "same word");
        CheckPaths(FindLadders("a", "c", new[] { "a", "b", "c" }), new[] { new[] { "a", "c" } }, "one-letter direct hop");
        CheckPaths(FindLadders("ab", "abc", new[] { "abc" }), new string[0][], "mismatched lengths");
        CheckPaths(FindLadders("hot", "hot", new[] { "dog" }), new string[0][], "end not in list wins");
        CheckPaths(FindLadders("lost", "cost", new[] { "most", "fost", "cost", "cast" }),
            new[] { new[] { "lost", "cost" } }, "direct hop");

        var rng = new Random(269);
        for (int test = 0; test < 200; test++)          // cross-check vs the path-enumeration brute
        {
            var pool = new SortedSet<string>();
            int poolSize = 1 + rng.Next(6);
            while (pool.Count < poolSize)
                pool.Add("" + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)));
            var wordList = new List<string>(pool);
            string begin = "" + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3));
            string end = rng.NextDouble() < 0.5
                ? wordList[rng.Next(wordList.Count)]
                : "" + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3));
            var got = FindLadders(begin, end, wordList);
            var want = FindLaddersBrute(begin, end, wordList);
            bool ok = got.Count == want.Count;
            for (int i = 0; ok && i < want.Count; i++)
            {
                ok = got[i].Count == want[i].Count;
                for (int j = 0; ok && j < want[i].Count; j++) ok = got[i][j] == want[i][j];
            }
            Check(ok, $"random agreement begin={begin} end={end}");
        }
        Console.WriteLine("PASS");
    }

    private static void CheckPaths(List<List<string>> got, string[][] expected, string label)
    {
        bool ok = got.Count == expected.Length;
        for (int i = 0; ok && i < expected.Length; i++)
        {
            ok = got[i].Count == expected[i].Length;
            for (int j = 0; ok && j < expected[i].Length; j++) ok = got[i][j] == expected[i][j];
            if (!ok) label = label + " (row " + i + " got=[" + string.Join(", ", got[i]) + "])";
        }
        Check(ok, label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
