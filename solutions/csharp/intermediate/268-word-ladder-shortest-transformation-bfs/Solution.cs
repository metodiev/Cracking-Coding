// Task 268 - Word ladder shortest transformation (BFS). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>
    /// BFS over the implicit one-letter-difference graph with a wildcard bucket index
    /// (h*t -> [hot, hit, ...]) built once, so all neighbours are found in O(L) instead
    /// of scanning the whole list. Returns the sequence LENGTH in words (begin and end
    /// included), or 0 when no sequence exists (documented, as in LeetCode 127).
    /// end not in the list -> 0; begin == end -> 1; mismatched lengths -> 0; duplicate
    /// entries collapse to one word.
    /// </summary>
    public static int LadderLength(string begin, string end, IList<string> wordList)
    {
        var words = new HashSet<string>(wordList);
        if (!words.Contains(end) || begin.Length != end.Length) return 0;
        if (begin == end) return 1;
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
        var queue = new Queue<string>();
        queue.Enqueue(begin);
        var seen = new HashSet<string> { begin };
        int steps = 1;
        while (queue.Count > 0)
        {
            for (int size = queue.Count; size > 0; size--)
            {
                string word = queue.Dequeue();
                for (int i = 0; i < length; i++)
                {
                    string pattern = word.Substring(0, i) + "*" + word.Substring(i + 1);
                    if (!buckets.TryGetValue(pattern, out var neighbours)) continue;
                    foreach (var nxt in neighbours)
                    {
                        if (nxt == end) return steps + 1;
                        if (seen.Add(nxt)) queue.Enqueue(nxt);
                    }
                }
            }
            steps++;
        }
        return 0;
    }

    /// <summary>Naive BFS that scans every word for a one-letter difference; cross-check only.</summary>
    public static int LadderLengthOracle(string begin, string end, IList<string> wordList)
    {
        var words = new HashSet<string>(wordList);
        if (!words.Contains(end) || begin.Length != end.Length) return 0;
        if (begin == end) return 1;
        var queue = new Queue<(string Word, int Dist)>();
        queue.Enqueue((begin, 1));
        var seen = new HashSet<string> { begin };
        while (queue.Count > 0)
        {
            var (word, dist) = queue.Dequeue();
            foreach (var candidate in words)
            {
                if (seen.Contains(candidate) || candidate.Length != word.Length) continue;
                int diff = 0;
                for (int i = 0; i < word.Length && diff <= 1; i++) if (word[i] != candidate[i]) diff++;
                if (diff == 1)
                {
                    if (candidate == end) return dist + 1;
                    seen.Add(candidate);
                    queue.Enqueue((candidate, dist + 1));
                }
            }
        }
        return 0;
    }

    public static void Main()
    {
        Check(LadderLength("hit", "cog", new[] { "hot", "dot", "dog", "lot", "log", "cog" }) == 5, "hit -> cog = 5");
        Check(LadderLength("hit", "cog", new[] { "hot", "dot", "dog", "lot", "log" }) == 0, "end not in list");
        Check(LadderLength("a", "c", new[] { "a", "b", "c" }) == 2, "one-letter words");
        Check(LadderLength("hit", "hot", new[] { "hot", "dot", "dog" }) == 2, "one step");
        Check(LadderLength("hot", "hot", new[] { "hot" }) == 1, "same word -> 1");
        Check(LadderLength("ab", "abc", new[] { "abc" }) == 0, "mismatched lengths");
        Check(LadderLength("abc", "ab", new[] { "ab" }) == 0, "mismatched lengths reversed");
        Check(LadderLength("lost", "cost", new[] { "most", "fost", "cost", "cast" }) == 2, "lost -> cost direct");
        Check(LadderLength("ab", "cd", new[] { "ac", "bd" }) == 0, "unreachable");
        Check(LadderLength("hit", "cog", new[] { "hot", "hot", "dot", "dog", "lot", "log", "cog" }) == 5,
            "duplicates collapse");
        Check(LadderLength("hot", "hot", new[] { "dog" }) == 0, "end not in list wins over equality");
        Check(LadderLength("aaaa", "aaab", new[] { "aaaa", "aaab" }) == 2, "four-letter words");

        var rng = new Random(268);
        for (int test = 0; test < 300; test++)          // cross-check vs the naive-scan BFS
        {
            var pool = new HashSet<string>();
            int poolSize = 1 + rng.Next(6);
            while (pool.Count < poolSize)
                pool.Add("" + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)));
            var wordList = new List<string>(pool);
            if (rng.NextDouble() < 0.5) wordList.Add(wordList[0]);   // duplicate entry
            string begin = "" + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3));
            string end = rng.NextDouble() < 0.5
                ? wordList[rng.Next(wordList.Count)]
                : "" + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3)) + (char)('a' + rng.Next(3));
            Check(LadderLength(begin, end, wordList) == LadderLengthOracle(begin, end, wordList),
                $"random agreement begin={begin} end={end}");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
