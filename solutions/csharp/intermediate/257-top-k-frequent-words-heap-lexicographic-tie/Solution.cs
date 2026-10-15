// Task 257 - Top K frequent words (heap + lexicographic tie). (Tier: Intermediate, CTCI Ch 10)
using System;
using System.Collections.Generic;

public static class Solution
{
    // The heap priority (count, word) uses a custom comparer that puts the WORST word at the
    // root: lower count first, and on a count tie the lexicographically larger word. So
    // eviction is one dequeue, and the result is frequency-descending, word-ascending.
    private static List<string> TopKFrequentWords(string[] words, int k)
    {
        var freq = new Dictionary<string, int>();
        foreach (string w in words)
        {
            freq.TryGetValue(w, out int current);
            freq[w] = current + 1;
        }
        var result = new List<string>();
        if (k <= 0) return result;
        var comparer = Comparer<(int Count, string Word)>.Create((a, b) =>
        {
            int byCount = a.Count.CompareTo(b.Count);
            return byCount != 0 ? byCount : string.CompareOrdinal(b.Word, a.Word);
        });
        var heap = new PriorityQueue<string, (int Count, string Word)>(comparer);
        foreach (var pair in freq)
        {
            var entry = (pair.Value, pair.Key);
            if (heap.Count < k)
            {
                heap.Enqueue(pair.Key, entry);
            }
            else if (heap.TryPeek(out _, out var worst) && comparer.Compare(entry, worst) > 0)
            {
                heap.Dequeue();
                heap.Enqueue(pair.Key, entry);
            }
        }
        foreach (var item in heap.UnorderedItems) result.Add(item.Element);
        result.Sort((a, b) =>
        {
            int byCount = freq[b].CompareTo(freq[a]);
            return byCount != 0 ? byCount : string.CompareOrdinal(a, b);
        });
        return result;
    }

    public static void Main()
    {
        var classic = new[] { "i", "love", "leetcode", "i", "love", "coding" };
        Check(Render(TopKFrequentWords(classic, 1)) == "[i]", "classic k=1");
        Check(Render(TopKFrequentWords(classic, 2)) == "[i, love]", "classic k=2");
        Check(Render(TopKFrequentWords(classic, 3)) == "[i, love, coding]", "classic k=3");
        Check(Render(TopKFrequentWords(classic, 4)) == "[i, love, coding, leetcode]", "classic k=4");
        Check(Render(TopKFrequentWords(classic, 9)) == "[i, love, coding, leetcode]", "k > distinct");
        Check(TopKFrequentWords(classic, 0).Count == 0, "k=0");

        var other = new[] { "the", "day", "is", "sunny", "the", "the", "sunny", "is", "is" };
        Check(Render(TopKFrequentWords(other, 1)) == "[is]", "is beats the on ties");
        Check(Render(TopKFrequentWords(other, 2)) == "[is, the]", "is, the");
        Check(Render(TopKFrequentWords(other, 4)) == "[is, the, sunny, day]", "full order");

        Check(Render(TopKFrequentWords(new[] { "a", "aa", "aaa" }, 2)) == "[a, aa]", "lex ties");
        Check(Render(TopKFrequentWords(new[] { "a", "aa", "aaa" }, 3)) == "[a, aa, aaa]", "all three");
        var mixed = new[] { "b", "b", "a", "a", "c", "c", "d" };   // a:2 b:2 c:2 d:1
        Check(Render(TopKFrequentWords(mixed, 1)) == "[a]", "eviction tie k=1");
        Check(Render(TopKFrequentWords(mixed, 2)) == "[a, b]", "eviction tie k=2");
        Check(Render(TopKFrequentWords(mixed, 3)) == "[a, b, c]", "eviction tie k=3");
        Check(Render(TopKFrequentWords(new[] { "only" }, 1)) == "[only]", "single word");
        Check(TopKFrequentWords(new string[0], 3).Count == 0, "empty input");
        Console.WriteLine("PASS");
    }

    private static string Render(List<string> values)
    {
        return "[" + string.Join(", ", values) + "]";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
