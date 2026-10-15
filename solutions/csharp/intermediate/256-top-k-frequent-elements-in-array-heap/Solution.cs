// Task 256 - Top K frequent elements in array (heap). (Tier: Intermediate, CTCI Ch 10)
using System;
using System.Collections.Generic;

public static class Solution
{
    // Counts occurrences, keeps the k best in a min-heap and reports them
    // frequency-descending with value-ascending tie-breaks. An entry is "worse" when its
    // count is smaller, or - on equal counts - when its value is larger; the heap priority
    // (count, -value) therefore has the worst kept entry at the root. O(n + d log k).
    private static List<int> TopKFrequent(int[] nums, int k)
    {
        var freq = new Dictionary<int, int>();
        foreach (int x in nums)
        {
            freq.TryGetValue(x, out int current);
            freq[x] = current + 1;
        }
        var result = new List<int>();
        if (k <= 0) return result;
        var heap = new PriorityQueue<int, (int Count, int NegValue)>();
        foreach (var pair in freq)
        {
            var entry = (pair.Value, -pair.Key);
            if (heap.Count < k)
            {
                heap.Enqueue(pair.Key, entry);
            }
            else if (heap.TryPeek(out _, out var worst) && entry.CompareTo(worst) > 0)
            {
                heap.Dequeue();
                heap.Enqueue(pair.Key, entry);
            }
        }
        foreach (var item in heap.UnorderedItems) result.Add(item.Element);
        result.Sort((a, b) => freq[a] != freq[b] ? freq[b] - freq[a] : a - b);
        return result;
    }

    public static void Main()
    {
        Check(Render(TopKFrequent(new[] { 1, 1, 1, 2, 2, 3 }, 2)) == "[1, 2]", "classic k=2");
        Check(Render(TopKFrequent(new[] { 1, 1, 1, 2, 2, 3 }, 3)) == "[1, 2, 3]", "classic k=3");
        Check(Render(TopKFrequent(new[] { 1, 1, 1, 2, 2, 3 }, 6)) == "[1, 2, 3]", "k >= distinct");
        Check(Render(TopKFrequent(new[] { 1, 1, 1, 2, 2, 3 }, 99)) == "[1, 2, 3]", "huge k");
        Check(TopKFrequent(new[] { 1, 1, 1, 2, 2, 3 }, 0).Count == 0, "k=0");
        Check(Render(TopKFrequent(new[] { 1, 1, 1, 2, 2, 3 }, 1)) == "[1]", "k=1");

        Check(Render(TopKFrequent(new[] { 4, 4, 5, 5, 6 }, 2)) == "[4, 5]", "count tie");
        Check(Render(TopKFrequent(new[] { 3, 1, 2 }, 2)) == "[1, 2]", "all count 1");
        Check(Render(TopKFrequent(new[] { 2, 2, 3, 3, 1 }, 1)) == "[2]", "tie at the eviction edge");
        Check(Render(TopKFrequent(new[] { -1, -1, 2, 2 }, 2)) == "[-1, 2]", "negatives");
        Check(Render(TopKFrequent(new[] { 5 }, 1)) == "[5]", "single element");
        Check(TopKFrequent(new int[0], 3).Count == 0, "empty array");
        Check(Render(TopKFrequent(new[] { 7, 7, 7 }, 3)) == "[7]", "one distinct value");
        Console.WriteLine("PASS");
    }

    private static string Render(List<int> values)
    {
        return "[" + string.Join(", ", values) + "]";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
