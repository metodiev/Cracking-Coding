// Task 179 - Implement simple heap-based median-of-stream class. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Hand-rolled binary heap; maxFirst=true gives a max-heap on top of ints.</summary>
    private class IntHeap
    {
        private readonly List<int> data = new List<int>();
        private readonly bool maxFirst;

        public IntHeap(bool maxFirst) { this.maxFirst = maxFirst; }

        private bool Better(int a, int b) => maxFirst ? a > b : a < b;

        public void Push(int value)
        {
            data.Add(value);
            int i = data.Count - 1;
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (!Better(data[i], data[parent])) break;
                (data[i], data[parent]) = (data[parent], data[i]);
                i = parent;
            }
        }

        public int Pop()
        {
            if (data.Count == 0) throw new InvalidOperationException("heap empty");
            int top = data[0];
            int last = data[data.Count - 1];
            data.RemoveAt(data.Count - 1);
            if (data.Count > 0)
            {
                data[0] = last;                       // move last to the root, then sift down
                int i = 0;
                while (true)
                {
                    int best = i, left = 2 * i + 1, right = 2 * i + 2;
                    if (left < data.Count && Better(data[left], data[best])) best = left;
                    if (right < data.Count && Better(data[right], data[best])) best = right;
                    if (best == i) break;
                    (data[i], data[best]) = (data[best], data[i]);
                    i = best;
                }
            }
            return top;
        }

        public int Peek()
        {
            if (data.Count == 0) throw new InvalidOperationException("heap empty");
            return data[0];
        }

        public int Size => data.Count;
    }

    private class MedianOfStream
    {
        private readonly IntHeap low = new IntHeap(true), high = new IntHeap(false);
        private int count = 0;

        public void Add(int value)
        {
            count++;
            if (low.Size == 0 || value <= low.Peek()) low.Push(value);
            else high.Push(value);
            if (low.Size > high.Size + 1) high.Push(low.Pop());
            else if (high.Size > low.Size) low.Push(high.Pop());
        }

        public double Median()
        {
            if (count == 0) throw new InvalidOperationException("no values yet");
            if (low.Size == high.Size) return (low.Peek() + high.Peek()) / 2.0;
            return low.Peek();
        }
    }

    public static void Main()
    {
        var m = new MedianOfStream();
        var values = new[] { 1, 2, 3, 4, 5 };
        var expected = new[] { 1.0, 1.5, 2.0, 2.5, 3.0 };
        for (int i = 0; i < values.Length; i++)
        {
            m.Add(values[i]);
            Check(m.Median() == expected[i], "stream " + values[i]);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
