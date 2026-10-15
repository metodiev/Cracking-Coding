// Task 164 - Compute running median using two heaps. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class RunningMedian
    {
        private readonly PriorityQueue<int, int> low = new PriorityQueue<int, int>();
        private readonly PriorityQueue<int, int> high = new PriorityQueue<int, int>();

        public void Add(int value)
        {
            if (low.Count == 0 || value <= low.Peek()) low.Enqueue(value, -value);   // max-heap via negated priority
            else high.Enqueue(value, value);                                         // min-heap
            if (low.Count > high.Count + 1) { var m = low.Dequeue(); high.Enqueue(m, m); }
            else if (high.Count > low.Count) { var m = high.Dequeue(); low.Enqueue(m, -m); }
        }

        public double Median()
        {
            if (low.Count == 0) throw new InvalidOperationException("no values yet");
            if (low.Count == high.Count) return (low.Peek() + high.Peek()) / 2.0;
            return low.Peek();
        }
    }

    private static double[] RunningMedians(int[] stream)
    {
        var rm = new RunningMedian();
        var outArr = new double[stream.Length];
        for (int i = 0; i < stream.Length; i++)
        {
            rm.Add(stream[i]);
            outArr[i] = rm.Median();
        }
        return outArr;
    }

    private static void CheckSeq(int[] stream, double[] expected, string label)
    {
        var got = RunningMedians(stream);
        Check(got.Length == expected.Length, label + " length");
        for (int i = 0; i < expected.Length; i++)
            Check(Math.Abs(got[i] - expected[i]) < 1e-9, label + " at " + i + " -> " + got[i]);
    }

    public static void Main()
    {
        CheckSeq(new[] { 1, 2, 3 }, new[] { 1.0, 1.5, 2.0 }, "1,2,3");
        CheckSeq(new[] { 5, 15, 1, 3 }, new[] { 5.0, 10.0, 5.0, 4.0 }, "5,15,1,3");
        CheckSeq(new[] { 2, 2, 2 }, new[] { 2.0, 2.0, 2.0 }, "duplicates");
        CheckSeq(new[] { -1, -2, -3 }, new[] { -1.0, -1.5, -2.0 }, "negatives");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
