// Task 067 - Implement max-heap push/pop. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Text;

public class MaxHeap
{
    private readonly List<int> data = new List<int>();

    public void Push(int value)
    {
        data.Add(value);
        SiftUp(data.Count - 1);
    }

    public int Pop()
    {
        if (data.Count == 0) throw new InvalidOperationException("heap empty");
        int top = data[0];
        int last = data[data.Count - 1];
        data.RemoveAt(data.Count - 1);
        if (data.Count > 0)
        {
            data[0] = last;
            SiftDown(0);
        }
        return top;
    }

    public int Peek()
    {
        if (data.Count == 0) throw new InvalidOperationException("heap empty");
        return data[0];
    }

    public int Size => data.Count;

    private void SiftUp(int i)
    {
        while (i > 0)
        {
            int parent = (i - 1) / 2;
            if (data[parent] >= data[i]) break;
            Swap(parent, i);
            i = parent;
        }
    }

    private void SiftDown(int i)
    {
        while (true)
        {
            int largest = i, left = 2 * i + 1, right = 2 * i + 2;
            if (left < data.Count && data[left] > data[largest]) largest = left;
            if (right < data.Count && data[right] > data[largest]) largest = right;
            if (largest == i) break;
            Swap(i, largest);
            i = largest;
        }
    }

    private void Swap(int a, int b)
    {
        (data[a], data[b]) = (data[b], data[a]);
    }
}

public static class Solution
{
    public static void Main()
    {
        var h = new MaxHeap();
        foreach (var v in new[] { 5, 3, 8, 1, 9, 2 }) h.Push(v);
        Check(h.Peek() == 9, "max on top");
        var drained = new StringBuilder();
        for (int i = 0; i < 6; i++) drained.Append(h.Pop());
        Check(drained.ToString() == "985321", "descending drain, got " + drained);
        Check(h.Size == 0, "emptied");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
