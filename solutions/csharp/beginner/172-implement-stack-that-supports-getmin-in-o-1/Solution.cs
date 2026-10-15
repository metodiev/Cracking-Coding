// Task 172 - Implement stack that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class MinStack
    {
        private readonly Stack<int> main = new Stack<int>();
        private readonly Stack<int> mins = new Stack<int>();

        public void Push(int value)
        {
            main.Push(value);
            mins.Push(mins.Count == 0 ? value : Math.Min(value, mins.Peek()));
        }

        public int Pop()
        {
            if (main.Count == 0) throw new InvalidOperationException("stack empty");
            mins.Pop();
            return main.Pop();
        }

        public int Peek() => main.Count == 0 ? throw new InvalidOperationException("stack empty") : main.Peek();
        public int Min() => mins.Count == 0 ? throw new InvalidOperationException("stack empty") : mins.Peek();
        public bool IsEmpty => main.Count == 0;
        public int Size => main.Count;
    }

    public static void Main()
    {
        var s = new MinStack();
        foreach (var v in new[] { 5, 3, 7, 3, 9 }) s.Push(v);
        Check(s.Min() == 3 && s.Peek() == 9 && s.Size == 5, "state after pushes");
        Check(s.Pop() == 9 && s.Min() == 3, "pop 9");
        s.Push(1);
        Check(s.Min() == 1, "new minimum");
        Check(s.Pop() == 1 && s.Min() == 3, "pop 1");
        Check(s.Pop() == 3 && s.Min() == 3, "duplicate minimum preserved");
        Check(s.Pop() == 7 && s.Min() == 3, "pop 7");
        Check(s.Pop() == 3 && s.Min() == 5, "back to 5");
        Check(s.Pop() == 5 && s.IsEmpty, "drained");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
