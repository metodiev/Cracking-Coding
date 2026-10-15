// Task 049 - Implement stack with array (push/pop). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;
using System.Text;

public class StackArray
{
    private readonly int[] data;
    private int top = -1;

    public StackArray(int capacity) { data = new int[capacity]; }

    public void Push(int value)
    {
        if (top + 1 == data.Length) throw new InvalidOperationException("stack full");
        data[++top] = value;
    }

    public int Pop()
    {
        if (IsEmpty) throw new InvalidOperationException("stack empty");
        return data[top--];
    }

    public int Peek()
    {
        if (IsEmpty) throw new InvalidOperationException("stack empty");
        return data[top];
    }

    public bool IsEmpty => top < 0;
    public int Size => top + 1;
}

public static class Solution
{
    public static void Main()
    {
        var s = new StackArray(3);
        Check(s.IsEmpty, "starts empty");
        s.Push(1); s.Push(2); s.Push(3);
        Check(s.Peek() == 3 && s.Size == 3, "after pushes");
        Check(s.Pop() == 3 && s.Pop() == 2 && s.Pop() == 1, "LIFO order");
        Check(s.IsEmpty, "ends empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
