// Task 051 - Implement dynamic array (basic ArrayList). (Tier: Beginner, CTCI Ch 1)

using System;
using System.Collections.Generic;

public class ListNode
{
    public int Val;
    public ListNode? Next;
    public ListNode(int v) { Val = v; }
}

public static class Solution
{
    public static ListNode FromList(params int[] vals)
    {
        ListNode? head = null;
        for (int i = vals.Length - 1; i >= 0; i--) head = new ListNode(vals[i]) { Next = head };
        return head!;
    }

    public static List<int> ToList(ListNode? head)
    {
        var outList = new List<int>();
        for (var p = head; p != null; p = p.Next) outList.Add(p.Val);
        return outList;
    }
    private class DynamicArray
    {
        private int[] data = new int[2];
        private int size = 0;

        public void PushBack(int value)
        {
            if (size == data.Length) Grow();
            data[size++] = value;
        }

        private void Grow()
        {
            var bigger = new int[data.Length * 2];
            Array.Copy(data, bigger, size);
            data = bigger;
        }

        public int Get(int index) { Check(index); return data[index]; }
        public void Set(int index, int value) { Check(index); data[index] = value; }
        public int PopBack()
        {
            if (size == 0) throw new InvalidOperationException("empty");
            return data[--size];
        }
        private void Check(int index)
        {
            if (index < 0 || index >= size) throw new IndexOutOfRangeException();
        }
        public int Size => size;
        public int Capacity => data.Length;
    }

    public static void Main()
    {
        var a = new Solution.DynamicArray();
        for (int i = 0; i < 5; i++) a.PushBack(i * 10);
        Check(a.Size == 5, "size 5");
        Check(a.Capacity == 8, "capacity doubled");
        Check(a.Get(3) == 30, "get");
        a.Set(0, 99);
        Check(a.Get(0) == 99, "set");
        Check(a.PopBack() == 40 && a.Size == 4, "pop");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
