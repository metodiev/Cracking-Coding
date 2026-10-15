// Task 251 - Implement skip list insert/search/delete. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Linq;

public class SkipNode
{
    public int Key, Level;
    public SkipNode[] Next;

    public SkipNode(int key, int level)
    {
        Key = key;
        Level = level;
        Next = new SkipNode[level];
    }
}

public class SkipList
{
    public const int MaxLevel = 6;
    public readonly SkipNode Head = new SkipNode(-1, MaxLevel);
    public int Size;
    private long _state;

    public SkipList(long seed) { _state = seed; }

    // Deterministic coin: 31-bit LCG, bit 16 is the flip.
    private int Coin()
    {
        _state = (_state * 1103515245L + 12345) & 0x7FFFFFFFL;
        return (int)((_state >> 16) & 1);
    }

    private int RandomLevel()
    {
        int level = 1;
        while (level < MaxLevel && Coin() == 1) level++;
        return level;
    }

    // Last node with Key < target at every level.
    private SkipNode[] Find(int key)
    {
        var update = new SkipNode[MaxLevel];
        var node = Head;
        for (int i = MaxLevel - 1; i >= 0; i--)
        {
            while (node.Next[i] != null && node.Next[i].Key < key) node = node.Next[i];
            update[i] = node;
        }
        return update;
    }

    public bool Insert(int key)
    {
        var update = Find(key);
        var after = update[0].Next[0];
        if (after != null && after.Key == key) return false;
        var node = new SkipNode(key, RandomLevel());
        for (int i = 0; i < node.Level; i++)
        {
            node.Next[i] = update[i].Next[i];
            update[i].Next[i] = node;
        }
        Size++;
        return true;
    }

    public bool Search(int key)
    {
        var node = Head;
        for (int i = MaxLevel - 1; i >= 0; i--)
        {
            while (node.Next[i] != null && node.Next[i].Key < key) node = node.Next[i];
            if (node.Next[i] != null && node.Next[i].Key == key) return true;
        }
        return false;
    }

    public bool Delete(int key)
    {
        var update = Find(key);
        var target = update[0].Next[0];
        if (target == null || target.Key != key) return false;
        for (int i = 0; i < target.Level; i++) update[i].Next[i] = target.Next[i];
        Size--;
        return true;
    }

    public List<int> Walk(int level)
    {
        var outList = new List<int>();
        for (var node = Head.Next[level]; node != null; node = node.Next[level]) outList.Add(node.Key);
        return outList;
    }

    // Heights of the level-0 nodes, in sorted walk order.
    public int[] WalkHeights()
    {
        var heights = new List<int>();
        for (var node = Head.Next[0]; node != null; node = node.Next[0]) heights.Add(node.Level);
        return heights.ToArray();
    }
}

public static class Solution
{
    private static readonly int[] Keys = { 3, 6, 7, 9, 12, 19, 17, 26, 21, 25 };
    // Deterministic promotion bits for seed 20251 over Keys, in sorted walk order.
    private static readonly int[] Heights = { 1, 1, 1, 1, 1, 4, 1, 6, 1, 4 };

    public static void Main()
    {
        var sl = new SkipList(20251);
        foreach (int key in Keys) Check(sl.Insert(key), "insert " + key);
        Check(sl.Size == 10, "size after inserts");
        Check(sl.WalkHeights().SequenceEqual(Heights), "identical structure everywhere");
        Check(Heights.Max() == SkipList.MaxLevel, "top lane used");
        Check(sl.Walk(0).SequenceEqual(new[] { 3, 6, 7, 9, 12, 17, 19, 21, 25, 26 }), "level 0 walk sorted");

        foreach (int key in Keys) Check(sl.Search(key), "search hit " + key);
        foreach (int key in new[] { -1, 0, 5, 10, 20, 22, 30 }) Check(!sl.Search(key), "search miss " + key);

        var heights = sl.WalkHeights();
        var walk0 = sl.Walk(0);
        for (int i = 0; i < SkipList.MaxLevel; i++)            // index consistency
        {
            var expected = new List<int>();
            for (int j = 0; j < walk0.Count; j++) if (heights[j] > i) expected.Add(walk0[j]);
            var chain = sl.Walk(i);
            Check(chain.SequenceEqual(expected), "level " + i + " is a sorted subsequence of level 0");
            Check(chain.SequenceEqual(chain.OrderBy(x => x)), "level " + i + " sorted");
        }

        Check(!sl.Insert(3) && sl.Size == 10, "duplicates rejected");

        Check(sl.Delete(19) && sl.Size == 9, "delete 19");
        Check(!sl.Search(19), "19 gone");
        Check(!sl.Delete(19), "second delete fails");
        Check(!sl.Delete(4) && !sl.Delete(100), "missing deletes fail");
        heights = sl.WalkHeights();
        walk0 = sl.Walk(0);
        for (int i = 0; i < SkipList.MaxLevel; i++)
        {
            var expected = new List<int>();
            for (int j = 0; j < walk0.Count; j++) if (heights[j] > i) expected.Add(walk0[j]);
            Check(sl.Walk(i).SequenceEqual(expected), "consistency after delete");
        }

        Check(sl.Delete(3) && sl.Delete(25), "head-adjacent and near-tail keys");
        Check(sl.Delete(21), "top-lane node");
        foreach (int key in new[] { 6, 7, 9, 12, 17, 26 }) Check(sl.Delete(key), "delete " + key);
        Check(sl.Size == 0, "empty");
        for (int i = 0; i < SkipList.MaxLevel; i++) Check(sl.Walk(i).Count == 0, "lane drained");
        foreach (int key in Keys) Check(!sl.Search(key), "no stale key " + key);

        Check(sl.Insert(42) && sl.Search(42) && sl.Size == 1, "reuse after empty");
        Check(sl.Delete(42) && sl.Size == 0, "drain again");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
