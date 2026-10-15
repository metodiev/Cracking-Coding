// Task 436 - Implement rope with concatenation and split in O(log n). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Rope = balanced binary tree over text chunks. Documented thresholds: every leaf holds at most
// LEAF_LIMIT = 32 characters (a longer input string is pre-split into such leaves), and after every
// concat/split the root height is checked against 2 * ceil(log2(#leaves + 1)) + 1 - if a rope grows
// taller than that it is rebuilt from its leaves into a perfectly balanced tree. Each node caches its
// character size, height and leaf count, so concat is O(1) plus an amortised rebuild, and split
// recurses along one root-to-leaf path (plus an amortised rebuild of the two results). The rebuild
// counter is documented and asserted: building a 200-character rope by 200 single-character concats
// triggers at least one rebalance. Assertions compare ToString() with a plain-string oracle across
// LCG operation scripts including splits at 0 and at the full length, and check leaf, size and height
// invariants after every operation.
using System;
using System.Collections.Generic;

internal class RopeNode
{
    public readonly string? Text;
    public readonly RopeNode? Left;
    public readonly RopeNode? Right;
    public readonly int Size;
    public readonly int Height;
    public readonly int Leaves;

    public RopeNode(string text)
    {
        Text = text;
        Size = text.Length;
        Height = 1;
        Leaves = 1;
    }

    public RopeNode(RopeNode left, RopeNode right)
    {
        Text = null;
        Left = left;
        Right = right;
        Size = left.Size + right.Size;
        Height = 1 + Math.Max(left.Height, right.Height);
        Leaves = left.Leaves + right.Leaves;
    }
}

internal class Rope
{
    public const int LeafLimit = 32;

    private readonly RopeNode? root;
    public readonly int Rebuilds;

    public Rope(string text)
    {
        List<string> chunks = LeavesOf(text);
        root = BuildBalanced(chunks, 0, chunks.Count - 1);
        Rebuilds = 0;
    }

    private Rope(RopeNode? root, int rebuilds)
    {
        this.root = root;
        Rebuilds = rebuilds;
    }

    private sealed class Rebalanced
    {
        public readonly RopeNode? Root;
        public readonly int Rebuilt;

        public Rebalanced(RopeNode? root, int rebuilt)
        {
            Root = root;
            Rebuilt = rebuilt;
        }
    }

    public static List<string> LeavesOf(string text)
    {
        var chunks = new List<string>();
        for (int i = 0; i < text.Length; i += LeafLimit)
        {
            chunks.Add(text.Substring(i, Math.Min(LeafLimit, text.Length - i)));
        }
        return chunks;
    }

    public static RopeNode? BuildBalanced(List<string> chunks, int lo, int hi)
    {
        if (lo > hi) return null;
        if (lo == hi) return new RopeNode(chunks[lo]);
        int mid = (lo + hi) / 2;
        return new RopeNode(BuildBalanced(chunks, lo, mid)!, BuildBalanced(chunks, mid + 1, hi)!);
    }

    public static int HeightBound(int leafCount)
    {
        return 2 * (int)Math.Ceiling(Math.Log(leafCount + 1) / Math.Log(2)) + 1;
    }

    public static RopeNode? RawConcat(RopeNode? left, RopeNode? right)
    {
        if (left == null) return right;
        if (right == null) return left;
        return new RopeNode(left, right);
    }

    public static RopeNode?[] SplitRoot(RopeNode? node, int index)
    {
        if (node == null) return new RopeNode?[] { null, null };
        if (node.Text != null)
        {
            string head = node.Text.Substring(0, index);
            string tail = node.Text.Substring(index);
            return new RopeNode?[] {
                head.Length == 0 ? null : new RopeNode(head),
                tail.Length == 0 ? null : new RopeNode(tail) };
        }
        int leftSize = node.Left!.Size;
        if (index < leftSize)
        {
            RopeNode?[] parts = SplitRoot(node.Left, index);
            return new RopeNode?[] { parts[0], RawConcat(parts[1], node.Right) };
        }
        if (index == leftSize) return new RopeNode?[] { node.Left, node.Right };
        RopeNode?[] rest = SplitRoot(node.Right, index - leftSize);
        return new RopeNode?[] { RawConcat(node.Left, rest[0]), rest[1] };
    }

    public int Size() => root == null ? 0 : root.Size;

    public int Height() => root == null ? 0 : root.Height;

    public int LeafCount() => root == null ? 0 : root.Leaves;

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        var stack = new Stack<RopeNode?>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            RopeNode? node = stack.Pop();
            if (node == null) continue;
            if (node.Text != null) sb.Append(node.Text);
            else
            {
                stack.Push(node.Right);
                stack.Push(node.Left);
            }
        }
        return sb.ToString();
    }

    public List<string> LeafTexts()
    {
        var outList = new List<string>();
        var stack = new Stack<RopeNode?>();
        stack.Push(root);
        while (stack.Count > 0)
        {
            RopeNode? node = stack.Pop();
            if (node == null) continue;
            if (node.Text != null) outList.Add(node.Text);
            else
            {
                stack.Push(node.Right);
                stack.Push(node.Left);
            }
        }
        return outList;
    }

    public bool Validate()
    {
        int[]? result = Walk(root);
        return result != null && result[0] == Size() && result[1] == Height() && result[2] == LeafCount();
    }

    private static int[]? Walk(RopeNode? node)
    {
        if (node == null) return new[] { 0, 0, 0 };
        if (node.Text != null)
        {
            if (node.Text.Length > LeafLimit || node.Text.Length == 0) return null;
            return new[] { node.Size, node.Height, node.Leaves };
        }
        int[]? left = Walk(node.Left);
        int[]? right = Walk(node.Right);
        if (left == null || right == null) return null;
        if (node.Size != left[0] + right[0]) return null;
        if (node.Height != 1 + Math.Max(left[1], right[1])) return null;
        if (node.Leaves != left[2] + right[2]) return null;
        return new[] { node.Size, node.Height, node.Leaves };
    }

    private Rebalanced Rebalance(RopeNode? node)
    {
        if (node == null || node.Height <= HeightBound(node.Leaves)) return new Rebalanced(node, 0);
        var chunks = new List<string>();
        var stack = new Stack<RopeNode>();
        stack.Push(node);
        while (stack.Count > 0)
        {
            RopeNode current = stack.Pop();
            if (current.Text != null) chunks.Add(current.Text);
            else
            {
                stack.Push(current.Right!);
                stack.Push(current.Left!);
            }
        }
        return new Rebalanced(BuildBalanced(chunks, 0, chunks.Count - 1), 1);
    }

    public Rope Concat(Rope other)
    {
        if (root == null) return new Rope(other.root, Rebuilds + other.Rebuilds);
        if (other.root == null) return new Rope(root, Rebuilds + other.Rebuilds);
        Rebalanced merged = Rebalance(RawConcat(root, other.root));
        return new Rope(merged.Root, Rebuilds + other.Rebuilds + merged.Rebuilt);
    }

    public Rope[] Split(int index)
    {
        if (index < 0 || index > Size()) throw new IndexOutOfRangeException("split index out of bounds");
        RopeNode?[] parts = SplitRoot(root, index);
        Rebalanced first = Rebalance(parts[0]);
        Rebalanced second = Rebalance(parts[1]);
        return new[]
        {
            new Rope(first.Root, Rebuilds + first.Rebuilt),
            new Rope(second.Root, Rebuilds + second.Rebuilt)
        };
    }
}

public static class Solution
{
    private static string ExtraString(int step, int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++) chars[i] = (char)('a' + (step * 3 + i) % 26);
        return new string(chars);
    }

    public static void Main()
    {
        var rope = new Rope("hello world");
        Check(rope.ToString() == "hello world", "hand text");
        Check(rope.Size() == 11, "hand size");
        Check(rope.LeafCount() == 1, "hand leaves");
        Check(rope.Height() == 1, "hand height");
        Check(rope.Validate(), "hand validate");

        Rope[] parts = rope.Split(0);
        Check(parts[0].ToString() == "" && parts[0].Size() == 0, "split 0 left");
        Check(parts[1].ToString() == "hello world", "split 0 right");
        parts = rope.Split(11);
        Check(parts[0].ToString() == "hello world", "split len left");
        Check(parts[1].ToString() == "" && parts[1].Size() == 0, "split len right");
        parts = rope.Split(5);
        Check(parts[0].ToString() == "hello" && parts[1].ToString() == " world", "split middle");
        Check(parts[0].Concat(parts[1]).ToString() == "hello world", "concat back");
        Check(parts[0].Height() <= Rope.HeightBound(parts[0].LeafCount()), "small bound");

        var dataChars = new char[2000];
        for (int i = 0; i < 2000; i++) dataChars[i] = (char)('a' + (i * 7 + i * i) % 26);
        string data = new string(dataChars);
        rope = new Rope(data);
        Check(rope.ToString() == data, "long text");
        Check(rope.Size() == 2000, "long size");
        Check(rope.LeafCount() == (2000 + Rope.LeafLimit - 1) / Rope.LeafLimit, "long leaves");
        Check(rope.Height() <= Rope.HeightBound(rope.LeafCount()), "long height");
        Check(rope.Validate(), "long validate");
        int maxLeaf = 0;
        foreach (string chunk in rope.LeafTexts()) maxLeaf = Math.Max(maxLeaf, chunk.Length);
        Check(maxLeaf <= Rope.LeafLimit, "leaf limit");
        parts = rope.Split(999);
        Check(parts[0].ToString() + parts[1].ToString() == data, "long split parts");
        Check(parts[0].Concat(parts[1]).ToString() == data, "long concat back");

        rope = new Rope("");
        for (int index = 0; index < 200; index++)
        {
            rope = rope.Concat(new Rope(((char)('a' + index % 26)).ToString()));
        }
        Check(rope.Size() == 200, "build size");
        Check(rope.Rebuilds >= 1, "build rebuilds " + rope.Rebuilds);
        var expectedBuilt = new char[200];
        for (int i = 0; i < 200; i++) expectedBuilt[i] = (char)('a' + i % 26);
        Check(rope.ToString() == new string(expectedBuilt), "build text");
        Check(rope.Height() <= Rope.HeightBound(rope.LeafCount()), "build height");
        Check(rope.Validate(), "build validate");

        var initialChars = new char[64];
        for (int i = 0; i < 64; i++) initialChars[i] = (char)('a' + (i * 13 + 5) % 26);
        rope = new Rope(new string(initialChars));
        string oracle = new string(initialChars);
        int seed = 4360;
        for (int step = 0; step < 100; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 4;
            if (op == 0)
            {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.Length + 1);
                Rope[] halves = rope.Split(index);
                Check(halves[0].ToString() == oracle.Substring(0, index), "script split left " + step);
                Check(halves[1].ToString() == oracle.Substring(index), "script split right " + step);
                if (seed % 2 == 0)
                {
                    rope = halves[0].Concat(halves[1]);
                }
                else
                {
                    rope = halves[1].Concat(halves[0]);
                    oracle = oracle.Substring(index) + oracle.Substring(0, index);
                }
            }
            else if (op == 1)
            {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.Length + 1);
                rope = rope.Split(index)[0];
                oracle = oracle.Substring(0, index);
            }
            else if (op == 2)
            {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.Length + 1);
                rope = rope.Split(index)[1];
                oracle = oracle.Substring(index);
            }
            else
            {
                string extra = ExtraString(step, seed % 20);
                rope = rope.Concat(new Rope(extra));
                oracle += extra;
            }
            Check(rope.ToString() == oracle, "script text " + step);
            Check(rope.Size() == oracle.Length, "script size " + step);
            Check(rope.Height() <= Rope.HeightBound(rope.LeafCount()), "script height " + step);
            Check(rope.Validate(), "script validate " + step);
            Check(string.Join("", rope.LeafTexts()) == oracle, "script leaves " + step);
        }
        parts = rope.Split(0);
        Check(parts[0].ToString() == "" && parts[1].ToString() == oracle, "final split 0");
        parts = rope.Split(rope.Size());
        Check(parts[1].ToString() == "" && parts[0].ToString() == oracle, "final split len");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
