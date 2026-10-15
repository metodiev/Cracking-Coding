// Task 235 - Serialize and deserialize N-ary tree. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Text;

public class NAryNode
{
    public int Val;
    public List<NAryNode> Children = new List<NAryNode>();
    public NAryNode(int v) { Val = v; }
}

public static class Solution
{
    /// <summary>
    /// Wire format (identical across the Python, Java, JavaScript and C# solutions):
    /// a recursive preorder where every node is written as "value[childCount]" followed
    /// by the serialized children, e.g. root 1 with children [3, 2, 4] where 3 has
    /// children [5, 6] serializes to "1[3]3[2]5[0]6[0]2[0]4[0]"; empty tree -> "".
    /// </summary>
    public static string Serialize(NAryNode? root)
    {
        if (root == null) return "";
        var sb = new StringBuilder();
        SerializeInto(root, sb);
        return sb.ToString();
    }

    private static void SerializeInto(NAryNode node, StringBuilder sb)
    {
        sb.Append(node.Val).Append('[').Append(node.Children.Count).Append(']');
        foreach (var child in node.Children) SerializeInto(child, sb);
    }

    public static NAryNode? Deserialize(string data)
    {
        if (data.Length == 0) return null;
        int pos = 0;
        return ReadNode(data, ref pos);
    }

    private static NAryNode ReadNode(string data, ref int pos)
    {
        int start = pos;
        while (data[pos] != '[') pos++;
        int val = int.Parse(data.Substring(start, pos - start));
        pos++;                                     // skip '['
        start = pos;
        while (data[pos] != ']') pos++;
        int childCount = int.Parse(data.Substring(start, pos - start));
        pos++;                                     // skip ']'
        var node = new NAryNode(val);
        for (int i = 0; i < childCount; i++) node.Children.Add(ReadNode(data, ref pos));
        return node;
    }

    public static int Depth(NAryNode? node)
    {
        if (node == null) return 0;
        int best = 0;
        foreach (var child in node.Children) best = Math.Max(best, Depth(child));
        return 1 + best;
    }

    public static bool TreesEqual(NAryNode? a, NAryNode? b)
    {
        if (a == null || b == null) return a == b;
        if (a.Val != b.Val || a.Children.Count != b.Children.Count) return false;
        for (int i = 0; i < a.Children.Count; i++)
            if (!TreesEqual(a.Children[i], b.Children[i])) return false;
        return true;
    }

    public static void Main()
    {
        // exact wire format for the documented example
        var example = new NAryNode(1);
        var three = new NAryNode(3);
        three.Children.Add(new NAryNode(5));
        three.Children.Add(new NAryNode(6));
        example.Children.Add(three);
        example.Children.Add(new NAryNode(2));
        example.Children.Add(new NAryNode(4));
        Check(Serialize(example) == "1[3]3[2]5[0]6[0]2[0]4[0]", "example wire format");
        Check(TreesEqual(Deserialize("1[3]3[2]5[0]6[0]2[0]4[0]"), example), "example parsed");
        Check(Serialize(Deserialize(Serialize(example))!) == Serialize(example), "wire round trip");

        // empty tree
        Check(Serialize(null) == "", "empty serializes to blank");
        Check(Deserialize("") == null, "blank parses to null");

        // single node
        Check(Serialize(new NAryNode(42)) == "42[0]", "single node format");
        Check(TreesEqual(Deserialize("42[0]"), new NAryNode(42)), "single node round trip");

        // deep chain of 50 nodes
        NAryNode chain = new NAryNode(50);
        for (int v = 49; v >= 1; v--)
        {
            var parent = new NAryNode(v);
            parent.Children.Add(chain);
            chain = parent;
        }
        Check(Depth(chain) == 50, "chain depth built");
        Check(Serialize(chain).StartsWith("1[1]2[1]3[1]"), "chain prefix");
        Check(TreesEqual(Deserialize(Serialize(chain)), chain), "chain round trip");
        Check(Depth(Deserialize(Serialize(chain))) == 50, "chain depth restored");

        // node with many children
        var many = new NAryNode(1);
        for (int v = 0; v < 10; v++) many.Children.Add(new NAryNode(v));
        var expected = new StringBuilder("1[10]");
        for (int v = 0; v < 10; v++) expected.Append(v).Append("[0]");
        Check(Serialize(many) == expected.ToString(), "ten children format");
        var restoredMany = Deserialize(Serialize(many))!;
        Check(TreesEqual(restoredMany, many), "ten children round trip");
        Check(restoredMany.Children.Count == 10, "ten children restored");
        for (int v = 0; v < 10; v++) Check(restoredMany.Children[v].Val == v, "child value " + v);

        // negative values are handled by the scanner
        var negative = new NAryNode(-5);
        negative.Children.Add(new NAryNode(-7));
        Check(Serialize(negative) == "-5[1]-7[0]", "negative values");
        Check(TreesEqual(Deserialize("-5[1]-7[0]"), negative), "negative round trip");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
