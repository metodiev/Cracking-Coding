// Task 974 - Implement step-by-step solution explainer generator using AST of canonical solution.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// A canonical solution is modelled as a small AST: sequence, assignment, conditional with then/else
// branches, loop and return nodes, each carrying its own prose. The explainer walks the tree in
// pre-order, numbers every node from 1 and indents by two spaces per level of nesting, which is the
// standard shape of an auto-generated walkthrough of how a solution proceeds.
//
// Verified: the node count and contiguous numbering of the is-prime walkthrough, exact depth and
// indentation of selected lines, equal indentation for sibling statements, the deepest line equals
// the deepest node, twice-identical output, and three independent oracles - an explicit-stack
// pre-order traversal, a breadth-first depth map and a recursive height computation.
// Simplification: the AST is a hand-built tree of 13 nodes with prose already attached; there is no
// parser, type checker or natural-language surface realisation, and indentation is fixed at 2 spaces.
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Node
    {
        public readonly string Kind;
        public readonly string Text;
        public readonly List<Node> Children = new List<Node>();

        public Node(string kind, string text)
        {
            Kind = kind;
            Text = text;
        }

        public Node With(params Node[] children)
        {
            Children.AddRange(children);
            return this;
        }
    }

    private sealed class Line
    {
        public readonly int Number;
        public readonly int Depth;
        public readonly string Text;

        public Line(int number, int depth, string text)
        {
            Number = number;
            Depth = depth;
            Text = text;
        }

        public string Render() => Number.ToString() + ". " + new string(' ', Depth * 2) + Text;
    }

    private static void Walk(Node node, int depth, List<Line> lines)
    {
        lines.Add(new Line(lines.Count + 1, depth, node.Text));
        foreach (Node child in node.Children) Walk(child, depth + 1, lines);
    }

    /** Oracle 1: pre-order produced with an explicit stack instead of recursion. */
    private static List<string> FlattenIterative(Node root)
    {
        var order = new List<string>();
        var stack = new Stack<KeyValuePair<Node, int>>();
        stack.Push(new KeyValuePair<Node, int>(root, 0));
        while (stack.Count > 0)
        {
            KeyValuePair<Node, int> current = stack.Pop();
            order.Add(current.Value + "|" + current.Key.Text);
            for (int i = current.Key.Children.Count - 1; i >= 0; i--)
                stack.Push(new KeyValuePair<Node, int>(current.Key.Children[i], current.Value + 1));
        }
        return order;
    }

    /** Oracle 2: node depth by edge distance from the root, computed breadth first. */
    private static Dictionary<Node, int> DepthsByBreadth(Node root)
    {
        var depths = new Dictionary<Node, int>();
        var queue = new Queue<Node>();
        depths[root] = 0;
        queue.Enqueue(root);
        while (queue.Count > 0)
        {
            Node node = queue.Dequeue();
            foreach (Node child in node.Children)
            {
                depths[child] = depths[node] + 1;
                queue.Enqueue(child);
            }
        }
        return depths;
    }

    /** Oracle 3: tree height by recursion, independent of the traversal that produced the lines. */
    private static int Height(Node node)
    {
        int best = 0;
        foreach (Node child in node.Children) best = Math.Max(best, Height(child) + 1);
        return best;
    }

    private static void CollectNodes(Node node, List<Node> nodes)
    {
        nodes.Add(node);
        foreach (Node child in node.Children) CollectNodes(child, nodes);
    }

    private static List<Line> Explain(Node root)
    {
        var lines = new List<Line>();
        if (root != null) Walk(root, 0, lines);
        return lines;
    }

    private static string Render(List<Line> lines)
    {
        var builder = new StringBuilder();
        foreach (Line line in lines)
        {
            if (builder.Length > 0) builder.Append('\n');
            builder.Append(line.Render());
        }
        return builder.ToString();
    }

    private static Node IsPrimeProgram()
    {
        var thenBranch = new Node("then", "then").With(new Node("return", "return false"));
        var elseBranch = new Node("else", "else").With(
            new Node("assign", "candidate = 2"),
            new Node("loop", "while candidate * candidate <= n:").With(
                new Node("if", "if n % candidate == 0:").With(
                    new Node("then", "then").With(new Node("return", "return false"))),
                new Node("assign", "candidate = candidate + 1")));
        return new Node("seq", "solve is-prime").With(
            new Node("assign", "read n"),
            new Node("if", "if n < 2:").With(thenBranch, elseBranch),
            new Node("return", "return true"));
    }

    public static void Main()
    {
        Node root = IsPrimeProgram();
        List<Line> lines = Explain(root);
        Check(lines.Count == 13, "the is-prime walkthrough explains all 13 nodes (got " + lines.Count + ")");

        bool contiguous = true;
        for (int i = 0; i < lines.Count; i++) contiguous &= lines[i].Number == i + 1;
        Check(contiguous, "step numbers run 1..13 without gaps or repeats");

        Check(lines[0].Number == 1 && lines[0].Depth == 0 && lines[0].Text == "solve is-prime",
            "the walkthrough opens with the root step at depth 0");
        Check(lines[4].Render() == "5. " + new string(' ', 6) + "return false",
            "the guarded early return is step 5 at depth 3 (got '" + lines[4].Render() + "')");
        Check(lines[10].Number == 11 && lines[10].Depth == 6,
            "the innermost return of the loop is step 11 at depth 6");
        Check(lines[12].Number == 13 && lines[12].Depth == 1 && lines[12].Text == "return true",
            "the final return closes the walkthrough at depth 1");

        // Oracle 1: recursive and explicit-stack pre-orders must agree.
        List<string> iterative = FlattenIterative(root);
        bool sameOrder = iterative.Count == lines.Count;
        for (int i = 0; i < lines.Count && sameOrder; i++) sameOrder &= iterative[i] == lines[i].Depth + "|" + lines[i].Text;
        Check(sameOrder, "the explicit-stack pre-order oracle reproduces the recursive walkthrough");

        // Oracle 2: every reported depth is the true edge distance from the root.
        Dictionary<Node, int> breadth = DepthsByBreadth(root);
        var nodes = new List<Node>();
        CollectNodes(root, nodes);
        int depthMismatch = 0, indentMismatch = 0;
        for (int i = 0; i < lines.Count; i++)
        {
            if (!breadth.ContainsKey(nodes[i]) || breadth[nodes[i]] != lines[i].Depth) depthMismatch++;
            if (lines[i].Render() != lines[i].Number + ". " + new string(' ', breadth[nodes[i]] * 2) + lines[i].Text) indentMismatch++;
        }
        Check(depthMismatch == 0, "the breadth-first depth oracle confirms every reported depth");
        Check(indentMismatch == 0, "indentation is exactly two spaces per level for every step");

        // Oracle 3: the height of the AST equals the deepest indentation in the output.
        int height = Height(root);
        int deepest = 0;
        foreach (Line line in lines) deepest = Math.Max(deepest, line.Depth);
        Check(height == deepest && height == 6, "recursive height 6 matches the deepest explained step");

        Check(lines[8].Depth == lines[11].Depth && lines[8].Depth > lines[7].Depth,
            "statements inside the while body share one indentation deeper than the loop header");

        // Boundaries: a leaf-only program, an empty root list and a deep chain.
        List<Line> leaf = Explain(new Node("seq", "solve trivially"));
        Check(leaf.Count == 1 && leaf[0].Depth == 0, "a single-node program produces exactly one step at depth 0");
        Check(Explain(null).Count == 0, "an absent root produces no steps instead of throwing");

        Node chain = new Node("seq", "level 0");
        Node cursor = chain;
        for (int level = 1; level <= 10; level++)
        {
            var next = new Node("seq", "level " + level);
            cursor.Children.Add(next);
            cursor = next;
        }
        List<Line> deep = Explain(chain);
        Check(deep.Count == 11 && deep[10].Depth == 10 && deep[10].Render().StartsWith("11. " + new string(' ', 20)),
            "a ten-level nest indents the last step by 20 spaces");

        string first = Render(lines), second = Render(Explain(IsPrimeProgram()));
        Check(first == second && first.Split('\n').Length == 13, "explaining the same AST twice yields identical text");

        Console.WriteLine("walkthrough: 13 steps, height " + height + ", max indent " + deepest * 2 + " spaces");
        Console.WriteLine("sample line 5: " + lines[4].Render());
        Console.WriteLine("sample line 11: " + lines[10].Render());
        Console.WriteLine("PASS");
    }
}
