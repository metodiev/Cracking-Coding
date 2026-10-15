// Task 236 - Count nodes at distance K from target in binary tree. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class TreeNode
{
    public int Val;
    public TreeNode? Left, Right;
    public TreeNode(int v) { Val = v; }
}

public static class Solution
{
    /// <summary>Build a binary tree from level-order values (null marks a missing child).</summary>
    public static TreeNode? BuildTree(params int?[] values)
    {
        if (values.Length == 0 || values[0] == null) return null;
        var root = new TreeNode(values[0]!.Value);
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        int i = 1;
        while (q.Count > 0 && i < values.Length)
        {
            var node = q.Dequeue();
            if (i < values.Length && values[i] != null)
            {
                node.Left = new TreeNode(values[i]!.Value);
                q.Enqueue(node.Left);
            }
            i++;
            if (i < values.Length && values[i] != null)
            {
                node.Right = new TreeNode(values[i]!.Value);
                q.Enqueue(node.Right);
            }
            i++;
        }
        return root;
    }

    /// <summary>
    /// Distance counts edges, so k = 0 is the target. A BFS records every node's parent;
    /// a second BFS radiates outward from the target k steps and counts the frontier.
    /// O(n) time. A missing target, empty tree or k beyond the diameter return 0; for
    /// repeated values the first node in level order is the target.
    /// </summary>
    public static int CountNodesAtDistance(TreeNode? root, int targetVal, int k)
    {
        if (root == null || k < 0) return 0;
        var parent = new Dictionary<TreeNode, TreeNode?> { [root] = null };
        TreeNode? target = null;
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        while (q.Count > 0)
        {
            var node = q.Dequeue();
            if (target == null && node.Val == targetVal) target = node;
            if (node.Left != null) { parent[node.Left] = node; q.Enqueue(node.Left); }
            if (node.Right != null) { parent[node.Right] = node; q.Enqueue(node.Right); }
        }
        if (target == null) return 0;
        var seen = new HashSet<TreeNode> { target };
        var frontier = new List<TreeNode> { target };   // nodes exactly `step` edges away
        for (int step = 0; step < k; step++)
        {
            var next = new List<TreeNode>();
            foreach (var node in frontier)
            {
                TreeNode? up = parent[node];
                foreach (var nxt in new[] { node.Left, node.Right, up })
                {
                    if (nxt != null && seen.Add(nxt)) next.Add(nxt);
                }
            }
            frontier = next;
            if (frontier.Count == 0) return 0;
        }
        return frontier.Count;
    }

    /// <summary>O(n * h) reference: distance = root paths without their common prefix.</summary>
    public static int BruteCount(TreeNode? root, int targetVal, int k)
    {
        if (root == null || k < 0) return 0;
        var targetPath = FindPath(root, targetVal);
        if (targetPath == null) return 0;
        return BruteDfs(root, new List<TreeNode>(), targetPath, k);
    }

    private static int BruteDfs(TreeNode node, List<TreeNode> path, List<TreeNode> targetPath, int k)
    {
        path.Add(node);
        int common = 0;
        while (common < path.Count && common < targetPath.Count && path[common] == targetPath[common]) common++;
        int count = path.Count - common + targetPath.Count - common == k ? 1 : 0;
        if (node.Left != null) count += BruteDfs(node.Left, path, targetPath, k);
        if (node.Right != null) count += BruteDfs(node.Right, path, targetPath, k);
        path.RemoveAt(path.Count - 1);
        return count;
    }

    /// <summary>Root-to-target path via BFS parent pointers (same target rule as the main one).</summary>
    private static List<TreeNode>? FindPath(TreeNode root, int targetVal)
    {
        var parent = new Dictionary<TreeNode, TreeNode?> { [root] = null };
        var q = new Queue<TreeNode>();
        q.Enqueue(root);
        TreeNode? target = null;
        while (q.Count > 0 && target == null)
        {
            var node = q.Dequeue();
            if (node.Val == targetVal) { target = node; break; }
            if (node.Left != null) { parent[node.Left] = node; q.Enqueue(node.Left); }
            if (node.Right != null) { parent[node.Right] = node; q.Enqueue(node.Right); }
        }
        if (target == null) return null;
        var path = new List<TreeNode>();
        for (TreeNode? n = target; n != null; n = parent[n]) path.Add(n);
        path.Reverse();
        return path;
    }

    private static TreeNode? Prune(TreeNode? node, Random rng)
    {
        if (node == null || rng.NextDouble() < 0.25) return null;
        node.Left = Prune(node.Left, rng);
        node.Right = Prune(node.Right, rng);
        return node;
    }

    public static void Main()
    {
        var classic = BuildTree(3, 5, 1, 6, 2, 0, 8, null, null, 7, 4);
        Check(CountNodesAtDistance(classic, 5, 0) == 1, "k=0 is the target");
        Check(CountNodesAtDistance(classic, 5, 1) == 3, "{3, 6, 2}");
        Check(CountNodesAtDistance(classic, 5, 2) == 3, "{1, 7, 4}");
        Check(CountNodesAtDistance(classic, 5, 3) == 2, "{0, 8}");
        Check(CountNodesAtDistance(classic, 5, 4) == 0, "k past distance from target");
        Check(CountNodesAtDistance(classic, 3, 0) == 1, "root k=0");
        Check(CountNodesAtDistance(classic, 3, 2) == 4, "root level 2");
        Check(CountNodesAtDistance(classic, 3, 3) == 2, "root reaches deepest nodes");
        Check(CountNodesAtDistance(classic, 99, 1) == 0, "missing target");

        var skewed = BuildTree(1, 2, null, 3, null, 4);
        Check(CountNodesAtDistance(skewed, 1, 3) == 1, "root to deepest in chain");
        Check(CountNodesAtDistance(skewed, 1, 4) == 0, "chain k past diameter");
        Check(CountNodesAtDistance(BuildTree(7), 7, 0) == 1, "single node k=0");
        Check(CountNodesAtDistance(BuildTree(7), 7, 1) == 0, "single node k=1");
        Check(CountNodesAtDistance(null, 1, 0) == 0, "empty tree");
        Check(CountNodesAtDistance(BuildTree(1, 2, 1), 1, 1) == 2, "duplicate target, level order first");
        Check(CountNodesAtDistance(BuildTree(1, 2, 1), 1, 2) == 0, "duplicate target explored");

        var rng = new Random(236);
        for (int test = 0; test < 200; test++)      // cross-check vs path-based brute force
        {
            int n = rng.Next(13);
            var nodes = new TreeNode[n];
            for (int i = 0; i < n; i++) nodes[i] = new TreeNode(i + 1);
            for (int i = 0; i < n; i++)
            {
                if (2 * i + 1 < n) nodes[i].Left = nodes[2 * i + 1];
                if (2 * i + 2 < n) nodes[i].Right = nodes[2 * i + 2];
            }
            TreeNode? root = n == 0 ? null : Prune(nodes[0], rng);
            var reachable = new List<TreeNode>();
            if (root != null)
            {
                var stack = new Stack<TreeNode>();
                stack.Push(root);
                while (stack.Count > 0)
                {
                    var node = stack.Pop();
                    reachable.Add(node);
                    if (node.Left != null) stack.Push(node.Left);
                    if (node.Right != null) stack.Push(node.Right);
                }
            }
            int targetVal = reachable.Count == 0 ? 1 : reachable[rng.Next(reachable.Count)].Val;
            for (int k = 0; k <= n + 2; k++)
            {
                Check(CountNodesAtDistance(root, targetVal, k) == BruteCount(root, targetVal, k),
                    "matches brute force: target=" + targetVal + " k=" + k);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
