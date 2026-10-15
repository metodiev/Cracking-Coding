/** Task 240 - Sum of nodes at each level (level order). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

public class Solution {
    /** Build a binary tree from level-order values (null marks a missing child). */
    static TreeNode buildTree(Integer... values) {
        if (values.length == 0 || values[0] == null) return null;
        TreeNode root = new TreeNode(values[0]);
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int i = 1;
        while (!q.isEmpty() && i < values.length) {
            TreeNode node = q.poll();
            if (i < values.length && values[i] != null) {
                node.left = new TreeNode(values[i]);
                q.add(node.left);
            }
            i++;
            if (i < values.length && values[i] != null) {
                node.right = new TreeNode(values[i]);
                q.add(node.right);
            }
            i++;
        }
        return root;
    }

    /**
     * Level-order BFS with a per-level snapshot sums each level independently:
     * one pass over the tree, O(n) time. The empty tree yields [].
     */
    static List<Integer> levelSums(TreeNode root) {
        List<Integer> sums = new ArrayList<>();
        if (root == null) return sums;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        while (!q.isEmpty()) {
            int total = 0;
            for (int size = q.size(); size > 0; size--) {   // snapshot of the level size
                TreeNode node = q.poll();
                total += node.val;
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            sums.add(total);
        }
        return sums;
    }

    /** Independent helper: accumulate sums by depth with a DFS. */
    static List<Integer> levelSumsDfs(TreeNode root) {
        List<Integer> sums = new ArrayList<>();
        dfs(root, 0, sums);
        return sums;
    }

    private static void dfs(TreeNode node, int depth, List<Integer> sums) {
        if (node == null) return;
        if (depth == sums.size()) sums.add(0);
        sums.set(depth, sums.get(depth) + node.val);
        dfs(node.left, depth + 1, sums);
        dfs(node.right, depth + 1, sums);
    }

    private static TreeNode prune(TreeNode node, Random rng) {
        if (node == null || rng.nextDouble() < 0.25) return null;
        node.left = prune(node.left, rng);
        node.right = prune(node.right, rng);
        return node;
    }

    public static void main(String[] args) {
        check(levelSums(buildTree(1, 2, 3, 4, 5, null, 6)).equals(Arrays.asList(1, 5, 15)),
                "classic: {1}, {2,3}, {4,5,6}");
        check(levelSums(buildTree(1, 2, 3)).equals(Arrays.asList(1, 5)), "two levels");
        check(levelSums(buildTree(1)).equals(Arrays.asList(1)), "single node");
        check(levelSums(null).isEmpty(), "empty");
        check(levelSums(buildTree(1, 2, null, 3, null, 4)).equals(Arrays.asList(1, 2, 3, 4)), "left chain");
        check(levelSums(buildTree(-1, -2, -3)).equals(Arrays.asList(-1, -5)), "negatives");
        check(levelSums(buildTree(0, 0, 0, 0)).equals(Arrays.asList(0, 0, 0)), "zeros");

        Random rng = new Random(240);
        for (int test = 0; test < 200; test++) {    // BFS result == DFS depth accumulation
            int n = rng.nextInt(15);
            TreeNode[] nodes = new TreeNode[n];
            for (int i = 0; i < n; i++) nodes[i] = new TreeNode(rng.nextInt(41) - 20);
            for (int i = 0; i < n; i++) {
                if (2 * i + 1 < n) nodes[i].left = nodes[2 * i + 1];
                if (2 * i + 2 < n) nodes[i].right = nodes[2 * i + 2];
            }
            TreeNode root = n == 0 ? null : prune(nodes[0], rng);
            check(levelSums(root).equals(levelSumsDfs(root)), "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
