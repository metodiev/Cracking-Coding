/** Task 241 - Maximum path sum in binary tree (any path). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

public class Solution {
    static TreeNode buildTree(Integer... values) {
        if (values.length == 0 || values[0] == null) return null;
        TreeNode root = new TreeNode(values[0]);
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int i = 1;
        while (!q.isEmpty() && i < values.length) {
            TreeNode node = q.poll();
            if (i < values.length && values[i] != null) { node.left = new TreeNode(values[i]); q.add(node.left); }
            i++;
            if (i < values.length && values[i] != null) { node.right = new TreeNode(values[i]); q.add(node.right); }
            i++;
        }
        return root;
    }

    static List<Integer> preorder(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        pre(node, out);
        return out;
    }

    private static void pre(TreeNode node, List<Integer> out) {
        if (node == null) return;
        out.add(node.val);
        pre(node.left, out);
        pre(node.right, out);
    }

    static List<Integer> inorder(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        in(node, out);
        return out;
    }

    private static void in(TreeNode node, List<Integer> out) {
        if (node == null) return;
        in(node.left, out);
        out.add(node.val);
        in(node.right, out);
    }

    private static long best;

    static long maxPathSum(TreeNode root) {
        if (root == null) throw new IllegalArgumentException("empty tree has no path");
        best = Long.MIN_VALUE;
        gain(root);
        return best;
    }

    private static long gain(TreeNode node) {
        if (node == null) return 0;
        long left = Math.max(0, gain(node.left));
        long right = Math.max(0, gain(node.right));
        best = Math.max(best, node.val + left + right);
        return node.val + Math.max(left, right);
    }

    public static void main(String[] args) {
        check(maxPathSum(buildTree(1, 2, 3)) == 6, "small tree");
        check(maxPathSum(buildTree(-10, 9, 20, null, null, 15, 7)) == 42, "classic");
        check(maxPathSum(buildTree(-3)) == -3, "single negative");
        check(maxPathSum(buildTree(2, -1)) == 2, "skip negative child");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
