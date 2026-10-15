/** Task 151 - Path sum existence in tree. (Tier: Beginner, CTCI Ch 4) */
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

    /** True iff a root-to-leaf path sums to target (leaf = both children null). */
    static boolean hasPathSum(TreeNode root, int target) {
        if (root == null) return false;
        if (root.left == null && root.right == null) return root.val == target;
        return hasPathSum(root.left, target - root.val) || hasPathSum(root.right, target - root.val);
    }

    public static void main(String[] args) {
        TreeNode root = buildTree(5, 4, 8, 11, null, 13, 4, 7, 2, null, null, null, 1);
        check(hasPathSum(root, 22), "5+4+11+2 exists");
        check(hasPathSum(root, 18), "5+8+4+1 exists");
        check(!hasPathSum(root, 5), "5 is not a leaf so it is not a path end");
        check(!hasPathSum(null, 0), "empty tree has no paths");
        check(hasPathSum(buildTree(1), 1), "single node match");
        check(!hasPathSum(buildTree(1), 2), "single node miss");
        check(!hasPathSum(buildTree(1, 2), 1), "one missing child is not a path end");
        check(hasPathSum(buildTree(1, 2), 3), "one-child path 1 -> 2");
        check(hasPathSum(buildTree(-2, null, -3), -5), "negative values");
        check(hasPathSum(buildTree(0, 1, 1), 1), "zero root");
        check(!hasPathSum(buildTree(0, 1, 1), 0), "zero root, no zero path");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
