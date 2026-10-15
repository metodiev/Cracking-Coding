/** Task 141 - Min depth of binary tree. (Tier: Beginner, CTCI Ch 4)
 *
 * Minimum depth = nodes on the shortest root-to-leaf path; a node with only
 * one child is not a leaf, so {1, 2, null, 3} has min depth 3 (not 2).
 */
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

    static int minDepth(TreeNode root) {
        if (root == null) return 0;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int depth = 0;
        while (!q.isEmpty()) {
            depth++;                              // BFS level count: first leaf level wins
            for (int i = q.size(); i > 0; i--) {
                TreeNode node = q.poll();
                if (node.left == null && node.right == null) return depth;
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
        }
        return depth;
    }

    static int minDepthRecursive(TreeNode node) {
        if (node == null) return 0;
        if (node.left == null && node.right == null) return 1;
        if (node.left == null) return 1 + minDepthRecursive(node.right);
        if (node.right == null) return 1 + minDepthRecursive(node.left);
        return 1 + Math.min(minDepthRecursive(node.left), minDepthRecursive(node.right));
    }

    public static void main(String[] args) {
        check(minDepth(null) == 0 && minDepthRecursive(null) == 0, "empty");
        check(minDepth(buildTree(7)) == 1 && minDepthRecursive(buildTree(7)) == 1, "single node");
        check(minDepth(buildTree(1, 2, null, 3)) == 3, "one-sided child is not a leaf");
        check(minDepthRecursive(buildTree(1, 2, null, 3)) == 3, "one-sided child recursive");
        check(minDepth(buildTree(3, 9, 20, null, null, 15, 7)) == 2, "sample tree");
        check(minDepth(buildTree(1, 2, null, 3, null, 4)) == 4, "skewed chain");
        check(minDepth(buildTree(1, 2, 3, 4, 5, null, 6)) ==
              minDepthRecursive(buildTree(1, 2, 3, 4, 5, null, 6)), "bfs == recursive");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
