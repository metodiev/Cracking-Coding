/** Task 140 - Max depth of binary tree. (Tier: Beginner, CTCI Ch 4) */
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

    static int maxDepth(TreeNode node) {
        if (node == null) return 0;
        return 1 + Math.max(maxDepth(node.left), maxDepth(node.right));
    }

    static int maxDepthIterative(TreeNode root) {
        if (root == null) return 0;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int depth = 0;
        while (!q.isEmpty()) {
            depth++;
            for (int i = q.size(); i > 0; i--) {
                TreeNode node = q.poll();
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
        }
        return depth;
    }

    public static void main(String[] args) {
        check(maxDepth(buildTree(1, 2, 3, 4, 5, null, 6)) == 3, "balanced three levels");
        check(maxDepth(null) == 0, "empty");
        check(maxDepth(buildTree(7)) == 1, "single");
        check(maxDepth(buildTree(1, 2, null, 3, null, 4)) == 4, "skewed");
        check(maxDepth(buildTree(1, 2, 3, 4, 5, null, 6)) == maxDepthIterative(buildTree(1, 2, 3, 4, 5, null, 6)),
                "recursive == iterative");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
