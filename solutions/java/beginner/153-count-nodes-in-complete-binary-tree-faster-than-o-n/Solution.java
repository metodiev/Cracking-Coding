/** Task 153 - Count nodes in complete binary tree faster than O(n). (Tier: Beginner, CTCI Ch 4) */
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

    static int leftHeight(TreeNode node) {
        int h = 0;
        while (node != null) { h++; node = node.left; }
        return h;
    }

    static int rightHeight(TreeNode node) {
        int h = 0;
        while (node != null) { h++; node = node.right; }
        return h;
    }

    /**
     * O(log^2 n) node count for a complete binary tree. If the left and right
     * spine heights match, the subtree is perfect and holds 2^h - 1 nodes.
     */
    static int countNodes(TreeNode root) {
        if (root == null) return 0;
        int lh = leftHeight(root), rh = rightHeight(root);
        if (lh == rh) return (1 << lh) - 1;
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    /** Plain O(n) count used only to cross-check countNodes. */
    static int naiveCount(TreeNode node) {
        if (node == null) return 0;
        return 1 + naiveCount(node.left) + naiveCount(node.right);
    }

    public static void main(String[] args) {
        check(countNodes(null) == 0, "empty");
        check(countNodes(buildTree(1)) == 1, "single");

        int[] sizes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 31, 63, 64, 65, 100, 1000};
        for (int n : sizes) {
            Integer[] values = new Integer[n];
            for (int i = 0; i < n; i++) values[i] = i;
            TreeNode root = buildTree(values);   // no gaps -> complete tree
            check(countNodes(root) == n, "count == " + n);
            check(countNodes(root) == naiveCount(root), "fast == naive for n=" + n);
        }

        TreeNode full = buildTree(1, 2, 3, 4, 5, 6, 7);   // perfect, 2^3 - 1
        check(countNodes(full) == 7, "perfect tree");

        TreeNode partial = buildTree(1, 2, 3, 4, 5, 6, null);   // last level partially filled
        check(countNodes(partial) == 6 && naiveCount(partial) == 6, "partial last level");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
