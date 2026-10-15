/** Task 142 - Check if tree is balanced (height difference <= 1). (Tier: Beginner, CTCI Ch 4)
 *
 * Balanced = at EVERY node the two subtree heights differ by at most 1.
 * The bottom-up pass returns a height, or -1 as soon as a node fails: O(n).
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

    static int height(TreeNode node) {
        if (node == null) return 0;
        return 1 + Math.max(height(node.left), height(node.right));
    }

    /** Subtree height, or -1 when some node inside the subtree is unbalanced. */
    static int heightOrFailure(TreeNode node) {
        if (node == null) return 0;
        int left = heightOrFailure(node.left);
        if (left == -1) return -1;
        int right = heightOrFailure(node.right);
        if (right == -1) return -1;
        if (Math.abs(left - right) > 1) return -1;
        return 1 + Math.max(left, right);
    }

    static boolean isBalanced(TreeNode root) {
        return heightOrFailure(root) != -1;
    }

    /** O(n^2) contrast: recompute both subtree heights at every node. */
    static boolean isBalancedNaive(TreeNode root) {
        if (root == null) return true;
        if (Math.abs(height(root.left) - height(root.right)) > 1) return false;
        return isBalancedNaive(root.left) && isBalancedNaive(root.right);
    }

    public static void main(String[] args) {
        check(isBalanced(null) && isBalancedNaive(null), "empty");
        check(isBalanced(buildTree(1)) && isBalanced(buildTree(1, 2)), "tiny");
        check(isBalanced(buildTree(3, 9, 20, null, null, 15, 7)), "balanced sample");
        check(isBalanced(buildTree(1, 2, 3, 4, 5)), "balanced deeper");
        check(!isBalanced(buildTree(1, 2, 2, 3, 3, null, null, 4, 4)), "unbalanced sample");
        check(!isBalanced(buildTree(1, 2, null, 3, null, 4)), "left chain");
        check(!isBalanced(buildTree(1, null, 2, null, 3)), "right chain");
        check(height(buildTree(1, 2, 3, 4, 5, null, 6)) == 3, "height of sample");
        check(isBalanced(buildTree(1, 2, 2, 3, 3, null, null, 4, 4)) ==
              isBalancedNaive(buildTree(1, 2, 2, 3, 3, null, null, 4, 4)), "bottom-up == naive");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
