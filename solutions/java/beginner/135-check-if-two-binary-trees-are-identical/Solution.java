/** Task 135 - Check if two binary trees are identical. (Tier: Beginner, CTCI Ch 4) */
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

    static boolean areIdentical(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.val == b.val
                && areIdentical(a.left, b.left)
                && areIdentical(a.right, b.right);
    }

    public static void main(String[] args) {
        check(areIdentical(buildTree(1, 2, 3, null, 4), buildTree(1, 2, 3, null, 4)), "equal trees");
        check(!areIdentical(buildTree(1, 2, 3, null, 4), buildTree(1, 2, 3, 4)), "different shape");
        check(!areIdentical(buildTree(1, 2), buildTree(1, 3)), "different values");
        check(areIdentical(null, null), "both empty");
        check(!areIdentical(buildTree(1), null), "one empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
