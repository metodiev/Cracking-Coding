/** Task 143 - Symmetric tree check (mirror). (Tier: Beginner, CTCI Ch 4)
 *
 * A tree is symmetric when its left and right subtrees are mirror images.
 * Traversal order alone is not enough: compare a.left with b.right and
 * a.right with b.left.
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

    static boolean isMirror(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.val == b.val && isMirror(a.left, b.right) && isMirror(a.right, b.left);
    }

    static boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isMirror(root.left, root.right);
    }

    /** Iterative variant: keep a stack of mirrored pairs (a, b). */
    static boolean isSymmetricIterative(TreeNode root) {
        if (root == null) return true;
        List<TreeNode> stack = new ArrayList<>();   // ArrayList tolerates null entries
        stack.add(root.left);
        stack.add(root.right);
        while (!stack.isEmpty()) {
            TreeNode b = stack.remove(stack.size() - 1);
            TreeNode a = stack.remove(stack.size() - 1);
            if (a == null && b == null) continue;
            if (a == null || b == null || a.val != b.val) return false;
            stack.add(a.left);
            stack.add(b.right);
            stack.add(a.right);
            stack.add(b.left);
        }
        return true;
    }

    public static void main(String[] args) {
        check(isSymmetric(null) && isSymmetricIterative(null), "empty");
        check(isSymmetric(buildTree(1)), "single node");
        check(isSymmetric(buildTree(1, 2, 2)), "two mirrors");
        check(isSymmetric(buildTree(1, 2, 2, 3, 4, 4, 3)), "classic symmetric");
        // both 3s hang as RIGHT children, so the mirrored pairing fails
        check(!isSymmetric(buildTree(1, 2, 2, null, 3, null, 3)), "mirrored pairing fails");
        check(isSymmetric(buildTree(1, 2, 2, null, 3, 3, null)), "mirrored placement");
        check(!isSymmetric(buildTree(1, 2, 3)), "different values");
        check(!isSymmetricIterative(buildTree(1, 2, 2, null, 3, null, 3)), "iterative mismatch");
        check(isSymmetricIterative(buildTree(1, 2, 2, null, 3, 3, null)), "iterative match");
        check(isSymmetric(buildTree(1, 2, 2, 3, null, null, 3)) &&
              isSymmetricIterative(buildTree(1, 2, 2, 3, null, null, 3)), "deeper symmetric");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
