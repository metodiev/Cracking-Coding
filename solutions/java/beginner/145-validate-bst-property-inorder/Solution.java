/** Task 145 - Validate BST property (inorder). (Tier: Beginner, CTCI Ch 4)
 *
 * Two equivalent O(n) routes:
 *   1. min/max range recursion - every node must stay inside the open interval
 *      inherited from its ancestors (long bounds, not int bounds, so
 *      Integer.MIN_VALUE / Integer.MAX_VALUE values are still legal);
 *   2. inorder must be strictly increasing (see isValidBstInorder below).
 * Comparing only a node with its own children is NOT enough:
 * {5, 1, 4, null, null, 3, 6} passes that weaker test although 3 < 5.
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

    static boolean isValidBst(TreeNode root) {
        return validRange(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private static boolean validRange(TreeNode node, long lo, long hi) {
        if (node == null) return true;
        if (node.val <= lo || node.val >= hi) return false;
        return validRange(node.left, lo, node.val) && validRange(node.right, node.val, hi);
    }

    /** Alternative route: walk the tree in order and require a strict increase. */
    static boolean isValidBstInorder(TreeNode root) {
        long previous = Long.MIN_VALUE;
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            if (node.val <= previous) return false;
            previous = node.val;
            node = node.right;
        }
        return true;
    }

    public static void main(String[] args) {
        check(isValidBst(null) && isValidBstInorder(null), "empty");
        check(isValidBst(buildTree(2, 1, 3)), "2,1,3");
        check(isValidBst(buildTree(7)), "single");
        check(isValidBst(buildTree(3, 2, null, 1)), "left chain");
        // classic pitfall: 4 sits in the right subtree of 5 but is smaller than 5
        check(!isValidBst(buildTree(5, 1, 4, null, null, 3, 6)), "weak test would pass");
        check(!isValidBst(buildTree(10, 5, 15, null, null, 6, 20)), "deep violation");
        check(!isValidBst(buildTree(5, 4, 6, null, null, 3, 7)), "3 under the right subtree");
        check(!isValidBst(buildTree(1, 1)), "duplicates");
        // long bounds: int extremes are legal node values
        check(isValidBst(buildTree(Integer.MAX_VALUE, Integer.MIN_VALUE, null)), "int extremes");
        check(isValidBstInorder(buildTree(2, 1, 3)) && isValidBstInorder(buildTree(3, 2, null, 1)),
              "inorder route agrees");
        check(!isValidBstInorder(buildTree(5, 1, 4, null, null, 3, 6)) &&
              !isValidBstInorder(buildTree(1, 1)), "inorder route rejects");
        check(isValidBst(buildTree(Integer.MAX_VALUE, Integer.MIN_VALUE, null)) ==
              isValidBstInorder(buildTree(Integer.MAX_VALUE, Integer.MIN_VALUE, null)), "routes agree");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
