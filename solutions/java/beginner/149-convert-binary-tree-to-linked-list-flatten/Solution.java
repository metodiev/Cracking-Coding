/** Task 149 - Convert binary tree to linked list (flatten). (Tier: Beginner, CTCI Ch 4)
 *
 * LeetCode 114: rewrite the tree IN PLACE into its preorder order, using the right
 * pointers as the "next" links and leaving every left pointer empty.
 * {1, 2, 5, 3, 4, null, 6} becomes the chain 1 -> 2 -> 3 -> 4 -> 5 -> 6.
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

    /** Recursive: flatten both subtrees, then splice the left chain in front of the right one. */
    static TreeNode flatten(TreeNode root) {
        if (root == null) return null;
        TreeNode left = root.left, right = root.right;
        flatten(left);
        flatten(right);
        root.left = null;
        root.right = left;
        TreeNode node = root;
        while (node.right != null) node = node.right;   // tail of the flattened left part
        node.right = right;
        return root;
    }

    /** Preorder with an explicit stack: link each node to the previously visited one. */
    static TreeNode flattenIterative(TreeNode root) {
        if (root == null) return null;
        Deque<TreeNode> stack = new ArrayDeque<>();
        stack.push(root);
        TreeNode prev = null;
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            if (prev != null) {
                prev.left = null;
                prev.right = node;
            }
            if (node.right != null) stack.push(node.right);   // right first: left is visited first
            if (node.left != null) stack.push(node.left);
            prev = node;
        }
        return root;
    }

    static List<Integer> preorderValues(TreeNode root) {
        List<Integer> values = new ArrayList<>();
        preorderInto(root, values);
        return values;
    }

    private static void preorderInto(TreeNode node, List<Integer> values) {
        if (node == null) return;
        values.add(node.val);
        preorderInto(node.left, values);
        preorderInto(node.right, values);
    }

    /** Values following the right pointers (the flattened preorder order). */
    static List<Integer> rightChain(TreeNode root) {
        List<Integer> values = new ArrayList<>();
        while (root != null) {
            values.add(root.val);
            root = root.right;
        }
        return values;
    }

    /** After flattening nothing may remain on a left pointer. */
    static boolean allLeftNull(TreeNode node) {
        while (node != null) {
            if (node.left != null) return false;
            node = node.right;
        }
        return true;
    }

    public static void main(String[] args) {
        check(flatten(null) == null && flattenIterative(null) == null, "empty tree");
        check(rightChain(flatten(buildTree(7))).toString().equals("[7]"), "single node");
        check(rightChain(flatten(buildTree(1, 2, 5, 3, 4, null, 6))).toString()
                .equals("[1, 2, 3, 4, 5, 6]"), "sample becomes the preorder chain");
        Integer[][] cases = {
            {}, {7}, {1, 2, 5, 3, 4, null, 6}, {1, 2, null, 3}, {1, null, 2},
            {1, 2, null, 3, null, 4}, {1, 2, 3, 4, 5, null, 6}
        };
        for (Integer[] data : cases) {
            List<Integer> expected = preorderValues(buildTree(data));      // preorder is the target order
            TreeNode recursive = flatten(buildTree(data));
            TreeNode iterative = flattenIterative(buildTree(data));
            check(rightChain(recursive).equals(expected), "recursive chain");
            check(rightChain(iterative).equals(expected), "iterative chain");
            check(allLeftNull(recursive) && allLeftNull(iterative), "left pointers are empty");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
