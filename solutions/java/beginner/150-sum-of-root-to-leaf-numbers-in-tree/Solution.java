/** Task 150 - Sum of root-to-leaf numbers in tree. (Tier: Beginner, CTCI Ch 4)
 *
 * Every root-to-leaf path spells a decimal number (1 -> 2 -> 3 means 123). The DFS
 * carries the prefix built so far: acc * 10 + node.val, and adds the accumulated
 * value whenever a leaf closes a path.
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

    /** Recursive DFS with the running prefix; an empty tree contributes 0. */
    static int sumNumbers(TreeNode root) {
        return walk(root, 0);
    }

    private static int walk(TreeNode node, int acc) {
        if (node == null) return 0;
        acc = acc * 10 + node.val;
        if (node.left == null && node.right == null) return acc;   // leaf closes one number
        return walk(node.left, acc) + walk(node.right, acc);
    }

    /** Stack of (node, prefix) pairs; a leaf adds its prefix to the total. */
    static int sumNumbersIterative(TreeNode root) {
        if (root == null) return 0;
        int total = 0;
        Deque<TreeNode> nodes = new ArrayDeque<>();
        Deque<Integer> prefixes = new ArrayDeque<>();
        nodes.push(root);
        prefixes.push(root.val);
        while (!nodes.isEmpty()) {
            TreeNode node = nodes.pop();
            int acc = prefixes.pop();
            if (node.left == null && node.right == null) {
                total += acc;
                continue;
            }
            if (node.left != null) {
                nodes.push(node.left);
                prefixes.push(acc * 10 + node.left.val);
            }
            if (node.right != null) {
                nodes.push(node.right);
                prefixes.push(acc * 10 + node.right.val);
            }
        }
        return total;
    }

    public static void main(String[] args) {
        check(sumNumbers(null) == 0 && sumNumbersIterative(null) == 0, "empty tree");
        check(sumNumbers(buildTree(7)) == 7, "single node");
        check(sumNumbers(buildTree(1, 2, 3)) == 25, "12 + 13");
        check(sumNumbers(buildTree(4, 9, 0, 5, 1)) == 1026, "495 + 491 + 40");
        check(sumNumbers(buildTree(1, 0, 0)) == 20, "zeros inside the path");
        check(sumNumbers(buildTree(1, 2, null, 3)) == 123, "one path only");
        check(sumNumbers(buildTree(1, 2, 3, 4, 5)) == 262, "three paths");
        Integer[][] cases = {
            {}, {7}, {1, 2, 3}, {4, 9, 0, 5, 1}, {1, 0, 0}, {1, 2, null, 3}, {1, 2, 3, 4, 5},
            {1, 9, 9, null, 1}
        };
        for (Integer[] data : cases) {
            TreeNode tree = buildTree(data);
            check(sumNumbers(tree) == sumNumbersIterative(tree), "recursive == iterative");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
