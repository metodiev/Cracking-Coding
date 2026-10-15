/** Task 144 - Convert sorted array to balanced BST. (Tier: Beginner, CTCI Ch 4)
 *
 * The middle element becomes the root, so both halves stay equally deep and the
 * resulting tree has the minimal possible height for those values.
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

    static TreeNode sortedArrayToBst(int[] nums) {
        return buildBst(nums, 0, nums.length - 1);
    }

    private static TreeNode buildBst(int[] nums, int lo, int hi) {
        if (lo > hi) return null;
        int mid = (lo + hi) / 2;                  // left-biased middle keeps it stable
        TreeNode node = new TreeNode(nums[mid]);
        node.left = buildBst(nums, lo, mid - 1);
        node.right = buildBst(nums, mid + 1, hi);
        return node;
    }

    /** Left - node - right values; sorted output proves the BST ordering. */
    static List<Integer> inorder(TreeNode root) {
        List<Integer> values = new ArrayList<>();
        inorderInto(root, values);
        return values;
    }

    private static void inorderInto(TreeNode node, List<Integer> values) {
        if (node == null) return;
        inorderInto(node.left, values);
        values.add(node.val);
        inorderInto(node.right, values);
    }

    static int height(TreeNode node) {
        if (node == null) return 0;
        return 1 + Math.max(height(node.left), height(node.right));
    }

    /** Check the requirement directly: every node's subtree heights differ by <= 1. */
    static boolean isHeightBalanced(TreeNode root) {
        if (root == null) return true;
        return Math.abs(height(root.left) - height(root.right)) <= 1
                && isHeightBalanced(root.left) && isHeightBalanced(root.right);
    }

    public static void main(String[] args) {
        check(sortedArrayToBst(new int[]{}) == null, "empty array");
        check(sortedArrayToBst(new int[]{5}).val == 5, "single element");
        TreeNode small = sortedArrayToBst(new int[]{1, 2, 3});
        check(small.val == 2 && small.left.val == 1 && small.right.val == 3, "1,2,3");
        int[][] cases = {
            {}, {1}, {1, 2, 3}, {1, 2, 3, 4, 5, 6}, {-10, -3, 0, 5, 9},
            {1, 2, 3, 4, 5, 6, 7}, {1, 1, 1}, {0, 1000000}
        };
        for (int[] nums : cases) {
            TreeNode tree = sortedArrayToBst(nums);
            List<Integer> expected = new ArrayList<>();
            for (int v : nums) expected.add(v);
            check(inorder(tree).equals(expected), "inorder equals input of length " + nums.length);
            check(isHeightBalanced(tree), "balanced for length " + nums.length);
            if (nums.length > 0) {
                check(tree.val == nums[(nums.length - 1) / 2], "middle element is the root");
            }
        }
        check(height(sortedArrayToBst(new int[]{1, 2, 3, 4, 5, 6, 7})) == 3, "height 3 for 7 values");
        check(height(sortedArrayToBst(new int[]{-10, -3, 0, 5, 9})) == 3, "height 3 for 5 values");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
