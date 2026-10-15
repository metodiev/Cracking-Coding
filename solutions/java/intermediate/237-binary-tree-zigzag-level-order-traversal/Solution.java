/** Task 237 - Binary tree zigzag level order traversal. (Tier: Intermediate, CTCI Ch 4) */
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

    /**
     * Level-order BFS with a per-level snapshot; odd levels (0-based) are reversed so
     * the output alternates left->right / right->left. An empty tree yields []. O(n).
     */
    static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> levels = new ArrayList<>();
        if (root == null) return levels;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        boolean leftToRight = true;
        while (!q.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            for (int size = q.size(); size > 0; size--) {   // snapshot of the level size
                TreeNode node = q.poll();
                level.add(node.val);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            if (!leftToRight) Collections.reverse(level);
            levels.add(level);
            leftToRight = !leftToRight;
        }
        return levels;
    }

    public static void main(String[] args) {
        check(zigzagLevelOrder(buildTree(3, 9, 20, null, null, 15, 7)).equals(Arrays.asList(
                Arrays.asList(3),
                Arrays.asList(20, 9),
                Arrays.asList(15, 7))), "classic");
        check(zigzagLevelOrder(buildTree(1, 2, 3, 4, 5, 6, 7)).equals(Arrays.asList(
                Arrays.asList(1),
                Arrays.asList(3, 2),
                Arrays.asList(4, 5, 6, 7))), "perfect tree");
        check(zigzagLevelOrder(buildTree(1, 2, 3, 4, null, null, 5)).equals(Arrays.asList(
                Arrays.asList(1),
                Arrays.asList(3, 2),
                Arrays.asList(4, 5))), "ragged last level");
        check(zigzagLevelOrder(buildTree(1)).equals(Arrays.asList(Arrays.asList(1))), "single node");
        check(zigzagLevelOrder(buildTree(1, 2)).equals(Arrays.asList(
                Arrays.asList(1), Arrays.asList(2))), "two nodes");
        check(zigzagLevelOrder(null).isEmpty(), "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
