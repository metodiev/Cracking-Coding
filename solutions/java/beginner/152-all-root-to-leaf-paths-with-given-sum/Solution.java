/** Task 152 - All root-to-leaf paths with given sum. (Tier: Beginner, CTCI Ch 8) */
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

    /** All root-to-leaf paths summing to target, left subtree before right. */
    static List<List<Integer>> pathSumPaths(TreeNode root, int target) {
        List<List<Integer>> result = new ArrayList<>();
        dfs(root, target, new ArrayList<>(), result);
        return result;
    }

    private static void dfs(TreeNode node, int remaining, List<Integer> path, List<List<Integer>> result) {
        if (node == null) return;
        path.add(node.val);
        if (node.left == null && node.right == null) {
            if (remaining == node.val) result.add(new ArrayList<>(path));
        } else {
            dfs(node.left, remaining - node.val, path, result);
            dfs(node.right, remaining - node.val, path, result);
        }
        path.remove(path.size() - 1);
    }

    public static void main(String[] args) {
        TreeNode root = buildTree(5, 4, 8, 11, null, 13, 4, 7, 2, null, null, 5, 1);
        check(pathSumPaths(root, 22).toString().equals("[[5, 4, 11, 2], [5, 8, 4, 5]]"), "two paths");
        check(pathSumPaths(root, 26).toString().equals("[[5, 8, 13]]"), "single path");
        check(pathSumPaths(root, 27).toString().equals("[[5, 4, 11, 7]]"), "left-most path");
        check(pathSumPaths(root, 100).isEmpty(), "no matching path");
        check(pathSumPaths(null, 0).isEmpty(), "empty tree");
        check(pathSumPaths(buildTree(7), 7).toString().equals("[[7]]"), "single node match");
        check(pathSumPaths(buildTree(7), 1).isEmpty(), "single node miss");
        check(pathSumPaths(buildTree(1, 2), 1).isEmpty(), "one missing child is not an end");
        check(pathSumPaths(buildTree(1, 2), 3).toString().equals("[[1, 2]]"), "one-child path");
        check(pathSumPaths(buildTree(-1, -2, -3), -3).toString().equals("[[-1, -2]]"), "negatives");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
