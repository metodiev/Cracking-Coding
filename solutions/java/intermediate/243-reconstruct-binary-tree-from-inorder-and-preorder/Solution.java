/** Task 243 - Reconstruct binary tree from inorder and preorder. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

public class Solution {
    static TreeNode buildTree(Integer... values) {
        if (values.length == 0 || values[0] == null) return null;
        TreeNode root = new TreeNode(values[0]);
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int i = 1;
        while (!q.isEmpty() && i < values.length) {
            TreeNode node = q.poll();
            if (i < values.length && values[i] != null) { node.left = new TreeNode(values[i]); q.add(node.left); }
            i++;
            if (i < values.length && values[i] != null) { node.right = new TreeNode(values[i]); q.add(node.right); }
            i++;
        }
        return root;
    }

    static List<Integer> preorder(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        pre(node, out);
        return out;
    }

    private static void pre(TreeNode node, List<Integer> out) {
        if (node == null) return;
        out.add(node.val);
        pre(node.left, out);
        pre(node.right, out);
    }

    static List<Integer> inorder(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        in(node, out);
        return out;
    }

    private static void in(TreeNode node, List<Integer> out) {
        if (node == null) return;
        in(node.left, out);
        out.add(node.val);
        in(node.right, out);
    }

    private static int prePos;
    private static Map<Integer, Integer> inIndex;

    static TreeNode buildFromPreIn(int[] preorder, int[] inorder) {
        if (preorder.length == 0 || preorder.length != inorder.length)
            throw new IllegalArgumentException("preorder and inorder must be non-empty and equal-length");
        prePos = 0;
        inIndex = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) inIndex.put(inorder[i], i);
        return build(preorder, 0, inorder.length - 1);
    }

    private static TreeNode build(int[] preorder, int lo, int hi) {
        if (lo > hi) return null;
        int rootVal = preorder[prePos++];
        TreeNode node = new TreeNode(rootVal);
        int mid = inIndex.get(rootVal);
        node.left = build(preorder, lo, mid - 1);
        node.right = build(preorder, mid + 1, hi);
        return node;
    }

    public static void main(String[] args) {
        TreeNode root = buildFromPreIn(new int[]{3, 9, 20, 15, 7}, new int[]{9, 3, 15, 20, 7});
        check(preorder(root).toString().equals("[3, 9, 20, 15, 7]"), "preorder round trip");
        check(inorder(root).toString().equals("[9, 3, 15, 20, 7]"), "inorder round trip");
        check(inorder(buildFromPreIn(new int[]{1}, new int[]{1})).toString().equals("[1]"), "single");
        TreeNode spine = buildFromPreIn(new int[]{1, 2, 3}, new int[]{3, 2, 1});
        check(preorder(spine).toString().equals("[1, 2, 3]") && inorder(spine).toString().equals("[3, 2, 1]"), "left spine");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
