/** Task 244 - Reconstruct binary tree from inorder and postorder. (Tier: Intermediate, CTCI Ch 4) */
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

    private static int postPos;
    private static Map<Integer, Integer> inIndex;

    static TreeNode buildFromInPost(int[] inorder, int[] postorder) {
        if (postorder.length == 0 || postorder.length != inorder.length)
            throw new IllegalArgumentException("inorder and postorder must be non-empty and equal-length");
        postPos = postorder.length - 1;
        inIndex = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) inIndex.put(inorder[i], i);
        return build(postorder, 0, inorder.length - 1);
    }

    private static TreeNode build(int[] postorder, int lo, int hi) {
        if (lo > hi) return null;
        int rootVal = postorder[postPos--];
        TreeNode node = new TreeNode(rootVal);
        int mid = inIndex.get(rootVal);
        node.right = build(postorder, mid + 1, hi);
        node.left = build(postorder, lo, mid - 1);
        return node;
    }

    static List<Integer> postorderList(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        post(node, out);
        return out;
    }

    private static void post(TreeNode node, List<Integer> out) {
        if (node == null) return;
        post(node.left, out);
        post(node.right, out);
        out.add(node.val);
    }

    public static void main(String[] args) {
        int[] ino = {9, 3, 15, 20, 7};
        int[] post = {9, 15, 7, 20, 3};
        TreeNode root = buildFromInPost(ino, post);
        check(inorder(root).toString().equals("[9, 3, 15, 20, 7]"), "inorder round trip");
        check(postorderList(root).toString().equals("[9, 15, 7, 20, 3]"), "postorder round trip");
        check(inorder(buildFromInPost(new int[]{1}, new int[]{1})).toString().equals("[1]"), "single");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
