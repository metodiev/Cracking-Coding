/** Task 245 - Reconstruct binary tree from preorder with null markers. (Tier: Intermediate, CTCI Ch 4) */
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

    private static int tokenPos;

    /** tokens like "1,2,#,#,3,#,#" where "#" marks a missing child. */
    static TreeNode buildFromPreorderTokens(String csv) {
        if (csv.isEmpty()) return null;
        String[] tokens = csv.split(",");
        tokenPos = 0;
        return build(tokens);
    }

    private static TreeNode build(String[] tokens) {
        if (tokenPos >= tokens.length || tokens[tokenPos].equals("#")) {
            tokenPos++;
            return null;
        }
        TreeNode node = new TreeNode(Integer.parseInt(tokens[tokenPos++]));
        node.left = build(tokens);
        node.right = build(tokens);
        return node;
    }

    public static void main(String[] args) {
        TreeNode root = buildFromPreorderTokens("1,2,#,#,3,#,#");
        check(preorder(root).toString().equals("[1, 2, 3]"), "three nodes preorder");
        check(inorder(root).toString().equals("[2, 1, 3]"), "three nodes inorder");

        TreeNode root2 = buildFromPreorderTokens("1,2,4,#,#,5,#,#,3,#,6,#,#");
        check(preorder(root2).toString().equals("[1, 2, 4, 5, 3, 6]"), "six nodes preorder");
        check(inorder(root2).toString().equals("[4, 2, 5, 1, 3, 6]"), "six nodes inorder");

        check(buildFromPreorderTokens("#") == null, "all-null tree");
        TreeNode single = buildFromPreorderTokens("9,#,#");
        check(single.val == 9 && single.left == null && single.right == null, "single node");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
