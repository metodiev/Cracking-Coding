/** Task 246 - Serialize/deserialize binary search tree compactly. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

public class Solution {
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

    static String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        for (int v : preorder(root)) {
            if (sb.length() > 0) sb.append(',');
            sb.append(v);
        }
        return sb.toString();
    }

    /** Naive form kept for comparison: explicit '#' markers for missing children. */
    static String serializeWithNulls(TreeNode node, StringBuilder sb) {
        if (sb.length() > 0) sb.append(',');
        if (node == null) { sb.append('#'); return sb.toString(); }
        sb.append(node.val);
        serializeWithNulls(node.left, sb);
        serializeWithNulls(node.right, sb);
        return sb.toString();
    }

    private static int[] values;
    private static int pos;

    static TreeNode deserialize(String data) {
        if (data.isEmpty()) return null;
        String[] parts = data.split(",");
        values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) values[i] = Integer.parseInt(parts[i]);
        pos = 0;
        return build(Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private static TreeNode build(long lo, long hi) {
        if (pos >= values.length) return null;
        long value = values[pos];
        if (value < lo || value > hi) return null;
        pos++;
        TreeNode node = new TreeNode((int) value);
        node.left = build(lo, value - 1);
        node.right = build(value + 1, hi);
        return node;
    }

    static TreeNode buildBst(int[] input) {
        TreeNode root = null;
        for (int value : input) {
            if (root == null) { root = new TreeNode(value); continue; }
            TreeNode node = root;
            while (true) {
                if (value < node.val) {
                    if (node.left == null) { node.left = new TreeNode(value); break; }
                    node = node.left;
                } else {
                    if (node.right == null) { node.right = new TreeNode(value); break; }
                    node = node.right;
                }
            }
        }
        return root;
    }

    public static void main(String[] args) {
        int[] vals = {5, 3, 8, 2, 4, 7, 9};
        TreeNode root = buildBst(vals);
        String data = serialize(root);
        check(data.equals("5,3,2,4,8,7,9"), "preorder serialization");
        check(serializeWithNulls(root, new StringBuilder()).equals("5,3,2,#,#,4,#,#,8,7,#,#,9,#,#"), "null-marker form");

        TreeNode clone = deserialize(data);
        check(preorder(clone).toString().equals("[5, 3, 2, 4, 8, 7, 9]"), "clone preorder");
        check(inorder(clone).toString().equals("[2, 3, 4, 5, 7, 8, 9]"), "clone inorder sorted");
        check(data.split(",").length < serializeWithNulls(root, new StringBuilder()).split(",").length, "compact form is shorter");
        check(deserialize("") == null, "empty string");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
