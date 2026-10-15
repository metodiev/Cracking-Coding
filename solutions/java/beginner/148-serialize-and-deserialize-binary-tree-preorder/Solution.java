/** Task 148 - Serialize and deserialize binary tree (preorder). (Tier: Beginner, CTCI Ch 4)
 *
 * Format (identical in all four languages): preorder values separated by commas,
 * with "#" marking every missing child, e.g. "1,2,#,#,3,#,#". An empty tree is "#"
 * and a single node is "7,#,#", so every tree with n nodes uses exactly 2n + 1
 * tokens and the stream can be rebuilt by a single left-to-right pass.
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

    /** Preorder walk that also emits a sentinel for each missing child. */
    static String serialize(TreeNode root) {
        List<String> parts = new ArrayList<>();
        serializeInto(root, parts);
        return String.join(",", parts);
    }

    private static void serializeInto(TreeNode node, List<String> parts) {
        if (node == null) {
            parts.add("#");
            return;
        }
        parts.add(String.valueOf(node.val));
        serializeInto(node.left, parts);
        serializeInto(node.right, parts);
    }

    /** Rebuild the tree from the preorder stream produced by serialize. */
    static TreeNode deserialize(String data) {
        if (data.isEmpty()) return null;
        return build(data.split(","), new int[] {0});     // pos[0] is the stream cursor
    }

    private static TreeNode build(String[] tokens, int[] pos) {
        String token = tokens[pos[0]++];
        if (token.equals("#")) return null;
        TreeNode node = new TreeNode(Integer.parseInt(token));
        node.left = build(tokens, pos);                  // the left subtree comes next
        node.right = build(tokens, pos);
        return node;
    }

    /** Preorder values only (the sentinels are dropped). */
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

    /** Round-trip check: same shape and same values everywhere. */
    static boolean areIdentical(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.val == b.val && areIdentical(a.left, b.left) && areIdentical(a.right, b.right);
    }

    public static void main(String[] args) {
        check(serialize(null).equals("#"), "empty tree");
        check(deserialize("#") == null && deserialize("") == null, "deserialize empty");
        check(serialize(buildTree(7)).equals("7,#,#"), "single node");
        check(serialize(buildTree(1, 2, null, 3)).equals("1,2,3,#,#,#,#"), "left chain");
        check(serialize(buildTree(1, 2, 5, 3, 4, null, 6)).equals("1,2,3,#,#,4,#,#,5,#,6,#,#"), "sample");
        check(serialize(buildTree(1, 2, 3)).split(",").length == 7, "2n + 1 tokens");
        Integer[][] cases = {
            {}, {7}, {1, 2, null, 3}, {1, 2, 5, 3, 4, null, 6}, {1, 2, 3, 4, 5, null, 6}
        };
        for (Integer[] data : cases) {
            TreeNode tree = buildTree(data);
            String text = serialize(tree);
            List<String> fromStream = new ArrayList<>();
            for (String token : text.split(",")) {
                if (!token.equals("#")) fromStream.add(token);
            }
            List<String> fromTree = new ArrayList<>();
            for (int value : preorderValues(tree)) fromTree.add(String.valueOf(value));
            check(fromTree.equals(fromStream), "preorder matches the stream");
            TreeNode rebuilt = deserialize(text);
            check(areIdentical(tree, rebuilt), "round trip keeps shape and values");
            check(serialize(rebuilt).equals(text), "string round trip");
            if (tree != null) {
                check(rebuilt != tree, "deserialize builds a fresh structure");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
