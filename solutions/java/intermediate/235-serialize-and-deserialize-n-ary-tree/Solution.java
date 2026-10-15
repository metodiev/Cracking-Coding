/** Task 235 - Serialize and deserialize N-ary tree. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class NAryNode {
    int val;
    List<NAryNode> children = new ArrayList<>();
    NAryNode(int v) { val = v; }
}

public class Solution {
    /**
     * Wire format (identical across the Python, Java, JavaScript and C# solutions):
     * a recursive preorder where every node is written as "value[childCount]" followed
     * by the serialized children, e.g. root 1 with children [3, 2, 4] where 3 has
     * children [5, 6] serializes to "1[3]3[2]5[0]6[0]2[0]4[0]"; empty tree -> "".
     */
    static String serialize(NAryNode root) {
        if (root == null) return "";
        StringBuilder sb = new StringBuilder();
        serializeInto(root, sb);
        return sb.toString();
    }

    private static void serializeInto(NAryNode node, StringBuilder sb) {
        sb.append(node.val).append('[').append(node.children.size()).append(']');
        for (NAryNode child : node.children) serializeInto(child, sb);
    }

    static NAryNode deserialize(String data) {
        if (data.isEmpty()) return null;
        int[] pos = {0};
        return readNode(data, pos);
    }

    private static NAryNode readNode(String data, int[] pos) {
        int start = pos[0];
        while (data.charAt(pos[0]) != '[') pos[0]++;
        int val = Integer.parseInt(data.substring(start, pos[0]));
        pos[0]++;                                  // skip '['
        start = pos[0];
        while (data.charAt(pos[0]) != ']') pos[0]++;
        int childCount = Integer.parseInt(data.substring(start, pos[0]));
        pos[0]++;                                  // skip ']'
        NAryNode node = new NAryNode(val);
        for (int i = 0; i < childCount; i++) node.children.add(readNode(data, pos));
        return node;
    }

    static int depth(NAryNode node) {
        if (node == null) return 0;
        int best = 0;
        for (NAryNode child : node.children) best = Math.max(best, depth(child));
        return 1 + best;
    }

    static boolean treesEqual(NAryNode a, NAryNode b) {
        if (a == null || b == null) return a == b;
        if (a.val != b.val || a.children.size() != b.children.size()) return false;
        for (int i = 0; i < a.children.size(); i++)
            if (!treesEqual(a.children.get(i), b.children.get(i))) return false;
        return true;
    }

    public static void main(String[] args) {
        // exact wire format for the documented example
        NAryNode example = new NAryNode(1);
        NAryNode three = new NAryNode(3), two = new NAryNode(2), four = new NAryNode(4);
        three.children.add(new NAryNode(5));
        three.children.add(new NAryNode(6));
        example.children.add(three);
        example.children.add(two);
        example.children.add(four);
        check(serialize(example).equals("1[3]3[2]5[0]6[0]2[0]4[0]"), "example wire format");
        check(treesEqual(deserialize("1[3]3[2]5[0]6[0]2[0]4[0]"), example), "example parsed");
        check(serialize(deserialize(serialize(example))).equals(serialize(example)), "wire round trip");

        // empty tree
        check(serialize(null).equals(""), "empty serializes to blank");
        check(deserialize("") == null, "blank parses to null");

        // single node
        check(serialize(new NAryNode(42)).equals("42[0]"), "single node format");
        check(treesEqual(deserialize("42[0]"), new NAryNode(42)), "single node round trip");

        // deep chain of 50 nodes
        NAryNode chain = new NAryNode(50);
        for (int v = 49; v >= 1; v--) {
            NAryNode parent = new NAryNode(v);
            parent.children.add(chain);
            chain = parent;
        }
        check(depth(chain) == 50, "chain depth built");
        check(serialize(chain).startsWith("1[1]2[1]3[1]"), "chain prefix");
        check(treesEqual(deserialize(serialize(chain)), chain), "chain round trip");
        check(depth(deserialize(serialize(chain))) == 50, "chain depth restored");

        // node with many children
        NAryNode many = new NAryNode(1);
        for (int v = 0; v < 10; v++) many.children.add(new NAryNode(v));
        StringBuilder expected = new StringBuilder("1[10]");
        for (int v = 0; v < 10; v++) expected.append(v).append("[0]");
        check(serialize(many).equals(expected.toString()), "ten children format");
        NAryNode restoredMany = deserialize(serialize(many));
        check(treesEqual(restoredMany, many), "ten children round trip");
        check(restoredMany.children.size() == 10, "ten children restored");
        for (int v = 0; v < 10; v++) check(restoredMany.children.get(v).val == v, "child value " + v);

        // negative values are handled by the scanner
        NAryNode negative = new NAryNode(-5);
        negative.children.add(new NAryNode(-7));
        check(serialize(negative).equals("-5[1]-7[0]"), "negative values");
        check(treesEqual(deserialize("-5[1]-7[0]"), negative), "negative round trip");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
