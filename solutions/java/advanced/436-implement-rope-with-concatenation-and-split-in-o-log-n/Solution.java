/** Task 436 - Implement rope with concatenation and split in O(log n). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Rope = balanced binary tree over text chunks. Documented thresholds: every leaf holds at most
 * LEAF_LIMIT = 32 characters (a longer input string is pre-split into such leaves), and after every
 * concat/split the root height is checked against 2 * ceil(log2(#leaves + 1)) + 1 - if a rope grows
 * taller than that it is rebuilt from its leaves into a perfectly balanced tree. Each node caches its
 * character size, height and leaf count, so concat is O(1) plus an amortised rebuild, and split
 * recurses along one root-to-leaf path (plus an amortised rebuild of the two results). The rebuild
 * counter is documented and asserted: building a 200-character rope by 200 single-character concats
 * triggers at least one rebalance. Assertions compare toString() with a plain-string oracle across
 * LCG operation scripts including splits at 0 and at the full length, and check leaf, size and height
 * invariants after every operation.
 */
import java.util.*;

class RopeNode {
    final String text;
    final RopeNode left;
    final RopeNode right;
    final int size;
    final int height;
    final int leaves;

    RopeNode(String text) {
        this.text = text;
        this.left = null;
        this.right = null;
        this.size = text.length();
        this.height = 1;
        this.leaves = 1;
    }

    RopeNode(RopeNode left, RopeNode right) {
        this.text = null;
        this.left = left;
        this.right = right;
        this.size = left.size + right.size;
        this.height = 1 + Math.max(left.height, right.height);
        this.leaves = left.leaves + right.leaves;
    }
}

class Rope {
    static final int LEAF_LIMIT = 32;

    private final RopeNode root;
    final int rebuilds;

    Rope() {
        this("");
    }

    Rope(String text) {
        List<String> chunks = leavesOf(text);
        this.root = buildBalanced(chunks, 0, chunks.size() - 1);
        this.rebuilds = 0;
    }

    private Rope(RopeNode root, int rebuilds) {
        this.root = root;
        this.rebuilds = rebuilds;
    }

    private static class Balanced {
        final RopeNode root;
        final int rebuilt;

        Balanced(RopeNode root, int rebuilt) {
            this.root = root;
            this.rebuilt = rebuilt;
        }
    }

    static List<String> leavesOf(String text) {
        List<String> chunks = new ArrayList<>();
        for (int i = 0; i < text.length(); i += LEAF_LIMIT) {
            chunks.add(text.substring(i, Math.min(i + LEAF_LIMIT, text.length())));
        }
        return chunks;
    }

    static RopeNode buildBalanced(List<String> chunks, int lo, int hi) {
        if (lo > hi) return null;
        if (lo == hi) return new RopeNode(chunks.get(lo));
        int mid = (lo + hi) / 2;
        return new RopeNode(buildBalanced(chunks, lo, mid), buildBalanced(chunks, mid + 1, hi));
    }

    static int heightBound(int leafCount) {
        return 2 * (int) Math.ceil(Math.log(leafCount + 1) / Math.log(2)) + 1;
    }

    static RopeNode rawConcat(RopeNode left, RopeNode right) {
        if (left == null) return right;
        if (right == null) return left;
        return new RopeNode(left, right);
    }

    static RopeNode[] splitRoot(RopeNode node, int index) {
        if (node == null) return new RopeNode[]{null, null};
        if (node.text != null) {
            String head = node.text.substring(0, index);
            String tail = node.text.substring(index);
            return new RopeNode[]{
                    head.isEmpty() ? null : new RopeNode(head),
                    tail.isEmpty() ? null : new RopeNode(tail)};
        }
        int leftSize = node.left.size;
        if (index < leftSize) {
            RopeNode[] parts = splitRoot(node.left, index);
            return new RopeNode[]{parts[0], rawConcat(parts[1], node.right)};
        }
        if (index == leftSize) return new RopeNode[]{node.left, node.right};
        RopeNode[] parts = splitRoot(node.right, index - leftSize);
        return new RopeNode[]{rawConcat(node.left, parts[0]), parts[1]};
    }

    int size() {
        return root == null ? 0 : root.size;
    }

    int height() {
        return root == null ? 0 : root.height;
    }

    int leafCount() {
        return root == null ? 0 : root.leaves;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Deque<RopeNode> stack = new ArrayDeque<>();
        if (root != null) stack.push(root);
        while (!stack.isEmpty()) {
            RopeNode node = stack.pop();
            if (node.text != null) sb.append(node.text);
            else {
                stack.push(node.right);
                stack.push(node.left);
            }
        }
        return sb.toString();
    }

    List<String> leafTexts() {
        List<String> out = new ArrayList<>();
        Deque<RopeNode> stack = new ArrayDeque<>();
        if (root != null) stack.push(root);
        while (!stack.isEmpty()) {
            RopeNode node = stack.pop();
            if (node.text != null) out.add(node.text);
            else {
                stack.push(node.right);
                stack.push(node.left);
            }
        }
        return out;
    }

    boolean validate() {
        int[] result = walk(root);
        return result != null && result[0] == size() && result[1] == height() && result[2] == leafCount();
    }

    private int[] walk(RopeNode node) {
        if (node == null) return new int[]{0, 0, 0};
        if (node.text != null) {
            if (node.text.length() > LEAF_LIMIT || node.text.isEmpty()) return null;
            return new int[]{node.size, node.height, node.leaves};
        }
        int[] left = walk(node.left);
        int[] right = walk(node.right);
        if (left == null || right == null) return null;
        if (node.size != left[0] + right[0]) return null;
        if (node.height != 1 + Math.max(left[1], right[1])) return null;
        if (node.leaves != left[2] + right[2]) return null;
        return new int[]{node.size, node.height, node.leaves};
    }

    private Balanced balanced(RopeNode node) {
        if (node == null || node.height <= heightBound(node.leaves)) return new Balanced(node, 0);
        List<String> chunks = new ArrayList<>();
        Deque<RopeNode> stack = new ArrayDeque<>();
        stack.push(node);
        while (!stack.isEmpty()) {
            RopeNode current = stack.pop();
            if (current.text != null) chunks.add(current.text);
            else {
                stack.push(current.right);
                stack.push(current.left);
            }
        }
        return new Balanced(buildBalanced(chunks, 0, chunks.size() - 1), 1);
    }

    Rope concat(Rope other) {
        if (root == null) return new Rope(other.root, rebuilds + other.rebuilds);
        if (other.root == null) return new Rope(root, rebuilds + other.rebuilds);
        Balanced merged = balanced(rawConcat(root, other.root));
        return new Rope(merged.root, rebuilds + other.rebuilds + merged.rebuilt);
    }

    Rope[] split(int index) {
        if (index < 0 || index > size()) throw new IndexOutOfBoundsException("split index out of bounds");
        RopeNode[] parts = splitRoot(root, index);
        Balanced first = balanced(parts[0]);
        Balanced second = balanced(parts[1]);
        return new Rope[]{
                new Rope(first.root, rebuilds + first.rebuilt),
                new Rope(second.root, rebuilds + second.rebuilt)};
    }
}

public class Solution {
    static String extraString(int step, int length) {
        char[] chars = new char[length];
        for (int i = 0; i < length; i++) chars[i] = (char) ('a' + (step * 3 + i) % 26);
        return new String(chars);
    }

    public static void main(String[] args) {
        Rope rope = new Rope("hello world");
        check(rope.toString().equals("hello world"), "hand text");
        check(rope.size() == 11, "hand size");
        check(rope.leafCount() == 1, "hand leaves");
        check(rope.height() == 1, "hand height");
        check(rope.validate(), "hand validate");

        Rope[] parts = rope.split(0);
        check(parts[0].toString().isEmpty() && parts[0].size() == 0, "split 0 left");
        check(parts[1].toString().equals("hello world"), "split 0 right");
        parts = rope.split(11);
        check(parts[0].toString().equals("hello world"), "split len left");
        check(parts[1].toString().isEmpty() && parts[1].size() == 0, "split len right");
        parts = rope.split(5);
        check(parts[0].toString().equals("hello") && parts[1].toString().equals(" world"), "split middle");
        check(parts[0].concat(parts[1]).toString().equals("hello world"), "concat back");
        check(parts[0].height() <= Rope.heightBound(parts[0].leafCount()), "small bound");

        char[] dataChars = new char[2000];
        for (int i = 0; i < 2000; i++) dataChars[i] = (char) ('a' + (i * 7 + i * i) % 26);
        String data = new String(dataChars);
        rope = new Rope(data);
        check(rope.toString().equals(data), "long text");
        check(rope.size() == 2000, "long size");
        check(rope.leafCount() == (2000 + Rope.LEAF_LIMIT - 1) / Rope.LEAF_LIMIT, "long leaves");
        check(rope.height() <= Rope.heightBound(rope.leafCount()), "long height");
        check(rope.validate(), "long validate");
        int maxLeaf = 0;
        for (String chunk : rope.leafTexts()) maxLeaf = Math.max(maxLeaf, chunk.length());
        check(maxLeaf <= Rope.LEAF_LIMIT, "leaf limit");
        parts = rope.split(999);
        check((parts[0].toString() + parts[1].toString()).equals(data), "long split parts");
        check(parts[0].concat(parts[1]).toString().equals(data), "long concat back");

        rope = new Rope();
        for (int index = 0; index < 200; index++) {
            rope = rope.concat(new Rope(String.valueOf((char) ('a' + index % 26))));
        }
        check(rope.size() == 200, "build size");
        check(rope.rebuilds >= 1, "build rebuilds " + rope.rebuilds);
        StringBuilder expectedBuilt = new StringBuilder();
        for (int i = 0; i < 200; i++) expectedBuilt.append((char) ('a' + i % 26));
        check(rope.toString().equals(expectedBuilt.toString()), "build text");
        check(rope.height() <= Rope.heightBound(rope.leafCount()), "build height");
        check(rope.validate(), "build validate");

        char[] initialChars = new char[64];
        for (int i = 0; i < 64; i++) initialChars[i] = (char) ('a' + (i * 13 + 5) % 26);
        rope = new Rope(new String(initialChars));
        String oracle = new String(initialChars);
        int seed = 4360;
        for (int step = 0; step < 100; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 4;
            if (op == 0) {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.length() + 1);
                Rope[] halves = rope.split(index);
                check(halves[0].toString().equals(oracle.substring(0, index)), "script split left " + step);
                check(halves[1].toString().equals(oracle.substring(index)), "script split right " + step);
                if (seed % 2 == 0) {
                    rope = halves[0].concat(halves[1]);
                } else {
                    rope = halves[1].concat(halves[0]);
                    oracle = oracle.substring(index) + oracle.substring(0, index);
                }
            } else if (op == 1) {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.length() + 1);
                rope = rope.split(index)[0];
                oracle = oracle.substring(0, index);
            } else if (op == 2) {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.length() + 1);
                rope = rope.split(index)[1];
                oracle = oracle.substring(index);
            } else {
                String extra = extraString(step, seed % 20);
                rope = rope.concat(new Rope(extra));
                oracle = oracle + extra;
            }
            check(rope.toString().equals(oracle), "script text " + step);
            check(rope.size() == oracle.length(), "script size " + step);
            check(rope.height() <= Rope.heightBound(rope.leafCount()), "script height " + step);
            check(rope.validate(), "script validate " + step);
            check(String.join("", rope.leafTexts()).equals(oracle), "script leaves " + step);
        }
        parts = rope.split(0);
        check(parts[0].toString().isEmpty() && parts[1].toString().equals(oracle), "final split 0");
        parts = rope.split(rope.size());
        check(parts[1].toString().isEmpty() && parts[0].toString().equals(oracle), "final split len");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
