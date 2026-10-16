/** Task 611 - Implement dynamic succinct trees (conceptual). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A dynamic succinct tree keeps the balanced-parenthesis word of the tree in a dynamic bit sequence and
 * derives the tree from rank/select on that sequence. Here the sequence is a treap whose nodes are
 * 64-bit chunks; every chunk carries size, popcount and the minimum prefix excess of its subtree, so
 * rank1, select1, bitAt, excessAt and findClose all descend the tree instead of scanning it. insertBit
 * and deleteBit are split/merge operations that concat two neighbouring chunks that fit into one, so
 * the chunks stay dense; a node of the tree is a matched pair, a leaf is the two bits "()", and
 * addLeaf/removeLeaf insert or erase such a pair.
 *
 * Simplified (this is the conceptual variant): the tree is not rebalanced - the treap priorities come
 * from the deterministic PRNG, so the operations are O(log n) expected with no guarantee, whereas a
 * real dynamic succinct tree uses a weight-balanced or finger-tree layer; only leaves can be added and
 * removed because inserting an internal node needs the heavier split/join of subtrees; and the root is
 * fixed at the first pair. Verified: after every one of several hundred randomized operations the
 * stored word is byte-for-byte the canonical parenthesis word of an independent tree oracle, stays a
 * balanced word, and rank1/select1/bitAt/excessAt/findClose agree with naive scans of that word.
 */
import java.util.*;

class BpChunk {
    long bits;                 // bit p of the chunk (0-based) is (bits >>> p) & 1
    int len;
    int size;                  // bits in the subtree
    int ones;                  // one-bits in the subtree
    int minRel;                // minimum relative excess over all prefixes, including the empty one
    int priority;
    BpChunk left, right;
}

class DynamicBp {
    private static final int CHUNK_BITS = 64;
    private BpChunk root;
    private long state = 0x9E3779B97F4A7C15L;

    private long nextPriority() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long mask(int bits) { return bits >= 64 ? -1L : (1L << bits) - 1; }

    static int size(BpChunk n) { return n == null ? 0 : n.size; }

    static int delta(BpChunk n) { return n == null ? 0 : 2 * n.ones - n.size; }

    static int minRel(BpChunk n) { return n == null ? 0 : n.minRel; }

    private BpChunk chunk(long bits, int len) {
        BpChunk n = new BpChunk();
        n.bits = bits & mask(len);
        n.len = len;
        n.priority = (int) (nextPriority() >>> 33);
        update(n);
        return n;
    }

    private static void update(BpChunk n) {
        int dl = delta(n.left);
        n.size = size(n.left) + n.len + size(n.right);
        n.ones = (n.left == null ? 0 : n.left.ones) + Long.bitCount(n.bits & mask(n.len))
                + (n.right == null ? 0 : n.right.ones);
        int best = Math.min(minRel(n.left), dl + chunkMinRel(n));
        best = Math.min(best, dl + chunkDelta(n) + minRel(n.right));
        n.minRel = best;
    }

    /** Minimum relative excess inside one chunk: min over k of (2 * ones(first k) - k). */
    private static int chunkMinRel(BpChunk n) {
        int best = 0, e = 0;
        for (int k = 0; k < n.len; k++) {
            e += ((n.bits >>> k) & 1L) != 0 ? 1 : -1;
            best = Math.min(best, e);
        }
        return best;
    }

    private static int chunkDelta(BpChunk n) {
        return 2 * Long.bitCount(n.bits & mask(n.len)) - n.len;
    }

    private static int chunkOnes(BpChunk n) { return Long.bitCount(n.bits & mask(n.len)); }

    /** Treap merge that fuses two neighbouring chunks whenever they still fit into one chunk. */
    private BpChunk merge(BpChunk a, BpChunk b) {
        if (a == null) return b;
        if (b == null) return a;
        if (a.right == null && b.left == null && a.len + b.len <= CHUNK_BITS) {
            BpChunk fused = chunk(a.bits | (b.bits << a.len), a.len + b.len);
            fused.left = a.left;
            fused.right = b.right;
            fused.priority = Math.max(a.priority, b.priority);   // keeps the heap order of the merge
            update(fused);
            return fused;
        }
        if (a.priority >= b.priority) {
            a.right = merge(a.right, b);
            update(a);
            return a;
        }
        b.left = merge(a, b.left);
        update(b);
        return b;
    }

    /** Left part keeps the first k bits. */
    private BpChunk[] split(BpChunk n, int k) {
        if (n == null) return new BpChunk[]{null, null};
        int leftSize = size(n.left);
        if (k < leftSize) {
            BpChunk[] parts = split(n.left, k);
            n.left = parts[1];
            update(n);
            return new BpChunk[]{parts[0], n};
        }
        if (k > leftSize + n.len) {
            BpChunk[] parts = split(n.right, k - leftSize - n.len);
            n.right = parts[0];
            update(n);
            return new BpChunk[]{n, parts[1]};
        }
        int inChunk = k - leftSize;
        BpChunk left = n.left, right = n.right;
        if (inChunk > 0) left = merge(left, chunk(n.bits & mask(inChunk), inChunk));
        if (inChunk < n.len) right = merge(chunk(n.bits >>> inChunk, n.len - inChunk), right);
        return new BpChunk[]{left, right};
    }

    int size() { return size(root); }

    int ones() { return root == null ? 0 : root.ones; }

    int height() { return height(root); }

    private static int height(BpChunk n) {
        return n == null ? 0 : 1 + Math.max(height(n.left), height(n.right));
    }

    /** Ones in positions 1..p. */
    int rank1(int p) {
        if (p <= 0) return 0;
        if (p > size(root)) p = size(root);
        return rank1(root, p);
    }

    private static int rank1(BpChunk n, int p) {
        int leftSize = size(n.left);
        if (p <= leftSize) return rank1(n.left, p);
        if (p <= leftSize + n.len) {
            return (n.left == null ? 0 : n.left.ones) + Long.bitCount(n.bits & mask(p - leftSize));
        }
        return (n.left == null ? 0 : n.left.ones) + chunkOnes(n) + rank1(n.right, p - leftSize - n.len);
    }

    int bitAt(int p) {
        if (p < 1 || p > size(root)) return -1;
        return bitAt(root, p);
    }

    private static int bitAt(BpChunk n, int p) {
        int leftSize = size(n.left);
        if (p <= leftSize) return bitAt(n.left, p);
        if (p <= leftSize + n.len) return (int) ((n.bits >>> (p - leftSize - 1)) & 1L);
        return bitAt(n.right, p - leftSize - n.len);
    }

    int excessAt(int p) { return 2 * rank1(p) - Math.min(p, size(root)); }

    int select1(int k) {
        if (k < 1 || k > ones()) return -1;
        return select1(root, k);
    }

    private static int select1(BpChunk n, int k) {
        int leftOnes = n.left == null ? 0 : n.left.ones;
        if (k <= leftOnes) return select1(n.left, k);
        if (k <= leftOnes + chunkOnes(n)) {
            int need = k - leftOnes;
            for (int i = 0; i < n.len; i++) {
                if (((n.bits >>> i) & 1L) != 0 && --need == 0) return size(n.left) + i + 1;
            }
        }
        return size(n.left) + n.len + select1(n.right, k - leftOnes - chunkOnes(n));
    }

    void insertBit(int p, int bit) {
        if (p < 1 || p > size(root) + 1) throw new IllegalArgumentException("insert at " + p);
        BpChunk[] parts = split(root, p - 1);
        root = merge(merge(parts[0], chunk(bit, 1)), parts[1]);
    }

    void deleteBit(int p) {
        if (p < 1 || p > size(root)) throw new IllegalArgumentException("delete at " + p);
        BpChunk[] head = split(root, p - 1);
        BpChunk[] tail = split(head[1], 1);
        root = merge(head[0], tail[1]);
    }

    /** Position of the ')' matching the '(' at openPos, found by an aggregate descend. */
    int findClose(int openPos) {
        if (openPos < 1 || openPos > size(root) || bitAt(openPos) != 1) {
            throw new IllegalArgumentException("not an open paren: " + openPos);
        }
        int level = excessAt(openPos);                 // the excess just after this open paren
        BpChunk[] head = split(root, openPos);
        int offset = firstOffsetWithExcess(head[1], -1);   // the matching close returns to level - 1
        root = merge(head[0], head[1]);                // the split is undone, the sequence is unchanged
        if (offset < 0) return -1;
        int closePos = openPos + offset;
        if (excessAt(closePos) != level - 1) throw new IllegalStateException("unbalanced word");
        return closePos;
    }

    /** First 0-based offset o >= 1 in the subtree where the relative excess equals target. */
    private static int firstOffsetWithExcess(BpChunk n, int target) {
        if (n == null) return -1;
        if (minRel(n.left) <= target) return firstOffsetWithExcess(n.left, target);
        int afterLeft = delta(n.left);
        int inChunk = firstInChunk(n, target - afterLeft);
        if (inChunk > 0) return size(n.left) + inChunk;
        int rest = firstOffsetWithExcess(n.right, target - afterLeft - chunkDelta(n));
        return rest < 0 ? -1 : size(n.left) + n.len + rest;
    }

    /** First 1-based offset inside a chunk whose prefix excess equals target, or -1. */
    private static int firstInChunk(BpChunk n, int target) {
        int e = 0;
        for (int k = 0; k < n.len; k++) {
            e += ((n.bits >>> k) & 1L) != 0 ? 1 : -1;
            if (e == target) return k + 1;
        }
        return -1;
    }

    String materialize() {
        StringBuilder sb = new StringBuilder();
        collect(root, sb);
        return sb.toString();
    }

    private static void collect(BpChunk n, StringBuilder sb) {
        if (n == null) return;
        collect(n.left, sb);
        for (int i = 0; i < n.len; i++) sb.append(((n.bits >>> i) & 1L) != 0 ? '(' : ')');
        collect(n.right, sb);
    }
}

/** Independent oracle: an ordinary tree of child lists plus its canonical parenthesis word. */
class TreeOracle {
    private final Map<Integer, List<Integer>> children = new LinkedHashMap<>();
    private final Set<Integer> alive = new LinkedHashSet<>();
    private int nextId = 1;

    TreeOracle() {
        children.put(1, new ArrayList<>());
        alive.add(1);
        nextId = 2;
    }

    int addLeaf(int parentId, boolean asFirst) {
        int id = nextId++;
        List<Integer> kids = children.get(parentId);
        if (asFirst) kids.add(0, id); else kids.add(id);
        children.put(id, new ArrayList<>());
        alive.add(id);
        return id;
    }

    boolean removeLeaf(int id) {
        if (id == 1 || !children.get(id).isEmpty()) return false;
        for (List<Integer> kids : children.values()) kids.remove((Integer) id);
        children.remove(id);
        alive.remove(id);
        return true;
    }

    int nodeCount() { return alive.size(); }

    int[] preorderIds() {
        List<Integer> order = new ArrayList<>();
        walk(1, order);
        int[] out = new int[order.size()];
        for (int i = 0; i < out.length; i++) out[i] = order.get(i);
        return out;
    }

    private void walk(int id, List<Integer> order) {
        order.add(id);
        for (int child : children.get(id)) walk(child, order);
    }

    int preorderIndex(int id) {
        int[] order = preorderIds();
        for (int i = 0; i < order.length; i++) if (order[i] == id) return i;
        return -1;
    }

    int[] leavesExcludingRoot() {
        List<Integer> leaves = new ArrayList<>();
        for (int id : alive) if (id != 1 && children.get(id).isEmpty()) leaves.add(id);
        int[] out = new int[leaves.size()];
        for (int i = 0; i < out.length; i++) out[i] = leaves.get(i);
        return out;
    }

    String canonicalBp() {
        StringBuilder sb = new StringBuilder();
        emit(1, sb);
        return sb.toString();
    }

    private void emit(int id, StringBuilder sb) {
        sb.append('(');
        for (int child : children.get(id)) emit(child, sb);
        sb.append(')');
    }
}

public class Solution {
    private static long state = 0x243F6A8885A308D3L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static boolean isBalanced(String bp) {
        int e = 0;
        for (int i = 0; i < bp.length(); i++) {
            e += bp.charAt(i) == '(' ? 1 : -1;
            if (e < 0) return false;
        }
        return e == 0;
    }

    static int naiveRank(String bp, int p) {
        int count = 0;
        for (int i = 0; i < p; i++) if (bp.charAt(i) == '(') count++;
        return count;
    }

    static int naiveExcess(String bp, int p) { return 2 * naiveRank(bp, p) - p; }

    static int naiveMatch(String bp, int openPos) {
        int e = 0;
        for (int p = openPos; p <= bp.length(); p++) {
            e += bp.charAt(p - 1) == '(' ? 1 : -1;
            if (e == 0) return p;
        }
        return -1;
    }

    static void checkBitvector(DynamicBp bp, String flat, String label) {
        int p = 0;
        while (p <= flat.length()) {
            check(bp.rank1(p) == naiveRank(flat, p), label + ": rank1 at " + p);
            p++;
        }
        check(bp.excessAt(0) == 0 && bp.excessAt(flat.length()) == 0, label + ": excess at the ends");
        for (int k = 1; k <= bp.ones(); k++) {
            int pos = bp.select1(k);
            check(flat.charAt(pos - 1) == '(', label + ": select1 points at an open paren");
            check(bp.findClose(pos) == naiveMatch(flat, pos), label + ": findClose of open " + k);
        }
        check(bp.select1(0) == -1 && bp.select1(bp.ones() + 1) == -1, label + ": select1 bounds");
        for (int i = 1; i <= flat.length(); i++) {
            check(bp.bitAt(i) == (flat.charAt(i - 1) == '(' ? 1 : 0), label + ": bitAt " + i);
            check(bp.excessAt(i) == naiveExcess(flat, i), label + ": excessAt " + i);
        }
        check(bp.bitAt(0) == -1 && bp.bitAt(flat.length() + 1) == -1, label + ": bitAt bounds");
    }

    public static void main(String[] args) {
        // a bare sequence: insert/delete at the ends and in the middle
        DynamicBp seq = new DynamicBp();
        seq.insertBit(1, 1);
        seq.insertBit(2, 0);
        check(seq.materialize().equals("()"), "a single pair");
        check(seq.size() == 2 && seq.ones() == 1 && seq.excessAt(1) == 1, "statistics of a pair");
        seq.insertBit(2, 1);
        seq.insertBit(3, 0);
        check(seq.materialize().equals("(())"), "first child inserted");
        seq.deleteBit(3);
        seq.deleteBit(2);
        check(seq.materialize().equals("()"), "the child was removed again");
        for (int i = 0; i < 200; i++) seq.insertBit(seq.size() + 1, (i % 2 == 0) ? 1 : 0);
        check(seq.size() == 202, "size after bulk inserts");
        seq.insertBit(1, 0);
        check(seq.bitAt(1) == 0 && seq.size() == 203, "insert at the front");
        seq.deleteBit(1);
        check(seq.size() == 202 && seq.bitAt(1) == 1, "delete at the front");
        boolean threw = false;
        try { seq.insertBit(seq.size() + 5, 1); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "insert out of range rejected");
        threw = false;
        try { seq.deleteBit(seq.size() + 1); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "delete out of range rejected");
        threw = false;
        try { seq.findClose(2); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "findClose of a close paren rejected");

        // tree level: a few hundred randomized leaf insert/remove steps, checked against the oracle
        DynamicBp bp = new DynamicBp();
        bp.insertBit(1, 1);
        bp.insertBit(2, 0);
        TreeOracle oracle = new TreeOracle();
        check(bp.materialize().equals(oracle.canonicalBp()), "the root word");
        int truncated = 0;
        for (int step = 0; step < 300; step++) {
            boolean grow = oracle.nodeCount() < 40 && (nextLong() >>> 33) % 100 < 60;
            if (grow) {
                int[] ids = oracle.preorderIds();
                int parentId = ids[(int) ((nextLong() >>> 33) % ids.length)];
                boolean asFirst = ((nextLong() >>> 33) & 1L) == 0;
                int openPos = bp.select1(oracle.preorderIndex(parentId) + 1);
                if (asFirst) {
                    bp.insertBit(openPos + 1, 1);
                    bp.insertBit(openPos + 2, 0);
                } else {
                    int closePos = bp.findClose(openPos);
                    bp.insertBit(closePos, 1);
                    bp.insertBit(closePos + 1, 0);
                }
                oracle.addLeaf(parentId, asFirst);
            } else {
                int[] leaves = oracle.leavesExcludingRoot();
                if (leaves.length == 0) {
                    truncated++;
                    continue;
                }
                int leafId = leaves[(int) ((nextLong() >>> 33) % leaves.length)];
                int openPos = bp.select1(oracle.preorderIndex(leafId) + 1);
                check(bp.findClose(openPos) == openPos + 1, "a leaf is the pair at its open paren");
                bp.deleteBit(openPos);
                bp.deleteBit(openPos);
                check(oracle.removeLeaf(leafId), "the oracle drops the leaf");
            }
            String flat = bp.materialize();
            check(flat.equals(oracle.canonicalBp()), "step " + step + ": word equals the canonical BP");
            check(isBalanced(flat), "step " + step + ": the word stays balanced");
            check(bp.size() == flat.length() && bp.ones() == oracle.nodeCount(),
                    "step " + step + ": one open paren per node");
            checkBitvector(bp, flat, "step " + step);
        }
        check(oracle.nodeCount() > 5, "the random walk kept a real tree");
        System.out.println("final tree: " + oracle.nodeCount() + " nodes, " + bp.size()
                + " bits, treap height " + bp.height() + ", " + truncated + " no-op steps");
        check(bp.height() <= 40, "the height stays far below the number of bits");

        // the bitvector layer at scale: many chunks, rank/select/bit still exact
        DynamicBp wide = new DynamicBp();
        int[] expect = new int[1500];
        for (int i = 0; i < expect.length; i++) {
            expect[i] = (nextLong() >>> 33) % 3 == 0 ? 1 : 0;
            wide.insertBit(i + 1, expect[i]);
        }
        int running = 0;
        for (int p = 0; p <= expect.length; p++) {
            check(wide.rank1(p) == running, "wide rank1 at " + p);
            if (p < expect.length) {
                check(wide.bitAt(p + 1) == expect[p], "wide bitAt at " + (p + 1));
                running += expect[p];
            }
        }
        for (int k = 1; k <= wide.ones(); k++) check(wide.bitAt(wide.select1(k)) == 1, "wide select1");
        check(wide.select1(0) == -1 && wide.select1(wide.ones() + 1) == -1, "wide select1 bounds");
        System.out.println("wide bitvector: " + wide.size() + " bits, " + wide.ones()
                + " ones, height " + wide.height());
        check(wide.height() <= 40, "the height stays far below the number of chunks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
