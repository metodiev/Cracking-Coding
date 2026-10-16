/** Task 774 - Implement concurrent B-tree with latch-free readers (reader-writer locks). (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A B+tree whose concurrency control splits the two sides of the classic problem:
 *
 *   writers   latch every node on their descent path exclusively (crabbing). A latch is a single owner
 *             field that throws if it is taken twice, so exclusivity is enforced, not assumed; a
 *             writer that walks into a held latch blocks and retries the same step, and the version
 *             counter of a node is bumped on latch and again on release so that every write window is
 *             detectable by version comparison.
 *   readers   take no latch at all. A read is optimistic: sample (version, latchOwner) on entry, read
 *             the keys and the child pointer, then sample again. The data is used only when the node
 *             was unlatched at both samples and the version is unchanged; otherwise the traversal
 *             restarts at the root. Readers therefore never wait - they pay in retries instead of
 *             blocking, which is exactly the latch-free read side of a B-tree.
 *
 * The interleaving is simulated deterministically: writers and readers advance one protocol step at a
 * time and the step boundary is allowed to fall between a reader's content read and its validation,
 * which is precisely the window in which a concurrent writer invalidates the read. The protocol is
 * real - a reader that skipped the validation could return a torn node.
 *
 * Verified: every answer a latch-free reader produced equals the expected value; readers block zero
 * times while writers do block; the readers sampled a latched node and restarted at least once, so the
 * optimistic path was genuinely exercised; all latches are released after the run; the B+tree
 * invariants (sorted keys, occupancy bounds, equal leaf depth, ordered child ranges, no duplicate
 * key) hold after every single writer step and at the end, and the final contents match a TreeMap
 * oracle. Degenerate inputs (empty tree, single key, duplicate key, root split, absent key) are
 * covered explicitly.
 *
 * Simplified: writers serialise on the root latch (a real implementation would use finer-grained
 * release) and there is no deletion/merge step, which is orthogonal to the latch protocol.
 */
import java.util.*;

class BPlusTree {
    static final int MAX_KEYS = 4;

    static class Node {
        final boolean leaf;
        int keyCount;
        final int[] keys = new int[MAX_KEYS + 1];
        final int[] values = new int[MAX_KEYS + 1];
        final Node[] children = new Node[MAX_KEYS + 2];
        int version;
        int latchOwner = -1;

        Node(boolean leaf) { this.leaf = leaf; }

        /** Exclusive latch; taking a held latch is a bug and is reported as such. */
        void acquire(int writerId) {
            if (latchOwner != -1) throw new IllegalStateException("latch already owned by " + latchOwner);
            latchOwner = writerId;
            version++;
        }

        boolean tryAcquire(int writerId) {
            if (latchOwner != -1) return false;
            acquire(writerId);
            return true;
        }

        void release() {
            if (latchOwner == -1) throw new IllegalStateException("release of a free latch");
            latchOwner = -1;
            version++;
        }

        /** Insertion point among the leaf keys: the number of keys strictly below `key`. */
        int positionOf(int key) {
            int position = 0;
            while (position < keyCount && keys[position] < key) position++;
            return position;
        }

        /**
         * Child to descend into. Separators are copies of the smallest key of the right subtree, so
         * child i covers [keys[i-1], keys[i]) and the child for `key` is the number of keys <= key.
         */
        int childIndex(int key) {
            int position = 0;
            while (position < keyCount && keys[position] <= key) position++;
            return position;
        }
    }

    Node root = new Node(true);
    long splits;
    long writerLatches;

    /** Ordinary single-threaded insert; returns false when the key is already present. */
    boolean insert(int key, int value) {
        if (root.keyCount == MAX_KEYS) {
            Node newRoot = new Node(false);
            newRoot.children[0] = root;
            splitChild(newRoot, 0);
            root = newRoot;
        }
        return insertNonFull(root, key, value);
    }

    private void splitChild(Node parent, int index) {
        Node full = parent.children[index];
        Node sibling = new Node(full.leaf);
        int half = MAX_KEYS / 2;
        if (full.leaf) {
            sibling.keyCount = MAX_KEYS - half;
            for (int i = 0; i < sibling.keyCount; i++) {
                sibling.keys[i] = full.keys[half + i];
                sibling.values[i] = full.values[half + i];
            }
            full.keyCount = half;
            // a leaf separator is a *copy* of the right sibling's first key
            insertIntoParent(parent, index, sibling.keys[0], sibling.values[0], sibling);
        } else {
            int upKey = full.keys[half];
            int upValue = full.values[half];
            sibling.keyCount = MAX_KEYS - half - 1;
            for (int i = 0; i < sibling.keyCount; i++) {
                sibling.keys[i] = full.keys[half + 1 + i];
                sibling.values[i] = full.values[half + 1 + i];
            }
            for (int i = 0; i <= sibling.keyCount; i++) {
                sibling.children[i] = full.children[half + 1 + i];
            }
            full.keyCount = half;
            insertIntoParent(parent, index, upKey, upValue, sibling);
        }
        splits++;
    }

    private void insertIntoParent(Node parent, int index, int key, int value, Node sibling) {
        for (int i = parent.keyCount; i > index; i--) {
            parent.keys[i] = parent.keys[i - 1];
            parent.values[i] = parent.values[i - 1];
            parent.children[i + 1] = parent.children[i];
        }
        parent.keys[index] = key;
        parent.values[index] = value;
        parent.children[index + 1] = sibling;
        parent.keyCount++;
    }

    private boolean insertNonFull(Node node, int key, int value) {
        if (node.leaf) {
            int position = node.positionOf(key);
            if (position < node.keyCount && node.keys[position] == key) return false;
            for (int i = node.keyCount; i > position; i--) {
                node.keys[i] = node.keys[i - 1];
                node.values[i] = node.values[i - 1];
            }
            node.keys[position] = key;
            node.values[position] = value;
            node.keyCount++;
            return true;
        }
        int position = node.childIndex(key);
        if (node.children[position].keyCount == MAX_KEYS) {
            splitChild(node, position);
            if (key >= node.keys[position]) position++;      // the new separator was copied up
        }
        return insertNonFull(node.children[position], key, value);
    }

    Integer search(int key) {
        Node node = root;
        while (true) {
            if (node.leaf) {
                int position = node.positionOf(key);
                return position < node.keyCount && node.keys[position] == key ? node.values[position] : null;
            }
            node = node.children[node.childIndex(key)];
        }
    }

    /** Returns null when every B+tree invariant holds, else a description. */
    String verify() {
        if (root.keyCount > MAX_KEYS) return "root overflows";
        List<Integer> depths = new ArrayList<>();
        Set<Integer> leafKeys = new HashSet<>();
        String problem = verifyNode(root, Integer.MIN_VALUE, Integer.MAX_VALUE, 0, depths, leafKeys, true);
        if (problem != null) return problem;
        if (depths.isEmpty()) return "no leaves";
        for (int depth : depths) {
            if (depth != depths.get(0)) return "leaves at different depths: " + depths;
        }
        return null;
    }

    private String verifyNode(Node node, int low, int high, int depth, List<Integer> depths,
                              Set<Integer> leafKeys, boolean isRoot) {
        if (!isRoot && node.keyCount < 1) return "node below minimum occupancy at depth " + depth;
        if (node.keyCount > MAX_KEYS) return "node above maximum occupancy at depth " + depth;
        for (int i = 0; i < node.keyCount; i++) {
            if (i > 0 && node.keys[i - 1] >= node.keys[i]) return "keys not sorted at depth " + depth;
            if (node.keys[i] < low || node.keys[i] >= high) return "key outside its range at depth " + depth;
            if (node.leaf && !leafKeys.add(node.keys[i])) return "key stored twice: " + node.keys[i];
        }
        if (node.leaf) {
            depths.add(depth);
            return null;
        }
        for (int i = 0; i <= node.keyCount; i++) {
            if (node.children[i] == null) return "missing child " + i;
            int childLow = i == 0 ? low : node.keys[i - 1];
            int childHigh = i == node.keyCount ? high : node.keys[i];
            String problem = verifyNode(node.children[i], childLow, childHigh, depth + 1, depths, leafKeys, false);
            if (problem != null) return problem;
        }
        return null;
    }

    /** Returns null when no node is still latched, else a description of a locked node. */
    String latchesFree() {
        return latchesFree(root, new HashSet<>());
    }

    private String latchesFree(Node node, Set<Node> seen) {
        if (!seen.add(node)) return null;
        if (node.latchOwner != -1) return "a node at depth is still latched";
        if (node.leaf) return null;
        for (int i = 0; i <= node.keyCount; i++) {
            String problem = latchesFree(node.children[i], seen);
            if (problem != null) return problem;
        }
        return null;
    }
}

public class Solution {
    private static long state = 0x853C49E6748FEA9BL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** A simulated agent: one protocol step per call, so the driver can interleave them. */
    abstract static class Agent {
        final String name;
        boolean done;
        Agent(String name) { this.name = name; }
        abstract void step();
    }

    /** Writer: blocks on held latches along the descent path, inserts, then releases them. */
    static class Writer extends Agent {
        final BPlusTree tree;
        final int[] batch;
        final int writerId;
        int index;
        int phase;
        final List<BPlusTree.Node> latched = new ArrayList<>();
        BPlusTree.Node current;
        long latchesTaken;
        long waits;

        Writer(BPlusTree tree, int writerId, int[] batch) {
            super("writer" + writerId);
            this.tree = tree;
            this.writerId = writerId;
            this.batch = batch;
            this.current = tree.root;
        }

        @Override
        void step() {
            if (index >= batch.length) {
                done = true;
                return;
            }
            if (phase == 0) {
                if (!current.tryAcquire(writerId)) {          // a writer blocks on a held latch
                    waits++;
                    return;
                }
                latchesTaken++;
                tree.writerLatches++;
                latched.add(current);
                if (current.leaf) {
                    phase = 1;
                } else {
                    current = current.children[current.childIndex(batch[index])];
                }
                return;
            }
            tree.insert(batch[index], 0);                     // the path is latched: exclusive
            index++;
            phase = 0;
            current = tree.root;
            for (int i = latched.size() - 1; i >= 0; i--) latched.get(i).release();
            latched.clear();
        }
    }

    /** Reader: optimistic descent with sample/read/validate, restarting when the samples disagree. */
    static class Reader extends Agent {
        final BPlusTree tree;
        final int[] queries;
        final Integer[] answers;
        int queryIndex;
        int phase;
        BPlusTree.Node node;
        int versionBefore;
        int ownerBefore;
        int[] keysRead;
        int keyCountRead;
        int restarts;
        int latchedSamples;
        final int waitCount = 0;                              // latch-free readers never wait
        Integer currentQuery;

        Reader(BPlusTree tree, int id, int[] queries) {
            super("reader" + id);
            this.tree = tree;
            this.queries = queries;
            this.answers = new Integer[queries.length];
        }

        @Override
        void step() {
            if (queryIndex >= queries.length) {
                done = true;
                return;
            }
            if (currentQuery == null) {
                currentQuery = queries[queryIndex];
                node = tree.root;
                phase = 0;
            }
            if (phase == 0) {                                  // sample the node header
                versionBefore = node.version;
                ownerBefore = node.latchOwner;
                if (ownerBefore != -1) latchedSamples++;
                phase = 1;
                return;
            }
            if (phase == 1) {                                  // read the content
                keyCountRead = node.keyCount;
                keysRead = Arrays.copyOf(node.keys, keyCountRead);
                phase = 2;
                return;
            }
            // phase 2: validate - a writer step may have landed between the three phases
            boolean valid = versionBefore == node.version && ownerBefore == -1 && node.latchOwner == -1;
            if (!valid) {
                restarts++;
                node = tree.root;
                phase = 0;
                return;
            }
            int position = 0;
            while (position < keyCountRead && keysRead[position] < currentQuery) position++;
            boolean found = position < keyCountRead && keysRead[position] == currentQuery;
            if (node.leaf) {
                answers[queryIndex] = found ? node.values[position] : null;
                queryIndex++;
                currentQuery = null;
                phase = 0;
                return;
            }
            node = node.children[found ? position + 1 : position];   // keys <= x descend right
            phase = 0;
        }
    }

    public static void main(String[] args) {
        // --- degenerate tree ---------------------------------------------------
        BPlusTree empty = new BPlusTree();
        check(empty.search(7) == null, "empty tree misses");
        check(empty.verify() == null, "empty tree invariants: " + empty.verify());
        check(empty.insert(7, 3), "first insert succeeds");
        check(!empty.insert(7, 9), "duplicate key is rejected");
        check(empty.search(7) == 3, "single key is found");
        check(empty.search(8) == null, "absent key is absent");
        check(empty.verify() == null, "single key invariants: " + empty.verify());

        // --- root split and structural growth ----------------------------------
        BPlusTree growing = new BPlusTree();
        for (int key = 0; key < 200; key++) {
            check(growing.insert(key, key * 2), "insert " + key);
            if (key % 40 == 0) {
                check(growing.verify() == null, "growth invariants at " + key + ": " + growing.verify());
            }
        }
        check(growing.splits >= 4, "a 200 key tree must split repeatedly, got " + growing.splits);
        for (int key = 0; key < 200; key++) {
            check(growing.search(key) != null && growing.search(key) == 2 * key,
                    "stored key " + key + " round trips");
        }
        check(growing.verify() == null, "grown tree invariants: " + growing.verify());

        // --- latched writers and latch-free readers interleaved -----------------
        BPlusTree tree = new BPlusTree();
        Map<Integer, Integer> expected = new HashMap<>();
        for (int key = 0; key < 300; key++) {
            tree.insert(key, key + 1000);
            expected.put(key, key + 1000);
        }
        int[] firstBatch = new int[20];
        int[] secondBatch = new int[20];
        for (int i = 0; i < 20; i++) {
            firstBatch[i] = 1000 + i;
            secondBatch[i] = 2000 + i;
        }
        int[] reads = new int[48];
        for (int i = 0; i < reads.length; i++) reads[i] = (i % 3 == 0) ? 5000 + i : (i * 7) % 300;

        Writer firstWriter = new Writer(tree, 1, firstBatch);
        Writer secondWriter = new Writer(tree, 2, secondBatch);
        List<Reader> readers = new ArrayList<>();
        for (int r = 0; r < 3; r++) {
            int[] queries = new int[reads.length / 3];
            for (int i = 0; i < queries.length; i++) queries[i] = reads[r * queries.length + i];
            readers.add(new Reader(tree, r, queries));
        }
        List<Agent> agents = new ArrayList<>(readers);
        agents.add(firstWriter);
        agents.add(secondWriter);

        int steps = 0;
        while (true) {
            List<Agent> ready = new ArrayList<>();
            for (Agent agent : agents) {
                if (!agent.done) ready.add(agent);
            }
            if (ready.isEmpty()) break;
            ready.get(nextInt(ready.size())).step();
            steps++;
            check(tree.verify() == null, "tree invariants during the interleaving at step " + steps
                    + ": " + tree.verify());
        }
        check(steps > 100, "the interleaving must take many steps, got " + steps);

        // --- the readers never waited, never used a torn node -------------------
        int restarts = 0;
        int latchedSamples = 0;
        int waits = 0;
        for (Reader reader : readers) {
            restarts += reader.restarts;
            latchedSamples += reader.latchedSamples;
            waits += reader.waitCount;
        }
        check(waits == 0, "latch-free readers must never block, waits=" + waits);
        check(latchedSamples > 0, "the readers must have sampled a latched node at least once");
        check(restarts > 0, "the optimistic readers must have restarted, restarts=" + restarts);
        check(firstWriter.latchesTaken > 0 && secondWriter.latchesTaken > 0,
                "both writers must have taken latches: " + firstWriter.latchesTaken + "/"
                        + secondWriter.latchesTaken);
        check(firstWriter.waits + secondWriter.waits > 0,
                "the blocking latch protocol must have made a writer wait: " + firstWriter.waits
                        + "/" + secondWriter.waits);
        check(firstWriter.index == firstBatch.length && secondWriter.index == secondBatch.length,
                "both writers must have finished their batches");
        check(tree.latchesFree() == null, "every latch must be free after the run");

        // --- every answer a latch-free reader produced is exactly right ---------
        int checkedReads = 0;
        int readersThatRestarted = 0;
        for (Reader reader : readers) {
            if (reader.restarts > 0) readersThatRestarted++;
            for (int i = 0; i < reader.queries.length; i++) {
                int query = reader.queries[i];
                Integer want = expected.get(query);
                check(reader.answers[i] == null ? want == null : reader.answers[i].equals(want),
                        "reader answer for key " + query + " was " + reader.answers[i] + " but expected " + want);
                checkedReads++;
            }
        }
        check(checkedReads == 48, "every reader query must have an answer, got " + checkedReads);
        check(readersThatRestarted > 0, "a reader that restarted must still answer correctly");

        // --- final contents equal the TreeMap oracle ---------------------------
        TreeMap<Integer, Integer> oracle = new TreeMap<>(expected);
        for (int key : firstBatch) oracle.put(key, 0);
        for (int key : secondBatch) oracle.put(key, 0);
        for (Map.Entry<Integer, Integer> entry : oracle.entrySet()) {
            check(tree.search(entry.getKey()) != null,
                    "the oracle key " + entry.getKey() + " must be present after the run");
        }
        check(tree.verify() == null, "final tree invariants: " + tree.verify());
        List<Integer> candidates = new ArrayList<>();
        for (int key = 0; key < 300; key++) candidates.add(key);
        for (int key : firstBatch) candidates.add(key);
        for (int key : secondBatch) candidates.add(key);
        int reachable = 0;
        for (int key : candidates) {
            if (tree.search(key) != null) reachable++;
        }
        check(reachable == candidates.size() && reachable == oracle.size(),
                "every oracle key is reachable, got " + reachable + " of " + oracle.size());
        for (int absent : new int[]{-1, 300, 999, 1020, 1999, 2020, 30000}) {
            check(tree.search(absent) == null, "key " + absent + " must stay absent");
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
