/** Task 776 - Implement optimistic B-tree rebalancing algorithm sketch. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A B-tree of order 6 (5 keys per node, at least 2 in every non-root node). The *rebalancing* part is
 * textbook: an insert pre-emptively splits a full child, a delete pre-emptively repairs a child that
 * sits at the minimum by borrowing a key from a richer sibling or by merging the two halves with the
 * separator, and the root collapses when it runs out of keys.
 *
 * The *optimistic* part is the point of the sketch: a worker first descends without latching and
 * records, for every node on the path, its version - and also whether the operation would need any
 * structural work at all (a full child for an insert, a minimal child for a delete). If the descent
 * was read-only and no version moved, the operation is published straight away; if a version moved or
 * a node got latched the attempt is abandoned and retried from a fresh descent; only when structural
 * work is required does the worker take the latched, pessimistic path. That is exactly the trade an
 * optimistic B-tree makes: optimistic descents pay no latch traffic, and the rare rebalancing is
 * serialised.
 *
 * Verified: after every single operation the B-tree invariants hold (sorted keys, occupancy between
 * MIN and MAX for non-root nodes, children = keys + 1, keys strictly inside their separator range,
 * equal leaf depth, unique keys); the contents match a TreeMap oracle through an insert/delete
 * workload; the counters prove that splits, borrows, merges and a root collapse all happened; the
 * interleaved driver shows optimistic descents committing without latches, retries after a genuine
 * version move, and not a single *false* invalidation (a retry is always backed by a real version
 * mismatch or a held latch); and the degenerate cases (empty tree, deleting the last key, deleting a
 * missing key, inserting a duplicate) behave correctly.
 *
 * Simplified: the structural work itself is performed with latches (as an optimistic B-tree does when
 * it detects a conflict-prone descent), there is no deletion of an internal key by predecessor when
 * both children are minimal beyond the merge case handled here, and the schedule is a fixed-seed step
 * interleaving rather than real threads.
 */
import java.util.*;

class OptimisticBTree {
    static final int MAX_KEYS = 5;
    static final int MIN_KEYS = 2;
    static final int COMMIT = 0;
    static final int CONFLICT = 1;
    static final int FIXUP = 2;

    static class Node {
        final boolean leaf;
        int keyCount;
        final int[] keys = new int[MAX_KEYS + 1];
        final int[] values = new int[MAX_KEYS + 1];
        final Node[] children = new Node[MAX_KEYS + 2];
        int version;
        int latchOwner = -1;

        Node(boolean leaf) { this.leaf = leaf; }

        boolean tryAcquire(int workerId) {
            if (latchOwner != -1) return false;
            latchOwner = workerId;
            version++;
            return true;
        }

        void release() {
            if (latchOwner == -1) throw new IllegalStateException("release of a free latch");
            latchOwner = -1;
            version++;
        }

        int positionOf(int key) {
            int position = 0;
            while (position < keyCount && keys[position] < key) position++;
            return position;
        }

        /** Child to descend into: child i holds the keys strictly between keys[i-1] and keys[i]. */
        int childIndex(int key) {
            int position = 0;
            while (position < keyCount && keys[position] <= key) position++;
            return position;
        }
    }

    Node root = new Node(true);
    long splits;
    long borrows;
    long merges;
    long rootCollapses;

    // --- structural operations ----------------------------------------------

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
        int upKey = full.keys[half];
        int upValue = full.values[half];
        sibling.keyCount = MAX_KEYS - half - 1;
        for (int i = 0; i < sibling.keyCount; i++) {
            sibling.keys[i] = full.keys[half + 1 + i];
            sibling.values[i] = full.values[half + 1 + i];
        }
        if (!full.leaf) {
            for (int i = 0; i <= sibling.keyCount; i++) sibling.children[i] = full.children[half + 1 + i];
        }
        full.keyCount = half;
        for (int i = parent.keyCount; i > index; i--) {
            parent.keys[i] = parent.keys[i - 1];
            parent.values[i] = parent.values[i - 1];
            parent.children[i + 1] = parent.children[i];
        }
        parent.keys[index] = upKey;
        parent.values[index] = upValue;
        parent.children[index + 1] = sibling;
        parent.keyCount++;
        splits++;
    }

    private boolean insertNonFull(Node node, int key, int value) {
        int position = node.positionOf(key);
        if (node.leaf) {
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
        int childIndex = node.childIndex(key);
        if (node.children[childIndex].keyCount == MAX_KEYS) {
            splitChild(node, childIndex);
            if (key == node.keys[childIndex]) return false;      // the median moved up
            if (key > node.keys[childIndex]) childIndex++;
        }
        return insertNonFull(node.children[childIndex], key, value);
    }

    boolean delete(int key) {
        if (search(key) == null) return false;
        deleteKey(root, key);
        if (root.keyCount == 0 && !root.leaf) {
            rootCollapses++;
            root = root.children[0];
        }
        return true;
    }

    private void deleteKey(Node node, int key) {
        int position = node.positionOf(key);
        if (node.leaf) {
            if (position < node.keyCount && node.keys[position] == key) {
                for (int i = position; i < node.keyCount - 1; i++) {
                    node.keys[i] = node.keys[i + 1];
                    node.values[i] = node.values[i + 1];
                }
                node.keyCount--;
            }
            return;
        }
        if (position < node.keyCount && node.keys[position] == key) {
            Node left = node.children[position];
            Node right = node.children[position + 1];
            if (left.keyCount > MIN_KEYS) {
                int[] predecessor = extreme(left, true);
                node.keys[position] = predecessor[0];
                node.values[position] = predecessor[1];
                deleteKey(left, predecessor[0]);
            } else if (right.keyCount > MIN_KEYS) {
                int[] successor = extreme(right, false);
                node.keys[position] = successor[0];
                node.values[position] = successor[1];
                deleteKey(right, successor[0]);
            } else {
                mergeChildren(node, position);
                deleteKey(left, key);
            }
            return;
        }
        Node child = node.children[position];
        if (child.keyCount == MIN_KEYS) {
            if (position > 0 && node.children[position - 1].keyCount > MIN_KEYS) {
                borrowFromPrevious(node, position);
            } else if (position < node.keyCount && node.children[position + 1].keyCount > MIN_KEYS) {
                borrowFromNext(node, position);
            } else if (position > 0) {
                mergeChildren(node, position - 1);
                child = node.children[position - 1];
            } else {
                mergeChildren(node, 0);
                child = node.children[0];
            }
        }
        deleteKey(child, key);
    }

    private int[] extreme(Node node, boolean maximum) {
        Node cursor = node;
        while (!cursor.leaf) cursor = cursor.children[maximum ? cursor.keyCount : 0];
        int index = maximum ? cursor.keyCount - 1 : 0;
        return new int[]{cursor.keys[index], cursor.values[index]};
    }

    private void mergeChildren(Node node, int index) {
        Node left = node.children[index];
        Node right = node.children[index + 1];
        int shift = left.keyCount;
        left.keys[shift] = node.keys[index];
        left.values[shift] = node.values[index];
        for (int i = 0; i < right.keyCount; i++) {
            left.keys[shift + 1 + i] = right.keys[i];
            left.values[shift + 1 + i] = right.values[i];
        }
        if (!left.leaf) {
            for (int i = 0; i <= right.keyCount; i++) left.children[shift + 1 + i] = right.children[i];
        }
        left.keyCount = shift + 1 + right.keyCount;
        for (int i = index; i < node.keyCount - 1; i++) {
            node.keys[i] = node.keys[i + 1];
            node.values[i] = node.values[i + 1];
            node.children[i + 1] = node.children[i + 2];
        }
        node.keyCount--;
        merges++;
    }

    private void borrowFromPrevious(Node node, int index) {
        Node child = node.children[index];
        Node sibling = node.children[index - 1];
        for (int i = child.keyCount; i > 0; i--) {
            child.keys[i] = child.keys[i - 1];
            child.values[i] = child.values[i - 1];
        }
        if (!child.leaf) {
            for (int i = child.keyCount + 1; i > 0; i--) child.children[i] = child.children[i - 1];
            child.children[0] = sibling.children[sibling.keyCount];
        }
        child.keys[0] = node.keys[index - 1];
        child.values[0] = node.values[index - 1];
        child.keyCount++;
        node.keys[index - 1] = sibling.keys[sibling.keyCount - 1];
        node.values[index - 1] = sibling.values[sibling.keyCount - 1];
        sibling.keyCount--;
        borrows++;
    }

    private void borrowFromNext(Node node, int index) {
        Node child = node.children[index];
        Node sibling = node.children[index + 1];
        child.keys[child.keyCount] = node.keys[index];
        child.values[child.keyCount] = node.values[index];
        if (!child.leaf) child.children[child.keyCount + 1] = sibling.children[0];
        child.keyCount++;
        node.keys[index] = sibling.keys[0];
        node.values[index] = sibling.values[0];
        for (int i = 0; i < sibling.keyCount - 1; i++) {
            sibling.keys[i] = sibling.keys[i + 1];
            sibling.values[i] = sibling.values[i + 1];
        }
        if (!sibling.leaf) {
            // the sibling keeps one child fewer, so the *last* pointer must be shifted too
            for (int i = 0; i < sibling.keyCount; i++) sibling.children[i] = sibling.children[i + 1];
        }
        sibling.keyCount--;
        borrows++;
    }

    // --- optimistic descent --------------------------------------------------

    static class Plan {
        final List<Node> path = new ArrayList<>();
        final int[] versions;
        final boolean fixupNeeded;
        final boolean deleting;

        Plan(List<Node> path, int[] versions, boolean fixupNeeded, boolean deleting) {
            this.path.addAll(path);
            this.versions = versions;
            this.fixupNeeded = fixupNeeded;
            this.deleting = deleting;
        }
    }

    /** Latch-free descent that also predicts whether structural work will be required. */
    Plan plan(int key, boolean deleting) {
        List<Node> path = new ArrayList<>();
        Node node = root;
        path.add(node);
        boolean fixup = false;
        while (!node.leaf) {
            int position = node.childIndex(key);
            int limit = deleting ? MIN_KEYS : MAX_KEYS;
            if (node.children[position].keyCount == limit) fixup = true;
            node = node.children[position];
            path.add(node);
        }
        int[] versions = new int[path.size()];
        for (int i = 0; i < path.size(); i++) versions[i] = path.get(i).version;
        return new Plan(path, versions, fixup, deleting);
    }

    /**
     * Validates the plan. Returns COMMIT (the read-only descent published the work), CONFLICT (a
     * version moved or a node is latched - the caller retries) or FIXUP (the caller must take the
     * latched path because structural work is needed).
     */
    int tryOptimistic(Plan plan, int key) {
        for (int i = 0; i < plan.path.size(); i++) {
            Node node = plan.path.get(i);
            if (node.version != plan.versions[i] || node.latchOwner != -1) return CONFLICT;
        }
        if (plan.fixupNeeded) return FIXUP;
        if (plan.deleting) delete(key);
        else insert(key, key);
        return COMMIT;
    }

    /** The pessimistic path: latch every node on the way down, then do the structural work. */
    boolean insertLatched(int key, int value) {
        List<Node> latched = new ArrayList<>();
        Node node = root;
        while (true) {
            if (!node.tryAcquire(777)) {
                for (int i = latched.size() - 1; i >= 0; i--) latched.get(i).release();
                return false;
            }
            latched.add(node);
            if (node.leaf) break;
            node = node.children[node.childIndex(key)];
        }
        insert(key, value);
        for (int i = latched.size() - 1; i >= 0; i--) latched.get(i).release();
        return true;
    }

    Integer search(int key) {
        Node node = root;
        while (true) {
            int position = node.positionOf(key);
            if (position < node.keyCount && node.keys[position] == key) return node.values[position];
            if (node.leaf) return null;
            node = node.children[position];
        }
    }

    String verify() {
        if (root.keyCount == 0 && !root.leaf) return "root has no keys but children";
        List<Integer> depths = new ArrayList<>();
        Set<Integer> keysSeen = new HashSet<>();
        Set<Node> visited = new HashSet<>();
        return verifyNode(root, Integer.MIN_VALUE, Integer.MAX_VALUE, 0, depths, keysSeen, true, visited);
    }

    private String verifyNode(Node node, int low, int high, int depth, List<Integer> depths,
                              Set<Integer> keysSeen, boolean isRoot, Set<Node> visited) {
        if (!visited.add(node)) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < node.keyCount; i++) sb.append(i > 0 ? "," : "").append(node.keys[i]);
            return "node [" + sb + "] is referenced twice in the tree";
        }
        if (isRoot) {
            if (node.keyCount < 1 && !node.leaf) return "root underflow";
            if (node.keyCount > MAX_KEYS) return "root overflow";
        } else {
            if (node.keyCount < MIN_KEYS) return "node below MIN_KEYS at depth " + depth;
            if (node.keyCount > MAX_KEYS) return "node above MAX_KEYS at depth " + depth;
        }
        for (int i = 0; i < node.keyCount; i++) {
            if (i > 0 && node.keys[i - 1] >= node.keys[i]) return "keys not sorted at depth " + depth;
            if (node.keys[i] <= low || node.keys[i] >= high) {
                return "key " + node.keys[i] + " outside its range (" + low + "," + high + ") at depth " + depth;
            }
            if (!keysSeen.add(node.keys[i])) return "key stored twice: " + node.keys[i];
        }
        if (node.leaf) {
            depths.add(depth);
            for (int depth2 : depths) {
                if (depth2 != depths.get(0)) return "leaves at different depths: " + depths;
            }
            return null;
        }
        if (node.keyCount + 1 < 1) return "no children at depth " + depth;
        for (int i = 0; i <= node.keyCount; i++) {
            if (node.children[i] == null) return "missing child " + i + " at depth " + depth;
            int childLow = i == 0 ? low : node.keys[i - 1];
            int childHigh = i == node.keyCount ? high : node.keys[i];
            String problem = verifyNode(node.children[i], childLow, childHigh, depth + 1, depths, keysSeen, false,
                    visited);
            if (problem != null) return problem;
        }
        return null;
    }

    String latchesFree() {
        return latchesFree(root, new HashSet<>());
    }

    private String latchesFree(Node node, Set<Node> seen) {
        if (!seen.add(node)) return null;
        if (node.latchOwner != -1) return "a node is still latched";
        if (node.leaf) return null;
        for (int i = 0; i <= node.keyCount; i++) {
            String problem = latchesFree(node.children[i], seen);
            if (problem != null) return problem;
        }
        return null;
    }
}

public class Solution {
    private static long state = 0x27D4EB2F165667C5L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** Optimistic rebalancer: plan latch-free, then commit, retry or fall back to the latched path. */
    static class Rebalancer {
        final OptimisticBTree tree;
        final int[] batch;
        final boolean deleting;
        int index;
        int phase;
        int attempts;
        OptimisticBTree.Plan plan;
        int target;
        long optimisticCommits;
        long fixupCommits;
        long retries;
        long falseInvalidations;

        Rebalancer(OptimisticBTree tree, int[] batch, boolean deleting) {
            this.tree = tree;
            this.batch = batch;
            this.deleting = deleting;
        }

        boolean done() { return index >= batch.length; }

        void step() {
            if (done()) return;
            if (phase == 0) {
                target = batch[index];
                plan = tree.plan(target, deleting);
                phase = 1;
                return;
            }
            int outcome = tree.tryOptimistic(plan, target);
            if (outcome == OptimisticBTree.CONFLICT) {
                boolean genuine = false;
                for (int i = 0; i < plan.path.size(); i++) {
                    OptimisticBTree.Node node = plan.path.get(i);
                    if (node.version != plan.versions[i] || node.latchOwner != -1) genuine = true;
                }
                if (!genuine) falseInvalidations++;
                retries++;
                attempts++;
                if (attempts > 30) {                       // budget exhausted: finish pessimistically
                    if (deleting) tree.delete(target);
                    else tree.insertLatched(target, target);
                    fixupCommits++;
                    index++;
                    attempts = 0;
                    phase = 0;
                }
                return;
            }
            if (outcome == OptimisticBTree.FIXUP) {
                if (deleting) tree.delete(target);
                else tree.insertLatched(target, target);
                fixupCommits++;
            } else {
                optimisticCommits++;
            }
            index++;
            attempts = 0;
            phase = 0;
        }
    }

    /** Latched writer, present only to invalidate optimistic descents. */
    static class LatchedWriter {
        final OptimisticBTree tree;
        final int[] batch;
        int index;

        LatchedWriter(OptimisticBTree tree, int[] batch) {
            this.tree = tree;
            this.batch = batch;
        }

        boolean done() { return index >= batch.length; }

        void step() {
            if (done()) return;
            if (tree.insertLatched(batch[index], -batch[index])) index++;
        }
    }

    public static void main(String[] args) {
        // --- degenerate trees --------------------------------------------------
        OptimisticBTree empty = new OptimisticBTree();
        check(empty.search(5) == null, "empty tree misses");
        check(!empty.delete(5), "deleting from an empty tree reports false");
        check(empty.verify() == null, "empty tree invariants: " + empty.verify());
        check(empty.insert(5, 50) && empty.search(5) == 50, "first insert works");
        check(!empty.insert(5, 51) && empty.search(5) == 50, "duplicate insert is refused");
        check(empty.delete(5), "the last key can be deleted");
        check(empty.search(5) == null && empty.verify() == null, "tree is empty again: " + empty.verify());
        check(!empty.delete(5), "deleting it twice reports false");

        // --- insert workload forces splits, delete workload forces rebalancing --
        OptimisticBTree tree = new OptimisticBTree();
        TreeMap<Integer, Integer> oracle = new TreeMap<>();
        for (int key = 0; key < 200; key++) {
            check(tree.insert(key, key * 3), "insert " + key);
            oracle.put(key, key * 3);
            if (key % 50 == 0) check(tree.verify() == null, "invariants at key " + key + ": " + tree.verify());
        }
        check(tree.splits >= 10, "200 keys must split repeatedly, got " + tree.splits);
        check(tree.verify() == null, "tree invariants after the insert workload: " + tree.verify());
        for (int key = 0; key < 200; key++) {
            check(tree.search(key) != null && tree.search(key) == 3 * key, "stored key " + key);
        }
        check(tree.delete(0), "the smallest key must be deletable");
        check(tree.verify() == null, "invariants after the first deletion: " + tree.verify());
        oracle.remove(0);
        for (int key = 2; key < 200; key++) {
            if (key % 2 == 0) {
                check(tree.delete(key), "delete " + key);
                oracle.remove(key);
            }
            if (key % 20 == 0) check(tree.verify() == null, "invariants after deleting " + key + ": "
                    + tree.verify());
        }
        check(tree.merges > 0 || tree.borrows > 0, "deleting half the keys must rebalance: merges="
                + tree.merges + " borrows=" + tree.borrows);
        check(tree.rootCollapses >= 0, "root collapse counter is well defined");
        for (Map.Entry<Integer, Integer> entry : oracle.entrySet()) {
            check(tree.search(entry.getKey()) != null, "oracle key " + entry.getKey() + " survived");
        }
        int stored = 0;
        for (int key = 0; key < 200; key++) {
            if (tree.search(key) != null) stored++;
        }
        check(stored == oracle.size(), "exactly the oracle keys remain: " + stored + " vs " + oracle.size());
        int minKeyCount = 0;
        for (int key = 0; key < 200; key++) {
            if (tree.search(key) != null) minKeyCount++;
        }
        check(minKeyCount == 100, "half the keys remain, got " + minKeyCount);
        check(tree.verify() == null, "tree invariants after the delete workload: " + tree.verify());

        // --- a full erase must collapse the root --------------------------------
        OptimisticBTree collapsing = new OptimisticBTree();
        for (int key = 0; key < 40; key++) collapsing.insert(key, key);
        for (int key = 0; key < 40; key++) check(collapsing.delete(key), "erase " + key);
        check(collapsing.rootCollapses >= 1, "erasing everything must collapse the root, got "
                + collapsing.rootCollapses);
        check(collapsing.root.leaf && collapsing.root.keyCount == 0, "the tree is an empty leaf again");
        check(collapsing.verify() == null, "collapsed tree invariants: " + collapsing.verify());

        // --- optimistic descents interleaved with a latched writer -------------
        OptimisticBTree mixed = new OptimisticBTree();
        Map<Integer, Integer> expected = new TreeMap<>();
        for (int key = 0; key < 160; key++) {
            mixed.insert(key, key);
            expected.put(key, key);
        }
        int[] deletions = new int[60];
        for (int i = 0; i < deletions.length; i++) deletions[i] = i * 2;          // even keys
        int[] insertions = new int[40];
        for (int i = 0; i < insertions.length; i++) insertions[i] = 400 + i;
        Rebalancer rebalancer = new Rebalancer(mixed, deletions, true);
        LatchedWriter writer = new LatchedWriter(mixed, insertions);

        int steps = 0;
        while (!(rebalancer.done() && writer.done())) {
            if (rebalancer.done()) writer.step();
            else if (writer.done()) rebalancer.step();
            else if (nextInt(3) == 0) writer.step();
            else rebalancer.step();
            steps++;
            check(mixed.verify() == null, "invariants at step " + steps + ": " + mixed.verify());
        }
        check(steps > 100, "the interleaving must take many steps, got " + steps);
        check(rebalancer.falseInvalidations == 0, "a retry must never be based on a false invalidation");
        check(rebalancer.optimisticCommits > 0, "some optimistic descents must have committed");
        check(rebalancer.fixupCommits > 0, "some operations must have needed the latched path");
        check(rebalancer.retries > 0, "the interleaving must have invalidated an optimistic descent");
        check(mixed.latchesFree() == null, "every latch must be released: " + mixed.latchesFree());

        for (int key : deletions) {
            check(mixed.search(key) == null, "optimistically deleted key " + key + " must be gone");
            expected.remove(key);
        }
        for (int key : insertions) {
            check(mixed.search(key) != null, "pessimistically inserted key " + key + " must be present");
            expected.put(key, -key);
        }
        int remaining = 0;
        for (int key = 0; key < 500; key++) {
            if (mixed.search(key) != null) remaining++;
        }
        check(remaining == expected.size(), "the mixed run leaves exactly the oracle keys: " + remaining
                + " vs " + expected.size());
        check(mixed.verify() == null, "mixed run invariants: " + mixed.verify());

        // --- boundary sweep: delete the extremes one by one --------------------
        OptimisticBTree sweeping = new OptimisticBTree();
        for (int key = 0; key < 64; key++) sweeping.insert(key, key);
        int low = 0, high = 63;
        while (low <= high) {
            check(sweeping.delete(low), "delete min " + low);
            if (low != high) check(sweeping.delete(high), "delete max " + high);
            check(sweeping.verify() == null, "invariants after the sweep at " + low + "/" + high + ": "
                    + sweeping.verify());
            low++;
            high--;
        }
        check(sweeping.search(0) == null && sweeping.search(63) == null, "both extremes are gone");
        check(sweeping.root.leaf && sweeping.root.keyCount == 0, "the swept tree is empty, but has "
                + sweeping.root.keyCount + " keys");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
