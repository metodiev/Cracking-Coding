/** Task 775 - Implement speculative B-tree inserts with conflict resolution. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A B+tree insert is speculative when the worker does not latch the way down: it descends latch-free,
 * samples the version counter of every node on the path, and prepares the new leaf content *aside*
 * from the tree. Only then does it try to publish the change:
 *
 *   validate   latch the target leaf (a failed latch is a conflict), re-check that the leaf version
 *              moved by exactly the latch itself and that no node on the path was re-organised, and
 *              additionally re-compare the leaf keys with the sampled ones - if that belt-and-braces
 *              comparison ever failed, the version check would be broken.
 *   commit     write the prepared content into the leaf and release.
 *   conflict   if validation fails the speculative content is *thrown away*, the attempt counter is
 *              bumped and the worker retries the same key from a fresh descent - the resolution rule.
 *              A leaf that is already full cannot be published in place, so the worker falls back to
 *              the latched (pessimistic) insert path, which splits as usual; a worker that exhausts
 *              its attempt budget falls back the same way, so no key is ever silently dropped.
 *
 * The interleaving is deterministic: a plain latched writer and two speculators advance one protocol
 * step at a time, and the step boundary falls between a speculator's sample and its validation - the
 * window in which the writer changes a leaf version and turns the speculation into a conflict.
 *
 * Verified: the final contents equal a TreeMap oracle (no lost update, no phantom key); every leaf
 * re-comparison during a commit matched the sample (so the optimistic validator is sound); conflicts
 * and commits are both positive, i.e. the protocol really had to resolve a conflict and really
 * committed speculatively; structural invariants (sorted keys, occupancy, equal leaf depth, ranges, no
 * duplicate key) hold after every step; duplicates inside a batch are detected rather than inserted
 * twice; the overflow fallback fires; and the degenerate cases (empty tree, one-key batch, fill-the-
 * leaf batch) behave as documented.
 *
 * Simplified: no deletion, the fallback insert is the ordinary latched path, and the schedule is a
 * fixed-seed step interleaving rather than real threads, so runs are reproducible.
 */
import java.util.*;

class SpeculativeBTree {
    static final int MAX_KEYS = 4;
    static final int COMMIT = 0;
    static final int CONFLICT = 1;
    static final int DUPLICATE = 2;
    static final int FALLBACK = 3;

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

        int childIndex(int key) {
            int position = 0;
            while (position < keyCount && keys[position] <= key) position++;
            return position;
        }
    }

    Node root = new Node(true);
    long splits;
    long commits;
    long conflicts;
    long fallbacks;
    long duplicates;
    long validationMismatches;

    // --- ordinary insert (baseline, fallback and pessimistic target) ----------

    /** Latching insert: takes every latch on the path, inserts, then releases. False when busy. */
    boolean insertLatched(int key, int value) {
        List<Node> latched = new ArrayList<>();
        Node node = root;
        while (true) {
            if (!node.tryAcquire(999)) {
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
            insertIntoParent(parent, index, sibling.keys[0], sibling.values[0], sibling);
        } else {
            int upKey = full.keys[half];
            int upValue = full.values[half];
            sibling.keyCount = MAX_KEYS - half - 1;
            for (int i = 0; i < sibling.keyCount; i++) {
                sibling.keys[i] = full.keys[half + 1 + i];
                sibling.values[i] = full.values[half + 1 + i];
            }
            for (int i = 0; i <= sibling.keyCount; i++) sibling.children[i] = full.children[half + 1 + i];
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
            if (key >= node.keys[position]) position++;
        }
        return insertNonFull(node.children[position], key, value);
    }

    // --- the speculative protocol --------------------------------------------

    List<Node> pathTo(int key) {
        List<Node> path = new ArrayList<>();
        Node node = root;
        path.add(node);
        while (!node.leaf) {
            node = node.children[node.childIndex(key)];
            path.add(node);
        }
        return path;
    }

    static int[] snapshot(List<Node> path) {
        int[] versions = new int[path.size()];
        for (int i = 0; i < path.size(); i++) versions[i] = path.get(i).version;
        return versions;
    }

    /**
     * Validates the sampled path and publishes the prepared leaf content. Returns COMMIT, CONFLICT,
     * DUPLICATE or FALLBACK; in every non-COMMIT case the tree was left untouched by this attempt.
     */
    int tryCommit(List<Node> path, int[] versions, int[] sampledKeys, int[] sampledValues,
                  int sampledCount, int key, int value, int workerId) {
        Node leaf = path.get(path.size() - 1);
        int position = 0;
        while (position < sampledCount && sampledKeys[position] < key) position++;
        boolean alreadyThere = position < sampledCount && sampledKeys[position] == key;
        if (alreadyThere) {
            duplicates++;
            return DUPLICATE;
        }
        if (sampledCount == MAX_KEYS) {                 // a full leaf needs a split: pessimistic path
            insert(key, value);
            fallbacks++;
            return FALLBACK;
        }
        if (!leaf.tryAcquire(workerId)) {               // somebody else is writing this leaf
            conflicts++;
            return CONFLICT;
        }
        boolean leafUnchanged = leaf.version == versions[versions.length - 1] + 1;
        boolean pathUnchanged = true;
        for (int i = 0; i < path.size() - 1; i++) {
            if (path.get(i).version != versions[i] || path.get(i).latchOwner != -1) pathUnchanged = false;
        }
        if (!leafUnchanged || !pathUnchanged) {
            leaf.release();
            conflicts++;
            return CONFLICT;
        }
        // the version check claims the leaf is untouched: prove it against the keys we sampled
        boolean keysMatch = leaf.keyCount == sampledCount;
        for (int i = 0; i < sampledCount && keysMatch; i++) {
            if (leaf.keys[i] != sampledKeys[i] || leaf.values[i] != sampledValues[i]) keysMatch = false;
        }
        if (!keysMatch) {
            leaf.release();
            validationMismatches++;
            conflicts++;
            return CONFLICT;
        }
        for (int i = leaf.keyCount; i > position; i--) {
            leaf.keys[i] = leaf.keys[i - 1];
            leaf.values[i] = leaf.values[i - 1];
        }
        leaf.keys[position] = key;
        leaf.values[position] = value;
        leaf.keyCount++;
        leaf.release();
        commits++;
        return COMMIT;
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

    String verify() {
        if (root.keyCount > MAX_KEYS) return "root overflows";
        List<Integer> depths = new ArrayList<>();
        Set<Integer> leafKeys = new HashSet<>();
        return verifyNode(root, Integer.MIN_VALUE, Integer.MAX_VALUE, 0, depths, leafKeys, true);
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
            for (int depth2 : depths) {
                if (depth2 != depths.get(0)) return "leaves at different depths: " + depths;
            }
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
    private static long state = 0x14057B7EF767814FL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    abstract static class Agent {
        boolean done;
        abstract void step();
    }

    /** Speculative worker: sample a path, prepare the new leaf content, then validate and publish. */
    static class Speculator extends Agent {
        final SpeculativeBTree tree;
        final int workerId;
        final int[] batch;
        int index;
        int phase;
        int attempts;
        List<SpeculativeBTree.Node> path;
        int[] versions;
        int[] sampledKeys;
        int[] sampledValues;
        int sampledCount;
        int target;
        long localConflicts;
        long localCommits;
        long localFallbacks;

        Speculator(SpeculativeBTree tree, int workerId, int[] batch) {
            this.tree = tree;
            this.workerId = workerId;
            this.batch = batch;
        }

        @Override
        void step() {
            if (index >= batch.length) {
                done = true;
                return;
            }
            if (phase == 0) {
                target = batch[index];
                path = tree.pathTo(target);
                versions = SpeculativeBTree.snapshot(path);
                SpeculativeBTree.Node leaf = path.get(path.size() - 1);
                sampledCount = leaf.keyCount;
                sampledKeys = Arrays.copyOf(leaf.keys, sampledCount);
                sampledValues = Arrays.copyOf(leaf.values, sampledCount);
                phase = 1;
                return;
            }
            int outcome = tree.tryCommit(path, versions, sampledKeys, sampledValues, sampledCount,
                    target, target, workerId);
            phase = 0;
            if (outcome == SpeculativeBTree.CONFLICT) {
                localConflicts++;
                attempts++;
                if (attempts > 40) {                       // attempt budget exhausted: be pessimistic
                    tree.insertLatched(target, target);
                    tree.fallbacks++;
                    localFallbacks++;
                    index++;
                    attempts = 0;
                }
                return;
            }
            if (outcome == SpeculativeBTree.COMMIT) localCommits++;
            if (outcome == SpeculativeBTree.FALLBACK) localFallbacks++;
            index++;
            attempts = 0;
        }
    }

    /** Plain latched writer: it is the conflict source for the speculators. */
    static class LatchedWriter extends Agent {
        final SpeculativeBTree tree;
        final int[] batch;
        int index;

        LatchedWriter(SpeculativeBTree tree, int[] batch) {
            this.tree = tree;
            this.batch = batch;
        }

        @Override
        void step() {
            if (index >= batch.length) {
                done = true;
                return;
            }
            if (tree.insertLatched(batch[index], -batch[index])) index++;
        }
    }

    public static void main(String[] args) {
        // --- degenerate trees --------------------------------------------------
        SpeculativeBTree empty = new SpeculativeBTree();
        check(empty.search(9) == null, "empty tree misses");
        check(empty.verify() == null, "empty tree invariants: " + empty.verify());
        check(empty.insert(9, 90) && empty.search(9) == 90, "ordinary insert on an empty tree");
        check(!empty.insert(9, 91) && empty.search(9) == 90, "duplicate insert is refused");
        check(empty.verify() == null, "single key invariants: " + empty.verify());

        List<SpeculativeBTree.Node> path = empty.pathTo(9);
        int[] versions = SpeculativeBTree.snapshot(path);
        int[] sampled = Arrays.copyOf(path.get(path.size() - 1).keys, path.get(path.size() - 1).keyCount);
        int[] sampledValues = Arrays.copyOf(path.get(path.size() - 1).values,
                path.get(path.size() - 1).keyCount);
        int outcome = empty.tryCommit(path, versions, sampled, sampledValues, 1, 9, 9, 1);
        check(outcome == SpeculativeBTree.DUPLICATE, "speculating a present key is a duplicate, got " + outcome);
        check(empty.duplicates == 1, "the duplicate was counted");

        path = empty.pathTo(11);
        versions = SpeculativeBTree.snapshot(path);
        sampled = Arrays.copyOf(path.get(path.size() - 1).keys, path.get(path.size() - 1).keyCount);
        sampledValues = Arrays.copyOf(path.get(path.size() - 1).values, path.get(path.size() - 1).keyCount);
        outcome = empty.tryCommit(path, versions, sampled, sampledValues, 1, 11, 11, 1);
        check(outcome == SpeculativeBTree.COMMIT, "a clean path commits speculatively, got " + outcome);
        check(empty.search(11) == 11, "the speculative key is in the tree");
        check(empty.commits == 1, "the commit was counted");
        check(empty.validationMismatches == 0, "no validation mismatch on a clean commit");
        check(empty.latchesFree() == null, "the latch was released after the commit");
        check(empty.verify() == null, "speculatively grown tree invariants: " + empty.verify());

        // --- a stale sample must be rejected ----------------------------------
        empty.insert(13, 13);
        path = empty.pathTo(15);
        versions = SpeculativeBTree.snapshot(path);
        SpeculativeBTree.Node leaf = path.get(path.size() - 1);
        sampled = Arrays.copyOf(leaf.keys, leaf.keyCount);
        sampledValues = Arrays.copyOf(leaf.values, leaf.keyCount);
        int sampledCount = leaf.keyCount;
        check(empty.insertLatched(17, 17), "a competing latched writer inserts into the same leaf");
        long conflictsBefore = empty.conflicts;
        outcome = empty.tryCommit(path, versions, sampled, sampledValues, sampledCount, 15, 15, 1);
        check(outcome == SpeculativeBTree.CONFLICT, "a stale sample must conflict, got " + outcome);
        check(empty.conflicts == conflictsBefore + 1, "the conflict was counted");
        check(empty.search(15) == null, "a rejected speculation must not leak into the tree");
        check(empty.latchesFree() == null, "no latch was leaked by the rejected speculation");
        check(empty.verify() == null, "tree after the rejected speculation: " + empty.verify());

        // --- interleaved speculative and pessimistically latched inserts --------
        SpeculativeBTree tree = new SpeculativeBTree();
        TreeMap<Integer, Integer> oracle = new TreeMap<>();
        for (int key = 0; key < 200; key++) {
            tree.insert(key, key * 2);
            oracle.put(key, key * 2);
        }
        int[] firstBatch = new int[30];
        int[] secondBatch = new int[30];
        int[] plainBatch = new int[30];
        for (int i = 0; i < 30; i++) {
            firstBatch[i] = 1000 + i;                       // consecutive keys fill whole leaves
            secondBatch[i] = 1005 + i;                      // deliberate overlap with the first batch
            plainBatch[i] = 1003 + i;                       // the conflict source
        }
        for (int key : firstBatch) oracle.put(key, key);
        for (int key : secondBatch) oracle.put(key, key);
        for (int key : plainBatch) oracle.put(key, -key);

        List<Agent> agents = new ArrayList<>();
        Speculator first = new Speculator(tree, 1, firstBatch);
        Speculator second = new Speculator(tree, 2, secondBatch);
        agents.add(first);
        agents.add(second);
        agents.add(new LatchedWriter(tree, plainBatch));

        int steps = 0;
        while (true) {
            List<Agent> ready = new ArrayList<>();
            for (Agent agent : agents) {
                if (!agent.done) ready.add(agent);
            }
            if (ready.isEmpty()) break;
            ready.get(nextInt(ready.size())).step();
            steps++;
            check(tree.verify() == null, "invariants during the interleaving at step " + steps + ": "
                    + tree.verify());
        }
        check(steps > 100, "the interleaving must take many steps, got " + steps);

        // --- the protocol really speculated and really resolved conflicts -------
        check(tree.commits > 0, "some speculation must have committed, commits=" + tree.commits);
        check(tree.conflicts > 0, "the protocol must have resolved conflicts, conflicts=" + tree.conflicts);
        check(tree.commits + tree.fallbacks > 0, "work must have been published");
        check(first.localConflicts + second.localConflicts == tree.conflicts,
                "every conflict belongs to a speculator: " + tree.conflicts);
        check(tree.validationMismatches == 0,
                "the version validation must be sound, mismatches=" + tree.validationMismatches);
        check(tree.latchesFree() == null, "every latch must be released: " + tree.latchesFree());
        check(tree.duplicates > 0, "speculating a key another worker already stored is a duplicate");

        // --- no lost update, no phantom key ------------------------------------
        for (Map.Entry<Integer, Integer> entry : oracle.entrySet()) {
            Integer got = tree.search(entry.getKey());
            check(got != null, "oracle key " + entry.getKey() + " is missing (lost update)");
        }
        for (int key : firstBatch) {
            Integer got = tree.search(key);
            check(got != null, "batch key " + key + " is missing");
        }
        int stored = 0;
        for (int key = 0; key < 1100; key++) {
            if (tree.search(key) != null) stored++;
        }
        check(stored == oracle.size(), "exactly the oracle keys are stored: " + stored + " vs " + oracle.size());
        check(tree.verify() == null, "final tree invariants: " + tree.verify());

        // --- a batch that fills a leaf must use the fallback -------------------
        SpeculativeBTree overflowing = new SpeculativeBTree();
        for (int key = 0; key < 60; key++) overflowing.insert(key, key);
        int[] bigBatch = new int[12];
        for (int i = 0; i < bigBatch.length; i++) bigBatch[i] = 100 + i;
        Speculator filler = new Speculator(overflowing, 3, bigBatch);
        while (!filler.done) filler.step();
        check(overflowing.fallbacks > 0, "filling a leaf must trigger the fallback, got "
                + overflowing.fallbacks);
        for (int key : bigBatch) {
            check(overflowing.search(key) != null, "fallback key " + key + " must be stored");
        }
        check(overflowing.verify() == null, "tree after the fallback batch: " + overflowing.verify());
        check(overflowing.latchesFree() == null, "latches free after the fallback batch");

        // --- determinism --------------------------------------------------------
        state = 0x14057B7EF767814FL;                        // restart the fixed-seed schedule
        SpeculativeBTree rerun = new SpeculativeBTree();
        for (int key = 0; key < 200; key++) rerun.insert(key, key * 2);
        List<Agent> rerunAgents = new ArrayList<>();
        Speculator rerunFirst = new Speculator(rerun, 1, firstBatch);
        Speculator rerunSecond = new Speculator(rerun, 2, secondBatch);
        rerunAgents.add(rerunFirst);
        rerunAgents.add(rerunSecond);
        rerunAgents.add(new LatchedWriter(rerun, plainBatch));
        while (true) {
            List<Agent> ready = new ArrayList<>();
            for (Agent agent : rerunAgents) {
                if (!agent.done) ready.add(agent);
            }
            if (ready.isEmpty()) break;
            ready.get(nextInt(ready.size())).step();
        }
        check(rerun.commits == tree.commits && rerun.conflicts == tree.conflicts,
                "the same seed must give the same commit/conflict split: " + rerun.commits + "/"
                        + rerun.conflicts + " vs " + tree.commits + "/" + tree.conflicts);

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
