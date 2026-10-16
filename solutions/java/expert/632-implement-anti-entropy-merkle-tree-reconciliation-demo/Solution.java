/** Task 632 - Implement anti-entropy Merkle tree reconciliation demo. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Anti-entropy compares two replicas by *hash tree*, not by content. The key space is split into
 * fixed buckets; each bucket hashes its entries, and each internal node hashes the concatenation
 * of its children, so the root commits to the whole key space. Two replicas exchange roots first:
 * equal roots mean "identical", and otherwise they descend only into the children whose hashes
 * differ. Because a single changed key dirties just one leaf and the path above it, reconciliation
 * costs O(diff * log B) hash comparisons and transfers exactly the differing entries.
 *
 * Simulation model, and what is simplified: there are no nodes, sockets or bandwidth. A bucket is
 * an index and a replica is a sorted map, so "exchanging a hash" is an integer comparison and
 * "transferring an entry" is a map put. The tree is a fixed complete binary tree over 16 buckets,
 * which keeps it deterministic and small; the hashing is FNV-1a over a canonical serialization, so
 * it does not depend on iteration or insertion order. What is *not* simplified is the algorithm:
 * prune-on-equal-hash, descend-on-differ, and reconcile only the leaves that disagree.
 *
 * Verified: (1) identical replicas reconcile with exactly one hash comparison and an empty diff;
 * (2) a single changed key is found while visiting O(log B) nodes, far below the whole tree;
 * (3) on PRNG-generated divergence the Merkle diff equals a brute-force comparison of both maps,
 * key for key; (4) applying the diff in both directions makes the replicas identical and transfers
 * exactly the differing entries; (5) additions, removals, empty replicas and single-entry
 * replicas behave; (6) the root changes when any leaf changes, while sibling subtrees off the
 * changed path keep their hashes; (7) the range hash matches an independently computed fold over
 * the leaf hashes, so the tree is not self-confirming.
 */
import java.util.*;

class Replica {
    final Map<Integer, Integer> entries = new TreeMap<>();
    final int buckets;

    Replica(int buckets) {
        this.buckets = buckets;
    }

    void put(int key, int value) {
        entries.put(key, value);
    }

    void remove(int key) {
        entries.remove(key);
    }

    int bucketOf(int key) {
        return Math.floorMod(key, buckets);
    }

    Map<Integer, Integer> bucketEntries(int bucket) {
        Map<Integer, Integer> out = new TreeMap<>();
        for (Map.Entry<Integer, Integer> e : entries.entrySet()) {
            if (bucketOf(e.getKey()) == bucket) out.put(e.getKey(), e.getValue());
        }
        return out;
    }

    String fingerprint() {
        return entries.toString();
    }
}

/** Complete binary hash tree over the buckets plus the prune-and-descend diff. */
class MerkleTree {
    final Replica replica;
    final int buckets;
    int visited;                     // hash comparisons performed by the last diff
    int leavesCompared;

    MerkleTree(Replica replica) {
        this.replica = replica;
        this.buckets = replica.buckets;
    }

    static long fnv(String text) {
        long hash = 0xcbf29ce484222325L;
        for (int i = 0; i < text.length(); i++) {
            hash ^= text.charAt(i);
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    long leafHash(int bucket) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, Integer> e : replica.bucketEntries(bucket).entrySet()) {
            sb.append(e.getKey()).append('=').append(e.getValue()).append(';');
        }
        return fnv(sb.toString());
    }

    long childHash(long left, long right) {
        return fnv(left + ":" + right);
    }

    long rangeHash(int lo, int hi) {
        if (hi - lo == 1) return leafHash(lo);
        int mid = (lo + hi) >>> 1;
        return childHash(rangeHash(lo, mid), rangeHash(mid, hi));
    }

    long rootHash() {
        return rangeHash(0, buckets);
    }

    /** Independent oracle: the same tree computed bottom-up, without the recursive descent. */
    long rootHashBottomUp() {
        long[] level = new long[buckets];
        for (int i = 0; i < buckets; i++) level[i] = leafHash(i);
        int width = buckets;
        while (width > 1) {
            for (int i = 0; i < width / 2; i++) level[i] = childHash(level[2 * i], level[2 * i + 1]);
            width /= 2;
        }
        return level[0];
    }

    int totalNodes() {
        return 2 * buckets - 1;
    }

    /** Keys whose value differs between the two replicas, found by descending hash mismatches. */
    List<Integer> diff(MerkleTree other) {
        visited = 0;
        leavesCompared = 0;
        List<Integer> out = new ArrayList<>();
        diffRange(other, 0, buckets, out);
        Collections.sort(out);
        return out;
    }

    private void diffRange(MerkleTree other, int lo, int hi, List<Integer> out) {
        visited++;
        if (rangeHash(lo, hi) == other.rangeHash(lo, hi)) return;     // prune: subtrees agree
        if (hi - lo == 1) {
            leavesCompared++;
            Map<Integer, Integer> mine = replica.bucketEntries(lo);
            Map<Integer, Integer> theirs = other.replica.bucketEntries(lo);
            Set<Integer> keys = new TreeSet<>(mine.keySet());
            keys.addAll(theirs.keySet());
            for (int key : keys) {
                Integer a = mine.get(key);
                Integer b = theirs.get(key);
                if (a == null ? b != null : !a.equals(b)) out.add(key);
            }
            return;
        }
        int mid = (lo + hi) >>> 1;
        diffRange(other, lo, mid, out);
        diffRange(other, mid, hi, out);
    }

    /** Brute force oracle: walk both maps and compare. */
    static List<Integer> bruteForceDiff(Replica a, Replica b) {
        List<Integer> out = new ArrayList<>();
        Set<Integer> keys = new TreeSet<>(a.entries.keySet());
        keys.addAll(b.entries.keySet());
        for (int key : keys) {
            Integer x = a.entries.get(key);
            Integer y = b.entries.get(key);
            if (x == null ? y != null : !x.equals(y)) out.add(key);
        }
        return out;
    }

    /** Copy every differing entry from `from` to `to`; returns how many entries moved. */
    static int reconcile(Replica from, Replica to, List<Integer> keys) {
        int transferred = 0;
        for (int key : keys) {
            Integer value = from.entries.get(key);
            if (value == null) to.entries.remove(key);
            else to.entries.put(key, value);
            transferred++;
        }
        return transferred;
    }
}

public class Solution {
    static final int BUCKETS = 16;
    static long seed = 0x1234ABCD5678L;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static Replica replicaWith(int count) {
        Replica r = new Replica(BUCKETS);
        for (int i = 0; i < count; i++) r.put(i, i * 7 + 1);
        return r;
    }

    public static void main(String[] args) {
        identicalReplicas();
        singleChangeIsCheap();
        diffMatchesBruteForce();
        reconciliationConverges();
        additionsAndRemovals();
        hashTreeStructure();
        degenerateReplicas();
        System.out.println("PASS");
    }

    /** Equal roots mean identical: one comparison, nothing transferred. */
    static void identicalReplicas() {
        Replica a = replicaWith(64);
        Replica b = replicaWith(64);
        MerkleTree ta = new MerkleTree(a);
        MerkleTree tb = new MerkleTree(b);
        check(ta.rootHash() == tb.rootHash(), "identical replicas share a root hash");
        check(ta.rootHash() == ta.rootHashBottomUp(), "range hash matches the independent fold");
        List<Integer> diff = ta.diff(tb);
        check(diff.isEmpty(), "no differing keys");
        check(ta.visited == 1 && ta.leavesCompared == 0, "exactly the root was compared: "
                + ta.visited);
        check(ta.totalNodes() == 2 * BUCKETS - 1, "tree size: " + ta.totalNodes());

        Replica c = replicaWith(64);
        MerkleTree tc = new MerkleTree(c);
        c.put(5, 999);
        c.put(5, 5 * 7 + 1);                       // restored: hashes must agree again
        check(tc.rootHash() == ta.rootHash(), "a value restored to the original restores the root");
    }

    /** One changed key descends only the path to its leaf. */
    static void singleChangeIsCheap() {
        Replica a = replicaWith(64);
        Replica b = replicaWith(64);
        b.put(37, 12345);
        MerkleTree ta = new MerkleTree(a);
        MerkleTree tb = new MerkleTree(b);
        check(ta.rootHash() != tb.rootHash(), "the root changed");
        List<Integer> diff = ta.diff(tb);
        check(diff.equals(Arrays.asList(37)), "the changed key was found: " + diff);
        check(ta.visited <= 2 * 5 + 1, "visited a path, not the tree: " + ta.visited);
        check(ta.visited < ta.totalNodes() / 2, "far below the whole tree: " + ta.visited
                + " of " + ta.totalNodes());
        check(ta.leavesCompared == 1, "exactly one leaf bucket was inspected: " + ta.leavesCompared);

        int oldVisited = ta.visited;
        for (int i = 0; i < 40; i++) b.put(100 + i, i);        // spread changes over buckets
        ta.diff(tb);
        check(ta.visited > oldVisited, "more differences cost more comparisons: " + ta.visited);
        check(ta.visited <= ta.totalNodes(), "never more than the whole tree");
    }

    /** The Merkle diff must equal a brute-force comparison of both maps. */
    static void diffMatchesBruteForce() {
        for (int trial = 0; trial < 6; trial++) {
            Replica a = new Replica(BUCKETS);
            Replica b = new Replica(BUCKETS);
            for (int i = 0; i < 40 + trial * 5; i++) {
                int key = nextInt(200);
                a.put(key, nextInt(1000));
            }
            for (int i = 0; i < 40 + trial * 5; i++) {
                int key = nextInt(200);
                b.put(key, nextInt(1000));
            }
            // Introduce a few deliberate mutations on each side.
            for (int i = 0; i < 3; i++) b.put(nextInt(200), 5000 + i);
            for (int i = 0; i < 3; i++) a.remove(nextInt(200));
            for (int i = 0; i < 3; i++) b.remove(nextInt(200));

            MerkleTree ta = new MerkleTree(a);
            MerkleTree tb = new MerkleTree(b);
            List<Integer> viaTree = ta.diff(tb);
            List<Integer> viaBrute = MerkleTree.bruteForceDiff(a, b);
            check(viaTree.equals(viaBrute), "trial " + trial + ": merkle " + viaTree
                    + " vs brute " + viaBrute);
            check(!viaTree.isEmpty(), "trial " + trial + ": the replicas really differ");
            check(ta.rootHash() != tb.rootHash(), "trial " + trial + ": roots differ");
        }
    }

    /** Applying the diff both ways converges the replicas and moves only differing entries. */
    static void reconciliationConverges() {
        Replica a = replicaWith(50);
        Replica b = replicaWith(50);
        for (int key : new int[]{3, 17, 31, 44}) b.put(key, key + 1000);
        b.put(120, 7);
        a.remove(9);
        MerkleTree ta = new MerkleTree(a);
        MerkleTree tb = new MerkleTree(b);
        List<Integer> diff = ta.diff(tb);
        check(diff.equals(Arrays.asList(3, 9, 17, 31, 44, 120)), "the union of differences: " + diff);

        int toB = MerkleTree.reconcile(a, b, diff);
        int toA = MerkleTree.reconcile(b, a, diff);
        check(toB == diff.size() && toA == diff.size(), "one transfer per differing key: "
                + toB + "/" + toA);
        check(a.fingerprint().equals(b.fingerprint()), "the replicas are now identical");
        check(new MerkleTree(a).rootHash() == new MerkleTree(b).rootHash(), "and share a root");
        check(new MerkleTree(a).diff(new MerkleTree(b)).isEmpty(), "re-running the diff finds nothing");
    }

    /** Additions and removals in every bucket, including the first and the last. */
    static void additionsAndRemovals() {
        Replica a = new Replica(BUCKETS);
        Replica b = new Replica(BUCKETS);
        a.put(0, 1);                     // bucket 0
        a.put(15, 1);                    // the last bucket
        b.put(0, 2);
        b.remove(15);
        b.put(31, 9);                    // bucket 15 again
        MerkleTree ta = new MerkleTree(a);
        MerkleTree tb = new MerkleTree(b);
        List<Integer> diff = ta.diff(tb);
        check(diff.equals(Arrays.asList(0, 15, 31)), "boundary buckets detected: " + diff);
        check(ta.visited > 1, "the diff had to descend: " + ta.visited);

        MerkleTree.reconcile(a, b, diff);
        MerkleTree.reconcile(b, a, diff);
        check(a.fingerprint().equals(b.fingerprint()), "converged");
        check(a.entries.get(0) == 1 && a.entries.containsKey(15) && !a.entries.containsKey(31),
                "the update propagated down and the extra key was dropped: " + a.entries);
        check(b.entries.get(0) == 1 && b.entries.containsKey(15) && !b.entries.containsKey(31),
                "both replicas ended on the same content: " + b.entries);
    }

    /** Root sensitivity and sibling stability: only the path to the change moves. */
    static void hashTreeStructure() {
        Replica a = replicaWith(64);
        MerkleTree ta = new MerkleTree(a);
        long root = ta.rootHash();
        a.put(1, 42);
        check(ta.rootHash() != root, "changing a leaf changes the root");

        Replica b = replicaWith(64);
        MerkleTree tb = new MerkleTree(b);
        b.put(1, 42);
        check(ta.rootHash() == tb.rootHash(), "the same change leads to the same root");

        MerkleTree clean = new MerkleTree(replicaWith(64));
        check(ta.rangeHash(8, 16) == clean.rangeHash(8, 16),
                "the untouched half keeps its hash");
        check(ta.rangeHash(1, 2) != clean.rangeHash(1, 2),
                "the changed bucket has a different hash");
        check(ta.leafHash(1) != clean.leafHash(1) && ta.leafHash(9) == clean.leafHash(9),
                "only the touched leaf hash moved");

        Set<Long> hashes = new TreeSet<>();
        for (int i = 0; i < 16; i++) hashes.add(clean.leafHash(i));
        check(hashes.size() == 16, "leaf hashes differ across buckets");
        int counted = 0;
        for (int i = 0; i < BUCKETS; i++) counted += clean.replica.bucketEntries(i).size();
        check(counted == 64 && clean.replica.bucketEntries(0).size() == 4,
                "keys are partitioned over the buckets: " + counted);
    }

    /** Empty, single-entry and one-sided replicas. */
    static void degenerateReplicas() {
        Replica empty = new Replica(BUCKETS);
        Replica other = new Replica(BUCKETS);
        MerkleTree te = new MerkleTree(empty);
        MerkleTree to = new MerkleTree(other);
        check(te.rootHash() == to.rootHash(), "two empty replicas agree");
        check(!te.diff(to).iterator().hasNext(), "nothing to reconcile");

        other.put(4, 4);
        List<Integer> diff = te.diff(to);
        check(diff.equals(Arrays.asList(4)), "one-sided insertion is detected: " + diff);
        check(te.visited <= 2 * 5 + 1 && te.leavesCompared == 1,
                "descended one path only: " + te.visited + " nodes, " + te.leavesCompared + " leaf");
        MerkleTree.reconcile(other, empty, diff);
        check(empty.fingerprint().equals(other.fingerprint()), "the empty replica caught up");

        Replica single = new Replica(BUCKETS);
        single.put(7, 7);
        Replica missing = new Replica(BUCKETS);
        List<Integer> removal = new MerkleTree(single).diff(new MerkleTree(missing));
        check(removal.equals(Arrays.asList(7)), "deleting the only entry is detected: " + removal);
        MerkleTree.reconcile(missing, single, removal);
        check(single.fingerprint().equals(missing.fingerprint()), "converged to empty");
        check(new MerkleTree(single).rootHash()
                        == new MerkleTree(new Replica(BUCKETS)).rootHash(),
                "an empty replica has the same root as any other empty replica");
        check(!new MerkleTree(single).diff(new MerkleTree(new Replica(BUCKETS))).iterator()
                .hasNext(), "and reconciles with a fresh replica without work");
    }
}
