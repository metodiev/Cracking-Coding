/** Task 757 - Implement an ORAM (oblivious RAM) sketch with an access-pattern checker.
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * Path ORAM: the server holds a binary tree of buckets of capacity Z, plus one block per leaf position
 * mapping. A client keeps the position map (block -> leaf) and a stash. To touch a block the client
 *
 *   1. looks up the block's leaf, picks a *fresh* pseudo-random leaf for it,
 *   2. reads the whole root-to-leaf path (the only thing the server ever sees),
 *   3. pulls every block on that path into the stash, answers the query from the stash,
 *   4. writes the path back, pushing blocks as deep as their own leaf allows.
 *
 * Obliviousness is then a property of the *trace*: the leaf of every access is uniform and independent
 * of the block id, and every access touches exactly levels+1 buckets. The file makes that testable:
 *  * the trace histogram is checked with a chi-square statistic against the uniform distribution;
 *  * the histograms of accesses to different blocks must be statistically indistinguishable;
 *  * a deliberately non-oblivious baseline (home-bucket lookup) must FAIL the same test, so the check
 *    is shown to have power rather than being a tautology.
 *
 * Structural invariants verified after every single access: a block only ever sits in a bucket that
 * lies on the path of that block's own leaf; no block id exists twice in tree+stash; tree+stash always
 * holds exactly the live blocks and their latest values (checked against a plain mirror array).
 *
 * Simplified: the tree, buckets and stash are plain Java structures instead of bytes encrypted with a
 * fresh key per access, so this is an access-pattern ORAM with no real cryptography.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** Path ORAM over a fixed universe of block ids. A block is {id, value, leaf}. */
final class PathOram {
    final int levels;                 // tree height; buckets live on levels 0..levels
    final int leaves;                 // number of leaves = 2^levels
    final int capacity;               // blocks per bucket (Z)
    final Map<Integer, List<int[]>> tree = new LinkedHashMap<>();
    final int[] position;
    final List<int[]> stash = new ArrayList<>();
    final Prng rng;
    final List<Integer> accessedLeaves = new ArrayList<>();
    long bucketsTouched;
    int bucketOverflows;
    int maxStash;

    PathOram(int levels, int capacity, int blockCount, long seed) {
        this.levels = levels;
        this.leaves = 1 << levels;
        this.capacity = capacity;
        this.position = new int[blockCount];
        this.rng = new Prng(seed);
        for (int node = 0; node < (1 << (levels + 1)) - 1; node++) tree.put(node, new ArrayList<>());
        for (int id = 0; id < blockCount; id++) position[id] = rng.nextInt(leaves);
    }

    /** Node index on level `level` of the path to `leaf`, where level 0 is the root. */
    int pathNode(int leaf, int level) { return ((1 << level) - 1) + (leaf >> (levels - level)); }

    boolean onPath(int leaf, int node) {
        for (int level = 0; level <= levels; level++) if (pathNode(leaf, level) == node) return true;
        return false;
    }

    /** Reads a block (or writes when `store` is true); returns null for a block never written. */
    int[] access(int id, int value, boolean store) {
        int leaf = position[id];
        accessedLeaves.add(leaf);
        int[] result;
        for (int level = 0; level <= levels; level++) {                  // step 2: read the path
            int node = pathNode(leaf, level);
            bucketsTouched++;
            List<int[]> bucket = tree.get(node);
            stash.addAll(bucket);                                       // step 3: into the stash
            bucket.clear();
        }
        int[] existing = null;
        for (int[] block : stash) {
            if (block[0] == id) { existing = block; break; }
        }
        position[id] = rng.nextInt(leaves);                             // step 1b: a fresh leaf
        if (existing != null) {
            if (store) existing[1] = value;                             // answered from the stash
            existing[2] = position[id];
            result = existing.clone();
        } else if (store) {
            int[] created = new int[]{id, value, position[id]};
            stash.add(created);
            result = created.clone();
        } else {
            result = null;                                              // never written
        }
        writeBack(leaf);                                                // step 4: push blocks down
        maxStash = Math.max(maxStash, stash.size());
        return result;
    }

    /** Greedy eviction: deepest bucket first, only where the block's own leaf allows it. */
    private void writeBack(int leaf) {
        for (int level = levels; level >= 0; level--) {
            int node = pathNode(leaf, level);
            List<int[]> bucket = tree.get(node);
            for (Iterator<int[]> it = stash.iterator(); it.hasNext(); ) {
                int[] block = it.next();
                if (bucket.size() >= capacity) break;
                if (!onPath(block[2], node)) continue;
                bucket.add(block);
                it.remove();
            }
            if (bucket.size() > capacity) bucketOverflows++;
        }
    }

    /** Invariant 1: every stored block sits on the path of its own leaf. */
    boolean placementInvariantHolds() {
        for (Map.Entry<Integer, List<int[]>> entry : tree.entrySet()) {
            for (int[] block : entry.getValue()) {
                if (!onPath(block[2], entry.getKey())) return false;
            }
        }
        return true;
    }

    /** Invariant 2: no block id appears twice in tree+stash. */
    boolean noDuplicates() {
        Set<Integer> seen = new HashSet<>();
        for (List<int[]> bucket : tree.values()) for (int[] block : bucket) if (!seen.add(block[0])) return false;
        for (int[] block : stash) if (!seen.add(block[0])) return false;
        return true;
    }

    /** Invariant 3: the server plus stash holds exactly the live blocks, with the newest values. */
    boolean matchesMirror(Map<Integer, Integer> mirror) {
        Map<Integer, Integer> found = new HashMap<>();
        for (List<int[]> bucket : tree.values()) for (int[] block : bucket) found.put(block[0], block[1]);
        for (int[] block : stash) found.put(block[0], block[1]);
        return found.equals(mirror);
    }
}

/** Access-pattern checks: uniformity and indistinguishability of the observed leaf trace. */
final class TraceChecker {

    static int[] histogram(List<Integer> leaves, int buckets) {
        int[] counts = new int[buckets];
        for (int leaf : leaves) counts[leaf]++;
        return counts;
    }

    /** Chi-square statistic of a trace against the uniform distribution over the leaves. */
    static double chiSquare(List<Integer> leaves, int bucketCount) {
        int[] counts = histogram(leaves, bucketCount);
        double expected = leaves.size() / (double) bucketCount;
        double statistic = 0;
        for (int count : counts) statistic += (count - expected) * (count - expected) / expected;
        return statistic;
    }

    /** Total variation distance between two traces, in [0, 1]. */
    static double totalVariation(List<Integer> first, List<Integer> second, int bucketCount) {
        int[] a = histogram(first, bucketCount), b = histogram(second, bucketCount);
        double distance = 0;
        for (int i = 0; i < bucketCount; i++) distance += Math.abs(a[i] - b[i]);
        return distance / (2.0 * first.size());
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    public static void main(String[] args) {
        int blocks = 64, levels = 4, capacity = 4, accesses = 600;
        PathOram oram = new PathOram(levels, capacity, blocks, 0x0A5EEDL);
        Map<Integer, Integer> mirror = new HashMap<>();

        // 1. workload correctness: every read agrees with a plain mirror array
        Prng rng = new Prng(0xA11CE5EEDL);
        int reads = 0, writes = 0, missing = 0;
        for (int i = 0; i < accesses; i++) {
            int id = rng.nextInt(blocks);
            boolean store = rng.nextInt(2) == 0;
            int value = rng.nextInt(1_000_000);
            int[] block = oram.access(id, value, store);
            if (store) {
                mirror.put(id, value);
                writes++;
            } else {
                if (!mirror.containsKey(id)) {
                    missing++;
                    check(block == null, "an unwritten block reads back as absent");
                } else {
                    check(block != null && block[1] == mirror.get(id), "read of block " + id + " returned "
                            + (block == null ? "null" : block[1]) + " but the mirror holds " + mirror.get(id));
                }
                reads++;
            }
            check(oram.placementInvariantHolds(), "block placement invariant after access " + i);
            check(oram.noDuplicates(), "no duplicated block after access " + i);
            check(oram.matchesMirror(mirror), "tree+stash equals the mirror after access " + i);
            check(oram.bucketOverflows == 0, "no bucket overflow at access " + i);
        }
        check(reads + writes == accesses, "every access was classified: " + reads + " reads, " + writes
                + " writes");
        check(missing > 0, "the workload included trivially missing blocks: " + missing);

        // 2. every access touches the same number of buckets: the path length leaks nothing
        check(oram.bucketsTouched == (long) accesses * (levels + 1), "every access read a full path of "
                + (levels + 1) + " buckets: " + oram.bucketsTouched);
        check(oram.bucketOverflows == 0, "the whole workload ran without a bucket overflow");
        check(oram.maxStash <= 3 * capacity * levels, "the stash stayed bounded: max " + oram.maxStash
                + " vs bound " + (3 * capacity * levels));

        // 3. obliviousness of one block: the leaves chosen are uniform (chi-square over the trace)
        List<Integer> trace = oram.accessedLeaves;
        double chi = TraceChecker.chiSquare(trace, oram.leaves);
        check(trace.size() == accesses, "the trace recorded every access");
        check(chi < 40.0, "one-block trace passes the uniformity test: chi2=" + chi
                + " (chi2(15) at 99.9% is about 38)");
        int[] counts = TraceChecker.histogram(trace, oram.leaves);
        int max = 0;
        for (int count : counts) max = Math.max(max, count);
        check(max < 3 * (accesses / oram.leaves), "no leaf dominates the trace: max count " + max
                + " vs expected " + (accesses / oram.leaves));

        // 4. two different access patterns are indistinguishable: block 0 vs block 1, 400 times each
        PathOram first = new PathOram(levels, capacity, blocks, 0x1111L);
        PathOram second = new PathOram(levels, capacity, blocks, 0x2222L);
        for (int i = 0; i < 400; i++) {
            first.access(0, i, true);
            second.access(1, i, true);
        }
        double distance = TraceChecker.totalVariation(first.accessedLeaves, second.accessedLeaves, first.leaves);
        check(distance < 0.25, "traces for block 0 and block 1 are statistically indistinguishable: TV="
                + distance);
        check(TraceChecker.chiSquare(first.accessedLeaves, first.leaves) < 40.0
                && TraceChecker.chiSquare(second.accessedLeaves, second.leaves) < 40.0,
                "both single-block traces are uniform");

        // 5. the test has power: a home-bucket baseline (every access goes to a fixed leaf) fails it
        List<Integer> home = new ArrayList<>();
        for (int access = 0; access < 400; access++) home.add(0);           // block 7 always hits leaf 0
        double leakyChi = TraceChecker.chiSquare(home, first.leaves);
        check(leakyChi > 40.0, "a home-bucket baseline is rejected by the same test: chi2=" + leakyChi);
        check(TraceChecker.totalVariation(home, first.accessedLeaves, first.leaves) > 0.5,
                "the leaky baseline is far from the oblivious trace");

        // 6. boundaries: smallest tree, repeated access to one block, deep path bookkeeping
        PathOram tiny = new PathOram(1, 2, 4, 0x5EEDL);
        check(tiny.leaves == 2 && tiny.tree.size() == 3, "a one-level tree has two leaves and three nodes");
        for (int i = 0; i < 20; i++) check(tiny.access(2, i, true)[1] == i, "repeated writes to one block "
                + "read back the latest value");
        check(tiny.placementInvariantHolds() && tiny.noDuplicates(), "the tiny tree keeps its invariants");
        check(tiny.bucketsTouched == (long) 20 * 2, "the tiny tree reads two buckets per access");
        check(oram.pathNode(0, 0) == 0, "the root is level 0");
        check(oram.pathNode(0, levels) == (1 << levels) - 1, "leaf 0 is the leftmost leaf node");
        check(oram.pathNode(oram.leaves - 1, levels) == (1 << (levels + 1)) - 2, "the last leaf is the "
                + "rightmost");

        // 7. determinism: the same seed replays the same leaf trace
        PathOram replay = new PathOram(levels, capacity, blocks, 0x0A5EEDL);
        Prng replayRng = new Prng(0xA11CE5EEDL);
        for (int i = 0; i < accesses; i++) {
            int id = replayRng.nextInt(blocks);
            boolean store = replayRng.nextInt(2) == 0;
            int value = replayRng.nextInt(1_000_000);
            replay.access(id, value, store);
        }
        check(replay.accessedLeaves.equals(oram.accessedLeaves), "same seed replays the same leaf trace");

        System.out.println("path ORAM: " + blocks + " blocks, " + oram.leaves + " leaves, Z=" + capacity
                + ", " + accesses + " accesses");
        System.out.println("trace uniformity: chi2=" + chi + " over " + oram.leaves + " leaves, "
                + "TV(block 0, block 1)=" + distance + ", home-bucket baseline chi2=" + leakyChi);
        System.out.println("max stash " + oram.maxStash + ", bucket overflows " + oram.bucketOverflows
                + ", bucket reads " + oram.bucketsTouched);
        System.out.println("PASS");
    }
}
