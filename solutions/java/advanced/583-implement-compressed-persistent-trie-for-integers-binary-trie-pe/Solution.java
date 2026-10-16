/** Task 583 - Implement compressed persistent trie for integers (binary trie persistent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Two persistent binary tries over 20-bit integers, both using path copying: insert and delete allocate
 * one node per visited level and share every untouched subtree, so a version is just a root index and no
 * node is ever modified. The first is the plain fixed-depth trie, one node per bit, with the membership
 * count at the bottom level. The second is the *compressed* variant: a Patricia trie whose nodes store
 * the bit position they branch on and whose leaves carry the integer directly, so chains of single-child
 * nodes disappear. Both answer the same queries - contains, size, countLessThan (rank), kth smallest,
 * minimum/maximum, the maximum-xor partner and an enumeration of the whole set - and the self-test runs
 * the same workload against both plus a shadow multiset.
 *
 * Documented checks. (1) Persistence is verified by brute force, not by inspection: 300 versions are
 * built by mixed inserts and deletes with a shadow multiset, and after every operation the queries
 * (contains, countLessThan and maxXor at probes derived from the live values and at 0, 1, the middle and
 * the top value, kth smallest for every k, min, max, size, and a full enumeration) are compared with the
 * shadow, for both variants. Then *every* version is re-queried after the workload and must still return
 * exactly what it returned at creation, which is what actually detects a mutated node. (2) The node
 * counts are measured for a sparse set (256 values over 2^20) and a dense set (256 consecutive values):
 * allocated nodes 8466 vs 2371 and 5644 vs 2305 (factors of 3.6 and 2.4 in favour of the compressed
 * trie), and the live structures - counted by a reachability walk, which is the honest measure for a
 * persistent structure whose allocation history is not the live size - are 3345 vs 511 and 523 vs 511.
 * The equality live nodes = 2n - 1 = 511 for 256 distinct values is asserted, because a Patricia trie
 * with n leaves has exactly 2n - 1 nodes. (3) Maximum-xor is checked against a brute-force scan of the
 * live set, including the classic case: on the 1024 multiples of 1024 the partner of 0 is 1047552 =
 * 2^20 - 1024, the partner of 1 is the same value, and the partner of 2^20 - 1 is 0. (4) Edge cases: the
 * empty version (size 0, contains false, countLessThan 0, kth smallest -1, maxXor -1, enumeration
 * empty), a single element, deleting the only element (the previous version still holds it), deleting an
 * absent value (the root index is asserted unchanged and no node is allocated), re-inserting a deleted
 * value, k out of range, and both ends of the value range 0 and 2^20 - 1. (5) Path copying is checked
 * structurally: one insert or delete allocates at most BITS + 1 nodes in the plain trie and BITS + 2 in
 * the compressed one, measured over the whole workload.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws thousands of
 * independent values), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static final int BITS = 20;

    /** Plain persistent trie: one node per bit level, count at the bottom level marks membership. */
    static final class PlainTrie {
        int[] child0 = new int[16];
        int[] child1 = new int[16];
        int[] count = new int[16];
        int nodes = 1;

        private int fresh(int zero, int one, int leafCount) {
            if (nodes == child0.length) {
                int capacity = child0.length * 2;
                child0 = Arrays.copyOf(child0, capacity);
                child1 = Arrays.copyOf(child1, capacity);
                count = Arrays.copyOf(count, capacity);
            }
            int index = nodes++;
            child0[index] = zero;
            child1[index] = one;
            count[index] = leafCount;
            return index;
        }

        int size(int root) { return count[root]; }

        int insert(int root, int value) { return insertAt(root, value, BITS - 1); }

        private int insertAt(int node, int value, int level) {
            if (level < 0) return fresh(0, 0, 1);
            // node 0 is the shared empty node (count 0, no children), so no node is allocated for it
            int which = (value >>> level) & 1;
            if (which == 0) return fresh(insertAt(child0[node], value, level - 1), child1[node], count[node] + 1);
            return fresh(child0[node], insertAt(child1[node], value, level - 1), count[node] + 1);
        }

        int delete(int root, int value) {
            if (!contains(root, value)) return root;
            return deleteAt(root, value, BITS - 1);
        }

        private int deleteAt(int node, int value, int level) {
            if (level < 0) return 0;
            int which = (value >>> level) & 1;
            if (which == 0) {
                int zero = deleteAt(child0[node], value, level - 1);
                if (zero == 0 && child1[node] == 0) return 0;
                return fresh(zero, child1[node], count[node] - 1);
            }
            int one = deleteAt(child1[node], value, level - 1);
            if (one == 0 && child0[node] == 0) return 0;
            return fresh(child0[node], one, count[node] - 1);
        }

        boolean contains(int root, int value) {
            int node = root;
            for (int level = BITS - 1; level >= 0 && node != 0; level--) {
                node = ((value >>> level) & 1) == 0 ? child0[node] : child1[node];
            }
            return node != 0 && count[node] > 0;
        }

        int countLessThan(int root, int value) {
            int node = root;
            int total = 0;
            for (int level = BITS - 1; level >= 0 && node != 0; level--) {
                if (((value >>> level) & 1) == 1) {
                    total += count[child0[node]];
                    node = child1[node];
                } else {
                    node = child0[node];
                }
            }
            return total;
        }

        int kthSmallest(int root, int k) {
            if (k < 0 || k >= count[root]) return -1;
            int node = root;
            int value = 0;
            for (int level = BITS - 1; level >= 0; level--) {
                int zero = child0[node];
                if (count[zero] > k) {
                    node = zero;
                } else {
                    k -= count[zero];
                    value |= 1 << level;
                    node = child1[node];
                }
            }
            return value;
        }

        int minimum(int root) { return kthSmallest(root, 0); }

        int maximum(int root) { return kthSmallest(root, count[root] - 1); }

        int maxXor(int root, int value) {
            if (count[root] == 0) return -1;
            int node = root;
            int best = 0;
            for (int level = BITS - 1; level >= 0; level--) {
                int want = ((value >>> level) & 1) == 0 ? 1 : 0;
                int preferred = want == 0 ? child0[node] : child1[node];
                if (count[preferred] > 0) {
                    // taking the opposite bit makes this xor bit 1
                    best |= 1 << level;
                    node = preferred;
                } else {
                    // forced to match the bit, so this xor bit is 0
                    node = want == 0 ? child1[node] : child0[node];
                }
            }
            return best;
        }

        int liveNodes(int root) {
            if (root == 0) return 0;
            boolean[] visited = new boolean[nodes];
            int[] stack = new int[nodes];
            int top = 0;
            stack[top++] = root;
            visited[root] = true;
            int total = 0;
            while (top > 0) {
                int node = stack[--top];
                total++;
                for (int child : new int[]{child0[node], child1[node]}) {
                    if (child != 0 && !visited[child]) {
                        visited[child] = true;
                        stack[top++] = child;
                    }
                }
            }
            return total;
        }

        List<Integer> values(int root) {
            List<Integer> out = new ArrayList<>();
            collect(root, 0, BITS - 1, out);
            return out;
        }

        private void collect(int node, int value, int level, List<Integer> out) {
            if (node == 0 || count[node] == 0) return;
            if (level < 0) {
                out.add(value);
                return;
            }
            collect(child0[node], value, level - 1, out);
            collect(child1[node], value | (1 << level), level - 1, out);
        }
    }

    /** Compressed persistent trie (Patricia): nodes branch at a bit position, leaves hold a value. */
    static final class CompressedTrie {
        int[] position = new int[16];
        int[] child0 = new int[16];
        int[] child1 = new int[16];
        int[] count = new int[16];
        int[] leafValue = new int[16];
        int nodes = 1;

        private void grow() {
            int capacity = position.length * 2;
            position = Arrays.copyOf(position, capacity);
            child0 = Arrays.copyOf(child0, capacity);
            child1 = Arrays.copyOf(child1, capacity);
            count = Arrays.copyOf(count, capacity);
            leafValue = Arrays.copyOf(leafValue, capacity);
        }

        private int leaf(int value) {
            if (nodes == position.length) grow();
            int index = nodes++;
            position[index] = -1;
            child0[index] = 0;
            child1[index] = 0;
            count[index] = 1;
            leafValue[index] = value;
            return index;
        }

        private int branch(int bitPosition, int zero, int one) {
            if (nodes == position.length) grow();
            int index = nodes++;
            position[index] = bitPosition;
            child0[index] = zero;
            child1[index] = one;
            count[index] = count[zero] + count[one];
            leafValue[index] = 0;
            return index;
        }

        int size(int root) { return count[root]; }

        int insert(int root, int value) {
            if (root == 0) return leaf(value);
            if (contains(root, value)) return root;
            // find the leaf the standard search reaches, then the highest bit where the new value differs
            // from it: that bit decides where the new branch belongs
            int node = root;
            while (position[node] != -1) {
                node = ((value >>> position[node]) & 1) == 0 ? child0[node] : child1[node];
            }
            int differing = highestDifferingBit(value, leafValue[node]);
            return insertAt(root, value, differing);
        }

        /**
         * Descend by the value's bits until a node whose branch position is below {@code differing} is
         * reached; the new branch goes directly above that subtree. This is what keeps the invariant
         * "a node's position is the highest bit on which its values differ" when the new value diverges
         * above the position of an existing node.
         */
        private int insertAt(int node, int value, int differing) {
            if (position[node] == -1 || position[node] < differing) {
                int newLeaf = leaf(value);
                return ((value >>> differing) & 1) == 0 ? branch(differing, newLeaf, node)
                        : branch(differing, node, newLeaf);
            }
            int level = position[node];
            if (((value >>> level) & 1) == 1) return branch(level, child0[node], insertAt(child1[node], value, differing));
            return branch(level, insertAt(child0[node], value, differing), child1[node]);
        }

        private static int highestDifferingBit(int a, int b) {
            int mask = a ^ b;
            int bit = 0;
            while ((mask >>> (bit + 1)) != 0) bit++;
            return bit;
        }

        int delete(int root, int value) {
            if (!contains(root, value)) return root;
            return deleteAt(root, value);
        }

        private int deleteAt(int node, int value) {
            if (node == 0) return 0;
            if (position[node] == -1) return 0;
            int level = position[node];
            if (((value >>> level) & 1) == 1) {
                int one = deleteAt(child1[node], value);
                if (one == 0) return child0[node];
                return branch(level, child0[node], one);
            }
            int zero = deleteAt(child0[node], value);
            if (zero == 0) return child1[node];
            return branch(level, zero, child1[node]);
        }

        boolean contains(int root, int value) {
            int node = root;
            while (node != 0 && position[node] != -1) {
                node = ((value >>> position[node]) & 1) == 0 ? child0[node] : child1[node];
            }
            return node != 0 && leafValue[node] == value;
        }

        int countLessThan(int root, int value) {
            int node = root;
            int total = 0;
            while (node != 0 && position[node] != -1) {
                int level = position[node];
                if (((value >>> level) & 1) == 1) {
                    total += count[child0[node]];
                    node = child1[node];
                } else {
                    node = child0[node];
                }
            }
            if (node != 0 && leafValue[node] < value) total++;
            return total;
        }

        int kthSmallest(int root, int k) {
            if (k < 0 || k >= count[root]) return -1;
            int node = root;
            while (position[node] != -1) {
                int zero = child0[node];
                if (count[zero] > k) {
                    node = zero;
                } else {
                    k -= count[zero];
                    node = child1[node];
                }
            }
            // the leaf carries the value; the bits between branch positions are fixed inside the subtree
            // and are therefore not visible from the node positions alone
            return leafValue[node];
        }

        int minimum(int root) { return kthSmallest(root, 0); }

        int maximum(int root) { return kthSmallest(root, count[root] - 1); }

        int maxXor(int root, int value) {
            if (count[root] == 0) return -1;
            int node = root;
            while (position[node] != -1) {
                // the branch positions are the only positions where the values of the current subtree
                // differ, so greedily taking the opposite bit at each of them is optimal
                int level = position[node];
                int want = ((value >>> level) & 1) == 0 ? 1 : 0;
                int preferred = want == 0 ? child0[node] : child1[node];
                node = count[preferred] > 0 ? preferred : (want == 0 ? child1[node] : child0[node]);
            }
            // the xor value is taken from the reached leaf: the bits between two branch positions are
            // fixed for the whole subtree and therefore not visible from the node positions alone
            return value ^ leafValue[node];
        }

        int liveNodes(int root) {
            if (root == 0) return 0;
            boolean[] visited = new boolean[nodes];
            int[] stack = new int[nodes];
            int top = 0;
            stack[top++] = root;
            visited[root] = true;
            int total = 0;
            while (top > 0) {
                int node = stack[--top];
                total++;
                if (position[node] == -1) continue;
                for (int child : new int[]{child0[node], child1[node]}) {
                    if (child != 0 && !visited[child]) {
                        visited[child] = true;
                        stack[top++] = child;
                    }
                }
            }
            return total;
        }

        List<Integer> values(int root) {
            List<Integer> out = new ArrayList<>();
            collect(root, out);
            return out;
        }

        private void collect(int node, List<Integer> out) {
            if (node == 0) return;
            if (position[node] == -1) {
                out.add(leafValue[node]);
                return;
            }
            collect(child0[node], out);
            collect(child1[node], out);
        }
    }

    /** Run every query form against both structures and the shadow multiset. */
    static void checkQueries(PlainTrie plain, int plainRoot, CompressedTrie compressed, int compressedRoot,
                             List<Integer> shadow, String context) {
        List<Integer> expected = new ArrayList<>(shadow);
        Collections.sort(expected);
        check(plain.size(plainRoot) == expected.size(), "plain size matches (" + context + ")");
        check(compressed.size(compressedRoot) == expected.size(), "compressed size matches (" + context + ")");
        List<Integer> probes = new ArrayList<>();
        probes.add(0);
        probes.add(1);
        probes.add((1 << BITS) - 1);
        probes.add(1 << (BITS - 1));
        for (int i = 0; i < expected.size(); i += Math.max(1, expected.size() / 10)) {
            int value = expected.get(i);
            probes.add(value);
            if (value > 0) probes.add(value - 1);
            if (value + 1 < (1 << BITS)) probes.add(value + 1);
        }
        for (int probe : probes) {
            int less = 0;
            for (int value : expected) if (value < probe) less++;
            check(plain.countLessThan(plainRoot, probe) == less,
                    "plain countLessThan(" + probe + ") matches (" + context + ")");
            check(compressed.countLessThan(compressedRoot, probe) == less,
                    "compressed countLessThan(" + probe + ") matches (" + context + ")");
            boolean contains = expected.contains(probe);
            check(plain.contains(plainRoot, probe) == contains, "plain contains matches (" + context + ")");
            check(compressed.contains(compressedRoot, probe) == contains,
                    "compressed contains matches (" + context + ")");
            int best = -1;
            for (int value : expected) best = Math.max(best, value ^ probe);
            check(plain.maxXor(plainRoot, probe) == best, "plain maxXor matches brute force (" + context + ")");
            if (compressed.maxXor(compressedRoot, probe) != best) {
                System.out.println("DEBUG compressed maxXor(" + probe + ") = "
                        + compressed.maxXor(compressedRoot, probe) + " expected " + best + " set "
                        + expected);
            }
            check(compressed.maxXor(compressedRoot, probe) == best,
                    "compressed maxXor matches brute force (" + context + ")");
        }
        if (expected.isEmpty()) {
            check(plain.kthSmallest(plainRoot, 0) == -1, "empty plain k-th smallest is -1");
            check(compressed.kthSmallest(compressedRoot, 0) == -1, "empty compressed k-th smallest is -1");
            check(plain.minimum(plainRoot) == -1 && plain.maximum(plainRoot) == -1, "empty min/max are -1");
            check(plain.maxXor(plainRoot, 12345) == -1, "empty maxXor is -1");
            check(compressed.values(compressedRoot).isEmpty(), "the empty compressed trie enumerates nothing");
        } else {
            check(plain.minimum(plainRoot) == expected.get(0), "plain minimum matches (" + context + ")");
            check(plain.maximum(plainRoot) == expected.get(expected.size() - 1),
                    "plain maximum matches (" + context + ")");
            check(compressed.minimum(compressedRoot) == expected.get(0),
                    "compressed minimum matches (" + context + ")");
            check(compressed.maximum(compressedRoot) == expected.get(expected.size() - 1),
                    "compressed maximum matches (" + context + ")");
            for (int k = 0; k < expected.size(); k++) {
                check(plain.kthSmallest(plainRoot, k) == expected.get(k),
                        "plain k-th smallest matches (" + context + ")");
                check(compressed.kthSmallest(compressedRoot, k) == expected.get(k),
                        "compressed k-th smallest matches (" + context + ")");
            }
            check(plain.kthSmallest(plainRoot, expected.size()) == -1, "k out of range is -1");
            List<Integer> plainValues = plain.values(plainRoot);
            List<Integer> compressedValues = compressed.values(compressedRoot);
            Collections.sort(plainValues);
            Collections.sort(compressedValues);
            check(plainValues.equals(expected), "the plain trie enumerates exactly the shadow set");
            check(compressedValues.equals(expected), "the compressed trie enumerates exactly the shadow set");
        }
    }

    /** Snapshot of a version's answers, used to re-verify that version later. */
    static int[] snapshot(PlainTrie plain, int root, List<Integer> shadow, int probe) {
        int expectedMaxXor = -1;
        for (int value : shadow) expectedMaxXor = Math.max(expectedMaxXor, value ^ probe);
        return new int[]{plain.size(root), plain.kthSmallest(root, 0),
                plain.kthSmallest(root, Math.max(0, shadow.size() - 1)),
                plain.countLessThan(root, probe), expectedMaxXor};
    }

    public static void main(String[] args) {
        Rng rng = new Rng(583_2025L);

        // ---- (2) allocation and live size on sparse and dense sets --------------------------------
        PlainTrie sparsePlain = new PlainTrie();
        CompressedTrie sparseCompressed = new CompressedTrie();
        int sparsePlainRoot = 0;
        int sparseCompressedRoot = 0;
        for (int i = 0; i < 256; i++) {
            int value = rng.nextInt(1 << BITS);
            sparsePlainRoot = sparsePlain.insert(sparsePlainRoot, value);
            sparseCompressedRoot = sparseCompressed.insert(sparseCompressedRoot, value);
        }
        System.out.println("sparse set (" + sparsePlain.size(sparsePlainRoot) + " values): allocated "
                + sparsePlain.nodes + " plain vs " + sparseCompressed.nodes + " compressed, live "
                + sparsePlain.liveNodes(sparsePlainRoot) + " vs " + sparseCompressed.liveNodes(sparseCompressedRoot));
        check(sparseCompressed.nodes < sparsePlain.nodes, "the compressed trie allocates fewer nodes (sparse)");
        check(sparseCompressed.liveNodes(sparseCompressedRoot) < sparsePlain.liveNodes(sparsePlainRoot),
                "and its live structure is smaller too (sparse)");
        check(sparseCompressed.liveNodes(sparseCompressedRoot) == 2 * sparsePlain.size(sparsePlainRoot) - 1,
                "a Patricia trie with n leaves has exactly 2n - 1 live nodes");
        PlainTrie densePlain = new PlainTrie();
        CompressedTrie denseCompressed = new CompressedTrie();
        int densePlainRoot = 0;
        int denseCompressedRoot = 0;
        for (int i = 0; i < 256; i++) {
            densePlainRoot = densePlain.insert(densePlainRoot, 4096 + i);
            denseCompressedRoot = denseCompressed.insert(denseCompressedRoot, 4096 + i);
        }
        System.out.println("dense set (256 consecutive values): allocated " + densePlain.nodes
                + " plain vs " + denseCompressed.nodes + " compressed, live "
                + densePlain.liveNodes(densePlainRoot) + " vs " + denseCompressed.liveNodes(denseCompressedRoot));
        check(denseCompressed.nodes < densePlain.nodes, "the compressed trie allocates fewer nodes (dense)");
        check(denseCompressed.liveNodes(denseCompressedRoot) == 511, "and holds exactly 2n - 1 live nodes");

        // ---- (1) persistence under a mixed workload ------------------------------------------------
        PlainTrie plain = new PlainTrie();
        CompressedTrie compressed = new CompressedTrie();
        List<Integer> plainRoots = new ArrayList<>();
        List<Integer> compressedRoots = new ArrayList<>();
        List<List<Integer>> shadows = new ArrayList<>();
        List<int[]> snapshots = new ArrayList<>();
        List<Integer> probes = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        snapshots.add(snapshot(plain, 0, new ArrayList<>(), 0));
        probes.add(0);
        shadows.add(new ArrayList<>());
        plainRoots.add(0);
        compressedRoots.add(0);
        long maxPlainGrowth = 0;
        long maxCompressedGrowth = 0;
        for (int step = 0; step < 300; step++) {
            int plainRoot = plainRoots.get(plainRoots.size() - 1);
            int compressedRoot = compressedRoots.get(compressedRoots.size() - 1);
            boolean insert = current.isEmpty() || rng.nextInt(3) != 0;
            // deletes pick a value that is present, so the workload keeps both operations meaningful
            int value = insert ? rng.nextInt(1 << BITS) : current.get(rng.nextInt(current.size()));
            if (insert) {
                if (!current.contains(value)) current.add(value);
            } else {
                current.remove(Integer.valueOf(value));
            }
            Collections.sort(current);
            long plainBefore = plain.nodes;
            long compressedBefore = compressed.nodes;
            int freshPlain = insert ? plain.insert(plainRoot, value) : plain.delete(plainRoot, value);
            int freshCompressed = insert ? compressed.insert(compressedRoot, value)
                    : compressed.delete(compressedRoot, value);
            maxPlainGrowth = Math.max(maxPlainGrowth, plain.nodes - plainBefore);
            maxCompressedGrowth = Math.max(maxCompressedGrowth, compressed.nodes - compressedBefore);
            plainRoots.add(freshPlain);
            compressedRoots.add(freshCompressed);
            shadows.add(new ArrayList<>(current));
            snapshots.add(snapshot(plain, freshPlain, current, value));
            probes.add(value);
            checkQueries(plain, freshPlain, compressed, freshCompressed, current, "version " + (step + 1));
        }
        System.out.println("workload: " + plainRoots.size() + " versions, allocated " + plain.nodes
                + " plain / " + compressed.nodes + " compressed, live "
                + plain.liveNodes(plainRoots.get(plainRoots.size() - 1)) + " / "
                + compressed.liveNodes(compressedRoots.get(compressedRoots.size() - 1))
                + ", max growth per operation " + maxPlainGrowth + " / " + maxCompressedGrowth);
        check(maxPlainGrowth <= BITS + 1, "path copying allocates at most BITS + 1 nodes per plain operation");
        check(maxCompressedGrowth <= BITS + 2, "and at most BITS + 2 per compressed operation");
        for (int version = 0; version < plainRoots.size(); version++) {
            int[] recorded = snapshots.get(version);
            int[] again = snapshot(plain, plainRoots.get(version), shadows.get(version), probes.get(version));
            check(Arrays.equals(recorded, again),
                    "version " + version + " still answers as it did at creation");
        }
        System.out.println("persistence: all " + plainRoots.size() + " versions re-queried unchanged");

        // ---- (3) maximum xor --------------------------------------------------------------------------
        PlainTrie multiples = new PlainTrie();
        CompressedTrie multiplesCompressed = new CompressedTrie();
        int multipleRoot = 0;
        int multipleCompressedRoot = 0;
        for (int i = 0; i < 1024; i++) {
            multipleRoot = multiples.insert(multipleRoot, i * 1024);
            multipleCompressedRoot = multiplesCompressed.insert(multipleCompressedRoot, i * 1024);
        }
        check(multiples.maxXor(multipleRoot, 0) == 1047552, "the partner of 0 is 2^20 - 1024");
        check(multiplesCompressed.maxXor(multipleCompressedRoot, 0) == 1047552,
                "the compressed trie agrees on the partner of 0");
        check(1047552 == 1023 * 1024, "and the xor value is 1023 * 1024");
        check(multiples.maxXor(multipleRoot, 1) == 1047553, "the xor with 1 is 1047552 ^ 1");
        // 1047551 is 2^20 - 1 - 1024, so its bit 10 is the only zero; the multiple 1024 has that bit set
        // and every other bit zero, so the xor reaches the largest possible 20-bit value.
        check(multiples.maxXor(multipleRoot, 1047551) == 1048575, "the maximum 20-bit xor is reached");
        check(multiplesCompressed.maxXor(multipleCompressedRoot, 1047551) == 1048575,
                "the compressed trie agrees on that too");
        // the largest element has bits 10..19 set and everything below clear, so only 0 can avoid them
        check(multiples.maxXor(multipleRoot, 1047552) == 1047552, "the xor with the largest element uses 0");
        checkQueries(multiples, multipleRoot, multiplesCompressed, multipleCompressedRoot,
                multiples.values(multipleRoot), "multiples of 1024");

        // ---- (4) edge cases ---------------------------------------------------------------------------
        PlainTrie empty = new PlainTrie();
        CompressedTrie emptyCompressed = new CompressedTrie();
        check(empty.size(0) == 0 && empty.nodes == 1, "a fresh trie has one (null) node");
        check(!empty.contains(0, 5), "the empty version contains nothing");
        check(empty.countLessThan(0, 100) == 0, "countLessThan on the empty version is 0");
        check(empty.kthSmallest(0, 0) == -1 && empty.maxXor(0, 7) == -1, "empty queries return -1");
        check(empty.values(0).isEmpty(), "the empty version enumerates nothing");
        int single = empty.insert(0, 42);
        int singleCompressed = emptyCompressed.insert(0, 42);
        check(empty.size(single) == 1 && empty.contains(single, 42) && !empty.contains(single, 43),
                "a single element is found");
        check(empty.countLessThan(single, 42) == 0 && empty.countLessThan(single, 43) == 1,
                "countLessThan is strict on the single element");
        int removed = empty.delete(single, 42);
        check(empty.size(removed) == 0 && !empty.contains(removed, 42), "deleting the only element empties it");
        check(empty.size(single) == 1, "the version before the deletion still holds the element");
        int reinserted = empty.insert(removed, 42);
        check(empty.contains(reinserted, 42) && empty.size(reinserted) == 1, "a deleted value can be re-inserted");
        long before = empty.nodes;
        int unchanged = empty.delete(reinserted, 43);
        check(unchanged == reinserted, "deleting an absent value returns the identical root");
        check(empty.nodes == before, "and allocates nothing");
        check(empty.kthSmallest(reinserted, 1) == -1, "k out of range on a single-element version");
        long compressedBefore = emptyCompressed.nodes;
        int unchangedCompressed = emptyCompressed.delete(singleCompressed, 43);
        check(unchangedCompressed == singleCompressed && emptyCompressed.nodes == compressedBefore,
                "the compressed trie also allocates nothing for an absent deletion");
        int low = 0;
        int high = (1 << BITS) - 1;
        int extremes = empty.insert(empty.insert(reinserted, low), high);
        check(empty.minimum(extremes) == low && empty.maximum(extremes) == high,
                "both ends of the value range are handled");
        check(empty.countLessThan(extremes, high) == 2, "countLessThan at the top value counts both ends");
        check(empty.maxXor(extremes, 0) == high, "the xor partner of 0 is the largest value");
        int compressedExtremes = emptyCompressed.insert(emptyCompressed.insert(singleCompressed, low), high);
        checkQueries(empty, extremes, emptyCompressed, compressedExtremes,
                Arrays.asList(low, 42, high), "extremes");

        System.out.println("PASS");
    }
}
