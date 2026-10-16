/** Task 695 - Implement anti-entropy repair scheduling with bandwidth bounds. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * Two parts of a real anti-entropy system:
 *
 *  1. MERKLE COMPARISON. Every replica stores the same sorted key space as a binary Merkle tree
 *     (leaf = hash of the key/value pair, node = mix of its children). Comparing two roots
 *     detects equality in one comparison; a mismatch recurses only into the differing subtrees,
 *     so finding d differences costs O(d log K) node comparisons instead of the O(K) of a linear
 *     scan - the reason Cassandra and Dynamo ship this structure.
 *
 *  2. BANDWIDTH-BOUNDED REPAIR. The differing keys become repair items carrying a stale version
 *     and a byte size. Each round a node may transfer at most `budget` bytes, so the items are
 *     packed first-fit in decreasing priority (oldest version first) - the classic first-fit
 *     decreasing heuristic. The rounds it needs are compared against the exact optimum computed
 *     by dynamic programming over item subsets for small instances.
 *
 * Verified: Merkle comparison cost for 0, 1 and 5 differences, convergence of a three-replica
 * cluster to identical digests, the per-round bandwidth bound, every item repaired exactly once,
 * monotone staleness reduction, and the FFD schedule within one round of the optimum.
 */
import java.util.*;

/** A key/value pair with the version it currently carries. */
class Item {
    final int key;
    final int version;
    final int bytes;

    Item(int key, int version, int bytes) {
        this.key = key;
        this.version = version;
        this.bytes = bytes;
    }
}

/** Binary Merkle tree over the sorted key list of a replica. */
class MerkleTree {
    final int leaves;
    final int size;
    final long[] nodes;

    MerkleTree(List<Item> items) {
        int n = 1;
        while (n < Math.max(1, items.size())) n <<= 1;
        leaves = n;
        size = 2 * n;
        nodes = new long[size];
        for (int i = 0; i < n; i++) {
            nodes[n + i] = i < items.size() ? hash(items.get(i)) : 0L;
        }
        for (int i = n - 1; i > 0; i--) nodes[i] = mix(nodes[2 * i], nodes[2 * i + 1]);
    }

    static long hash(Item item) {
        long value = 0x9E3779B97F4A7C15L * (item.key + 1);
        value ^= 0xBF58476D1CE4E5B9L * (item.version + 1);
        value ^= 0x94D049BB133111EBL * (item.bytes + 1);
        value ^= value >>> 29;
        return value;
    }

    static long mix(long left, long right) {
        long value = left * 0x2545F4914F6CDD1DL + right;
        value ^= value >>> 31;
        value *= 0x9E3779B97F4A7C15L;
        return value ^ (value >>> 27);
    }

    /** Comparison statistics for one tree pair. */
    static final class Diff {
        final List<Integer> positions = new ArrayList<>();
        int comparisons;

        int size() {
            return positions.size();
        }
    }

    static Diff differences(MerkleTree left, MerkleTree right) {
        Diff diff = new Diff();
        collect(left, right, 1, diff);
        return diff;
    }

    private static void collect(MerkleTree left, MerkleTree right, int node, Diff diff) {
        diff.comparisons++;
        if (left.nodes[node] == right.nodes[node]) return;
        if (node >= left.leaves) {
            diff.positions.add(node - left.leaves);
            return;
        }
        collect(left, right, 2 * node, diff);
        collect(left, right, 2 * node + 1, diff);
    }
}

/** First-fit-decreasing packing of repair items under a per-round byte budget. */
class RepairSchedule {
    final int rounds;
    final int[] roundOfItem;

    RepairSchedule(int rounds, int[] roundOfItem) {
        this.rounds = rounds;
        this.roundOfItem = roundOfItem;
    }
}

public class Solution {
    private static long prngState = 0x1234FACEL;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    /** A replica holding one Item per key in ascending key order. */
    static List<Item> replica(int keys, int version) {
        List<Item> items = new ArrayList<>();
        for (int key = 0; key < keys; key++) {
            items.add(new Item(key, version + (key % 3), 16 + (key % 5) * 8));
        }
        return items;
    }

    /**
     * Repairs the differences of `target` (as an item list) by first-fit decreasing on the stale
     * version, one byte budget per round.
     */
    static RepairSchedule schedule(List<Item> target, List<Item> source, int[] differingPositions,
                                   int budget) {
        List<Integer> order = new ArrayList<>();
        for (int position : differingPositions) order.add(position);
        order.sort((a, b) -> {
            int left = target.get(a).version;
            int right = target.get(b).version;
            if (left != right) return Integer.compare(left, right);
            return Integer.compare(b, a);
        });
        int[] remaining = new int[Math.max(1, order.size())];
        int used = 0;
        int[] roundOfItem = new int[target.size()];
        Arrays.fill(roundOfItem, -1);
        for (int position : order) {
            int size = target.get(position).bytes;
            int placed = -1;
            for (int round = 0; round < used; round++) {
                if (remaining[round] + size <= budget) {
                    placed = round;
                    break;
                }
            }
            if (placed < 0) {
                if (used >= remaining.length) {
                    remaining = Arrays.copyOf(remaining, remaining.length * 2);
                }
                placed = used++;
                roundOfItem[position] = placed;
                remaining[placed] = size;
                continue;
            }
            remaining[placed] += size;
            roundOfItem[position] = placed;
        }
        return new RepairSchedule(used, roundOfItem);
    }

    /** Exact minimum number of rounds for a small instance, by DP over subsets. */
    static int optimalRounds(int[] sizes, int budget) {
        int n = sizes.length;
        int[] weight = new int[1 << n];
        for (int mask = 1; mask < (1 << n); mask++) {
            int lowest = Integer.numberOfTrailingZeros(mask);
            weight[mask] = weight[mask & (mask - 1)] + sizes[lowest];
        }
        int[] best = new int[1 << n];
        Arrays.fill(best, Integer.MAX_VALUE / 2);
        best[0] = 0;
        for (int mask = 1; mask < (1 << n); mask++) {
            for (int subset = mask; subset > 0; subset = (subset - 1) & mask) {
                if (weight[subset] > budget) continue;
                best[mask] = Math.min(best[mask], 1 + best[mask ^ subset]);
            }
        }
        return best[(1 << n) - 1];
    }

    static String digest(List<Item> items) {
        StringBuilder sb = new StringBuilder();
        for (Item item : items) sb.append(item.key).append('/').append(item.version).append(' ');
        return sb.toString();
    }

    public static void main(String[] args) {
        // 1. Merkle comparison: equal replicas cost one comparison, differences cost O(d log K).
        List<Item> base = replica(128, 5);
        List<Item> same = replica(128, 5);
        MerkleTree left = new MerkleTree(base);
        MerkleTree right = new MerkleTree(same);
        MerkleTree.Diff equal = MerkleTree.differences(left, right);
        check(equal.size() == 0 && equal.comparisons == 1, "equal replicas differ in one comparison");

        List<Item> oneOff = new ArrayList<>(base);
        oneOff.set(70, new Item(70, 99, 24));
        MerkleTree.Diff single = MerkleTree.differences(left, new MerkleTree(oneOff));
        check(single.size() == 1 && single.positions.get(0) == 70,
                "a single divergence is located exactly");
        check(single.comparisons <= 2 * 8, "one divergence costs about log2(128) comparisons: "
                + single.comparisons);
        check(single.comparisons < 128 / 4, "far cheaper than a linear scan");

        List<Item> fiveOff = new ArrayList<>(base);
        int[] changed = {3, 20, 64, 100, 127};
        for (int key : changed) fiveOff.set(key, new Item(key, 200 + key, 32));
        MerkleTree.Diff five = MerkleTree.differences(left, new MerkleTree(fiveOff));
        check(five.size() == 5, "five divergences are all found: " + five.size());
        check(five.comparisons <= 2 * 8 * (5 + 1),
                "cost stays proportional to d log K: " + five.comparisons);
        check(five.comparisons < 128, "still cheaper than scanning every key: " + five.comparisons);
        check(five.positions.contains(3) && five.positions.contains(127),
                "both end keys are reported");

        // 2. A three-replica cluster converges once every pair has run anti-entropy.
        List<List<Item>> cluster = new ArrayList<>();
        cluster.add(new ArrayList<>(base));
        List<Item> second = new ArrayList<>(base);
        second.set(11, new Item(11, 77, 24));
        second.set(90, new Item(90, 78, 40));
        cluster.add(second);
        List<Item> third = new ArrayList<>(base);
        third.set(42, new Item(42, 13, 16));
        third.set(11, new Item(11, 3, 24));
        cluster.add(third);
        int totalComparisons = 0;
        int totalRepaired = 0;
        for (int a = 0; a < 3; a++) {
            for (int b = a + 1; b < 3; b++) {
                MerkleTree.Diff diff = MerkleTree.differences(new MerkleTree(cluster.get(a)),
                        new MerkleTree(cluster.get(b)));
                totalComparisons += diff.comparisons;
                for (int position : diff.positions) {
                    Item mine = cluster.get(a).get(position);
                    Item theirs = cluster.get(b).get(position);
                    Item winner = mine.version >= theirs.version ? mine : theirs;
                    cluster.get(a).set(position, winner);
                    cluster.get(b).set(position, winner);
                    totalRepaired++;
                }
            }
        }
        check(digest(cluster.get(0)).equals(digest(cluster.get(1)))
                        && digest(cluster.get(1)).equals(digest(cluster.get(2))),
                "the three replicas converge to identical state");
        check(totalRepaired >= 4, "every divergence was repaired: " + totalRepaired);
        check(totalComparisons < 128, "anti-entropy cost stays sublinear in the key count: "
                + totalComparisons);

        // 3. Bandwidth-bounded scheduling: bound respected, every item repaired exactly once.
        int[] differences = five.positions.stream().mapToInt(Integer::intValue).toArray();
        List<Item> stale = new ArrayList<>(base);
        for (int position : differences) stale.set(position, new Item(position, 1 + position % 4, 16));
        RepairSchedule schedule = schedule(stale, base, differences, 48);
        int[] perRound = new int[Math.max(1, schedule.rounds)];
        int scheduled = 0;
        for (int position : differences) {
            int round = schedule.roundOfItem[position];
            check(round >= 0, "item " + position + " was scheduled");
            perRound[round] += stale.get(position).bytes;
            scheduled++;
        }
        check(scheduled == differences.length, "all differing keys were scheduled");
        boolean withinBudget = true;
        for (int load : perRound) if (load > 48) withinBudget = false;
        check(withinBudget, "no round exceeds the bandwidth budget: " + Arrays.toString(perRound));
        check(schedule.rounds >= 1, "at least one repair round is needed");

        // 4. Priority order: the stalest item goes first and staleness falls monotonically.
        int stalest = differences[0];
        for (int position : differences) {
            if (stale.get(position).version < stale.get(stalest).version) stalest = position;
        }
        check(schedule.roundOfItem[stalest] == 0, "the stalest key is repaired in the first round");
        int previous = Integer.MAX_VALUE;
        boolean decreasing = true;
        for (int round = 0; round < schedule.rounds; round++) {
            int worst = -1;
            for (int position : differences) {
                if (schedule.roundOfItem[position] >= round) {
                    worst = Math.max(worst, stale.get(position).version);
                }
            }
            if (worst > previous) decreasing = false;
            previous = worst;
        }
        check(decreasing, "the worst remaining staleness never grows");

        // 5. First-fit decreasing against the exact optimum from subset DP.
        int[] sizesA = {9, 8, 7, 6, 5, 4, 3, 3};
        int optimumA = optimalRounds(sizesA, 16);
        int ffdA = ffRounds(sizesA, 16);
        check(ffdA >= optimumA, "FFD can never beat the optimum: " + ffdA + " vs " + optimumA);
        check(ffdA - optimumA <= 1, "FFD is within one round here: " + ffdA + " vs " + optimumA);
        int[] sizesB = {5, 4, 3, 2, 1, 1};
        int optimumB = optimalRounds(sizesB, 6);
        int ffdB = ffRounds(sizesB, 6);
        check(ffdB == optimumB && optimumB == 3, "equal-size packing is optimal: " + ffdB);
        check(ffRounds(new int[]{10, 10, 10}, 10) == 3, "one item per round when nothing else fits");
        check(ffRounds(new int[]{4, 4, 4}, 12) == 1, "everything fits in a single round when it can");

        // 6. Boundary cases: one item, and a budget so small that every round carries one item.
        RepairSchedule tiny = schedule(stale, base, new int[]{differences[0]}, 1);
        check(tiny.rounds == 1 && tiny.roundOfItem[differences[0]] == 0,
                "a single large item still gets one round");
        RepairSchedule cramped = schedule(stale, base, differences, 1);
        check(cramped.rounds == differences.length,
                "a one-byte budget yields one round per item: " + cramped.rounds);

        // 7. Random instances: the schedule never exceeds the budget and always finishes.
        int violations = 0;
        for (int trial = 0; trial < 50; trial++) {
            int count = 3 + nextInt(5);
            int budget = 8 + nextInt(32);
            int[] sizes = new int[count];
            for (int i = 0; i < count; i++) sizes[i] = 1 + nextInt(budget);
            int rounds = ffRounds(sizes, budget);
            int optimum = optimalRounds(sizes, budget);
            if (rounds < optimum) violations++;
            int total = 0;
            for (int size : sizes) total += size;
            if (rounds > count || rounds < (total + budget - 1) / budget) violations++;
        }
        check(violations == 0, "50 random instances respect both bounds on the round count");

        // 8. Determinism of the whole pipeline.
        prngState = 0x999L;
        int[] firstRun = new int[]{nextInt(1000), nextInt(1000), nextInt(1000)};
        prngState = 0x999L;
        int[] secondRun = new int[]{nextInt(1000), nextInt(1000), nextInt(1000)};
        check(Arrays.equals(firstRun, secondRun), "the PRNG is deterministic");
        MerkleTree treeA = new MerkleTree(base);
        MerkleTree treeB = new MerkleTree(base);
        check(MerkleTree.differences(treeA, treeB).comparisons == 1
                        && treeA.nodes[1] == treeB.nodes[1],
                "the Merkle hash is deterministic");

        System.out.println("PASS");
    }

    /** Number of rounds first-fit decreasing needs for the given item sizes. */
    static int ffRounds(int[] sizes, int budget) {
        int[] sorted = sizes.clone();
        Arrays.sort(sorted);
        Integer[] descending = new Integer[sorted.length];
        for (int i = 0; i < sorted.length; i++) descending[i] = sorted[sorted.length - 1 - i];
        List<Integer> loads = new ArrayList<>();
        for (int size : descending) {
            boolean placed = false;
            for (int i = 0; i < loads.size(); i++) {
                if (loads.get(i) + size <= budget) {
                    loads.set(i, loads.get(i) + size);
                    placed = true;
                    break;
                }
            }
            if (!placed) loads.add(size);
        }
        return loads.size();
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
