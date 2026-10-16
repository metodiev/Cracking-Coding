/** Task 773 - Implement self-adjusting skip lists with locality heuristics. (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * A skip list keeps a level-0 list of all keys plus coarse express lanes above it; a new node's tower
 * height is drawn from deterministic splitmix64 coin flips, so the structure is reproducible. On top
 * of that this implementation adds the two locality heuristics that make the list *self adjusting*:
 *
 *   1. frequency promotion - every successful search counts one access on the node it found. When a
 *      node reaches the promotion threshold it is lifted one lane, provided the target lane is still
 *      sparse (laneCount + 1 <= 1.5 * n / 2^lane) and the node has not used up its maxLift extra
 *      lanes. Lanes therefore stay close to the ideal skip-list density while the keys that are
 *      requested repeatedly gain express coverage of their own.
 *   2. a bounded hot cache - the most frequently requested keys are also kept in a small access
 * as a bounded access ordered map, so a repeated lookup is answered without any list descent at all.
 * The cache is capped, evicts in access order and is repaired by deletion, so it cannot grow with the
 * number of distinct keys ever seen. A key served from the cache therefore stops accumulating
 * accesses; it resumes counting when it is evicted, which is how a tower can climb more than once.
 *
 * Invariants verified after every mutation: for each lane i, the lane holds exactly the nodes whose
 * tower is taller than i, in increasing key order, and the maintained lane counters agree with the
 * lanes they describe (verifyLevels); level 0 is strictly increasing; every tower length is in
 * [1, maxLevel]; a promotion raises a tower by exactly one and never beyond baseLevel + maxLift;
 * delete removes a node from every lane and from the hot cache; and an oracle workload over a TreeMap
 * confirms that membership answers are unchanged by the heuristics. The locality claim is measured,
 * not asserted: on a skewed workload the adaptive list performs far fewer probe steps than the same
 * list with the heuristics switched off, and the cache really serves the repeated keys.
 *
 * Simplified: single threaded, promotion only (a stale entry leaves the hot cache by eviction rather
 * than demotion), and the measured locality experiments run on a few hundred keys.
 */
import java.util.*;

class SkipList {
    static class Node {
        final int key;
        final Node[] next;
        final int baseLevel;
        int level;
        int accesses;

        Node(int key, int maxLevel, int baseLevel) {
            this.key = key;
            this.next = new Node[maxLevel];
            this.baseLevel = baseLevel;
            this.level = baseLevel;
        }
    }

    final int maxLevel;
    final boolean adaptive;
    final int promoteThreshold;
    final int maxLift;
    final int hotCapacity;
    final Node head;
    final int[] laneCount;
    final LinkedHashMap<Integer, Node> hot;
    int size;
    long steps;
    long searches;
    long cacheHits;
    int promotions;
    private long randomState;

    SkipList(int maxLevel, boolean adaptive, int promoteThreshold, int maxLift, int hotCapacity, long seed) {
        if (maxLevel < 1) throw new IllegalArgumentException("maxLevel must be positive");
        if (promoteThreshold < 1) throw new IllegalArgumentException("promoteThreshold must be positive");
        if (maxLift < 0 || hotCapacity < 0) throw new IllegalArgumentException("negative heuristic bound");
        this.maxLevel = maxLevel;
        this.adaptive = adaptive;
        this.promoteThreshold = promoteThreshold;
        this.maxLift = maxLift;
        this.hotCapacity = hotCapacity;
        this.head = new Node(Integer.MIN_VALUE, maxLevel, 1);
        this.laneCount = new int[maxLevel];
        this.hot = new LinkedHashMap<>();
        this.randomState = seed | 1L;
    }

    /** splitmix64: deterministic coin flips for the tower height. */
    private long nextLong() {
        randomState += 0x9E3779B97F4A7C15L;
        long z = randomState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private int randomLevel() {
        int level = 1;
        while (level < maxLevel && (nextLong() & 1L) == 0L) level++;
        return level;
    }

    /** Standard descent from the top lane; returns the node holding the key or null. */
    private Node find(int key) {
        Node cur = head;
        long probe = 0;
        for (int i = maxLevel - 1; i >= 0; i--) {
            while (cur.next[i] != null && cur.next[i].key < key) {
                cur = cur.next[i];
                probe++;
            }
            probe++;
        }
        Node candidate = cur.next[0];
        probe++;
        steps += probe;
        searches++;
        return candidate != null && candidate.key == key ? candidate : null;
    }

    boolean contains(int key) {
        Node cached = hot.get(key);
        if (cached != null) {                       // O(1) hit: refresh the access order
            hot.remove(key);
            hot.put(key, cached);
            cacheHits++;
            searches++;
            steps++;
            return true;
        }
        Node node = find(key);
        if (node != null) onAccess(node);
        return node != null;
    }

    /** Counts the access and, at the threshold, promotes the node and caches the key. */
    private void onAccess(Node node) {
        if (!adaptive) return;
        node.accesses++;
        if (node.accesses % promoteThreshold != 0) return;
        if (!hot.containsKey(node.key)) {
            if (hot.size() >= hotCapacity) hot.remove(hot.keySet().iterator().next());
            hot.put(node.key, node);
        }
        promoteIfSparse(node);
    }

    /** Lifts the tower by one lane when that keeps the target lane close to its ideal density. */
    private void promoteIfSparse(Node node) {
        int lane = node.level;
        if (lane >= maxLevel || node.level >= node.baseLevel + maxLift) return;
        long ideal = Math.max(2L, size >> lane);              // a p = 1/2 lane holds about n / 2^lane
        if (laneCount[lane] + 1 > ideal + ideal / 2) return;  // leave the express lanes sparse
        Node cur = head;
        for (int i = maxLevel - 1; i >= lane; i--) {
            while (cur.next[i] != null && cur.next[i].key < node.key) cur = cur.next[i];
        }
        node.next[lane] = cur.next[lane];
        cur.next[lane] = node;
        node.level++;
        laneCount[lane]++;
        promotions++;
    }

    boolean insert(int key) {
        Node[] update = new Node[maxLevel];
        Node cur = head;
        for (int i = maxLevel - 1; i >= 0; i--) {
            while (cur.next[i] != null && cur.next[i].key < key) cur = cur.next[i];
            update[i] = cur;
        }
        if (cur.next[0] != null && cur.next[0].key == key) return false;
        int level = randomLevel();
        Node node = new Node(key, maxLevel, level);
        for (int i = 0; i < level; i++) {
            node.next[i] = update[i].next[i];
            update[i].next[i] = node;
            laneCount[i]++;
        }
        size++;
        return true;
    }

    boolean delete(int key) {
        hot.remove(key);
        Node[] update = new Node[maxLevel];
        Node cur = head;
        for (int i = maxLevel - 1; i >= 0; i--) {
            while (cur.next[i] != null && cur.next[i].key < key) cur = cur.next[i];
            update[i] = cur;
        }
        Node victim = cur.next[0];
        if (victim == null || victim.key != key) return false;
        for (int i = 0; i < victim.level; i++) {
            if (update[i].next[i] == victim) {
                update[i].next[i] = victim.next[i];
                laneCount[i]--;
            }
        }
        size--;
        return true;
    }

    /** Level-0 keys in order: ground truth for sortedness and membership. */
    List<Integer> keys() {
        List<Integer> keys = new ArrayList<>();
        Node cur = head.next[0];
        while (cur != null) {
            keys.add(cur.key);
            cur = cur.next[0];
        }
        return keys;
    }

    int levelOf(int key) {
        Node cur = head;
        for (int i = maxLevel - 1; i >= 0; i--) {
            while (cur.next[i] != null && cur.next[i].key < key) cur = cur.next[i];
        }
        Node candidate = cur.next[0];
        return candidate != null && candidate.key == key ? candidate.level : -1;
    }

    /** Returns null when every lane is exactly what a skip list with these towers should have. */
    String verifyLevels() {
        List<Integer> ground = keys();
        for (int i = 1; i < ground.size(); i++) {
            if (ground.get(i - 1) >= ground.get(i)) return "level 0 is not strictly increasing";
        }
        Node cur = head.next[0];
        while (cur != null) {
            if (cur.level < 1 || cur.level > maxLevel) return "node " + cur.key + " has level " + cur.level;
            if (cur.level < cur.baseLevel) return "node " + cur.key + " lost part of its random tower";
            cur = cur.next[0];
        }
        for (int level = 0; level < maxLevel; level++) {
            List<Integer> lane = new ArrayList<>();
            Node walk = head.next[level];
            while (walk != null) {
                lane.add(walk.key);
                walk = walk.next[level];
            }
            List<Integer> expected = new ArrayList<>();
            Node node = head.next[0];
            while (node != null) {
                if (node.level > level) expected.add(node.key);
                node = node.next[0];
            }
            if (!lane.equals(expected)) {
                return "lane " + level + " holds " + lane + " but should hold " + expected;
            }
            if (laneCount[level] != lane.size()) {
                return "lane counter " + laneCount[level] + " disagrees with lane length " + lane.size();
            }
        }
        return null;
    }

    Map<Integer, Integer> levelHistogram() {
        Map<Integer, Integer> histogram = new TreeMap<>();
        Node cur = head.next[0];
        while (cur != null) {
            histogram.merge(cur.level, 1, Integer::sum);
            cur = cur.next[0];
        }
        return histogram;
    }

    double averageSteps() { return searches == 0 ? 0 : (double) steps / searches; }
}

public class Solution {
    private static long state = 0x2246822519FEDCBAL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** Skewed workload: three of four queries target one of the `hot` keys (the locality pattern). */
    static int[] skewedWorkload(int length, int keySpace, int hot) {
        int[] workload = new int[length];
        for (int i = 0; i < length; i++) {
            workload[i] = (nextInt(4) != 0) ? nextInt(hot) : nextInt(keySpace);
        }
        return workload;
    }

    static SkipList build(int keySpace, boolean adaptive, long seed) {
        SkipList list = new SkipList(8, adaptive, 3, 2, adaptive ? 8 : 0, seed);
        for (int i = 0; i < keySpace; i++) list.insert(i);
        return list;
    }

    static List<Integer> intList(int count) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < count; i++) list.add(i);
        return list;
    }

    public static void main(String[] args) {
        // --- degenerate lists --------------------------------------------------
        SkipList empty = new SkipList(4, true, 3, 2, 2, 11L);
        check(!empty.contains(5), "search on an empty list misses");
        check(empty.keys().isEmpty() && empty.size == 0, "empty list has no keys");
        check(empty.verifyLevels() == null, "empty list lanes: " + empty.verifyLevels());
        check(empty.levelOf(5) == -1, "no level for a missing key");

        SkipList single = new SkipList(3, true, 1, 2, 2, 12L);
        check(single.insert(42), "insert into an empty list");
        check(!single.insert(42), "duplicate insert is rejected");
        check(single.size == 1, "duplicate insert does not grow the list");
        check(single.contains(42), "single key is found");
        check(!single.contains(41) && !single.contains(43), "neighbours of the single key are absent");
        check(single.verifyLevels() == null, "single node lanes: " + single.verifyLevels());
        check(!single.delete(41), "deleting a missing key reports false");
        check(single.delete(42), "deleting the only key works");
        check(single.keys().isEmpty() && single.verifyLevels() == null, "list is empty again");
        check(single.hot.isEmpty(), "the hot cache is repaired by deletion");

        // --- a flat list (maxLevel = 1) is still correct -----------------------
        SkipList flat = new SkipList(1, true, 2, 2, 4, 13L);
        for (int i = 0; i < 20; i++) flat.insert(i);
        check(flat.verifyLevels() == null, "maxLevel = 1 lanes: " + flat.verifyLevels());
        check(flat.keys().equals(intList(20)), "flat list keeps every key in order");
        check(flat.delete(0) && flat.delete(19), "flat list deletes head and tail");
        check(flat.promotions == 0, "a flat list has no lane to promote into");

        // --- rejection of invalid configuration --------------------------------
        boolean rejected = false;
        try {
            new SkipList(0, true, 2, 1, 1, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "maxLevel = 0 must be rejected");
        rejected = false;
        try {
            new SkipList(4, true, 0, 1, 1, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "promoteThreshold = 0 must be rejected");
        rejected = false;
        try {
            new SkipList(4, true, 2, -1, 1, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative maxLift must be rejected");

        // --- oracle workload against a TreeMap ---------------------------------
        SkipList oracleList = new SkipList(6, true, 4, 2, 6, 0xBEEF1234L);
        TreeMap<Integer, Integer> oracle = new TreeMap<>();
        int inserts = 0, removals = 0, hits = 0, misses = 0;
        for (int step = 0; step < 1500; step++) {
            int roll = nextInt(100);
            int key = nextInt(120);
            if (roll < 50) {
                if (oracleList.insert(key)) inserts++;
                oracle.merge(key, 1, Integer::sum);
            } else if (roll < 70) {
                if (oracleList.delete(key)) removals++;
                oracle.remove(key);
            } else {
                boolean expected = oracle.containsKey(key);
                boolean actual = oracleList.contains(key);
                check(expected == actual, "membership disagreement for key " + key + " at step " + step);
                if (actual) hits++;
                else misses++;
            }
            check(oracleList.verifyLevels() == null, "lane invariant at step " + step + ": "
                    + oracleList.verifyLevels());
        }
        check(oracleList.keys().equals(new ArrayList<>(oracle.keySet())), "final key set matches the oracle");
        check(oracleList.size == oracle.size(), "final size matches the oracle");
        check(inserts > 250 && removals > 120 && hits > 200 && misses > 100,
                "the oracle workload must exercise every operation: " + inserts + "/" + removals
                        + "/" + hits + "/" + misses);
        check(oracleList.promotions > 0, "the oracle workload must promote some hot keys");

        // --- promotion lifts a tower by exactly one lane per threshold ---------
        SkipList promoting = build(30, true, 0x5EEDL);
        int key = -1;
        for (int candidate = 0; candidate < 30; candidate++) {
            if (promoting.levelOf(candidate) <= 2) {
                key = candidate;
                break;
            }
        }
        check(key >= 0, "the test list must contain a low-tower key");
        int base = promoting.levelOf(key);
        for (int i = 0; i < 2; i++) promoting.contains(key);
        check(promoting.levelOf(key) == base, "two accesses are below the threshold of three");
        check(promoting.promotions == 0, "no promotion before the threshold is reached");
        promoting.contains(key);
        check(promoting.levelOf(key) == base + 1, "the third access promotes exactly one lane: "
                + base + " -> " + promoting.levelOf(key));
        check(promoting.promotions == 1, "exactly one promotion was performed");
        for (int i = 0; i < 200; i++) promoting.contains(key);
        check(promoting.levelOf(key) == base + 1,
                "while the key stays cached no further promotion happens, got " + promoting.levelOf(key));
        check(promoting.verifyLevels() == null, "promoted structure lanes: " + promoting.verifyLevels());

        // --- with a one-slot cache the key is evicted and keeps climbing -------
        SkipList climbing = new SkipList(8, true, 1, 3, 1, 0xABCDL);
        for (int i = 0; i < 30; i++) climbing.insert(i);
        int climber = 5;
        int climbBase = climbing.levelOf(climber);
        for (int round = 0; round < 60; round++) {
            climbing.contains(climber);
            climbing.contains(9);                    // evicts the climber from the size-one cache
        }
        check(climbing.levelOf(climber) == Math.min(climbBase + 3, 8),
                "repeated eviction lets the tower climb to baseLevel + maxLift, got "
                        + climbing.levelOf(climber) + " from base " + climbBase);
        check(climbing.promotions >= 3, "the climber must be promoted repeatedly, got " + climbing.promotions);
        check(climbing.verifyLevels() == null, "climbing structure lanes: " + climbing.verifyLevels());

        // --- the adaptive list answers exactly like the plain one -------------
        int[] workload = skewedWorkload(6000, 300, 6);
        SkipList adaptive = build(300, true, 0xC0FFEEL);
        SkipList plain = build(300, false, 0xC0FFEEL);
        int mismatches = 0;
        for (int query : workload) {
            if (adaptive.contains(query) != plain.contains(query)) mismatches++;
        }
        check(mismatches == 0, "adaptive and plain lists must agree, mismatches=" + mismatches);
        check(adaptive.verifyLevels() == null, "adaptive lanes: " + adaptive.verifyLevels());
        check(plain.verifyLevels() == null, "plain lanes: " + plain.verifyLevels());

        // --- locality: the heuristics must actually pay off --------------------
        check(adaptive.promotions > 0, "the adaptive run must promote hot keys");
        check(plain.promotions == 0, "the plain run must never promote");
        check(adaptive.cacheHits * 3 > adaptive.searches * 2,
                "at least two thirds of the skewed queries should hit the hot cache, got "
                        + adaptive.cacheHits + " of " + adaptive.searches);
        check(adaptive.hot.size() <= 8, "the hot cache stays bounded, size " + adaptive.hot.size());
        check(plain.hot.isEmpty(), "the plain list keeps no cache");
        check(adaptive.steps * 2 < plain.steps,
                "adaptive probe steps " + adaptive.steps + " should be far below plain " + plain.steps);
        check(adaptive.averageSteps() < plain.averageSteps(),
                "adaptive average steps " + adaptive.averageSteps() + " vs plain " + plain.averageSteps());

        // --- determinism --------------------------------------------------------
        SkipList rerun = build(300, true, 0xC0FFEEL);
        for (int query : workload) rerun.contains(query);
        check(rerun.levelHistogram().equals(adaptive.levelHistogram()),
                "the same seed and workload must give the same lanes");
        check(rerun.promotions == adaptive.promotions, "the same heuristics must fire equally often");

        // --- promotions stay inside the density budget -------------------------
        for (int level = 0; level < 8; level++) {
            long ideal = Math.max(2L, adaptive.size >> level);
            check(adaptive.laneCount[level] <= ideal + ideal / 2 + 1,
                    "lane " + level + " holds " + adaptive.laneCount[level] + " nodes, ideal " + ideal);
        }
        int tallTowers = 0;
        for (Map.Entry<Integer, Integer> entry : adaptive.levelHistogram().entrySet()) {
            if (entry.getKey() > 2) tallTowers += entry.getValue();
        }
        check(tallTowers >= 3, "the hot keys must end up in tall towers, got " + tallTowers);

        // --- deletion still removes the tower from every lane and the cache ----
        for (int hotKey : new int[]{0, 1, 2, 3, 4, 5}) {
            if (adaptive.contains(hotKey)) {
                check(adaptive.delete(hotKey), "hot key " + hotKey + " must be deletable");
                check(!adaptive.contains(hotKey), "deleted key " + hotKey + " is gone");
                check(!adaptive.hot.containsKey(hotKey), "the cache forgot the deleted key");
                check(adaptive.verifyLevels() == null, "lanes after deleting " + hotKey + ": "
                        + adaptive.verifyLevels());
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
