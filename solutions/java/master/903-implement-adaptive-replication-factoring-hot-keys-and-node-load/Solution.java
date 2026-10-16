/** Task 903 - Implement adaptive replication factoring hot keys and node load.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A consistent-hashing ring places each key on a primary node, and a key's replication factor (RF) decides
 * how many distinct nodes carry a copy of it. Two forces fight each other:
 *
 *   hot keys  -- a few keys absorb most of the reads, so raising their RF spreads the request stream over
 *                more nodes (more memory, but the hottest node stops being a bottleneck);
 *   node load -- a replica may only land on a node with spare capacity, otherwise replication just relocates
 *                the hotspot; when the whole cluster is undersized the planner must say so instead of
 *                silently dropping a copy.
 *
 * The loop below measures a Zipf-ish request stream per key, derives RF from two heat thresholds
 * (hot -> MAX_RF, warm -> 2, cold -> 1), places replicas with capacity awareness, and re-measures balance.
 *
 * Simplifications: 24 keys, 8 nodes, 32 virtual nodes per node, 4000 simulated requests, and counts divided
 * evenly among a key's replicas (reads are assumed to spread uniformly over copies). Invariants verified:
 * replicas are distinct live nodes, RF is clamped by the live node count, a generous capacity is never
 * exceeded, the adapted placement beats the single-copy baseline on max-node-load and on max/mean imbalance,
 * the reported load vector matches an independent recomputation from the placement, a lost node never
 * appears in a repaired replica set, and the whole loop replays identically for a fixed seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness. */
final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** Consistent-hash ring: sorted virtual-node positions, each owned by one node. */
final class HashRing {
    private static final int VNODES = 32;
    private final long[] positions;
    private final int[] owners;

    HashRing(List<Integer> nodes) {
        TreeMap<Long, Integer> ring = new TreeMap<>();
        for (int node : nodes) {
            for (int v = 0; v < VNODES; v++) ring.put(position(node, v), node);
        }
        positions = new long[ring.size()];
        owners = new int[ring.size()];
        int i = 0;
        for (Map.Entry<Long, Integer> e : ring.entrySet()) {
            positions[i] = e.getKey();
            owners[i] = e.getValue();
            i++;
        }
    }

    private static long position(int node, int vnode) {
        long h = (0x9E3779B97F4A7C15L * (node + 1)) ^ (0xBF58476D1CE4E5B9L * (vnode + 1));
        h ^= h >>> 30;
        h *= 0xBF58476D1CE4E5B9L;
        h ^= h >>> 27;
        return h ^ (h >>> 31);
    }

    static long hashKey(String key) {
        long h = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < key.length(); i++) h = (h ^ key.charAt(i)) * 0xBF58476D1CE4E5B9L;
        return h ^ (h >>> 31);
    }

    /** Distinct owners in clockwise order starting at the key's own position. */
    List<Integer> clockwise(String key) {
        List<Integer> order = new ArrayList<>();
        int start = lowerBound(hashKey(key));
        for (int step = 0; step < positions.length; step++) {
            int owner = owners[(start + step) % positions.length];
            if (!order.contains(owner)) order.add(owner);
        }
        return order;
    }

    /** The first `rf` distinct nodes clockwise - the classic primary-plus-succeeding-replicas rule. */
    List<Integer> replicas(String key, int rf) {
        List<Integer> order = clockwise(key);
        return new ArrayList<>(order.subList(0, Math.min(rf, order.size())));
    }

    private int lowerBound(long position) {
        int lo = 0;
        int hi = positions.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (positions[mid] < position) lo = mid + 1;
            else hi = mid;
        }
        return lo % positions.length;
    }
}

/** Result of a placement round: which nodes hold which key, and what that costs each node. */
final class Placement {
    final Map<String, List<Integer>> replicas = new LinkedHashMap<>();
    final Map<Integer, Long> load = new TreeMap<>();
    int saturatedPlan;
    long maxLoad;

    void observe(int node, long requests) {
        load.merge(node, requests, Long::sum);
        maxLoad = Math.max(maxLoad, load.get(node));
    }

    double imbalance(int nodeCount) {
        if (load.isEmpty()) return 0;
        long total = 0;
        for (long l : load.values()) total += l;
        return maxLoad / (total / (double) nodeCount);
    }
}

public class Solution {
    static final int KEY_COUNT = 24;
    static final int REQUEST_COUNT = 4000;
    static final int HOT_THRESHOLD = 300;
    static final int WARM_THRESHOLD = 120;
    static final int MAX_RF = 3;
    static final int NODE_COUNT = 8;

    public static void main(String[] args) {
        List<Integer> nodes = new ArrayList<>();
        for (int i = 0; i < NODE_COUNT; i++) nodes.add(i);
        HashRing ring = new HashRing(nodes);
        List<String> keys = new ArrayList<>();
        for (int i = 0; i < KEY_COUNT; i++) keys.add("key-" + (i / 10) + (i % 10));

        // ---- 1. the simulated workload ---------------------------------------------------------------
        Map<String, Long> frequency = measure(0x1234ABCDL, keys);
        long total = 0;
        for (long f : frequency.values()) total += f;
        String hottest = hottestKey(frequency);
        String coldest = coldestKey(frequency);
        check(total == REQUEST_COUNT, "every request must be attributed to a key, got " + total);
        check(frequency.get(hottest) > HOT_THRESHOLD,
                "the workload must have a hot key, hottest " + frequency.get(hottest));
        check(frequency.get(coldest) < WARM_THRESHOLD,
                "the workload must have a cold tail, coldest " + frequency.get(coldest));

        // ---- 2. baseline: one copy per key ------------------------------------------------------------
        Map<String, Integer> single = new LinkedHashMap<>();
        for (String key : keys) single.put(key, 1);
        Placement baseline = place(ring, keys, frequency, single, nodes, Long.MAX_VALUE);
        check(baseline.saturatedPlan == 0, "an unconstrained plan cannot be saturated");
        check(baseline.maxLoad > REQUEST_COUNT / NODE_COUNT,
                "the single-copy baseline must be unbalanced, hottest node " + baseline.maxLoad);

        // ---- 3. adapt RF from the measured heat -------------------------------------------------------
        Map<String, Integer> rf = adapt(frequency, nodes.size());
        check(rf.get(hottest) == MAX_RF, "the hottest key must reach MAX_RF = " + MAX_RF);
        check(rf.get(coldest) == 1, "the coldest key must stay at RF 1, got " + rf.get(coldest));
        check(countWith(rf, 1) > 0 && countWith(rf, 2) > 0 && countWith(rf, MAX_RF) > 0,
                "the adapted policy must use all three RF levels, got 1/" + countWith(rf, 1) + " 2/"
                        + countWith(rf, 2) + " " + MAX_RF + "/" + countWith(rf, MAX_RF));

        // ---- 4. capacity-aware placement -------------------------------------------------------------
        long generous = 2L * REQUEST_COUNT / NODE_COUNT;
        Placement adapted = place(ring, keys, frequency, rf, nodes, generous);
        check(adapted.saturatedPlan == 0, "with capacity " + generous + " the plan must not be saturated");
        for (Map.Entry<Integer, Long> e : adapted.load.entrySet()) {
            check(e.getValue() <= generous, "node " + e.getKey() + " exceeded capacity: " + e.getValue());
        }
        check(adapted.maxLoad < baseline.maxLoad,
                "adaptation must lower the hottest node: " + baseline.maxLoad + " -> " + adapted.maxLoad);
        check(adapted.imbalance(NODE_COUNT) < baseline.imbalance(NODE_COUNT),
                "adaptation must lower max/mean imbalance: " + round(baseline.imbalance(NODE_COUNT)) + " -> "
                        + round(adapted.imbalance(NODE_COUNT)));

        // ---- 5. replica-set invariants ----------------------------------------------------------------
        for (String key : keys) {
            List<Integer> replicas = adapted.replicas.get(key);
            check(replicas.size() == rf.get(key), key + " must have exactly " + rf.get(key) + " replicas");
            check(new HashSet<>(replicas).size() == replicas.size(), key + " replicas must be distinct nodes");
            for (int node : replicas) check(nodes.contains(node), key + " replica must be a live node");
        }
        check(adapted.replicas.get(hottest).size() > adapted.replicas.get(coldest).size(),
                "the hot key must be replicated more widely than the cold key");

        // ---- 6. independent oracle: rebuild the load vector from the placement ------------------------
        Map<Integer, Long> oracle = new TreeMap<>();
        for (String key : keys) {
            List<Integer> replicas = adapted.replicas.get(key);
            long requests = frequency.get(key);
            long share = requests / replicas.size();
            long remainder = requests % replicas.size();
            for (int i = 0; i < replicas.size(); i++) {
                oracle.merge(replicas.get(i), share + (i < remainder ? 1 : 0), Long::sum);
            }
        }
        check(oracle.equals(adapted.load), "the reported load must match a recomputation from the placement");
        long oracleTotal = 0;
        for (long l : oracle.values()) oracleTotal += l;
        check(oracleTotal == REQUEST_COUNT, "replicated requests must still add up to " + REQUEST_COUNT);

        // ---- 7. undersized cluster: reported, not hidden ----------------------------------------------
        long tight = REQUEST_COUNT / NODE_COUNT / 4;
        Placement squeezed = place(ring, keys, frequency, rf, nodes, tight);
        check(squeezed.saturatedPlan > 0, "capacity " + tight + " is below the mean, so the plan must saturate");
        for (String key : keys) {
            check(squeezed.replicas.get(key).size() == rf.get(key),
                    "a saturated plan must still keep every replica of " + key);
        }
        check(squeezed.saturatedPlan <= keys.size(), "saturation is recorded per key, so it cannot exceed "
                + keys.size() + ", got " + squeezed.saturatedPlan);
        Map<Integer, Long> squeezedOracle = new TreeMap<>();
        for (String key : keys) {
            List<Integer> replicas = squeezed.replicas.get(key);
            long share = frequency.get(key) / replicas.size();
            long rest = frequency.get(key) % replicas.size();
            for (int i = 0; i < replicas.size(); i++) squeezedOracle.merge(replicas.get(i), share
                    + (i < rest ? 1 : 0), Long::sum);
        }
        check(squeezedOracle.equals(squeezed.load),
                "even a saturated plan must account for every request: " + squeezed.load + " vs "
                        + squeezedOracle);

        // ---- 8. boundary: one-node cluster ------------------------------------------------------------
        HashRing solo = new HashRing(List.of(7));
        check(solo.replicas(hottest, MAX_RF).size() == 1, "a single node can only hold one replica");
        check(adapt(frequency, 1).get(hottest) == 1, "RF must be clamped by the live node count");
        Placement soloPlan = place(solo, keys, frequency, adapt(frequency, 1), List.of(7), Long.MAX_VALUE);
        check(soloPlan.load.size() == 1 && soloPlan.maxLoad == REQUEST_COUNT,
                "one node absorbs the whole stream, got " + soloPlan.maxLoad);

        // ---- 9. node loss: repaired replicas avoid the dead node --------------------------------------
        List<Integer> live = new ArrayList<>(nodes);
        int dead = adapted.replicas.get(hottest).get(0);
        live.remove(Integer.valueOf(dead));
        HashRing shrunk = new HashRing(live);
        Map<String, Integer> shrunkRf = adapt(frequency, live.size());
        Placement repaired = place(shrunk, keys, frequency, shrunkRf, live, Long.MAX_VALUE);
        for (String key : keys) {
            check(!repaired.replicas.get(key).contains(dead), key + " still points at the dead node");
            check(repaired.replicas.get(key).size() == shrunkRf.get(key),
                    key + " lost its RF during the repair");
        }
        check(repaired.maxLoad > adapted.maxLoad,
                "repairing onto 7 nodes must cost load: " + adapted.maxLoad + " -> " + repaired.maxLoad);

        // ---- 10. determinism --------------------------------------------------------------------------
        Map<String, Long> replay = measure(0x1234ABCDL, keys);
        check(replay.equals(frequency), "the same seed must replay the same frequencies");
        Placement replayPlacement = place(new HashRing(nodes), keys, replay, adapt(replay, NODE_COUNT), nodes,
                generous);
        check(replayPlacement.replicas.equals(adapted.replicas),
                "the adapted placement must be reproducible exactly");
        check(replayPlacement.load.equals(adapted.load), "and so must the load vector");

        System.out.println("keys " + KEY_COUNT + ", requests " + REQUEST_COUNT + ", hottest " + hottest + "="
                + frequency.get(hottest) + ", coldest " + coldest + "=" + frequency.get(coldest));
        System.out.println("max node load " + baseline.maxLoad + " -> " + adapted.maxLoad + " (capacity "
                + generous + "); imbalance " + round(baseline.imbalance(NODE_COUNT)) + " -> "
                + round(adapted.imbalance(NODE_COUNT)));
        System.out.println("RF levels 1/2/" + MAX_RF + " = " + countWith(rf, 1) + "/" + countWith(rf, 2) + "/"
                + countWith(rf, MAX_RF) + "; saturated keys at capacity " + tight + " = "
                + squeezed.saturatedPlan + "; after losing node " + dead + " max load " + repaired.maxLoad);
        System.out.println("PASS");
    }

    /** Deterministic Zipf sampler: rank r has weight 1/(r+1); ranks are drawn by inverse-CDF. */
    static Map<String, Long> measure(long seed, List<String> keys) {
        SplitMix64 rng = new SplitMix64(seed);
        Map<String, Long> counts = new LinkedHashMap<>();
        for (String key : keys) counts.put(key, 0L);
        double[] cumulative = new double[keys.size()];
        double running = 0;
        for (int r = 0; r < keys.size(); r++) {
            running += 1.0 / (r + 1);
            cumulative[r] = running;
        }
        for (int i = 0; i < REQUEST_COUNT; i++) {
            double u = rng.nextDouble() * running;
            int rank = 0;
            while (rank < cumulative.length - 1 && cumulative[rank] < u) rank++;
            String key = keys.get(rank);
            counts.put(key, counts.get(key) + 1);
        }
        return counts;
    }

    static Map<String, Integer> adapt(Map<String, Long> frequency, int liveNodes) {
        Map<String, Integer> rf = new LinkedHashMap<>();
        for (Map.Entry<String, Long> e : frequency.entrySet()) {
            int level = e.getValue() >= HOT_THRESHOLD ? MAX_RF : (e.getValue() >= WARM_THRESHOLD ? 2 : 1);
            rf.put(e.getKey(), Math.max(1, Math.min(level, liveNodes)));
        }
        return rf;
    }

    /**
     * Capacity-aware placement. Each key walks its ring order clockwise and takes the first `rf` nodes that
     * still have room for its share; if too few nodes have room the planner falls back to the globally
     * least-loaded live nodes, keeps the replica count, and records the saturation instead of dropping data.
     */
    static Placement place(HashRing ring, List<String> keys, Map<String, Long> frequency,
            Map<String, Integer> rf, List<Integer> live, long capacity) {
        Placement placement = new Placement();
        for (int node : live) placement.load.put(node, 0L);
        for (String key : keys) {
            int wanted = Math.min(rf.get(key), live.size());
            long requests = frequency.get(key);
            long share = requests / wanted;
            List<Integer> chosen = new ArrayList<>();
            for (int node : ring.clockwise(key)) {
                if (chosen.size() == wanted) break;
                if (placement.load.get(node) + share <= capacity) chosen.add(node);
            }
            if (chosen.size() < wanted) {
                List<Integer> byLoad = new ArrayList<>(live);
                byLoad.sort((a, b) -> {
                    int cmp = Long.compare(placement.load.get(a), placement.load.get(b));
                    return cmp != 0 ? cmp : Integer.compare(a, b);
                });
                for (int node : byLoad) {
                    if (chosen.size() == wanted) break;
                    if (!chosen.contains(node)) chosen.add(node);
                }
                placement.saturatedPlan++;
            }
            long remainder = requests % wanted;
            for (int i = 0; i < chosen.size(); i++) {
                placement.observe(chosen.get(i), share + (i < remainder ? 1 : 0));
            }
            placement.replicas.put(key, chosen);
        }
        return placement;
    }

    static String hottestKey(Map<String, Long> frequency) { return extremeKey(frequency, true); }

    static String coldestKey(Map<String, Long> frequency) { return extremeKey(frequency, false); }

    static String extremeKey(Map<String, Long> frequency, boolean max) {
        String best = null;
        for (Map.Entry<String, Long> e : frequency.entrySet()) {
            if (best == null || (max ? e.getValue() > frequency.get(best) : e.getValue() < frequency.get(best))) {
                best = e.getKey();
            }
        }
        return best;
    }

    static int countWith(Map<String, Integer> rf, int level) {
        int count = 0;
        for (int v : rf.values()) if (v == level) count++;
        return count;
    }

    static double round(double value) { return Math.round(value * 1000) / 1000.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
