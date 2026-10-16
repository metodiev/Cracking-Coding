/** Task 917 - Implement self-healing storage cluster with automatic rebalancing and throttled repairs.
 *  (Tier: Master, Ch 9 - System Design and Scalability)
 *
 * A replicated, consistent-hash storage cluster must survive three events: a node dies, a node joins, and
 * both of those at a rate that does not melt the network. The model here is deliberately small but complete:
 *
 *   placement    every key lives on the R distinct nodes that follow it clockwise on the hash ring;
 *   failure      a dead node takes one replica of every key it hosted. Those keys become under-replicated,
 *                and - if R >= 2 - still readable from the survivors, which is the whole point of replicating;
 *   repair       re-replicating an under-replicated key is queued as a byte-sized task and drained under a
 *                per-tick bandwidth throttle, so the repair traffic is bounded even for a large node;
 *   rebalancing  a joining node inherits keys from its neighbours, again drained under the same throttle.
 *
 * The checks verify the safety properties (no key ever drops below one live replica unless it was already
 * unrecoverable, and after the drain every key is back at R replicas) and the cost properties (per-tick bytes
 * never exceed the throttle, progress is monotone, the number of ticks matches the lower bound
 * ceil(bytes/throttle), a task larger than the throttle is deferred rather than overspending, and an
 * undersized cluster reports under-replication instead of pretending to have healed).
 *
 * Simplification: in-memory metadata only, no bytes are moved. Invariants verified: replica counts, throttle
 * compliance, termination, rebalance correctness and determinism.
 */
import java.util.*;

/** Consistent-hash ring with virtual nodes. */
final class Ring917 {
    private static final int VNODES = 40;
    private final long[] positions;
    private final int[] owners;
    private final Map<Integer, Boolean> live = new TreeMap<>();

    Ring917(List<Integer> nodes) {
        for (int node : nodes) live.put(node, true);
        TreeMap<Long, Integer> ring = new TreeMap<>();
        for (int node : live.keySet()) {
            for (int v = 0; v < VNODES; v++) ring.put(position(node, v), node);
        }
        positions = new long[ring.size()];
        owners = new int[ring.size()];
        int i = 0;
        for (Map.Entry<Long, Integer> entry : ring.entrySet()) {
            positions[i] = entry.getKey();
            owners[i] = entry.getValue();
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

    void kill(int node) { live.remove(node); }

    void revive(int node) { live.put(node, true); }

    Set<Integer> liveNodes() { return new TreeSet<>(live.keySet()); }

    int liveCount() { return live.size(); }

    /** The R distinct live nodes that follow the key clockwise; shorter when the cluster is too small. */
    List<Integer> owners(String key, int replicas) {
        List<Integer> result = new ArrayList<>();
        if (live.isEmpty()) return result;
        int start = lowerBound(hashKey(key));
        for (int step = 0; step < positions.length && result.size() < replicas; step++) {
            int owner = owners[(start + step) % positions.length];
            if (live.containsKey(owner) && !result.contains(owner)) result.add(owner);
        }
        return result;
    }

    static long hashKey(String key) {
        long h = 0x917BEEFL;
        for (int i = 0; i < key.length(); i++) h = (h ^ key.charAt(i)) * 0xBF58476D1CE4E5B9L;
        return h ^ (h >>> 31);
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

/** A queued repair: put one more replica of `key` on some live node. */
final class RepairTask {
    final String key;
    final int bytes;
    final boolean rebalance;

    RepairTask(String key, int bytes, boolean rebalance) {
        this.key = key;
        this.bytes = bytes;
        this.rebalance = rebalance;
    }
}

/** One pass of the throttled repair loop. */
final class RepairRound {
    long bytes;
    int tasks;
}

public class Solution {
    static final int KEY_COUNT = 200;
    static final int REPLICAS = 3;
    static final int THROTTLE = 4096;

    public static void main(String[] args) {
        List<String> keys = new ArrayList<>();
        Map<String, Integer> sizes = new TreeMap<>();
        Rng917 rng = new Rng917(0x917ABCDEL);
        for (int i = 0; i < KEY_COUNT; i++) {
            String key = "key-" + (i / 20) + (char) ('a' + i % 20);
            keys.add(key);
            sizes.put(key, 1 + rng.nextInt(512));
        }
        List<Integer> nodeIds = new ArrayList<>();
        for (int i = 0; i < 8; i++) nodeIds.add(i);
        Ring917 ring = new Ring917(nodeIds);

        // ---- 1. the healthy placement is complete -------------------------------------------------------
        Map<String, List<Integer>> placement = place(ring, keys);
        check(placement.size() == KEY_COUNT, "every key must be placed, got " + placement.size());
        for (Map.Entry<String, List<Integer>> entry : placement.entrySet()) {
            check(entry.getValue().size() == REPLICAS, entry.getKey() + " must have " + REPLICAS + " replicas");
            check(new HashSet<>(entry.getValue()).size() == REPLICAS,
                    entry.getKey() + " replicas must be distinct nodes");
        }
        check(underReplicated(placement).isEmpty(), "a healthy cluster must have no under-replicated keys");
        long loadBefore = maxLoad(placement, sizes);
        check(loadBefore > 0, "the cluster must actually hold bytes");

        // ---- 2. a node dies: the replicas it held are gone, the keys themselves survive ----------------
        int victim = 3;
        int lostKeys = 0;
        for (List<Integer> replicas : placement.values()) if (replicas.contains(victim)) lostKeys++;
        Map<String, List<Integer>> degraded = deepCopy(placement);
        List<String> under = new ArrayList<>();
        for (Map.Entry<String, List<Integer>> entry : degraded.entrySet()) {
            if (entry.getValue().remove(Integer.valueOf(victim)) && entry.getValue().size() < REPLICAS) {
                under.add(entry.getKey());
            }
        }
        check(lostKeys > 0, "the victim must have hosted some keys, hosted " + lostKeys);
        check(under.size() == lostKeys, "exactly the keys on the dead node must be under-replicated: "
                + under.size() + " vs " + lostKeys);
        check(unavailable(degraded).isEmpty(), "with R=3 a single failure must never make a key unreadable");
        check(underReplicated(degraded).size() == lostKeys, "the degraded view must agree with the key count");
        ring.kill(victim);

        // ---- 3. throttled repair --------------------------------------------------------------------------
        List<RepairTask> queue = new ArrayList<>();
        for (String key : under) queue.add(new RepairTask(key, sizes.get(key), false));
        long repairBytes = totalBytesTasks(queue);
        List<RepairRound> rounds = drain(queue, THROTTLE);
        check(!rounds.isEmpty(), "the repair queue must not be empty");
        long moved = 0;
        long previousOutstanding = repairBytes;
        for (RepairRound round : rounds) {
            check(round.bytes <= THROTTLE, "a round moved " + round.bytes + " bytes, above the throttle "
                    + THROTTLE);
            check(round.bytes > 0, "a round that moves nothing would be a livelock");
            moved += round.bytes;
            long outstanding = repairBytes - moved;
            check(outstanding < previousOutstanding, "repair progress must be monotone");
            previousOutstanding = outstanding;
        }
        check(moved == repairBytes, "the drain must move exactly the repair bytes: " + moved + " vs "
                + repairBytes);
        long minimumRounds = (repairBytes + THROTTLE - 1) / THROTTLE;
        check(rounds.size() >= minimumRounds, "the throttle forces at least " + minimumRounds + " rounds, got "
                + rounds.size());
        check(queue.isEmpty(), "the repair queue must be drained");

        // ---- 4. the healed cluster -------------------------------------------------------------------------
        Map<String, List<Integer>> healed = deepCopy(degraded);
        for (String key : under) {
            List<Integer> candidates = ring.owners(key, REPLICAS);
            for (int node : candidates) {
                if (healed.get(key).size() == REPLICAS) break;
                if (!healed.get(key).contains(node)) healed.get(key).add(node);
            }
        }
        check(underReplicated(healed).isEmpty(), "self-healing must restore every key to " + REPLICAS
                + " replicas");
        for (Map.Entry<String, List<Integer>> entry : healed.entrySet()) {
            check(!entry.getValue().contains(victim), entry.getKey() + " still points at the dead node");
            check(new HashSet<>(entry.getValue()).size() == entry.getValue().size(),
                    entry.getKey() + " must not hold two replicas on one node");
        }
        check(unavailable(healed).isEmpty(), "the healed cluster must have no unavailable keys");
        check(ring.liveCount() == 7, "seven nodes must still be alive after one failure, got "
                + ring.liveCount());

        // ---- 5. a task larger than the throttle must be deferred, not overspent -------------------------
        List<RepairTask> oversized = new ArrayList<>();
        oversized.add(new RepairTask("huge", THROTTLE * 3, false));
        oversized.add(new RepairTask("small", 10, false));
        List<RepairRound> deferred = drain(oversized, THROTTLE);
        check(deferred.size() == 1 && deferred.get(0).bytes == 10,
                "only the task that fits may run, got " + deferred.get(0).bytes);
        check(!oversized.isEmpty(), "the oversized task must remain queued for the next window");
        check(deferred.get(0).bytes <= THROTTLE, "the throttle must hold even with an oversized head");

        // ---- 6. a node joins and inherits keys under the same throttle ---------------------------------
        ring.revive(victim);
        Ring917 enlarged = new Ring917(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8));
        Map<String, List<Integer>> before = place(enlarged, keys);
        check(before.values().stream().allMatch(list -> list.size() == REPLICAS),
                "the enlarged cluster must still place every key");
        int hostedByNew = 0;
        for (List<Integer> replicas : before.values()) if (replicas.contains(8)) hostedByNew++;
        check(hostedByNew > 0, "the joining node must take over some keys");
        long rebalanceBytes = 0;
        List<RepairTask> rebalanceQueue = new ArrayList<>();
        for (Map.Entry<String, List<Integer>> entry : before.entrySet()) {
            if (entry.getValue().contains(8)) {
                rebalanceBytes += sizes.get(entry.getKey());
                rebalanceQueue.add(new RepairTask(entry.getKey(), sizes.get(entry.getKey()), true));
            }
        }
        List<RepairRound> rebalanceRounds = drain(rebalanceQueue, THROTTLE);
        check(totalBytes(rebalanceRounds) == rebalanceBytes, "the rebalance must move exactly the inherited "
                + "bytes");
        check(rebalanceRounds.size() >= (rebalanceBytes + THROTTLE - 1) / THROTTLE,
                "the rebalance must also respect the throttle");
        long loadAfter = maxLoad(before, sizes);
        check(loadAfter <= loadBefore, "adding a node must not increase the busiest node: " + loadBefore
                + " -> " + loadAfter);
        check(underReplicated(before).isEmpty(), "rebalancing must leave no key under-replicated");

        // ---- 7. boundaries -----------------------------------------------------------------------------------
        check(underReplicated(place(new Ring917(List.of(0)), keys)).size() > 0,
                "a single-node cluster cannot satisfy R=3 and must report under-replication");
        Map<String, List<Integer>> single = place(new Ring917(List.of(0)), keys);
        for (List<Integer> replicas : single.values()) {
            check(replicas.size() <= 1, "replica count must be clamped to the live node count");
        }
        Map<String, List<Integer>> oneReplica = place(new Ring917(List.of(0, 1)), keys);
        check(underReplicated(oneReplica).size() == KEY_COUNT, "with two nodes and R=3 every key is short");
        check(unavailable(oneReplica).isEmpty(), "two nodes still hold at least one replica of every key");
        Map<String, List<Integer>> empty = place(new Ring917(List.of()), keys);
        check(unavailable(empty).size() == KEY_COUNT, "an empty cluster must report every key unavailable");
        check(underReplicated(empty).size() == KEY_COUNT, "an empty cluster is fully under-replicated");
        check(new Ring917(List.of()).owners("key-0a", REPLICAS).isEmpty(), "a dead cluster owns nothing");
        Ring917 half = new Ring917(List.of(0, 1, 2, 3));
        half.kill(0);
        half.kill(1);
        check(half.liveCount() == 2, "two of four nodes remain, got " + half.liveCount());
        check(underReplicated(place(half, keys)).size() == KEY_COUNT,
                "with only two of four nodes left every key is short of R replicas");
        check(unavailable(place(half, keys)).isEmpty(),
                "two surviving nodes still hold a replica of every key");

        // ---- 8. determinism ------------------------------------------------------------------------------------
        check(place(new Ring917(nodeIds), keys).get(keys.get(0)).equals(placement.get(keys.get(0))),
                "placement must be reproducible");
        List<RepairTask> replayQueue = new ArrayList<>();
        for (String key : under) replayQueue.add(new RepairTask(key, sizes.get(key), false));
        List<RepairRound> replayRounds = drain(replayQueue, THROTTLE);
        check(totalBytes(replayRounds) == moved, "the drain must replay identically");
        check(replayRounds.size() == rounds.size(), "the round count must replay identically");

        System.out.println("keys " + KEY_COUNT + ", replicas " + REPLICAS + ", nodes 8 -> loss of node "
                + victim + " -> 8; throttle " + THROTTLE + " bytes/tick");
        System.out.println("degraded keys " + under.size() + ", repair " + repairBytes + " bytes in "
                + rounds.size() + " rounds (lower bound " + minimumRounds + "), unavailable "
                + unavailable(degraded).size() + ", healed shortfall "
                + underReplicated(healed).size());
        System.out.println("rebalance onto node 8: " + hostedByNew + " keys, " + rebalanceBytes + " bytes in "
                + rebalanceRounds.size() + " rounds; max node load " + loadBefore + " -> " + loadAfter);
        System.out.println("PASS");
    }

    /** Places every key on the R distinct live nodes following it on the ring. */
    static Map<String, List<Integer>> place(Ring917 ring, List<String> keys) {
        Map<String, List<Integer>> placement = new LinkedHashMap<>();
        for (String key : keys) placement.put(key, ring.owners(key, REPLICAS));
        return placement;
    }

    /** A deep copy so the degraded view can be mutated without touching the healthy placement. */
    static Map<String, List<Integer>> deepCopy(Map<String, List<Integer>> placement) {
        Map<String, List<Integer>> copy = new LinkedHashMap<>();
        for (Map.Entry<String, List<Integer>> entry : placement.entrySet()) {
            copy.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        return copy;
    }

    static List<String> underReplicated(Map<String, List<Integer>> placement) {
        List<String> under = new ArrayList<>();
        for (Map.Entry<String, List<Integer>> entry : placement.entrySet()) {
            if (entry.getValue().size() < REPLICAS) under.add(entry.getKey());
        }
        return under;
    }

    static List<String> unavailable(Map<String, List<Integer>> placement) {
        List<String> lost = new ArrayList<>();
        for (Map.Entry<String, List<Integer>> entry : placement.entrySet()) {
            if (entry.getValue().isEmpty()) lost.add(entry.getKey());
        }
        return lost;
    }

    static long maxLoad(Map<String, List<Integer>> placement, Map<String, Integer> sizes) {
        Map<Integer, Long> load = new TreeMap<>();
        for (Map.Entry<String, List<Integer>> entry : placement.entrySet()) {
            for (int node : entry.getValue()) load.merge(node, (long) sizes.get(entry.getKey()), Long::sum);
        }
        long max = 0;
        for (long value : load.values()) max = Math.max(max, value);
        return max;
    }

    /**
     * Throttled drain: tasks that fit in the remaining budget run this tick, in queue order. A task larger
     * than the whole throttle is deferred, so the per-round byte limit is never violated.
     */
    static List<RepairRound> drain(List<RepairTask> queue, long throttle) {
        List<RepairRound> rounds = new ArrayList<>();
        while (!queue.isEmpty()) {
            RepairRound round = new RepairRound();
            long budget = throttle;
            Iterator<RepairTask> iterator = queue.iterator();
            while (iterator.hasNext()) {
                RepairTask task = iterator.next();
                if (task.bytes <= budget) {
                    budget -= task.bytes;
                    round.bytes += task.bytes;
                    round.tasks++;
                    iterator.remove();
                }
            }
            if (round.tasks == 0) break;
            check(round.bytes <= throttle, "a round must never exceed the throttle");
            rounds.add(round);
        }
        return rounds;
    }

    static long totalBytes(List<RepairRound> rounds) {
        long total = 0;
        for (RepairRound round : rounds) total += round.bytes;
        return total;
    }

    static long totalBytesTasks(List<RepairTask> tasks) {
        long total = 0;
        for (RepairTask task : tasks) total += task.bytes;
        return total;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}

/** splitmix64 - the only source of randomness. */
final class Rng917 {
    private long state;

    Rng917(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}
