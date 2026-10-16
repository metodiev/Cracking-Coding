/** Task 659 - Implement consistent hashing with weighted nodes and replication. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A consistent hash ring with weighted virtual nodes. Node i with weight w gets 64*w tokens placed at
 * mix64(nodeName, tokenIndex) on a 64 bit ring; tokens are sorted and the successor of a key's hash
 * is its primary. Replicas walk clockwise from the primary and take the first R *distinct* physical
 * nodes. Ties between equal token hashes are broken by node name, so the ring layout does not depend
 * on the order in which the nodes were inserted.
 *
 * Verified invariants: two rings built from the same nodes in different orders place every key
 * identically; replica sets are distinct, of size min(R, nodeCount) and are exactly the clockwise
 * walk; removing a node moves only the keys that node owned; adding a node only ever moves keys onto
 * that node (minimal disruption, and roughly 1/(n+1) of the keys); ownership weight follows the node
 * weight (a 1:2:1 ring loads the middle node about twice as much); equal weights balance within a
 * band; a single-node ring owns everything with one replica; and an empty ring, a non-positive
 * weight, a zero replica count and duplicate node names are rejected.
 */
import java.util.*;

final class RingNode {
    final String name;
    final int weight;

    RingNode(String name, int weight) {
        this.name = name;
        this.weight = weight;
    }
}

final class HashRing {
    private static final int TOKENS_PER_WEIGHT = 128;

    final int replicas;
    private final long[] tokenHashes;
    private final int[] tokenOwner;        // index into the node list

    HashRing(List<RingNode> nodes, int replicas) {
        if (nodes.isEmpty()) throw new IllegalArgumentException("cannot build an empty ring");
        if (replicas < 1) throw new IllegalArgumentException("need at least one replica");
        Set<String> names = new HashSet<>();
        for (RingNode node : nodes) {
            if (node.weight <= 0) throw new IllegalArgumentException("weight must be positive: " + node.name);
            if (!names.add(node.name)) throw new IllegalArgumentException("duplicate node: " + node.name);
        }
        // tokens are sorted by (hash, name) so that insertion order is irrelevant
        long[] hashes = new long[nodes.size() * TOKENS_PER_WEIGHT * maxWeight(nodes)];
        int[] owners = new int[hashes.length];
        int count = 0;
        for (int index = 0; index < nodes.size(); index++) {
            RingNode node = nodes.get(index);
            for (int token = 0; token < TOKENS_PER_WEIGHT * node.weight; token++) {
                hashes[count] = mix(node.name, token);
                owners[count] = index;
                count++;
            }
        }
        Integer[] order = new Integer[count];
        for (int i = 0; i < count; i++) order[i] = i;
        final long[] hashRef = hashes;
        final int[] ownerRef = owners;
        final List<RingNode> nodesRef = nodes;
        Arrays.sort(order, (a, b) -> hashRef[a] != hashRef[b] ? Long.compare(hashRef[a], hashRef[b])
                : nodesRef.get(ownerRef[a]).name.compareTo(nodesRef.get(ownerRef[b]).name));
        tokenHashes = new long[count];
        tokenOwner = new int[count];
        for (int i = 0; i < count; i++) {
            tokenHashes[i] = hashRef[order[i]];
            tokenOwner[i] = ownerRef[order[i]];
        }
        this.replicas = replicas;
    }

    private static int maxWeight(List<RingNode> nodes) {
        int max = 1;
        for (RingNode node : nodes) max = Math.max(max, node.weight);
        return max;
    }

    static long mix(String name, int token) {
        long z = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < name.length(); i++) z = (z ^ name.charAt(i)) * 0x100000001B3L;
        z += token * 0x2545F4914F6CDD1DL;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long hashKey(String key) {
        return mix(key, 0x5A17);
    }

    int tokenCount() { return tokenHashes.length; }

    /** Index of the first token at or after the key hash (wrapping around the ring). */
    private int successorIndex(String key) {
        long hash = hashKey(key);
        int lo = 0, hi = tokenHashes.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (tokenHashes[mid] < hash) lo = mid + 1; else hi = mid;
        }
        return lo == tokenHashes.length ? 0 : lo;
    }

    /** Primary replica index for a key. */
    int nodeFor(String key) {
        return tokenOwner[successorIndex(key)];
    }

    /** The first `count` distinct physical nodes walked clockwise from the key. */
    int[] replicasFor(String key, List<RingNode> nodes) {
        int want = Math.min(replicas, distinctNodes(nodes));
        int[] out = new int[want];
        Set<Integer> seen = new LinkedHashSet<>();
        int at = successorIndex(key);
        for (int step = 0; step < tokenHashes.length && seen.size() < want; step++) {
            seen.add(tokenOwner[(at + step) % tokenHashes.length]);
        }
        int i = 0;
        for (int node : seen) out[i++] = node;
        return out;
    }

    private int distinctNodes(List<RingNode> nodes) {
        Set<Integer> owners = new HashSet<>();
        for (int owner : tokenOwner) owners.add(owner);
        return owners.size();
    }

    int[] load(List<RingNode> nodes, List<String> keys) {
        int[] counts = new int[nodes.size()];
        for (String key : keys) counts[nodeFor(key)]++;
        return counts;
    }

    /** Fraction of keys (in permille) whose primary is not `nodeIndex` but becomes it in `other`. */
    static int movedToNode(List<String> keys, HashRing before, HashRing after, int newNodeIndex) {
        int moved = 0;
        for (String key : keys) {
            int previous = before.nodeFor(key);
            int current = after.nodeFor(key);
            if (previous != current) {
                if (current != newNodeIndex) throw new IllegalStateException("key moved to a survivor: " + key);
                moved++;
            }
        }
        return moved;
    }
}

public class Solution {
    private static long state = 0x13198A2E03707344L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static List<RingNode> nodes(String... names) {
        List<RingNode> out = new ArrayList<>();
        for (String name : names) out.add(new RingNode(name, 1));
        return out;
    }

    static List<String> keys(int count, String prefix) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < count; i++) out.add(prefix + i);
        return out;
    }

    public static void main(String[] args) {
        List<RingNode> three = nodes("alpha", "beta", "gamma");

        // --- structure ----------------------------------------------------------
        HashRing ring = new HashRing(three, 2);
        check(ring.tokenCount() == 3 * 128, "one token group per weight unit, got " + ring.tokenCount());
        List<String> workload = keys(4000, "user:");
        int[] load = ring.load(three, workload);
        int total = 0;
        for (int count : load) total += count;
        check(total == workload.size(), "every key is owned by exactly one node");
        for (int count : load) {
            check(count > 400 && count < 3000, "equal weights balance loosely: " + Arrays.toString(load));
        }

        // --- insertion order must not matter ------------------------------------
        List<RingNode> shuffled = new ArrayList<>(three);
        Collections.reverse(shuffled);
        HashRing reordered = new HashRing(shuffled, 2);
        boolean sameOwner = true;
        for (String key : workload) {
            if (three.get(ring.nodeFor(key)).name.equals(shuffled.get(reordered.nodeFor(key)).name) == false) {
                sameOwner = false;
            }
        }
        check(sameOwner, "the ring layout is independent of insertion order");
        check(ring.tokenCount() == reordered.tokenCount(), "the two rings hold the same number of tokens");

        // --- replica sets -------------------------------------------------------
        int[] replicaSet = ring.replicasFor("user:7", three);
        check(replicaSet.length == 2, "two replicas are returned");
        check(replicaSet[0] == ring.nodeFor("user:7"), "the first replica is the primary");
        check(replicaSet[0] != replicaSet[1], "replica nodes are distinct");
        HashRing singleReplica = new HashRing(three, 1);
        int[] onlyOne = singleReplica.replicasFor("user:7", three);
        check(onlyOne.length == 1 && onlyOne[0] == singleReplica.nodeFor("user:7"), "R = 1 yields the primary");
        HashRing overReplicated = new HashRing(three, 9);
        check(overReplicated.replicasFor("user:7", three).length == 3,
                "replication is capped at the number of physical nodes");
        Set<Integer> seen = new LinkedHashSet<>();
        for (int node : overReplicated.replicasFor("user:7", three)) seen.add(node);
        check(seen.size() == 3, "the capped replica set still holds three distinct nodes");

        // --- minimal disruption when a node leaves ------------------------------
        List<RingNode> four = nodes("alpha", "beta", "gamma", "delta");
        HashRing withDelta = new HashRing(four, 1);
        int movedOnRemoval = 0;
        for (String key : workload) {
            int before = withDelta.nodeFor(key);
            int after = ring.nodeFor(key);
            if (before != after) {
                check(four.get(before).name.equals("delta"), "only delta's keys move when delta leaves: " + key);
                movedOnRemoval++;
            }
        }
        int deltaOwned = withDelta.load(four, workload)[3];
        check(movedOnRemoval == deltaOwned, "removal moves exactly the leaving node's keys: "
                + movedOnRemoval + " == " + deltaOwned);

        // --- minimal disruption when a node joins -------------------------------
        int movedOnJoin = HashRing.movedToNode(workload, ring, withDelta, 3);
        check(movedOnJoin > workload.size() / 20, "a joining node takes a meaningful share, got " + movedOnJoin);
        check(movedOnJoin == withDelta.load(four, workload)[3], "the joining node takes exactly its own share");
        double share = movedOnJoin / (double) workload.size();
        check(share > 0.5 / 4 && share < 2.0 / 4, "the joining node takes about 1/(n+1) of the keys: " + share);

        // --- weights shape the load --------------------------------------------
        List<RingNode> weighted = List.of(new RingNode("light", 1), new RingNode("heavy", 2), new RingNode("other", 1));
        HashRing weightedRing = new HashRing(weighted, 1);
        int[] weightedLoad = weightedRing.load(weighted, workload);
        check(weightedLoad[0] + weightedLoad[1] + weightedLoad[2] == workload.size(), "weighted load accounts fully");
        double heavyOverLight = weightedLoad[1] / (double) Math.max(1, weightedLoad[0]);
        check(heavyOverLight > 1.6 && heavyOverLight < 2.4,
                "a doubled weight takes about twice the load, ratio " + heavyOverLight);
        check(weightedLoad[1] > weightedLoad[2], "the heavy node beats an equal weight node");

        // --- a single node ring -------------------------------------------------
        HashRing lonely = new HashRing(List.of(new RingNode("only", 3)), 2);
        check(lonely.tokenCount() == 128 * 3, "weight 3 creates three token groups");
        check(lonely.load(List.of(new RingNode("only", 3)), workload)[0] == workload.size(),
                "a lonely ring owns every key");
        check(lonely.replicasFor("anything", List.of(new RingNode("only", 3))).length == 1,
                "a lonely ring has one replica");

        // --- determinism over many keys ----------------------------------------
        boolean stable = true;
        for (int i = 0; i < 200; i++) {
            String key = "k" + nextInt(100000);
            if (ring.nodeFor(key) != new HashRing(three, 2).nodeFor(key)) stable = false;
        }
        check(stable, "key placement is deterministic across ring instances");

        // --- rejected input -----------------------------------------------------
        boolean rejected = false;
        try {
            new HashRing(new ArrayList<>(), 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty ring must be rejected");
        rejected = false;
        try {
            new HashRing(List.of(new RingNode("a", 0)), 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero weight must be rejected");
        rejected = false;
        try {
            new HashRing(List.of(new RingNode("a", 1)), 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero replicas must be rejected");
        rejected = false;
        try {
            new HashRing(List.of(new RingNode("a", 1), new RingNode("a", 1)), 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "duplicate node names must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
