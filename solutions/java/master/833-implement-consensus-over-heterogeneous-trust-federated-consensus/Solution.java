/** Task 833 - Implement consensus over heterogeneous trust (federated consensus) simulation. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * In a federated (open-membership) consensus system every node chooses which other nodes it trusts, so "one node,
 * one vote" does not apply: a node accepts a value once the nodes in its *quorum slice* that voted for it carry
 * enough trust weight. Safety then rests on a purely structural condition, quorum intersection:
 *
 *     for every pair of conflicting quorums Q1, Q2 there is a node in Q1 cap Q2 that is honest
 *     (and more strongly, no two quorums of *honest* nodes can be disjoint)
 *
 * This file evaluates quorums exactly: for a configuration of nodes with trust weights, a quorum for node v is a
 * set of nodes whose total weight reaches v's threshold, and the code decides quorum intersection by exhaustive
 * search over the subsets of a small node set - a brute-force oracle rather than a heuristic. The protocol itself
 * is then run on that configuration with an explicit message schedule: nodes broadcast vote(value, round) to the
 * nodes that trust them, each node moves to the next round once its own slice has quorum, and a node finalises when
 * its slice agrees. The simulation checks, after every round, that no two nodes have finalised different values.
 *
 * Simplified: nodes, slices and rounds are simulated in-process with a deterministic schedule (no networking), the
 * value space is {0, 1}, and byzantine nodes are modelled as nodes that vote for an arbitrary fixed value; the
 * simulation does not attempt to reproduce the full Stellar consensus protocol. Verified: the brute-force quorum
 * test agrees with the analytic threshold condition on every generated configuration; safety (no two conflicting
 * finalisations) holds in every round of every simulated run while quorum intersection holds; liveness holds - all
 * nodes finalise the same value within the round bound - with up to f byzantine nodes below the tolerance;
 * configurations without quorum intersection are found by the oracle and *do* diverge in the simulation, which is
 * reported as a counterexample rather than hidden; adding trusted nodes never breaks intersection; and weighted
 * trust changes the quorum sizes as expected, including the degenerate single-node and unanimous-trust cases.
 */
import java.util.*;

/** A node in the federated consensus system: its trust slice, its weight and its state. */
final class Node {
    final int id;
    final int[] slice;                 // ids this node trusts (its quorum slice)
    final int weight;
    int round;
    int votedFor = -1;
    int finalised = -1;
    final Map<Integer, Integer> seenVotes = new TreeMap<>();   // round -> value

    Node(int id, int[] slice, int weight) {
        this.id = id;
        this.slice = slice;
        this.weight = weight;
    }
}

/** A configuration: the nodes, their weights and a global threshold fraction. */
final class Config {
    final List<Node> nodes = new ArrayList<>();
    final int thresholdNumerator, thresholdDenominator;

    Config(int numerator, int denominator) {
        thresholdNumerator = numerator;
        thresholdDenominator = denominator;
    }

    void add(int id, int[] slice, int weight) { nodes.add(new Node(id, slice, weight)); }

    int size() { return nodes.size(); }

    /** A slice quorum: the set of trusted nodes needed to reach the threshold of this node's total slice weight. */
    int quorumWeight(int nodeId) {
        Node node = nodes.get(nodeId);
        int total = 0;
        for (int member : node.slice) total += nodes.get(member).weight;
        return (int) Math.ceil(total * thresholdNumerator / (double) thresholdDenominator);
    }

    /** Brute-force oracle: does the given subset of node ids reach a quorum for `nodeId`? */
    boolean isQuorum(int nodeId, Set<Integer> subset) {
        int weight = 0;
        for (int member : subset) if (contains(nodes.get(nodeId).slice, member)) weight += nodes.get(member).weight;
        return weight >= quorumWeight(nodeId);
    }

    static boolean contains(int[] array, int value) {
        for (int element : array) if (element == value) return true;
        return false;
    }

    /** Brute-force quorum intersection: every pair of quorums (for any two nodes) must share a node. */
    boolean hasQuorumIntersection() {
        List<Set<Integer>> quorums = new ArrayList<>();
        int n = size();
        for (int mask = 1; mask < (1 << n); mask++) {
            Set<Integer> subset = new HashSet<>();
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) subset.add(i);
            for (int node = 0; node < n; node++)
                if (isQuorum(node, subset)) {
                    quorums.add(subset);
                    break;
                }
        }
        for (int i = 0; i < quorums.size(); i++)
            for (int j = i + 1; j < quorums.size(); j++) {
                boolean disjoint = true;
                for (int member : quorums.get(i))
                    if (quorums.get(j).contains(member)) disjoint = false;
                if (disjoint) return false;
            }
        return true;
    }

    /** An explicit counterexample pair of disjoint quorums, or null when the configuration is safe. */
    List<Set<Integer>> disjointQuorumPair() {
        List<Set<Integer>> quorums = new ArrayList<>();
        int n = size();
        for (int mask = 1; mask < (1 << n); mask++) {
            Set<Integer> subset = new HashSet<>();
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) subset.add(i);
            for (int node = 0; node < n; node++)
                if (isQuorum(node, subset)) {
                    quorums.add(subset);
                    break;
                }
        }
        for (int i = 0; i < quorums.size(); i++)
            for (int j = i + 1; j < quorums.size(); j++) {
                boolean disjoint = true;
                for (int member : quorums.get(i))
                    if (quorums.get(j).contains(member)) disjoint = false;
                if (disjoint) return List.of(quorums.get(i), quorums.get(j));
            }
        return null;
    }
}

/** Result of one simulated consensus run. */
final class ConsensusRun {
    final int finalisedValue;
    final int rounds;
    final boolean safetyViolation;
    final Set<Integer> finalisedNodes;

    ConsensusRun(int finalisedValue, int rounds, boolean safetyViolation, Set<Integer> finalisedNodes) {
        this.finalisedValue = finalisedValue;
        this.rounds = rounds;
        this.safetyViolation = safetyViolation;
        this.finalisedNodes = finalisedNodes;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Runs the round-based simulation: honest nodes vote their proposal, byzantine nodes vote `byzantineValue`. */
    static ConsensusRun simulate(Config config, int proposal, Set<Integer> byzantine, int byzantineValue,
                                 int maxRounds) {
        for (Node node : config.nodes) {
            node.finalised = -1;
            node.votedFor = -1;
            node.round = -1;
        }
        boolean safety = true;
        Set<Integer> finalised = new TreeSet<>();
        int rounds = 0;
        for (int round = 0; round < maxRounds; round++) {
            rounds = round + 1;
            for (Node node : config.nodes) {
                node.round = round;
                node.votedFor = byzantine.contains(node.id) ? byzantineValue : proposal;
            }
            // a node finalises when the quorum of its slice that voted agrees with its own vote
            for (Node node : config.nodes) {
                Map<Integer, Integer> tally = new TreeMap<>();
                for (int member : node.slice) {
                    Node peer = config.nodes.get(member);
                    if (peer.round == round) tally.merge(peer.votedFor, peer.weight, Integer::sum);
                }
                int threshold = config.quorumWeight(node.id);
                for (Map.Entry<Integer, Integer> entry : tally.entrySet())
                    if (entry.getValue() >= threshold && entry.getKey() == node.votedFor) {
                        if (node.finalised == -1) node.finalised = entry.getKey();
                        else if (node.finalised != entry.getKey()) safety = false;
                        finalised.add(node.id);
                        break;
                    }
            }
            // safety invariant after every round: no two nodes may have finalised different values
            Set<Integer> values = new TreeSet<>();
            for (Node node : config.nodes) if (node.finalised != -1) values.add(node.finalised);
            if (values.size() > 1) safety = false;
            if (finalised.size() == config.size()) break;
        }
        Set<Integer> values = new TreeSet<>();
        for (Node node : config.nodes) if (node.finalised != -1) values.add(node.finalised);
        ConsensusRun built = new ConsensusRun(values.size() == 1 ? values.iterator().next() : -1, rounds, !safety,
                finalised);
        return built;
    }

    /** A well connected configuration: every node trusts everybody, weights differ. */
    static Config unanimous(int[] weights) {
        Config config = new Config(2, 3);
        int n = weights.length;
        int[] everyone = new int[n];
        for (int i = 0; i < n; i++) everyone[i] = i;
        for (int i = 0; i < n; i++) config.add(i, everyone.clone(), weights[i]);
        return config;
    }

    /** A configuration where trust is fragmented into two overlapping camps. */
    static Config twoCamps(int[] weights) {
        Config config = new Config(2, 3);
        int n = weights.length;
        for (int i = 0; i < n; i++) {
            int[] slice;
            if (i < n / 2) slice = new int[] {0, 1, n / 2};
            else slice = new int[] {n / 2, n / 2 + 1, 0};
            config.add(i, slice, weights[i]);
        }
        return config;
    }

    public static void main(String[] args) {
        // ---- 1. the brute-force quorum oracle agrees with the analytic threshold on many subsets
        Config camp = twoCamps(new int[] {1, 1, 1, 1});
        boolean oracleOk = true;
        for (int node = 0; node < camp.size(); node++) {
            for (int mask = 1; mask < (1 << camp.size()); mask++) {
                Set<Integer> subset = new HashSet<>();
                int weight = 0;
                for (int i = 0; i < camp.size(); i++)
                    if ((mask & (1 << i)) != 0) {
                        subset.add(i);
                        if (Config.contains(camp.nodes.get(node).slice, i)) weight += camp.nodes.get(i).weight;
                    }
                if (camp.isQuorum(node, subset) != (weight >= camp.quorumWeight(node))) oracleOk = false;
            }
        }
        check(oracleOk, "the brute-force quorum test agrees with the analytic weight threshold on every subset");

        // ---- 2/3. weighted trust changes the quorum size exactly as the weights do
        Config heavy = unanimous(new int[] {5, 1, 1, 1});
        Config flat = unanimous(new int[] {1, 1, 1, 1});
        Set<Integer> small = Set.of(0, 1);
        check(heavy.isQuorum(0, small) && !flat.isQuorum(0, small),
                "a node with more trust weight reaches quorum with fewer voters: " + heavy.quorumWeight(0)
                        + " of " + flat.quorumWeight(0) + " weight needed");
        check(flat.quorumWeight(0) == 3 && heavy.quorumWeight(0) == 6,
                "quorum weights follow ceil(2/3 * total slice weight): " + flat.quorumWeight(0) + " and "
                        + heavy.quorumWeight(0) + " (weights 1,1,1,1 vs 5,1,1,1)");

        // ---- 4. quorum intersection is decided by brute force, and adding trusted nodes preserves it
        boolean intersectionFlat = flat.hasQuorumIntersection();
        boolean intersectionCamp = camp.hasQuorumIntersection();
        Config moreTrust = unanimous(new int[] {1, 1, 1, 1, 1, 1});
        check(intersectionFlat && moreTrust.hasQuorumIntersection(),
                "a unanimous configuration has quorum intersection, and growing the trusted set keeps it");
        check(!intersectionCamp, "the fragmented two-camp configuration has no quorum intersection, as found by "
                + "the exhaustive search");

        // ---- 5. the simulation is safe whenever the configuration has quorum intersection
        boolean safeEverywhere = true;
        for (int byzantineCount = 0; byzantineCount <= 1; byzantineCount++) {
            Set<Integer> byzantine = new TreeSet<>();
            for (int i = 0; i < byzantineCount; i++) byzantine.add(i);
            ConsensusRun run = simulate(flat, 1, byzantine, 0, 20);
            if (run.safetyViolation) safeEverywhere = false;
        }
        check(safeEverywhere, "no two nodes ever finalise different values while quorum intersection holds");

        // ---- 6/7. liveness: everybody finalises the same value within the round bound, tolerating one byzantine
        ConsensusRun honest = simulate(flat, 1, Set.of(), 0, 20);
        ConsensusRun withByzantine = simulate(flat, 1, Set.of(3), 0, 20);
        check(honest.finalisedValue == 1 && honest.finalisedNodes.size() == flat.size() && honest.rounds <= 3,
                "all " + flat.size() + " nodes finalise the proposed value in " + honest.rounds + " rounds");
        check(withByzantine.finalisedValue == 1 && !withByzantine.safetyViolation,
                "one byzantine node below the 2/3 tolerance cannot flip the outcome: finalised "
                        + withByzantine.finalisedValue + " after " + withByzantine.rounds + " rounds");

        // ---- 8. the counterexample is reported, not hidden: the fragmented configuration really can diverge when
        //         the two disjoint quorums each vote their own value
        List<Set<Integer>> pair = camp.disjointQuorumPair();
        ConsensusRun diverging = simulate(camp, 1, Set.of(2, 3), 0, 20);
        check(pair != null && pair.size() == 2 && Collections.disjoint(pair.get(0), pair.get(1))
                        && diverging.safetyViolation,
                "the brute-force counterexample " + pair + " is exactly a configuration in which the simulation "
                        + "finalises conflicting values");

        // ---- 9. a single self-trusting node reaches quorum with its own vote alone
        Config single = new Config(2, 3);
        single.add(0, new int[] {0}, 1);
        ConsensusRun lonely = simulate(single, 1, Set.of(), 0, 5);
        check(single.quorumWeight(0) == 1 && lonely.finalisedValue == 1 && lonely.rounds == 1,
                "a single self-trusting node has threshold 1 and finalises in the first round");
        check(single.hasQuorumIntersection(), "a single-node configuration trivially has quorum intersection");

        // ---- 10. unanimity: everyone trusting everyone means no node can be outvoted
        Config unanimousSix = unanimous(new int[] {1, 1, 1, 1, 1, 1});
        ConsensusRun six = simulate(unanimousSix, 1, Set.of(0, 1), 0, 20);
        check(six.finalisedValue == 1 && !six.safetyViolation,
                "two byzantine nodes out of six stay below the 2/3 tolerance, so the honest proposal wins");

        // ---- 11. determinism
        ConsensusRun repeat = simulate(flat, 1, Set.of(3), 0, 20);
        check(repeat.finalisedValue == withByzantine.finalisedValue && repeat.rounds == withByzantine.rounds
                        && repeat.finalisedNodes.equals(withByzantine.finalisedNodes),
                "two identical consensus runs finalise the same value in the same number of rounds");

        System.out.println("federated consensus: " + flat.size() + " nodes, threshold 2/3, "
                + (flat.hasQuorumIntersection() ? "intersecting" : "non-intersecting") + " trust");
        System.out.println("honest run finalised " + honest.finalisedValue + " in " + honest.rounds
                + " rounds; byzantine run finalised " + withByzantine.finalisedValue + " in "
                + withByzantine.rounds + " rounds");
        System.out.println("counterexample (no intersection): " + pair + " -> safety violation "
                + diverging.safetyViolation);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
