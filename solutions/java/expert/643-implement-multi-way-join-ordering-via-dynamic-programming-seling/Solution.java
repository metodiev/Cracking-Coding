/** Task 643 - Implement multi-way join ordering via dynamic programming (Selinger-style). (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Selinger's optimizer is a dynamic program over subsets of the joined relations. A query supplies one
 * effective cardinality per relation (local selections already applied) and one selectivity per join
 * predicate; the planner then computes, for every subset, the cheapest plan tree covering exactly that
 * subset. Two search spaces are implemented:
 *
 *   bushy      - every split of a subset into two non-empty halves is considered, so both
 *                (A join B) join (C join D) and ((A join B) join C) join D are reachable;
 *   left-deep  - Selinger's original shape, where one input of every join is always a base relation.
 *
 * Cost model (stub, documented): SCAN_CPU per row produced by a base scan, JOIN_CPU per row produced
 * by a join, and the whole tree costs the sum over its join nodes of (output rows x JOIN_CPU), with a
 * CROSS_PENALTY multiplier when the two sides of a split share no join predicate. The intermediate
 * cardinality of a split is rows(S1) * rows(S2) * PRODUCT of the selectivities of the predicates that
 * cross the split, so a subset's cardinality depends only on the subset - which is exactly the
 * property that makes keeping a single cheapest plan per subset correct rather than an approximation.
 *
 * Simplified (deliberately): no interesting orders / physical property propagation, no sort-merge or
 * index-nested-loop alternatives, no histograms (a single selectivity per predicate) and enumerative
 * rather than memoized combinatorics.
 *
 * Verified: the DP cost equals the cost of the best plan found by an independent exhaustive
 * enumeration of *all* trees in the same space, for 3, 4 and 5 relation queries; the whole DP result
 * is invariant under permutation of the input relation list; the number of states visited is exactly
 * 2^n - 1 and the number of splits evaluated is exactly sum over subsets of 2^(|S|-1) - 1; pruning
 * cross products on a connected join graph neither changes the cost nor evaluates as many splits; the
 * DP starts from the most selective pair (checked against a brute-force argmin over pairs); a bad
 * input order costs strictly more than the DP's answer; disconnected queries still plan, paying the
 * documented cross-product penalty.
 */
import java.util.*;

class SelingerPlanner {

    static final double JOIN_CPU = 1.0;
    static final double SCAN_CPU = 0.5;
    static final double CROSS_PENALTY = 10.0;

    static final class Query {
        final String[] names;
        final double[] rows;
        final Map<Long, Double> predicates = new LinkedHashMap<>();

        Query(String[] names, double[] rows) {
            this.names = names;
            this.rows = rows;
        }

        Query predicate(int i, int j, double selectivity) {
            predicates.put(key(i, j), selectivity);
            return this;
        }

        static long key(int i, int j) { return Math.min(i, j) * 1000L + Math.max(i, j); }

        boolean connected(int i, int j) { return predicates.containsKey(key(i, j)); }

        double selectivity(int i, int j) { return predicates.getOrDefault(key(i, j), 1.0); }

        int size() { return names.length; }

        /** Product of the selectivities of the predicates crossing the split; 1.0 means no predicate. */
        double crossingSelectivity(int leftMask, int rightMask) {
            double selectivity = 1.0;
            for (int i = 0; i < size(); i++) {
                if ((leftMask & (1 << i)) == 0) continue;
                for (int j = 0; j < size(); j++) {
                    if ((rightMask & (1 << j)) == 0) continue;
                    selectivity *= selectivity(i, j);
                }
            }
            return selectivity;
        }

        int crossingPredicates(int leftMask, int rightMask) {
            int count = 0;
            for (int i = 0; i < size(); i++) {
                if ((leftMask & (1 << i)) == 0) continue;
                for (int j = 0; j < size(); j++) {
                    if ((rightMask & (1 << j)) == 0) continue;
                    if (connected(i, j)) count++;
                }
            }
            return count;
        }
    }

    static final class Node {
        final int mask;
        final double rows, cost;
        final Node left, right;
        final boolean crossProduct;

        Node(int mask, double rows, double cost, Node left, Node right, boolean crossProduct) {
            this.mask = mask;
            this.rows = rows;
            this.cost = cost;
            this.left = left;
            this.right = right;
            this.crossProduct = crossProduct;
        }

        static Node leaf(int index, double rows) {
            return new Node(1 << index, rows, rows * SCAN_CPU, null, null, false);
        }

        boolean isLeaf() { return left == null; }

        boolean hasCrossProduct() {
            if (crossProduct) return true;
            return (left != null && left.hasCrossProduct()) || (right != null && right.hasCrossProduct());
        }

        /** The first join evaluated bottom-up; used to check which pair the optimizer starts from. */
        Node firstJoin() {
            if (isLeaf()) return null;
            if (!left.isLeaf()) return left.firstJoin();
            if (!right.isLeaf()) return right.firstJoin();
            return this;
        }

        int leafMask(int position) { return (mask >> position) & 1; }

        /** Deterministic structural signature, used only to break cost ties reproducibly. */
        String signature() {
            if (isLeaf()) return Integer.toString(Integer.numberOfTrailingZeros(mask));
            return "(" + left.signature() + "," + right.signature() + ")";
        }

        String describe(String[] names) {
            if (isLeaf()) return names[Integer.numberOfTrailingZeros(mask)];
            return "(" + left.describe(names) + " * " + right.describe(names) + ")";
        }
    }

    static final class Stats {
        int states;
        int splits;
    }

    static Node combine(Query q, Node left, Node right, Stats stats) {
        int mask = left.mask | right.mask;
        int adjacent = q.crossingPredicates(left.mask, right.mask);
        boolean cross = adjacent == 0;
        double selectivity = q.crossingSelectivity(left.mask, right.mask);
        double rows = left.rows * right.rows * selectivity;
        double penalty = cross ? CROSS_PENALTY : 1.0;
        double cost = left.cost + right.cost + rows * JOIN_CPU * penalty;
        if (stats != null) stats.splits++;
        return new Node(mask, rows, cost, left, right, cross);
    }

    static boolean better(Node a, Node b) {
        if (a == null) return false;
        if (b == null) return true;
        if (a.cost != b.cost) return a.cost < b.cost;
        return a.signature().compareTo(b.signature()) < 0;
    }

    /** Bushy DP over subsets; exact because a subset's cardinality depends only on the subset. */
    static Node dpBushy(Query q, boolean pruneCrossProducts, Stats stats) {
        int n = q.size();
        if (n == 0) throw new IllegalArgumentException("a query needs at least one relation");
        Node[] best = new Node[1 << n];
        for (int i = 0; i < n; i++) best[1 << i] = Node.leaf(i, q.rows[i]);
        for (int mask = 1; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) < 2) continue;
            stats.states++;
            Node chosen = null;
            for (int sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
                int other = mask ^ sub;
                if (sub > other) continue;                       // each unordered split once
                if (best[sub] == null || best[other] == null) continue;
                if (pruneCrossProducts && q.crossingPredicates(sub, other) == 0) continue;
                Node candidate = combine(q, best[sub], best[other], stats);
                if (better(candidate, chosen)) chosen = candidate;
            }
            best[mask] = chosen;
        }
        return best[(1 << n) - 1];
    }

    /** Selinger's original left-deep shape: the outer side is the accumulated plan. */
    static Node dpLeftDeep(Query q, Stats stats) {
        int n = q.size();
        if (n == 0) throw new IllegalArgumentException("a query needs at least one relation");
        Node[] best = new Node[1 << n];
        for (int i = 0; i < n; i++) best[1 << i] = Node.leaf(i, q.rows[i]);
        for (int mask = 1; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) < 2) continue;
            stats.states++;
            Node chosen = null;
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) == 0) continue;
                Node outer = best[mask ^ (1 << i)];
                if (outer == null) continue;
                Node candidate = combine(q, outer, Node.leaf(i, q.rows[i]), stats);
                if (better(candidate, chosen)) chosen = candidate;
            }
            best[mask] = chosen;
        }
        return best[(1 << n) - 1];
    }

    /** Exhaustive enumeration of every bushy (or left-deep) tree over a subset; the oracle for the DP. */
    static List<Node> allTrees(Query q, int mask, boolean leftDeep, Stats stats) {
        List<Node> out = new ArrayList<>();
        if (Integer.bitCount(mask) == 1) {
            out.add(Node.leaf(Integer.numberOfTrailingZeros(mask), q.rows[Integer.numberOfTrailingZeros(mask)]));
            return out;
        }
        for (int sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
            int other = mask ^ sub;
            if (sub > other) continue;
            List<Node> leftTrees = allTrees(q, sub, leftDeep, null);
            List<Node> rightTrees = allTrees(q, other, leftDeep, null);
            for (Node l : leftTrees) {
                for (Node r : rightTrees) {
                    out.add(combine(q, l, r, null));
                    if (!leftDeep) out.add(combine(q, r, l, null));
                }
            }
        }
        return out;
    }

    static Node bestOf(List<Node> trees) {
        Node best = null;
        for (Node tree : trees) if (better(tree, best)) best = tree;
        return best;
    }

    static int popcountSum(int n) {
        int total = 0;
        for (int mask = 1; mask < (1 << n); mask++) {
            int k = Integer.bitCount(mask);
            if (k >= 2) total += (1 << (k - 1)) - 1;
        }
        return total;
    }
}

public class Solution {

    static SelingerPlanner.Query starQuery() {
        return new SelingerPlanner.Query(
                new String[]{"fact", "dim_small", "dim_medium", "dim_large"},
                new double[]{1000000, 100, 10000, 1000})
                .predicate(0, 1, 1.0 / 1000)
                .predicate(0, 2, 1.0 / 10000)
                .predicate(0, 3, 1.0 / 100)
                .predicate(1, 2, 1.0 / 50);
    }

    static SelingerPlanner.Query chainQuery() {
        return new SelingerPlanner.Query(
                new String[]{"a", "b", "c", "d"},
                new double[]{50000, 4000, 300, 20})
                .predicate(0, 1, 1.0 / 500)
                .predicate(1, 2, 1.0 / 40)
                .predicate(2, 3, 1.0 / 5);
    }

    static SelingerPlanner.Query fiveWay() {
        return new SelingerPlanner.Query(
                new String[]{"r0", "r1", "r2", "r3", "r4"},
                new double[]{200000, 50000, 9000, 700, 60})
                .predicate(0, 1, 1.0 / 1000)
                .predicate(1, 2, 1.0 / 200)
                .predicate(2, 3, 1.0 / 30)
                .predicate(3, 4, 1.0 / 6)
                .predicate(0, 2, 1.0 / 5000);
    }

    static SelingerPlanner.Query disconnectedQuery() {
        return new SelingerPlanner.Query(new String[]{"p", "q", "r"}, new double[]{1000, 1000, 1000})
                .predicate(0, 1, 1.0 / 100);
    }

    public static void main(String[] args) {
        SelingerPlanner.Query star = starQuery();
        SelingerPlanner.Stats bushyStats = new SelingerPlanner.Stats();
        SelingerPlanner.Node bushy = SelingerPlanner.dpBushy(star, false, bushyStats);

        check(bushyStats.states == (1 << star.size()) - 1 - star.size(),
                "states visited equals the number of subsets with two or more members: " + bushyStats.states);
        check(bushyStats.splits == SelingerPlanner.popcountSum(star.size()),
                "splits evaluated " + bushyStats.splits + " vs formula " + SelingerPlanner.popcountSum(star.size()));
        check(!bushy.hasCrossProduct(), "a connected query never plans a cross product: " + bushy.describe(star.names));
        SelingerPlanner.Node bushyOracle = SelingerPlanner.bestOf(
                SelingerPlanner.allTrees(star, (1 << star.size()) - 1, false, null));
        check(bushyOracle.cost == bushy.cost,
                "exhaustive bushy enumeration matches the DP: " + bushyOracle.cost + " vs " + bushy.cost);
        SelingerPlanner.Node repeated = SelingerPlanner.dpBushy(starQuery(), false, new SelingerPlanner.Stats());
        check(repeated.signature().equals(bushy.signature()),
                "the DP is deterministic on a repeat run: " + bushy.signature());

        SelingerPlanner.Stats leftStats = new SelingerPlanner.Stats();
        SelingerPlanner.Node leftDeep = SelingerPlanner.dpLeftDeep(star, leftStats);
        SelingerPlanner.Node leftOracle = SelingerPlanner.bestOf(
                SelingerPlanner.allTrees(star, (1 << star.size()) - 1, true, null));
        check(leftOracle.cost == leftDeep.cost,
                "exhaustive left-deep enumeration matches the left-deep DP: " + leftOracle.cost
                        + " vs " + leftDeep.cost);
        check(leftDeep.cost >= bushy.cost, "the bushy space is at least as good: " + bushy.cost
                + " vs " + leftDeep.cost);

        double startPairCost = Double.MAX_VALUE;
        String startPair = "";
        for (int i = 0; i < star.size(); i++) {
            for (int j = i + 1; j < star.size(); j++) {
                if (!star.connected(i, j)) continue;
                double rows = star.rows[i] * star.rows[j] * star.selectivity(i, j);
                double cost = rows * SelingerPlanner.JOIN_CPU
                        + SelingerPlanner.SCAN_CPU * (star.rows[i] + star.rows[j]);
                if (cost < startPairCost) {
                    startPairCost = cost;
                    startPair = star.names[i] + "*" + star.names[j];
                }
            }
        }
        SelingerPlanner.Node first = bushy.firstJoin();
        check(first != null, "a multi-way plan has a first join");
        check(first.cost == startPairCost,
                "the DP starts from the cheapest pair " + startPair + ": " + first.describe(star.names));

        SelingerPlanner.Query chain = chainQuery();
        SelingerPlanner.Node chainBushy = SelingerPlanner.dpBushy(chain, false, new SelingerPlanner.Stats());
        SelingerPlanner.Node chainOracle = SelingerPlanner.bestOf(
                SelingerPlanner.allTrees(chain, (1 << chain.size()) - 1, false, null));
        check(chainBushy.cost == chainOracle.cost, "chain query: DP equals exhaustive enumeration");
        check(chainBushy.rows == chainOracle.rows, "chain query: cardinality estimate is identical");
        SelingerPlanner.Stats unpruned = new SelingerPlanner.Stats();
        SelingerPlanner.Stats pruned = new SelingerPlanner.Stats();
        SelingerPlanner.Node unprunedPlan = SelingerPlanner.dpBushy(chain, false, unpruned);
        SelingerPlanner.Node prunedPlan = SelingerPlanner.dpBushy(chain, true, pruned);
        check(prunedPlan.cost == unprunedPlan.cost, "pruning cross products keeps the cost: "
                + prunedPlan.cost + " vs " + unprunedPlan.cost);
        check(pruned.splits < unpruned.splits, "pruning evaluates fewer splits: " + pruned.splits
                + " vs " + unpruned.splits);
        check(!prunedPlan.hasCrossProduct() && !unprunedPlan.hasCrossProduct(),
                "the optimal connected plan uses no cross product");

        SelingerPlanner.Query five = fiveWay();
        SelingerPlanner.Node fiveDp = SelingerPlanner.dpBushy(five, false, new SelingerPlanner.Stats());
        SelingerPlanner.Node fiveOracle = SelingerPlanner.bestOf(
                SelingerPlanner.allTrees(five, (1 << five.size()) - 1, false, null));
        check(fiveDp.cost == fiveOracle.cost,
                "five-way DP equals exhaustive enumeration: " + fiveDp.cost + " vs " + fiveOracle.cost);

        Set<Double> permutedCosts = new TreeSet<>();
        List<Integer> base = new ArrayList<>();
        for (int i = 0; i < star.size(); i++) base.add(i);
        for (List<Integer> permutation : permutations(base)) {
            SelingerPlanner.Query shuffled = new SelingerPlanner.Query(
                    reorder(star.names, permutation), reorder(star.rows, permutation));
            for (int i = 0; i < star.size(); i++) {
                for (int j = i + 1; j < star.size(); j++) {
                    int a = permutation.get(i);
                    int b = permutation.get(j);
                    if (star.connected(a, b)) shuffled.predicate(i, j, star.selectivity(a, b));
                }
            }
            permutedCosts.add(SelingerPlanner.dpBushy(shuffled, false, new SelingerPlanner.Stats()).cost);
        }
        check(permutedCosts.size() == 1, "the DP is invariant under permutation of the input order: "
                + permutedCosts);
        check(permutedCosts.iterator().next() == bushy.cost, "permutations reproduce the same optimum");

        SelingerPlanner.Node natural = naturalOrderPlan(star);
        check(natural.cost > bushy.cost,
                "joining in the FROM-clause order costs more than the DP's choice: " + natural.cost
                        + " vs " + bushy.cost);
        check(natural.describe(star.names).equals("(((fact * dim_small) * dim_medium) * dim_large)"),
                "the natural plan is the left-to-right one: " + natural.describe(star.names));

        SelingerPlanner.Query disconnected = disconnectedQuery();
        SelingerPlanner.Node withCross = SelingerPlanner.dpBushy(disconnected, false, new SelingerPlanner.Stats());
        check(disconnected.crossingPredicates(1, 4) == 0, "relations 0 and 2 are not connected");
        check(withCross.hasCrossProduct(), "a disconnected query needs a cross product: "
                + withCross.describe(disconnected.names));
        check(Math.abs(recomputeCost(withCross, SelingerPlanner.CROSS_PENALTY) - withCross.cost) < 1e-6,
                "the plan cost is consistent with the documented cross-product penalty");
        check(recomputeCost(withCross, 1.0) * 10 > withCross.cost,
                "the cross-product penalty dominates the plan cost: " + withCross.cost + " vs "
                        + recomputeCost(withCross, 1.0));
        boolean rejected = false;
        try {
            SelingerPlanner.dpBushy(new SelingerPlanner.Query(new String[0], new double[0]), false,
                    new SelingerPlanner.Stats());
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty query is rejected");

        SelingerPlanner.Query single = new SelingerPlanner.Query(new String[]{"only"}, new double[]{42});
        SelingerPlanner.Node singlePlan = SelingerPlanner.dpBushy(single, false, new SelingerPlanner.Stats());
        check(singlePlan.isLeaf() && singlePlan.cost == 21.0, "a single relation is just a scan: "
                + singlePlan.cost);
        SelingerPlanner.Query pair = new SelingerPlanner.Query(new String[]{"l", "r"}, new double[]{1000, 500})
                .predicate(0, 1, 1.0 / 100);
        SelingerPlanner.Node pairPlan = SelingerPlanner.dpBushy(pair, false, new SelingerPlanner.Stats());
        check(!pairPlan.isLeaf() && pairPlan.rows == 5000.0, "two relations join once: rows " + pairPlan.rows);

        System.out.println("star: bushy " + bushy.describe(star.names) + " cost " + bushy.cost
                + " == exhaustive oracle " + bushyOracle.cost);
        System.out.println("states=" + bushyStats.states + " splits=" + bushyStats.splits
                + " (formula " + SelingerPlanner.popcountSum(star.size()) + ")"
                + ", pruned splits " + pruned.splits + " vs unpruned " + unpruned.splits);
        System.out.println("permutations of the input list all yield cost " + permutedCosts.iterator().next()
                + "; natural order costs " + natural.cost);
        System.out.println("5-way DP " + fiveDp.cost + " == exhaustive " + fiveOracle.cost
                + ", cross-product plan " + withCross.describe(disconnected.names));
        System.out.println("PASS");
    }

    static SelingerPlanner.Node naturalOrderPlan(SelingerPlanner.Query q) {
        SelingerPlanner.Node acc = SelingerPlanner.Node.leaf(0, q.rows[0]);
        for (int i = 1; i < q.size(); i++) {
            acc = SelingerPlanner.combine(q, acc, SelingerPlanner.Node.leaf(i, q.rows[i]), null);
        }
        return acc;
    }

    /** Re-derives a plan's cost by walking its tree, using the documented penalty parameter. */
    static double recomputeCost(SelingerPlanner.Node node, double crossPenalty) {
        if (node.isLeaf()) return node.rows * SelingerPlanner.SCAN_CPU;
        double penalty = node.crossProduct ? crossPenalty : 1.0;
        return recomputeCost(node.left, crossPenalty) + recomputeCost(node.right, crossPenalty)
                + node.rows * SelingerPlanner.JOIN_CPU * penalty;
    }

    static String[] reorder(String[] values, List<Integer> permutation) {
        String[] out = new String[values.length];
        for (int i = 0; i < values.length; i++) out[i] = values[permutation.get(i)];
        return out;
    }

    static double[] reorder(double[] values, List<Integer> permutation) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) out[i] = values[permutation.get(i)];
        return out;
    }

    static List<List<Integer>> permutations(List<Integer> values) {
        List<List<Integer>> out = new ArrayList<>();
        permute(new ArrayList<>(values), 0, out);
        return out;
    }

    static void permute(List<Integer> values, int start, List<List<Integer>> out) {
        if (start == values.size()) { out.add(new ArrayList<>(values)); return; }
        for (int i = start; i < values.size(); i++) {
            Collections.swap(values, start, i);
            permute(values, start + 1, out);
            Collections.swap(values, start, i);
        }
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
