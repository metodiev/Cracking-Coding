/** Task 782 - Implement probabilistic graphical models inference (belief propagation). (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * A binary pairwise Markov random field: unary potentials psi_i(x) for every variable and edge
 * potentials psi_ij(x, y) for every edge. Belief propagation passes messages
 *
 *     m_{i->j}(y)  proportional to  sum_x psi_i(x) psi_ij(x, y) prod_{k in N(i)\j} m_{k->i}(x)
 *
 * around the graph. On a graph without cycles (a tree) one sweep in each direction already gives the
 * exact marginals, and rooting the model at one variable and multiplying the incoming unnormalised
 * messages into it recovers the exact partition function. On a graph with cycles the same fixed point
 * iteration is only approximate (loopy BP, run on normalised messages with damping so it stays
 * numerically stable), which is what the grid experiment measures against exact enumeration.
 *
 * Verified: on random trees the node marginals, the pairwise marginals and the rooted log Z formula all
 * match the exact enumeration over all 2^n assignments to 1e-9; the message order does not matter on a
 * tree (a different schedule reaches the same fixed point); conditioning by clamping a unary potential
 * to a one-hot gives exactly the conditional marginal of the enumeration; on a 3x3 grid loopy BP
 * converges and its marginals stay within the documented tolerance of the exact ones, while a strongly
 * coupled grid degrades measurably - the documented limitation of loopy BP; and the degenerate inputs
 * (a single variable, two variables, a hard 0/1 potential that makes some messages vanish) stay finite
 * and correct, a vanishing message falling back to the uniform message.
 *
 * Simplified: binary variables, pairwise factors only, a fixed sweep budget with damping instead of a
 * convergence proof, and tiny graphs (n <= 12) so that exact enumeration is available as the oracle.
 */
import java.util.*;

class Mrf {
    final int variables;
    final double[][] unary;
    final List<int[]> edges = new ArrayList<>();
    final List<double[][]> potentials = new ArrayList<>();
    double[] messageAB;
    double[] messageBA;
    double[] rawAB;
    double[] rawBA;
    boolean rawMode = true;

    Mrf(int variables) {
        this.variables = variables;
        this.unary = new double[variables][2];
        for (int i = 0; i < variables; i++) {
            unary[i][0] = 1.0;
            unary[i][1] = 1.0;
        }
    }

    void setUnary(int variable, double zero, double one) {
        unary[variable][0] = zero;
        unary[variable][1] = one;
    }

    void addEdge(int a, int b, double[][] potential) {
        edges.add(new int[]{a, b});
        potentials.add(potential);
    }

    private static int size(int count) { return Math.max(1, count); }

    /**
     * Runs BP with damping for a fixed number of sweeps; returns the largest message change. The
     * default keeps the unnormalised messages (they are what the rooted partition function needs).
     */
    double runBeliefPropagation(int sweeps, double damping) {
        return runSweeps(sweeps, damping);
    }

    /**
     * Loopy variant: only the normalised messages are iterated, which is the standard, numerically
     * safe form for a graph with cycles (the unnormalised recursion can diverge there).
     */
    double runLoopy(int sweeps, double damping) {
        rawMode = false;
        try {
            return runSweeps(sweeps, damping);
        } finally {
            rawMode = true;
        }
    }

    private double runSweeps(int sweeps, double damping) {
        int edgeCount = edges.size();        if (edgeCount == 0) return 0;
        if (messageAB == null) {
            messageAB = new double[edgeCount * 2];
            messageBA = new double[edgeCount * 2];
            rawAB = new double[edgeCount * 2];
            rawBA = new double[edgeCount * 2];
            Arrays.fill(messageAB, 0.5);
            Arrays.fill(messageBA, 0.5);
            Arrays.fill(rawAB, 0.5);
            Arrays.fill(rawBA, 0.5);
        }
        double lastChange = 0;
        for (int sweep = 0; sweep < sweeps; sweep++) {
            lastChange = 0;
            for (int e = 0; e < edgeCount; e++) {
                double change = updateMessage(e, true, damping);
                lastChange = Math.max(lastChange, change);
                change = updateMessage(e, false, damping);
                lastChange = Math.max(lastChange, change);
            }
            if (lastChange < 1e-12) break;
        }
        return lastChange;
    }

    private double updateMessage(int edge, boolean forward, double damping) {
        int[] pair = edges.get(edge);
        int from = forward ? pair[0] : pair[1];
        int to = forward ? pair[1] : pair[0];
        double[] raw = forward ? rawAB : rawBA;
        double[] normalised = forward ? messageAB : messageBA;
        double[] fresh = new double[2];
        for (int toValue = 0; toValue < 2; toValue++) {
            double total = 0;
            for (int fromValue = 0; fromValue < 2; fromValue++) {
                double incomingProduct = rawMode
                        ? incomingRaw(from, edge, fromValue) : incoming(from, edge, fromValue);
                total += unary[from][fromValue]
                        * potentials.get(edge)[fromValue][toValue] * incomingProduct;
            }
            fresh[toValue] = total;
        }
        if (rawMode) {
            double change = 0;
            for (int i = 0; i < 2; i++) {
                double updated = (1 - damping) * raw[edge * 2 + i] + damping * fresh[i];
                change = Math.max(change, Math.abs(updated - raw[edge * 2 + i]));
                raw[edge * 2 + i] = updated;
            }
            double sum = raw[edge * 2] + raw[edge * 2 + 1];
            normalised[edge * 2] = sum > 0 ? raw[edge * 2] / sum : 0.5;
            normalised[edge * 2 + 1] = sum > 0 ? raw[edge * 2 + 1] / sum : 0.5;
            return change;
        }
        double sum = fresh[0] + fresh[1];
        if (!(sum > 0)) {
            fresh[0] = 0.5;
            fresh[1] = 0.5;
        } else {
            fresh[0] /= sum;
            fresh[1] /= sum;
        }
        double change = 0;
        for (int i = 0; i < 2; i++) {
            double updated = (1 - damping) * normalised[edge * 2 + i] + damping * fresh[i];
            change = Math.max(change, Math.abs(updated - normalised[edge * 2 + i]));
            normalised[edge * 2 + i] = updated;
        }
        return change;
    }

    /** Product of the unnormalised messages arriving at `node` from every edge except `excluded`. */
    private double incomingRaw(int node, int excluded, int value) {
        double product = 1;
        for (int e = 0; e < edges.size(); e++) {
            if (e == excluded) continue;
            int[] pair = edges.get(e);
            if (pair[1] == node) product *= rawAB[e * 2 + value];
            else if (pair[0] == node) product *= rawBA[e * 2 + value];
        }
        return product;
    }

    /** Product of the normalised messages arriving at `node` from every edge except `excluded`. */
    private double incoming(int node, int excluded, int value) {
        double product = 1;
        for (int e = 0; e < edges.size(); e++) {
            if (e == excluded) continue;
            int[] pair = edges.get(e);
            if (pair[1] == node) product *= messageAB[e * 2 + value];
            else if (pair[0] == node) product *= messageBA[e * 2 + value];
        }
        return product;
    }

    double[] marginals() {
        double[] result = new double[variables * 2];
        for (int variable = 0; variable < variables; variable++) {
            for (int value = 0; value < 2; value++) {
                result[variable * 2 + value] = unary[variable][value] * incoming(variable, -1, value);
            }
            normalise(result, variable * 2);
        }
        return result;
    }

    /** Pairwise beliefs; on a tree these are the exact pairwise marginals. */
    double[][] pairwiseBeliefs() {
        double[][] beliefs = new double[edges.size()][4];
        for (int e = 0; e < edges.size(); e++) {
            int[] pair = edges.get(e);
            for (int x = 0; x < 2; x++) {
                for (int y = 0; y < 2; y++) {
                    beliefs[e][x * 2 + y] = unary[pair[0]][x] * unary[pair[1]][y] * potentials.get(e)[x][y]
                            * incoming(pair[0], e, x) * incoming(pair[1], e, y);
                }
            }
            normaliseFour(beliefs[e]);
        }
        return beliefs;
    }

    private void normaliseFour(double[] values) {
        double sum = values[0] + values[1] + values[2] + values[3];
        if (!(sum > 0)) {
            for (int i = 0; i < 4; i++) values[i] = 0.25;
        } else {
            for (int i = 0; i < 4; i++) values[i] /= sum;
        }
    }

    /**
     * log Z recovered from the unnormalised fixed-point messages by rooting the model at variable 0:
     * on a tree Z = sum_{x_0} psi_0(x_0) prod_{e in N(0)} m_{e->0}(x_0), which is exact when the
     * messages are the true sum-product messages (they are, up to the sweep budget).
     */
    double logZFromMessages() {
        double total = 0;
        for (int x = 0; x < 2; x++) {
            double product = unary[0][x];
            for (int e = 0; e < edges.size(); e++) {
                int[] pair = edges.get(e);
                if (pair[1] == 0) product *= rawAB[e * 2 + x];
                else if (pair[0] == 0) product *= rawBA[e * 2 + x];
            }
            total += product;
        }
        return Math.log(Math.max(total, Double.MIN_NORMAL));
    }

    private int degree(int variable) {
        int degree = 0;
        for (int[] pair : edges) {
            if (pair[0] == variable || pair[1] == variable) degree++;
        }
        return degree;
    }

    private void normalise(double[] values, int offset) {
        double sum = values[offset] + values[offset + 1];
        if (!(sum > 0)) {
            values[offset] = 0.5;
            values[offset + 1] = 0.5;
        } else {
            values[offset] /= sum;
            values[offset + 1] /= sum;
        }
    }

    // --- exact inference by enumeration over all 2^n assignments ------------

    double exactLogZ() {
        double total = 0;
        for (int mask = 0; mask < (1 << variables); mask++) total += weight(mask);
        return Math.log(total);
    }

    double[] exactMarginals() {
        double total = 0;
        double[] sums = new double[variables * 2];
        for (int mask = 0; mask < (1 << variables); mask++) {
            double weight = weight(mask);
            total += weight;
            for (int variable = 0; variable < variables; variable++) {
                sums[variable * 2 + ((mask >> variable) & 1)] += weight;
            }
        }
        for (int variable = 0; variable < variables; variable++) {
            sums[variable * 2] /= total;
            sums[variable * 2 + 1] /= total;
        }
        return sums;
    }

    double[][] exactPairwise() {
        double total = 0;
        double[][] sums = new double[edges.size()][4];
        for (int mask = 0; mask < (1 << variables); mask++) {
            double weight = weight(mask);
            total += weight;
            for (int e = 0; e < edges.size(); e++) {
                int[] pair = edges.get(e);
                int x = (mask >> pair[0]) & 1;
                int y = (mask >> pair[1]) & 1;
                sums[e][x * 2 + y] += weight;
            }
        }
        for (double[] row : sums) {
            for (int i = 0; i < 4; i++) row[i] /= total;
        }
        return sums;
    }

    private double weight(int mask) {
        double product = 1;
        for (int variable = 0; variable < variables; variable++) {
            product *= unary[variable][(mask >> variable) & 1];
        }
        for (int e = 0; e < edges.size(); e++) {
            int[] pair = edges.get(e);
            product *= potentials.get(e)[(mask >> pair[0]) & 1][(mask >> pair[1]) & 1];
        }
        return product;
    }
}

public class Solution {
    private static long state = 0x3C6EF372FE94F82BL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** A random tree on n variables: every new node attaches to an earlier one. */
    static Mrf randomTree(int variables, double coupling) {
        Mrf mrf = new Mrf(variables);
        for (int i = 0; i < variables; i++) mrf.setUnary(i, 0.3 + nextDouble(), 0.3 + nextDouble());
        for (int i = 1; i < variables; i++) {
            int parent = nextInt(i);
            double same = coupling * (0.5 + nextDouble());
            double different = (1 - coupling) * (0.2 + nextDouble());
            mrf.addEdge(parent, i, new double[][]{{same, different}, {different, same}});
        }
        return mrf;
    }

    /** A square grid with nearest-neighbour couplings. */
    static Mrf grid(int side, double agreement, double disagreement) {
        Mrf mrf = new Mrf(side * side);
        for (int i = 0; i < side * side; i++) mrf.setUnary(i, 0.5 + nextDouble(), 0.5 + nextDouble());
        for (int row = 0; row < side; row++) {
            for (int column = 0; column < side; column++) {
                int here = row * side + column;
                if (column + 1 < side) {
                    mrf.addEdge(here, here + 1, new double[][]{{agreement, disagreement},
                            {disagreement, agreement}});
                }
                if (row + 1 < side) {
                    mrf.addEdge(here, here + side, new double[][]{{agreement, disagreement},
                            {disagreement, agreement}});
                }
            }
        }
        return mrf;
    }

    static double maxDifference(double[] a, double[] b) {
        double worst = 0;
        for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        return worst;
    }

    public static void main(String[] args) {
        // --- a single variable -------------------------------------------------
        Mrf single = new Mrf(1);
        single.setUnary(0, 1.0, 3.0);
        double[] singleMarginal = single.marginals();
        check(Math.abs(singleMarginal[0] - 0.25) < 1e-12 && Math.abs(singleMarginal[1] - 0.75) < 1e-12,
                "a single variable marginalises its unary potential");
        check(Math.abs(single.exactLogZ() - Math.log(4.0)) < 1e-12, "the partition function of one variable");
        check(Math.abs(single.logZFromMessages() - Math.log(4.0)) < 1e-9, "the message formula is exact without edges");

        // --- two variables, including a hard constraint ------------------------
        Mrf pair = new Mrf(2);
        pair.setUnary(0, 1.0, 1.0);
        pair.setUnary(1, 1.0, 1.0);
        pair.addEdge(0, 1, new double[][]{{1.0, 0.0}, {0.0, 1.0}});       // x0 == x1
        pair.runBeliefPropagation(20, 1.0);
        double[] pairMarginal = pair.marginals();
        double[] pairExact = pair.exactMarginals();
        check(maxDifference(pairMarginal, pairExact) < 1e-9, "a hard constraint must be handled exactly");
        check(Math.abs(pairMarginal[0] - 0.5) < 1e-9, "the equality constraint leaves a fair coin");
        for (double value : pairMarginal) check(!Double.isNaN(value), "no NaN may leak out of a vanished message");
        check(Math.abs(pair.logZFromMessages() - pair.exactLogZ()) < 1e-9, "the message formula is exact on a two-node tree: "
                + pair.logZFromMessages() + " vs " + pair.exactLogZ());

        // --- conditioning ------------------------------------------------------
        double[][] referencePair = new double[2][2];
        referencePair[0][0] = 3.0;
        referencePair[0][1] = 1.0;
        referencePair[1][0] = 1.0;
        referencePair[1][1] = 3.0;
        Mrf reference = new Mrf(2);
        reference.setUnary(0, 1.0, 1.0);
        reference.setUnary(1, 1.0, 1.0);
        reference.addEdge(0, 1, referencePair);
        double[][] referenceExact = reference.exactPairwise();
        double conditionalZero = referenceExact[0][1] / (referenceExact[0][1] + referenceExact[0][3]);
        Mrf clampedMrf = new Mrf(2);
        clampedMrf.setUnary(0, 1.0, 1.0);
        clampedMrf.setUnary(1, 0.0, 1.0);                     // variable 1 is evidence: it must be 1
        clampedMrf.addEdge(0, 1, copy(referencePair));
        clampedMrf.runBeliefPropagation(20, 1.0);
        double[] clampedMarginal = clampedMrf.marginals();
        check(Math.abs(clampedMarginal[0] - conditionalZero) < 1e-9,
                "clamping a unary potential must equal conditioning: " + clampedMarginal[0] + " vs "
                        + conditionalZero);
        check(Math.abs(clampedMarginal[2]) < 1e-12 && Math.abs(clampedMarginal[3] - 1) < 1e-12,
                "the clamped variable sits entirely on the evidence value");

        // --- random trees against exact enumeration ---------------------------
        double worstMarginal = 0;
        double worstPairwise = 0;
        double worstLogZ = 0;
        for (int trial = 0; trial < 8; trial++) {
            int variables = 3 + nextInt(9);
            Mrf tree = randomTree(variables, 0.8);
            tree.runBeliefPropagation(40, 1.0);
            double[] approximate = tree.marginals();
            double[] exact = tree.exactMarginals();
            worstMarginal = Math.max(worstMarginal, maxDifference(approximate, exact));
            double[][] approximatePairs = tree.pairwiseBeliefs();
            double[][] exactPairs = tree.exactPairwise();
            for (int e = 0; e < approximatePairs.length; e++) {
                worstPairwise = Math.max(worstPairwise, maxDifference(approximatePairs[e], exactPairs[e]));
            }
            worstLogZ = Math.max(worstLogZ, Math.abs(tree.logZFromMessages() - tree.exactLogZ()));
            for (double value : approximate) check(!Double.isNaN(value), "marginals stay finite");
        }
        System.out.println("tree BP: worst marginal error=" + worstMarginal + " pairwise=" + worstPairwise
                + " logZ=" + worstLogZ);
        check(worstMarginal < 1e-9, "tree marginals must be exact, worst error " + worstMarginal);
        check(worstPairwise < 1e-9, "tree pairwise marginals must be exact, worst error " + worstPairwise);
        check(worstLogZ < 1e-9, "the message formula must give log Z exactly on trees, got " + worstLogZ);

        // --- the message order must not matter on a tree ----------------------
        Mrf ordered = randomTree(7, 0.7);
        Mrf reordered = new Mrf(7);
        for (int i = 0; i < 7; i++) {
            reordered.setUnary(i, ordered.unary[i][0], ordered.unary[i][1]);
        }
        List<int[]> edges = new ArrayList<>(ordered.edges);
        for (int i = edges.size() - 1; i >= 0; i--) {
            int[] edge = edges.get(i);
            reordered.addEdge(edge[0], edge[1], copy(ordered.potentials.get(i)));
        }
        ordered.runBeliefPropagation(40, 1.0);
        reordered.runBeliefPropagation(40, 1.0);
        check(maxDifference(ordered.marginals(), reordered.marginals()) < 1e-9,
                "the message schedule must not change a tree result");

        // --- loopy BP on a grid ------------------------------------------------
        Mrf loopy = grid(3, 1.2, 0.7);
        double change = loopy.runLoopy(400, 0.5);
        double[] loopyMarginal = loopy.marginals();
        double[] gridExact = loopy.exactMarginals();
        double gridError = maxDifference(loopyMarginal, gridExact);
        check(change < 1e-6, "loopy BP must converge on this grid, last change " + change);
        check(gridError < 0.1, "loopy marginals must be close to exact, error " + gridError);

        // with a strong coupling the same fixed point iteration gets much worse - a documented limit
        Mrf strong = grid(3, 2.5, 0.4);
        strong.runLoopy(2000, 0.3);
        double strongError = maxDifference(strong.marginals(), strong.exactMarginals());
        System.out.println("grid loopy BP: weak coupling error=" + gridError + " strong coupling error="
                + strongError);
        check(strongError > gridError, "stronger coupling must degrade loopy BP: " + strongError + " vs "
                + gridError);
        check(Double.isFinite(strongError), "loopy marginals stay finite");
        for (int variable = 0; variable < 9; variable++) {
            double total = loopyMarginal[variable * 2] + loopyMarginal[variable * 2 + 1];
            check(Math.abs(total - 1) < 1e-9, "loopy marginals stay normalised");
        }

        System.out.println("PASS");
    }

    static double[][] copy(double[][] potential) {
        double[][] clone = new double[2][2];
        for (int i = 0; i < 2; i++) {
            System.arraycopy(potential[i], 0, clone[i], 0, 2);
        }
        return clone;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
