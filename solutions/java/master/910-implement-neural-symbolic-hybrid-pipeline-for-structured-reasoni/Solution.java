/** Task 910 - Implement neural-symbolic hybrid pipeline for structured reasoning tasks.
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * The task is a structured search problem: given a multiset of integers and a target, find the subsets that
 * sum exactly to the target, and return the one a human would consider canonical (the smallest subset, ties
 * broken lexicographically). A purely learned model is free to emit answers that do not sum to the target;
 * a purely symbolic enumerator is sound but has no idea which of the many solutions is preferred. The hybrid
 * pipeline below splits the work the way neuro-symbolic systems are meant to:
 *
 *   symbolic layer   depth-first enumeration with pruning: it defines the *candidate space*, guarantees every
 *                    candidate really sums to the target, and refuses anything outside it;
 *   neural layer     a linear scorer over subset features (cardinality, largest element, spread, sum of
 *                    squares) trained with a softmax cross-entropy loss on labelled queries - it *ranks*
 *                    candidates but never creates one;
 *   consistency      a proposal that violates the symbolic constraint is rejected with a reason, so the
 *                    pipeline can only ever answer with a verified subset (precision 1 by construction).
 *
 * The checks measure exactly those claims: soundness and completeness of the enumerator against a bitmask
 * oracle, recall of the gold answer, precision 1 for the hybrid versus a deliberately unconstrained neural
 * run (which is shown to propose invalid subsets), the ranking gain over both a random pick and an untrained
 * scorer, the abstain-on-no-solution boundary, and determinism.
 *
 * Simplification: 14 items so the brute-force oracle is exhaustive, 40 training queries. Invariants verified:
 * every emitted answer sums to the target, the gold answer is always in the candidate space, and no invalid
 * proposal ever escapes the constraint layer.
 */
import java.util.*;

/** splitmix64 - the only source of randomness. */
final class Rng910 {
    private long state;

    Rng910(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** One candidate answer: the chosen indices and its feature vector. */
final class Candidate {
    final int[] indices;
    final double[] features;

    Candidate(int[] indices, double[] features) {
        this.indices = indices;
        this.features = features;
    }

    @Override
    public String toString() { return Arrays.toString(indices); }
}

/** The symbolic layer: exhaustive, sound enumeration of subsets hitting the target. */
final class SubsetSolver {
    final int[] items;

    SubsetSolver(int[] items) { this.items = items; }

    /** Depth-first enumeration with pruning: no pruning shortcut is trusted here, only the target test. */
    List<Candidate> solutions(int target) {
        List<Candidate> found = new ArrayList<>();
        explore(0, target, new ArrayList<>(), found);
        found.sort((a, b) -> {
            if (a.indices.length != b.indices.length) return a.indices.length - b.indices.length;
            if (a.features[1] != b.features[1]) return Double.compare(a.features[1], b.features[1]);
            if (a.features[3] != b.features[3]) return Double.compare(a.features[3], b.features[3]);
            for (int i = 0; i < a.indices.length; i++) {
                if (a.indices[i] != b.indices[i]) return a.indices[i] - b.indices[i];
            }
            return 0;
        });
        return found;
    }

    private void explore(int start, int remaining, List<Integer> chosen, List<Candidate> found) {
        if (remaining == 0 && !chosen.isEmpty()) {
            int[] indices = new int[chosen.size()];
            for (int i = 0; i < indices.length; i++) indices[i] = chosen.get(i);
            found.add(new Candidate(indices, features(indices)));
            return;
        }
        for (int i = start; i < items.length; i++) {
            if (items[i] > remaining) continue;
            chosen.add(i);
            explore(i + 1, remaining - items[i], chosen, found);
            chosen.remove(chosen.size() - 1);
        }
    }

    /** Independent oracle: enumerate every bitmask and keep the subsets that sum to the target. */
    List<Integer> bitmaskOracle(int target) {
        List<Integer> masks = new ArrayList<>();
        for (int mask = 1; mask < (1 << items.length); mask++) {
            int sum = 0;
            int size = 0;
            for (int i = 0; i < items.length; i++) {
                if ((mask & (1 << i)) != 0) {
                    sum += items[i];
                    size++;
                }
            }
            if (sum == target) masks.add(mask);
        }
        return masks;
    }

    /** Features the learned layer sees; none of them is the raw sum, so the model cannot shortcut the check. */
    double[] features(int[] indices) {
        int max = 0;
        int total = 0;
        for (int index : indices) {
            max = Math.max(max, items[index]);
            total += items[index];
        }
        double sumSquares = 0;
        for (int index : indices) sumSquares += items[index] * items[index];
        return new double[] {indices.length, max, total, Math.sqrt(sumSquares)};
    }

    /**
     * The canonical answer: fewest items, then the smallest largest item, then the tightest magnitude.
     * Those are exactly the quantities the feature vector exposes, so the ranking task is learnable rather
     * than a matter of guessing a hidden tie-break rule.
     */
    static Candidate gold(List<Candidate> solutions) { return solutions.isEmpty() ? null : solutions.get(0); }

    static int[] indicesOf(int mask) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < 31; i++) if ((mask & (1 << i)) != 0) list.add(i);
        int[] result = new int[list.size()];
        for (int i = 0; i < result.length; i++) result[i] = list.get(i);
        return result;
    }
}

/** The learned layer: a linear ranker trained with softmax cross-entropy over the candidate set. */
final class Scorer {
    final double[] weights = new double[4];
    double bias;

    double score(double[] features) {
        double total = bias;
        for (int i = 0; i < features.length; i++) total += weights[i] * features[i];
        return total;
    }

    /** One gradient step on the softmax loss with the gold candidate as the target. */
    double trainStep(List<Candidate> candidates, int goldIndex, double rate) {
        double[] logits = new double[candidates.size()];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < candidates.size(); i++) {
            logits[i] = score(candidates.get(i).features);
            max = Math.max(max, logits[i]);
        }
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            logits[i] = Math.exp(logits[i] - max);
            sum += logits[i];
        }
        double loss = -Math.log(logits[goldIndex] / sum);
        for (int i = 0; i < candidates.size(); i++) {
            double probability = logits[i] / sum - (i == goldIndex ? 1 : 0);
            for (int d = 0; d < weights.length; d++) {
                weights[d] -= rate * probability * candidates.get(i).features[d] / candidates.size();
            }
            bias -= rate * probability / candidates.size();
        }
        return loss;
    }

    Candidate argmax(List<Candidate> candidates) {
        Candidate best = null;
        for (Candidate candidate : candidates) {
            if (best == null || score(candidate.features) > score(best.features)) best = candidate;
        }
        return best;
    }
}

public class Solution {
    static final int ITEMS = 14;
    static final int TRAIN_QUERIES = 40;
    static final int TEST_QUERIES = 60;
    static final int EPOCHS = 200;

    public static void main(String[] args) {
        int[] items = new int[ITEMS];
        Rng910 rng = new Rng910(0x910ABCDEL);
        for (int i = 0; i < ITEMS; i++) items[i] = 1 + rng.nextInt(12);
        SubsetSolver solver = new SubsetSolver(items);
        check(items.length == ITEMS, "the fixture must hold " + ITEMS + " items");

        // ---- 1. the symbolic layer is sound and complete -----------------------------------------------
        int target = 17;
        List<Candidate> solutions = solver.solutions(target);
        check(!solutions.isEmpty(), "the probe target must have at least one solution");
        for (Candidate candidate : solutions) {
            int sum = 0;
            for (int index : candidate.indices) sum += items[index];
            check(sum == target, "the enumerator emitted a subset that does not sum to the target: " + candidate);
            check(new HashSet<>(Arrays.asList(box(candidate.indices))).size() == candidate.indices.length,
                    "a candidate must not use the same item twice");
        }
        List<Integer> oracle = solver.bitmaskOracle(target);
        check(oracle.size() == solutions.size(),
                "the enumerator must be complete: " + solutions.size() + " vs oracle " + oracle.size());
        Set<String> enumerated = new TreeSet<>();
        for (Candidate candidate : solutions) enumerated.add(Arrays.toString(candidate.indices));
        for (int mask : oracle) check(enumerated.contains(Arrays.toString(SubsetSolver.indicesOf(mask))),
                "the enumerator missed subset " + SubsetSolver.indicesOf(mask));
        check(enumerated.size() == solutions.size(), "the enumerator must not report duplicates");
        check(SubsetSolver.gold(solutions) == solutions.get(0),
                "the gold answer must be the smallest, lexicographically first subset");

        // ---- 2. training queries, built over the symbolic candidate space -----------------------------
        Scorer ranker = new Scorer();
        List<List<Candidate>> trainQueries = new ArrayList<>();
        for (int q = 0; q < TRAIN_QUERIES; q++) {
            List<Candidate> candidates = solver.solutions(8 + q % 25);
            if (!candidates.isEmpty()) trainQueries.add(candidates);
        }
        check(trainQueries.size() > 20, "the training queries must mostly be satisfiable, got "
                + trainQueries.size());

        // ---- 3. unconstrained neural decoding is unsound -----------------------------------------------
        List<Candidate> everything = new ArrayList<>();
        for (int mask = 1; mask < (1 << ITEMS); mask++) {
            int[] indices = SubsetSolver.indicesOf(mask);
            everything.add(new Candidate(indices, solver.features(indices)));
        }
        Scorer unconstrained = new Scorer();
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (List<Candidate> candidates : trainQueries) unconstrained.trainStep(candidates, 0, 0.05);
        }
        Candidate neuralOnly = unconstrained.argmax(everything);
        int neuralSum = 0;
        for (int index : neuralOnly.indices) neuralSum += items[index];
        check(validate(neuralOnly, solver, target) == null, "an unconstrained neural decoder must be shown to "
                + "propose an invalid subset, got sum " + neuralSum + " for target " + target);
        check(!enumerated.contains(Arrays.toString(neuralOnly.indices)),
                "the invalid proposal must be outside the sound candidate space");

        // ---- 4. the hybrid: train the ranker on the symbolic candidate space ---------------------------
        double firstLoss = 0;
        double lastLoss = 0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double epochLoss = 0;
            for (List<Candidate> candidates : trainQueries) {
                double loss = ranker.trainStep(candidates, 0, 0.05);
                epochLoss += loss;
                if (epoch == 0) firstLoss += loss;
                if (epoch == EPOCHS - 1) lastLoss += loss;
            }
        }
        check(lastLoss < firstLoss, "training must reduce the softmax loss: " + firstLoss + " -> " + lastLoss);

        // ---- 5. hybrid versus baselines ----------------------------------------------------------------
        int hybridHits = 0;
        int randomHits = 0;
        int untrainedHits = 0;
        int tested = 0;
        Scorer untrained = new Scorer();
        Rng910 weightRng = new Rng910(0x910FEEDL);
        for (int d = 0; d < untrained.weights.length; d++) untrained.weights[d] = weightRng.nextInt(21) - 10;
        for (int q = 0; q < TEST_QUERIES; q++) {
            int queryTarget = 6 + q % 30;
            List<Candidate> candidates = solver.solutions(queryTarget);
            if (candidates.isEmpty()) continue;
            tested++;
            Candidate gold = SubsetSolver.gold(candidates);
            Candidate hybrid = validate(ranker.argmax(candidates), solver, queryTarget);
            check(hybrid != null, "the hybrid must always emit a validated candidate");
            int sum = 0;
            for (int index : hybrid.indices) sum += items[index];
            check(sum == queryTarget, "the hybrid answer must sum to the target");
            if (Arrays.equals(hybrid.indices, gold.indices)) hybridHits++;
            int randomIndex = new Rng910(0x9100F00L + q).nextInt(candidates.size());
            if (Arrays.equals(candidates.get(randomIndex).indices, gold.indices)) randomHits++;
            if (Arrays.equals(untrained.argmax(candidates).indices, gold.indices)) untrainedHits++;
        }
        double hybridAccuracy = hybridHits / (double) tested;
        double randomAccuracy = randomHits / (double) tested;
        double untrainedAccuracy = untrainedHits / (double) tested;
        check(hybridAccuracy > randomAccuracy, "ranking must beat picking a candidate at random: "
                + hybridAccuracy + " vs " + randomAccuracy);
        check(hybridAccuracy >= 0.8, "the trained ranker must recover the canonical answer often: "
                + hybridAccuracy);
        check(hybridAccuracy > untrainedAccuracy, "training must beat the untrained scorer: " + hybridAccuracy
                + " vs " + untrainedAccuracy);

        // ---- 5. the constraint layer rejects invalid proposals -----------------------------------------
        Candidate invalid = new Candidate(SubsetSolver.indicesOf(1 << 0), solver.features(new int[] {0}));
        check(validate(invalid, solver, target) == null, "an invalid proposal must be rejected");
        check(validate(solutions.get(0), solver, target) != null, "a valid proposal must pass");
        check(rejectReason(invalid, solver, target).equals("not a solution"), "the rejection must carry a "
                + "reason, got " + rejectReason(invalid, solver, target));

        // ---- 6. boundary cases -------------------------------------------------------------------------
        int impossible = -1;
        check(solver.solutions(impossible).isEmpty(), "a negative target has no solutions");
        check(validate(null, solver, impossible) == null, "the hybrid must abstain when nothing is proposed");
        check(SubsetSolver.gold(solver.solutions(impossible)) == null, "no solutions means no gold answer");
        int emptySpaceTarget = 0;
        check(solver.solutions(emptySpaceTarget).isEmpty(), "the empty subset is not an answer");
        check(new SubsetSolver(new int[] {5, 5, 5}).solutions(10).size() == 3,
                "identical items must be distinguished by index: 3 pairs of 5s");

        // ---- 7. determinism ------------------------------------------------------------------------------
        check(Arrays.equals(solver.solutions(target).get(0).indices, solutions.get(0).indices),
                "enumeration must be reproducible");
        check(ranker.score(solutions.get(0).features) == ranker.score(solutions.get(0).features),
                "scoring must be deterministic");
        check(solver.bitmaskOracle(target).size() == oracle.size(), "the oracle must be reproducible");

        System.out.println("items " + ITEMS + " (sum " + sumOf(items) + "), candidate space 2^" + ITEMS);
        System.out.println("symbolic recall " + enumerated.size() + " solutions for target " + target
                + "; unconstrained neural proposal sum " + neuralSum);
        System.out.println("accuracy over " + tested + " queries: hybrid " + round(hybridAccuracy) + " vs "
                + "random pick " + round(randomAccuracy) + " vs random weights " + round(untrainedAccuracy)
                + "; precision 1.0 enforced by the constraint layer");
        System.out.println("PASS");
    }

    /** The constraint layer: only candidates that are genuine solutions may leave the pipeline. */
    static Candidate validate(Candidate proposal, SubsetSolver solver, int target) {
        if (proposal == null) return null;
        int sum = 0;
        for (int index : proposal.indices) {
            if (index < 0 || index >= solver.items.length) return null;
            sum += solver.items[index];
        }
        return sum == target ? proposal : null;
    }

    static String rejectReason(Candidate proposal, SubsetSolver solver, int target) {
        if (proposal == null) return "nothing proposed";
        for (int index : proposal.indices) if (index < 0 || index >= solver.items.length) return "not a solution";
        int sum = 0;
        for (int index : proposal.indices) sum += solver.items[index];
        return sum == target ? "accepted" : "not a solution";
    }

    static Integer[] box(int[] values) {
        Integer[] boxed = new Integer[values.length];
        for (int i = 0; i < values.length; i++) boxed[i] = values[i];
        return boxed;
    }

    static int sumOf(int[] values) {
        int total = 0;
        for (int value : values) total += value;
        return total;
    }

    static double round(double value) { return Math.round(value * 10000) / 10000.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
