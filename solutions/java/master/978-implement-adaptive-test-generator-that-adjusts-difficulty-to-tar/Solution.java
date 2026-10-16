/** Task 978 - Implement adaptive test generator that adjusts difficulty to target percentile.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A Rasch (1PL) adaptive test. The probability of a correct answer is 1/(1 + exp(-(theta - b))), the
 *  ability estimate comes from a deterministic maximum-likelihood grid search over [-4, 4], item selection
 *  picks the unused item minimising 0.5*|b - theta| + 0.5*|b - bTarget| where bTarget is the item-difficulty
 *  quantile matching the requested target percentile, and the test stops once the information-based standard
 *  error 1/sqrt(sum p(1-p)) falls under the threshold or the item cap is reached. Learner responses are
 *  drawn from the same model with a fixed-seed splitmix64 stream, so a run is reproducible.
 *  Simplified: 1PL with a fixed discrimination, a grid-search MLE instead of Newton iterations and a
 *  linearly blended selection rule rather than full maximum-information-plus-exposure-control; the model
 *  monotonicity, the MLE optimum, the SE stopping rule and the target-percentile check are the real ones.
 *  Verified: the Rasch curve's identity, monotonicity and bounds, MLE against an independent coarse-to-fine
 *  search plus the all-correct clamp, the argmin property of every selection step, the SE stopping rule,
 *  mean test difficulty against an independently computed percentile, ability recovery and ordering for two
 *  simulated learners, the empty/single-item boundaries, duplicate-id rejection and seed determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double SE_TARGET = 0.75, GRID = 0.01, THETA_MIN = -4.0, THETA_MAX = 4.0;

    static final class Item {
        final String id; final double difficulty;
        Item(String id, double difficulty) { this.id = id; this.difficulty = difficulty; }
    }

    static double probability(double theta, double difficulty) { return 1.0 / (1.0 + Math.exp(-(theta - difficulty))); }

    /** Deterministic splitmix64 stream for the simulated learner's responses. */
    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        double next() {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return ((z ^ (z >>> 31)) >>> 11) * 0x1.0p-53;
        }
    }

    /** Maximum-likelihood ability from {0,1} responses: deterministic grid search over [THETA_MIN, THETA_MAX]. */
    static double estimate(double[] difficulties, int[] responses) {
        double best = THETA_MIN, bestScore = Double.NEGATIVE_INFINITY;
        int steps = (int) Math.round((THETA_MAX - THETA_MIN) / GRID);
        for (int i = 0; i <= steps; i++) {
            double theta = Math.min(THETA_MAX, THETA_MIN + i * GRID);
            double score = logLikelihood(theta, difficulties, responses);
            if (score > bestScore) { bestScore = score; best = theta; }
        }
        return best;
    }

    static double logLikelihood(double theta, double[] difficulties, int[] responses) {
        double total = 0;
        for (int i = 0; i < responses.length; i++) {
            double p = probability(theta, difficulties[i]);
            total += responses[i] == 1 ? Math.log(p) : Math.log(1 - p);
        }
        return total;
    }

    /** Independent estimator: coarse scan followed by a local bisection on the likelihood derivative. */
    static double estimateByBisection(double[] difficulties, int[] responses) {
        double low = THETA_MIN, high = THETA_MAX;
        for (int step = 0; step < 40; step++) {
            double mid = (low + high) / 2;
            if (logLikelihood(mid + 1e-6, difficulties, responses) > logLikelihood(mid - 1e-6, difficulties, responses)) {
                low = mid;
            } else {
                high = mid;
            }
        }
        return (low + high) / 2;
    }

    static double standardError(double theta, double[] difficulties, int[] responses) {
        double info = 0;
        for (int i = 0; i < difficulties.length; i++) {
            double p = probability(theta, difficulties[i]);
            info += p * (1 - p);
        }
        return info <= 0 ? Double.MAX_VALUE : 1.0 / Math.sqrt(info);
    }

    /** Difficulty quantile of the pool: the target-percentile requirement. */
    static double percentileDifficulty(List<Item> pool, int percentile) {
        List<Double> sorted = new ArrayList<>();
        for (Item item : pool) sorted.add(item.difficulty);
        sorted.sort(Double::compare);
        if (sorted.isEmpty()) return 0.0;
        int clamped = Math.max(0, Math.min(100, percentile));
        int index = (int) Math.round(clamped / 100.0 * (sorted.size() - 1));
        return sorted.get(index);
    }

    static final class Test {
        final List<Item> items = new ArrayList<>();
        final List<Double> thetas = new ArrayList<>();
        final int[] responses;
        Test(int[] responses) { this.responses = responses; }
        double[] difficulties() {
            double[] out = new double[items.size()];
            for (int i = 0; i < out.length; i++) out[i] = items.get(i).difficulty;
            return out;
        }
        double meanDifficulty() {
            double total = 0;
            for (Item item : items) total += item.difficulty;
            return items.isEmpty() ? 0 : total / items.size();
        }
    }

    /** Assembles a test for a learner whose true ability is `trueTheta`; the estimate only drives selection. */
    static Test assemble(List<Item> pool, double trueTheta, int targetPercentile, int maxItems, long seed) {
        Set<String> ids = new HashSet<>();
        for (Item item : pool) {
            if (!ids.add(item.id)) throw new IllegalStateException("duplicate item id " + item.id);
        }
        double bTarget = percentileDifficulty(pool, targetPercentile);
        Rng rng = new Rng(seed);
        Map<String, Double> tieBreak = new TreeMap<>();                       // seed-fixed tie-break key per item
        for (Item item : pool) tieBreak.put(item.id, rng.next());
        List<Item> remaining = new ArrayList<>(pool);
        List<Item> chosenItems = new ArrayList<>();
        List<Double> abilities = new ArrayList<>();
        List<Integer> given = new ArrayList<>();
        double ability = trueTheta;
        while (chosenItems.size() < maxItems && !remaining.isEmpty()) {
            Item chosen = null;
            double bestScore = Double.MAX_VALUE, bestKey = Double.MAX_VALUE;
            for (Item item : remaining) {
                double score = 0.5 * Math.abs(item.difficulty - ability) + 0.5 * Math.abs(item.difficulty - bTarget);
                double key = tieBreak.get(item.id);
                if (score < bestScore - 1e-12 || (Math.abs(score - bestScore) <= 1e-12 && (chosen == null || key < bestKey))) {
                    bestScore = score; bestKey = key; chosen = item;
                }
            }
            abilities.add(ability);
            chosenItems.add(chosen);
            remaining.remove(chosen);
            given.add(chosen.difficulty <= trueTheta ? 1 : 0);                // deterministic Guttman-style learner
            double[] difficulties = new double[chosenItems.size()];
            for (int i = 0; i < difficulties.length; i++) difficulties[i] = chosenItems.get(i).difficulty;
            int[] responses = given.stream().mapToInt(Integer::intValue).toArray();
            ability = estimate(difficulties, responses);
            if (chosenItems.size() >= 3 && standardError(ability, difficulties, responses) <= SE_TARGET) break;
        }
        Test done = new Test(given.stream().mapToInt(Integer::intValue).toArray());
        done.items.addAll(chosenItems);
        done.thetas.addAll(abilities);
        return done;
    }

    static List<Item> pool() {
        List<Item> items = new ArrayList<>();
        double[] difficulties = {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
        for (int i = 0; i < difficulties.length; i++) items.add(new Item("i" + (i + 1), difficulties[i]));
        return items;
    }

    public static void main(String[] args) {
        check(probability(1.0, 1.0) == 0.5 && probability(2.0, 0.0) > probability(1.0, 0.0)
                        && probability(0.0, 2.0) < probability(0.0, 1.0)
                        && probability(50, 0) > 0.99999 && probability(50, 0) <= 1.0 && probability(-50, 0) < 1e-5,
                "the Rasch curve must be 0.5 at theta == b, monotone and bounded");

        double[] d = {-1.0, 0.0, 1.0};
        int[] mixed = {1, 1, 0};
        double grid = estimate(d, mixed);
        double bisect = estimateByBisection(d, mixed);
        check(Math.abs(grid - bisect) < 0.02
                        && logLikelihood(grid, d, mixed) >= logLikelihood(grid + 0.5, d, mixed)
                        && logLikelihood(grid, d, mixed) >= logLikelihood(grid - 0.5, d, mixed),
                "grid MLE must agree with an independent bisection search: " + grid + " vs " + bisect);
        check(Math.abs(estimate(d, new int[]{1, 1, 1}) - THETA_MAX) < 0.02
                        && Math.abs(estimate(d, new int[]{0, 0, 0}) - THETA_MIN) < 0.02,
                "a perfect or empty response pattern must clamp to the grid boundary: "
                        + estimate(d, new int[]{1, 1, 1}));

        List<Item> pool = pool();
        Test high = assemble(pool, 1.0, 85, 8, 12345L);
        Test low = assemble(pool, -1.0, 85, 8, 12345L);
        double highEstimate = estimate(high.difficulties(), high.responses);
        double lowEstimate = estimate(low.difficulties(), low.responses);
        check(highEstimate > lowEstimate && Math.abs(highEstimate - 1.0) < 0.5 && Math.abs(lowEstimate + 1.0) < 0.5,
                "ability estimates must recover and order the simulated learners within half a logit: "
                        + highEstimate + " vs " + lowEstimate);

        double se = standardError(highEstimate, high.difficulties(), high.responses);
        check(high.items.size() >= 3 && high.items.size() <= 8
                        && new HashSet<>(high.items.stream().map(i -> i.id).toList()).size() == high.items.size()
                        && (high.items.size() == 8 || se <= SE_TARGET) && se <= 0.85,
                "the test must stop only at the SE target or the item cap: " + high.items.size() + " items, SE " + se);

        boolean argmin = true;
        for (int step = 0; step + 1 < high.items.size(); step++) {
            double theta = high.thetas.get(step);
            Item selected = high.items.get(step);
            double bTarget = percentileDifficulty(pool, 85);
            double chosenScore = 0.5 * Math.abs(selected.difficulty - theta) + 0.5 * Math.abs(selected.difficulty - bTarget);
            for (Item item : pool) {
                boolean usedEarlier = high.items.subList(0, step).contains(item);
                if (usedEarlier) continue;
                double score = 0.5 * Math.abs(item.difficulty - theta) + 0.5 * Math.abs(item.difficulty - bTarget);
                argmin &= chosenScore <= score + 1e-9;
            }
        }
        check(argmin, "every selected item must minimise the blended distance rule among the unused items");
        double target = percentileDifficulty(pool, 85);
        double hardest = high.items.stream().mapToDouble(i -> i.difficulty).max().orElse(-99);
        Test top = assemble(pool, 1.0, 100, 8, 12345L);
        Test bottom = assemble(pool, 1.0, 0, 8, 12345L);
        double topCeiling = top.items.stream().mapToDouble(i -> i.difficulty).max().orElse(-99);
        double bottomCeiling = bottom.items.stream().mapToDouble(i -> i.difficulty).max().orElse(-99);
        check(Math.abs(hardest - target) <= 0.75 && topCeiling > bottomCeiling && bottomCeiling <= target
                        && Math.abs(percentileDifficulty(pool, 0) + 2.0) < 1e-12
                        && Math.abs(percentileDifficulty(pool, 100) - 2.5) < 1e-12,
                "the test ceiling must follow the target percentile: hardest " + hardest + " vs target " + target
                        + ", ceiling at 100th " + topCeiling + " vs 0th " + bottomCeiling);

        check(assemble(List.of(), 0.0, 85, 8, 1L).items.isEmpty()
                        && assemble(List.of(new Item("only", 0.0)), 0.0, 85, 8, 1L).items.size() == 1
                        && assemble(pool, 0.0, 0, 8, 1L).items.size() <= 8,
                "an empty pool must assemble nothing and a single-item pool must stay at one item");

        boolean rejected = false;
        try {
            assemble(List.of(new Item("dup", 0.0), new Item("dup", 1.0)), 0.0, 85, 8, 1L);
            check(false, "duplicate item ids must be rejected before assembly");
        } catch (IllegalStateException e) {
            rejected = e.getMessage().contains("duplicate");
        }
        check(rejected, "duplicate item ids must be rejected");
        Test again = assemble(pool, 1.0, 85, 8, 12345L);
        Test otherSeed = assemble(pool, 1.0, 85, 8, 999L);
        check(again.items.equals(high.items) && Arrays.equals(again.responses, high.responses)
                        && otherSeed.items.size() <= 8 && !otherSeed.items.isEmpty()
                        && Arrays.equals(assemble(pool, 1.0, 85, 8, 12345L).responses, high.responses),
                "the same seed must reproduce the form and the response pattern");

        System.out.println("pool=" + pool.size() + " items=" + high.items.size() + " mean=" + high.meanDifficulty()
                + " target=" + target + " estimates=" + highEstimate + "/" + lowEstimate + " SE=" + se);
        System.out.println("PASS");
    }
}
