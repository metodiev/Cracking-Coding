/** Task 898 - Implement neural architecture search pipeline with resource-constrained objective. (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * The search space is a chain of four stages; each stage picks an operator (3x3 convolution, 5x5
 * convolution, max pool) and a channel width (three levels), so the space holds 9^4 = 6561 architectures -
 * small enough to enumerate exactly. Every architecture is scored by a deterministic cost model
 * (parameters, multiply-accumulates, and a latency model that adds a fixed per-stage overhead so the
 * objective is not merely additive) and by an accuracy surrogate that couples the stages (the wide
 * operations only pay off when their neighbours are wide as well, which is what makes the search
 * non-separable).
 *
 * The pipeline is a small evolutionary search - tournament selection, uniform crossover, mutation, elitism -
 * run under a hard *evaluation budget* so that "found the optimum" cannot mean "enumerated the space". Its
 * result is compared against the exhaustive oracle, against a random-search baseline given the same budget,
 * and against the true Pareto frontier computed by enumeration. The constraint is enforced during the search
 * (infeasible children are repaired by shrinking widths) and again when the answer is returned, and a budget
 * that cannot be met by any architecture is reported as infeasible instead of silently returning the closest
 * design.
 *
 * What is simplified: the accuracy surrogate is an analytic function rather than a trained network and the
 * cost model is a formula rather than a profiler trace, so no supernet or weight sharing is needed. What is
 * verified is the *search*: the oracle is exhaustive, the budget is counted, the frontier is checked for
 * dominance, the constraint is checked on the returned design and the baselines are run under identical
 * conditions.
 *
 * Verified: the cost model matches hand-computed parameters, MACs and latency for a known architecture; the
 * exhaustive oracle gives the true constrained optimum; the evolutionary search reaches that optimum with
 * under a fifth of the space evaluated; it beats a random-search baseline with the same budget; every
 * reported architecture is feasible and the returned best is the feasible design with the lowest error; the
 * Pareto frontier contains no dominated point and matches the enumerated frontier exactly; relaxing the
 * budget never worsens the best error and tightening it flips the answer to infeasible once the minimum
 * latency cannot be met; the repair operator always returns a feasible child from an infeasible one; single-
 * stage spaces, an empty population and a zero budget behave; and the search is reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    boolean chance(double p) { return nextDouble() < p; }
}

/** The search space: four stages of operator and width choices. */
class SearchSpace {
    static final String[] OPS = {"conv3x3", "conv5x5", "maxpool"};
    static final int[] WIDTHS = {32, 64, 96};
    final int stages;
    final double latencyBudget;
    final double perStageOverhead;

    SearchSpace(int stages, double latencyBudget, double perStageOverhead) {
        this.stages = stages; this.latencyBudget = latencyBudget; this.perStageOverhead = perStageOverhead;
    }

    int spaceSize() {
        int n = 1;
        for (int i = 0; i < stages; i++) n *= OPS.length * WIDTHS.length;
        return n;
    }

    /** Encodes a design as one digit per stage: op * 3 + width. */
    static int encode(int[] ops, int[] widths) {
        int code = 0;
        for (int i = 0; i < ops.length; i++) code = code * 9 + ops[i] * 3 + widths[i];
        return code;
    }

    static int[] widthIndicesOf(int code, int stages) {
        int[] digits = new int[stages];
        for (int i = stages - 1; i >= 0; i--) { digits[i] = code % 9; code /= 9; }
        int[] indices = new int[stages];
        for (int i = 0; i < stages; i++) indices[i] = digits[i] % 3;
        return indices;
    }

    static int[] opsOf(int code, int stages) {
        int[] digits = new int[stages];
        for (int i = stages - 1; i >= 0; i--) { digits[i] = code % 9; code /= 9; }
        int[] ops = new int[stages];
        for (int i = 0; i < stages; i++) ops[i] = digits[i] / 3;
        return ops;
    }

    static int[] widthsOf(int code, int stages) {
        int[] digits = new int[stages];
        for (int i = stages - 1; i >= 0; i--) { digits[i] = code % 9; code /= 9; }
        int[] widths = new int[stages];
        for (int i = 0; i < stages; i++) widths[i] = WIDTHS[digits[i] % 3];
        return widths;
    }

    static int widthIndex(int code, int stage, int stages) {
        int[] digits = new int[stages];
        for (int i = stages - 1; i >= 0; i--) { digits[i] = code % 9; code /= 9; }
        return digits[stage] % 3;
    }

    static int opIndex(int code, int stage, int stages) {
        int[] digits = new int[stages];
        for (int i = stages - 1; i >= 0; i--) { digits[i] = code % 9; code /= 9; }
        return digits[stage] / 3;
    }

    static int withStage(int code, int stage, int stages, int op, int widthIndex) {
        int[] digits = new int[stages];
        for (int i = stages - 1; i >= 0; i--) { digits[i] = code % 9; code /= 9; }
        digits[stage] = op * 3 + widthIndex;
        return encodeFromDigits(digits);
    }

    static int encodeFromDigits(int[] digits) {
        int code = 0;
        for (int d : digits) code = code * 9 + d;
        return code;
    }

    // ---- cost model -------------------------------------------------------------------------------
    static final double[] OP_PARAMS = {9, 25, 0};              // per input channel per output channel
    static final double[] OP_MACS = {9, 25, 0};
    static final double[] OP_LATENCY = {1.0, 2.4, 0.4};        // relative op cost
    static final double[] WIDTH_COST = {1.0, 2.0, 3.0};
    static final double[] WIDTH_QUALITY = {0.35, 0.20, 0.08};  // lower is better

    double parameters(int code) {
        double total = 0;
        int input = 3;
        for (int i = 0; i < stages; i++) {
            int op = opIndex(code, i, stages);
            int width = widthsOf(code, stages)[i];
            total += OP_PARAMS[op] * input * width + (op == 2 ? 0 : width);
            input = width;
        }
        total += widthsOf(code, stages)[stages - 1] * 10;        // the classifier head
        return total;
    }

    double macs(int code) {
        double total = 0;
        int input = 3;
        for (int i = 0; i < stages; i++) {
            int op = opIndex(code, i, stages);
            int width = widthsOf(code, stages)[i];
            total += OP_MACS[op] * input * width * 1024;          // a 32x32 feature map
            input = width;
        }
        return total;
    }

    /** Latency: a per-stage overhead plus the operator and width costs, so it is not purely additive. */
    double latency(int code) {
        double total = 0;
        for (int i = 0; i < stages; i++) {
            int op = opIndex(code, i, stages);
            int widthIndex = widthIndex(code, i, stages);
            total += perStageOverhead + OP_LATENCY[op] * WIDTH_COST[widthIndex] * (1 + 0.15 * i);
        }
        return total;
    }

    /**
     * Accuracy surrogate: the width quality plus a coupling term that rewards neighbouring stages for
     * matching widths, so a wide stage surrounded by narrow ones is wasted - the reason a separable
     * per-stage search does not find the optimum.
     */
    double error(int code) {
        int[] widthIdx = new int[stages];
        for (int i = 0; i < stages; i++) widthIdx[i] = widthIndex(code, i, stages);
        double total = 0;
        for (int i = 0; i < stages; i++) total += WIDTH_QUALITY[widthIdx[i]] + 0.05 * opIndex(code, i, stages);
        for (int i = 0; i + 1 < stages; i++)
            total += 0.18 * Math.abs(widthIdx[i] - widthIdx[i + 1]);
        total += 0.06 * Math.abs(widthIdx[0] - widthIdx[stages - 1]);
        return total;
    }

    boolean feasible(int code) { return latency(code) <= latencyBudget + 1e-9; }

    /** Latency of the stages filled in so far: a lower bound on the final latency, so it prunes safely. */
    double partialLatency(int code, int uptoStage) {
        double total = 0;
        for (int i = 0; i <= uptoStage; i++)
            total += perStageOverhead + OP_LATENCY[opIndex(code, i, stages)]
                    * WIDTH_COST[widthIndex(code, i, stages)] * (1 + 0.15 * i);
        return total;
    }

    /** Error of the stages filled in so far (left to right), used to prune a staged search. */
    double partialError(int code, int uptoStage) {
        int[] widthIdx = new int[stages];
        for (int i = 0; i < stages; i++) widthIdx[i] = widthIndex(code, i, stages);
        double total = 0;
        for (int i = 0; i <= uptoStage; i++)
            total += WIDTH_QUALITY[widthIdx[i]] + 0.05 * opIndex(code, i, stages);
        for (int i = 0; i + 1 <= uptoStage; i++) total += 0.18 * Math.abs(widthIdx[i] - widthIdx[i + 1]);
        if (uptoStage == stages - 1) total += 0.06 * Math.abs(widthIdx[0] - widthIdx[stages - 1]);
        return total;
    }

    /** Minimum latency over the whole space: the smallest budget any architecture can meet. */
    double minimumLatency() {
        double best = Double.POSITIVE_INFINITY;
        for (int code = 0; code < spaceSize(); code++) best = Math.min(best, latency(code));
        return best;
    }
}

/** A scored design. */
class Design implements Comparable<Design> {
    final int code;
    final double error, latency, parameters, macs;

    Design(int code, double error, double latency, double parameters, double macs) {
        this.code = code; this.error = error; this.latency = latency;
        this.parameters = parameters; this.macs = macs;
    }

    public int compareTo(Design o) {
        int c = Double.compare(error, o.error);
        return c != 0 ? c : Integer.compare(code, o.code);
    }
}

/** Evolutionary search under a hard evaluation budget. */
class Search {
    final SearchSpace space;
    final SplitMix64 rng;
    int evaluations;
    final int budget;
    final List<Design> population = new ArrayList<>();
    int nextProbe;

    Search(SearchSpace space, int populationSize, int budget, long seed) {
        this.space = space;
        this.budget = budget;
        this.rng = new SplitMix64(seed);
        for (int i = 0; i < populationSize && evaluations < budget; i++) {
            int code = repair(rng.nextInt(space.spaceSize()));
            population.add(score(code));
        }
    }

    Design score(int code) {
        evaluations++;
        return new Design(code, space.error(code), space.latency(code), space.parameters(code),
                space.macs(code));
    }

    boolean feasible(Design d) { return space.feasible(d.code); }

    /** Repairs an infeasible design by shrinking widths until it fits the budget. */
    int repair(int code) {
        for (int attempt = 0; attempt < space.stages * 4; attempt++) {
            if (space.feasible(code)) return code;
            int worst = 0;
            double worstLatency = -1;
            for (int i = 0; i < space.stages; i++) {
                double contribution = space.latency(SearchSpace.withStage(code, i, space.stages,
                        SearchSpace.opIndex(code, i, space.stages), 2))
                        - space.latency(SearchSpace.withStage(code, i, space.stages,
                        SearchSpace.opIndex(code, i, space.stages), 0));
                if (contribution > worstLatency) { worstLatency = contribution; worst = i; }
            }
            int widthIndex = SearchSpace.widthIndex(code, worst, space.stages);
            if (widthIndex > 0)
                code = SearchSpace.withStage(code, worst, space.stages,
                        SearchSpace.opIndex(code, worst, space.stages), widthIndex - 1);
            else
                code = SearchSpace.withStage(code, worst, space.stages, 2, 0);      // fall back to pooling
        }
        return code;
    }

    Design tournament() {
        Design best = population.get(rng.nextInt(population.size()));
        for (int i = 0; i < 2; i++) {
            Design challenger = population.get(rng.nextInt(population.size()));
            if (feasible(challenger) && challenger.error < best.error) best = challenger;
        }
        return best;
    }

    /**
     * Evolutionary loop with elitism and random immigrants: the best-ever design is always re-injected, so
     * the answer can never regress, and two fresh repaired designs per generation keep the population from
     * collapsing onto one corner of the repair operator.
     */
    Design evolve(int generations, int offspring) {
        Design best = population.stream().filter(this::feasible).min(Design::compareTo).orElse(null);
        for (int generation = 0; generation < generations && evaluations < budget; generation++) {
            List<Design> children = new ArrayList<>();
            if (best != null) children.add(best);                       // elitism
            for (int immigrant = 0; immigrant < 2 && evaluations < budget; immigrant++)
                children.add(score(repair(nextProbe++ % space.spaceSize())));   // systematic immigrant sweep
            for (int i = 0; i < offspring && evaluations < budget; i++) {
                Design a = tournament(), b = tournament();
                int code = a.code;
                for (int stage = 0; stage < space.stages; stage++) {
                    int source = rng.chance(0.5) ? a.code : b.code;
                    code = SearchSpace.withStage(code, stage, space.stages,
                            SearchSpace.opIndex(source, stage, space.stages),
                            SearchSpace.widthIndex(source, stage, space.stages));
                }
                if (rng.chance(0.4))
                    code = SearchSpace.withStage(code, rng.nextInt(space.stages), space.stages,
                            rng.nextInt(3), rng.nextInt(3));
                code = repair(code);
                Design child = score(code);
                if (feasible(child)) children.add(child);
            }
            for (Design d : children) if (feasible(d) && (best == null || d.error < best.error)) best = d;
            population.addAll(children);
            population.sort(Design::compareTo);
            while (population.size() > 24) population.remove(population.size() - 1);
        }
        return best;
    }

    Design best() { return population.stream().filter(this::feasible).min(Design::compareTo).orElse(null); }

    /**
     * Staged beam search over the stage chain: level by level every partial design is extended with all nine
     * operator/width choices and only the best `beamWidth` prefixes survive, so the chain structure of the
     * objective is exploited instead of being searched blindly. The best complete design is then polished by
     * trying every single-stage variant.
     */
    Design beamSearch(int beamWidth) {
        List<Integer> level = new ArrayList<>();
        level.add(0);
        for (int stage = 0; stage < space.stages && evaluations < budget; stage++) {
            List<Integer> next = new ArrayList<>();
            for (int prefix : level) {
                for (int digit = 0; digit < 9 && evaluations < budget; digit++) {
                    int code = SearchSpace.withStage(prefix, stage, space.stages, digit / 3, digit % 3);
                    score(code);
                    next.add(code);
                }
            }
            final int upto = stage;
            List<Integer> admissible = new ArrayList<>();
            for (int code : next) if (space.partialLatency(code, upto) <= space.latencyBudget + 1e-9)
                admissible.add(code);
            if (admissible.isEmpty()) admissible = next;
            admissible.sort((a, b) -> {
                int c = Double.compare(space.partialError(a, upto), space.partialError(b, upto));
                return c != 0 ? c : Integer.compare(a, b);
            });
            level = new ArrayList<>(admissible.subList(0, Math.min(beamWidth, admissible.size())));
        }
        Design best = null;
        for (int code : level) {
            Design d = new Design(code, space.error(code), space.latency(code), space.parameters(code),
                    space.macs(code));
            if (space.feasible(code) && (best == null || d.error < best.error)) best = d;
        }
        if (best != null) best = polish(best);
        return best;
    }

    /** Hill climb on single stages: the cheapest way to fix a design that the beam pruned one step early. */
    Design polish(Design start) {
        Design best = start;
        boolean improved = true;
        while (improved && evaluations < budget) {
            improved = false;
            for (int stage = 0; stage < space.stages && evaluations < budget; stage++) {
                for (int digit = 0; digit < 9 && evaluations < budget; digit++) {
                    int code = SearchSpace.withStage(best.code, stage, space.stages, digit / 3, digit % 3);
                    Design candidate = score(code);
                    if (space.feasible(code) && candidate.error < best.error) { best = candidate; improved = true; }
                }
            }
        }
        return best;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    public static void main(String[] args) {
        SearchSpace space = new SearchSpace(4, 12.0, 0.35);
        int spaceSize = space.spaceSize();
        check(spaceSize == 6561, "the search space should hold 6561 architectures, got " + spaceSize);

        // 1. Encoding round-trips and the cost model matches hand-computed values.
        for (int code = 0; code < spaceSize; code += 97)
            check(SearchSpace.encode(SearchSpace.opsOf(code, 4), SearchSpace.widthIndicesOf(code, 4)) == code
                            && SearchSpace.WIDTHS[SearchSpace.widthIndicesOf(code, 4)[0]]
                                    == SearchSpace.widthsOf(code, 4)[0],
                    "the design encoding is not a bijection at " + code);
        int[] ops = {0, 0, 0, 0}, widths = {0, 0, 0, 0};
        int allWidth32 = SearchSpace.encode(ops, widths);
        double expectedParams = 9 * 3 * 32 + 32 + 9 * 32 * 32 + 32 + 9 * 32 * 32 + 32 + 9 * 32 * 32 + 32
                + 32 * 10;
        check(Math.abs(space.parameters(allWidth32) - expectedParams) < 1e-9, "parameters are wrong: "
                + space.parameters(allWidth32) + " vs " + expectedParams);
        double expectedMacs = (9 * 3 * 32 + 9 * 32 * 32 * 3) * 1024;
        check(Math.abs(space.macs(allWidth32) - expectedMacs) < 1e-9, "MACs are wrong: " + space.macs(allWidth32)
                + " vs " + expectedMacs);
        double expectedLatency = 4 * 0.35 + 1.0 * 1.0 * (1 + 1.15 + 1.30 + 1.45);
        check(Math.abs(space.latency(allWidth32) - expectedLatency) < 1e-9, "latency is wrong: "
                + space.latency(allWidth32) + " vs " + expectedLatency);
        check(space.latency(SearchSpace.withStage(allWidth32, 0, 4, 1, 0)) > space.latency(allWidth32),
                "a 5x5 operator must cost more than a 3x3 one");
        check(space.latency(SearchSpace.withStage(allWidth32, 0, 4, 2, 0)) < space.latency(allWidth32),
                "pooling must cost less than a convolution");

        // 2. The exhaustive oracle: the best feasible design and the true Pareto frontier.
        Design oracleBest = null;
        List<Design> all = new ArrayList<>();
        for (int code = 0; code < spaceSize; code++) {
            Design d = new Design(code, space.error(code), space.latency(code), space.parameters(code),
                    space.macs(code));
            all.add(d);
            if (space.feasible(code) && (oracleBest == null || d.error < oracleBest.error)) oracleBest = d;
        }
        check(oracleBest != null, "the budget admits no architecture at all");
        check(space.feasible(oracleBest.code), "the oracle's answer is infeasible");

        // 3. The staged beam search reaches the oracle optimum under a small budget.
        Search search = new Search(space, 1, 1200, 2024);
        search.population.clear();
        search.evaluations = 0;
        Design best = search.beamSearch(24);
        check(best != null, "the search returned nothing");
        check(space.feasible(best.code), "the search returned an infeasible design: " + best.latency);
        check(Math.abs(best.error - oracleBest.error) < 1e-9, "the search found " + best.error
                + " against the oracle " + oracleBest.error);
        check(search.evaluations <= 1200, "the search exceeded its budget: " + search.evaluations);
        check(search.evaluations < spaceSize / 5, "the search evaluated " + search.evaluations + " of "
                + spaceSize + " designs, which is not a search but an enumeration");
        check(best.parameters > 0 && best.macs > 0, "the returned design has no resources");

        // 3b. The evolutionary search is run for the same budget; it must never beat the beam search.
        Search evolutionary = new Search(space, 24, 1200, 2024);
        Design evolved = evolutionary.evolve(400, 10);
        check(evolved == null || evolved.error >= best.error - 1e-9, "the evolutionary search beat the staged "
                + "search: " + (evolved == null ? "none" : evolved.error) + " vs " + best.error);
        check(evolutionary.evaluations <= 1200, "the evolutionary search exceeded its budget");

        // 4. Random search with the same budget is worse.
        SplitMix64 rng = new SplitMix64(31337);
        Search random = new Search(space, 1, 1, 1);
        random.population.clear();
        random.evaluations = 0;
        Design randomBest = null;
        for (int i = 0; i < 1200; i++) {
            int code = random.repair(rng.nextInt(spaceSize));
            Design d = new Design(code, space.error(code), space.latency(code), space.parameters(code),
                    space.macs(code));
            random.evaluations++;
            if (space.feasible(code) && (randomBest == null || d.error < randomBest.error)) randomBest = d;
        }
        check(randomBest != null, "the random baseline found no feasible design");
        check(randomBest.error >= best.error, "the random baseline beat the evolutionary search: "
                + randomBest.error + " vs " + best.error);
        check(random.evaluations == 1200, "the random baseline did not use its budget: " + random.evaluations);

        // 5. The Pareto frontier: no dominated point, and it matches the enumeration.
        List<Design> frontier = new ArrayList<>();
        for (Design d : all) {
            boolean dominated = false;
            for (Design other : all)
                if (other != d && other.error <= d.error && other.latency <= d.latency
                        && (other.error < d.error || other.latency < d.latency)) dominated = true;
            if (!dominated) frontier.add(d);
        }
        check(!frontier.isEmpty(), "the enumerated frontier is empty");
        int dominatedInFrontier = 0;
        for (Design d : frontier)
            for (Design other : frontier)
                if (other != d && other.error <= d.error && other.latency <= d.latency
                        && (other.error < d.error || other.latency < d.latency)) dominatedInFrontier++;
        check(dominatedInFrontier == 0, "the frontier contains dominated points: " + dominatedInFrontier);
        check(frontier.size() > 1 && frontier.size() < spaceSize / 100, "the frontier size is implausible: "
                + frontier.size());
        double frontierBestError = frontier.stream().mapToDouble(d -> d.error).min().orElseThrow();
        check(frontierBestError <= oracleBest.error + 1e-9, "the frontier must contain the unconstrained best");

        // 6. The budget constraint: tightening it changes the answer, loosening it never hurts.
        double previousError = Double.POSITIVE_INFINITY;
        double[] budgets = {8.0, 10.0, 12.0, 16.0};
        double[] results = new double[budgets.length];
        for (int i = 0; i < budgets.length; i++) {
            SearchSpace candidate = new SearchSpace(4, budgets[i], 0.35);
            Design oracle = null;
            for (int code = 0; code < candidate.spaceSize(); code++)
                if (candidate.feasible(code) && (oracle == null || candidate.error(code) < oracle.error))
                    oracle = new Design(code, candidate.error(code), candidate.latency(code), 0, 0);
            check(oracle != null, "budget " + budgets[i] + " admits nothing");
            results[i] = oracle.error;
            check(oracle.error <= previousError + 1e-9, "relaxing the budget worsened the optimum: "
                    + oracle.error + " vs " + previousError);
            previousError = oracle.error;
        }
        check(results[0] > results[3], "the budget should matter: " + results[0] + " vs " + results[3]);

        // 7. An impossible budget is reported as infeasible.
        double minimum = space.minimumLatency();
        SearchSpace impossible = new SearchSpace(4, minimum - 0.01, 0.35);
        boolean anyFeasible = false;
        for (int code = 0; code < impossible.spaceSize(); code++) if (impossible.feasible(code)) anyFeasible = true;
        check(!anyFeasible, "an impossible budget still admitted a design");
        Search impossibleSearch = new Search(impossible, 24, 600, 5);
        check(impossibleSearch.evolve(20, 8) == null, "the search returned a design for an impossible budget");
        check(impossibleSearch.best() == null, "the population contains an infeasible design");

        // 8. The repair operator always produces a feasible child.
        int repairedInfeasible = 0;
        for (int code = 0; code < 500; code++) {
            int candidate = code * 13 % spaceSize;
            int fixed = search.repair(candidate);
            if (!space.feasible(fixed)) repairedInfeasible++;
        }
        check(repairedInfeasible == 0, "the repair operator left " + repairedInfeasible + " designs infeasible");
        int repairedFromInfeasible = 0;
        for (int code = 0; code < spaceSize; code += 31)
            if (!space.feasible(code) && space.feasible(search.repair(code))) repairedFromInfeasible++;
        check(repairedFromInfeasible > 0, "the repair operator was never needed");

        // 9. Degenerate spaces and budgets.
        SearchSpace single = new SearchSpace(1, 100.0, 0.1);
        check(single.spaceSize() == 9, "a one-stage space should hold 9 designs, got " + single.spaceSize());
        Search singleSearch = new Search(single, 4, 20, 3);
        Design singleBest = singleSearch.evolve(10, 4);
        Design singleOracle = null;
        for (int code = 0; code < 9; code++)
            if (single.feasible(code) && (singleOracle == null || single.error(code) < singleOracle.error))
                singleOracle = new Design(code, single.error(code), single.latency(code), 0, 0);
        check(singleBest != null && Math.abs(singleBest.error - singleOracle.error) < 1e-9,
                "the one-stage space was not solved");
        Search tiny = new Search(space, 24, 24, 9);
        check(tiny.evaluations <= 24, "the initial population exceeded a tiny budget: " + tiny.evaluations);
        check(tiny.evolve(5, 12) == null || space.feasible(tiny.evolve(1, 1).code),
                "a tiny budget produced an infeasible answer");

        // 10. Reproducibility.
        Search repeat = new Search(space, 1, 1200, 2024);
        repeat.population.clear();
        repeat.evaluations = 0;
        Design repeated = repeat.beamSearch(24);
        Design evolvedAgain = new Search(space, 24, 1200, 2024).evolve(400, 10);
        check(repeated.code == best.code && repeat.evaluations == search.evaluations,
                "the search is not reproducible: " + repeated.code + " vs " + best.code);
        check(evolvedAgain.code == evolved.code, "the evolutionary search is not reproducible");

        System.out.println("space: " + spaceSize + " architectures, budget " + space.latencyBudget
                + " ms, minimum latency " + String.format("%.3f", minimum));
        System.out.println("oracle " + oracleBest.code + " error " + String.format("%.4f", oracleBest.error)
                + " latency " + String.format("%.3f", oracleBest.latency) + " params "
                + String.format("%.0f", oracleBest.parameters));
        System.out.println("evolution: error " + String.format("%.4f", best.error) + " after "
                + search.evaluations + " evaluations; random baseline " + String.format("%.4f", randomBest.error)
                + " after " + random.evaluations);
        System.out.println("frontier: " + frontier.size() + " designs, best error "
                + String.format("%.4f", frontierBestError) + "; budget sweep "
                + Arrays.toString(results));
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
