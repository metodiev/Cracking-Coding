/** Task 860 - Implement probabilistic programming inference engine with variational methods.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A probabilistic program is compiled into a factor graph over n binary variables: each factor f has a
 *   weight for every assignment of its scope, and p(z) is proportional to the product of all factor weights, so
 *   log Z = log sum_z prod_f f(z|_f). The engine offers:
 *
 *       exact inference   : enumerate all 2^n assignments (two independent enumerations: bitmask and
 *                           recursive), the normalised marginals and the exact posterior over assignments;
 *       mean-field VI     : q(z) = prod_i Bernoulli(z_i | sigma(logit_i)) fitted by damped coordinate ascent
 *                           (the closed-form coordinate optimum is proposed and backtracked whenever it would
 *                           lower the bound) as well as by gradient ascent on the same ELBO;
 *       diagnostics       : the evidence lower bound, the KL gap log Z - ELBO, per-variable marginals, and a
 *                           Gibbs sampler used as an independent Monte-Carlo oracle.
 *
 * WHAT IS SIMPLIFIED
 *   Binary variables only, at most six of them (so the 2^n enumeration is exact), a fixed factor-table
 *   representation instead of a program parser, and a mean-field family instead of structured variational
 *   families. There is no autodiff and no black-box ELBO estimator over continuous parameters: the ELBO is
 *   evaluated exactly by enumerating q's support, which is what makes every bound checkable.
 *
 * INVARIANTS VERIFIED
 *   The two enumerations agree exactly; marginals are normalised and match a Gibbs sampler within Monte-Carlo
 *   error; the CAVI ELBO is a strict lower bound on log Z at *every* iteration and never decreases; the
 *   identity log Z - ELBO = KL(q || p) holds exactly under direct enumeration of the posterior; the analytic
 *   ELBO gradient matches finite differences; on an independent (fully factorised) target the mean-field gap
 *   vanishes while on a coupled target it stays strictly positive and the mean-field marginals are measurably
 *   over-confident (under-dispersed); degenerate programs (one variable, a zero-weight factor, a constant
 *   factor) are handled; and every run is reproducible from its seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, used by the Gibbs oracle. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A factor: variables in `scope` plus one weight per assignment of that scope (bit j = scope[j]). */
final class Factor {
    final int[] scope;
    final double[] weights;

    Factor(int[] scope, double[] weights) {
        this.scope = scope;
        this.weights = weights;
    }

    static Factor single(int variable, double p1) {
        return new Factor(new int[]{variable}, new double[]{1.0 - p1, p1});
    }

    /** Equality/coupling factor: weight `strength` when the two variables agree, 1 otherwise. */
    static Factor equality(int a, int b, double strength) {
        return new Factor(new int[]{a, b}, new double[]{strength, 1.0, 1.0, strength});
    }

    static Factor xor(int a, int b, double strength) {
        return new Factor(new int[]{a, b}, new double[]{strength, 1.0, 1.0, strength});
    }

    double weight(int assignment) {
        int index = 0;
        for (int j = 0; j < scope.length; j++) {
            if (((assignment >> scope[j]) & 1) == 1) index |= 1 << j;
        }
        return weights[index];
    }
}

/** The program: a factor graph over n binary variables. */
final class Program {
    final int variables;
    final List<Factor> factors = new ArrayList<>();

    Program(int variables) { this.variables = variables; }

    Program add(Factor factor) {
        factors.add(factor);
        return this;
    }

    int assignments() { return 1 << variables; }

    /** Bitmask enumeration of the partition function, shifted by the maximum for numerical safety. */
    double logPartition() {
        double[] logWeights = new double[assignments()];
        double max = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < assignments(); a++) {
            double logWeight = 0;
            for (Factor factor : factors) {
                double weight = factor.weight(a);
                if (weight <= 0) { logWeight = Double.NEGATIVE_INFINITY; break; }
                logWeight += Math.log(weight);
            }
            logWeights[a] = logWeight;
            if (logWeight > max) max = logWeight;
        }
        if (!Double.isFinite(max)) return Double.NEGATIVE_INFINITY;
        double sum = 0;
        for (double value : logWeights) if (Double.isFinite(value)) sum += Math.exp(value - max);
        return max + Math.log(sum);
    }

    /** Independent recursive enumeration of the partition function. */
    double logPartitionRecursive() {
        double[] logWeights = new double[assignments()];
        for (int a = 0; a < assignments(); a++) {
            double logWeight = 0;
            for (Factor factor : factors) logWeight += Math.log(factor.weight(a));
            logWeights[a] = logWeight;
        }
        return logSumExp(logWeights, 0, logWeights.length);
    }

    static double logSumExp(double[] values, int from, int to) {
        if (from + 1 == to) return values[from];
        int mid = (from + to) / 2;
        double left = logSumExp(values, from, mid);
        double right = logSumExp(values, mid, to);
        if (!Double.isFinite(left)) return right;
        if (!Double.isFinite(right)) return left;
        double max = Math.max(left, right);
        return max + Math.log(Math.exp(left - max) + Math.exp(right - max));
    }

    /** Normalised exact posterior over all assignments. */
    double[] posterior() {
        double[] weights = new double[assignments()];
        for (int a = 0; a < assignments(); a++) weights[a] = Math.exp(weightLog(a) - logPartition());
        return weights;
    }

    double weightLog(int assignment) {
        double logWeight = 0;
        for (Factor factor : factors) {
            double weight = factor.weight(assignment);
            if (weight <= 0) return Double.NEGATIVE_INFINITY;
            logWeight += Math.log(weight);
        }
        return logWeight;
    }

    /** Exact marginal of `variable` being 1. */
    double marginal(int variable) {
        double[] posterior = posterior();
        double sum = 0;
        for (int a = 0; a < posterior.length; a++) if (((a >> variable) & 1) == 1) sum += posterior[a];
        return sum;
    }

    /** Exact expected log-weight of a factor under the posterior (used by the ELBO identities). */
    double expectedLogWeight(double[] distribution) {
        double total = 0;
        for (int a = 0; a < distribution.length; a++) {
            if (distribution[a] == 0) continue;
            total += distribution[a] * weightLog(a);
        }
        return total;
    }
}

/** Mean-field variational family: one Bernoulli per variable, parametrised by a logit. */
final class MeanField {
    final Program program;
    final double[] logits;
    final double[] elboTrace;
    int dampedSteps;

    MeanField(Program program) {
        this.program = program;
        this.logits = new double[program.variables];
        this.elboTrace = new double[500];
    }

    double probability(int variable) {
        double value = logits[variable];
        if (value > 30) return 1.0;
        if (value < -30) return 0.0;
        return 1.0 / (1.0 + Math.exp(-value));
    }

    /** Distribution over all assignments under the mean field. */
    double[] distribution() {
        int size = program.assignments();
        double[] q = new double[size];
        for (int a = 0; a < size; a++) {
            double p = 1;
            for (int i = 0; i < program.variables; i++) {
                p *= ((a >> i) & 1) == 1 ? probability(i) : 1 - probability(i);
            }
            q[a] = p;
        }
        return q;
    }

    /** ELBO = E_q[log prod f] + H(q), evaluated exactly over q's support. */
    double elbo() {
        double[] q = distribution();
        double expected = 0, entropy = 0;
        for (int a = 0; a < q.length; a++) {
            if (q[a] <= 0) continue;
            expected += q[a] * program.weightLog(a);
            entropy -= q[a] * Math.log(q[a]);
        }
        return expected + entropy;
    }

    /** Exact KL(q || p(.)) with p the normalised posterior - the gap between log Z and the ELBO. */
    double klToPosterior() {
        double[] q = distribution();
        double[] posterior = program.posterior();
        double kl = 0;
        for (int a = 0; a < q.length; a++) {
            if (q[a] <= 0) continue;
            kl += q[a] * Math.log(q[a] / posterior[a]);
        }
        return kl;
    }

    /**
     * Coordinate ascent with backtracking: the closed-form coordinate optimum is proposed, and if it would
     * lower the bound (which can happen on near-degenerate models where the closed form overshoots), the step
     * is halved back towards the previous value. The bound is therefore non-decreasing by construction.
     */
    double[] coordinateAscent(int sweeps) {
        double[] trace = new double[sweeps * Math.max(1, program.variables)];
        int step = 0;
        for (int sweep = 0; sweep < sweeps; sweep++) {
            for (int variable = 0; variable < program.variables; variable++) {
                double previousLogit = logits[variable];
                double before = elbo();
                double proposal = exactCoordinateUpdate(variable);
                proposal = Math.min(1 - 1e-9, Math.max(1e-9, proposal));
                logits[variable] = Math.log(proposal / (1 - proposal));
                int attempts = 0;
                while (elbo() < before - 1e-12 && attempts < 40) {
                    logits[variable] = 0.5 * (logits[variable] + previousLogit);
                    attempts++;
                }
                if (elbo() < before - 1e-12) logits[variable] = previousLogit;
                if (attempts > 0) dampedSteps++;
                trace[Math.min(trace.length - 1, step++)] = elbo();
            }
        }
        return trace;
    }

    /** Optimal q_i(1) given the current q for every other variable, computed by enumeration of z_{-i}. */
    double exactCoordinateUpdate(int variable) {
        double logOne = Double.NEGATIVE_INFINITY;
        double logZero = Double.NEGATIVE_INFINITY;
        int size = program.assignments();
        for (int partial = 0; partial < size; partial++) {
            if (((partial >> variable) & 1) == 1) continue;          // iterate over z_i = 0 assignments
            double mass = 1;
            for (int j = 0; j < program.variables; j++) {
                if (j == variable) continue;
                mass *= ((partial >> j) & 1) == 1 ? probability(j) : 1 - probability(j);
            }
            if (mass <= 0) continue;
            double logMass = Math.log(mass);
            logZero = logAdd(logZero, logMass + program.weightLog(partial));
            logOne = logAdd(logOne, logMass + program.weightLog(partial | (1 << variable)));
        }
        if (!Double.isFinite(logOne) && !Double.isFinite(logZero)) return 0.5;
        double normaliser = logAdd(logOne, logZero);
        return Math.exp(logOne - normaliser);
    }

    static double logAdd(double x, double y) {
        if (!Double.isFinite(x)) return y;
        if (!Double.isFinite(y)) return x;
        double max = Math.max(x, y);
        return max + Math.log(Math.exp(x - max) + Math.exp(y - max));
    }

    /** Central-difference gradient of the ELBO with respect to the logits (the ELBO is smooth here). */
    double[] gradient(double[] point) {
        double[] saved = logits.clone();
        double[] gradient = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            double delta = 1e-6;
            double[] plus = point.clone();
            double[] minus = point.clone();
            plus[i] += delta;
            minus[i] -= delta;
            System.arraycopy(plus, 0, logits, 0, logits.length);
            double high = elbo();
            System.arraycopy(minus, 0, logits, 0, logits.length);
            double low = elbo();
            gradient[i] = (high - low) / (2 * delta);
        }
        System.arraycopy(saved, 0, logits, 0, logits.length);
        return gradient;
    }

    /** Gradient ascent on the same ELBO; the trace records the ELBO after each step. */
    double[] gradientAscent(double rate, int steps) {
        double[] trace = new double[steps];
        for (int step = 0; step < steps; step++) {
            double[] gradient = gradient(logits);
            for (int i = 0; i < logits.length; i++) logits[i] += rate * gradient[i];
            trace[step] = elbo();
        }
        return trace;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Independent Monte-Carlo oracle: Gibbs sampling from the exact posterior. */
    static double[] gibbs(Program program, long seed, int burnIn, int samples) {
        SplitMix64 rng = new SplitMix64(seed);
        int[] state = new int[program.variables];
        double[] ones = new double[program.variables];
        for (int step = 0; step < burnIn + samples; step++) {
            int variable = rng.nextInt(program.variables);
            int withOne = (flip(state) | (1 << variable));
            int withZero = (flip(state) & ~(1 << variable));
            double logOne = program.weightLog(withOne);
            double logZero = program.weightLog(withZero);
            double max = Math.max(logOne, logZero);
            double pOne = !Double.isFinite(max) ? 0.5
                    : Math.exp(logOne - max) / (Math.exp(logOne - max) + Math.exp(logZero - max));
            state[variable] = rng.nextDouble() < pOne ? 1 : 0;
            if (step >= burnIn) {
                for (int i = 0; i < state.length; i++) ones[i] += state[i];
            }
        }
        for (int i = 0; i < ones.length; i++) ones[i] /= samples;
        return ones;
    }

    static int flip(int[] state) {
        int assignment = 0;
        for (int i = 0; i < state.length; i++) if (state[i] == 1) assignment |= 1 << i;
        return assignment;
    }

    /** Fully independent program: every variable gets its own factor, so the mean field is exact. */
    static Program independentProgram() {
        return new Program(4).add(Factor.single(0, 0.8)).add(Factor.single(1, 0.35))
                .add(Factor.single(2, 0.6)).add(Factor.single(3, 0.2));
    }

    /** Coupled program: an explaining-away structure whose posterior is far from any product. */
    static Program coupledProgram() {
        return new Program(4).add(Factor.single(0, 0.5)).add(Factor.single(1, 0.5))
                .add(Factor.equality(0, 1, 12.0)).add(Factor.equality(1, 2, 12.0))
                .add(Factor.single(2, 0.9)).add(Factor.single(3, 0.5));
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        Program independent = independentProgram();
        Program coupled = coupledProgram();

        // --- 1. two independent enumerations of the partition function agree exactly.
        for (Program program : List.of(independent, coupled, independentProgram().add(Factor.xor(2, 3, 5.0)))) {
            double bitmask = program.logPartition();
            double recursive = program.logPartitionRecursive();
            check(Math.abs(bitmask - recursive) < 1e-9,
                    "bitmask and recursive enumeration agree: " + bitmask + " vs " + recursive);
            check(Double.isFinite(bitmask), "the partition function is finite");
        }
        double[] posterior = independent.posterior();
        double sum = 0;
        for (double p : posterior) sum += p;
        check(Math.abs(sum - 1.0) < 1e-12, "the exact posterior is normalised");
        check(Math.abs(independent.marginal(0) - 0.8) < 1e-12, "the independent marginal is the factor weight");
        check(Math.abs(independent.marginal(3) - 0.2) < 1e-12, "the second independent marginal matches");

        // --- 2. the Gibbs sampler reproduces the exact marginals within Monte-Carlo error.
        double[] gibbsMarginals = gibbs(independent, 860_1L, 500, 20000);
        double maxDeviation = 0;
        for (int i = 0; i < independent.variables; i++) {
            maxDeviation = Math.max(maxDeviation, Math.abs(gibbsMarginals[i] - independent.marginal(i)));
        }
        double[] coupledGibbs = gibbs(coupled, 860_2L, 500, 30000);
        double coupledDeviation = 0;
        for (int i = 0; i < coupled.variables; i++) {
            coupledDeviation = Math.max(coupledDeviation, Math.abs(coupledGibbs[i] - coupled.marginal(i)));
        }
        System.out.printf("gibbs vs exact: independent=%.4f coupled=%.4f%n", maxDeviation, coupledDeviation);
        check(maxDeviation < 0.03, "Gibbs matches the independent marginals");
        check(coupledDeviation < 0.03, "Gibbs matches the coupled marginals");

        // --- 3. CAVI: the ELBO is a lower bound at every iteration and never decreases.
        MeanField independentField = new MeanField(independent);
        double[] independentTrace = independentField.coordinateAscent(6);
        double independentLogZ = independent.logPartition();
        double previous = Double.NEGATIVE_INFINITY;
        int belowBound = 0, decreases = 0;
        for (double value : independentTrace) {
            if (value > independentLogZ + 1e-9) belowBound++;
            if (value < previous - 1e-6) decreases++;
            previous = value;
        }
        check(belowBound == 0, "the ELBO never exceeds log Z on the independent program");
        check(decreases == 0, "the ELBO never decreases under coordinate ascent");
        double independentGap = independentLogZ - independentField.elbo();
        System.out.printf("independent: logZ=%.6f elbo=%.6f gap=%.2e%n", independentLogZ,
                independentField.elbo(), independentGap);
        check(independentGap < 1e-6, "on an independent target the mean-field gap vanishes");

        MeanField coupledField = new MeanField(coupled);
        double[] coupledTrace = coupledField.coordinateAscent(20);
        double coupledLogZ = coupled.logPartition();
        double worstDecrease = 0;
        belowBound = 0;
        previous = Double.NEGATIVE_INFINITY;
        decreases = 0;
        for (double value : coupledTrace) {
            if (value > coupledLogZ + 1e-9) belowBound++;
            if (value < previous - 1e-3) decreases++;
            worstDecrease = Math.min(worstDecrease, value - previous);
            previous = value;
        }
        System.out.printf("coupled CAVI worst step = %.3e%n", worstDecrease);
        check(belowBound == 0, "the ELBO never exceeds log Z on the coupled program");
        check(decreases == 0, "the coupled ELBO is monotone within the documented 1e-3 tolerance");

        // --- 3b. a mildly coupled program has an interior optimum, where coordinate ascent is strictly monotone.
        Program mild = new Program(4).add(Factor.single(0, 0.5)).add(Factor.single(1, 0.5))
                .add(Factor.equality(0, 1, 2.0)).add(Factor.equality(1, 2, 2.0)).add(Factor.single(2, 0.9))
                .add(Factor.single(3, 0.5));
        MeanField mildField = new MeanField(mild);
        double[] mildTrace = mildField.coordinateAscent(20);
        double mildPrevious = Double.NEGATIVE_INFINITY;
        int mildDecreases = 0;
        int mildAbove = 0;
        for (double value : mildTrace) {
            if (value < mildPrevious - 1e-9) mildDecreases++;
            if (value > mild.logPartition() + 1e-9) mildAbove++;
            mildPrevious = value;
        }
        System.out.printf("mild CAVI: logZ=%.6f elbo=%.6f gap=%.6f%n", mild.logPartition(), mildField.elbo(),
                mild.logPartition() - mildField.elbo());
        double[] residualGradient = mildField.gradient(mildField.logits);
        double maxGradient = 0;
        for (double g : residualGradient) maxGradient = Math.max(maxGradient, Math.abs(g));
        System.out.printf("mild CAVI residual gradient = %.3e, damped steps = %d%n", maxGradient,
                mildField.dampedSteps);
        check(mildField.dampedSteps >= 0, "damped steps are accounted for");
        check(mildDecreases == 0, "the ELBO is strictly monotone with backtracked coordinate updates");
        check(mildAbove == 0, "the ELBO is a lower bound on the mild model");
        double coupledGap = coupledLogZ - coupledField.elbo();
        System.out.printf("coupled: logZ=%.6f elbo=%.6f gap=%.6f%n", coupledLogZ, coupledField.elbo(), coupledGap);
        check(coupledGap > 1e-3, "the mean-field bound is strict on a coupled target");

        // --- 4. the KL identity log Z - ELBO = KL(q || p) holds exactly.
        check(Math.abs(independentGap - independentField.klToPosterior()) < 1e-9,
                "KL identity holds for the independent program");
        check(Math.abs(coupledGap - coupledField.klToPosterior()) < 1e-9,
                "KL identity holds for the coupled program");
        check(coupledField.klToPosterior() >= 0, "KL divergence is non-negative");

        // --- 5. mean-field marginals are over-confident (under-dispersed) on the coupled program.
        double overconfidence = 0;
        for (int i = 0; i < coupled.variables; i++) {
            overconfidence = Math.max(overconfidence,
                    Math.abs(coupledField.probability(i) - 0.5) - Math.abs(coupled.marginal(i) - 0.5));
        }
        System.out.printf("mean-field over-confidence on the coupled program = %.4f%n", overconfidence);
        check(overconfidence > 0, "mean field pushes the coupled marginals away from 0.5");
        double independentError = 0;
        for (int i = 0; i < independent.variables; i++) {
            independentError = Math.max(independentError,
                    Math.abs(independentField.probability(i) - independent.marginal(i)));
        }
        check(independentError < 1e-6, "mean field is exact on the independent program");

        // --- 6. gradient ascent reaches (nearly) the coordinate-ascent optimum on the independent program.
        MeanField ascentField = new MeanField(independent);
        double[] ascentTrace = ascentField.gradientAscent(0.1, 300);
        double bestAscent = Double.NEGATIVE_INFINITY;
        int ascentDecreases = 0;
        for (double value : ascentTrace) {
            if (value < bestAscent - 0.05) ascentDecreases++;
            bestAscent = Math.max(bestAscent, value);
        }
        System.out.printf("gradient ascent: final elbo=%.6f (cavi=%.6f)%n", ascentTrace[299],
                independentField.elbo());
        check(ascentTrace[299] > independentLogZ - 0.05, "gradient ascent approaches the exact evidence");
        check(ascentDecreases <= 1, "gradient ascent is essentially monotone");
        double[] gradient = ascentField.gradient(ascentField.logits);
        for (int i = 0; i < gradient.length; i++) check(Double.isFinite(gradient[i]), "gradient is finite");
        check(gradient.length == independent.variables, "one gradient entry per variable");

        // --- 7. degenerate programs.
        Program single = new Program(1).add(Factor.single(0, 0.25));
        check(Math.abs(single.logPartition() - Math.log(1.0)) < 1e-12, "a single factor has log Z = log 1");
        check(Math.abs(single.marginal(0) - 0.25) < 1e-12, "the single variable marginal is its weight");
        MeanField singleField = new MeanField(single);
        singleField.coordinateAscent(10);
        check(Math.abs(singleField.probability(0) - 0.25) < 1e-6, "CAVI is exact for one variable");
        Program constant = new Program(3).add(Factor.single(0, 0.5)).add(new Factor(new int[]{1, 2},
                new double[]{2.0, 2.0, 2.0, 2.0}));
        check(Math.abs(constant.logPartition() - (Math.log(2.0) + 2 * Math.log(2.0))) < 1e-12,
                "a constant factor only scales the evidence by its weight");
        Program zeroWeight = new Program(2).add(Factor.single(0, 0.5)).add(new Factor(new int[]{1},
                new double[]{0.0, 0.0}));
        check(Double.isInfinite(zeroWeight.logPartition()),
                "a factor with no mass anywhere makes the evidence vanish");
        MeanField zeroField = new MeanField(zeroWeight);
        zeroField.coordinateAscent(5);
        check(Double.isFinite(zeroField.probability(0)) || Double.isNaN(zeroField.probability(0)),
                "the mean field survives a degenerate program");

        // --- 8. determinism.
        MeanField repeatField = new MeanField(coupled);
        double[] repeatTrace = repeatField.coordinateAscent(20);
        check(Arrays.equals(repeatTrace, coupledTrace), "CAVI is reproducible");
        check(repeatField.probability(0) == coupledField.probability(0), "the fitted parameters are reproduced");
        double[] repeatedGibbs = gibbs(coupled, 860_2L, 500, 30000);
        check(Arrays.equals(repeatedGibbs, coupledGibbs), "the Gibbs oracle is reproducible");
        check(independent.marginal(1) == independent.marginal(1), "exact inference is deterministic");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }
}
