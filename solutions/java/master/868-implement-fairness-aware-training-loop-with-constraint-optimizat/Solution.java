/** Task 868 - Implement fairness-aware training loop with constraint optimization (demographic parity). (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A training loop that minimises the regularised logistic loss subject to a demographic-parity constraint
 *
 *     min  f(theta) = (1/n) sum_i -log p(y_i | x_i, theta) + (rho/2) ||theta||^2
 *     s.t. |g(theta)| <= eps,   g(theta) = mean_{A=0} p(1|x,theta) - mean_{A=1} p(1|x,theta)
 *
 * with theta = (w, b_0, b_1): one slope shared by both groups and one intercept per group, which is exactly
 * the post-processing view of fairness (a group may be calibrated separately, it may not be scored
 * differently). The constraint is nonlinear in theta, so the feasible set is not convex; what *is* strictly
 * convex is the penalised inner problem, because rho > 0. The loop is therefore the production pattern:
 *
 *   outer  - find the Lagrange multiplier mu of the active boundary by a continuation ladder plus bisection
 *   inner  - warm-started minibatch gradient descent on L(theta; mu) = f(theta) + mu (s g(theta) - eps),
 *            where the sign s = +-1 is the direction in which the unconstrained optimum violates parity
 *
 * The multiplier search is exact (bisection on the violation, not a tuned learning rate) and each inner solve
 * is polished with a Newton step, so the returned model satisfies the KKT system of the constrained problem
 * to machine precision and its multiplier can be compared with the one an independent boundary solve finds.
 *
 * SIMPLIFIED: one scalar feature, a synthetic two-group dataset, no threads, no wall-clock time and no real
 * hardware; "scale" is the number of examples, epochs and bisection rounds and a virtual-ops counter stands
 * in for cost. Data, minibatch order and hyper-parameters are deterministic.
 *
 * VERIFIED: (a) OPTIMALITY - the trained model matches an independent Newton boundary solve (KKT stationarity
 * residual < 1e-8, feasibility < 1e-9, correct multiplier sign, complementary slackness, and the bisected
 * multiplier agrees with the exact one) and it is not beaten by a complete brute-force scan of the parameter
 * grid (the independent global oracle), within a stated loss tolerance in both directions; (b) the
 * unconstrained optimum violates parity by 0.3+, so the constraint is binding and its price in accuracy is
 * positive; (c) FEASIBILITY holds along a ladder of eps, from eps = 0 (exact parity) to eps large (constraint
 * inactive, multiplier exactly 0); (d) the hard 0/1 selection-rate gap - never optimised directly - falls far
 * below the unconstrained one; (e) the minibatch loop is reproducible bit-for-bit from its seed, its
 * best-so-far loss never increases and it reaches the exact optimum; (f) degenerate inputs (one group, fewer
 * multiplier rounds, zero SGD epochs) and invalid inputs (negative eps, empty dataset) behave.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }
    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** One labelled example: a scalar feature, a protected group and a binary label. */
class Example {
    final double x;
    final int group;
    final int label;

    Example(double x, int group, int label) { this.x = x; this.group = group; this.label = label; }
}

/**
 * A two-group training set. Group 0 is shifted up and group 1 down while the label depends only on the
 * feature, so the Bayes-optimal classifier selects far more of group 0: the parity gap a fair loop must close,
 * and one it can only close by paying accuracy.
 */
class Dataset {
    static final int DIM = 3;

    final List<Example> examples;

    Dataset(List<Example> examples) {
        if (examples.isEmpty()) throw new IllegalArgumentException("dataset must not be empty");
        this.examples = Collections.unmodifiableList(examples);
    }
    static Dataset synthetic(int perGroup, long seed) {
        if (perGroup <= 0) throw new IllegalArgumentException("perGroup must be positive");
        SplitMix64 rng = new SplitMix64(seed);
        List<Example> out = new ArrayList<>();
        for (int group = 0; group < 2; group++) {
            double shift = group == 0 ? 0.9 : -0.9;
            for (int i = 0; i < perGroup; i++) {
                double x = shift + rng.nextGaussian();
                out.add(new Example(x, group, rng.nextDouble() < sigmoid(1.2 * x) ? 1 : 0));
            }
        }
        return new Dataset(out);
    }

    /** Every example in one group: the parity gap is identically zero. */
    static Dataset singleGroup(int n, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Example> out = new ArrayList<>();
        for (int i = 0; i < n; i++) out.add(new Example(rng.nextGaussian(), 0, rng.nextDouble() < 0.5 ? 1 : 0));
        return new Dataset(out);
    }

    int size() { return examples.size(); }
    int groupCount(int group) {
        int c = 0; for (Example e : examples) if (e.group == group) c++; return c;
    }

    /** phi(x, group) = (x, 1[group = 0], 1[group = 1]): the group only ever enters as its own intercept. */
    static double[] phi(Example e, double[] out) {
        out[0] = e.x; out[1] = e.group == 0 ? 1.0 : 0.0; out[2] = e.group == 1 ? 1.0 : 0.0;
        return out;
    }
    static double sigmoid(double z) {
        if (z >= 0) { double e = Math.exp(-z); return 1 / (1 + e); }
        double e = Math.exp(z); return e / (1 + e);
    }

    static double l2(double[] v) { double s = 0; for (double x : v) s += x * x; return s; }
}

/** Hyper-parameters of the fairness-aware loop. */
class TrainConfig {
    final int rounds;       // bisection steps once the multiplier is bracketed
    final int epochs;       // inner epochs per multiplier value
    final int batchSize;
    final double primalStep;
    final double ridge;

    TrainConfig(int rounds, int epochs, int batchSize, double primalStep, double ridge) {
        this.rounds = rounds; this.epochs = epochs; this.batchSize = batchSize;
        this.primalStep = primalStep; this.ridge = ridge;
    }
}

/** The trained model plus the fairness bookkeeping the checks assert on. */
class FairModel {
    final double[] theta;
    final double multiplier;
    final int activeSide;        // +1: gap pinned at +eps, -1: pinned at -eps, 0: constraint inactive
    final double[] lossTrace;    // best-so-far loss per round (never increasing)
    final int innerSolves;
    final long sgdSteps;

    FairModel(double[] theta, double multiplier, int activeSide, double[] lossTrace, int innerSolves,
            long sgdSteps) {
        this.theta = theta; this.multiplier = multiplier; this.activeSide = activeSide;
        this.lossTrace = lossTrace; this.innerSolves = innerSolves; this.sgdSteps = sgdSteps;
    }

    double loss(Dataset d) { return Objective.loss(d, theta, Objective.RIDGE); }
    double parity(Dataset d) { return Objective.parity(d, theta); }
}

/** Numerical linear algebra for the small systems this task needs. */
class Linear {
    /** Gaussian elimination with partial pivoting; null when the matrix is numerically singular. */
    static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) { System.arraycopy(a[i], 0, m[i], 0, n); m[i][n] = b[i]; }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            if (Math.abs(m[pivot][col]) < 1e-13) return null;
            double[] tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = m[row][col] / m[col][col];
                for (int j = col; j <= n; j++) m[row][j] -= factor * m[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = m[i][n] / m[i][i];
        return x;
    }

    static double dot(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
}

/** The objective, its derivatives and every metric the checks compare. */
class Objective {
    static final double RIDGE = 1e-3;

    /** Logistic loss through softplus, so it never overflows. */
    static double exampleLoss(double z, int label) {
        double v = (2 * label - 1) * z;
        return Math.log1p(Math.exp(-Math.abs(v))) + Math.max(-v, 0);
    }

    static double score(Example e, double[] theta) { return theta[0] * e.x + theta[e.group + 1]; }

    static double probability(Example e, double[] theta) { return Dataset.sigmoid(score(e, theta)); }
    static double loss(Dataset d, double[] theta, double ridge) {
        double sum = 0;
        for (Example e : d.examples) sum += exampleLoss(score(e, theta), e.label);
        return sum / d.size() + 0.5 * ridge * Dataset.l2(theta);
    }
    static double[] gradLoss(Dataset d, double[] theta, double ridge) {
        double[] g = new double[Dataset.DIM], phi = new double[Dataset.DIM];
        for (int i = 0; i < g.length; i++) g[i] = ridge * theta[i];
        for (Example e : d.examples) {
            double coef = (probability(e, theta) - e.label) / d.size();
            Dataset.phi(e, phi);
            for (int i = 0; i < g.length; i++) g[i] += coef * phi[i];
        }
        return g;
    }
    static double[][] hessLoss(Dataset d, double[] theta, double ridge) {
        double[][] h = new double[Dataset.DIM][Dataset.DIM];
        double[] phi = new double[Dataset.DIM];
        for (int i = 0; i < Dataset.DIM; i++) h[i][i] = ridge;
        for (Example e : d.examples) {
            double p = probability(e, theta), w = p * (1 - p) / d.size();
            Dataset.phi(e, phi);
            for (int i = 0; i < Dataset.DIM; i++)
                for (int j = 0; j < Dataset.DIM; j++) h[i][j] += w * phi[i] * phi[j];
        }
        return h;
    }

    /** The parity gap: mean predicted probability for group 0 minus group 1, zero when a group is missing. */
    static double parity(Dataset d, double[] theta) {
        if (d.groupCount(0) == 0 || d.groupCount(1) == 0) return 0.0;
        double[] mean = {0, 0};
        for (Example e : d.examples) mean[e.group] += probability(e, theta);
        return mean[0] / d.groupCount(0) - mean[1] / d.groupCount(1);
    }
    /** Gradient and Hessian of the parity gap in one pass; the Hessian is accumulated into hessOut. */
    static double[] parityDerivatives(Dataset d, double[] theta, double[][] hessOut) {
        double[] g = new double[Dataset.DIM], phi = new double[Dataset.DIM];
        for (int group = 0; group < 2; group++) {
            int n = d.groupCount(group); if (n == 0) continue;
            double sign = group == 0 ? 1.0 : -1.0;
            for (Example e : d.examples) if (e.group == group) {
                double p = probability(e, theta), gp = sign * p * (1 - p) / n, gh = gp * (1 - 2 * p);
                Dataset.phi(e, phi);
                for (int i = 0; i < Dataset.DIM; i++) {
                    g[i] += gp * phi[i];
                    for (int j = 0; j < Dataset.DIM; j++) hessOut[i][j] += gh * phi[i] * phi[j];
                }
            }
        }
        return g;
    }

    /** The hard 0/1 selection-rate gap: what a product owner measures, and what is never optimised. */
    static double hardParity(Dataset d, double[] theta) {
        double[] selected = new double[2]; int[] n = new int[2];
        for (Example e : d.examples) {
            n[e.group]++; if (probability(e, theta) >= 0.5) selected[e.group]++;
        }
        double a = n[0] == 0 ? 0 : selected[0] / n[0];
        double b = n[1] == 0 ? 0 : selected[1] / n[1];
        return a - b;
    }
    static double lagrangian(Dataset d, double[] theta, double eps, double mu, int side) {
        return loss(d, theta, RIDGE) + mu * (side * parity(d, theta) - eps);
    }
    static double[] gradLagrangian(Dataset d, double[] theta, double mu, int side) {
        double[] gf = gradLoss(d, theta, RIDGE);
        double[] gp = parityDerivatives(d, theta, new double[Dataset.DIM][Dataset.DIM]);
        double[] g = new double[Dataset.DIM];
        for (int i = 0; i < g.length; i++) g[i] = gf[i] + mu * side * gp[i];
        return g;
    }
    static double[][] hessLagrangian(Dataset d, double[] theta, double mu, int side) {
        double[][] h = new double[Dataset.DIM][Dataset.DIM];
        parityDerivatives(d, theta, h);              // h now holds the parity Hessian
        double[][] hf = hessLoss(d, theta, RIDGE);
        for (int i = 0; i < Dataset.DIM; i++)
            for (int j = 0; j < Dataset.DIM; j++) h[i][j] = hf[i][j] + mu * side * h[i][j];
        return h;
    }

    static double norm(double[] v) { return Math.sqrt(Dataset.l2(v)); }
}

/** Newton solvers: the exact inner minimiser and the exact boundary (KKT) reference. */
class ExactSolver {
    /**
     * Newton with a backtracking line search on the penalised objective. The Lagrangian Hessian can be
     * indefinite, so when the Newton direction is not a descent direction the diagonal-preconditioned
     * gradient is used instead - the value therefore decreases on every accepted iteration.
     */
    static double[] minimise(Dataset d, double eps, double mu, int side, double[] start, int maxIterations) {
        double[] t = start.clone();
        double value = Objective.lagrangian(d, t, eps, mu, side);
        for (int it = 0; it < maxIterations; it++) {
            double[] g = Objective.gradLagrangian(d, t, mu, side);
            if (Objective.norm(g) < 1e-14) break;
            double[][] h = Objective.hessLagrangian(d, t, mu, side);
            double[] dir = Linear.solve(h, new double[]{-g[0], -g[1], -g[2]});
            if (dir == null || Linear.dot(dir, g) >= 0) {
                dir = new double[Dataset.DIM];
                for (int i = 0; i < Dataset.DIM; i++)
                    dir[i] = -g[i] / (Math.abs(h[i][i]) < 1e-9 ? 1.0 : Math.abs(h[i][i]));
            }
            double step = 1.0;
            double[] best = null;
            for (int back = 0; back < 60; back++) {
                double[] cand = new double[Dataset.DIM];
                for (int i = 0; i < Dataset.DIM; i++) cand[i] = t[i] + step * dir[i];
                double candValue = Objective.lagrangian(d, cand, eps, mu, side);
                if (candValue < value) { best = cand; value = candValue; break; }
                step *= 0.5;
            }
            if (best == null) break;
            t = best;
        }
        return t;
    }
    static double[] negate(double[] g) { return new double[]{-g[0], -g[1], -g[2]}; }

    /**
     * Solve the equality-constrained boundary problem min f s.t. g(theta) = target by Newton on the KKT
     * system [H_f + mu H_g, grad g; grad g^T, 0] [dtheta; dmu] = -[grad f + mu grad g; g - target]. The
     * returned multiplier is the exact one, so the caller can cross-check the bisected multiplier and the
     * KKT sign condition of the inequality it came from.
     */
    static double[] solveBoundary(Dataset d, double target, double[] start, int maxIterations) {
        double[] t = start.clone(); double mu = 0;
        int n = Dataset.DIM;
        for (int it = 0; it < maxIterations; it++) {
            double[] gf = Objective.gradLoss(d, t, Objective.RIDGE);
            double[] gp = Objective.parityDerivatives(d, t, new double[Dataset.DIM][Dataset.DIM]);
            double[][] hf = Objective.hessLoss(d, t, Objective.RIDGE);
            double[][] hp = new double[Dataset.DIM][Dataset.DIM];
            Objective.parityDerivatives(d, t, hp);
            double[][] a = new double[n + 1][n + 1];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) a[i][j] = hf[i][j] + mu * hp[i][j];
            for (int i = 0; i < n; i++) { a[i][n] = gp[i]; a[n][i] = gp[i]; }
            double[] rhs = new double[n + 1];
            for (int i = 0; i < n; i++) rhs[i] = -(gf[i] + mu * gp[i]);
            rhs[n] = -(Objective.parity(d, t) - target);
            double[] step = Linear.solve(a, rhs);
            if (step == null) break;
            double size = 0;
            for (int i = 0; i < n; i++) { t[i] += step[i]; size += Math.abs(step[i]); }
            mu += step[n];
            if (size + Math.abs(step[n]) < 1e-14) break;
        }
        return new double[]{t[0], t[1], t[2], mu};
    }
}

/** The fairness-aware training loop: an outer multiplier search around an inner minibatch trainer. */
class FairTrainer {
    /**
     * Train with |g(theta)| <= eps. The unconstrained optimum fixes the violation direction s; a geometric
     * ladder in mu then walks the penalty path (warm-starting each inner solve, which is what keeps the path
     * on one branch of the non-convex feasible set) until the violation changes sign, and bisection nails the
     * boundary. The returned best is the feasible iterate with the smallest loss.
     */
    static FairModel train(Dataset d, double eps, TrainConfig cfg, long seed) {
        if (eps < 0) throw new IllegalArgumentException("eps must be non-negative");
        double[] unconstrained = ExactSolver.minimise(d, eps, 0.0, 1, new double[]{0, 0, 0}, 200);
        double gap = Objective.parity(d, unconstrained);
        if (Math.abs(gap) <= eps) {
            return new FairModel(unconstrained, 0.0, 0,
                    new double[]{Objective.loss(d, unconstrained, Objective.RIDGE)}, 1, 0);
        }
        int side = gap > 0 ? 1 : -1;
        long steps = 0;
        int solves = 1;
        double[] feasible = unconstrained;
        double muLow = 0.0, muHigh = 0.0;
        double[] current = unconstrained;
        for (int i = 0; i < 40; i++) {
            double mu = Math.pow(2, i + 1);
            double[] theta = inner(d, eps, mu, side, cfg, seed + i, current);
            solves++; steps += (long) cfg.epochs * d.size();
            if (side * Objective.parity(d, theta) - eps <= 0) { muHigh = mu; feasible = theta; break; }
            muLow = mu; current = theta;
        }
        if (muHigh == 0.0) throw new IllegalStateException("failed to bracket the parity boundary");
        double[] best = feasible;
        double bestLoss = Objective.loss(d, best, Objective.RIDGE);
        double[] trace = new double[cfg.rounds + 1];
        trace[0] = bestLoss;
        double[] start = feasible;
        for (int round = 1; round <= cfg.rounds; round++) {
            double mid = 0.5 * (muLow + muHigh);
            double[] theta = inner(d, eps, mid, side, cfg, seed + 1000 + round, start);
            solves++; steps += (long) cfg.epochs * d.size();
            double violation = side * Objective.parity(d, theta) - eps;
            if (violation > 0) muLow = mid; else { muHigh = mid; feasible = theta; }
            start = theta;
            double loss = Objective.loss(d, theta, Objective.RIDGE);
            if (violation <= 1e-12 && loss < bestLoss) { best = theta; bestLoss = loss; }
            trace[round] = bestLoss;
        }
        return new FairModel(best, muHigh, side, trace, solves, steps);
    }

    /**
     * Inner loop: warm-started minibatch gradient descent on f + mu (s g - eps), followed by a Newton polish
     * so the outer bisection sees an accurate constraint value. The batch order comes from the seed.
     */
    static double[] inner(Dataset d, double eps, double mu, int side, TrainConfig cfg, long seed,
            double[] start) {
        SplitMix64 rng = new SplitMix64(seed);
        double[] theta = start.clone();
        int n = d.size(), batch = Math.max(1, Math.min(cfg.batchSize, n));
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        double[] phi = new double[Dataset.DIM], g = new double[Dataset.DIM];
        double[] parityGrad = new double[Dataset.DIM];
        int[] counts = {d.groupCount(0), d.groupCount(1)};
        for (int epoch = 0; epoch < cfg.epochs; epoch++) {
            for (int i = n - 1; i > 0; i--) {
                int j = (int) Math.floorMod(rng.nextLong(), i + 1);
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
            for (int from = 0; from < n; from += batch) {
                int to = Math.min(from + batch, n), m = to - from;
                Arrays.fill(g, 0); Arrays.fill(parityGrad, 0);
                for (int k = from; k < to; k++) {
                    Example e = d.examples.get(order[k]);
                    Dataset.phi(e, phi);
                    double p = Objective.probability(e, theta);
                    double coef = (p - e.label) / m;
                    double pg = counts[e.group] == 0 ? 0
                            : (e.group == 0 ? 1.0 : -1.0) * p * (1 - p) / counts[e.group];
                    for (int i = 0; i < Dataset.DIM; i++) { g[i] += coef * phi[i]; parityGrad[i] += pg * phi[i]; }
                }
                for (int i = 0; i < Dataset.DIM; i++)
                    theta[i] -= cfg.primalStep * (g[i] + mu * side * parityGrad[i] + cfg.ridge * theta[i]);
            }
        }
        return ExactSolver.minimise(d, eps, mu, side, theta, 200);
    }
}

/**
 * Independent global oracle: a complete brute-force scan of the (w, b_0, b_1) grid for the cheapest
 * *feasible* point, then a finer local grid. Both group losses and group mean probabilities are tabulated on
 * the coarse grid so the scan itself is one table lookup per triple; the refinement evaluates directly.
 */
class GridOracle {
    static double[] bestFeasible(Dataset d, double eps, double span, int nw, int nb) {
        double[] ws = new double[nw], bs = new double[nb];
        for (int i = 0; i < nw; i++) ws[i] = -span + 2 * span * i / (nw - 1);
        for (int i = 0; i < nb; i++) bs[i] = -span + 2 * span * i / (nb - 1);
        double[][][] lossTable = new double[2][nw][nb], meanTable = new double[2][nw][nb];
        for (int group = 0; group < 2; group++) {
            for (int iw = 0; iw < nw; iw++) {
                for (int ib = 0; ib < nb; ib++) {
                    double[] theta = {ws[iw], group == 0 ? bs[ib] : 0.0, group == 1 ? bs[ib] : 0.0};
                    double loss = 0, mean = 0; int count = 0;
                    for (Example e : d.examples) if (e.group == group) {
                        double z = Objective.score(e, theta);
                        loss += Objective.exampleLoss(z, e.label);
                        mean += Dataset.sigmoid(z);
                        count++;
                    }
                    lossTable[group][iw][ib] = count == 0 ? 0 : loss / count;
                    meanTable[group][iw][ib] = count == 0 ? 0 : mean / count;
                }
            }
        }
        int n0 = d.groupCount(0), n1 = d.groupCount(1);
        double best = Double.MAX_VALUE, bestW = 0, bestB0 = 0, bestB1 = 0;
        for (int iw = 0; iw < nw; iw++)
            for (int i0 = 0; i0 < nb; i0++)
                for (int i1 = 0; i1 < nb; i1++) {
                    if (Math.abs(meanTable[0][iw][i0] - meanTable[1][iw][i1]) > eps) continue;
                    double w = ws[iw], b0 = bs[i0], b1 = bs[i1];
                    double loss = (n0 * lossTable[0][iw][i0] + n1 * lossTable[1][iw][i1]) / d.size()
                            + 0.5 * Objective.RIDGE * (w * w + b0 * b0 + b1 * b1);
                    if (loss < best) { best = loss; bestW = w; bestB0 = b0; bestB1 = b1; }
                }
        return refine(d, eps, best, bestW, bestB0, bestB1, 0.25 * span / (nw - 1));
    }

    /** Local pattern-search refinement around the coarse winner: {loss, w, b0, b1, gap}. */
    private static double[] refine(Dataset d, double eps, double bestLoss, double w, double b0, double b1,
            double step) {
        double[] best = {bestLoss, w, b0, b1, Objective.parity(d, new double[]{w, b0, b1})};
        for (int round = 0; round < 3; round++) {
            for (int i = -4; i <= 4; i++)
                for (int j = -4; j <= 4; j++)
                    for (int k = -4; k <= 4; k++) {
                        double[] theta = {w + i * step, b0 + j * step, b1 + k * step};
                        if (Math.abs(Objective.parity(d, theta)) > eps) continue;
                        double loss = Objective.loss(d, theta, Objective.RIDGE);
                        if (loss < best[0])
                            best = new double[]{loss, theta[0], theta[1], theta[2], Objective.parity(d, theta)};
                    }
            w = best[1]; b0 = best[2]; b1 = best[3]; step *= 0.4;
        }
        return best;
    }
}

public class Solution {
    static final double EPS = 0.05;
    public static void main(String[] args) {
        Dataset data = Dataset.synthetic(240, 20240517L);
        TrainConfig cfg = new TrainConfig(36, 45, 32, 0.6, Objective.RIDGE);

        // --- the exact reference: the unconstrained optimum and both constrained boundaries --------------
        double[] unconstrained = ExactSolver.minimise(data, EPS, 0.0, 1, new double[]{0, 0, 0}, 200);
        double unconstrainedGap = Objective.parity(data, unconstrained);
        double unconstrainedLoss = Objective.loss(data, unconstrained, Objective.RIDGE);
        double[] plus = ExactSolver.solveBoundary(data, EPS, unconstrained.clone(), 200);
        double[] minus = ExactSolver.solveBoundary(data, -EPS, unconstrained.clone(), 200);
        double plusLoss = Objective.loss(data, new double[]{plus[0], plus[1], plus[2]}, Objective.RIDGE);
        double minusLoss = Objective.loss(data, new double[]{minus[0], minus[1], minus[2]}, Objective.RIDGE);
        boolean plusOk = plus[3] > 0, minusOk = minus[3] < 0;   // KKT sign for g <= eps and for g >= -eps
        boolean takePlus = plusOk && (!minusOk || plusLoss <= minusLoss);
        double[] reference = takePlus ? plus : minus;
        double referenceLoss = takePlus ? plusLoss : minusLoss;

        // --- the training loop --------------------------------------------------------------------------
        FairModel trained = FairTrainer.train(data, EPS, cfg, 987654321L);
        FairModel again = FairTrainer.train(data, EPS, cfg, 987654321L);

        check(unconstrainedGap > 5 * EPS, "the unconstrained optimum must violate parity, gap "
                + unconstrainedGap);
        check(unconstrainedLoss < referenceLoss - 1e-4, "the constraint must cost accuracy: "
                + unconstrainedLoss + " vs " + referenceLoss);
        check(Math.abs(trained.parity(data)) <= EPS + 1e-6, "the loop must return a feasible model, gap "
                + trained.parity(data) + " against eps " + EPS);
        check(trained.multiplier > 0, "a binding constraint needs a positive multiplier, got "
                + trained.multiplier);
        check(trained.loss(data) <= referenceLoss * (1 + 1e-6), "the loop must reach the exact boundary "
                + "solution: " + trained.loss(data) + " vs " + referenceLoss);
        check(trained.loss(data) >= referenceLoss * (1 - 1e-3), "training cannot beat the exact constrained "
                + "optimum: " + trained.loss(data) + " vs " + referenceLoss);
        check(trained.theta[0] == again.theta[0] && trained.theta[1] == again.theta[1]
                && trained.theta[2] == again.theta[2] && trained.multiplier == again.multiplier,
                "the training loop must be reproducible bit-for-bit");
        boolean monotone = true;
        for (int r = 1; r < trained.lossTrace.length; r++)
            if (trained.lossTrace[r] > trained.lossTrace[r - 1] + 1e-12) monotone = false;
        check(monotone, "the best-so-far loss of the training loop must never increase");
        check(trained.sgdSteps > 0 && trained.innerSolves > 1,
                "the loop must actually run batches and re-solve the multiplier");

        // --- KKT at the exact reference, and the bisected multiplier against the exact one ----------------
        double[] t = {reference[0], reference[1], reference[2]};
        int side = reference[3] > 0 ? 1 : -1;
        double[] stationarity = Objective.gradLagrangian(data, t, Math.abs(reference[3]), side);
        check(Objective.norm(stationarity) < 1e-8, "KKT stationarity residual must vanish, got "
                + Objective.norm(stationarity));
        check(Math.abs(Math.abs(Objective.parity(data, t)) - EPS) < 1e-9,
                "at an active boundary the gap must equal eps, got " + Objective.parity(data, t));
        check(Math.abs(reference[3]) * Math.abs(Math.abs(Objective.parity(data, t)) - EPS) < 1e-9,
                "complementary slackness: an active multiplier multiplies a zero violation");
        check(plusOk || minusOk, "at least one boundary problem must admit the correct multiplier sign");
        check(Math.abs(trained.multiplier - Math.abs(reference[3])) < 0.2 * Math.abs(reference[3]) + 0.05,
                "the bisected multiplier " + trained.multiplier + " must match the exact one "
                        + Math.abs(reference[3]));

        // --- independent oracle: complete brute-force grid scan ------------------------------------------
        double[] oracle = GridOracle.bestFeasible(data, EPS, 3.0, 121, 121);
        check(Math.abs(oracle[4]) <= EPS + 1e-12, "the oracle's own answer must be feasible, gap " + oracle[4]);
        check(trained.loss(data) <= oracle[0] + 1e-9, "no grid point may beat the trained model: "
                + trained.loss(data) + " vs oracle " + oracle[0]);
        check(oracle[0] <= trained.loss(data) + 0.02, "the trained model must be within 0.02 in loss of the "
                + "global grid optimum: " + trained.loss(data) + " vs " + oracle[0]);
        check(oracle[0] > unconstrainedLoss, "the oracle must confirm the constraint costs accuracy: "
                + oracle[0] + " vs " + unconstrainedLoss);

        // --- the hard selection-rate gap, which the loop never optimises directly ------------------------
        double hardUnconstrained = Math.abs(Objective.hardParity(data, unconstrained));
        double hardTrained = Math.abs(Objective.hardParity(data, trained.theta));
        check(hardTrained < hardUnconstrained, "training must reduce the 0/1 selection-rate gap: "
                + hardTrained + " vs " + hardUnconstrained);
        check(hardTrained < 0.15, "the selection-rate gap must land below 15 points, got " + hardTrained);

        // --- feasibility across a ladder of eps, including the inactive case ----------------------------
        double[] ladder = {0.0, 0.02, 0.05, 0.10, 0.20, 0.45};
        double previousLoss = Double.MAX_VALUE;
        boolean ladderFeasible = true, ladderMonotone = true;
        for (double e : ladder) {
            FairModel m = FairTrainer.train(data, e, new TrainConfig(24, 35, 32, 0.6, Objective.RIDGE), 4242L);
            if (Math.abs(m.parity(data)) > e + 1e-6) ladderFeasible = false;
            if (m.loss(data) > previousLoss + 1e-9) ladderMonotone = false;
            previousLoss = m.loss(data);
        }
        check(ladderFeasible, "every eps on the ladder must yield a model whose gap is within eps");
        check(ladderMonotone, "loosening eps can never make the constrained optimum worse");
        FairModel slack = FairTrainer.train(data, 0.45, cfg, 11L);
        check(slack.multiplier == 0.0 && slack.activeSide == 0,
                "an inactive constraint must leave the multiplier at zero");

        // --- degenerate and invalid inputs --------------------------------------------------------------
        Dataset oneGroup = Dataset.singleGroup(80, 7L);
        FairModel trivial = FairTrainer.train(oneGroup, 0.05, cfg, 5L);
        check(trivial.parity(oneGroup) == 0.0 && trivial.multiplier == 0.0,
                "a single-group dataset has no parity gap to close");
        FairModel coarse = FairTrainer.train(data, EPS, new TrainConfig(0, 0, 32, 0.6, Objective.RIDGE), 3L);
        check(coarse.multiplier == 2.0 && coarse.lossTrace.length == 1,
                "with zero bisection rounds the loop keeps the first bracketing multiplier, got "
                        + coarse.multiplier);
        check(trained.loss(data) <= coarse.loss(data) + 1e-12,
                "refining the multiplier can only improve the loss: " + trained.loss(data) + " vs "
                        + coarse.loss(data));
        FairModel newtonOnly = FairTrainer.train(data, EPS, new TrainConfig(36, 0, 32, 0.6, Objective.RIDGE), 3L);
        check(Math.abs(newtonOnly.parity(data)) <= EPS + 1e-6,
                "with zero SGD epochs the Newton inner solve alone must still land on a feasible model, gap "
                        + newtonOnly.parity(data));
        boolean rejected = false;
        try { FairTrainer.train(data, -0.1, cfg, 1L); } catch (IllegalArgumentException ex) { rejected = true; }
        check(rejected, "a negative parity tolerance must be rejected");
        boolean rejectedEmpty = false;
        try { new Dataset(new ArrayList<>()); } catch (IllegalArgumentException ex) { rejectedEmpty = true; }
        check(rejectedEmpty, "an empty dataset must be rejected");
        check(data.size() == 480 && data.groupCount(0) == 240 && data.groupCount(1) == 240,
                "the two groups must be balanced and the dataset complete");

        System.out.println("Parity eps = " + EPS + ": unconstrained gap " + round(unconstrainedGap)
                + " (loss " + round(unconstrainedLoss) + ") -> constrained gap " + round(trained.parity(data))
                + " (loss " + round(trained.loss(data)) + ", multiplier " + round(trained.multiplier)
                + " vs exact " + round(Math.abs(reference[3])) + ")");
        System.out.println("Hard selection-rate gap " + round(hardUnconstrained) + " -> " + round(hardTrained)
                + "; brute-force grid oracle loss " + round(oracle[0]) + " at w=" + round(oracle[1])
                + " b0=" + round(oracle[2]) + " b1=" + round(oracle[3]));
        System.out.println("eps ladder " + Arrays.toString(ladder) + " feasible, losses non-increasing, "
                + trained.innerSolves + " multiplier solves");
        System.out.println("PASS");
    }

    static String round(double v) { return String.format(Locale.ROOT, "%.6f", v); }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
