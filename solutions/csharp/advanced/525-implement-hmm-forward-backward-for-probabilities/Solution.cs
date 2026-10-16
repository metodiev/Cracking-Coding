// Task 525 - Implement HMM forward-backward for probabilities. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Forward-backward for a discrete HMM (start distribution, row-stochastic transition and emission
// matrices). Four computations are implemented and cross-checked:
//   * alpha_t(j) = P(o_1..o_t, q_t = j) and beta_t(j) = P(o_{t+1}..o_T | q_t = j), both in the linear
//     domain and (for the likelihood) again in log space, plus a scaled variant that divides every
//     alpha row by its sum c_t and folds the same factors into beta, so that
//     log P(O) = sum_t log c_t with no underflow on long sequences;
//   * gamma_t(j) = alpha_t(j) beta_t(j) / P(O), the posterior state distribution;
//   * xi_t(i, j) = alpha_t(i) a_ij b_j(o_{t+1}) beta_{t+1}(j) / P(O), the posterior of a transition
//     at time t;
//   * one Baum-Welch (EM) iteration: pi'_i = gamma_0(i), a'_ij = sum_t xi_t(i,j) / sum_t gamma_t(i)
//     and b'_i(k) proportional to sum over the times with o_t = k of gamma_t(i).
//
// Documented checks. (1) The classical weather HMM on Walk, Shop, Clean has the hand-checked
// likelihood 0.033612 (the sum of the eight path probabilities, enumerated independently) and the
// last-row posterior gamma_2 = (0.02904/0.033612, 0.004572/0.033612); forward, log-space forward and
// scaled forward agree to 1e-12 relative. (2) Identity sum_j alpha_t(j) beta_t(j) = P(O) holds at
// every t and in both domains (1e-12 relative). (3) gamma_t(j) equals the brute-force ratio of the
// joint probability of the paths through state j at t to the total joint probability (1e-10), sums to
// 1 at every t, and lies in [0, 1]; xi_t equals its explicit formula, sums to 1 over (i, j) for every
// t, and the scaled gamma reproduces the unscaled one. (4) A single observation gives the closed-form
// posterior start_i*emission_i(o) / sum_k start_k*emission_k(o). (5) Baum-Welch: three iterations on
// three LCG models are non-decreasing in log-likelihood (each step at least the previous minus 1e-12)
// and strictly increase overall, the re-estimated rows stay stochastic to 1e-12, and the number of
// parameters stays fixed. (6) Errors: an empty observation, an unknown symbol and a non-stochastic
// model each throw ArgumentException.
using System;

internal sealed class Hmm
{
    internal readonly double[] Start;
    internal readonly double[][] Trans;
    internal readonly double[][] Emit;

    internal Hmm(double[] start, double[][] trans, double[][] emit)
    {
        if (trans.Length != start.Length || emit.Length != start.Length)
        {
            throw new ArgumentException("start, transition and emission sizes disagree");
        }
        Start = (double[])start.Clone();
        Trans = Clone(trans, "transition");
        Emit = Clone(emit, "emission");
        double startSum = 0.0;
        foreach (double value in Start)
        {
            if (value < 0.0) throw new ArgumentException("negative start probability");
            startSum += value;
        }
        if (Math.Abs(startSum - 1.0) > 1e-9) throw new ArgumentException("the start distribution must sum to 1");
    }

    private static double[][] Clone(double[][] rows, string role)
    {
        var copy = new double[rows.Length][];
        for (int i = 0; i < rows.Length; i++)
        {
            copy[i] = (double[])rows[i].Clone();
            double sum = 0.0;
            foreach (double value in copy[i])
            {
                if (value < 0.0) throw new ArgumentException("negative " + role + " probability");
                sum += value;
            }
            if (Math.Abs(sum - 1.0) > 1e-9)
            {
                throw new ArgumentException("every " + role + " row must sum to 1");
            }
        }
        return copy;
    }

    internal int States => Start.Length;
    internal int Symbols => Emit[0].Length;

    internal static Hmm Weather()
    {
        return new Hmm(
                new[] { 0.6, 0.4 },
                new[] { new[] { 0.7, 0.3 }, new[] { 0.4, 0.6 } },
                new[] { new[] { 0.1, 0.4, 0.5 }, new[] { 0.6, 0.3, 0.1 } });
    }

    internal static Hmm Lcg(int states, int symbols, int seed)
    {
        var start = new double[states];
        var trans = new double[states][];
        var emit = new double[states][];
        double startSum = 0.0;
        for (int i = 0; i < states; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            start[i] = 1.0 + seed % 100;
            startSum += start[i];
        }
        for (int i = 0; i < states; i++) start[i] /= startSum;
        for (int i = 0; i < states; i++)
        {
            trans[i] = new double[states];
            emit[i] = new double[symbols];
            double rowSum = 0.0;
            for (int j = 0; j < states; j++)
            {
                seed = (seed * 75 + 74) % 65537;
                trans[i][j] = 1.0 + seed % 100;
                rowSum += trans[i][j];
            }
            for (int j = 0; j < states; j++) trans[i][j] /= rowSum;
            rowSum = 0.0;
            for (int k = 0; k < symbols; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                emit[i][k] = 1.0 + seed % 100;
                rowSum += emit[i][k];
            }
            for (int k = 0; k < symbols; k++) emit[i][k] /= rowSum;
        }
        return new Hmm(start, trans, emit);
    }
}

internal sealed class ScaledResult
{
    internal double[][] Alpha = Array.Empty<double[]>();
    internal double[][] Beta = Array.Empty<double[]>();
    internal double[] Scale = Array.Empty<double>();
    internal double LogLikelihood;
    internal double Likelihood;
}

internal sealed class BwResult
{
    internal Hmm Model = Hmm.Weather();
    internal double LogLikelihood;
}

internal static class ForwardBackward
{
    internal static void Validate(Hmm hmm, int[] observation)
    {
        if (observation.Length == 0) throw new ArgumentException("empty observation");
        foreach (int symbol in observation)
        {
            if (symbol < 0 || symbol >= hmm.Symbols) throw new ArgumentException("unknown symbol " + symbol);
        }
    }

    internal static double[][] Forward(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var alpha = new double[steps][];
        alpha[0] = new double[states];
        for (int k = 0; k < states; k++) alpha[0][k] = hmm.Start[k] * hmm.Emit[k][observation[0]];
        for (int t = 1; t < steps; t++)
        {
            alpha[t] = new double[states];
            for (int k = 0; k < states; k++)
            {
                double sum = 0.0;
                for (int j = 0; j < states; j++) sum += alpha[t - 1][j] * hmm.Trans[j][k];
                alpha[t][k] = sum * hmm.Emit[k][observation[t]];
            }
        }
        return alpha;
    }

    internal static double[][] Backward(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var beta = new double[steps][];
        beta[steps - 1] = new double[states];
        for (int k = 0; k < states; k++) beta[steps - 1][k] = 1.0;
        for (int t = steps - 2; t >= 0; t--)
        {
            beta[t] = new double[states];
            for (int j = 0; j < states; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < states; k++)
                {
                    sum += hmm.Trans[j][k] * hmm.Emit[k][observation[t + 1]] * beta[t + 1][k];
                }
                beta[t][j] = sum;
            }
        }
        return beta;
    }

    internal static double Likelihood(double[][] alpha)
    {
        double total = 0.0;
        foreach (double value in alpha[alpha.Length - 1]) total += value;
        return total;
    }

    /** Log-space forward pass; LogLikelihood = log sum_j alpha_T(j). */
    internal static double LogLikelihood(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var alpha = new double[states];
        for (int k = 0; k < states; k++)
        {
            alpha[k] = Log(hmm.Start[k]) + Log(hmm.Emit[k][observation[0]]);
        }
        for (int t = 1; t < steps; t++)
        {
            var next = new double[states];
            for (int k = 0; k < states; k++)
            {
                double sum = double.NegativeInfinity;
                for (int j = 0; j < states; j++)
                {
                    sum = LogSumExp(sum, alpha[j] + Log(hmm.Trans[j][k]));
                }
                next[k] = sum + Log(hmm.Emit[k][observation[t]]);
            }
            alpha = next;
        }
        double total = double.NegativeInfinity;
        foreach (double value in alpha) total = LogSumExp(total, value);
        return total;
    }

    /** Scaled forward-backward: alpha rows are normalised, the factors are folded into beta.
     *  With alphaHat_t = alpha_t / (c_0..c_t), betaHat_t = beta_t / (c_{t+1}..c_{T-1}) and the
     *  row sums c_t = sum_j alpha_t(j) the product alphaHat_t * betaHat_t is exactly gamma_t and
     *  log P(O) = sum_t log c_t. */
    internal static ScaledResult ScaledForwardBackward(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var result = new ScaledResult();
        result.Alpha = new double[steps][];
        result.Scale = new double[steps];
        var alpha = new double[states];
        for (int k = 0; k < states; k++) alpha[k] = hmm.Start[k] * hmm.Emit[k][observation[0]];
        double scale = 0.0;
        foreach (double value in alpha) scale += value;
        result.Scale[0] = scale;
        for (int k = 0; k < states; k++) alpha[k] /= scale;
        result.Alpha[0] = (double[])alpha.Clone();
        for (int t = 1; t < steps; t++)
        {
            var next = new double[states];
            double sum = 0.0;
            for (int k = 0; k < states; k++)
            {
                double value = 0.0;
                for (int j = 0; j < states; j++) value += alpha[j] * hmm.Trans[j][k];
                next[k] = value * hmm.Emit[k][observation[t]];
                sum += next[k];
            }
            result.Scale[t] = sum;
            for (int k = 0; k < states; k++) next[k] /= sum;
            result.Alpha[t] = next;
            alpha = next;
        }
        result.Beta = new double[steps][];
        result.Beta[steps - 1] = new double[states];
        for (int k = 0; k < states; k++) result.Beta[steps - 1][k] = 1.0;
        for (int t = steps - 2; t >= 0; t--)
        {
            result.Beta[t] = new double[states];
            for (int j = 0; j < states; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < states; k++)
                {
                    sum += hmm.Trans[j][k] * hmm.Emit[k][observation[t + 1]] * result.Beta[t + 1][k];
                }
                result.Beta[t][j] = sum / result.Scale[t + 1];
            }
        }
        double logLikelihood = 0.0;
        foreach (double factor in result.Scale) logLikelihood += Math.Log(factor);
        result.LogLikelihood = logLikelihood;
        result.Likelihood = Math.Exp(logLikelihood);
        return result;
    }

    internal static double[][] Gamma(double[][] alpha, double[][] beta, double likelihood)
    {
        int steps = alpha.Length;
        int states = alpha[0].Length;
        var gamma = new double[steps][];
        for (int t = 0; t < steps; t++)
        {
            gamma[t] = new double[states];
            for (int j = 0; j < states; j++)
            {
                gamma[t][j] = alpha[t][j] * beta[t][j] / likelihood;
            }
        }
        return gamma;
    }

    internal static double[][][] Xi(Hmm hmm, int[] observation, double[][] alpha, double[][] beta,
            double likelihood)
    {
        int steps = observation.Length;
        int states = hmm.States;
        var xi = new double[steps - 1][][];
        for (int t = 0; t < steps - 1; t++)
        {
            xi[t] = new double[states][];
            for (int i = 0; i < states; i++)
            {
                xi[t][i] = new double[states];
                for (int j = 0; j < states; j++)
                {
                    xi[t][i][j] = alpha[t][i] * hmm.Trans[i][j] * hmm.Emit[j][observation[t + 1]]
                            * beta[t + 1][j] / likelihood;
                }
            }
        }
        return xi;
    }

    /** One Baum-Welch re-estimation step; the returned model is a fresh instance. */
    internal static double[] BaumWelch(Hmm hmm, int[] observation, out Hmm updated)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        int symbols = hmm.Symbols;
        ScaledResult scaled = ScaledForwardBackward(hmm, observation);
        var gamma = new double[steps][];
        for (int t = 0; t < steps; t++)
        {
            gamma[t] = new double[states];
            for (int j = 0; j < states; j++)
            {
                gamma[t][j] = scaled.Alpha[t][j] * scaled.Beta[t][j];
            }
        }
        var xi = new double[steps - 1][][];
        for (int t = 0; t < steps - 1; t++)
        {
            xi[t] = new double[states][];
            for (int i = 0; i < states; i++)
            {
                xi[t][i] = new double[states];
                for (int j = 0; j < states; j++)
                {
                    xi[t][i][j] = scaled.Alpha[t][i] * hmm.Trans[i][j]
                            * hmm.Emit[j][observation[t + 1]] * scaled.Beta[t + 1][j];
                }
            }
        }
        var start = new double[states];
        var trans = new double[states][];
        var emit = new double[states][];
        for (int i = 0; i < states; i++)
        {
            start[i] = gamma[0][i];
            trans[i] = new double[states];
            emit[i] = new double[symbols];
            double denominator = 0.0;
            for (int t = 0; t < steps - 1; t++) denominator += gamma[t][i];
            for (int j = 0; j < states; j++)
            {
                double sum = 0.0;
                for (int t = 0; t < steps - 1; t++) sum += xi[t][i][j];
                trans[i][j] = denominator > 0.0 ? sum / denominator : 0.0;
            }
            double stateTotal = 0.0;
            for (int t = 0; t < steps; t++) stateTotal += gamma[t][i];
            for (int k = 0; k < symbols; k++)
            {
                double sum = 0.0;
                for (int t = 0; t < steps; t++)
                {
                    if (observation[t] == k) sum += gamma[t][i];
                }
                emit[i][k] = stateTotal > 0.0 ? sum / stateTotal : 0.0;
            }
        }
        Normalise(start);
        for (int i = 0; i < states; i++)
        {
            Normalise(trans[i]);
            Normalise(emit[i]);
        }
        updated = new Hmm(start, trans, emit);
        return gamma[0];
    }

    private static void Normalise(double[] row)
    {
        double sum = 0.0;
        foreach (double value in row) sum += value;
        if (sum <= 0.0)
        {
            for (int i = 0; i < row.Length; i++) row[i] = 1.0 / row.Length;
            return;
        }
        for (int i = 0; i < row.Length; i++) row[i] /= sum;
    }

    /** Sum of the joint probabilities of all s^T state paths (the brute-force likelihood). */
    internal static double BruteForceLikelihood(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var path = new int[steps];
        double total = 0.0;
        while (true)
        {
            total += Joint(hmm, path, observation);
            int position = steps - 1;
            while (position >= 0)
            {
                path[position]++;
                if (path[position] < states) break;
                path[position] = 0;
                position--;
            }
            if (position < 0) break;
        }
        return total;
    }

    /** Brute-force posterior over the state at time t. */
    internal static double[] BruteForcePosterior(Hmm hmm, int[] observation, int time)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var marginal = new double[states];
        var path = new int[steps];
        double total = 0.0;
        while (true)
        {
            double probability = Joint(hmm, path, observation);
            total += probability;
            marginal[path[time]] += probability;
            int position = steps - 1;
            while (position >= 0)
            {
                path[position]++;
                if (path[position] < states) break;
                path[position] = 0;
                position--;
            }
            if (position < 0) break;
        }
        for (int j = 0; j < states; j++) marginal[j] /= total;
        return marginal;
    }

    internal static double Joint(Hmm hmm, int[] path, int[] observation)
    {
        double probability = hmm.Start[path[0]] * hmm.Emit[path[0]][observation[0]];
        for (int t = 1; t < path.Length; t++)
        {
            probability *= hmm.Trans[path[t - 1]][path[t]] * hmm.Emit[path[t]][observation[t]];
        }
        return probability;
    }

    internal static double Log(double value)
    {
        return value > 0.0 ? Math.Log(value) : double.NegativeInfinity;
    }

    internal static double LogSumExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        double high = Math.Max(a, b);
        return high + Math.Log(Math.Exp(a - high) + Math.Exp(b - high));
    }
}

public static class Solution
{
    public static void Main()
    {
        Hmm weather = Hmm.Weather();
        int[] observation = { 0, 1, 2 };
        double[][] alpha = ForwardBackward.Forward(weather, observation);
        double[][] beta = ForwardBackward.Backward(weather, observation);
        double likelihood = ForwardBackward.Likelihood(alpha);
        CheckClose(likelihood, 0.033612, 1e-12, "hand-checked weather likelihood");
        CheckClose(ForwardBackward.BruteForceLikelihood(weather, observation), 0.033612, 1e-12,
                "the likelihood is the sum over the eight path probabilities");
        CheckClose(Math.Exp(ForwardBackward.LogLikelihood(weather, observation)), 0.033612, 1e-12,
                "log-space likelihood");
        ScaledResult scaled = ForwardBackward.ScaledForwardBackward(weather, observation);
        CheckClose(scaled.Likelihood, 0.033612, 1e-12, "scaled likelihood");

        double[][] gamma = ForwardBackward.Gamma(alpha, beta, likelihood);
        CheckClose(gamma[0][0] * likelihood, alpha[0][0] * beta[0][0], 1e-15, "gamma_0(0) definition");
        CheckClose(gamma[2][0], 0.02904 / 0.033612, 1e-12, "hand-checked gamma_2(Rainy)");
        CheckClose(gamma[2][1], 0.004572 / 0.033612, 1e-12, "hand-checked gamma_2(Sunny)");
        for (int t = 0; t < observation.Length; t++)
        {
            double sum = 0.0;
            for (int j = 0; j < weather.States; j++) sum += gamma[t][j];
            CheckClose(sum, 1.0, 1e-12, "gamma sums to 1 at t = " + t);
            double identity = 0.0;
            for (int j = 0; j < weather.States; j++) identity += alpha[t][j] * beta[t][j];
            CheckClose(identity, likelihood, 1e-12, "sum_j alpha_t beta_t = P(O) at t = " + t);
            double[] brutePosterior = ForwardBackward.BruteForcePosterior(weather, observation, t);
            for (int j = 0; j < weather.States; j++)
            {
                CheckClose(gamma[t][j], brutePosterior[j], 1e-10,
                        "gamma_ " + t + "(" + j + ") equals the brute-force posterior");
                Check(Math.Abs(scaled.Alpha[t][j] * scaled.Beta[t][j] - gamma[t][j]) <= 1e-12,
                        "the scaled auxiliary equals gamma at t = " + t + ", state " + j);
            }
        }

        double[][][] xi = ForwardBackward.Xi(weather, observation, alpha, beta, likelihood);
        for (int t = 0; t < xi.Length; t++)
        {
            double sum = 0.0;
            for (int i = 0; i < weather.States; i++)
            {
                for (int j = 0; j < weather.States; j++)
                {
                    sum += xi[t][i][j];
                    double expected = alpha[t][i] * weather.Trans[i][j] * weather.Emit[j][observation[t + 1]]
                            * beta[t + 1][j] / likelihood;
                    CheckClose(xi[t][i][j], expected, 1e-12, "xi formula at t = " + t);
                }
            }
            CheckClose(sum, 1.0, 1e-12, "xi sums to 1 at t = " + t);
        }

        var single = new[] { 2 };
        double[][] singleAlpha = ForwardBackward.Forward(weather, single);
        double[] singleGamma = ForwardBackward.Gamma(singleAlpha, ForwardBackward.Backward(weather, single),
                ForwardBackward.Likelihood(singleAlpha))[0];
        double singleTotal = 0.0;
        for (int j = 0; j < weather.States; j++) singleTotal += weather.Start[j] * weather.Emit[j][2];
        for (int j = 0; j < weather.States; j++)
        {
            CheckClose(singleGamma[j], weather.Start[j] * weather.Emit[j][2] / singleTotal, 1e-12,
                    "single observation closed-form posterior at state " + j);
        }
        CheckClose(ForwardBackward.BruteForceLikelihood(weather, single), singleTotal, 1e-12,
                "single observation brute-force likelihood");

        int[] seeds = { 11, 22, 33 };
        double[] expectedGrowth = { 0.0, 0.0, 0.0 };
        for (int s = 0; s < seeds.Length; s++)
        {
            Hmm model = Hmm.Lcg(3, 5, seeds[s]);
            var sequence = new int[8];
            int seed = seeds[s] + 1;
            for (int t = 0; t < sequence.Length; t++)
            {
                seed = (seed * 75 + 74) % 65537;
                sequence[t] = seed % 5;
            }
            double previous = ForwardBackward.LogLikelihood(model, sequence);
            CheckClose(Math.Exp(previous), ForwardBackward.Likelihood(ForwardBackward.Forward(model, sequence)),
                    1e-9, "seed " + seeds[s] + ": log and linear likelihood");
            CheckClose(ForwardBackward.ScaledForwardBackward(model, sequence).Likelihood, Math.Exp(previous),
                    1e-9, "seed " + seeds[s] + ": scaled likelihood");
            CheckClose(ForwardBackward.BruteForceLikelihood(model, sequence), Math.Exp(previous), 1e-9,
                    "seed " + seeds[s] + ": brute-force likelihood");
            Hmm current = model;
            for (int iteration = 0; iteration < 3; iteration++)
            {
                ForwardBackward.BaumWelch(current, sequence, out Hmm updated);
                double next = ForwardBackward.LogLikelihood(updated, sequence);
                Check(next >= previous - 1e-12,
                        "seed " + seeds[s] + ": the log-likelihood never decreases (iteration " + iteration + ")");
                CheckStochastic(updated, "seed " + seeds[s] + " iteration " + iteration);
                previous = next;
                current = updated;
            }
            expectedGrowth[s] = previous - ForwardBackward.LogLikelihood(model, sequence);
            Check(expectedGrowth[s] > 1e-6, "seed " + seeds[s] + ": the likelihood improves overall");

            double[][] localAlpha = ForwardBackward.Forward(model, sequence);
            double[][] localBeta = ForwardBackward.Backward(model, sequence);
            double localLikelihood = ForwardBackward.Likelihood(localAlpha);
            for (int t = 0; t < sequence.Length; t++)
            {
                double identity = 0.0;
                for (int j = 0; j < model.States; j++) identity += localAlpha[t][j] * localBeta[t][j];
                CheckClose(identity, localLikelihood, 1e-12,
                        "seed " + seeds[s] + ": alpha beta identity at t = " + t);
                double[] posterior = ForwardBackward.Gamma(localAlpha, localBeta, localLikelihood)[t];
                double[] brute = ForwardBackward.BruteForcePosterior(model, sequence, t);
                for (int j = 0; j < model.States; j++)
                {
                    CheckClose(posterior[j], brute[j], 1e-10,
                            "seed " + seeds[s] + ": posterior at t = " + t + ", state " + j);
                }
            }
        }

        CheckThrows(() => ForwardBackward.Forward(weather, new int[0]), "an empty observation throws");
        CheckThrows(() => ForwardBackward.Forward(weather, new[] { 7 }), "an unknown symbol throws");
        CheckThrows(() => ForwardBackward.LogLikelihood(weather, new[] { 0, 9 }), "an unknown symbol throws");
        CheckThrows(() => new Hmm(new[] { 0.4, 0.4 }, new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } },
                new[] { new[] { 1.0 }, new[] { 1.0 } }), "a non-stochastic start throws");
        CheckThrows(() => new Hmm(new[] { 0.5, 0.5 }, new[] { new[] { 0.5, 0.5 }, new[] { 0.4, 0.4 } },
                new[] { new[] { 1.0 }, new[] { 1.0 } }), "a non-stochastic transition row throws");
        CheckThrows(() => new Hmm(new[] { 0.5, 0.5 }, new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } },
                new[] { new[] { 1.5, -0.5 }, new[] { 1.0, 0.0 } }), "a negative emission throws");

        Console.WriteLine("PASS");
    }

    private static void CheckStochastic(Hmm model, string label)
    {
        double startSum = 0.0;
        foreach (double value in model.Start) startSum += value;
        CheckClose(startSum, 1.0, 1e-12, label + ": re-estimated start row");
        foreach (double[] row in model.Trans) CheckRow(row, label + ": re-estimated transition row");
        foreach (double[] row in model.Emit) CheckRow(row, label + ": re-estimated emission row");
    }

    private static void CheckRow(double[] row, string label)
    {
        double sum = 0.0;
        foreach (double value in row)
        {
            Check(value >= 0.0, label + ": non-negative entry");
            sum += value;
        }
        CheckClose(sum, 1.0, 1e-12, label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        if (!(Math.Abs(actual - expected) <= tolerance * Math.Max(1.0, Math.Abs(expected))))
        {
            throw new Exception("FAILED: " + label + " (" + actual + " != " + expected + ")");
        }
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
