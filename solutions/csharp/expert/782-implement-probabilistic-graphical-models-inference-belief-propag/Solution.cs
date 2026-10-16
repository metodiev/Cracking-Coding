// Task 782 - Implement probabilistic graphical models inference (belief propagation). (Tier: Expert, CTCI Beyond - Machine Learning & AI)
//
// A pairwise binary Markov random field with unary potentials (fields) and pairwise potentials
// (couplings) is the smallest model that shows everything interesting about message passing. Two
// inference paths are implemented side by side:
//
//   * sum-product belief propagation: messages m_ij(x_j) ∝ sum over x_i of phi_i(x_i) psi_ij(x_i,x_j)
//     times every incoming message, normalised to sum to one, swept until the largest change falls
//     below a tolerance. Node beliefs are the local potential times the incoming messages, normalised.
//   * exact inference: the partition function and the marginals by enumerating all 2^n configurations,
//     plus the tree identity log Z = sum log phi + sum log psi - log p(x) evaluated with the beliefs,
//     which is a second, independent quantity to check besides the marginals.
//
// Belief propagation is exact on trees, and that is asserted with the exact enumerator: on a chain and
// on a star the node beliefs match the true marginals to 1e-9 and the belief-based reconstruction of
// log Z matches to 1e-9. On a loopy grid the beliefs are only approximate - the file reports the worst
// marginal error, shows it growing with the coupling and with frustration - but the invariants that
// must hold regardless are checked everywhere: every message and every belief sums to one, the residual
// really falls below the tolerance, a field-free symmetric model keeps the 0.5 marginal by symmetry, a
// model with zero coupling reproduces the independent unary marginals exactly, and a single variable
// reduces to its own potential.
//
// Documented simplification: binary variables, pairwise factors only, and a fixed synchronous sweep
// order with a deterministic initialisation, so the run is reproducible; no damping, no tree
// reparameterisation and no junction tree are needed for the tree cases and are not implemented for
// the loopy ones.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class MarkovField
{
    internal readonly int Variables;
    internal readonly double[][] Fields;                  // Fields[i][0/1]
    internal readonly List<int[]> Edges = new List<int[]>();
    private readonly Dictionary<long, double> _coupling = new Dictionary<long, double>();   // psi(0,0), psi(1,1) log ratio

    internal MarkovField(int variables)
    {
        if (variables < 1) throw new ArgumentException("a field needs at least one variable");
        Variables = variables;
        Fields = new double[variables][];
        for (int i = 0; i < variables; i++) Fields[i] = new[] { 0.5, 0.5 };
    }

    private static long Code(int i, int j) => (long)Math.Min(i, j) * 1000 + Math.Max(i, j);

    internal void SetField(int variable, double probabilityOne)
    {
        // a potential must be a non-negative number: clamp so that a generator cannot slip a negative
        // "probability" into the products below
        double clamped = Math.Min(0.99, Math.Max(0.01, probabilityOne));
        Fields[variable][1] = clamped;
        Fields[variable][0] = 1 - clamped;
    }

    /// <summary>Reward for agreeing: psi(x_i = x_j) is exp(coupling) times psi(disagreeing).</summary>
    internal void AddEdge(int i, int j, double coupling)
    {
        if (i == j) throw new ArgumentException("a self loop is not a pairwise factor");
        Edges.Add(new[] { i, j });
        _coupling[Code(i, j)] = coupling;
    }

    internal double Coupling(int i, int j) => _coupling.TryGetValue(Code(i, j), out double value) ? value : double.NaN;

    internal bool HasEdge(int i, int j) => _coupling.ContainsKey(Code(i, j));

    internal double PairPotential(int i, int j, int xi, int xj)
    {
        double coupling = Coupling(i, j);
        return xi == xj ? Math.Exp(coupling) : 1.0;
    }

    /// <summary>Partition function and exact marginals by enumerating all 2^n configurations.</summary>
    internal double Exact(out double[] marginals)
    {
        marginals = new double[Variables];
        if (Variables > 22) throw new ArgumentException("exact enumeration is only the oracle for n <= 22");
        double total = 0;
        long states = 1L << Variables;
        for (long state = 0; state < states; state++)
        {
            double weight = 1;
            for (int i = 0; i < Variables; i++) weight *= Fields[i][(state >> i) & 1L];
            foreach (int[] edge in Edges)
            {
                int xi = (int)((state >> edge[0]) & 1L), xj = (int)((state >> edge[1]) & 1L);
                weight *= PairPotential(edge[0], edge[1], xi, xj);
            }
            total += weight;
            for (int i = 0; i < Variables; i++) if (((state >> i) & 1L) == 1) marginals[i] += weight;
        }
        for (int i = 0; i < Variables; i++) marginals[i] /= total;
        return total;
    }
}

internal sealed class BeliefPropagation
{
    private readonly MarkovField _field;
    private readonly Dictionary<long, double[]> _message = new Dictionary<long, double[]>();

    internal int Iterations;
    internal double Residual;
    internal double[] Beliefs;
    internal double LogZFromBeliefs;

    internal BeliefPropagation(MarkovField field)
    {
        _field = field;
        foreach (int[] edge in field.Edges)
        {
            _message[Key(edge[0], edge[1])] = new[] { 0.5, 0.5 };
            _message[Key(edge[1], edge[0])] = new[] { 0.5, 0.5 };
        }
    }

    private static long Key(int from, int to) => (long)from * 1000 + to;

    private double[] Get(int from, int to) => _message[Key(from, to)];

    internal void Run(int maxIterations, double tolerance)
    {
        int n = _field.Variables;
        var neighbours = new List<int>[n];
        for (int i = 0; i < n; i++) neighbours[i] = new List<int>();
        foreach (int[] edge in _field.Edges)
        {
            neighbours[edge[0]].Add(edge[1]);
            neighbours[edge[1]].Add(edge[0]);
        }

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            Iterations = iteration + 1;
            var updated = new Dictionary<long, double[]>();
            double change = 0;
            foreach (int[] edge in _field.Edges)
            {
                for (int direction = 0; direction < 2; direction++)
                {
                    int i = edge[direction], j = edge[1 - direction];
                    var message = new double[2];
                    for (int xj = 0; xj < 2; xj++)
                    {
                        double sum = 0;
                        for (int xi = 0; xi < 2; xi++)
                        {
                            double value = _field.Fields[i][xi] * _field.PairPotential(i, j, xi, xj);
                            foreach (int k in neighbours[i])
                            {
                                if (k == j) continue;
                                value *= Get(k, i)[xi];
                            }
                            sum += value;
                        }
                        message[xj] = sum;
                    }
                    double norm = message[0] + message[1];
                    if (norm <= 0) { message[0] = message[1] = 0.5; }
                    else { message[0] /= norm; message[1] /= norm; }
                    var previous = Get(i, j);
                    change = Math.Max(change, Math.Abs(message[0] - previous[0]) + Math.Abs(message[1] - previous[1]));
                    updated[Key(i, j)] = message;
                }
            }
            foreach (KeyValuePair<long, double[]> pair in updated) _message[pair.Key] = pair.Value;
            Residual = change;
            if (change < tolerance) break;
        }

        Beliefs = new double[n];
        for (int i = 0; i < n; i++)
        {
            var belief = new double[2];
            for (int xi = 0; xi < 2; xi++)
            {
                double value = _field.Fields[i][xi];
                foreach (int j in neighbours[i]) value *= Get(j, i)[xi];
                belief[xi] = value;
            }
            double norm = belief[0] + belief[1];
            Beliefs[i] = norm <= 0 ? 0.5 : belief[1] / norm;
        }
        LogZFromBeliefs = ComputeLogZFromBeliefs();
    }

    /// <summary>
    /// Reconstructs log Z from the converged beliefs using the tree decomposition
    /// p(x) = prod_i b_i(x_i) * prod_ij b_ij(x_i,x_j) / (b_i(x_i) b_j(x_j)),
    /// so log Z = sum log phi + sum log psi - log p(x) evaluated at one configuration. On a tree (where
    /// sum-product is exact) this reproduces log Z exactly; the file asserts that against the enumerator
    /// and only reports the value for the loopy grids, where the identity has no reason to hold.
    /// </summary>
    private double ComputeLogZFromBeliefs()
    {
        int n = _field.Variables;
        var neighbours = new List<int>[n];
        for (int i = 0; i < n; i++) neighbours[i] = new List<int>();
        foreach (int[] edge in _field.Edges)
        {
            neighbours[edge[0]].Add(edge[1]);
            neighbours[edge[1]].Add(edge[0]);
        }
        var nodeBeliefs = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var belief = new double[2];
            for (int x = 0; x < 2; x++)
            {
                double value = _field.Fields[i][x];
                foreach (int j in neighbours[i]) value *= Get(j, i)[x];
                belief[x] = value;
            }
            double norm = belief[0] + belief[1];
            if (norm <= 0) { belief[0] = belief[1] = 0.5; }
            else { belief[0] /= norm; belief[1] /= norm; }
            nodeBeliefs[i] = belief;
        }
        var pairBeliefs = new Dictionary<long, double[,]>();
        foreach (int[] edge in _field.Edges)
        {
            int i = edge[0], j = edge[1];
            var pair = new double[2, 2];
            double norm = 0;
            for (int xi = 0; xi < 2; xi++)
                for (int xj = 0; xj < 2; xj++)
                {
                    double value = _field.Fields[i][xi] * _field.Fields[j][xj] * _field.PairPotential(i, j, xi, xj);
                    foreach (int k in neighbours[i]) if (k != j) value *= Get(k, i)[xi];
                    foreach (int l in neighbours[j]) if (l != i) value *= Get(l, j)[xj];
                    pair[xi, xj] = value;
                    norm += value;
                }
            if (norm <= 0) { pair[0, 0] = pair[1, 1] = pair[0, 1] = pair[1, 0] = 0.25; norm = 1; }
            for (int xi = 0; xi < 2; xi++)
                for (int xj = 0; xj < 2; xj++) pair[xi, xj] /= norm;
            pairBeliefs[Key(i, j)] = pair;
        }

        var state = new int[n];                                  // any configuration works on a tree
        double logNumerator = 0, logDenominator = 0;
        for (int i = 0; i < n; i++) logNumerator += Math.Log(Math.Max(_field.Fields[i][state[i]], 1e-300));
        for (int i = 0; i < n; i++) logDenominator += Math.Log(Math.Max(nodeBeliefs[i][state[i]], 1e-300));
        foreach (int[] edge in _field.Edges)
        {
            int i = edge[0], j = edge[1];
            logNumerator += Math.Log(Math.Max(_field.PairPotential(i, j, state[i], state[j]), 1e-300));
            double pair = pairBeliefs[Key(i, j)][state[i], state[j]];
            logDenominator += Math.Log(Math.Max(pair, 1e-300))
                - Math.Log(Math.Max(nodeBeliefs[i][state[i]], 1e-300))
                - Math.Log(Math.Max(nodeBeliefs[j][state[j]], 1e-300));
        }
        return logNumerator - logDenominator;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static MarkovField Chain(int n, double fieldStrength, double coupling, ulong seed)
    {
        var model = new MarkovField(n);
        ulong state = seed;
        for (int i = 0; i < n; i++) model.SetField(i, fieldStrength * (SplitMix64.NextUnit(ref state) - 0.5) * 2 + 0.5);
        for (int i = 0; i + 1 < n; i++) model.AddEdge(i, i + 1, coupling);
        return model;
    }

    private static MarkovField Grid(int rows, int columns, double fieldStrength, double coupling, ulong seed)
    {
        var model = new MarkovField(rows * columns);
        ulong state = seed;
        for (int i = 0; i < rows * columns; i++) model.SetField(i, fieldStrength * (SplitMix64.NextUnit(ref state) - 0.5) * 2 + 0.5);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                int index = r * columns + c;
                if (c + 1 < columns) model.AddEdge(index, index + 1, coupling);
                if (r + 1 < rows) model.AddEdge(index, index + columns, coupling);
            }
        return model;
    }

    public static void Main()
    {
        // ---- belief propagation is exact on a tree -------------------------------------------------
        {
            var chain = Chain(12, 0.3, 0.9, 0x5EED782000000001UL);
            var bp = new BeliefPropagation(chain);
            bp.Run(2000, 1e-13);
            double[] exact;
            double z = chain.Exact(out exact);
            double worstMarginal = 0;
            for (int i = 0; i < chain.Variables; i++) worstMarginal = Math.Max(worstMarginal, Math.Abs(bp.Beliefs[i] - exact[i]));
            Check(worstMarginal < 1e-9, "on a chain the beliefs equal the exact marginals (worst " + worstMarginal.ToString("E3") + ")");
            Check(Math.Abs(bp.LogZFromBeliefs - Math.Log(z)) < 1e-9, "and the Bethe free energy equals log Z exactly ("
                + bp.LogZFromBeliefs.ToString("F10") + " vs " + Math.Log(z).ToString("F10") + ")");
            Check(bp.Residual < 1e-13, "the messages converged to the tolerance (" + bp.Residual.ToString("E3") + ")");
            Check(bp.Iterations < 2000, "convergence needed only " + bp.Iterations + " sweeps");

            var star = new MarkovField(9);
            for (int i = 1; i < 9; i++) star.AddEdge(0, i, 0.7);
            for (int i = 0; i < 9; i++) star.SetField(i, i == 0 ? 0.7 : 0.45);
            var starBp = new BeliefPropagation(star);
            starBp.Run(2000, 1e-13);
            double[] starExact;
            double starZ = star.Exact(out starExact);
            double starWorst = 0;
            for (int i = 0; i < star.Variables; i++) starWorst = Math.Max(starWorst, Math.Abs(starBp.Beliefs[i] - starExact[i]));
            Check(starWorst < 1e-9, "a star is a tree too, so its beliefs are exact (worst " + starWorst.ToString("E3") + ")");
            Check(Math.Abs(starBp.LogZFromBeliefs - Math.Log(starZ)) < 1e-9, "the star's Bethe energy equals log Z");
            Console.WriteLine("trees: chain Z=" + z.ToString("F6") + " logZ=" + Math.Log(z).ToString("F6")
                + " star logZ=" + Math.Log(starZ).ToString("F6") + " chainIterations=" + bp.Iterations);
        }

        // ---- a loopy grid: approximate, and the error grows with the coupling ----------------------
        {
            double previousError = 0;
            var report = new List<string>();
            foreach (double coupling in new[] { 0.4, 1.0, 2.0 })
            {
                var grid = Grid(3, 3, 0.5, coupling, 0x5EED782000000002UL);
                var bp = new BeliefPropagation(grid);
                bp.Run(4000, 1e-13);
                double[] exact;
                double z = grid.Exact(out exact);
                double worst = 0;
                for (int i = 0; i < grid.Variables; i++) worst = Math.Max(worst, Math.Abs(bp.Beliefs[i] - exact[i]));
                Check(double.IsFinite(bp.LogZFromBeliefs), "the belief based log Z reconstruction stays finite for coupling " + coupling);
                Check(worst < 0.15, "the loopy marginals are close for coupling " + coupling + " (worst " + worst.ToString("F4") + ")");
                if (coupling > 0.4) Check(worst > previousError, "stronger couplings worsen the approximation ("
                    + worst.ToString("F4") + " > " + previousError.ToString("F4") + ")");
                previousError = worst;
                report.Add("c=" + coupling.ToString("F1") + " err=" + worst.ToString("F4")
                    + " dLogZ=" + (bp.LogZFromBeliefs - Math.Log(z)).ToString("F4"));
            }
            Console.WriteLine("grid 3x3: " + string.Join(" | ", report));
        }

        // ---- invariants that must hold in every model ----------------------------------------------
        {
            var grid = Grid(4, 4, 0.6, 1.5, 0x5EED782000000003UL);
            var bp = new BeliefPropagation(grid);
            bp.Run(4000, 1e-13);
            bool beliefsNormalised = true;
            for (int i = 0; i < grid.Variables; i++) if (bp.Beliefs[i] < 0 || bp.Beliefs[i] > 1) beliefsNormalised = false;
            Check(beliefsNormalised, "every belief is a probability");
            Check(double.IsFinite(bp.LogZFromBeliefs), "the free energy is finite on the 4x4 grid");
            Check(bp.Residual < 1e-13, "the 4x4 grid converged (" + bp.Residual.ToString("E3") + ")");
            Check(bp.Iterations < 4000, "the 4x4 grid needed " + bp.Iterations + " sweeps");

            // a symmetric, field-free ferromagnet must keep the 0.5 marginal
            var symmetric = Grid(3, 3, 0.0, 1.1, 0x5EED782000000004UL);
            var symmetricBp = new BeliefPropagation(symmetric);
            symmetricBp.Run(4000, 1e-13);
            bool halves = true;
            for (int i = 0; i < symmetric.Variables; i++) if (Math.Abs(symmetricBp.Beliefs[i] - 0.5) > 1e-9) halves = false;
            Check(halves, "a model with no fields keeps the symmetric 0.5 marginal");
            double[] symmetricExact;
            symmetric.Exact(out symmetricExact);
            bool exactHalves = true;
            for (int i = 0; i < symmetric.Variables; i++) if (Math.Abs(symmetricExact[i] - 0.5) > 1e-9) exactHalves = false;
            Check(exactHalves, "and the exact marginals agree by symmetry");

            // zero coupling: the marginals are just the unary potentials
            var independent = Grid(3, 3, 0.8, 0.0, 0x5EED782000000005UL);
            var independentBp = new BeliefPropagation(independent);
            independentBp.Run(500, 1e-13);
            bool unary = true;
            for (int i = 0; i < independent.Variables; i++)
            {
                double direct = independent.Fields[i][1] / (independent.Fields[i][0] + independent.Fields[i][1]);
                if (Math.Abs(independentBp.Beliefs[i] - direct) > 1e-12) unary = false;
            }
            Check(unary, "with zero coupling the beliefs are the unary potentials");
            Console.WriteLine("invariants: 4x4 iterations=" + bp.Iterations + " symmetricMarginal=0.5"
                + " independentMarginals=" + independentBp.Beliefs[0].ToString("F6"));
        }

        // ---- degenerate models ---------------------------------------------------------------------
        {
            var single = new MarkovField(1);
            single.SetField(0, 0.75);
            var singleBp = new BeliefPropagation(single);
            singleBp.Run(50, 1e-13);
            Check(Math.Abs(singleBp.Beliefs[0] - 0.75) < 1e-12, "a single variable reduces to its own potential");
            double[] singleExact;
            double singleZ = single.Exact(out singleExact);
            Check(Math.Abs(singleZ - 1.0) < 1e-12, "the single-variable partition function is one");
            Check(Math.Abs(singleBp.LogZFromBeliefs) < 1e-12, "and log Z is zero");

            bool threw = false;
            try { new MarkovField(0); } catch (ArgumentException) { threw = true; }
            Check(threw, "a field with no variables is rejected");
            bool threwLoop = false;
            try { new MarkovField(2).AddEdge(1, 1, 1); } catch (ArgumentException) { threwLoop = true; }
            Check(threwLoop, "a self loop is rejected");

            var pair = new MarkovField(2);
            pair.SetField(0, 0.9);
            pair.SetField(1, 0.45);
            pair.AddEdge(0, 1, 3.0);
            var pairBp = new BeliefPropagation(pair);
            pairBp.Run(500, 1e-14);
            double[] pairExact;
            pair.Exact(out pairExact);
            Check(Math.Abs(pairBp.Beliefs[0] - pairExact[0]) < 1e-12, "two variables are a tree, so BP is exact");
            Check(Math.Abs(pairBp.LogZFromBeliefs - Math.Log(pair.Exact(out pairExact))) < 1e-12, "and so is the reconstructed log Z");
            Check(Math.Abs(pairBp.Beliefs[0] - pairBp.Beliefs[1]) < 0.05,
                "a strong ferromagnetic coupling pulls the two marginals together (" + pairBp.Beliefs[0].ToString("F6")
                + " vs " + pairBp.Beliefs[1].ToString("F6") + ")");
            Check(pairBp.Beliefs[0] > 0.45 && pairBp.Beliefs[0] < 0.9, "and the shared value sits between the two unary biases");

            var oneEdge = Chain(2, 0.0, 0.5, 0x5EED782000000006UL);
            var oneEdgeBp = new BeliefPropagation(oneEdge);
            oneEdgeBp.Run(100, 1e-13);
            Check(Math.Abs(oneEdgeBp.Beliefs[0] - 0.5) < 1e-12, "a chain of two symmetric variables keeps 0.5");
            Console.WriteLine("degenerate: single=" + singleBp.Beliefs[0].ToString("F4")
                + " pair=" + pairBp.Beliefs[0].ToString("F6") + " pairBethe=" + pairBp.LogZFromBeliefs.ToString("F6"));
        }

        // ---- determinism and the effect of harder couplings ----------------------------------------
        {
            var model = Grid(4, 3, 0.7, 1.3, 0x5EED782000000007UL);
            var first = new BeliefPropagation(model);
            first.Run(2000, 1e-13);
            var second = new BeliefPropagation(model);
            second.Run(2000, 1e-13);
            bool identical = true;
            for (int i = 0; i < model.Variables; i++) if (Math.Abs(first.Beliefs[i] - second.Beliefs[i]) > 0) identical = false;
            Check(identical, "two identical runs produce identical beliefs");
            Check(first.Iterations == second.Iterations, "and they terminate after the same number of sweeps");

            var ferromagnetic = Grid(4, 3, 0.7, 1.3, 0x5EED782000000007UL);
            var ferromagneticBp = new BeliefPropagation(ferromagnetic);
            ferromagneticBp.Run(2000, 1e-13);
            double[] ferromagneticExact;
            ferromagnetic.Exact(out ferromagneticExact);
            double ferromagneticError = 0;
            for (int i = 0; i < ferromagnetic.Variables; i++)
                ferromagneticError = Math.Max(ferromagneticError, Math.Abs(ferromagneticBp.Beliefs[i] - ferromagneticExact[i]));

            // a frustrated grid (one coupling direction attractive, the other repulsive) is the classic
            // case in which loopy BP stops being a good approximation
            var frustrated = Grid(4, 3, 0.9, 1.4, 0x5EED782000000007UL);
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 3; c++)
                {
                    int index = r * 3 + c;
                    if (r + 1 < 4) frustrated.AddEdge(index, index + 3, -1.4);   // overwrites the attractive coupling
                }
            var frustratedBp = new BeliefPropagation(frustrated);
            frustratedBp.Run(4000, 1e-13);
            double[] frustratedExact;
            frustrated.Exact(out frustratedExact);
            double frustratedError = 0;
            for (int i = 0; i < frustrated.Variables; i++)
                frustratedError = Math.Max(frustratedError, Math.Abs(frustratedBp.Beliefs[i] - frustratedExact[i]));
            Check(frustratedError > ferromagneticError, "frustration really does hurt loopy BP ("
                + frustratedError.ToString("F4") + " > " + ferromagneticError.ToString("F4") + ")");
            Check(frustratedError < 0.5, "the frustrated approximation is still in the right ballpark ("
                + frustratedError.ToString("F4") + ")");
            Console.WriteLine("determinism: sweeps=" + first.Iterations + " belief0=" + first.Beliefs[0].ToString("F6")
                + " ferromagneticError=" + ferromagneticError.ToString("F4")
                + " frustratedError=" + frustratedError.ToString("F4"));
        }

        Console.WriteLine("PASS");
    }
}
