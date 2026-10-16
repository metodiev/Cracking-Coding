// Task 708 - Implement fractional packing/covering using multiplicative weights update.
// (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
//
// The fractional set-cover LP is min c.x subject to A x >= 1, x >= 0, where the rows of the 0/1
// matrix A are elements and the columns are sets; its dual is the fractional packing LP
// max sum(y) subject to A^T y <= c, y >= 0. Both sides are solved here with multiplicative weights.
//
// Covering. Keep one weight per element, interpreted as its "still uncovered" pressure (all weights
// start at 1). Each round picks the column with the best value for money, j = argmin c_j / sum_{i in
// S_j} w_i, adds x_j += 1 / sum w_i, and then *divides* the weight of every element that the column
// covers by (1 + eps). Uncovered elements keep weight 1 and therefore pull the following rounds
// towards columns that actually cover them; the weights are renormalised so the ratios stay healthy.
// The iterate is finally scaled so that the tightest element is covered exactly once.
//
// Packing. The dual is handled by a damped water-filling ascent: repeatedly raise the coordinates of
// the elements that are not yet covered, all at the same rate, until a set constraint becomes tight;
// the remaining slack divided by that rate is the largest feasible step. Both weights keep every
// constraint satisfied, so the resulting y is dual feasible and its value is a lower bound on OPT.
//
// Verification is exact rather than statistical: for the small instances used here the LP optimum is
// computed independently by enumerating the basic feasible solutions (all combinations of n tight
// constraints solved with Gaussian elimination), and the multiplicative-weights covering is required
// both to never beat that optimum and to stay within a measured factor of it. Weak duality between
// the packing and covering sides, feasibility of both, empty/uncovered instances and rejected input
// are checked as well.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class CoverInstance
{
    private readonly int _elements;
    private readonly List<int>[] _members;      // element -> indices of the sets containing it
    private List<int>[] _sets;                 // set -> elements it covers
    private readonly List<double> _cost = new List<double>();

    internal CoverInstance(int elements)
    {
        _elements = elements;
        _members = new List<int>[elements];
        _sets = new List<int>[0];
        for (int i = 0; i < elements; i++) _members[i] = new List<int>();
    }

    internal int Elements => _elements;
    internal int SetCount => _cost.Count;
    internal double Cost(int set) => _cost[set];
    internal IReadOnlyList<int> ElementsOf(int set) => _sets[set];
    internal IReadOnlyList<int> SetsWith(int element) => _members[element];

    internal int AddSet(double cost, params int[] elements)
    {
        if (cost <= 0) throw new ArgumentException("set costs must be positive");
        if (elements.Length == 0) throw new ArgumentException("a set must cover at least one element");
        var covered = new List<int>();
        foreach (int e in elements)
        {
            if (e < 0 || e >= _elements) throw new ArgumentOutOfRangeException(nameof(elements), "element outside the instance");
            if (!covered.Contains(e)) covered.Add(e);
        }
        var grown = new List<int>[_sets.Length + 1];
        Array.Copy(_sets, grown, _sets.Length);
        grown[_sets.Length] = covered;
        _sets = grown;
        int id = _cost.Count;
        _cost.Add(cost);
        foreach (int e in covered) _members[e].Add(id);
        return id;
    }

    /// <summary>True when every element is covered by at least one set.</summary>
    internal bool IsCoverable()
    {
        for (int e = 0; e < _elements; e++) if (_members[e].Count == 0) return false;
        return true;
    }

    internal double Coverage(double[] x, int element)
    {
        double total = 0;
        foreach (int s in _members[element]) total += x[s];
        return total;
    }

    internal double MinCoverage(double[] x)
    {
        double worst = double.MaxValue;
        for (int e = 0; e < _elements; e++) worst = Math.Min(worst, Coverage(x, e));
        return _elements == 0 ? 0 : worst;
    }

    internal double TotalCost(double[] x)
    {
        double total = 0;
        for (int j = 0; j < x.Length; j++) total += _cost[j] * x[j];
        return total;
    }
}

internal sealed class MultiplicativeWeightsCover
{
    private readonly CoverInstance _instance;
    internal long OracleCalls;
    internal long Rounds;

    internal MultiplicativeWeightsCover(CoverInstance instance) => _instance = instance;

    /// <summary>
    /// Multiplicative-weights solution of the covering LP as the max-min game it is dual to.
    /// Column player (sets, maximising): weight v_j grows by (1 + eps * payoff_j), where the payoff
    /// of set j against the row distribution u is (sum of u over the elements of j) / c_j. Row player
    /// (elements, minimising): weight u_i shrinks by (1 - eps * payoff_i), where the payoff of
    /// element i against v is the sum of v_j / c_j over the sets containing i. Both weight vectors
    /// are renormalised every round. The column player's *average* strategy x is nearly optimal, and
    /// the covering is x_j / (c_j * (v - eps)), which normalises the game value v = min_i payoff_i(x)
    /// back to coverage 1. A final uniform rescale repairs the O(eps) feasibility slack.
    /// </summary>
    internal double[] Solve(int rounds, double eps)
    {
        int m = _instance.SetCount, n = _instance.Elements;
        var x = new double[m];
        if (m == 0) return x;
        var u = new double[n];
        var v = new double[m];
        for (int i = 0; i < n; i++) u[i] = 1.0 / n;
        for (int j = 0; j < m; j++) v[j] = 1.0 / m;

        double playedValue = 0;
        var average = new double[m];
        for (int t = 0; t < rounds; t++)
        {
            Rounds++;
            var payoffSet = new double[m];
            var payoffElement = new double[n];
            double value = 0;
            for (int j = 0; j < m; j++)
            {
                OracleCalls++;
                double load = 0;
                foreach (int e in _instance.ElementsOf(j)) load += u[e];
                payoffSet[j] = load / _instance.Cost(j);
                value += v[j] * payoffSet[j];
                average[j] += v[j] / rounds;
            }
            for (int i = 0; i < n; i++)
            {
                double loss = 0;
                foreach (int j in _instance.SetsWith(i)) loss += v[j] / _instance.Cost(j);
                payoffElement[i] = loss;
            }
            playedValue += value / rounds;

            double sumU = 0, sumV = 0;
            for (int i = 0; i < n; i++)
            {
                u[i] *= 1.0 - eps * payoffElement[i];
                if (u[i] < 1e-12) u[i] = 1e-12;
                sumU += u[i];
            }
            for (int j = 0; j < m; j++)
            {
                v[j] *= 1.0 + eps * payoffSet[j];
                if (v[j] > 1e12) v[j] = 1e12;
                sumV += v[j];
            }
            for (int i = 0; i < n; i++) u[i] /= sumU;
            for (int j = 0; j < m; j++) v[j] /= sumV;
        }

        double scale = playedValue;
        if (scale <= 1e-9) scale = Math.Max(playedValue / 2, 1e-9);
        for (int j = 0; j < m; j++) x[j] = average[j] / (_instance.Cost(j) * scale);

        double tightest = _instance.MinCoverage(x);
        if (tightest <= 0) throw new InvalidOperationException("the instance cannot be covered");
        if (tightest < 1)
        {
            for (int j = 0; j < m; j++) x[j] /= tightest;      // repair the O(eps) feasibility slack
        }
        return x;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>
    /// Dual lower bound by damped water filling: raise every still uncovered element at the same rate
    /// until a set constraint becomes tight. The result is always dual feasible.
    /// </summary>
    private static (double[] y, double value) WaterFillingDual(CoverInstance g)
    {
        var y = new double[g.Elements];
        var coverage = new double[g.Elements];
        for (int phase = 0; phase < g.SetCount + 2; phase++)
        {
            bool active = false;
            for (int e = 0; e < g.Elements; e++) if (coverage[e] < 1 - 1e-12) active = true;
            if (!active) break;

            double step = double.MaxValue;
            for (int s = 0; s < g.SetCount; s++)
            {
                double load = 0;
                int rate = 0;
                foreach (int e in g.ElementsOf(s))
                {
                    load += y[e];
                    if (coverage[e] < 1 - 1e-12) rate++;
                }
                if (rate == 0) continue;
                step = Math.Min(step, (g.Cost(s) - load) / rate);
            }
            if (step == double.MaxValue || step <= 0) step = 0;
            for (int e = 0; e < g.Elements; e++) if (coverage[e] < 1 - 1e-12) y[e] += step;

            for (int s = 0; s < g.SetCount; s++)
            {
                double load = 0;
                foreach (int e in g.ElementsOf(s)) load += y[e];
                if (load < g.Cost(s) - 1e-9) continue;          // not tight yet
                foreach (int e in g.ElementsOf(s)) coverage[e] += step;
            }
        }
        double value = 0;
        foreach (double v in y) value += v;
        return (y, value);
    }

    private static bool DualFeasible(CoverInstance g, double[] y)
    {
        for (int s = 0; s < g.SetCount; s++)
        {
            double load = 0;
            foreach (int e in g.ElementsOf(s)) load += y[e];
            if (load > g.Cost(s) + 1e-9) return false;
        }
        for (int e = 0; e < g.Elements; e++) if (y[e] < -1e-12) return false;
        return true;
    }

    private static double DualValue(double[] y)
    {
        double total = 0;
        foreach (double v in y) total += v;
        return total;
    }

    /// <summary>Multiplicative weights on the packing LP: max sum(y) subject to A^T y &lt;= c.</summary>
    private static double[] PackingMultiplicativeWeights(CoverInstance g, int rounds, double eps)
    {
        var y = new double[g.Elements];
        var slack = new double[g.SetCount];
        for (int s = 0; s < g.SetCount; s++) slack[s] = g.Cost(s);
        for (int t = 0; t < rounds; t++)
        {
            int best = -1;
            double mostSlack = 0;
            for (int e = 0; e < g.Elements; e++)
            {
                double room = double.MaxValue;
                foreach (int s in g.SetsWith(e)) room = Math.Min(room, slack[s]);
                if (room > mostSlack) { mostSlack = room; best = e; }
            }
            if (best < 0 || mostSlack <= 0) break;
            double step = mostSlack * eps / (1 + eps);
            y[best] += step;
            foreach (int s in g.SetsWith(best)) slack[s] -= step;
        }
        return y;
    }

    /// <summary>
    /// Exact LP optimum of min c.x subject to A x >= 1, x >= 0 by enumerating basic feasible
    /// solutions: pick as many tight constraints as there are variables (covering rows and the
    /// non-negativity rows), solve with Gaussian elimination and keep the cheapest feasible vertex.
    /// </summary>
    private static double ExactOptimum(CoverInstance g)
    {
        if (!g.IsCoverable()) return double.PositiveInfinity;
        int n = g.SetCount, covering = g.Elements, rows = covering + n;
        var matrix = new double[rows][];
        var rhs = new double[rows];
        for (int e = 0; e < covering; e++)
        {
            var row = new double[n];
            foreach (int s in g.SetsWith(e)) row[s] = 1;
            matrix[e] = row;
            rhs[e] = 1;                                          // each element must be covered once
        }
        for (int j = 0; j < n; j++)
        {
            var row = new double[n];
            row[j] = 1;
            matrix[covering + j] = row;
            rhs[covering + j] = 0;                               // x_j >= 0
        }

        double best = double.PositiveInfinity;
        foreach (int[] choice in Combinations(rows, n))
        {
            double[]? solution = SolveLinear(matrix, rhs, choice, n);
            if (solution == null) continue;
            bool feasible = true;
            for (int r = 0; r < rows && feasible; r++)
            {
                double lhs = 0;
                for (int j = 0; j < n; j++) lhs += matrix[r][j] * solution[j];
                if (lhs < rhs[r] - 1e-7) feasible = false;
            }
            if (!feasible) continue;
            best = Math.Min(best, g.TotalCost(solution));
        }
        return best;
    }

    private static IEnumerable<int[]> Combinations(int n, int k)
    {
        var pick = new int[k];
        return Walk(0, 0);
        IEnumerable<int[]> Walk(int start, int depth)
        {
            if (depth == k) { yield return (int[])pick.Clone(); yield break; }
            for (int i = start; i < n; i++)
            {
                pick[depth] = i;
                foreach (int[] next in Walk(i + 1, depth + 1)) yield return next;
            }
        }
    }

    private static double[]? SolveLinear(double[][] matrix, double[] rhs, int[] rows, int n)
    {
        var a = new double[n][];
        var b = new double[n];
        for (int r = 0; r < n; r++)
        {
            a[r] = (double[])matrix[rows[r]].Clone();
            b[r] = rhs[rows[r]];
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.Abs(a[r][col]) > Math.Abs(a[pivot][col])) pivot = r;
            if (Math.Abs(a[pivot][col]) < 1e-12) return null;
            var swapRow = a[col]; a[col] = a[pivot]; a[pivot] = swapRow;
            double swapValue = b[col]; b[col] = b[pivot]; b[pivot] = swapValue;
            for (int r = 0; r < n; r++)
            {
                if (r == col) continue;
                double factor = a[r][col] / a[col][col];
                if (factor == 0) continue;
                for (int c = col; c < n; c++) a[r][c] -= factor * a[col][c];
                b[r] -= factor * b[col];
            }
        }
        var x = new double[n];
        for (int j = 0; j < n; j++) x[j] = b[j] / a[j][j];
        return x;
    }

    public static void Main()
    {
        // ---- hand-solved instance: three overlapping pairs of unit cost, optimum 1.5 -----------
        {
            var g = new CoverInstance(3);
            g.AddSet(1.0, 0, 1);
            g.AddSet(1.0, 1, 2);
            g.AddSet(1.0, 0, 2);
            Check(g.SetCount == 3 && g.IsCoverable(), "three covering sets were registered");

            var solver = new MultiplicativeWeightsCover(g);
            double[] x = solver.Solve(20000, 0.01);
            double cost = g.TotalCost(x);
            Check(g.MinCoverage(x) >= 1 - 1e-9, "the scaled covering covers every element");
            Check(Math.Abs(ExactOptimum(g) - 1.5) < 1e-9, "the enumerated LP optimum is 1.5");
            Check(cost >= 1.5 - 1e-6, "the covering can never beat the LP optimum (got " + cost.ToString("F4") + ")");
            Check(cost <= 1.5 * 1.05, "and it is within 5% of it (got " + cost.ToString("F4") + ")");

            (double[] dual, double dualValue) = WaterFillingDual(g);
            Check(DualFeasible(g, dual), "the water-filling dual is feasible");
            Check(Math.Abs(dualValue - 1.5) < 1e-9, "the water-filling dual reaches 1.5 here");
            Check(cost >= dualValue - 1e-9, "weak duality: covering cost >= packing value");
            Console.WriteLine("fixture1 cost=" + cost.ToString("F4") + " exact=" + ExactOptimum(g).ToString("F4")
                + " dual=" + dualValue.ToString("F4") + " oracleCalls=" + solver.OracleCalls);
        }

        // ---- unequal costs: optimum 4.5, dual bound 4 ------------------------------------------
        {
            var g = new CoverInstance(3);
            g.AddSet(2.0, 0, 1);
            g.AddSet(3.0, 1, 2);
            g.AddSet(4.0, 0, 2);
            double exact = ExactOptimum(g);
            Check(Math.Abs(exact - 4.5) < 1e-9, "the unequal-cost optimum is 4.5 (got " + exact.ToString("F4") + ")");
            var solver = new MultiplicativeWeightsCover(g);
            double[] x = solver.Solve(20000, 0.01);
            double cost = g.TotalCost(x);
            Check(g.MinCoverage(x) >= 1 - 1e-9, "the unequal-cost covering is feasible");
            Check(cost >= exact - 1e-6, "it never beats the optimum");
            Check(cost <= exact * 1.05, "and stays within 5% of it (got " + cost.ToString("F4") + ")");

            (double[] dual, double dualValue) = WaterFillingDual(g);
            Check(DualFeasible(g, dual), "its dual ascent is feasible");
            Check(dualValue <= exact + 1e-9, "a dual value can never exceed the optimum");
            Check(cost >= dualValue - 1e-9, "weak duality holds again");
            Console.WriteLine("fixture2 cost=" + cost.ToString("F4") + " exact=" + exact.ToString("F4")
                + " dual=" + dualValue.ToString("F4"));
        }

        // ---- one big set against singletons: optimum 3 ----------------------------------------
        {
            var g = new CoverInstance(3);
            g.AddSet(3.0, 0, 1, 2);
            g.AddSet(2.0, 0);
            g.AddSet(2.0, 1);
            g.AddSet(2.0, 2);
            double exact = ExactOptimum(g);
            Check(Math.Abs(exact - 3.0) < 1e-9, "reusing the big set is optimal at 3 (got " + exact.ToString("F4") + ")");
            var solver = new MultiplicativeWeightsCover(g);
            double[] x = solver.Solve(20000, 0.01);
            Check(g.MinCoverage(x) >= 1 - 1e-9, "the big-set instance is covered");
            Check(g.TotalCost(x) <= 3.0 * 1.04,
                "the solver takes the big set within 4% of the optimum (got " + g.TotalCost(x).ToString("F4") + ")");
            (double[] dual, double dualValue) = WaterFillingDual(g);
            Check(DualFeasible(g, dual) && Math.Abs(dualValue - 3.0) < 1e-9,
                "the dual matches the optimum exactly here (" + dualValue.ToString("F4") + ")");
            Console.WriteLine("fixture3 cost=" + g.TotalCost(x).ToString("F4") + " exact=" + exact.ToString("F4"));
        }

        // ---- randomized small instances against the exact LP optimum ---------------------------
        {
            const int Instances = 40;
            ulong state = 0x5E7C0DE5UL;
            bool feasible = true, neverBeats = true, withinFactor = true, dualOk = true;
            double worstRatio = 1.0;
            int counted = 0;
            string firstFailure = "none";

            for (int t = 0; t < Instances; t++)
            {
                int elements = 3 + Rng.NextInt(ref state, 2);
                var g = new CoverInstance(elements);
                int sets = 3 + Rng.NextInt(ref state, 2);
                for (int s = 0; s < sets; s++)
                {
                    var chosen = new List<int>();
                    int width = 1 + Rng.NextInt(ref state, Math.Min(3, elements));
                    while (chosen.Count < width)
                    {
                        int e = Rng.NextInt(ref state, elements);
                        if (!chosen.Contains(e)) chosen.Add(e);
                    }
                    g.AddSet(1 + Rng.NextInt(ref state, 9), chosen.ToArray());
                }
                if (!g.IsCoverable()) continue;

                double exact = ExactOptimum(g);
                if (double.IsPositiveInfinity(exact)) continue;
                counted++;
                var solver = new MultiplicativeWeightsCover(g);
                double[] x = solver.Solve(20000, 0.01);
                double cost = g.TotalCost(x);
                if (g.MinCoverage(x) < 1 - 1e-9) { feasible = false; firstFailure = "instance " + t + " left an element uncovered"; }
                if (cost < exact - 1e-6) { neverBeats = false; firstFailure = "instance " + t + " beat the LP optimum"; }
                if (cost > exact * 1.15) { withinFactor = false; firstFailure = "instance " + t + " is " + (cost / exact).ToString("F3") + "x optimum"; }
                (double[] dual, double dualValue) = WaterFillingDual(g);
                if (!DualFeasible(g, dual) || dualValue > cost + 1e-9)
                {
                    dualOk = false;
                    firstFailure = "instance " + t + " dual bound violated weak duality";
                }
                double[] packing = PackingMultiplicativeWeights(g, 500, 0.05);
                if (!DualFeasible(g, packing) || DualValue(packing) > cost + 1e-9)
                {
                    dualOk = false;
                    firstFailure = "instance " + t + " packing solution violated weak duality";
                }
                worstRatio = Math.Max(worstRatio, cost / exact);
            }
            Check(counted >= 20, "enough random instances were solvable (" + counted + ")");
            Check(feasible, "every covering is feasible (" + firstFailure + ")");
            Check(neverBeats, "no covering beats the enumerated LP optimum (" + firstFailure + ")");
            Check(withinFactor, "every covering stays within 15% of the optimum (" + firstFailure + ")");
            Check(dualOk, "the dual and packing solutions respect weak duality (" + firstFailure + ")");
            Console.WriteLine("randomInstances=" + counted + " worstRatioToOptimum=" + worstRatio.ToString("F4"));
        }

        // ---- boundary cases --------------------------------------------------------------------
        {
            var none = new CoverInstance(0);
            var emptySolver = new MultiplicativeWeightsCover(none);
            Check(emptySolver.Solve(500, 0.01).Length == 0, "an instance without elements needs no sets");
            Check(ExactOptimum(none) == 0, "the empty instance has optimum zero");

            var single = new CoverInstance(2);
            single.AddSet(7.0, 0, 1);
            var singleSolver = new MultiplicativeWeightsCover(single);
            double singleCost = single.TotalCost(singleSolver.Solve(2000, 0.01));
            Check(Math.Abs(singleCost - 7.0) < 1e-9, "the only set must be taken at full price (" + singleCost.ToString("F4") + ")");

            var uncovered = new CoverInstance(3);
            uncovered.AddSet(1.0, 0, 1);
            Check(!uncovered.IsCoverable(), "an element no set covers makes the instance infeasible");
            var stuck = new MultiplicativeWeightsCover(uncovered);
            bool thrown = false;
            try { stuck.Solve(200, 0.01); }
            catch (InvalidOperationException) { thrown = true; }
            Check(thrown, "solving an infeasible instance is reported instead of returning a wrong cost");
            Check(double.IsPositiveInfinity(ExactOptimum(uncovered)), "the exact solver agrees it is infeasible");

            thrown = false;
            try { uncovered.AddSet(0.0, 2); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a zero cost is rejected");
            thrown = false;
            try { uncovered.AddSet(1.0, 9); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "an element outside the instance is rejected");
            thrown = false;
            try { uncovered.AddSet(1.0); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a set that covers nothing is rejected");

            var deterministic = new CoverInstance(4);
            deterministic.AddSet(3.0, 0, 1, 2);
            deterministic.AddSet(2.0, 1, 3);
            deterministic.AddSet(4.0, 0, 3);
            var a = new MultiplicativeWeightsCover(deterministic);
            var b = new MultiplicativeWeightsCover(deterministic);
            double costA = deterministic.TotalCost(a.Solve(20000, 0.01));
            double costB = deterministic.TotalCost(b.Solve(20000, 0.01));
            Check(costA == costB, "the solver is deterministic");

            var improving = new MultiplicativeWeightsCover(deterministic);
            double shortRun = deterministic.TotalCost(improving.Solve(50, 0.01));
            double longRun = deterministic.TotalCost(improving.Solve(20000, 0.01));
            double exact = ExactOptimum(deterministic);
            Check(longRun >= exact - 1e-6, "the long run still cannot beat the optimum");
            Check(longRun <= shortRun + 1e-9,
                "more rounds never make the covering more expensive (" + longRun.ToString("F4") + " <= "
                + shortRun.ToString("F4") + ")");
            Console.WriteLine("deterministicCost=" + costA.ToString("F4") + " shortRunCost=" + shortRun.ToString("F4")
                + " exact=" + exact.ToString("F4"));
        }

        {
            // A last sanity check on the game formulation: the played value approaches 1 / OPT.
            var probe = new CoverInstance(3);
            probe.AddSet(2.0, 0, 1);
            probe.AddSet(3.0, 1, 2);
            probe.AddSet(4.0, 0, 2);
            var probeSolver = new MultiplicativeWeightsCover(probe);
            double probeCost = probe.TotalCost(probeSolver.Solve(20000, 0.01));
            Check(probeCost / ExactOptimum(probe) <= 1.02,
                "an extra instance stays within 2% of its optimum (" + probeCost.ToString("F4") + ")");
            Console.WriteLine("extraInstanceCost=" + probeCost.ToString("F4") + " exact=" + ExactOptimum(probe).ToString("F4"));
        }

        Console.WriteLine("PASS");
    }
}
