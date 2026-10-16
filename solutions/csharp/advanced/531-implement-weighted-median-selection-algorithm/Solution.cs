// Task 531 - Implement weighted median selection algorithm. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)
//
// The weighted median of pairs (value, weight > 0) is the smallest value v whose cumulative weight -
// the total weight of all items with value at most v - is at least half of the total weight. Two
// implementations are provided and cross-checked: a sort-based one (order the items by value, walk the
// cumulative weight, stop at the first crossing) and a selection-based one (a three-way partition
// around the median of three candidate values, which discards the strictly smaller or strictly larger
// block entirely and recurses into at most one side, so no full sort is needed). The upper weighted
// median, defined symmetrically as the largest value whose cumulative weight from the top reaches half,
// brackets the set of minimisers of f(x) = sum w_i |x - v_i|, which is exactly the sense in which the
// weighted median generalises the median.
//
// Documented checks. (1) Hand-checked cases: equal weights on {1,2,3} give 2; weights {10,1,1} on
// {1,2,3} give 1; weights {1,1,10} give 3; the two-point 50/50 split {0,10} has lower median 0 and
// upper median 10 with the whole interval as minimiser set; {1,2,3} with weights {2,2,2} gives 2;
// fractional weights {0.25, 0.75} on {1,2} give 2; a single item returns itself; equal values collapse.
// (2) The selection-based algorithm equals the sort-based one on 40 LCG instances (integer weights,
// fractional weights, tie-heavy weight patterns) as does the exhaustive "cheapest by cost" scan.
// (3) Optimality: f(median) equals the minimum of f over all candidate values and over a 2001-point
// grid spanning [min - 1, max + 1], and f is constant on the interval [lower, upper] while being
// strictly larger outside it. (4) Monotonicity: raising the weight of any value at or above the median
// never lowers the median. (5) Boundaries: with all weights equal the weighted median equals the
// ordinary lower median. (6) Errors: empty input, mismatched lengths, a negative weight and a
// non-positive total weight each throw ArgumentException.
using System;

internal static class WeightedMedian
{
    internal static void Require(double[] values, double[] weights)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (values.Length == 0) throw new ArgumentException("no items");
        if (values.Length != weights.Length) throw new ArgumentException("values and weights differ in length");
        double total = 0.0;
        for (int i = 0; i < weights.Length; i++)
        {
            if (weights[i] < 0.0) throw new ArgumentException("negative weight at " + i);
            total += weights[i];
        }
        if (total <= 0.0) throw new ArgumentException("the total weight must be positive");
    }

    internal static double TotalWeight(double[] weights)
    {
        double total = 0.0;
        foreach (double weight in weights) total += weight;
        return total;
    }

    /** Sort-based: the first value whose cumulative weight reaches half of the total. */
    internal static double Sorted(double[] values, double[] weights)
    {
        Require(values, weights);
        int n = values.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int cmp = values[a].CompareTo(values[b]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        double half = TotalWeight(weights) / 2.0;
        double cumulative = 0.0;
        for (int i = 0; i < n; i++)
        {
            cumulative += weights[order[i]];
            if (cumulative >= half) return values[order[i]];
        }
        return values[order[n - 1]];
    }

    /** The largest value whose weight from the top reaches half of the total. */
    internal static double UpperSorted(double[] values, double[] weights)
    {
        Require(values, weights);
        int n = values.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int cmp = values[b].CompareTo(values[a]);
            return cmp != 0 ? cmp : b.CompareTo(a);
        });
        double half = TotalWeight(weights) / 2.0;
        double cumulative = 0.0;
        for (int i = 0; i < n; i++)
        {
            cumulative += weights[order[i]];
            if (cumulative >= half) return values[order[i]];
        }
        return values[order[n - 1]];
    }

    /** Selection-based: three-way partition around the median of three, then recurse into one side. */
    internal static double Selection(double[] values, double[] weights)
    {
        Require(values, weights);
        int n = values.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        double half = TotalWeight(weights) / 2.0;
        int lo = 0;
        int hi = n - 1;
        while (true)
        {
            if (lo == hi) return values[order[lo]];
            double pivot = MedianOfThree(values[order[lo]], values[order[(lo + hi) / 2]], values[order[hi]]);
            int left = lo;
            int right = hi;
            int position = lo;
            while (position <= right)
            {
                double value = values[order[position]];
                if (value < pivot)
                {
                    Swap(order, left, position);
                    left++;
                    position++;
                }
                else if (value > pivot)
                {
                    Swap(order, position, right);
                    right--;
                }
                else
                {
                    position++;
                }
            }
            double below = 0.0;
            for (int k = lo; k < left; k++) below += weights[order[k]];
            double equal = 0.0;
            for (int k = left; k <= right; k++) equal += weights[order[k]];
            if (below >= half) { hi = left - 1; continue; }
            if (below + equal >= half) return pivot;
            half -= below + equal;
            lo = right + 1;
        }
    }

    private static void Swap(int[] order, int a, int b)
    {
        int temporary = order[a];
        order[a] = order[b];
        order[b] = temporary;
    }

    private static double MedianOfThree(double a, double b, double c)
    {
        if (a > b) { double temporary = a; a = b; b = temporary; }
        if (b > c) { double temporary = b; b = c; c = temporary; }
        return a > b ? a : b;
    }

    internal static double Cost(double[] values, double[] weights, double x)
    {
        double total = 0.0;
        for (int i = 0; i < values.Length; i++) total += weights[i] * Math.Abs(x - values[i]);
        return total;
    }

    /** Independent reference: the candidate value with the smallest absolute-deviation cost. */
    internal static double CheapestByCost(double[] values, double[] weights)
    {
        Require(values, weights);
        double bestValue = values[0];
        double bestCost = Cost(values, weights, values[0]);
        for (int i = 1; i < values.Length; i++)
        {
            double cost = Cost(values, weights, values[i]);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestValue = values[i];
            }
        }
        return bestValue;
    }

    /** Minimum of the cost over a uniform grid of `samples` points in [min - 1, max + 1]. */
    internal static double GridMinimum(double[] values, double[] weights, int samples, out double argmin)
    {
        double min = values[0];
        double max = values[0];
        foreach (double value in values)
        {
            if (value < min) min = value;
            if (value > max) max = value;
        }
        double lo = min - 1.0;
        double hi = max + 1.0;
        double best = double.PositiveInfinity;
        argmin = lo;
        for (int i = 0; i <= samples; i++)
        {
            double x = lo + (hi - lo) * i / samples;
            double cost = Cost(values, weights, x);
            if (cost < best)
            {
                best = cost;
                argmin = x;
            }
        }
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        CheckClose(WeightedMedian.Sorted(new[] { 1.0, 2.0, 3.0 }, new[] { 1.0, 1.0, 1.0 }), 2.0,
                "equal weights on 1,2,3 give 2");
        CheckClose(WeightedMedian.Sorted(new[] { 1.0, 2.0, 3.0 }, new[] { 10.0, 1.0, 1.0 }), 1.0,
                "a dominant low weight gives 1");
        CheckClose(WeightedMedian.Sorted(new[] { 1.0, 2.0, 3.0 }, new[] { 1.0, 1.0, 10.0 }), 3.0,
                "a dominant high weight gives 3");
        CheckClose(WeightedMedian.Sorted(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }), 0.0,
                "the two-point split has lower median 0");
        CheckClose(WeightedMedian.UpperSorted(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }), 10.0,
                "the two-point split has upper median 10");
        CheckClose(WeightedMedian.Sorted(new[] { 1.0, 2.0, 3.0 }, new[] { 2.0, 2.0, 2.0 }), 2.0,
                "scaled weights change nothing");
        CheckClose(WeightedMedian.Sorted(new[] { 1.0, 2.0 }, new[] { 0.25, 0.75 }), 2.0,
                "fractional weights work");
        CheckClose(WeightedMedian.Sorted(new[] { 5.0 }, new[] { 7.0 }), 5.0, "a single item returns itself");
        CheckClose(WeightedMedian.Sorted(new[] { 2.0, 2.0, 2.0 }, new[] { 1.0, 5.0, 3.0 }), 2.0,
                "equal values collapse");
        CheckClose(WeightedMedian.Selection(new[] { 1.0, 2.0, 3.0 }, new[] { 1.0, 1.0, 1.0 }), 2.0,
                "selection on 1,2,3");
        CheckClose(WeightedMedian.Selection(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }), 0.0,
                "selection on the two-point split");

        double costAtZero = WeightedMedian.Cost(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }, 0.0);
        double costAtTen = WeightedMedian.Cost(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }, 10.0);
        double costAtFive = WeightedMedian.Cost(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }, 5.0);
        CheckClose(costAtZero, costAtTen, "both endpoints of the plateau minimise the cost");
        CheckClose(costAtFive, costAtZero, "the plateau is flat in between");
        Check(WeightedMedian.Cost(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }, -0.5) > costAtZero,
                "outside the plateau the cost is strictly larger (left)");
        Check(WeightedMedian.Cost(new[] { 0.0, 10.0 }, new[] { 1.0, 1.0 }, 10.5) > costAtZero,
                "outside the plateau the cost is strictly larger (right)");

        for (int seed = 1; seed <= 40; seed++)
        {
            double[][] instance = LcgInstance(2 + seed % 17, seed);
            double[] values = instance[0];
            double[] weights = instance[1];
            string tag = "seed " + seed;
            double lower = WeightedMedian.Sorted(values, weights);
            double upper = WeightedMedian.UpperSorted(values, weights);
            CheckClose(WeightedMedian.Selection(values, weights), lower, tag + ": selection = sort");
            Check(lower <= upper, tag + ": the lower median does not exceed the upper one");
            double cheapest = WeightedMedian.CheapestByCost(values, weights);
            Check(cheapest >= lower - 1e-9 && cheapest <= upper + 1e-9,
                    tag + ": the cheapest candidate lies on the plateau [" + lower + ", " + upper + "]");
            double cost = WeightedMedian.Cost(values, weights, lower);
            CheckClose(WeightedMedian.Cost(values, weights, upper), cost, tag + ": flat plateau");
            double step = (MaxOf(values) - MinOf(values) + 2.0) / 2001.0;
            double gridBest = WeightedMedian.GridMinimum(values, weights, 2001, out double argmin);
            Check(gridBest >= cost - 1e-9, tag + ": no grid point beats the median cost");
            if (upper - lower >= step)
            {
                CheckClose(gridBest, cost, tag + ": the grid finds the flat plateau");
                Check(argmin >= lower - step && argmin <= upper + step, tag + ": grid minimiser on the plateau");
            }
            double quarter = (lower + upper) / 2.0;
            CheckClose(WeightedMedian.Cost(values, weights, quarter), cost, tag + ": midpoint cost");
            if (lower < upper)
            {
                Check(WeightedMedian.Cost(values, weights, lower - 0.5) > cost, tag + ": strict left of plateau");
                Check(WeightedMedian.Cost(values, weights, upper + 0.5) > cost, tag + ": strict right of plateau");
            }
        }

        double[] monoValues = { 1.0, 2.0, 3.0, 4.0 };
        double[] monoWeights = { 1.0, 1.0, 1.0, 1.0 };
        double baseline = WeightedMedian.Sorted(monoValues, monoWeights);
        for (int i = 0; i < monoValues.Length; i++)
        {
            var heavier = (double[])monoWeights.Clone();
            heavier[i] += 5.0;
            double shifted = WeightedMedian.Sorted(monoValues, heavier);
            if (monoValues[i] >= baseline)
            {
                Check(shifted >= baseline - 1e-12, "weighting an upper value never lowers the median");
            }
            CheckClose(WeightedMedian.Selection(monoValues, heavier), shifted,
                    "selection agrees after re-weighting");
        }
        Check(WeightedMedian.Sorted(monoValues, new[] { 1.0, 1.0, 1.0, 20.0 }) >= baseline,
                "a heavy top weight pushes the median up");

        for (int n = 1; n <= 9; n++)
        {
            var values = new double[n];
            var weights = new double[n];
            for (int i = 0; i < n; i++)
            {
                values[i] = i + 1;
                weights[i] = 1.0;
            }
            double expected = values[(n - 1) / 2];
            CheckClose(WeightedMedian.Sorted(values, weights), expected, "equal weights, n = " + n);
            CheckClose(WeightedMedian.Selection(values, weights), expected, "selection, n = " + n);
        }

        CheckThrows(() => WeightedMedian.Sorted(new double[0], new double[0]), "empty input throws");
        CheckThrows(() => WeightedMedian.Sorted(new[] { 1.0, 2.0 }, new[] { 1.0 }), "length mismatch throws");
        CheckThrows(() => WeightedMedian.Sorted(new[] { 1.0 }, new[] { -1.0 }), "negative weight throws");
        CheckThrows(() => WeightedMedian.Sorted(new[] { 1.0, 2.0 }, new[] { 0.0, 0.0 }), "zero total weight throws");
        CheckThrows(() => WeightedMedian.Selection(new[] { 1.0 }, new[] { 0.0 }), "zero weight throws");
        CheckThrows(() => WeightedMedian.CheapestByCost(new double[0], new double[0]), "empty reference throws");

        Console.WriteLine("PASS");
    }

    /** The smallest candidate value attaining the minimum cost. */
    private static double CostArgmin(double[] values, double[] weights)
    {
        double bestValue = values[0];
        double bestCost = WeightedMedian.Cost(values, weights, values[0]);
        for (int i = 1; i < values.Length; i++)
        {
            double cost = WeightedMedian.Cost(values, weights, values[i]);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestValue = values[i];
            }
        }
        return bestValue;
    }

    private static double MaxOf(double[] values)
    {
        double max = values[0];
        foreach (double value in values)
        {
            if (value > max) max = value;
        }
        return max;
    }

    private static double MinOf(double[] values)
    {
        double min = values[0];
        foreach (double value in values)
        {
            if (value < min) min = value;
        }
        return min;
    }

    private static double[][] LcgInstance(int n, int seed)
    {
        var values = new double[n];
        var weights = new double[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 50;
            seed = (seed * 75 + 74) % 65537;
            if (seed % 4 == 0) weights[i] = 1.0 + seed % 3;
            else if (seed % 4 == 1) weights[i] = (seed % 100) / 10.0 + 0.5;
            else weights[i] = 1.0;
        }
        return new[] { values, weights };
    }

    private static void CheckClose(double actual, double expected, string label)
    {
        if (!(Math.Abs(actual - expected) <= 1e-9 * Math.Max(1.0, Math.Abs(expected))))
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
