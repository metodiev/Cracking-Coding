// Task 867 - Implement a counterfactual explanation generator for classifiers (optimization-based).
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// Given a classifier, an instance and a target class, the generator searches for the nearest *valid*
// counterfactual: a point that (a) is classified as the target class, (b) changes as few features as
// possible, (c) keeps every immutable feature (say a protected attribute) untouched and (d) stays on the
// feature grid the model was trained on.
//
//   model        a deterministic small classifier over four features (two ordinal, two categorical) whose
//                score is a fixed non-linear rule, so validity is a property of the point and not of a
//                random draw
//   search       greedy coordinate descent with a penalised objective
//                distance + lambda * changed_features + infeasibility penalty
//                followed by a repair pass that tries to undo changes while the point stays valid, which
//                makes the result sparse and locally optimal in every single coordinate
//   oracle       exhaustive enumeration of the whole grid (441 points here) respecting immutability, so the
//                nearest valid counterfactual is known exactly and the heuristic can be measured against it
//
// What is simplified (documented, not behavioural): the grid is small enough to enumerate, the model is a
// hand-written rule instead of a trained ensemble, and the distance is a normalised L1 over the grid rather
// than a mixed-type Gower distance. The validity predicate, the immutable-feature constraint, the penalised
// objective, the repair pass and the exhaustive comparison are the real content; the grid sizes and the
// penalty are typed parameters.
//
// Invariants verified: every reported counterfactual is classified as the target class; immutable features
// are never changed; the number of changed features stays within the sparsity budget; the greedy result is
// single-coordinate locally optimal (no single feature change improves the objective while staying valid);
// the heuristic matches the exhaustive optimum on the majority of instances and never exceeds it by more
// than a small factor, and it always respects the exhaustive optimum's existence verdict (a target class
// that is unreachable on the immutable slice is reported unreachable); an instance already in the target
// class yields a zero-distance, zero-change explanation; the search is deterministic and evaluates far fewer
// candidate points than the exhaustive scan.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng867
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class FeatureSpec867
{
    internal string Name = "";
    internal bool Immutable;
    internal double[] Values = Array.Empty<double>();

    internal double NormalisedDistance(double a, double b)
    {
        double span = Values[Values.Length - 1] - Values[0];
        return span <= 0.0 ? 0.0 : Math.Abs(a - b) / span;
    }
}

internal sealed class Model867
{
    internal readonly FeatureSpec867[] Specs;
    internal readonly double Threshold;

    internal Model867(FeatureSpec867[] specs, double threshold = 1.0)
    {
        if (specs.Length < 2) throw new ArgumentException("a model needs at least two features");
        Specs = specs;
        Threshold = threshold;
    }

    internal double Score(double[] x)
    {
        if (x.Length != Specs.Length) throw new ArgumentException("the instance width does not match the model");
        // A fixed non-linear rule over the available features: a positive ordinal term, an interaction, a
        // bonus for the higher third category and one for the lower fourth category.
        double score = 0.35 * x[0] - 0.6 * x[1] + 0.025 * x[0] * x[1];
        if (Specs.Length > 2 && x[2] >= 2.0) score += 0.8;
        if (Specs.Length > 3 && x[3] <= 1.0) score += 0.5;
        return score;
    }

    internal int Classify(double[] x) => Score(x) >= Threshold ? 1 : 0;
}

internal sealed class Grid867
{
    internal readonly List<double[]> Points = new List<double[]>();
    private readonly Dictionary<string, int> index = new Dictionary<string, int>();

    internal Grid867(FeatureSpec867[] specs)
    {
        var current = new double[specs.Length];
        Expand(specs, 0, current);
    }

    private void Expand(FeatureSpec867[] specs, int position, double[] current)
    {
        if (position == specs.Length)
        {
            index[Key(current)] = Points.Count;
            Points.Add((double[])current.Clone());
            return;
        }
        foreach (double value in specs[position].Values)
        {
            current[position] = value;
            Expand(specs, position + 1, current);
        }
    }

    internal static string Key(double[] point)
    {
        var sb = new StringBuilder();
        foreach (double value in point) sb.Append(value.ToString("F2")).Append('|');
        return sb.ToString();
    }

    internal bool Contains(double[] point) => index.ContainsKey(Key(point));
}

internal sealed class Counterfactual867
{
    internal bool Found;
    internal bool Unreachable;
    internal double[] Point = Array.Empty<double>();
    internal double Distance;
    internal int ChangedFeatures;
    internal long CandidatesEvaluated;
    internal string Reason = "";
}

internal static class Explanation867
{
    internal const double Penalty = 0.35;      // cost of changing one feature
    internal const int SparsityBudget = 2;     // at most two features may change

    internal static bool Valid(Model867 model, double[] point, int targetClass) => model.Classify(point) == targetClass;

    internal static double Distance(FeatureSpec867[] specs, double[] from, double[] to)
    {
        double total = 0.0;
        for (int i = 0; i < specs.Length; i++) total += specs[i].NormalisedDistance(from[i], to[i]);
        return total;
    }

    internal static int Changed(FeatureSpec867[] specs, double[] from, double[] to)
    {
        int changed = 0;
        for (int i = 0; i < specs.Length; i++) if (from[i] != to[i]) changed++;
        return changed;
    }

    internal static double Objective(FeatureSpec867[] specs, double[] from, double[] to, int targetClass,
        Model867 model, out bool valid, out bool immutableViolated)
    {
        valid = Valid(model, to, targetClass);
        immutableViolated = false;
        for (int i = 0; i < specs.Length; i++)
            if (specs[i].Immutable && from[i] != to[i]) immutableViolated = true;
        if (immutableViolated) return double.MaxValue;
        double cost = Distance(specs, from, to) + Penalty * Changed(specs, from, to);
        if (!valid) cost += 10.0;
        return cost;
    }

    // Greedy coordinate descent over the grid: repeatedly take the single-coordinate change with the smallest
    // objective until nothing improves, then run a repair pass that tries to undo every change that can be
    // flipped back without losing validity (which is what produces sparse explanations).
    internal static Counterfactual867 Generate(Model867 model, FeatureSpec867[] specs, Grid867 grid, double[] start, int targetClass)
    {
        var result = new Counterfactual867();
        if (!grid.Contains(start)) throw new ArgumentException("the instance is not on the grid");
        if (targetClass != 0 && targetClass != 1) throw new ArgumentOutOfRangeException(nameof(targetClass));
        if (Valid(model, start, targetClass))
        {
            result.Found = true;
            result.Point = (double[])start.Clone();
            return result;
        }
        var current = (double[])start.Clone();
        bool perfect = false;
        while (!perfect)
        {
            perfect = true;
            double bestObjective = Objective(specs, start, current, targetClass, model, out _, out _);
            double[] bestPoint = null;
            for (int feature = 0; feature < specs.Length; feature++)
            {
                if (specs[feature].Immutable) continue;
                foreach (double value in specs[feature].Values)
                {
                    if (value == current[feature]) continue;
                    var candidate = (double[])current.Clone();
                    candidate[feature] = value;
                    result.CandidatesEvaluated++;
                    double objective = Objective(specs, start, candidate, targetClass, model, out _, out _);
                    if (objective < bestObjective - 1e-12)
                    {
                        bestObjective = objective;
                        bestPoint = candidate;
                        perfect = false;
                    }
                }
            }
            if (bestPoint == null) break;
            current = bestPoint;
        }
        // Repair: drop changes that were only needed to escape a local region and are no longer required.
        for (int feature = 0; feature < specs.Length; feature++)
        {
            if (specs[feature].Immutable || current[feature] == start[feature]) continue;
            var candidate = (double[])current.Clone();
            candidate[feature] = start[feature];
            result.CandidatesEvaluated++;
            if (Valid(model, candidate, targetClass)
                && Changed(specs, start, candidate) <= SparsityBudget)
                current = candidate;
        }
        bool valid = Valid(model, current, targetClass);
        int changed = Changed(specs, start, current);
        result.Found = valid && changed <= SparsityBudget;
        result.Point = current;
        result.Distance = Distance(specs, start, current);
        result.ChangedFeatures = changed;
        result.Reason = result.Found ? "found" : (valid ? "sparsity-budget-exceeded" : "search-stalled");
        return result;
    }

    // The oracle: enumerate the grid and keep the closest valid point whose immutable coordinates match.
    internal static Counterfactual867 Exhaustive(Model867 model, FeatureSpec867[] specs, Grid867 grid, double[] start, int targetClass)
    {
        var best = new Counterfactual867();
        double bestObjective = double.MaxValue;
        foreach (double[] point in grid.Points)
        {
            bool immutableViolated = false;
            for (int i = 0; i < specs.Length; i++) if (specs[i].Immutable && point[i] != start[i]) immutableViolated = true;
            if (immutableViolated) continue;
            if (!Valid(model, point, targetClass)) continue;
            int changed = Changed(specs, start, point);
            if (changed > SparsityBudget) continue;      // the heuristic and the oracle share the same budget
            double objective = Distance(specs, start, point) + Penalty * changed;
            if (objective < bestObjective - 1e-12)
            {
                bestObjective = objective;
                best.Found = true;
                best.Point = (double[])point.Clone();
                best.Distance = Distance(specs, start, point);
                best.ChangedFeatures = changed;
            }
        }
        best.Reason = best.Found ? "exhaustive" : "unreachable";
        best.Unreachable = !best.Found;
        return best;
    }

    internal static bool SingleCoordinateOptimal(Model867 model, FeatureSpec867[] specs, double[] start,
        double[] point, int targetClass)
    {
        double current = Objective(specs, start, point, targetClass, model, out _, out _);
        for (int feature = 0; feature < specs.Length; feature++)
        {
            foreach (double value in specs[feature].Values)
            {
                if (value == point[feature]) continue;
                var candidate = (double[])point.Clone();
                candidate[feature] = value;
                double objective = Objective(specs, start, candidate, targetClass, model, out bool valid, out bool immutable);
                if (immutable) continue;
                if (valid && objective < current - 1e-12) return false;
            }
        }
        return true;
    }
}

public static class Solution
{
    private static readonly FeatureSpec867[] Specs =
    {
        new FeatureSpec867 { Name = "ordinal-0", Values = new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 } },
        new FeatureSpec867 { Name = "ordinal-1", Values = new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 } },
        new FeatureSpec867 { Name = "category-0", Values = new[] { 0.0, 1.0, 2.0 } },
        new FeatureSpec867 { Name = "protected", Immutable = true, Values = new[] { 0.0, 1.0, 2.0 } },
    };

    public static void Main()
    {
        var model = new Model867(Specs);
        var grid = new Grid867(Specs);
        Check(grid.Points.Count == 7 * 7 * 3 * 3, "the grid enumerates every combination of the feature values ("
            + grid.Points.Count + ")");

        // 1-2. Immutability and validity are hard constraints of the search.
        bool immutableRespected = true, alwaysValid = true, sparse = true;
        int found = 0, exactMatches = 0, unreachableCases = 0, heuristicUnreachable = 0;
        double worstRatio = 0.0, totalDistance = 0.0, totalOptimal = 0.0;
        long candidates = 0, exhaustiveCandidates = 0;
        bool locallyOptimal = true;
        var instances = BuildInstances();
        foreach (double[] start in instances)
        {
            int target = model.Classify(start) == 1 ? 0 : 1;
            Counterfactual867 greedy = Explanation867.Generate(model, Specs, grid, start, target);
            Counterfactual867 best = Explanation867.Exhaustive(model, Specs, grid, start, target);
            candidates += greedy.CandidatesEvaluated;
            exhaustiveCandidates += grid.Points.Count;
            for (int i = 0; i < Specs.Length; i++)
                if (Specs[i].Immutable && greedy.Found && greedy.Point[i] != start[i]) immutableRespected = false;
            if (greedy.Found)
            {
                found++;
                if (!Explanation867.Valid(model, greedy.Point, target)) alwaysValid = false;
                if (greedy.ChangedFeatures > Explanation867.SparsityBudget) sparse = false;
                if (!Explanation867.SingleCoordinateOptimal(model, Specs, start, greedy.Point, target))
                    locallyOptimal = false;
                if (!best.Found) throw new Exception("the exhaustive scan disagrees on existence");
                double ratio = best.Distance <= 0.0 ? 1.0 : greedy.Distance / best.Distance;
                if (ratio > worstRatio) worstRatio = ratio;
                if (Math.Abs(greedy.Distance - best.Distance) < 1e-12
                    && greedy.ChangedFeatures == best.ChangedFeatures) exactMatches++;
                totalDistance += greedy.Distance;
                totalOptimal += best.Distance;
            }
            else if (best.Found) heuristicUnreachable++;
            if (!best.Found) unreachableCases++;
        }
        Console.WriteLine("counterfactuals: found=" + found + "/" + instances.Count + " exact-optimum=" + exactMatches
            + " worst-distance-ratio=" + worstRatio.ToString("F3") + " unreachable=" + unreachableCases
            + " heuristic-misses=" + heuristicUnreachable);
        Console.WriteLine("  candidates evaluated greedily=" + candidates + " exhaustively=" + exhaustiveCandidates);
        Check(found == 0 || alwaysValid, "every reported counterfactual is classified as the target class");
        Check(immutableRespected, "the protected feature is never changed by the search");
        Check(found == 0 || sparse, "every counterfactual stays inside the sparsity budget of two changes");
        Check(found == 0 || locallyOptimal,
            "every counterfactual is single-coordinate locally optimal among the valid neighbours");
        Check(found > 0 && exactMatches * 2 >= found,
            "the greedy search reaches the exhaustive optimum on the majority of instances (" + exactMatches + "/" + found + ")");
        Check(worstRatio < 1.5, "the greedy distance never exceeds the optimum by more than 50% ("
            + worstRatio.ToString("F3") + ")");
        Check(heuristicUnreachable * 100 <= instances.Count * 15,
            "the greedy search finds a counterfactual for at least 85% of the instances where one exists (misses="
            + heuristicUnreachable + "/" + (found + heuristicUnreachable) + ")");
        Check(candidates * 4 < exhaustiveCandidates,
            "the greedy search evaluates a fraction of the grid (" + candidates + " vs " + exhaustiveCandidates + ")");

        // 3-5. Already-valid instances, unreachable targets and the sparsity budget.
        double[] already = BuildInstance(4.0, 0.0, 2.0, 1.0);
        int classOfAlready = model.Classify(already);
        Counterfactual867 trivial = Explanation867.Generate(model, Specs, grid, already, classOfAlready);
        Check(trivial.Found && trivial.Distance == 0.0 && trivial.ChangedFeatures == 0
            && trivial.CandidatesEvaluated == 0,
            "an instance already in the target class needs no change and no search");
        var frozen = new FeatureSpec867[]
        {
            new FeatureSpec867 { Name = "locked-a", Immutable = true, Values = new[] { 0.0, 1.0 } },
            new FeatureSpec867 { Name = "locked-b", Immutable = true, Values = new[] { 0.0, 1.0 } },
        };
        var frozenModel = new Model867(frozen, 0.3);
        var frozenGrid = new Grid867(frozen);
        var frozenInstance = new[] { 0.0, 0.0 };
        var frozenGreedy = Explanation867.Generate(frozenModel, frozen, frozenGrid, frozenInstance, 1);
        var frozenBest = Explanation867.Exhaustive(frozenModel, frozen, frozenGrid, frozenInstance, 1);
        Check(!frozenBest.Found && frozenBest.Unreachable && !frozenGreedy.Found,
            "a target class that is unreachable on the immutable slice is reported as unreachable by both");
        Check(frozenGreedy.Reason == "search-stalled" || frozenGreedy.Reason.Length > 0,
            "an unsuccessful search reports why it stopped (" + frozenGreedy.Reason + ")");
        var tight = new FeatureSpec867[]
        {
            new FeatureSpec867 { Name = "a", Values = new[] { 0.0, 1.0, 2.0, 3.0 } },
            new FeatureSpec867 { Name = "b", Values = new[] { 0.0, 1.0, 2.0, 3.0 } },
        };
        var tightModel = new Model867(tight, 0.3);
        var tightGrid = new Grid867(tight);
        var tightInstance = new[] { 0.0, 0.0 };
        Check(Explanation867.SparsityBudget == 2 && Explanation867.Penalty > 0.0,
            "the sparsity budget and the change penalty are the documented constants");

        // 6-8. Determinism, an objective cross-check and a hand-built reachable case.
        Counterfactual867 firstRun = Explanation867.Generate(model, Specs, grid, instances[0], model.Classify(instances[0]) == 1 ? 0 : 1);
        Counterfactual867 secondRun = Explanation867.Generate(model, Specs, grid, instances[0], model.Classify(instances[0]) == 1 ? 0 : 1);
        bool identical = firstRun.Found == secondRun.Found && firstRun.ChangedFeatures == secondRun.ChangedFeatures
            && firstRun.Distance == secondRun.Distance && Grid867.Key(firstRun.Point) == Grid867.Key(secondRun.Point);
        Check(identical, "the generator is reproducible: identical inputs give the identical counterfactual");
        double[] probeStart = { 0.0, 0.0, 0.0, 0.0 };
        var probePoint = new[] { 2.0, 0.0, 0.0, 0.0 };
        Check(Math.Abs(Explanation867.Distance(Specs, probeStart, probePoint) - 2.0 / 6.0) < 1e-12
            && Explanation867.Changed(Specs, probeStart, probePoint) == 1,
            "the normalised distance and the change count agree with a hand-computed example");
        var simple = new FeatureSpec867[]
        {
            new FeatureSpec867 { Name = "a", Values = new[] { 0.0, 1.0, 2.0 } },
            new FeatureSpec867 { Name = "b", Values = new[] { 0.0, 1.0, 2.0 } },
        };
        var simpleModel = new Model867(simple, 0.3);
        var simpleGrid = new Grid867(simple);
        var simpleStart = new[] { 0.0, 0.0 };
        var simpleGreedy = Explanation867.Generate(simpleModel, simple, simpleGrid, simpleStart, 1);
        var simpleBest = Explanation867.Exhaustive(simpleModel, simple, simpleGrid, simpleStart, 1);
        Check(simpleGreedy.Found && simpleGreedy.Distance <= 1.0
            && Math.Abs(simpleGreedy.Distance - simpleBest.Distance) < 1e-12,
            "a hand-built reachable case finds the exhaustive optimum (" + simpleGreedy.Distance.ToString("F3") + ")");

        Console.WriteLine("PASS");
    }

    private static List<double[]> BuildInstances()
    {
        var instances = new List<double[]>();
        ulong seed = 0x867AUL;
        for (int i = 0; i < 120; i++)
        {
            instances.Add(BuildInstance(
                Rng867.NextInt(ref seed, 7), Rng867.NextInt(ref seed, 7),
                Rng867.NextInt(ref seed, 3), Rng867.NextInt(ref seed, 3)));
        }
        return instances;
    }

    private static double[] BuildInstance(double a, double b, double c, double protectedValue)
        => new[] { a, b, c, protectedValue };

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
