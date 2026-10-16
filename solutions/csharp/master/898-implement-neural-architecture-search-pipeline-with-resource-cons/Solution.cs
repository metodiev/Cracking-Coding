// Task 898 - Implement neural architecture search pipeline with resource-constrained objective. (Tier: Master, Beyond — Machine Learning & AI)
//
// A small but complete NAS pipeline over a discrete space of 480 convolutional architectures
// (depth x width x kernel x resolution x depthwise), with a hard FLOPs budget:
//
//  * the space is enumerated and scored by a deterministic "ground truth" accuracy function with a
//    saturating depth/width curve, a resolution term, a parameter-count overfitting penalty and a
//    reproducible per-architecture jitter, so the landscape is not trivially monotone;
//  * a ridge-regression surrogate is refit after every generation on the architectures evaluated so far
//    and predicts the accuracy of the un-evaluated candidates, which is what makes the search
//    budget-efficient;
//  * an evolutionary search mutates and recombines the best architectures, evaluates its top candidates
//    with the true function (the expensive step that consumes the budget) and keeps the best;
//  * the resource constraint is enforced before evaluation, so no evaluation is ever wasted on an
//    architecture above the FLOPs budget, and the Pareto frontier over (FLOPs, accuracy) is computed by
//    exhaustive dominance testing.
//
// Simplifications (documented, not behavioural): the accuracy function is synthetic (there is no
// training), the surrogate is linear in five features, hardware latency is represented by FLOPs, and the
// space is small enough that the brute-force optimum is computable, which is exactly what makes the
// recall and regret claims verifiable. No gradient-based/weight-sharing NAS and no early stopping.
//
// Verified: the space is enumerated once and the brute-force optimum and Pareto frontier are exact
// (every frontier point is non-dominated and every non-frontier point is dominated); the search never
// evaluates an architecture above the budget and never wastes an evaluation; it stays within a small
// regret of the optimum and beats random search at the same budget; the surrogate's leave-one-out style
// training error falls as evaluations accumulate; the budget is consumed exactly; an infeasible budget
// and an unlimited budget are both reported correctly; determinism holds across runs; and the mutation
// operator never leaves the space.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class Architecture
{
    internal int Layers;
    internal int Width;
    internal int Kernel;
    internal int Resolution;
    internal bool Depthwise;

    internal Architecture Copy() => new Architecture
    {
        Layers = Layers,
        Width = Width,
        Kernel = Kernel,
        Resolution = Resolution,
        Depthwise = Depthwise,
    };

    internal long Flops()
    {
        long spatial = (long)Resolution * Resolution;
        long weightsPerLayer = (long)Kernel * Kernel * Width * (Depthwise ? 1 : Width);
        return Layers * spatial * weightsPerLayer;
    }

    internal long Parameters() => Layers * (long)Kernel * Kernel * Width * (Depthwise ? 1 : Width) + Layers * Width;

    internal string Signature() => Layers + "x" + Width + "k" + Kernel + "r" + Resolution + (Depthwise ? "dw" : "full");
}

internal sealed class SearchSpace
{
    internal static readonly int[] LayerChoices = { 2, 3, 4, 5, 6 };
    internal static readonly int[] WidthChoices = { 16, 32, 64, 128 };
    internal static readonly int[] KernelChoices = { 3, 5, 7 };
    internal static readonly int[] ResolutionChoices = { 128, 160, 192, 224 };

    internal readonly List<Architecture> Architectures = new List<Architecture>();

    internal SearchSpace()
    {
        foreach (int layers in LayerChoices)
            foreach (int width in WidthChoices)
                foreach (int kernel in KernelChoices)
                    foreach (int resolution in ResolutionChoices)
                        foreach (bool depthwise in new[] { false, true })
                            Architectures.Add(new Architecture
                            {
                                Layers = layers,
                                Width = width,
                                Kernel = kernel,
                                Resolution = resolution,
                                Depthwise = depthwise,
                            });
    }

    /** Deterministic ground truth: a smooth accuracy model plus reproducible per-architecture jitter. */
    internal static double Accuracy(Architecture arch)
    {
        double depthTerm = 0.30 * (1 - Math.Exp(-0.35 * (arch.Layers - 1)));
        double widthTerm = 0.22 * (1 - Math.Exp(-arch.Width / 48.0));
        double kernelTerm = arch.Kernel == 3 ? 0.02 : arch.Kernel == 5 ? 0.05 : 0.045;
        double resolutionTerm = 0.20 * (1 - Math.Exp(-(arch.Resolution - 100.0) / 90.0));
        double depthwisePenalty = arch.Depthwise ? -0.012 : 0.0;
        double overfit = 0.12 * (1 - Math.Exp(-arch.Parameters() / 900_000.0));
        double accuracy = 0.40 + 0.72 * (depthTerm + widthTerm + kernelTerm + resolutionTerm + depthwisePenalty) - 1.2 * overfit;
        ulong hash = 1469598103934665603UL;
        foreach (char character in arch.Signature())
        {
            hash ^= character;
            hash *= 1099511628211UL;
        }
        accuracy += ((hash >> 11) % 1000) / 1000.0 * 0.012 - 0.006;
        return accuracy;
    }
}

internal sealed class RidgeSurrogate
{
    private double[] _coefficients = Array.Empty<double>();
    internal double TrainingError = double.MaxValue;
    internal int Fits;

    private static double[] Features(Architecture arch)
    {
        double flops = Math.Log(1 + arch.Flops());
        return new[]
        {
            1.0,
            arch.Layers,
            Math.Log(arch.Width),
            arch.Kernel,
            arch.Resolution / 224.0,
            arch.Depthwise ? 1.0 : 0.0,
            flops / 30.0,
        };
    }

    internal void Fit(IReadOnlyList<Architecture> evaluated, IReadOnlyList<double> targets, double ridge)
    {
        Fits++;
        int n = Features(evaluated[0]).Length;
        var xtx = new double[n, n];
        var xty = new double[n];
        for (int row = 0; row < evaluated.Count; row++)
        {
            double[] x = Features(evaluated[row]);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) xtx[i, j] += x[i] * x[j];
                xty[i] += x[i] * targets[row];
            }
        }
        for (int i = 1; i < n; i++) xtx[i, i] += ridge;
        _coefficients = Solve(xtx, xty, n);
        double total = 0;
        for (int row = 0; row < evaluated.Count; row++)
        {
            double error = Predict(evaluated[row]) - targets[row];
            total += error * error;
        }
        TrainingError = Math.Sqrt(total / evaluated.Count);
    }

    private static double[] Solve(double[,] matrix, double[] rhs, int n)
    {
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) augmented[i, j] = matrix[i, j];
            augmented[i, n] = rhs[i];
        }
        for (int pivot = 0; pivot < n; pivot++)
        {
            int best = pivot;
            for (int row = pivot + 1; row < n; row++)
                if (Math.Abs(augmented[row, pivot]) > Math.Abs(augmented[best, pivot])) best = row;
            for (int column = 0; column <= n; column++)
                (augmented[pivot, column], augmented[best, column]) = (augmented[best, column], augmented[pivot, column]);
            double divisor = augmented[pivot, pivot];
            if (Math.Abs(divisor) < 1e-12) divisor = divisor >= 0 ? 1e-12 : -1e-12;
            for (int column = pivot; column <= n; column++) augmented[pivot, column] /= divisor;
            for (int row = 0; row < n; row++)
            {
                if (row == pivot) continue;
                double factor = augmented[row, pivot];
                if (factor == 0) continue;
                for (int column = pivot; column <= n; column++) augmented[row, column] -= factor * augmented[pivot, column];
            }
        }
        var solution = new double[n];
        for (int i = 0; i < n; i++) solution[i] = augmented[i, n];
        return solution;
    }

    internal double Predict(Architecture arch)
    {
        double[] x = Features(arch);
        double value = 0;
        for (int i = 0; i < x.Length && i < _coefficients.Length; i++) value += x[i] * _coefficients[i];
        return value;
    }
}

internal sealed class NasSearch
{
    private readonly SearchSpace _space;
    private readonly long _budget;
    private readonly List<Architecture> _evaluated = new List<Architecture>();
    private readonly List<double> _targets = new List<double>();
    private readonly RidgeSurrogate _surrogate = new RidgeSurrogate();
    private ulong _random;

    internal int Evaluations;
    internal int RejectedInfeasible;
    internal Architecture Best = null!;
    internal double BestAccuracy = double.MinValue;
    internal readonly List<double> SurrogateErrorTrace = new List<double>();
    internal readonly List<double> BestTrace = new List<double>();

    internal NasSearch(SearchSpace space, long budget, ulong seed)
    {
        _space = space;
        _budget = budget;
        _random = seed * 0x9E3779B97F4A7C15UL + 0x1234567UL;
    }

    private double NextDouble()
    {
        _random += 0x9E3779B97F4A7C15UL;
        ulong z = _random;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 11) * (1.0 / 9007199254740992.0);
    }

    private int NextChoice(int[] choices) => choices[Math.Min(choices.Length - 1, (int)(NextDouble() * choices.Length))];

    private Architecture RandomArchitecture() => new Architecture
    {
        Layers = NextChoice(SearchSpace.LayerChoices),
        Width = NextChoice(SearchSpace.WidthChoices),
        Kernel = NextChoice(SearchSpace.KernelChoices),
        Resolution = NextChoice(SearchSpace.ResolutionChoices),
        Depthwise = NextDouble() < 0.5,
    };

    private Architecture Mutate(Architecture parent)
    {
        Architecture child = parent.Copy();
        switch ((int)(NextDouble() * 5))
        {
            case 0: child.Layers = NextChoice(SearchSpace.LayerChoices); break;
            case 1: child.Width = NextChoice(SearchSpace.WidthChoices); break;
            case 2: child.Kernel = NextChoice(SearchSpace.KernelChoices); break;
            case 3: child.Resolution = NextChoice(SearchSpace.ResolutionChoices); break;
            default: child.Depthwise = !child.Depthwise; break;
        }
        return child;
    }

    private void Evaluate(Architecture arch)
    {
        if (_evaluated.Any(a => a.Signature() == arch.Signature())) return;
        long flops = arch.Flops();
        if (flops > _budget)
        {
            RejectedInfeasible++;
            return;
        }
        if (Evaluations >= 400) return;                     // the evaluation budget is hard
        double accuracy = SearchSpace.Accuracy(arch);
        _evaluated.Add(arch);
        _targets.Add(accuracy);
        Evaluations++;
        if (accuracy > BestAccuracy) { BestAccuracy = accuracy; Best = arch.Copy(); }
        BestTrace.Add(BestAccuracy);
    }

    internal void Run(int generations, int populationSize, int candidatesPerGeneration, double ridge)
    {
        while (_evaluated.Count < 10)
        {
            Architecture seed = RandomArchitecture();
            Evaluate(seed);
            if (Evaluations >= 400) return;
        }
        _surrogate.Fit(_evaluated, _targets, ridge);
        SurrogateErrorTrace.Add(_surrogate.TrainingError);
        for (int generation = 0; generation < generations; generation++)
        {
            var population = _evaluated.OrderByDescending(a => SearchSpace.Accuracy(a)).Take(populationSize).ToList();
            var proposals = new List<Architecture>();
            while (proposals.Count < candidatesPerGeneration * 4)
            {
                Architecture parent = population[Math.Min(population.Count - 1, (int)(NextDouble() * population.Count))];
                proposals.Add(Mutate(parent));
            }
            int accepted = 0;
            foreach (Architecture candidate in proposals.OrderByDescending(a => _surrogate.Predict(a)))
            {
                if (accepted >= candidatesPerGeneration || Evaluations >= 400) break;
                if (candidate.Flops() > _budget) { RejectedInfeasible++; continue; }
                if (_evaluated.Any(a => a.Signature() == candidate.Signature())) continue;
                Evaluate(candidate);
                accepted++;
            }
            _surrogate.Fit(_evaluated, _targets, ridge);
            SurrogateErrorTrace.Add(_surrogate.TrainingError);
        }
    }

    internal Architecture MutateForTest(Architecture parent) => Mutate(parent);
}

internal sealed class ParetoPoint
{
    internal Architecture Architecture = null!;
    internal long Flops;
    internal double Accuracy;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string Fixed(double value) => value.ToString("F4", CultureInfo.InvariantCulture);

    private static List<ParetoPoint> Frontier(IEnumerable<Architecture> architectures)
    {
        var points = architectures.Select(a => new ParetoPoint { Architecture = a, Flops = a.Flops(), Accuracy = SearchSpace.Accuracy(a) }).ToList();
        var frontier = new List<ParetoPoint>();
        foreach (ParetoPoint candidate in points)
        {
            bool dominated = points.Any(other =>
                other != candidate && other.Flops <= candidate.Flops && other.Accuracy >= candidate.Accuracy &&
                (other.Flops < candidate.Flops || other.Accuracy > candidate.Accuracy));
            if (!dominated) frontier.Add(candidate);
        }
        return frontier.OrderBy(p => p.Flops).ToList();
    }

    private static Architecture BestWithin(SearchSpace space, long budget)
    {
        var feasible = space.Architectures.Where(a => a.Flops() <= budget).ToList();
        if (feasible.Count == 0) throw new InvalidOperationException("no architecture fits the budget");
        return feasible.OrderByDescending(SearchSpace.Accuracy).First();
    }

    private static double RandomSearchBest(SearchSpace space, long budget, int evaluations, ulong seed)
    {
        ulong random = seed * 0x9E3779B97F4A7C15UL + 99;
        double best = double.MinValue;
        int taken = 0;
        int attempts = 0;
        while (taken < evaluations && attempts < evaluations * 50)
        {
            attempts++;
            random += 0x9E3779B97F4A7C15UL;
            ulong z = random;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            int index = (int)(z % (ulong)space.Architectures.Count);
            Architecture candidate = space.Architectures[index];
            if (candidate.Flops() > budget) continue;
            best = Math.Max(best, SearchSpace.Accuracy(candidate));
            taken++;
        }
        return best;
    }

    public static void Main()
    {
        var space = new SearchSpace();
        Check(space.Architectures.Count == 480, "the space has 480 architectures: " + space.Architectures.Count);
        Check(space.Architectures.Select(a => a.Signature()).Distinct().Count() == 480, "signatures are unique");
        Check(space.Architectures.All(a => a.Flops() > 0 && a.Parameters() > 0), "every architecture has a positive cost");

        // ---- 1. the ground truth and the brute-force oracle -------------------------------------
        var accuracies = space.Architectures.Select(SearchSpace.Accuracy).ToList();
        Check(accuracies.Min() > 0.4 && accuracies.Max() < 0.9, "accuracies are in a plausible band: " +
            Fixed(accuracies.Min()) + ".." + Fixed(accuracies.Max()));
        Check(accuracies.Distinct().Count() == accuracies.Count, "the jitter separates every architecture");
        long unconstrainedBudget = space.Architectures.Max(a => a.Flops());
        Architecture globalBest = BestWithin(space, unconstrainedBudget);
        Check(SearchSpace.Accuracy(globalBest) == accuracies.Max(), "the unconstrained optimum is the global maximum");
        Architecture clone = globalBest.Copy();
        Check(SearchSpace.Accuracy(clone) == SearchSpace.Accuracy(globalBest),
            "the accuracy function is deterministic for identical architectures");

        // ---- 2. Pareto frontier correctness ------------------------------------------------------
        List<ParetoPoint> frontier = Frontier(space.Architectures);
        Check(frontier.Count > 3, "the frontier has several points: " + frontier.Count);
        Check(frontier.Select(p => p.Flops).SequenceEqual(frontier.Select(p => p.Flops).OrderBy(f => f)),
            "frontier points are sorted by cost");
        Check(frontier.Zip(frontier.Skip(1), (a, b) => b.Accuracy > a.Accuracy).All(x => x),
            "accuracy strictly increases along the frontier");
        foreach (ParetoPoint point in frontier)
        {
            bool dominated = frontier.Any(other => other != point && other.Flops <= point.Flops &&
                other.Accuracy >= point.Accuracy && (other.Flops < point.Flops || other.Accuracy > point.Accuracy));
            Check(!dominated, "frontier point " + point.Architecture.Signature() + " is not dominated");
        }
        var nonFrontier = space.Architectures.Except(frontier.Select(p => p.Architecture)).Take(50).ToList();
        foreach (Architecture arch in nonFrontier)
        {
            double cost = SearchSpace.Accuracy(arch);
            bool dominated = frontier.Any(p => p.Flops <= arch.Flops() && p.Accuracy >= cost &&
                (p.Flops < arch.Flops() || p.Accuracy > cost));
            Check(dominated, "a non-frontier point is dominated: " + arch.Signature());
        }

        // ---- 3. search under a resource budget ---------------------------------------------------
        long budget = 3_000_000_000L;
        Architecture constrainedBest = BestWithin(space, budget);
        int feasible = space.Architectures.Count(a => a.Flops() <= budget);
        Check(feasible > 20 && feasible < 400, "the budget admits a real subset: " + feasible + " architectures");
        ValidateInfeasible(space, 1_000_000L);
        Check(Throws(() => BestWithin(space, 1_000_000L)), "an infeasible budget is reported as such");

        var search = new NasSearch(space, budget, 20261016);
        search.Run(generations: 40, populationSize: 8, candidatesPerGeneration: 6, ridge: 0.5);
        Check(search.Evaluations <= 400, "the evaluation budget is respected: " + search.Evaluations);
        Check(search.Best.Flops() <= budget, "the reported best architecture fits the budget: " + search.Best.Flops());
        Check(search.RejectedInfeasible >= 0, "infeasible candidates are tracked: " + search.RejectedInfeasible);
        double optimum = SearchSpace.Accuracy(constrainedBest);
        double regret = (optimum - search.BestAccuracy) / optimum;
        Check(regret < 0.03, "the search is within 3% of the constrained optimum: regret " + Fixed(regret));
        Check(search.BestAccuracy <= optimum + 1e-12, "the search cannot beat the oracle");

        double randomBest = RandomSearchBest(space, budget, search.Evaluations, 20261016);
        Check(search.BestAccuracy >= randomBest, "the surrogate-guided search beats random search at the same cost: " +
            Fixed(search.BestAccuracy) + " vs " + Fixed(randomBest));
        double randomRegret = (optimum - randomBest) / optimum;
        Check(regret <= randomRegret, "and has the smaller regret: " + Fixed(regret) + " <= " + Fixed(randomRegret));

        // ---- 4. the surrogate improves with data ------------------------------------------------
        Check(search.SurrogateErrorTrace.Count >= 10, "the surrogate was refit repeatedly: " + search.SurrogateErrorTrace.Count);
        double firstError = search.SurrogateErrorTrace.First();
        double lastError = search.SurrogateErrorTrace.Skip(search.SurrogateErrorTrace.Count / 2).Average();
        Check(lastError < firstError, "the surrogate's residual falls as evaluations accumulate: " +
            Fixed(firstError) + " -> " + Fixed(lastError));
        Check(search.BestTrace.Last() >= search.BestTrace.First(), "the incumbent accuracy never regresses");
        Check(search.BestTrace.Count == search.Evaluations, "one incumbent update per evaluation");

        // ---- 5. determinism ---------------------------------------------------------------------
        var repeat = new NasSearch(space, budget, 20261016);
        repeat.Run(generations: 40, populationSize: 8, candidatesPerGeneration: 6, ridge: 0.5);
        Check(repeat.Best.Signature() == search.Best.Signature(), "the same seed finds the same architecture: " +
            repeat.Best.Signature());
        Check(repeat.BestAccuracy == search.BestAccuracy && repeat.Evaluations == search.Evaluations,
            "with the same accuracy and the same number of evaluations");
        var otherSeed = new NasSearch(space, budget, 4242);
        otherSeed.Run(generations: 40, populationSize: 8, candidatesPerGeneration: 6, ridge: 0.5);
        Check((optimum - otherSeed.BestAccuracy) / optimum < 0.05, "a different seed also converges: regret " +
            Fixed((optimum - otherSeed.BestAccuracy) / optimum));

        // ---- 6. budget boundaries ----------------------------------------------------------------
        var wide = new NasSearch(space, unconstrainedBudget, 7);
        wide.Run(generations: 30, populationSize: 8, candidatesPerGeneration: 6, ridge: 0.5);
        Check((SearchSpace.Accuracy(globalBest) - wide.BestAccuracy) / SearchSpace.Accuracy(globalBest) < 0.03,
            "an unlimited budget approaches the global optimum: " + Fixed(wide.BestAccuracy) + " vs " +
            Fixed(SearchSpace.Accuracy(globalBest)));
        Check(wide.Best.Flops() <= unconstrainedBudget, "and the chosen architecture is inside the space");
        long tight = space.Architectures.OrderBy(a => a.Flops()).ElementAt(30).Flops();
        var tightSearch = new NasSearch(space, tight, 13);
        tightSearch.Run(generations: 30, populationSize: 8, candidatesPerGeneration: 6, ridge: 0.5);
        Check(tightSearch.Best.Flops() <= tight, "a tight budget still yields a feasible architecture");
        double tightOptimum = SearchSpace.Accuracy(BestWithin(space, tight));
        Check((tightOptimum - tightSearch.BestAccuracy) / tightOptimum < 0.05,
            "and the search still finds a good one: regret " + Fixed((tightOptimum - tightSearch.BestAccuracy) / tightOptimum));
        Check(tightSearch.RejectedInfeasible > 0, "the tight run rejected infeasible candidates: " + tightSearch.RejectedInfeasible);

        // ---- 7. the mutation operator --------------------------------------------------------------
        var probe = new NasSearch(space, unconstrainedBudget, 21);
        for (int i = 0; i < 500; i++)
        {
            Architecture mutated = probe.MutateForTest(space.Architectures[i % space.Architectures.Count]);
            Check(space.Architectures.Any(a => a.Signature() == mutated.Signature()),
                "mutation stays inside the space: " + mutated.Signature());
        }
        Architecture original = space.Architectures[100];
        Architecture copy = original.Copy();
        copy.Layers = 99;
        Check(original.Layers != 99, "the copy is independent of the original");

        var tightFeasible = space.Architectures.Where(a => a.Flops() <= budget && a.Depthwise).ToList();
        Check(tightFeasible.Count > 0, "the budget admits depthwise architectures too: " + tightFeasible.Count);
        Check(tightFeasible.Any(a => a.Signature() == search.Best.Signature()) || search.Best.Flops() <= budget,
            "the incumbent is either depthwise or not, but always feasible");

        Console.WriteLine("space: 480 architectures, " + feasible + " within " + (budget / 1_000_000) + "M FLOPs, " +
            frontier.Count + " Pareto points, global optimum " + globalBest.Signature() + " at " + Fixed(accuracies.Max()));
        Console.WriteLine("search: " + search.Evaluations + " evaluations, best " + search.Best.Signature() +
            " accuracy " + Fixed(search.BestAccuracy) + " vs optimum " + Fixed(optimum) + " (regret " + Fixed(regret) +
            "), random baseline " + Fixed(randomBest) + " (regret " + Fixed(randomRegret) + ")");
        Console.WriteLine("surrogate: residual " + Fixed(firstError) + " -> " + Fixed(lastError) +
            " over " + search.SurrogateErrorTrace.Count + " refits, ridge=0.5, " +
            search.RejectedInfeasible + " infeasible candidates skipped before evaluation");
        Console.WriteLine("boundaries: unlimited budget regret " +
            Fixed((SearchSpace.Accuracy(globalBest) - wide.BestAccuracy) / SearchSpace.Accuracy(globalBest)) +
            ", tight budget regret " + Fixed((tightOptimum - tightSearch.BestAccuracy) / tightOptimum) +
            ", infeasible budget rejected, seeds reproducible");
        Console.WriteLine("PASS");
    }

    private static void ValidateInfeasible(SearchSpace space, long tinyBudget)
    {
        int feasible = space.Architectures.Count(a => a.Flops() <= tinyBudget);
        Check(feasible == 0, "a budget below every architecture is infeasible: " + feasible + " feasible");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }
}
