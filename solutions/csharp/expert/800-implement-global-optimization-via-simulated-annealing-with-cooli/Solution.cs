// Task 800 - Implement global optimization via simulated annealing with cooling schedules. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Simulated annealing as a Metropolis chain whose temperature follows a schedule. From the current point x with
// energy E, a random neighbour x' is drawn; it is accepted unconditionally when E' <= E and with probability
// exp(-(E' - E)/T) otherwise. Three schedules are implemented and compared
//   geometric:    T_k = T0 * alpha^k
//   linear:       T_k = max(0, T0 - k * step)
//   logarithmic:  T_k = T0 / ln(2 + k)          (the schedule that guarantees convergence in the limit)
// and the best-so-far energy is tracked separately, so the returned solution is never worse than the trajectory
// end. Acceptance is decided by a hand-written splitmix64 uniform, so every run is reproducible.
//
// Invariants asserted: the best-so-far energy never increases, the geometric schedule reproduces its closed form
// T0 * alpha^k exactly, the linear schedule clamps at zero and stops there, and the accepted tour of the TSP
// instance is always a valid permutation (checked after every accepted move).
//
// Independent oracles:
//   * the empirical uphill-acceptance rate at a fixed temperature is compared with the Boltzmann factor
//     exp(-delta/T) over many draws;
//   * the 1D multimodal objective f(x) = 10 + x^2 - 10 cos(2 pi x) has its global minimum at x = 0 with f = 0,
//     confirmed by a fine grid scan that shares no code with the sampler;
//   * the 8-city TSP instance is solved exactly by enumerating all permutations, so the annealing result can be
//     compared with the true optimum rather than with a heuristic bound.
//
// Documented simplification: small problems (a 1D landscape and an 8-city tour), a fixed iteration budget per
// schedule and no reheating, parallel tempering or adaptive schedules; the Metropolis rule, the three cooling
// schedules and the best-so-far bookkeeping are the real algorithm. Deterministic randomness only.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum Schedule { Geometric, Linear, Logarithmic }

internal sealed class Annealer
{
    private readonly Func<double, double> _proposal;     // current temperature -> proposal scale
    internal ulong State;

    internal Annealer(ulong seed, Func<double, double> proposal)
    {
        State = seed;
        _proposal = proposal;
    }

    internal static double Temperature(Schedule schedule, double initial, double parameter, int step) => schedule switch
    {
        Schedule.Geometric => initial * Math.Pow(parameter, step),
        Schedule.Linear => Math.Max(0.0, initial - step * parameter),
        _ => initial / Math.Log(2.0 + step)
    };

    /**
     * Runs the chain on a double-valued objective. Returns the best value found and the best point;
     * `bestTrailNonIncreasing` reports whether the best-so-far sequence was monotone.
     */
    internal double Optimize(Func<double, double> energy, double start, Schedule schedule, double initialTemperature,
        double parameter, int iterations, out double bestPoint, out bool bestTrailNonIncreasing, out double finalTemperature)
    {
        if (initialTemperature <= 0.0) throw new ArgumentException("initial temperature must be positive", nameof(initialTemperature));
        if (iterations < 0) throw new ArgumentException("iteration count must not be negative", nameof(iterations));
        if (schedule == Schedule.Geometric && (parameter <= 0.0 || parameter > 1.0))
        {
            throw new ArgumentException("geometric cooling factor must lie in (0, 1]", nameof(parameter));
        }

        double current = start;
        double currentEnergy = energy(current);
        double best = currentEnergy;
        bestPoint = current;
        bestTrailNonIncreasing = true;
        double temperature = initialTemperature;
        for (int step = 0; step < iterations; step++)
        {
            temperature = Temperature(schedule, initialTemperature, parameter, step);
            double scale = _proposal(temperature);
            double candidate = current + scale * (2.0 * SplitMix64.NextDouble(ref State) - 1.0);
            double candidateEnergy = energy(candidate);
            double delta = candidateEnergy - currentEnergy;
            if (delta <= 0.0 || SplitMix64.NextDouble(ref State) < Math.Exp(-delta / temperature))
            {
                current = candidate;
                currentEnergy = candidateEnergy;
            }
            double bestBefore = best;
            if (currentEnergy < best - 1e-15)
            {
                best = currentEnergy;
                bestPoint = current;
            }
            if (best > bestBefore + 1e-15) bestTrailNonIncreasing = false;
        }
        finalTemperature = Temperature(schedule, initialTemperature, parameter, Math.Max(0, iterations - 1));
        return best;
    }

    /** Empirical acceptance rate of uphill moves of a fixed size at a fixed temperature. */
    internal static double AcceptanceRate(double delta, double temperature, int trials, ulong seed)
    {
        var rng = seed;
        int accepted = 0;
        for (int i = 0; i < trials; i++)
        {
            if (SplitMix64.NextDouble(ref rng) < Math.Exp(-delta / temperature)) accepted++;
        }
        return accepted / (double)trials;
    }
}

public static class Solution
{
    /** Rastrigin-style multimodal landscape: global minimum 0 at x = 0. */
    private static double Landscape(double x) => 10.0 + x * x - 10.0 * Math.Cos(2.0 * Math.PI * x);

    private static readonly double[][] Cities =
    {
        new[] { 0.0, 0.0 }, new[] { 4.0, 0.0 }, new[] { 5.0, 3.0 }, new[] { 2.0, 5.0 },
        new[] { -1.0, 4.0 }, new[] { -3.0, 1.0 }, new[] { 1.0, -2.0 }, new[] { 3.5, -2.5 }
    };

    private static double TourLength(IReadOnlyList<int> tour, IReadOnlyList<double[]> cities)
    {
        double total = 0.0;
        for (int i = 0; i < tour.Count; i++)
        {
            double[] a = cities[tour[i]];
            double[] b = cities[tour[(i + 1) % tour.Count]];
            total += Math.Sqrt(Math.Pow(a[0] - b[0], 2) + Math.Pow(a[1] - b[1], 2));
        }
        return total;
    }

    private static bool IsPermutation(IReadOnlyList<int> tour, int cityCount)
    {
        var seen = new bool[cityCount];
        foreach (int city in tour)
        {
            if (city < 0 || city >= cityCount || seen[city]) return false;
            seen[city] = true;
        }
        return tour.Count == cityCount;
    }

    /** Exact optimum by enumerating every permutation (8! = 40320 tours). */
    private static double OptimalTour(IReadOnlyList<double[]> cities)
    {
        var order = new int[cities.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        double best = double.MaxValue;
        Permute(order, 0, cities, ref best);
        return best;
    }

    private static void Permute(int[] order, int at, IReadOnlyList<double[]> cities, ref double best)
    {
        if (at == order.Length)
        {
            double total = 0.0;
            for (int i = 0; i < order.Length; i++)
            {
                double[] a = cities[order[i]];
                double[] b = cities[order[(i + 1) % order.Length]];
                total += Math.Sqrt(Math.Pow(a[0] - b[0], 2) + Math.Pow(a[1] - b[1], 2));
            }
            best = Math.Min(best, total);
            return;
        }
        for (int i = at; i < order.Length; i++)
        {
            (order[at], order[i]) = (order[i], order[at]);
            Permute(order, at + 1, cities, ref best);
            (order[at], order[i]) = (order[i], order[at]);
        }
    }

    /** Anneals a tour with 2-opt proposals; tracks the best tour and its length. */
    private static double AnnealTour(ulong seed, Schedule schedule, double initialTemperature, double parameter,
        int iterations, out int[] bestTour, out bool alwaysPermutation)
    {
        var rng = seed;
        var tour = new int[Cities.Length];
        for (int i = 0; i < tour.Length; i++) tour[i] = i;
        double current = TourLength(tour, Cities);
        double best = current;
        bestTour = (int[])tour.Clone();
        alwaysPermutation = true;
        for (int step = 0; step < iterations; step++)
        {
            double temperature = Annealer.Temperature(schedule, initialTemperature, parameter, step);
            int i = SplitMix64.NextInt(ref rng, tour.Length);
            int j = SplitMix64.NextInt(ref rng, tour.Length);
            if (i == j) continue;
            int lo = Math.Min(i, j), hi = Math.Max(i, j);
            Array.Reverse(tour, lo, hi - lo + 1);
            double candidate = TourLength(tour, Cities);
            double delta = candidate - current;
            if (delta <= 0.0 || SplitMix64.NextDouble(ref rng) < Math.Exp(-delta / temperature))
            {
                current = candidate;
                if (current < best)
                {
                    best = current;
                    bestTour = (int[])tour.Clone();
                }
            }
            else
            {
                Array.Reverse(tour, lo, hi - lo + 1);              // reject: undo the 2-opt move
            }
            if (!IsPermutation(tour, Cities.Length)) alwaysPermutation = false;
        }
        return best;
    }

    public static void Main()
    {
        // 1-2. Boltzmann acceptance law: the measured uphill acceptance rate must match exp(-delta/T).
        double worstAcceptance = 0.0;
        foreach ((double delta, double temperature) in new[] { (0.5, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, 0.25) })
        {
            double measured = Annealer.AcceptanceRate(delta, temperature, 20000, 0xA1UL);
            double exact = Math.Exp(-delta / temperature);
            worstAcceptance = Math.Max(worstAcceptance, Math.Abs(measured - exact));
        }
        Console.WriteLine("  acceptance law: max deviation=" + worstAcceptance.ToString("F4") + " (exp(-delta/T))");
        Check(worstAcceptance < 0.02, "uphill acceptance frequency matches the Boltzmann factor");
        Check(Annealer.AcceptanceRate(1.0, 1e-6, 100, 0x1UL) < 0.01, "at zero temperature uphill moves are refused");

        // 3. Grid-scan oracle for the 1D landscape.
        double gridBest = double.MaxValue;
        double gridArgmin = 0.0;
        for (int i = 0; i <= 200000; i++)
        {
            double x = -5.12 + 10.24 * i / 200000.0;
            double value = Landscape(x);
            if (value < gridBest)
            {
                gridBest = value;
                gridArgmin = x;
            }
        }
        Console.WriteLine("  landscape oracle: grid minimum=" + gridBest.ToString("E3") + " at x=" + gridArgmin.ToString("F6")
            + " (closed form 0 at x = 0)");
        Check(gridBest < 1e-4 && Math.Abs(gridArgmin) < 1e-3, "the grid scan confirms the global minimum at x = 0");
        Check(Math.Abs(Landscape(0.0)) < 1e-15 && Landscape(0.5) > 20.0 && Landscape(1.0) > 0.9,
            "the landscape has a global well at 0, a barrier at 0.5 and local wells at +-1");

        // 4-6. Geometric schedule: closed form, monotone best-so-far, and the global minimum is located.
        var geometric = new Annealer(0x51A1UL, temperature => Math.Min(3.0, Math.Max(0.05, 0.5 * temperature)));
        double geometricBest = geometric.Optimize(Landscape, 4.0, Schedule.Geometric, 20.0, 0.9995, 20000,
            out double geometricPoint, out bool monotone, out double finalTemperature);
        double expectedFinal = 20.0 * Math.Pow(0.9995, 19999);
        Console.WriteLine("  geometric schedule: best=" + geometricBest.ToString("E3") + " at x="
            + geometricPoint.ToString("F6") + " final T=" + finalTemperature.ToString("E6")
            + " (closed form " + expectedFinal.ToString("E6") + ")");
        Check(Math.Abs(finalTemperature - expectedFinal) < 1e-15, "geometric cooling matches T0 * alpha^k exactly");
        Check(monotone, "the best-so-far energy never increases");
        Check(geometricBest < 0.5, "geometric annealing locates the global well");
        Check(Math.Abs(geometricPoint) < 0.2, "the best point sits at the global minimum");

        // 7. The other two schedules reach the same basin and cool differently.
        var linear = new Annealer(0x51A2UL, temperature => Math.Min(3.0, Math.Max(0.05, 0.5 * temperature)));
        double linearBest = linear.Optimize(Landscape, 4.0, Schedule.Linear, 20.0, 0.001, 20001, out double linearPoint,
            out bool linearMonotone, out double linearFinal);
        var logarithmic = new Annealer(0x51A3UL, temperature => Math.Min(3.0, Math.Max(0.05, 0.5 * temperature)));
        double logarithmicBest = logarithmic.Optimize(Landscape, 4.0, Schedule.Logarithmic, 20.0, 0.0, 20000,
            out double logarithmicPoint, out bool logarithmicMonotone, out double logarithmicFinal);
        Console.WriteLine("  linear: best=" + linearBest.ToString("E3") + " final T=" + linearFinal.ToString("E3")
            + " | logarithmic: best=" + logarithmicBest.ToString("E3") + " final T=" + logarithmicFinal.ToString("E3"));
        Check(linearFinal == 0.0, "the linear schedule clamps at zero once the budget is spent");
        Check(logarithmicFinal > linearFinal, "the logarithmic schedule is still warm at the end of the same budget");
        Check(linearMonotone && logarithmicMonotone, "best-so-far stays non-increasing for every schedule");
        Check(linearBest < 1.0 && logarithmicBest < 1.0, "linear and logarithmic schedules also find a deep well");

        // 8-9. Zero-temperature (greedy) descent gets stuck; the annealed run is strictly better.
        var frozen = new Annealer(0x7777UL, _ => 0.5);
        double frozenBest = frozen.Optimize(Landscape, 4.0, Schedule.Linear, 1e-9, 1e-9, 20000, out double frozenPoint,
            out _, out _);
        Console.WriteLine("  greedy (T -> 0): best=" + frozenBest.ToString("E3") + " at x=" + frozenPoint.ToString("F6"));
        Check(frozenBest > geometricBest, "a frozen chain is worse than the annealed one");
        Check(Math.Abs(frozenPoint) > 0.2, "the frozen chain stops in a local basin away from the optimum");

        // 10-12. Travelling salesman against the exact optimum.
        double optimal = OptimalTour(Cities);
        double annealedTour = AnnealTour(0x7A57UL, Schedule.Geometric, 3.0, 0.9995, 40000, out int[] bestTour,
            out bool alwaysPermutation);
        Console.WriteLine("  TSP (8 cities): annealed=" + annealedTour.ToString("F6") + " exact=" + optimal.ToString("F6")
            + " gap=" + ((annealedTour - optimal) / optimal * 100.0).ToString("F4") + "%");
        Check(alwaysPermutation, "every visited tour is a valid permutation of the cities");
        Check(IsPermutation(bestTour, Cities.Length), "the best tour returned is a valid permutation");
        Check(Math.Abs(TourLength(bestTour, Cities) - annealedTour) < 1e-9, "the reported length matches the returned tour");
        Check(annealedTour <= optimal * 1.002, "annealing reaches the exact optimum within 0.2%");

        // 13-15. Boundary cases and rejected parameters.
        var trivial = new Annealer(0x1UL, _ => 0.1);
        trivial.Optimize(Landscape, 0.0, Schedule.Geometric, 1.0, 0.9, 0, out double zeroPoint, out bool zeroMonotone, out _);
        Check(zeroPoint == 0.0 && zeroMonotone, "zero iterations return the start point unchanged");
        var pairCities = new[] { new[] { 0.0, 0.0 }, new[] { 3.0, 4.0 } };
        Check(Math.Abs(TourLength(new[] { 0, 1 }, pairCities) - 10.0) < 1e-12 && IsPermutation(new[] { 0, 1 }, 2),
            "a two-city tour is twice the inter-city distance and a valid permutation");
        Check(!IsPermutation(new[] { 0, 0 }, 2) && !IsPermutation(new[] { 0 }, 2),
            "repeated or missing cities fail the permutation test");
        Check(Throws(() => new Annealer(1UL, _ => 1.0).Optimize(Landscape, 0.0, Schedule.Geometric, 0.0, 0.9, 10, out _, out _, out _)),
            "non-positive initial temperature rejected");
        Check(Throws(() => new Annealer(1UL, _ => 1.0).Optimize(Landscape, 0.0, Schedule.Geometric, 1.0, 1.5, 10, out _, out _, out _)),
            "geometric factor above one rejected");
        Check(Throws(() => new Annealer(1UL, _ => 1.0).Optimize(Landscape, 0.0, Schedule.Linear, 1.0, 0.1, -1, out _, out _, out _)),
            "negative iteration count rejected");

        // 16. Determinism.
        var repeat = new Annealer(0x51A1UL, temperature => Math.Min(3.0, Math.Max(0.05, 0.5 * temperature)));
        double repeatBest = repeat.Optimize(Landscape, 4.0, Schedule.Geometric, 20.0, 0.9995, 20000, out _, out _, out _);
        Check(repeatBest == geometricBest, "identical seeds reproduce the annealed optimum bit for bit");

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
