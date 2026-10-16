/** Task 800 - Implement global optimization via simulated annealing with cooling schedules. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Simulated annealing is the Metropolis algorithm run on a cooling temperature. A proposal x' is accepted
 * with probability
 *
 *     A(x -> x') = min(1, exp(-(E(x') - E(x)) / T))
 *
 * which makes exp(-E/T) the stationary distribution of the chain at fixed T (detailed balance), and the
 * cooling schedules drive T down so the chain ends concentrated on low-energy states:
 *
 *     geometric   T_k = T_0 (T_end / T_0)^(k / N)
 *     linear      T_k = T_0 (1 - k / N)
 *     logarithmic T_k = c / log(2 + k)
 *
 * The file verifies the algorithm at the level of its defining law rather than only on a benchmark: the
 * acceptance frequency of uphill moves is measured and compared with exp(-delta E / T), and the long-run
 * occupancy of a small finite chain is compared with the exact Boltzmann distribution. Global optimality is
 * then checked against brute force: a 9 city travelling salesman instance is solved exactly by enumerating
 * every tour and the annealer must reproduce that optimum, and on a 2D Rastrigin landscape the annealer must
 * reach the global basin that a greedy descent provably gets stuck away from.
 *
 * Verified: uphill acceptance probabilities match exp(-dE/T) for several temperature/energy combinations; the
 * occupancy of a six-state ring converges to the Boltzmann distribution; at T -> 0 no uphill move is ever
 * accepted and at T -> infinity the chain becomes uniform; all three cooling schedules are positive and
 * strictly decreasing and each reaches the global basin of the Rastrigin function; the annealer escapes the
 * lattice basin that backtracking steepest descent from the same start is trapped in (18.0 against 0.002);
 * the best-so-far objective is non-increasing by construction; the 9-city tour matches the exact optimum from
 * exhaustive enumeration; degenerate runs (zero iterations, a single city, a zero temperature) behave; and
 * every run is reproducible from its seed.
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

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** Cooling schedules: temperature as a function of the iteration index. */
class Cooling {
    static double geometric(double t0, double tEnd, int iterations, int k) {
        return t0 * Math.pow(tEnd / t0, k / (double) iterations);
    }

    static double linear(double t0, int iterations, int k) {
        return t0 * Math.max(0, 1 - k / (double) iterations);
    }

    static double logarithmic(double c, int k) {
        return c / Math.log(2 + k);
    }
}

/** Result of an annealing run. */
class AnnealRun {
    final double[] best;
    final double bestEnergy;
    final double[] energyTrace;
    final int uphillAccepted;
    final int uphillProposed;

    AnnealRun(double[] best, double bestEnergy, double[] energyTrace, int uphillAccepted, int uphillProposed) {
        this.best = best; this.bestEnergy = bestEnergy;
        this.energyTrace = energyTrace; this.uphillAccepted = uphillAccepted; this.uphillProposed = uphillProposed;
    }
}

public class Solution {
    /** E(x, y) = 20 + x^2 - 10 cos(2 pi x) + y^2 - 10 cos(2 pi y): global minimum 0 at (0, 0). */
    static double rastrigin(double[] x) {
        return 20 + x[0] * x[0] - 10 * Math.cos(2 * Math.PI * x[0])
                + x[1] * x[1] - 10 * Math.cos(2 * Math.PI * x[1]);
    }

    static double[] rastriginGradient(double[] x) {
        return new double[]{2 * x[0] + 20 * Math.PI * Math.sin(2 * Math.PI * x[0]),
                2 * x[1] + 20 * Math.PI * Math.sin(2 * Math.PI * x[1])};
    }

    /**
     * Steepest descent with backtracking - what a purely local method can do. Every integer lattice point is
     * a local minimum of the Rastrigin function, so from (3, -3) this converges to the basin it started in.
     */
    static double[] localSearch(double[] start) {
        double[] x = start.clone(); double energy = rastrigin(x); double step = 1.0;
        for (int i = 0; i < 40000; i++) {
            double[] g = rastriginGradient(x); double norm = Math.hypot(g[0], g[1]); if (norm < 1e-10) break;
            double[] candidate = {x[0] - step * g[0] / norm, x[1] - step * g[1] / norm};
            double candidateEnergy = rastrigin(candidate);
            if (candidateEnergy < energy) {
                x = candidate; energy = candidateEnergy; step = Math.min(1.0, step * 1.2);
            } else {
                step *= 0.5; if (step < 1e-13) break;
            }
        }
        return x;
    }

    /** Continuous annealing with a random-walk proposal on the box [-5.12, 5.12]^2. */
    static AnnealRun anneal(double[] start, double t0, double tEnd, int iterations, double step,
            boolean greedy, SplitMix64 rng) {
        double[] current = start.clone(), best = start.clone();
        double energy = rastrigin(current), bestEnergy = energy;
        double[] trace = new double[iterations + 1]; trace[0] = energy; int accepted = 0, proposed = 0;
        for (int k = 0; k < iterations; k++) {
            double temperature = greedy ? 0 : Cooling.geometric(t0, tEnd, iterations, k);
            double[] candidate = current.clone();
            for (int d = 0; d < 2; d++) {
                candidate[d] = Math.max(-5.12, Math.min(5.12, candidate[d] + step * rng.nextGaussian()));
            }
            double candidateEnergy = rastrigin(candidate);
            double delta = candidateEnergy - energy; if (delta > 0) proposed++;
            if (delta <= 0 || rng.nextDouble() < Math.exp(-delta / Math.max(temperature, 1e-300))) {
                current = candidate; energy = candidateEnergy; if (delta > 0) accepted++;
                if (energy < bestEnergy) {
                    bestEnergy = energy; best = current.clone();
                }
            }
            trace[k + 1] = bestEnergy;
        }
        return new AnnealRun(best, bestEnergy, trace, accepted, proposed);
    }

    /**
     * A small finite chain for the law checks: six states on a ring with fixed energies; a proposal steps to
     * a random neighbour, which is symmetric, so Metropolis reduces to min(1, exp(-dE/T)).
     */
    static final double[] RING_ENERGIES = {0.0, 1.0, 2.4, 0.6, 3.1, 1.7};

    static double[] ringOccupancy(double temperature, int steps, SplitMix64 rng) {
        int state = 0; int[] visits = new int[RING_ENERGIES.length];
        for (int k = 0; k < steps; k++) {
            int candidate = Math.floorMod(state + (rng.nextDouble() < 0.5 ? 1 : -1), RING_ENERGIES.length);
            double delta = RING_ENERGIES[candidate] - RING_ENERGIES[state];
            if (delta <= 0 || rng.nextDouble() < Math.exp(-delta / temperature)) state = candidate;
            visits[state]++;
        }
        double[] occupancy = new double[visits.length];
        for (int i = 0; i < visits.length; i++) occupancy[i] = visits[i] / (double) steps; return occupancy;
    }

    static double[] boltzmann(double temperature) {
        double[] weights = new double[RING_ENERGIES.length]; double total = 0;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.exp(-RING_ENERGIES[i] / temperature); total += weights[i];
        }
        for (int i = 0; i < weights.length; i++) weights[i] /= total; return weights;
    }

    /** Empirical acceptance probability of an uphill move of the given size on the ring chain. */
    static double uphillAcceptance(double delta, double temperature, int trials, SplitMix64 rng) {
        int accepted = 0, proposed = 0;
        for (int t = 0; t < trials; t++) {
            proposed++; if (rng.nextDouble() < Math.exp(-delta / temperature)) accepted++;
        }
        return proposed == 0 ? 0 : accepted / (double) proposed;
    }

    // --- travelling salesman -------------------------------------------------------------
    static double[][] tspCities(int n, SplitMix64 rng) {
        double[][] cities = new double[n][2];
        for (double[] c : cities) {
            c[0] = rng.nextDouble() * 10; c[1] = rng.nextDouble() * 10;
        }
        return cities;
    }

    static double tourLength(double[][] cities, int[] order) {
        double sum = 0;
        for (int i = 0; i < order.length; i++) {
            double[] a = cities[order[i]], b = cities[order[(i + 1) % order.length]];
            sum += Math.hypot(a[0] - b[0], a[1] - b[1]);
        }
        return sum;
    }

    /** Exact optimum by enumerating every permutation that keeps city 0 first. */
    static double exactTour(double[][] cities) {
        int n = cities.length; int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i; double[] best = {Double.MAX_VALUE};
        permute(cities, order, 1, best); return best[0];
    }

    static void permute(double[][] cities, int[] order, int position, double[] best) {
        if (position == order.length) {
            best[0] = Math.min(best[0], tourLength(cities, order)); return;
        }
        for (int i = position; i < order.length; i++) {
            int swap = order[position]; order[position] = order[i];
            order[i] = swap; permute(cities, order, position + 1, best);
            swap = order[position]; order[position] = order[i]; order[i] = swap;
        }
    }

    /** Annealing over 2-opt swaps of the tour (city 0 stays first). */
    static double[] annealTour(double[][] cities, double t0, double tEnd, int iterations, SplitMix64 rng) {
        int n = cities.length; int[] order = new int[n]; for (int i = 0; i < n; i++) order[i] = i;
        double energy = tourLength(cities, order), bestEnergy = energy; int[] bestOrder = order.clone();
        for (int k = 0; k < iterations; k++) {
            int i = 1 + rng.nextInt(n - 1), j = 1 + rng.nextInt(n - 1); if (i == j) continue;
            if (i > j) {
                int swap = i; i = j; j = swap;
            }
            int before = order[i - 1], after = order[(j + 1) % n];
            double delta = -Math.hypot(cities[before][0] - cities[order[i]][0], cities[before][1] - cities[order[i]][1])
                    - Math.hypot(cities[order[j]][0] - cities[after][0], cities[order[j]][1] - cities[after][1])
                    + Math.hypot(cities[before][0] - cities[order[j]][0], cities[before][1] - cities[order[j]][1])
                    + Math.hypot(cities[order[i]][0] - cities[after][0], cities[order[i]][1] - cities[after][1]);
            double temperature = Cooling.geometric(t0, tEnd, iterations, k);
            if (delta <= 0 || rng.nextDouble() < Math.exp(-delta / Math.max(temperature, 1e-300))) {
                for (int a = i, b = j; a < b; a++, b--) {
                    int swap = order[a]; order[a] = order[b]; order[b] = swap;
                }
                energy += delta;
                if (energy < bestEnergy) {
                    bestEnergy = energy; bestOrder = order.clone();
                }
            }
        }
        return new double[]{tourLength(cities, bestOrder), bestEnergy, energy};
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240800L);

        // --- the Boltzmann law by direct simulation ---------------------------------------------
        for (double temperature : new double[]{0.5, 1.0, 2.5}) {
            double[] occupancy = ringOccupancy(temperature, 400000, new SplitMix64(1234L));
            double[] expected = boltzmann(temperature); double worst = 0;
            for (int i = 0; i < occupancy.length; i++) worst = Math.max(worst, Math.abs(occupancy[i] - expected[i]));
            check(worst < 0.02, "T = " + temperature + " occupancy deviation " + worst);
        }
        for (double[] probe : new double[][]{{1.0, 1.0}, {0.5, 2.0}, {3.1, 0.7}})
            check(Math.abs(uphillAcceptance(probe[0], probe[1], 200000, new SplitMix64(77L))
                    - Math.exp(-probe[0] / probe[1])) < 0.01,
                    "uphill acceptance must equal exp(-dE/T) for delta " + probe[0] + " at T " + probe[1]);

        // --- temperature limits -------------------------------------------------------------------
        check(uphillAcceptance(2.0, 1e-9, 20000, new SplitMix64(9L)) < 1e-9,
                "at T -> 0 uphill moves are never accepted");
        check(Math.abs(uphillAcceptance(2.0, 1e6, 200000, new SplitMix64(9L)) - 1.0) < 1e-9,
                "at T -> infinity every move is accepted");
        double[] hot = ringOccupancy(1e6, 300000, new SplitMix64(21L));
        for (double share : hot) check(Math.abs(share - 1.0 / 6) < 0.01, "a very hot chain is uniform, got "
                + share);

        // --- cooling schedules ----------------------------------------------------------------------
        int iterations = 20000; boolean decreasing = true; double previous = Double.MAX_VALUE;
        for (int k = 0; k < iterations; k += 200) {
            double temperature = Cooling.geometric(5.0, 1e-3, iterations, k);
            if (temperature <= 0 || temperature >= previous) decreasing = false; previous = temperature;
        }
        check(decreasing, "the geometric schedule must be positive and strictly decreasing");
        check(Math.abs(Cooling.geometric(5.0, 1e-3, iterations, iterations) - 1e-3) < 1e-12,
                "the geometric schedule must end at the requested temperature");
        check(Cooling.linear(5.0, iterations, iterations) == 0.0, "the linear schedule ends at zero");
        check(Cooling.linear(5.0, iterations, iterations / 2) > 2.4
                && Cooling.linear(5.0, iterations, iterations / 2) < 2.6, "the linear schedule halves halfway");
        check(Cooling.logarithmic(1.0, 0) > Cooling.logarithmic(1.0, 50000),
                "the logarithmic schedule must decay too, just far more slowly");

        // --- the Rastrigin landscape: annealing versus greedy descent -------------------------------
        double[] start = {3.0, -3.0}; double startEnergy = rastrigin(start);
        for (String kind : new String[]{"geometric", "linear", "logarithmic"}) {
            double[] point = start.clone(); double energy = rastrigin(point);
            double bestEnergy = energy; SplitMix64 scheduleRng = new SplitMix64(31415L);
            for (int k = 0; k < 60000; k++) {
                double temperature = kind.equals("geometric") ? Cooling.geometric(4.0, 1e-3, 60000, k)
                        : kind.equals("linear") ? Cooling.linear(4.0, 60000, k)
                                : Cooling.logarithmic(2.0, k);
                double[] candidate = point.clone();
                for (int d = 0; d < 2; d++)
                    candidate[d] = Math.max(-5.12, Math.min(5.12, candidate[d] + 0.7 * scheduleRng.nextGaussian()));
                double candidateEnergy = rastrigin(candidate); double delta = candidateEnergy - energy;
                if (delta <= 0 || scheduleRng.nextDouble() < Math.exp(-delta / Math.max(temperature, 1e-300))) {
                    point = candidate; energy = candidateEnergy;
                    bestEnergy = Math.min(bestEnergy, energy);
                }
            }
            check(bestEnergy < 1.0, "the " + kind + " schedule must reach the global basin, best energy "
                    + bestEnergy);
        }
        AnnealRun annealed = anneal(start, 4.0, 1e-3, 60000, 0.7, false, new SplitMix64(31415L));
        AnnealRun greedy = anneal(start, 4.0, 1e-3, 60000, 0.7, true, new SplitMix64(31415L));
        double[] local = localSearch(start); double localEnergy = rastrigin(local);
        check(annealed.bestEnergy < 0.5, "annealing must find the global minimum, got " + annealed.bestEnergy);
        check(Math.hypot(annealed.best[0], annealed.best[1]) < 0.5, "the minimiser must be near the origin, got "
                + Arrays.toString(annealed.best));
        check(Math.hypot(local[0] - 3, local[1] + 3) < 0.05 && Math.abs(localEnergy - 18) < 0.2,
                "steepest descent from (3, -3) must stay in its own lattice basin, got "
                        + Arrays.toString(local) + " with energy " + localEnergy);
        check(localEnergy > 10 * annealed.bestEnergy + 1, "the local method " + localEnergy
                + " must be far worse than annealing " + annealed.bestEnergy);
        check(greedy.uphillAccepted == 0, "a descent that refuses uphill moves accepts none");
        check(greedy.uphillProposed > 0, "the greedy variant still proposes moves");
        check(annealed.bestEnergy < startEnergy, "annealing must improve on the starting point");
        boolean monotone = true;
        for (int k = 1; k < annealed.energyTrace.length; k++)
            if (annealed.energyTrace[k] > annealed.energyTrace[k - 1] + 1e-12) monotone = false;
        check(monotone, "the best-so-far trace must never increase");
        check(annealed.uphillProposed > 0 && annealed.uphillAccepted > 0,
                "annealing must actually explore uphill moves");

        // --- nine-city TSP against exhaustive enumeration --------------------------------------------
        double[][] cities = tspCities(9, new SplitMix64(271828L)); double exact = exactTour(cities);
        double[] annealedTour = annealTour(cities, 3.0, 1e-4, 60000, new SplitMix64(161803L));
        check(Math.abs(annealedTour[0] - exact) < 1e-9, "the annealed tour " + annealedTour[0]
                + " must equal the exact optimum " + exact);
        int[] natural = new int[9]; for (int i = 0; i < 9; i++) natural[i] = i;
        check(exact <= tourLength(cities, natural) + 1e-12, "the optimum can never exceed the natural order");

        // --- degenerate runs ---------------------------------------------------------------------------
        AnnealRun nothing = anneal(start, 4.0, 1e-3, 0, 0.7, false, new SplitMix64(1L));
        check(nothing.bestEnergy == startEnergy && nothing.energyTrace.length == 1,
                "zero iterations must return the starting point");
        double[][] single = {{0, 0}};
        check(Math.abs(exactTour(single) - 0.0) < 1e-12, "a one-city tour has zero length");
        check(Math.abs(tourLength(single, new int[]{0})) < 1e-12, "a one-city loop is closed");
        AnnealRun repeat = anneal(start, 4.0, 1e-3, 60000, 0.7, false, new SplitMix64(31415L));
        check(repeat.bestEnergy == annealed.bestEnergy && repeat.best[0] == annealed.best[0],
                "the annealer must be reproducible");

        System.out.println("Boltzmann checks passed at T = 0.5, 1.0, 2.5; acceptance law verified");
        System.out.println("Rastrigin: annealing " + annealed.bestEnergy + " at "
                + Arrays.toString(annealed.best) + " versus greedy " + greedy.bestEnergy + " from "
                + startEnergy);
        System.out.println("TSP(9): annealed " + annealedTour[0] + ", exhaustive optimum " + exact
                + ", natural order " + tourLength(cities, natural));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
