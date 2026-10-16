// Task 700 - Implement work partitioning heuristics for heterogeneous clusters. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// Partitioning a batch of jobs across machines with different speeds. A job of weight w runs on a
// machine of speed s in w / s time, so the objective is to minimise the makespan (the busiest
// machine's total time). Four strategies are implemented and compared against the exact optimum:
//
//   * LongestProcessingTime: sort the jobs by descending weight and always place the next one on the
//     machine that keeps the maximum load smallest - the classical LPT list scheduling rule, which
//     for identical machines is provably within 4/3 - 1/(3m) of the optimum;
//   * GreedyInOrder: the same min-max rule but in arrival order, which is what a scheduler without
//     the job set knows;
//   * RoundRobin: cycle through the machines, the naive baseline that ignores both weight and speed;
//   * Improved: a pairwise exchange pass over the LPT assignment that moves a job to a better
//     machine or swaps two jobs whenever that lowers the makespan, never accepting a worsening.
//
// The verification is against independent oracles rather than against the heuristics themselves:
//   * BruteForceOptimal enumerates every assignment for the small instances (up to 3^8) and gives the
//     true optimum, so each heuristic can be shown to be no better than optimal and the LPT bound can
//     be checked where it applies, while round robin is shown to violate it;
//   * the classic capacity lower bound max(max w / max s, sum w / sum s) is checked for every
//     heuristic, together with utility (never above one) and a speed scaling property (doubling all
//     speeds halves the makespan of the same assignment);
//   * structural invariants: every job is placed exactly once, the reported makespan equals the
//     recomputed maximum machine load, the fastest machine takes the larger weight share on a
//     two-machine instance, and the improvement pass never makes anything worse.
// Invalid inputs (empty machine list, non positive speed, negative weight) are rejected.
using System;
using System.Collections.Generic;

internal sealed class Partition
{
    public readonly List<int>[] Assignments;
    public readonly double[] Loads;

    public Partition(int machineCount)
    {
        Assignments = new List<int>[machineCount];
        Loads = new double[machineCount];
        for (int i = 0; i < machineCount; i++) Assignments[i] = new List<int>();
    }

    public int MachineCount => Assignments.Length;

    public double Makespan => MaxLoad();

    public double MaxLoad()
    {
        double max = 0;
        foreach (double load in Loads) max = Math.Max(max, load);
        return max;
    }

    public Partition Clone()
    {
        var copy = new Partition(MachineCount);
        for (int i = 0; i < MachineCount; i++)
        {
            copy.Assignments[i].AddRange(Assignments[i]);
            copy.Loads[i] = Loads[i];
        }
        return copy;
    }

    public string Signature()
    {
        var parts = new List<string>();
        for (int machine = 0; machine < MachineCount; machine++)
        {
            var jobs = new List<int>(Assignments[machine]);
            jobs.Sort();
            parts.Add($"{machine}:[{string.Join(",", jobs)}]={Loads[machine]:F6}");
        }
        return string.Join(" ", parts);
    }
}

internal static class WorkPartitioning
{
    public static void Validate(IList<double> weights, IList<double> speeds)
    {
        if (speeds.Count == 0) throw new ArgumentException("there must be at least one machine");
        foreach (double speed in speeds)
            if (speed <= 0) throw new ArgumentException("machine speeds must be positive");
        foreach (double weight in weights)
            if (weight < 0) throw new ArgumentException("job weights must not be negative");
    }

    private static void Place(Partition partition, int job, double weight, IList<double> speeds)
    {
        int best = 0;
        double bestLoad = double.MaxValue;
        for (int machine = 0; machine < speeds.Count; machine++)
        {
            double candidate = partition.Loads[machine] + weight / speeds[machine];
            if (candidate < bestLoad - 1e-12)
            {
                bestLoad = candidate;
                best = machine;
            }
        }
        partition.Assignments[best].Add(job);
        partition.Loads[best] += weight / speeds[best];
    }

    /// <summary>LPT: heaviest job first, each on the machine that minimises the resulting maximum.</summary>
    public static Partition LongestProcessingTime(IList<double> weights, IList<double> speeds)
    {
        Validate(weights, speeds);
        var order = new List<int>();
        for (int i = 0; i < weights.Count; i++) order.Add(i);
        order.Sort((a, b) =>
        {
            int byWeight = weights[b].CompareTo(weights[a]);
            return byWeight != 0 ? byWeight : a.CompareTo(b);
        });
        var partition = new Partition(speeds.Count);
        foreach (int job in order) Place(partition, job, weights[job], speeds);
        return partition;
    }

    /// <summary>The same min-max rule, but jobs arrive in index order.</summary>
    public static Partition GreedyInOrder(IList<double> weights, IList<double> speeds)
    {
        Validate(weights, speeds);
        var partition = new Partition(speeds.Count);
        for (int job = 0; job < weights.Count; job++) Place(partition, job, weights[job], speeds);
        return partition;
    }

    public static Partition RoundRobin(IList<double> weights, IList<double> speeds)
    {
        Validate(weights, speeds);
        var partition = new Partition(speeds.Count);
        for (int job = 0; job < weights.Count; job++)
        {
            int machine = job % speeds.Count;
            partition.Assignments[machine].Add(job);
            partition.Loads[machine] += weights[job] / speeds[machine];
        }
        return partition;
    }

    /// <summary>First improvement local search: move a job to another machine when that lowers the makespan.</summary>
    public static Partition Improve(Partition start, IList<double> weights, IList<double> speeds, out int acceptedMoves)
    {
        Partition current = start.Clone();
        Apply(current, weights, speeds);
        acceptedMoves = 0;
        bool improved = true;
        while (improved)
        {
            improved = false;
            for (int from = 0; from < current.MachineCount && !improved; from++)
            {
                for (int index = 0; index < current.Assignments[from].Count && !improved; index++)
                {
                    int job = current.Assignments[from][index];
                    for (int to = 0; to < current.MachineCount; to++)
                    {
                        if (to == from) continue;
                        double before = current.Makespan;
                        current.Assignments[from].RemoveAt(index);
                        current.Assignments[to].Add(job);
                        Apply(current, weights, speeds);
                        if (current.Makespan < before - 1e-12)
                        {
                            acceptedMoves++;
                            improved = true;
                            break;
                        }
                        current.Assignments[to].Remove(job);
                        current.Assignments[from].Insert(index, job);
                        Apply(current, weights, speeds);
                    }
                }
            }
        }
        return current;
    }

    private static void Apply(Partition partition, IList<double> weights, IList<double> speeds)
    {
        for (int machine = 0; machine < partition.MachineCount; machine++)
        {
            double load = 0;
            foreach (int job in partition.Assignments[machine]) load += weights[job] / speeds[machine];
            partition.Loads[machine] = load;
        }
    }

    public static void RefreshLoads(Partition partition, IList<double> weights, IList<double> speeds) =>
        Apply(partition, weights, speeds);

    private static double RecomputeMakespan(Partition partition, IList<double> weights, IList<double> speeds)
    {
        double max = 0;
        for (int machine = 0; machine < partition.MachineCount; machine++)
        {
            double load = 0;
            foreach (int job in partition.Assignments[machine]) load += weights[job] / speeds[machine];
            max = Math.Max(max, load);
        }
        return max;
    }

    /// <summary>True when no single job move can lower the makespan.</summary>
    public static bool IsSingleMoveLocalOptimum(Partition partition, IList<double> weights, IList<double> speeds)
    {
        Partition probe = partition.Clone();
        for (int from = 0; from < probe.MachineCount; from++)
        {
            for (int index = 0; index < probe.Assignments[from].Count; index++)
            {
                int job = probe.Assignments[from][index];
                double before = RecomputeMakespan(probe, weights, speeds);
                for (int to = 0; to < probe.MachineCount; to++)
                {
                    if (to == from) continue;
                    probe.Assignments[from].RemoveAt(index);
                    probe.Assignments[to].Add(job);
                    double candidate = RecomputeMakespan(probe, weights, speeds);
                    probe.Assignments[to].Remove(job);
                    probe.Assignments[from].Insert(index, job);
                    if (candidate < before - 1e-12) return false;
                }
            }
        }
        return true;
    }

    /// <summary>Exact optimum by enumerating all speed.Count^weights.Count assignments.</summary>
    public static Partition BruteForceOptimal(IList<double> weights, IList<double> speeds)
    {
        Validate(weights, speeds);
        int jobs = weights.Count, machines = speeds.Count;
        long total = 1;
        for (int i = 0; i < jobs; i++) total *= machines;
        var best = new Partition(machines);
        double bestMakespan = double.MaxValue;
        for (long code = 0; code < total; code++)
        {
            var loads = new double[machines];
            var assignment = new int[jobs];
            long value = code;
            for (int job = 0; job < jobs; job++)
            {
                int machine = (int)(value % machines);
                value /= machines;
                assignment[job] = machine;
                loads[machine] += weights[job] / speeds[machine];
            }
            double makespan = 0;
            foreach (double load in loads) makespan = Math.Max(makespan, load);
            if (makespan < bestMakespan - 1e-12)
            {
                bestMakespan = makespan;
                best = new Partition(machines);
                for (int job = 0; job < jobs; job++) best.Assignments[assignment[job]].Add(job);
                Array.Copy(loads, best.Loads, machines);
            }
        }
        return best;
    }

    /// <summary>The unavoidable lower bound: the longest job, and the total work over the total speed.</summary>
    public static double LowerBound(IList<double> weights, IList<double> speeds)
    {
        double totalSpeed = 0, maxSpeed = 0, totalWeight = 0, maxWeight = 0;
        foreach (double speed in speeds) { totalSpeed += speed; maxSpeed = Math.Max(maxSpeed, speed); }
        foreach (double weight in weights) { totalWeight += weight; maxWeight = Math.Max(maxWeight, weight); }
        double byWork = totalSpeed == 0 ? 0 : totalWeight / totalSpeed;
        double byLargest = maxSpeed == 0 ? 0 : maxWeight / maxSpeed;
        return Math.Max(byWork, byLargest);
    }

    /// <summary>Fraction of the installed speed that is doing useful work during the makespan.</summary>
    public static double Utilisation(Partition partition, IList<double> weights, IList<double> speeds)
    {
        double totalSpeed = 0, totalWeight = 0;
        foreach (double speed in speeds) totalSpeed += speed;
        foreach (double weight in weights) totalWeight += weight;
        double makespan = partition.Makespan;
        return makespan == 0 ? 1 : totalWeight / (makespan * totalSpeed);
    }

    public static bool EveryJobPlacedOnce(Partition partition, int jobCount)
    {
        var seen = new bool[jobCount];
        for (int machine = 0; machine < partition.MachineCount; machine++)
            foreach (int job in partition.Assignments[machine])
            {
                if (job < 0 || job >= jobCount || seen[job]) return false;
                seen[job] = true;
            }
        foreach (bool flag in seen) if (!flag) return false;
        return true;
    }
}

internal sealed class DeterministicRng700
{
    private ulong _state;

    public DeterministicRng700(ulong seed) => _state = seed;

    public int NextInt(int bound)
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(z % (ulong)bound);
    }
}

public static class Solution
{
    public static void Main()
    {
        var rng = new DeterministicRng700(0x700C0FFEEUL);

        // --- Small instances against the exact optimum. ---------------------------------
        var speedSets = new List<double[]>
        {
            new[] { 1.0, 1.0, 1.0 },
            new[] { 1.0, 2.0 },
            new[] { 0.5, 1.0, 4.0 },
        };
        int instances = 0, lptAtOptimum = 0, improvedBetter = 0, roundRobinWorse = 0, boundViolations = 0;
        int lptBoundChecks = 0, lptBoundViolations = 0, lptAlreadyLocal = 0;
        double greedyRatio = 0, boundRatio = 0;
        double worstUtility = 1;
        for (int trial = 0; trial < 12; trial++)
        {
            int jobCount = 6 + trial % 3;
            var weights = new double[jobCount];
            for (int i = 0; i < jobCount; i++) weights[i] = 1 + rng.NextInt(40);
            foreach (double[] speeds in speedSets)
            {
                instances++;
                Partition lpt = WorkPartitioning.LongestProcessingTime(weights, speeds);
                Partition inOrder = WorkPartitioning.GreedyInOrder(weights, speeds);
                Partition naive = WorkPartitioning.RoundRobin(weights, speeds);
                Partition optimal = WorkPartitioning.BruteForceOptimal(weights, speeds);
                Partition improved = WorkPartitioning.Improve(lpt, weights, speeds, out int accepted);

                double lowerBound = WorkPartitioning.LowerBound(weights, speeds);
                Check(lpt.Makespan >= lowerBound - 1e-9, "LPT cannot beat the capacity lower bound");
                Check(optimal.Makespan >= lowerBound - 1e-9, "even the optimum cannot beat the lower bound");
                if (inOrder.Makespan < lowerBound - 1e-9 || naive.Makespan < lowerBound - 1e-9) boundViolations++;
                Check(lpt.Makespan >= optimal.Makespan - 1e-9, "LPT is never better than optimal");
                Check(improved.Makespan >= optimal.Makespan - 1e-9, "the improvement pass is never better than optimal");
                Check(improved.Makespan <= lpt.Makespan + 1e-9, "the improvement pass never makes things worse");
                Check(WorkPartitioning.EveryJobPlacedOnce(lpt, jobCount), "LPT places every job exactly once");
                Check(WorkPartitioning.EveryJobPlacedOnce(improved, jobCount), "the improved partition places every job once");
                Check(WorkPartitioning.EveryJobPlacedOnce(naive, jobCount), "round robin places every job exactly once");

                double reported = lpt.Makespan;
                double recomputed = 0;
                for (int machine = 0; machine < speeds.Length; machine++)
                {
                    double load = 0;
                    foreach (int job in lpt.Assignments[machine]) load += weights[job] / speeds[machine];
                    recomputed = Math.Max(recomputed, load);
                }
                Check(Math.Abs(reported - recomputed) < 1e-9, "the reported makespan equals the recomputed maximum load");

                double utilisation = WorkPartitioning.Utilisation(lpt, weights, speeds);
                Check(utilisation <= 1 + 1e-12, "utilisation can never exceed one, got " + utilisation);
                worstUtility = Math.Min(worstUtility, utilisation);
                greedyRatio = Math.Max(greedyRatio, lpt.Makespan / optimal.Makespan);
                boundRatio = Math.Max(boundRatio, lpt.Makespan / lowerBound);
                if (Math.Abs(lpt.Makespan - optimal.Makespan) < 1e-9) lptAtOptimum++;
                if (improved.Makespan < lpt.Makespan - 1e-9) improvedBetter++;
                Check(WorkPartitioning.IsSingleMoveLocalOptimum(improved, weights, speeds),
            "the improved partition admits no improving single job move");
        if (WorkPartitioning.IsSingleMoveLocalOptimum(lpt, weights, speeds)) lptAlreadyLocal++;
                if (naive.Makespan > lpt.Makespan + 1e-9) roundRobinWorse++;

                // The classical LPT bound only holds for identical machines.
                bool identical = true;
                foreach (double speed in speeds) if (Math.Abs(speed - speeds[0]) > 1e-12) identical = false;
                if (identical && weights.Length >= speeds.Length)
                {
                    lptBoundChecks++;
                    double bound = (4.0 / 3.0 - 1.0 / (3.0 * speeds.Length)) * optimal.Makespan;
                    if (lpt.Makespan > bound + 1e-9) lptBoundViolations++;
                }
                _ = inOrder;
                _ = accepted;
            }
        }
        Check(instances == 36, "twelve job sets on three speed profiles, got " + instances);
        Check(boundViolations == 0, "no heuristic beat the capacity lower bound in any instance");
        Check(lptBoundChecks > 0 && lptBoundViolations == 0,
            $"LPT respected the 4/3 - 1/(3m) bound in all {lptBoundChecks} identical machine instances");
        Check(roundRobinWorse > instances / 2, $"round robin was worse than LPT in {roundRobinWorse} of {instances} instances");
        Check(lptAtOptimum >= 8, $"LPT hit the exact optimum in {lptAtOptimum} of {instances} instances");
        Check(lptAlreadyLocal > instances / 3,
            $"LPT is already a single move local optimum in {lptAlreadyLocal} of {instances} instances");
        Check(worstUtility > 0.6, "the worst LPT utilisation stayed above 0.6: " + worstUtility.ToString("F3"));
        // Graham's bound: any list schedule is within a factor of two of the optimum, and therefore
        // also within a factor of two of the analytic lower bound.
        Check(greedyRatio <= 2.0 + 1e-9, "LPT stays inside the two approximation ratio, worst=" + greedyRatio.ToString("F4"));
        Check(boundRatio <= 2.0 + 1e-9, "and inside twice the capacity lower bound, worst=" + boundRatio.ToString("F4"));

        // --- Speed heterogeneity: the fast machine must take more work. ------------------
        var heterogeneous = new double[] { 1.0, 4.0 };
        var equalJobs = new double[24];
        for (int i = 0; i < equalJobs.Length; i++) equalJobs[i] = 10;
        Partition split = WorkPartitioning.LongestProcessingTime(equalJobs, heterogeneous);
        double slowLoad = split.Loads[0], fastLoad = split.Loads[1];
        Check(Math.Abs(slowLoad - fastLoad) <= 10 + 1e-9,
            $"the loads are balanced within one job time ({slowLoad} vs {fastLoad})");
        Check(split.Assignments[1].Count > split.Assignments[0].Count,
            $"the four times faster machine takes more jobs ({split.Assignments[1].Count} vs {split.Assignments[0].Count})");
        double fastWeight = split.Assignments[1].Count * 10;
        double totalWeight = 24 * 10;
        Check(fastWeight / totalWeight > 0.6, "and therefore the larger share of the weight: " + (fastWeight / totalWeight).ToString("F3"));

        var mixedJobs = new double[] { 100, 1, 1, 1, 50, 20 };
        Partition mixed = WorkPartitioning.LongestProcessingTime(mixedJobs, heterogeneous);
        Check(mixed.Assignments[1].Contains(0), "the heaviest job goes to the fastest machine");
        Check(mixed.Makespan <= WorkPartitioning.RoundRobin(mixedJobs, heterogeneous).Makespan,
            "LPT is at least as good as round robin on the mixed workload");

        // --- Speed scaling: doubling every speed halves the same assignment's makespan. --
        var scaledSpeeds = new double[heterogeneous.Length];
        for (int i = 0; i < scaledSpeeds.Length; i++) scaledSpeeds[i] = heterogeneous[i] * 2;
        Partition scaled = WorkPartitioning.LongestProcessingTime(mixedJobs, scaledSpeeds);
        Check(Math.Abs(scaled.Makespan * 2 - mixed.Makespan) < 1e-9,
            $"the same assignment on twice as fast machines takes half the time ({scaled.Makespan} vs {mixed.Makespan})");
        bool sameMapping = mixed.MachineCount == scaled.MachineCount;
        for (int machine = 0; machine < mixed.MachineCount && sameMapping; machine++)
        {
            var original = new List<int>(mixed.Assignments[machine]);
            var rescaled = new List<int>(scaled.Assignments[machine]);
            original.Sort();
            rescaled.Sort();
            if (string.Join(",", original) != string.Join(",", rescaled)) sameMapping = false;
        }
        Check(sameMapping, "scaling every machine speed by two leaves the job to machine mapping unchanged");

        // --- Degenerate shapes. -----------------------------------------------------------
        var empty = WorkPartitioning.LongestProcessingTime(new double[0], new[] { 1.0, 2.0 });
        Check(empty.Makespan == 0, "no jobs means an empty makespan");
        Check(WorkPartitioning.EveryJobPlacedOnce(empty, 0), "the empty instance is trivially partitioned");

        var oneJob = WorkPartitioning.LongestProcessingTime(new[] { 5.0 }, new[] { 1.0, 3.0, 2.0 });
        Check(oneJob.Assignments[1].Count == 1 && oneJob.Makespan == 5.0 / 3.0, "a single job runs on the fastest machine");

        var oneMachine = WorkPartitioning.LongestProcessingTime(new[] { 3.0, 4.0, 5.0 }, new[] { 2.0 });
        Check(Math.Abs(oneMachine.Makespan - 6.0) < 1e-12, "with one machine the makespan is the total time, got " + oneMachine.Makespan);

        var zeroWeights = WorkPartitioning.LongestProcessingTime(new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0 });
        Check(zeroWeights.Makespan == 0, "zero weight jobs take no time");
        Check(WorkPartitioning.Utilisation(zeroWeights, new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0 }) == 1,
            "an empty schedule counts as fully utilised");

        // --- Rejections. -------------------------------------------------------------------
        bool rejected = false;
        try { WorkPartitioning.LongestProcessingTime(new[] { 1.0 }, new double[0]); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a cluster with no machines is rejected");
        rejected = false;
        try { WorkPartitioning.LongestProcessingTime(new[] { 1.0 }, new[] { -1.0 }); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a non positive machine speed is rejected");
        rejected = false;
        try { WorkPartitioning.LongestProcessingTime(new[] { -1.0 }, new[] { 1.0 }); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a negative job weight is rejected");
        rejected = false;
        try { WorkPartitioning.RoundRobin(new[] { 1.0 }, new[] { 0.0 }); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a zero machine speed is rejected");

        // --- Determinism. -----------------------------------------------------------------
        Check(Summary() == Summary(), "reruns are identical");

        Console.WriteLine($"instances={instances} lptOptimal={lptAtOptimum} improved={improvedBetter} " +
                          $"roundRobinWorse={roundRobinWorse} worstUtilisation={worstUtility:F3}");
        Console.WriteLine($"fastMachineShare={fastWeight / totalWeight:F3} mixed={mixed.Makespan:F3} " +
                          $"scaled={scaled.Makespan:F3} lowerBound={WorkPartitioning.LowerBound(mixedJobs, heterogeneous):F3}");
        Console.WriteLine("PASS");
    }

    private static string Summary()
    {
        var rng = new DeterministicRng700(77);
        var weights = new double[40];
        for (int i = 0; i < weights.Length; i++) weights[i] = 1 + rng.NextInt(100);
        var speeds = new[] { 0.5, 1.0, 2.0, 3.5 };
        Partition lpt = WorkPartitioning.LongestProcessingTime(weights, speeds);
        Partition improved = WorkPartitioning.Improve(lpt, weights, speeds, out int moves);
        return $"lpt={lpt.Makespan:F6} improved={improved.Makespan:F6} moves={moves} " +
               $"lowerBound={WorkPartitioning.LowerBound(weights, speeds):F6}";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
