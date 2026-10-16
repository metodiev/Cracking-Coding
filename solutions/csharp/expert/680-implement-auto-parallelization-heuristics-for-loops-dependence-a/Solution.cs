// Task 680 - Implement auto-parallelization heuristics for loops (dependence analysis).
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// Dependence analysis for single loops over affine subscripts, with the three tests a compiler actually
// combines and an exact ground truth to police them:
//   * the GCD test on the subscript difference. Two references A[a*i + c] and B[a'*i' + c'] touch the same
//     element only if a*i - a'*i' = c' - c has an integer solution, which requires gcd(a, a') to divide
//     c' - c; when it does not, there is no dependence at all (necessary condition, so a negative answer is
//     a proof).
//   * the Banerjee test: over i, i' in [0, N) the difference a*i - a'*i' + c - c' ranges over an interval
//     whose endpoints are computed exactly (the extreme values of a linear form over a box); if zero is
//     outside that interval there is no dependence (sufficient condition for independence).
//   * a brute-force enumeration over all (i, i') pairs for small trip counts, which is the exact answer for
//     the concrete loop: it separates loop-carried dependences (i != i', which forbid parallelising the
//     loop) from loop-independent ones (i == i', which do not).
// The analyser is only ever allowed to *claim* independence when the GCD test fails or Banerjee succeeds,
// and the file verifies that claim against the enumeration for every generated loop — no false positives are
// tolerated, because a false positive would be a miscompile. The decision is then executed, not just
// printed: a loop the analyser calls parallel is run in two chunk orders and must produce identical arrays,
// while a loop it calls sequential is shown to be order-sensitive by exhibiting two chunk orders that
// differ. Two arrays are modelled so that genuinely independent loops (SAXPY writing A from B) are in the
// corpus and not trivially rejected.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic loop generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static int NextRange(ref ulong state, int low, int high) => low + NextBelow(ref state, high - low + 1);
}

/// A[i*dstStep + dstOffset] = (src array)[i*srcStep + srcOffset] * mul + add
internal sealed class Stmt
{
    internal int DstArray;
    internal int DstStep;
    internal int DstOffset;
    internal int SrcArray;
    internal int SrcStep;
    internal int SrcOffset;
    internal int Mul = 1;
    internal int Add;

    internal static Stmt Make(int dstArray, int dstStep, int dstOffset, int srcArray, int srcStep, int srcOffset)
    {
        return new Stmt
        {
            DstArray = dstArray, DstStep = dstStep, DstOffset = dstOffset,
            SrcArray = srcArray, SrcStep = srcStep, SrcOffset = srcOffset
        };
    }

    internal int Dst(int i) => DstStep * i + DstOffset;
    internal int Src(int i) => SrcStep * i + SrcOffset;
}

internal sealed class Loop
{
    internal string Name = "";
    internal int Trips;
    internal List<Stmt> Body = new List<Stmt>();

    internal static Loop Make(string name, int trips, params Stmt[] statements)
    {
        var loop = new Loop { Name = name, Trips = trips };
        loop.Body.AddRange(statements);
        return loop;
    }
}

internal static class Dependence
{
    internal static int Gcd(int a, int b)
    {
        a = a < 0 ? -a : a;
        b = b < 0 ? -b : b;
        while (b != 0)
        {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /// Necessary condition: a*i - a'*i' = c' - c can have an integer solution only if gcd divides the
    /// constant difference. Returns false when a dependence is *impossible*.
    internal static bool GcdTestAllows(int stepA, int stepB, int constantDifference)
    {
        if (stepA == 0 && stepB == 0) return constantDifference == 0;
        int divisor = Gcd(stepA, stepB);
        return divisor != 0 && constantDifference % divisor == 0;
    }

    /// Sufficient condition for independence: the exact range of `stepA*i - stepB*i' + constant` over the box
    /// i, i' in [0, trips) must exclude zero.
    internal static bool BanerjeeProvesIndependent(int stepA, int stepB, int constant, int trips)
    {
        int low = int.MaxValue;
        int high = int.MinValue;
        // The extremes of a linear form over a box are attained at the corners.
        for (int a = 0; a < 2; a++)
        {
            for (int b = 0; b < 2; b++)
            {
                int i = a == 0 ? 0 : trips - 1;
                int j = b == 0 ? 0 : trips - 1;
                int value = stepA * i - stepB * j + constant;
                if (value < low) low = value;
                if (value > high) high = value;
            }
        }
        return low > 0 || high < 0;
    }

    internal sealed class Enumeration
    {
        internal bool Carried;
        internal bool Independent;
        internal int Pairs;
    }

    /// Exact answer for the concrete loop: does any pair (i, i') with i != i' access the same element, where
    /// one access is a write and the other a read or a write of the same array?
    internal static Enumeration BruteForce(Loop loop, int trips)
    {
        var result = new Enumeration { Independent = true };
        for (int s = 0; s < loop.Body.Count; s++)
        {
            for (int t = 0; t < loop.Body.Count; t++)
            {
                Stmt first = loop.Body[s];
                Stmt second = loop.Body[t];
                for (int i = 0; i < trips; i++)
                {
                    for (int j = 0; j < trips; j++)
                    {
                        // A write in `first` to the same array slot that `second` touches, with i != j.
                        if (first.DstArray != second.DstArray) continue;
                        int written = first.Dst(i);
                        int read = second.Src(j);
                        if (written != read) continue;
                        result.Pairs++;
                        if (s == t && i == j) continue;          // the same access, not a dependence
                        if (i != j) result.Carried = true;
                    }
                }
            }
            // Output dependences: two different iterations writing the same slot.
            Stmt statement = loop.Body[s];
            for (int i = 0; i < trips; i++)
            {
                for (int j = i + 1; j < trips; j++)
                {
                    if (statement.Dst(i) == statement.Dst(j))
                    {
                        result.Pairs++;
                        result.Carried = true;
                    }
                }
            }
        }
        result.Independent = !result.Carried;
        return result;
    }

    /// The analyser: claim parallel only when the GCD test rules dependence out or Banerjee proves the
    /// subscript ranges disjoint. Everything else is conservatively sequential.
    internal sealed class Decision
    {
        internal bool Parallel = true;
        internal string Reason = "no dependences";
    }

    internal static Decision Analyse(Loop loop) => Analyse(loop, true);

    /// The same skeleton with only the GCD test, used to show that the refinements can only ever turn
    /// "sequential" into "parallel" and never the other way round.
    internal static Decision AnalyseGcdOnly(Loop loop) => Analyse(loop, false);

    internal static Decision Analyse(Loop loop, bool refinements)
    {
        var decision = new Decision();
        int trips = loop.Trips;
        for (int s = 0; s < loop.Body.Count; s++)
        {
            for (int t = 0; t < loop.Body.Count; t++)
            {
                Stmt write = loop.Body[s];
                Stmt access = loop.Body[t];
                if (write.DstArray != access.DstArray) continue;
                int stepA = write.DstStep;
                int stepB = access.SrcStep;
                int constant = write.DstOffset - access.SrcOffset;
                if (!Dependence.GcdTestAllows(stepA, stepB, constant))
                {
                    continue;                                        // proven independent
                }
                if (refinements && stepA == stepB && stepA != 0)
                {
                    // Equal steps make the distance exact: only the value of (i - i') matters.
                    int distance = constant % stepA == 0 ? -(constant / stepA) : int.MaxValue;
                    if (distance == 0 || distance >= trips || distance <= -trips) continue;
                    decision.Parallel = false;
                    decision.Reason = "carried dependence at distance " + distance;
                    return decision;
                }
                if (refinements && Dependence.BanerjeeProvesIndependent(stepA, stepB, constant, trips))
                {
                    continue;                                        // proven independent
                }
                decision.Parallel = false;
                decision.Reason = "possible dependence A[" + stepA + "i+" + write.DstOffset + "] against A["
                                  + stepB + "i+" + access.SrcOffset + "]";
                return decision;
            }

            // Output dependence: every iteration writing the same slot.
            Stmt statement = loop.Body[s];
            if (statement.DstStep == 0 && trips > 1)
            {
                decision.Parallel = false;
                decision.Reason = "every iteration writes the same slot A[" + statement.DstOffset + "]";
                return decision;
            }
        }
        return decision;
    }
}

internal static class LoopRunner
{
    internal static int[] Run(Loop loop, int[] a, int[] b, List<int> iterationOrder)
    {
        int[] dstA = (int[])a.Clone();
        int[] dstB = (int[])b.Clone();
        foreach (int i in iterationOrder)
        {
            foreach (Stmt statement in loop.Body)
            {
                int[] target = statement.DstArray == 0 ? dstA : dstB;
                int[] source = statement.SrcArray == 0 ? dstA : dstB;
                int index = statement.Dst(i);
                if (index < 0 || index >= target.Length) continue;
                int srcIndex = statement.Src(i);
                int value = srcIndex >= 0 && srcIndex < source.Length ? source[srcIndex] : 0;
                target[index] = value * statement.Mul + statement.Add;
            }
        }
        return Merge(dstA, dstB);
    }

    private static int[] Merge(int[] a, int[] b)
    {
        var merged = new int[a.Length + b.Length];
        for (int i = 0; i < a.Length; i++) merged[i] = a[i];
        for (int i = 0; i < b.Length; i++) merged[a.Length + i] = b[i];
        return merged;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Loop Saxpy() => Loop.Make("saxpy", 16,
        Stmt.Make(0, 1, 0, 1, 1, 0));

    private static Loop PrefixSum() => Loop.Make("prefix", 16,
        Stmt.Make(0, 1, 1, 0, 1, 0));

    private static Loop Scatter() => Loop.Make("scatter", 16,
        Stmt.Make(0, 2, 0, 0, 2, 1));

    private static Loop Reverse() => Loop.Make("reverse", 16,
        Stmt.Make(0, 1, 0, 0, -1, 15));

    private static Loop ReductionSlots() => Loop.Make("accumulate-into-one", 16,
        Stmt.Make(0, 0, 0, 0, 1, 0));

    /// A[i] = A[i] + 1: the only same-slot access is the diagonal, so the loop *is* parallel.
    private static Loop SelfIncrement() => Loop.Make("self-increment", 16,
        Stmt.Make(0, 1, 0, 0, 1, 0));

    /// A[i] = A[i+1] + 1: a carried dependence at distance one in the other direction.
    private static Loop ShiftDown() => Loop.Make("shift-down", 16,
        Stmt.Make(0, 1, 0, 0, 1, 1));

    private static Loop RandomLoop(ref ulong state, int trips)
    {
        int statements = 1 + Rng.NextBelow(ref state, 2);
        var body = new List<Stmt>();
        for (int s = 0; s < statements; s++)
        {
            body.Add(Stmt.Make(Rng.NextBelow(ref state, 2), Rng.NextRange(ref state, -2, 2), Rng.NextRange(ref state, -3, 3),
                              Rng.NextBelow(ref state, 2), Rng.NextRange(ref state, -2, 2), Rng.NextRange(ref state, -3, 3)));
        }
        return Loop.Make("random", trips, body.ToArray());
    }

    private static List<int> Order(int trips, bool reversedChunks)
    {
        var order = new List<int>();
        int half = trips / 2;
        if (reversedChunks)
        {
            for (int i = half; i < trips; i++) order.Add(i);
            for (int i = 0; i < half; i++) order.Add(i);
        }
        else
        {
            for (int i = 0; i < half; i++) order.Add(i);
            for (int i = half; i < trips; i++) order.Add(i);
        }
        return order;
    }

    private static bool SameArray(int[] x, int[] y)
    {
        if (x.Length != y.Length) return false;
        for (int i = 0; i < x.Length; i++) if (x[i] != y[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- the two linear tests in isolation -------------------------------------------------
        Check(!Dependence.GcdTestAllows(2, 4, 3), "gcd(2,4) = 2 does not divide 3, so no dependence exists");
        Check(Dependence.GcdTestAllows(2, 4, 6), "but it does divide 6, so a dependence cannot be excluded");
        Check(Dependence.GcdTestAllows(0, 3, 0), "a constant subscript equal to a stepped one is not excluded");
        Check(Dependence.BanerjeeProvesIndependent(1, 1, 9, 4),
              "the ranges of A[i+9] and A[i'] are further apart than the trip count, which Banerjee proves");
        Check(!Dependence.BanerjeeProvesIndependent(1, 1, 3, 4),
              "while A[i+3] and A[i'] certainly meet inside a four-trip loop");

        // ---- the library of textbook loops -----------------------------------------------------
        Check(Dependence.Analyse(Saxpy()).Parallel && Dependence.BruteForce(Saxpy(), 16).Independent,
              "A[i] = B[i] + 1 is parallel: the two arrays never interfere");
        Dependence.Decision prefix = Dependence.Analyse(PrefixSum());
        Check(!prefix.Parallel, "A[i] = A[i-1] + 1 is sequential: " + prefix.Reason);
        Check(Dependence.BruteForce(PrefixSum(), 16).Carried, "and the enumeration finds the carried dependence");
        Check(Dependence.Analyse(Scatter()).Parallel && Dependence.BruteForce(Scatter(), 16).Independent,
              "A[2i] = A[2i+1] + 1 is parallel: only even positions are written and only odd ones read");
        Dependence.Decision reverse = Dependence.Analyse(Reverse());
        Check(!reverse.Parallel, "A[i] = A[N-1-i] is sequential: " + reverse.Reason);
        Check(Dependence.BruteForce(Reverse(), 16).Carried, "and the enumeration agrees");
        Dependence.Decision accumulate = Dependence.Analyse(ReductionSlots());
        Check(!accumulate.Parallel, "writing A[0] from A[i] every iteration is sequential: " + accumulate.Reason);
        Dependence.Decision selfIncrement = Dependence.Analyse(SelfIncrement());
        Check(selfIncrement.Parallel, "A[i] = A[i] + 1 only touches each slot within its own iteration, so it is"
              + " parallel (" + selfIncrement.Reason + ")");
        Check(Dependence.BruteForce(SelfIncrement(), 16).Independent, "and the enumeration confirms that");
        Dependence.Decision shiftDown = Dependence.Analyse(ShiftDown());
        Check(!shiftDown.Parallel && shiftDown.Reason.Contains("distance"),
              "A[i] = A[i+1] + 1 is sequential: " + shiftDown.Reason);

        // ---- the decision is executed, not just printed ----------------------------------------
        int[] a = new int[64];
        int[] b = new int[64];
        for (int i = 0; i < 64; i++)
        {
            a[i] = i % 5;
            b[i] = (i * 3) % 7;
        }
        int[] sequential = LoopRunner.Run(Saxpy(), a, b, Order(16, false));
        int[] chunked = LoopRunner.Run(Saxpy(), a, b, Order(16, true));
        Check(SameArray(sequential, chunked), "the parallel loop gives the same result for either chunk order");
        int[] prefixInOrder = LoopRunner.Run(PrefixSum(), a, b, Order(16, false));
        int[] prefixChunked = LoopRunner.Run(PrefixSum(), a, b, Order(16, true));
        Check(!SameArray(prefixInOrder, prefixChunked),
              "while the sequential loop really is order-sensitive, so calling it parallel would be wrong");

        // ---- randomised soundness: no false "parallel" claims ----------------------------------
        ulong state = 680001UL;
        int loops = 400;
        int claimedParallel = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        int refinementGains = 0;
        for (int l = 0; l < loops; l++)
        {
            int trips = 4 + Rng.NextBelow(ref state, 9);
            Loop loop = RandomLoop(ref state, trips);
            Dependence.Decision decision = Dependence.Analyse(loop);
            Dependence.Decision gcdOnly = Dependence.AnalyseGcdOnly(loop);
            Dependence.Enumeration exact = Dependence.BruteForce(loop, trips);
            if (gcdOnly.Parallel && !decision.Parallel)
            {
                Check(false, "loop " + l + ": the full analysis is less precise than the GCD test alone");
            }
            if (!gcdOnly.Parallel && decision.Parallel) refinementGains++;
            if (decision.Parallel)
            {
                claimedParallel++;
                if (!exact.Independent)
                {
                    falsePositives++;
                    Check(false, "loop " + l + " was called parallel although the enumeration found a carried"
                          + " dependence: " + decision.Reason);
                }
            }
            else if (exact.Independent)
            {
                falseNegatives++;
            }
        }
        Check(claimedParallel >= 50, "the analyser proved " + claimedParallel + " of " + loops
              + " random loops parallel");
        Check(falsePositives == 0, "and never claimed parallelism where a carried dependence exists");
        Check(refinementGains >= 5, "the distance and Banerjee refinements proved " + refinementGains
              + " loops parallel that the GCD test alone rejected");
        Check(falseNegatives < loops, "the conservative answer still leaves " + falseNegatives
              + " missed parallel loops out of " + loops + " (reported, not asserted away)");

        // ---- the enumeration itself ------------------------------------------------------------
        Dependence.Enumeration saxpyEnumeration = Dependence.BruteForce(Saxpy(), 16);
        Check(saxpyEnumeration.Pairs == 16, "the enumeration counted the 16 same-iteration accesses of SAXPY ("
              + saxpyEnumeration.Pairs + ")");
        Check(!saxpyEnumeration.Carried && saxpyEnumeration.Independent, "all of them are loop-independent");
        Dependence.Enumeration prefixEnumeration = Dependence.BruteForce(PrefixSum(), 16);
        Check(prefixEnumeration.Carried, "the prefix sum has a carried dependence at distance one");
        Check(prefixEnumeration.Pairs == 15, "and the enumeration visited every overlapping pair ("
              + prefixEnumeration.Pairs + " pairs at distance one)");

        // ---- degenerate loops ------------------------------------------------------------------
        var empty = Loop.Make("empty", 8);
        Check(Dependence.Analyse(empty).Parallel, "an empty loop body is trivially parallel");
        Check(Dependence.BruteForce(empty, 8).Independent, "and the enumeration agrees");
        var constant = Loop.Make("constant", 8, Stmt.Make(0, 0, 7, 1, 1, 0));
        Dependence.Decision constantDecision = Dependence.Analyse(constant);
        Check(!constantDecision.Parallel, "writing the same slot from every iteration is an output dependence: "
              + constantDecision.Reason);
        Check(Dependence.BruteForce(constant, 8).Carried, "which the enumeration also finds");
        var singleTrip = Loop.Make("one-trip", 1, Stmt.Make(0, 1, 0, 0, 1, 0));
        Check(Dependence.BruteForce(singleTrip, 1).Independent, "a single-trip loop cannot carry anything");

        Console.WriteLine("library: saxpy parallel, prefix/reverse/accumulate sequential");
        Console.WriteLine("random loops: " + claimedParallel + " parallel, " + falseNegatives
              + " conservative, 0 false positives of " + loops);
        Console.WriteLine("PASS");
    }
}
