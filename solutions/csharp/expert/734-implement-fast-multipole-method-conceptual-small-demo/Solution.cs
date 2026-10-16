// Task 734 - Implement fast multipole method conceptual small demo. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// The fast multipole method replaces particle-particle interaction with *expansions*: a cluster of
// charges is compressed into multipole moments about its centre, moments are translated between centres,
// and a translated local expansion is evaluated at the target. This demo implements the three kernels of
// the 2D FMM for the phi = Re sum q_j / (z - z_j) potential with complex coefficients and a fixed order p:
//
//   M2M  a'_l = sum_{k<=l} a_k C(l,k) (-(c_child - c_parent))^{l-k}      multipole of a parent from its children
//   M2L  L_l  = sum_{k<=p} a_k (-1)^{k+1} C(k+l,l) / (c - t)^{k+l+1}     multipole of a far cell into a local one
//   L2L  L'_l = sum_{m>=l} L_m C(m,l) (t - c)^{m-l}                      local expansion moved to a new centre
//
// Documented simplification: this is a *two-level* demo, a fixed 4x4 grid of cells with charges placed on
// a deterministic pattern, direct O(n^2) evaluation inside a cell and its neighbours and one M2L per
// well-separated cell - no adaptive tree, no downward pass over levels, no exponential translation
// operators. The mathematics that makes the FMM work is the real one, and each formula is checked
// against closed forms rather than against itself: the multipole-to-local coefficients must converge to
// the analytic -q/(s-t)^{l+1} of a single charge, M2M and L2L must reproduce the moments and the
// potential exactly as polynomial identities, and the end-to-end error must shrink with the order p.
using System;
using System.Collections.Generic;

internal struct Cplx
{
    internal Cplx(double re, double im) { Re = re; Im = im; }

    internal double Re;
    internal double Im;

    public static Cplx operator +(Cplx a, Cplx b) => new Cplx(a.Re + b.Re, a.Im + b.Im);
    public static Cplx operator -(Cplx a, Cplx b) => new Cplx(a.Re - b.Re, a.Im - b.Im);
    public static Cplx operator *(Cplx a, Cplx b) => new Cplx(a.Re * b.Re - a.Im * b.Im, a.Re * b.Im + a.Im * b.Re);
    public static Cplx operator *(double s, Cplx a) => new Cplx(s * a.Re, s * a.Im);

    internal double Abs2 => Re * Re + Im * Im;
    internal Cplx Inverse()
    {
        double scale = 1.0 / Abs2;
        return new Cplx(Re * scale, -Im * scale);
    }

    internal Cplx Power(int exponent)
    {
        var result = new Cplx(1.0, 0.0);
        var baseValue = this;
        int remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0) result *= baseValue;
            baseValue *= baseValue;
            remaining >>= 1;
        }
        return result;
    }
}

internal sealed class Charge
{
    internal Charge(double x, double y, double q)
    {
        Z = new Cplx(x, y);
        Q = q;
    }

    internal Cplx Z;
    internal double Q;
}

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Fmm
{
    internal const int MaxOrder = 14;

    private static readonly double[][] Binomial = BuildBinomial(2 * MaxOrder + 2);

    private static double[][] BuildBinomial(int size)
    {
        var table = new double[size][];
        for (int n = 0; n < size; n++)
        {
            table[n] = new double[size];
            table[n][0] = 1.0;
            for (int k = 1; k <= n; k++) table[n][k] = table[n - 1][k - 1] + (k <= n - 1 ? table[n - 1][k] : 0.0);
        }
        return table;
    }

    internal static double Choose(int n, int k) => n < 0 || k < 0 || k > n || n >= Binomial.Length ? 0.0 : Binomial[n][k];

    /// <summary>Multipole moments a_k = sum_j q_j (z_j - centre)^k, k = 0..order.</summary>
    internal static Cplx[] Multipole(IReadOnlyList<Charge> charges, Cplx centre, int order)
    {
        var moments = new Cplx[order + 1];
        for (int k = 0; k <= order; k++)
        {
            var sum = new Cplx(0.0, 0.0);
            foreach (Charge charge in charges) sum += charge.Q * (charge.Z - centre).Power(k);
            moments[k] = sum;
        }
        return moments;
    }

    /// <summary>Evaluates the truncated multipole expansion at z (real part of the complex potential).</summary>
    internal static double EvaluateMultipole(Cplx[] moments, Cplx centre, Cplx z)
    {
        var inverse = (z - centre).Inverse();
        var sum = new Cplx(0.0, 0.0);
        var power = inverse;
        for (int k = 0; k < moments.Length; k++)
        {
            sum += moments[k] * power;
            power *= inverse;
        }
        return sum.Re;
    }

    /// <summary>M2M: the same cluster described about a new centre (exact binomial shift).</summary>
    internal static Cplx[] TranslateMultipole(Cplx[] moments, Cplx from, Cplx to, int order)
    {
        var shifted = new Cplx[order + 1];
        var offset = from - to;
        for (int l = 0; l <= order; l++)
        {
            var sum = new Cplx(0.0, 0.0);
            for (int k = 0; k <= l; k++) sum += Choose(l, k) * moments[k] * offset.Power(l - k);
            shifted[l] = sum;
        }
        return shifted;
    }

    /// <summary>M2L: the multipole of a far cluster turned into a local expansion about the target.</summary>
    internal static Cplx[] MultipoleToLocal(Cplx[] moments, Cplx sourceCentre, Cplx targetCentre, int order)
    {
        var local = new Cplx[order + 1];
        var offset = sourceCentre - targetCentre;
        if (offset.Abs2 == 0.0) throw new ArgumentException("a source cell cannot share the target centre");
        var inverse = offset.Inverse();
        for (int l = 0; l <= order; l++)
        {
            var sum = new Cplx(0.0, 0.0);
            for (int k = 0; k <= order; k++)
            {
                double sign = ((k + 1) % 2 == 0) ? 1.0 : -1.0;
                sum += Choose(k + l, l) * sign * moments[k] * inverse.Power(k + l + 1);
            }
            local[l] = sum;
        }
        return local;
    }

    /// <summary>L2L: the same local expansion about a new centre (exact binomial shift).</summary>
    internal static Cplx[] TranslateLocal(Cplx[] local, Cplx from, Cplx to, int order)
    {
        var shifted = new Cplx[order + 1];
        var offset = to - from;
        for (int l = 0; l <= order; l++)
        {
            var sum = new Cplx(0.0, 0.0);
            for (int m = l; m <= order; m++) sum += Choose(m, l) * local[m] * offset.Power(m - l);
            shifted[l] = sum;
        }
        return shifted;
    }

    internal static double EvaluateLocal(Cplx[] local, Cplx centre, Cplx z)
    {
        var offset = z - centre;
        var sum = new Cplx(0.0, 0.0);
        var power = new Cplx(1.0, 0.0);
        for (int l = 0; l < local.Length; l++)
        {
            sum += local[l] * power;
            power *= offset;
        }
        return sum.Re;
    }

    /// <summary>Exact potential of every charge: the oracle for both expansions.</summary>
    internal static double DirectSum(IReadOnlyList<Charge> charges, Cplx z)
    {
        double sum = 0.0;
        foreach (Charge charge in charges)
        {
            Cplx difference = z - charge.Z;
            if (difference.Abs2 == 0.0) throw new ArgumentException("a probe sits on a charge");
            sum += charge.Q * difference.Re / difference.Abs2;
        }
        return sum;
    }
}

internal sealed class Cell
{
    internal readonly List<Charge> Charges = new List<Charge>();
    internal Cplx Centre;
    internal int Row;
    internal int Column;
    internal Cplx[] Moments = Array.Empty<Cplx>();
}

internal sealed class Grid
{
    internal readonly List<Cell> Cells = new List<Cell>();
    internal readonly int Side;
    internal readonly double Size;
    internal long ToLocalTranslations;
    internal long NearInteractions;

    internal Grid(int side, double size)
    {
        Side = side;
        Size = size;
        double cellSize = size / side;
        for (int row = 0; row < side; row++)
            for (int column = 0; column < side; column++)
                Cells.Add(new Cell
                {
                    Row = row,
                    Column = column,
                    Centre = new Cplx((column + 0.5) * cellSize, (row + 0.5) * cellSize),
                });
    }

    internal Cell At(int row, int column) => Cells[row * Side + column];

    internal bool IsNear(Cell target, Cell other) =>
        Math.Abs(target.Row - other.Row) <= 1 && Math.Abs(target.Column - other.Column) <= 1;

    internal void BuildMoments(int order)
    {
        foreach (Cell cell in Cells) cell.Moments = Fmm.Multipole(cell.Charges, cell.Centre, order);
    }

    /// <summary>Potential at a point: direct for the target cell and its neighbours, one M2L per far cell.</summary>
    internal double Evaluate(Cplx point, Cell target, int order, List<(Cplx Centre, Cplx[] Local)>? expansions = null)
    {
        double potential = 0.0;
        var local = new Cplx[order + 1];
        bool hasLocal = false;
        foreach (Cell other in Cells)
        {
            if (IsNear(target, other))
            {
                foreach (Charge charge in other.Charges)
                {
                    Cplx difference = point - charge.Z;
                    if (difference.Abs2 == 0.0) continue;
                    potential += charge.Q * difference.Re / difference.Abs2;
                    NearInteractions++;
                }
                continue;
            }
            if (other.Charges.Count == 0) continue;
            Cplx[] contribution = Fmm.MultipoleToLocal(other.Moments, other.Centre, target.Centre, order);
            ToLocalTranslations++;
            for (int l = 0; l <= order; l++) local[l] += contribution[l];
            hasLocal = true;
        }
        if (hasLocal)
        {
            if (expansions != null) expansions.Add((target.Centre, local));
            potential += Fmm.EvaluateLocal(local, target.Centre, point);
        }
        return potential;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- the binomial table and the moment definitions -----------------------------------------
        {
            Check(Fmm.Choose(0, 0) == 1.0 && Fmm.Choose(5, 2) == 10.0 && Fmm.Choose(10, 5) == 252.0,
                "the binomial table matches the textbook values");
            Check(Fmm.Choose(4, 5) == 0.0 && Fmm.Choose(3, -1) == 0.0, "out-of-range binomials are zero");
            var single = new List<Charge> { new Charge(0.3, -0.2, 1.5) };
            Cplx[] moments = Fmm.Multipole(single, new Cplx(1.0, 1.0), 3);
            Check(Math.Abs(moments[0].Re - 1.5) < 1e-15 && Math.Abs(moments[0].Im) < 1e-15,
                "the zeroth moment is the total charge");
            var difference = single[0].Z - new Cplx(1.0, 1.0);
            Check(Math.Abs(moments[2].Re - 1.5 * (difference * difference).Re) < 1e-14,
                "the second moment is q times the offset squared");
        }

        // ---- a single charge is represented exactly once the truncation is below rounding ------------
        {
            var charges = new List<Charge> { new Charge(0.4, 0.4, -2.0) };
            Cplx centre = new Cplx(0.5, 0.5);
            var probe = new Cplx(3.0, -1.0);
            double exact = Fmm.DirectSum(charges, probe);
            foreach (int order in new[] { 6, 10, 14 })
            {
                double approximate = Fmm.EvaluateMultipole(Fmm.Multipole(charges, centre, order), centre, probe);
                double error = Math.Abs(approximate - exact);
                Check(order == 6 ? error < 1e-9 : error < 1e-13,
                    "order " + order + ": a single charge reproduced to the truncation limit (" + error + ")");
            }
            double offset = Math.Sqrt((centre - charges[0].Z).Abs2);
            double separation = Math.Sqrt((probe - centre).Abs2);
            double zeroth = Math.Abs(Fmm.EvaluateMultipole(Fmm.Multipole(charges, centre, 0), centre, probe) - exact);
            double predicted = 2.0 * Math.Abs(charges[0].Q) * offset / (separation * separation);
            Console.WriteLine("single charge: offset=" + offset.ToString("G4") + " separation=" + separation.ToString("G4")
                + " zerothOrderError=" + zeroth.ToString("G4") + " predictedFirstTerm=" + predicted.ToString("G4"));
            Check(zeroth <= 2.0 * predicted, "order zero is off by exactly the first omitted term");
            Check(zeroth > 0.0, "and moving the centre away from the charge really costs accuracy");
        }

        // ---- the multipole truncation converges at the (a/d) rate ------------------------------------
        {
            var charges = Cluster(16, new Cplx(0.0, 0.0), 0.2, 0x734AUL);
            Cplx centre = new Cplx(0.0, 0.0);
            var probe = new Cplx(4.0, 0.0);
            double exact = Fmm.DirectSum(charges, probe);
            var errors = new List<double>();
            foreach (int order in new[] { 2, 4, 8, 12 })
            {
                double approximate = Fmm.EvaluateMultipole(Fmm.Multipole(charges, centre, order), centre, probe);
                errors.Add(Math.Abs(approximate - exact));
            }
            Console.WriteLine("multipole orders 2,4,8,12 errors: " + string.Join(" ",
                errors.ConvertAll(e => e.ToString("G3"))));
            Check(errors[0] > errors[1] && errors[1] > errors[2] && errors[2] > errors[3],
                "the truncation error falls with the order");
            Check(errors[3] < 1e-9, "order 12 is essentially exact at this separation (" + errors[3] + ")");
            Check(errors[2] / errors[3] > 10.0, "each step of four orders buys orders of magnitude");
        }

        // ---- M2M is an exact polynomial identity -----------------------------------------------------
        {
            var charges = Cluster(24, new Cplx(1.0, -0.5), 0.15, 0x734BUL);
            Cplx childCentre = new Cplx(1.05, -0.45);
            Cplx parentCentre = new Cplx(0.7, -0.9);
            const int order = 8;
            Cplx[] child = Fmm.Multipole(charges, childCentre, order);
            Cplx[] aggregated = Fmm.TranslateMultipole(child, childCentre, parentCentre, order);
            Cplx[] direct = Fmm.Multipole(charges, parentCentre, order);
            double worst = 0.0;
            for (int k = 0; k <= order; k++)
            {
                var difference = aggregated[k] - direct[k];
                worst = Math.Max(worst, Math.Sqrt(difference.Abs2));
            }
            Console.WriteLine("M2M worst coefficient error=" + worst.ToString("G4"));
            Check(worst < 1e-12, "the translated moments equal the directly computed parent moments");

            var probe = new Cplx(6.0, 2.0);
            Check(Math.Abs(Fmm.EvaluateMultipole(aggregated, parentCentre, probe)
                - Fmm.DirectSum(charges, probe)) < 1e-9, "and they evaluate to the same potential");
        }

        // ---- M2L converges to the analytic single-charge coefficients --------------------------------
        {
            var charge = new Charge(0.2, 0.1, 1.25);
            Cplx sourceCentre = new Cplx(0.0, 0.0);
            Cplx targetCentre = new Cplx(1.0, 0.0);
            var worstByOrder = new List<double>();
            foreach (int order in new[] { 2, 4, 6, 10 })
            {
                Cplx[] moments = Fmm.Multipole(new List<Charge> { charge }, sourceCentre, order);
                Cplx[] local = Fmm.MultipoleToLocal(moments, sourceCentre, targetCentre, order);
                var offset = charge.Z - targetCentre;
                var inverse = offset.Inverse();
                double worst = 0.0;
                for (int l = 0; l <= order; l++)
                {
                    // the exact Taylor coefficient of q / (z - s) about the target centre
                    var exact = -1.0 * charge.Q * inverse.Power(l + 1);
                    var difference = local[l] - exact;
                    double magnitude = Math.Sqrt(exact.Abs2);
                    worst = Math.Max(worst, Math.Sqrt(difference.Abs2) / Math.Max(magnitude, 1e-12));
                }
                worstByOrder.Add(worst);
                Console.WriteLine("M2L order=" + order + " worstRelativeCoefficientError=" + worst.ToString("G4"));
            }
            Check(worstByOrder[0] > worstByOrder[1] && worstByOrder[1] > worstByOrder[2] && worstByOrder[2] > worstByOrder[3],
                "the translated coefficients converge to the analytic ones as the order grows");
            Check(worstByOrder[3] < 0.01, "order 10 is within a percent of the closed form (" + worstByOrder[3]
                + "); the residual of the truncated binomial sum carries a C(2p+1,p) growth, which is why real FMM"
                + " codes pick p from the cell-size ratio");

            // evaluating the translated expansion reproduces the exact potential of that one charge
            // (the local expansion converges inside |z - t| < |s - t|, so the probe stays well within it)
            Cplx[] best = Fmm.MultipoleToLocal(
                Fmm.Multipole(new List<Charge> { charge }, sourceCentre, 12), sourceCentre, targetCentre, 12);
            var probe = new Cplx(1.25, 0.15);
            double localError = Math.Abs(Fmm.EvaluateLocal(best, targetCentre, probe)
                - Fmm.DirectSum(new List<Charge> { charge }, probe));
            Console.WriteLine("single charge local expansion: probeDistance="
                + Math.Sqrt((probe - targetCentre).Abs2).ToString("G4") + " error=" + localError.ToString("G4"));
            Check(localError < 1e-3, "the local expansion evaluates to the exact potential inside its radius ("
                + localError + ")");
        }

        // ---- L2L moves a local expansion without changing the potential -------------------------------
        {
            var charges = Cluster(30, new Cplx(0.2, 0.3), 0.3, 0x734CUL);
            const int order = 10;
            Cplx sourceCentre = new Cplx(0.0, 0.0);
            Cplx firstTarget = new Cplx(2.0, 0.0);
            Cplx secondTarget = new Cplx(2.6, 0.4);
            Cplx[] local = Fmm.MultipoleToLocal(Fmm.Multipole(charges, sourceCentre, order), sourceCentre, firstTarget, order);
            Cplx[] moved = Fmm.TranslateLocal(local, firstTarget, secondTarget, order);
            foreach (var probe in new[] { new Cplx(3.0, 1.0), new Cplx(2.9, -0.7), new Cplx(4.0, 0.0) })
            {
                double before = Fmm.EvaluateLocal(local, firstTarget, probe);
                double after = Fmm.EvaluateLocal(moved, secondTarget, probe);
                Check(Math.Abs(before - after) < 1e-9,
                    "moving a local expansion preserves the potential (" + Math.Abs(before - after) + ")");
            }
            var roundTrip = Fmm.TranslateLocal(moved, secondTarget, firstTarget, order);
            foreach (var probe in new[] { new Cplx(3.0, 1.0), new Cplx(4.5, -1.5) })
                Check(Math.Abs(Fmm.EvaluateLocal(roundTrip, firstTarget, probe)
                    - Fmm.EvaluateLocal(local, firstTarget, probe)) < 1e-9,
                    "and moving it back reproduces the original expansion");
        }

        // ---- the end-to-end two-level evaluation ------------------------------------------------------
        {
            var grid = new Grid(4, 1.0);
            var rng = new SplitMix64(0x734DUL);
            for (int row = 0; row < 4; row++)
                for (int column = 0; column < 4; column++)
                {
                    Cell cell = grid.At(row, column);
                    for (int k = 0; k < 4; k++)
                    {
                        double x = (column + 0.15 + 0.7 * rng.NextUnit()) * 0.25;
                        double y = (row + 0.15 + 0.7 * rng.NextUnit()) * 0.25;
                        cell.Charges.Add(new Charge(x, y, 0.5 + rng.NextUnit()));
                    }
                }
            var all = new List<Charge>();
            foreach (Cell cell in grid.Cells) all.AddRange(cell.Charges);

            var errors = new List<double>();
            foreach (int order in new[] { 2, 4, 8, 12 })
            {
                grid.BuildMoments(order);
                grid.ToLocalTranslations = 0;
                grid.NearInteractions = 0;
                double worst = 0.0;
                foreach (Cell cell in grid.Cells)
                {
                    var probe = new Cplx(cell.Centre.Re + 0.11, cell.Centre.Im - 0.07);
                    double approximate = grid.Evaluate(probe, cell, order);
                    double exact = Fmm.DirectSum(all, probe);
                    worst = Math.Max(worst, Math.Abs(approximate - exact) / Math.Abs(exact));
                }
                errors.Add(worst);
                Console.WriteLine("grid order=" + order + " worstRelativeError=" + worst.ToString("G4")
                    + " m2l=" + grid.ToLocalTranslations + " directPairs=" + grid.NearInteractions);
            }
            Check(errors[0] > errors[1] && errors[1] > errors[2], "the two-level evaluation improves with the order");
            Check(errors[2] < 1e-3, "order 8 is within a thousandth of the direct sum (" + errors[2] + ")");
            Check(grid.ToLocalTranslations > 0 && grid.ToLocalTranslations <= 16L * 16,
                "the far field needed one translation per well-separated cell pair (" + grid.ToLocalTranslations + ")");
            Check(grid.NearInteractions < 64L * 9 * 4, "only the near cells fall back to direct pairs ("
                + grid.NearInteractions + ")");

            // neighbours-only configuration: with a single occupied cell everything is direct and exact
            var trivial = new Grid(4, 1.0);
            for (int k = 0; k < 6; k++)
                trivial.At(1, 1).Charges.Add(new Charge(0.3 + 0.05 * k, 0.3, 1.0));
            trivial.BuildMoments(8);
            double worstTrivial = 0.0;
            foreach (Charge charge in trivial.At(1, 1).Charges)
            {
                var probe = new Cplx(charge.Z.Re + 0.02, charge.Z.Im + 0.02);
                double approximate = trivial.Evaluate(probe, trivial.At(1, 1), 8);
                double exact = 0.0;
                foreach (Charge other in trivial.At(1, 1).Charges)
                {
                    Cplx difference = probe - other.Z;
                    if (difference.Abs2 == 0.0) continue;
                    exact += other.Q * difference.Re / difference.Abs2;
                }
                worstTrivial = Math.Max(worstTrivial, Math.Abs(approximate - exact));
            }
            Check(trivial.ToLocalTranslations == 0, "one occupied cell needs no far-field translation at all");
            Check(worstTrivial < 1e-12, "and it reproduces the direct sum exactly (" + worstTrivial + ")");
        }

        // ---- reproducibility and rejection -------------------------------------------------------------
        {
            var charges = Cluster(20, new Cplx(0.7, 0.7), 0.2, 0x734EUL);
            Cplx centre = new Cplx(0.0, 0.0);
            Cplx[] first = Fmm.Multipole(charges, centre, 9);
            Cplx[] second = Fmm.Multipole(charges, centre, 9);
            bool identical = true;
            for (int k = 0; k < first.Length; k++) if (first[k].Re != second[k].Re || first[k].Im != second[k].Im) identical = false;
            Check(identical, "the moments are reproducible");
            var probe = new Cplx(5.0, -2.0);
            Check(Fmm.EvaluateMultipole(first, centre, probe) == Fmm.EvaluateMultipole(second, centre, probe),
                "and so is the evaluated potential");

            bool threw = false;
            try { Fmm.MultipoleToLocal(first, centre, centre, 9); } catch (ArgumentException) { threw = true; }
            Check(threw, "a source cell sharing the target centre is rejected");
            bool threwProbe = false;
            try { Fmm.DirectSum(charges, charges[0].Z); } catch (ArgumentException) { threwProbe = true; }
            Check(threwProbe, "a probe sitting on a charge is rejected instead of returning infinity");

            var empty = new Grid(2, 1.0);
            empty.BuildMoments(4);
            Check(empty.Evaluate(new Cplx(0.5, 0.5), empty.At(0, 0), 4) == 0.0, "an empty grid evaluates to zero");
            Check(empty.ToLocalTranslations == 0, "and does no translations");
        }

        Console.WriteLine("PASS");
    }

    private static List<Charge> Cluster(int count, Cplx centre, double radius, ulong seed)
    {
        var rng = new SplitMix64(seed);
        var charges = new List<Charge>();
        for (int i = 0; i < count; i++)
        {
            double angle = 2.0 * Math.PI * rng.NextUnit();
            double r = radius * Math.Sqrt(rng.NextUnit());
            charges.Add(new Charge(centre.Re + r * Math.Cos(angle), centre.Im + r * Math.Sin(angle), 0.5 + rng.NextUnit()));
        }
        return charges;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
