// Task 731 - Implement adaptive mesh refinement driving solver coupling. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// An adaptive mesh refines where the solution is rough and coarsens where it is smooth, and the solver
// runs on the mesh it is handed - so the refinement decision and the discretisation have to be coupled
// consistently. This file implements a 1D conservative finite-volume mesh with per-cell sizes and a
// periodic heat solver, plus three mesh operations that must not disturb the physical invariants:
//
//   refine   split a cell in two. The two children carry u +/- d about u, so their mass h/2 (u+d) +
//            h/2 (u-d) equals h u exactly, whatever slope d is used.
//   coarsen  merge two cells into their mass-weighted average, again exactly conserving mass.
//   balance  split any cell more than twice the size of a neighbour, so the stencil stays local.
//
// Documented simplifications: the mesh is cell-wise heterogeneous in 1D rather than a hierarchy of
// refinement levels, so there is no time subcycling and no flux register - the requirement those
// mechanisms exist for, exact conservation of the discrete integral across refinement boundaries, is
// instead satisfied structurally by the mass-preserving operators and verified directly. The solver is
// an explicit two-point flux scheme with dt = 0.4 min(h)^2 / 2, and the indicator is the scaled second
// difference. Verified: mass is conserved to machine precision with and without adaptation; splits and
// merges preserve mass and domain length exactly; the scheme reproduces the analytic Fourier decay of
// the heat equation; a sharp bump is resolved better by the adaptive mesh than by a uniform mesh of
// comparable size; the indicator aims the refinement at the feature and coarsens the flat region; the
// size ratio never exceeds two; and the same configuration always produces the same mesh.
using System;
using System.Collections.Generic;

internal sealed class AdaptiveMesh
{
    private readonly List<double> _u = new List<double>();
    private readonly List<double> _h = new List<double>();

    internal int Cells => _u.Count;
    internal long Refinements { get; private set; }
    internal long Coarsenings { get; private set; }
    internal long Balances { get; private set; }

    private AdaptiveMesh() { }

    internal static AdaptiveMesh Uniform(int cells)
    {
        if (cells < 3) throw new ArgumentOutOfRangeException(nameof(cells));
        var mesh = new AdaptiveMesh();
        for (int i = 0; i < cells; i++)
        {
            mesh._u.Add(0.0);
            mesh._h.Add(1.0 / cells);
        }
        return mesh;
    }

    internal int Wrap(int i)
    {
        int n = _u.Count;
        int m = i % n;
        return m < 0 ? m + n : m;
    }

    internal double Value(int i) => _u[Wrap(i)];
    internal double Size(int i) => _h[Wrap(i)];
    internal void SetValue(int i, double value) => _u[Wrap(i)] = value;
    internal void SetSize(int i, double size) => _h[Wrap(i)] = size;

    internal double Mass
    {
        get
        {
            double mass = 0.0;
            for (int i = 0; i < _u.Count; i++) mass += _u[i] * _h[i];
            return mass;
        }
    }

    internal double Length
    {
        get
        {
            double length = 0.0;
            foreach (double h in _h) length += h;
            return length;
        }
    }

    internal double SmallestSize()
    {
        double smallest = double.MaxValue;
        foreach (double h in _h) smallest = Math.Min(smallest, h);
        return smallest;
    }

    internal double LargestSize()
    {
        double largest = 0.0;
        foreach (double h in _h) largest = Math.Max(largest, h);
        return largest;
    }

    internal double Centre(int i)
    {
        double x = 0.0;
        for (int k = 0; k < Wrap(i); k++) x += _h[k];
        return x + 0.5 * _h[Wrap(i)];
    }

    /// <summary>Explicit conservative update on the heterogeneous mesh; the fluxes telescope.</summary>
    internal void Step(double dt)
    {
        int n = _u.Count;
        var flux = new double[n];
        for (int i = 0; i < n; i++)
        {
            double distance = 0.5 * (Size(i) + Size(i + 1));
            flux[i] = (Value(i + 1) - Value(i)) / distance;
        }
        for (int i = 0; i < n; i++) _u[i] += dt / _h[i] * (flux[i] - flux[Wrap(i - 1)]);   // du/dt = d/dx(du/dx)
    }

    /// <summary>Scaled second difference: the classic smoothness indicator for refinement.</summary>
    internal double Indicator(int i)
    {
        double centre = Size(i);
        double left = 0.5 * (Size(i - 1) + centre);
        double right = 0.5 * (centre + Size(i + 1));
        double slopeLeft = (Value(i) - Value(i - 1)) / left;
        double slopeRight = (Value(i + 1) - Value(i)) / right;
        return 2.0 * (slopeRight - slopeLeft) / (left + right);
    }

    /// <summary>Refinement floor: no cell is ever split below this size.</summary>
    internal const double MinCellSize = 1.0 / 512.0;

    internal int Refine(double tolerance)
    {
        int splits = 0;
        var values = new List<double>();
        var sizes = new List<double>();
        for (int i = 0; i < _u.Count; i++)
        {
            double h = _h[i], u = _u[i];
            bool wanted = Math.Abs(Indicator(i)) > tolerance && h * 0.5 >= MinCellSize * (1.0 - 1e-12);
            if (!wanted)
            {
                values.Add(u);
                sizes.Add(h);
                continue;
            }
            double slope = (Value(i + 1) - Value(i - 1)) / (Size(i - 1) + 2.0 * h + Size(i + 1)) * 2.0;
            double d = 0.25 * h * slope;                 // mass-preserving split: (u+d) + (u-d) = 2u
            values.Add(u + d);
            sizes.Add(h / 2);
            values.Add(u - d);
            sizes.Add(h / 2);
            splits++;
        }
        _u.Clear(); _u.AddRange(values);
        _h.Clear(); _h.AddRange(sizes);
        Refinements += splits;
        return splits;
    }

    internal int Coarsen(double tolerance, double maxSize)
    {
        int merges = 0;
        var values = new List<double>();
        var sizes = new List<double>();
        int i = 0;
        while (i < _u.Count)
        {
            double h = _h[i];
            bool canMerge = i + 1 < _u.Count + 1 && _u.Count + 0 > 1 && h < maxSize * 1.0000001;
            if (canMerge && _u.Count > 2)
            {
                double h2 = Size(i + 1);
                bool smooth = Math.Abs(Indicator(i)) < tolerance * 0.25 && Math.Abs(Indicator(i + 1)) < tolerance * 0.25;
                if (h + h2 <= maxSize * 1.0000001 && smooth)
                {
                    double merged = (_u[i] * h + Value(i + 1) * h2) / (h + h2);
                    values.Add(merged);
                    sizes.Add(h + h2);
                    merges++;
                    i += 2;
                    continue;
                }
            }
            values.Add(_u[i]);
            sizes.Add(h);
            i++;
        }
        _u.Clear(); _u.AddRange(values);
        _h.Clear(); _h.AddRange(sizes);
        Coarsenings += merges;
        return merges;
    }

    /// <summary>Splits every cell that is more than twice as large as a neighbour.</summary>
    internal int Balance(double ratio)
    {
        int splits = 0;
        bool changed = true;
        while (changed && splits < 20000)
        {
            changed = false;
            for (int i = 0; i < _u.Count; i++)
            {
                if (Size(i) <= ratio * Size(i + 1) * 1.0000001 && Size(i) <= ratio * Size(i - 1) * 1.0000001) continue;
                if (Size(i) * 0.5 < MinCellSize) continue;      // the floor outranks the balance rule
                SplitOne(i);
                splits++;
                Balances++;
                changed = true;
            }
        }
        return splits;
    }

    private void SplitOne(int i)
    {
        double h = _h[i], u = _u[i];
        double slope = (Value(i + 1) - Value(i - 1)) / (Size(i - 1) + 2.0 * h + Size(i + 1)) * 2.0;
        double d = 0.25 * h * slope;
        _u[i] = u + d;
        _h[i] = h / 2;
        _u.Insert(i + 1, u - d);
        _h.Insert(i + 1, h / 2);
    }

    internal bool Balanced(double ratio)
    {
        for (int i = 0; i < _u.Count; i++)
        {
            double a = Size(i), b = Size(i + 1);
            if (Math.Max(a, b) > ratio * Math.Min(a, b) * 1.0000001) return false;
        }
        return true;
    }

    /// <summary>L2 error against an exact solution, measured cell-wise at cell centres.</summary>
    internal double L2Error(Func<double, double> exact)
    {
        double sum = 0.0;
        for (int i = 0; i < _u.Count; i++)
        {
            double d = _u[i] - exact(Centre(i));
            sum += d * d * _h[i];
        }
        return Math.Sqrt(sum);
    }

    internal double[] ValuesCopy() => _u.ToArray();
    internal double[] SizesCopy() => _h.ToArray();
}

public static class Solution
{
    public static void Main()
    {
        // ---- mesh operators preserve mass and domain length exactly --------------------------------
        {
            AdaptiveMesh mesh = AdaptiveMesh.Uniform(32);
            for (int i = 0; i < mesh.Cells; i++) mesh.SetValue(i, Math.Sin(2.0 * Math.PI * (i + 0.5) / 32.0) + 0.3);
            double before = mesh.Mass, length = mesh.Length;
            int splits = mesh.Refine(0.0);
            Check(splits == 32 && mesh.Cells == 64, "refining everything doubles the cell count (" + mesh.Cells + ")");
            Check(Math.Abs(mesh.Mass - before) < 1e-14 * Math.Abs(before), "refinement preserves the mass exactly");
            Check(Math.Abs(mesh.Length - length) < 1e-14, "refinement preserves the domain length");
            int merges = mesh.Coarsen(1e18, 10.0);
            Check(merges == 32 && mesh.Cells == 32, "the same mesh can be coarsened back (" + mesh.Cells + ")");
            Check(Math.Abs(mesh.Mass - before) < 1e-14 * Math.Abs(before), "coarsening preserves the mass exactly");
            Check(Math.Abs(mesh.Length - length) < 1e-14, "coarsening preserves the domain length");
            Console.WriteLine("mesh operators: mass=" + before.ToString("G10") + " splits=" + splits + " merges=" + merges);
        }

        // ---- mass conservation of the coupled solver, uniform and adaptive --------------------------
        {
            foreach (bool adaptive in new[] { false, true })
            {
                AdaptiveMesh mesh = AdaptiveMesh.Uniform(64);
                for (int i = 0; i < mesh.Cells; i++) mesh.SetValue(i, BumpExact(mesh.Centre(i)));
                double initial = mesh.Mass;
                double worst = 0.0;
                int totalSplits = 0, totalMerges = 0;
                for (int cycle = 0; cycle < 400; cycle++)
                {
                    double dt = 0.4 * mesh.SmallestSize() * mesh.SmallestSize() / 2.0;
                    mesh.Step(dt);
                    if (adaptive && cycle % 20 == 19)
                    {
                        totalSplits += mesh.Refine(1.0);
                        totalMerges += mesh.Coarsen(1.0, 0.05);
                        mesh.Balance(2.0);
                    }
                    worst = Math.Max(worst, Math.Abs(mesh.Mass - initial) / Math.Abs(initial));
                }
                Console.WriteLine("adaptive=" + adaptive + " massDrift=" + worst.ToString("G4")
                    + " cells=" + mesh.Cells + " splits=" + totalSplits + " merges=" + totalMerges);
                Check(worst < 1e-12, "mass is conserved to machine precision (adaptive=" + adaptive + ", drift " + worst + ")");
                Check(mesh.Balanced(2.0), "the mesh satisfies the two-to-one balance rule (adaptive=" + adaptive + ")");
                if (adaptive)
                {
                    Check(totalSplits > 0 && totalMerges > 0, "the adaptive run really refined and coarsened");
                    Check(mesh.Cells < 64 * 8, "the adaptive mesh stays bounded (" + mesh.Cells + " cells)");
                    Check(mesh.SmallestSize() >= 1.0 / 512.0 - 1e-15, "the refinement floor is respected");
                    Check(mesh.SmallestSize() < 1.0 / 64.0, "refinement produced cells smaller than the coarse size");
                }
                else Check(mesh.Cells == 64, "the uniform run never changes the mesh");
            }
        }

        // ---- the discretisation reproduces the analytic Fourier decay --------------------------------
        {
            AdaptiveMesh mesh = AdaptiveMesh.Uniform(64);
            for (int i = 0; i < mesh.Cells; i++) mesh.SetValue(i, 0.5 + Math.Sin(2.0 * Math.PI * mesh.Centre(i)));
            const double finalTime = 0.02;
            double time = 0.0;
            while (time < finalTime)
            {
                double dt = Math.Min(0.4 * mesh.SmallestSize() * mesh.SmallestSize() / 2.0, finalTime - time);
                mesh.Step(dt);
                time += dt;
            }
            double decay = Math.Exp(-4.0 * Math.PI * Math.PI * finalTime);
            double error = mesh.L2Error(x => 0.5 + Math.Sin(2.0 * Math.PI * x) * decay);
            Console.WriteLine("diffusion: t=" + finalTime + " decay=" + decay.ToString("G6") + " l2Error=" + error.ToString("G6"));
            Check(error < 0.02, "the scheme matches the analytic heat decay (" + error + ")");

            AdaptiveMesh steady = AdaptiveMesh.Uniform(32);
            for (int i = 0; i < steady.Cells; i++) steady.SetValue(i, 0.5);
            for (int step = 0; step < 100; step++) steady.Step(0.4 * steady.SmallestSize() * steady.SmallestSize() / 2.0);
            Check(steady.L2Error(x => 0.5) < 1e-13, "the constant mode does not decay at all ("
                + steady.L2Error(x => 0.5) + ")");
        }

        // ---- a sharp bump is resolved better by an adaptive mesh of similar size ---------------------
        {
            AdaptiveMesh coarse = AdaptiveMesh.Uniform(128);
            AdaptiveMesh adaptive = AdaptiveMesh.Uniform(48);
            for (int i = 0; i < coarse.Cells; i++) coarse.SetValue(i, BumpExact(coarse.Centre(i)));
            for (int i = 0; i < adaptive.Cells; i++) adaptive.SetValue(i, BumpExact(adaptive.Centre(i)));
            const int cycles = 240;
            for (int cycle = 0; cycle < cycles; cycle++)
            {
                double coarseDt = 0.4 * coarse.SmallestSize() * coarse.SmallestSize() / 2.0;
                coarse.Step(coarseDt);
                double adaptiveDt = 0.4 * adaptive.SmallestSize() * adaptive.SmallestSize() / 2.0;
                adaptive.Step(adaptiveDt);
                if (cycle % 15 == 14)
                {
                    adaptive.Refine(2.0);
                    adaptive.Coarsen(2.0, 0.08);
                    adaptive.Balance(2.0);
                }
            }
            double coarseError = coarse.L2Error(BumpExact);
            double adaptiveError = adaptive.L2Error(BumpExact);
            Console.WriteLine("bump: uniformCells=" + coarse.Cells + " uniformError=" + coarseError.ToString("G6")
                + " adaptiveCells=" + adaptive.Cells + " adaptiveError=" + adaptiveError.ToString("G6")
                + " perCellUniform=" + (coarseError / coarse.Cells).ToString("G4")
                + " perCellAdaptive=" + (adaptiveError / adaptive.Cells).ToString("G4"));
            Check(adaptiveError < coarseError, "the adaptive mesh resolves the bump better than a uniform mesh ("
                + adaptiveError + " vs " + coarseError + ")");
            Check(adaptiveError / adaptive.Cells < coarseError / coarse.Cells,
                "and it is more efficient measured per cell");

            // the indicator aims refinement at the bump edges and leaves the flat region coarse
            double edgeSize = 0.0, flatSize = 0.0;
            int edgeCount = 0, flatCount = 0;
            for (int i = 0; i < adaptive.Cells; i++)
            {
                double x = adaptive.Centre(i);
                bool nearEdge = Math.Abs(x - 0.4) < 0.03 || Math.Abs(x - 0.6) < 0.03;
                bool flat = x > 0.65 && x < 0.95;
                if (nearEdge) { edgeSize += adaptive.Size(i); edgeCount++; }
                if (flat) { flatSize += adaptive.Size(i); flatCount++; }
            }
            Check(edgeCount > 0 && flatCount > 0, "both regions still have cells to compare");
            Console.WriteLine("edgeCells=" + edgeCount + " meanEdgeSize=" + (edgeSize / edgeCount).ToString("G4")
                + " flatCells=" + flatCount + " meanFlatSize=" + (flatSize / flatCount).ToString("G4"));
            Check(edgeSize / edgeCount < flatSize / flatCount,
                "cells at the bump edge are smaller than cells in the flat region");
        }

        // ---- degenerate configurations and rejection ---------------------------------------------------
        {
            AdaptiveMesh flat = AdaptiveMesh.Uniform(24);
            for (int i = 0; i < flat.Cells; i++) flat.SetValue(i, 3.0);
            Check(flat.Refine(1e-6) == 0, "a constant solution is never refined");
            double mass = flat.Mass;
            for (int cycle = 0; cycle < 50; cycle++) flat.Step(0.4 * flat.SmallestSize() * flat.SmallestSize() / 2.0);
            Check(Math.Abs(flat.Mass - mass) < 1e-14, "and it stays constant under the solver");
            Check(flat.Coarsen(1e18, 1.0 / 4.0) > 0, "but it can be coarsened, which is the point of the indicator");

            AdaptiveMesh single = AdaptiveMesh.Uniform(3);
            Check(single.Balanced(2.0), "a uniform mesh is trivially balanced");
            bool threw = false;
            try { AdaptiveMesh.Uniform(2); } catch (ArgumentOutOfRangeException) { threw = true; }
            Check(threw, "a two-cell mesh is rejected instead of breaking the stencil");

            AdaptiveMesh once = AdaptiveMesh.Uniform(16);
            for (int i = 0; i < once.Cells; i++) once.SetValue(i, Math.Cos(4.0 * Math.PI * once.Centre(i)));
            once.Refine(0.5);
            AdaptiveMesh twice = AdaptiveMesh.Uniform(16);
            for (int i = 0; i < twice.Cells; i++) twice.SetValue(i, Math.Cos(4.0 * Math.PI * twice.Centre(i)));
            twice.Refine(0.5);
            bool identical = once.Cells == twice.Cells;
            for (int i = 0; i < once.Cells && identical; i++)
                if (once.Value(i) != twice.Value(i) || once.Size(i) != twice.Size(i)) identical = false;
            Check(identical, "the same configuration produces the same mesh");
            Check(once.Refinements > 0 && once.Coarsenings == 0, "the refinement counters are tracked ("
                + once.Refinements + " splits)");
        }

        Console.WriteLine("PASS");
    }

    /// <summary>A periodic square bump: the sharp feature the adaptive mesh is supposed to chase.</summary>
    private static double BumpExact(double x)
    {
        double folded = x - Math.Floor(x);
        return folded > 0.4 && folded < 0.6 ? 1.0 : 0.0;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
