// Task 952 - Implement autograder that runs tests, measures complexity, and reports regression.
// (Tier: Master, Ch 11 - Testing)
//
// The autograder runs a submission's unit-test records, estimates its asymptotic cost from operation
// counts sampled at growing input sizes, and diffs the measured counts against a stored baseline.
// Complexity is estimated by fitting a constant through the origin (least squares) for every candidate
// shape in a catalogue, then keeping the shape with the smallest normalised residual - which also
// recovers the hidden constant exactly when the counts really were generated from that shape.
//
// Verified: shape and constant recovery for counts generated exactly from each catalogue entry (the
// independent oracle), robustness to a 20% outlier, the verdict precedence compile error > test fail >
// pass, regression reporting at +-tolerance, and the two-point boundary that must refuse to classify.
// Simplification: tests are in-memory pass/fail records and "compiling" is a flag; no process is run.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Shape
    {
        public readonly string Name;
        public readonly Func<int, double> F;

        public Shape(string name, Func<int, double> f)
        {
            Name = name;
            F = f;
        }
    }

    // Ordered by growth: the strict residual comparison keeps the simpler shape on a tie.
    private static readonly Shape[] Catalogue =
    {
        new Shape("O(1)", n => 1.0),
        new Shape("O(log n)", n => Math.Log2(n)),
        new Shape("O(n)", n => n),
        new Shape("O(n log n)", n => n * Math.Log2(n)),
        new Shape("O(n^2)", n => (double)n * n),
        new Shape("O(n^3)", n => (double)n * n * n),
    };

    private sealed class Fit
    {
        public string Shape = "insufficient data";
        public double Constant;
        public double Residual;

        public bool IsEmpty => Shape == "insufficient data";
    }

    /** Least-squares fit of `ops ~ c * f(n)` through the origin for each shape; smallest residual wins. */
    private static Fit Classify(int[] sizes, long[] ops)
    {
        if (sizes.Length < 3 || sizes.Length != ops.Length) return new Fit();
        Fit best = new Fit();
        double bestResidual = double.MaxValue;
        foreach (Shape shape in Catalogue)
        {
            double num = 0.0, den = 0.0, energy = 0.0;
            for (int i = 0; i < sizes.Length; i++)
            {
                double f = shape.F(sizes[i]);
                double y = ops[i];
                num += f * y;
                den += f * f;
                energy += y * y;
            }
            if (den <= 0.0) continue;
            double c = num / den;
            double sse = 0.0;
            for (int i = 0; i < sizes.Length; i++)
            {
                double d = ops[i] - c * shape.F(sizes[i]);
                sse += d * d;
            }
            double residual = energy > 0.0 ? sse / energy : 0.0;
            if (residual < bestResidual - 1e-12)
            {
                bestResidual = residual;
                best = new Fit { Shape = shape.Name, Constant = c, Residual = residual };
            }
        }
        return best;
    }

    private sealed class Submission
    {
        public readonly string Id;
        public readonly bool Compiles;
        public readonly List<(string Name, bool Passed)> Tests = new List<(string, bool)>();

        public Submission(string id, bool compiles)
        {
            Id = id;
            Compiles = compiles;
        }
    }

    private static string Verdict(Submission submission)
    {
        if (!submission.Compiles) return "COMPILE_ERROR";
        foreach ((string _, bool passed) in submission.Tests)
        {
            if (!passed) return "FAIL";
        }
        return "PASS";
    }

    private static List<string> Regressions(int[] sizes, long[] candidate, long[] baseline, double tolerance)
    {
        var findings = new List<string>();
        int n = Math.Min(sizes.Length, Math.Min(candidate.Length, baseline.Length));
        for (int i = 0; i < n; i++)
        {
            if (baseline[i] <= 0) continue;
            double ratio = candidate[i] / (double)baseline[i];
            if (ratio > 1.0 + tolerance)
                findings.Add("size=" + sizes[i].ToString(CultureInfo.InvariantCulture) + " +" +
                    ((ratio - 1.0) * 100.0).ToString("F1", CultureInfo.InvariantCulture) + "%");
        }
        return findings;
    }

    private static long[] NoisyOps(ulong seed, int[] sizes, Func<int, double> exact)
    {
        _state = seed;
        var ops = new long[sizes.Length];
        for (int i = 0; i < sizes.Length; i++)
            ops[i] = (long)Math.Round(exact(sizes[i]) * (0.9 + 0.2 * NextUnit()));
        return ops;
    }

    public static void Main()
    {
        int[] sizes = { 64, 128, 256, 512, 1024 };
        const double constant = 4.0;

        // Oracle 1: counts generated exactly from c*f(n) must be classified back to f and recover c.
        int recovered = 0;
        foreach (Shape shape in Catalogue)
        {
            var ops = new long[sizes.Length];
            for (int i = 0; i < sizes.Length; i++) ops[i] = (long)Math.Round(constant * shape.F(sizes[i]));
            Fit fit = Classify(sizes, ops);
            Check(fit.Shape == shape.Name, "exact " + shape.Name + " counts classify as " + shape.Name + " (got " + fit.Shape + ")");
            Check(Math.Abs(fit.Constant - constant) <= 1e-9,
                "the hidden constant of " + shape.Name + " is recovered (" + fit.Constant.ToString("F6", CultureInfo.InvariantCulture) + ")");
            if (!fit.IsEmpty) recovered++;
        }
        Check(recovered == Catalogue.Length, "every catalogue shape is reachable, none is shadowed by a neighbour");

        // Oracle 2: the residual of a perfect fit is essentially zero, so the classifier is not luck.
        var exactQuadratic = new long[sizes.Length];
        for (int i = 0; i < sizes.Length; i++) exactQuadratic[i] = 4L * sizes[i] * sizes[i];
        Check(Classify(sizes, exactQuadratic).Residual < 1e-12, "an exact O(n^2) series fits with a residual below 1e-12");

        var outlier = (long[])exactQuadratic.Clone();
        outlier[2] = outlier[2] * 12 / 10;
        Check(Classify(sizes, outlier).Shape == "O(n^2)", "a single 20% outlier does not overturn the O(n^2) verdict");

        var passing = new Submission("ok", true);
        passing.Tests.Add(("empty input", true));
        passing.Tests.Add(("sorted input", true));
        var failing = new Submission("bad", true);
        failing.Tests.Add(("empty input", true));
        failing.Tests.Add(("sorted input", false));
        var broken = new Submission("broken", false);
        broken.Tests.Add(("empty input", false));

        Check(Verdict(passing) == "PASS", "all tests passing yields PASS");
        Check(Verdict(failing) == "FAIL", "one failing test yields FAIL");
        Check(Verdict(broken) == "COMPILE_ERROR" && Verdict(broken) != "FAIL",
            "a non-compiling submission outranks a test failure in the verdict precedence");

        var slower = new long[sizes.Length];
        for (int i = 0; i < sizes.Length; i++) slower[i] = exactQuadratic[i] * 112 / 100;
        Check(Regressions(sizes, slower, exactQuadratic, 0.05).Count == sizes.Length,
            "a 12% slowdown is reported at every sampled size when the tolerance is 5%");
        var barely = new long[sizes.Length];
        for (int i = 0; i < sizes.Length; i++) barely[i] = exactQuadratic[i] * 102 / 100;
        Check(Regressions(sizes, barely, exactQuadratic, 0.05).Count == 0,
            "a 2% drift stays inside the 5% tolerance and is not reported");
        Check(Regressions(sizes, exactQuadratic, exactQuadratic, 0.0).Count == 0, "an equal run never regresses even at zero tolerance");

        Check(Classify(new[] { 64, 128 }, new long[] { 4096, 16384 }).IsEmpty,
            "two sampled sizes are not enough to classify - the boundary refuses instead of guessing");
        Check(Classify(new int[0], new long[0]).IsEmpty, "an empty sample set classifies as insufficient data without crashing");

        long[] noisy1 = NoisyOps(20240901UL, sizes, n => 3.0 * n * n);
        long[] noisy2 = NoisyOps(20240901UL, sizes, n => 3.0 * n * n);
        long[] noisy3 = NoisyOps(99UL, sizes, n => 3.0 * n * n);
        bool sameStream = true, seedSensitive = false;
        for (int i = 0; i < sizes.Length; i++)
        {
            sameStream &= noisy1[i] == noisy2[i];
            seedSensitive |= noisy1[i] != noisy3[i];
        }
        Check(sameStream && seedSensitive, "the noisy measurement generator is seed-reproducible and seed-sensitive");
        Check(Classify(sizes, noisy2).Shape == "O(n^2)", "the noisy stream is still classified O(n^2)");

        Console.WriteLine("classifier: " + Catalogue.Length + " catalogue shapes recovered exactly, constants recovered to 1e-9");
        Console.WriteLine("regression: 12% slowdown -> " + Regressions(sizes, slower, exactQuadratic, 0.05).Count +
            " findings, 2% drift -> " + Regressions(sizes, barely, exactQuadratic, 0.05).Count + " findings");
        Console.WriteLine("verdicts: PASS=" + Verdict(passing) + " FAIL=" + Verdict(failing) + " BROKEN=" + Verdict(broken));
        Console.WriteLine("PASS");
    }
}
