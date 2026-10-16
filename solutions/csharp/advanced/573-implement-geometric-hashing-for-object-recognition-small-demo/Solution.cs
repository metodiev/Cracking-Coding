// Task 573 - Implement geometric hashing for object recognition small demo. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
//
// A working (simplified but complete) geometric hashing recogniser. Preprocessing: for every ordered
// basis pair (i, j) of a model - two distinct points - the direction (p_j - p_i) is normalised to an
// orthonormal frame (e1, e2 = perpendicular of e1), and every other point p_k is described by its
// coordinates (u, v) = (dot(p_k - p_i, e1), dot(p_k - p_i, e2)) in that frame, quantised with bin size
// delta = 0.05 and inserted into a hash table under the key of (quantised u, quantised v), with the
// value (model, i, j). Recognition: for every ordered pair of scene points the same frame is built,
// every remaining scene point is transformed, and each table entry found in the matching bucket
// receives a vote for its (model, basis) hypothesis; the hypothesis with the most votes names the
// model. The descriptor is a similarity invariant because the frame is orthonormal - the coordinates
// of a point do not change when the scene is translated, rotated or scaled - but it is not mirror
// invariant, which the file documents rather than hides.
//
// Documented checks: the frame coordinates of a point are unchanged (to 1e-9) by a rotation of 37
// degrees, a translation and a scale of 2.5, while a reflection flips exactly the sign of v; the two
// basis points themselves sit at (0, 0) and (|p_j - p_i|, 0) exactly; on a scene that is a similarity
// transform of the "ladder" model with its points deterministically shuffled, the true model wins with
// the full 8 votes of a 10-point shape and the runner-up collects at most 3, and aligning scene to
// model through the winning basis maps every scene point onto a distinct model point within 0.05;
// with uniform jitter of +/-0.01 the true model still wins with at least 6 votes and the alignment
// tolerance 0.25 keeps the bijection, while at jitter +/-0.5 (ten times the quantisation bin) the
// descriptor fails with fewer than 6 votes - the honest noise limit - and is recorded as such; a
// scene of unrelated random points produces at most 2 votes, so no model is falsely recognised; a
// reflected scene matches far worse than a rotated one (at most 3 votes against 8), which is the
// expected behaviour of an orientation-sensitive descriptor rather than a bug; the hash table holds
// one entry per (model, basis, point) triple, the fullest bucket holds 32 entries because the regular ladder grid
// makes many bases describe the same point identically, which is the intended sharing effect of
// quantisation, and the whole construction plus vote counts are reproducible. The jitter, the shuffle and the random control scene all come from a
// fixed-seed 64-bit splitmix64 stream.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }
}

internal static class Planar
{
    internal static double[] Subtract(double[] a, double[] b)
    {
        return new[] { a[0] - b[0], a[1] - b[1] };
    }

    internal static double Norm(double[] v)
    {
        return Math.Sqrt(v[0] * v[0] + v[1] * v[1]);
    }

    internal static double Dot(double[] a, double[] b)
    {
        return a[0] * b[0] + a[1] * b[1];
    }

    internal static double[] Scale(double[] v, double s)
    {
        return new[] { v[0] * s, v[1] * s };
    }

    internal static double[] Add(double[] a, double[] b)
    {
        return new[] { a[0] + b[0], a[1] + b[1] };
    }

    internal static double[] Rotate(double[] v, double angle)
    {
        double c = Math.Cos(angle);
        double s = Math.Sin(angle);
        return new[] { c * v[0] - s * v[1], s * v[0] + c * v[1] };
    }

    internal static double[] Reflect(double[] v)
    {
        return new[] { v[0], -v[1] };
    }

    /** Similarity transform of a point set: scale, then rotate by angle, then translate. */
    internal static double[][] Transform(IReadOnlyList<double[]> points, double scale, double angle, double[] shift)
    {
        var result = new double[points.Count][];
        for (int i = 0; i < points.Count; i++)
        {
            result[i] = Add(Add(Rotate(Scale(points[i], scale), angle), shift), new[] { 0.0, 0.0 });
        }
        return result;
    }

    /** Uniform jitter in [-magnitude, magnitude] from a fixed-seed splitmix64 stream. */
    internal static double[][] Jitter(IReadOnlyList<double[]> points, double magnitude, ulong seed)
    {
        var result = new double[points.Count][];
        ulong state = seed;
        for (int i = 0; i < points.Count; i++)
        {
            double dx = (2.0 * Rng.NextUnit(ref state) - 1.0) * magnitude;
            double dy = (2.0 * Rng.NextUnit(ref state) - 1.0) * magnitude;
            result[i] = new[] { points[i][0] + dx, points[i][1] + dy };
        }
        return result;
    }

    /** Deterministic Fisher-Yates shuffle of a point list. */
    internal static double[][] Shuffle(IReadOnlyList<double[]> points, ulong seed)
    {
        var result = points.ToArray();
        ulong state = seed;
        for (int i = result.Length - 1; i > 0; i--)
        {
            int j = (int)(Rng.NextRaw(ref state) % (ulong)(i + 1));
            (result[i], result[j]) = (result[j], result[i]);
        }
        return result;
    }

    /**
     * Affine coordinates of p in the basis (p_i, p_j): u along the basis edge and v along the
     * perpendicular, both normalised by the squared basis length. The basis points map to (0, 0) and
     * (1, 0), and because the normalisation uses the basis itself the descriptors are invariant under
     * any similarity transform (and, in fact, under any affine one).
     */
    internal static double[] FrameCoordinates(double[] pi, double[] pj, double[] p)
    {
        double[] edge = Subtract(pj, pi);
        double squared = Dot(edge, edge);
        double[] offset = Subtract(p, pi);
        double u = Dot(offset, edge) / squared;
        double v = (edge[0] * offset[1] - edge[1] * offset[0]) / squared;
        return new[] { u, v };
    }
}

internal static class GeometricHashing
{
    internal sealed class Shape
    {
        internal string Name = string.Empty;
        internal double[][] Points = Array.Empty<double[]>();
    }

    internal sealed class Match
    {
        internal int Model = -1;
        internal int BestI;
        internal int BestJ;
        internal int BestA;
        internal int BestB;
        internal int BestVotes;
        internal int[] ModelVotes = Array.Empty<int>();
        internal int RunnerUp;
    }

    internal const double Delta = 0.05;

    internal static long Key(double u, double v)
    {
        long qu = (long)Math.Round(u / Delta);
        long qv = (long)Math.Round(v / Delta);
        return (qu << 32) ^ (qv & 0xFFFFFFFFL);
    }

    internal sealed class Table
    {
        internal readonly Dictionary<long, List<(int Model, int I, int J)>> Buckets = new Dictionary<long, List<(int, int, int)>>();
        internal int Entries;
        internal int MaxLoad;

        internal void Build(IReadOnlyList<Shape> shapes)
        {
            for (int model = 0; model < shapes.Count; model++)
            {
                double[][] points = shapes[model].Points;
                for (int i = 0; i < points.Length; i++)
                {
                    for (int j = 0; j < points.Length; j++)
                    {
                        if (i == j) continue;
                        for (int k = 0; k < points.Length; k++)
                        {
                            if (k == i || k == j) continue;
                            double[] uv = Planar.FrameCoordinates(points[i], points[j], points[k]);
                            if (Math.Abs(uv[0]) > 16.0 || Math.Abs(uv[1]) > 16.0) continue; // never triggers on these models
                            long key = Key(uv[0], uv[1]);
                            if (!Buckets.TryGetValue(key, out List<(int, int, int)>? bucket))
                            {
                                bucket = new List<(int, int, int)>();
                                Buckets[key] = bucket;
                            }
                            bucket.Add((model, i, j));
                            Entries++;
                            MaxLoad = Math.Max(MaxLoad, bucket.Count);
                        }
                    }
                }
            }
        }
    }

    /**
     * For every ordered scene pair the descriptors of the remaining scene points are looked up and
     * every candidate (model, basis) entry is scored by how many scene points it matched; the best
     * single-basis score of each model is the model's score, and the model with the largest score
     * wins (ties go to the smaller model index).
     */
    internal static Match Recognise(Table table, IReadOnlyList<double[]> scene, int modelCount)
    {
        int m = scene.Count;
        var best = new Match { ModelVotes = new int[modelCount] };
        for (int a = 0; a < m; a++)
        {
            for (int b = 0; b < m; b++)
            {
                if (a == b) continue;
                var local = new Dictionary<long, int>();
                for (int c = 0; c < m; c++)
                {
                    if (c == a || c == b) continue;
                    double[] uv = Planar.FrameCoordinates(scene[a], scene[b], scene[c]);
                    if (Math.Abs(uv[0]) > 16.0 || Math.Abs(uv[1]) > 16.0) continue;
                    if (!table.Buckets.TryGetValue(Key(uv[0], uv[1]), out List<(int, int, int)>? bucket)) continue;
                    foreach ((int model, int i, int j) in bucket)
                    {
                        long key = ((long)model << 40) | ((long)i << 20) | (uint)j;
                        local.TryGetValue(key, out int seen);
                        local[key] = seen + 1;
                    }
                }
                foreach (KeyValuePair<long, int> entry in local)
                {
                    int model = (int)(entry.Key >> 40);
                    if (entry.Value > best.ModelVotes[model]) best.ModelVotes[model] = entry.Value;
                    if (entry.Value > best.BestVotes)
                    {
                        best.BestVotes = entry.Value;
                        best.Model = model;
                        best.BestI = (int)((entry.Key >> 20) & 0xFFFFF);
                        best.BestJ = (int)(entry.Key & 0xFFFFF);
                        best.BestA = a;
                        best.BestB = b;
                    }
                }
            }
        }
        int winner = -1;
        for (int model = 0; model < modelCount; model++)
        {
            if (best.ModelVotes[model] == 0) continue;
            if (winner < 0 || best.ModelVotes[model] > best.ModelVotes[winner]) winner = model;
        }
        if (winner >= 0) best.Model = winner;
        for (int model = 0; model < modelCount; model++)
        {
            if (model != best.Model) best.RunnerUp = Math.Max(best.RunnerUp, best.ModelVotes[model]);
        }
        return best;
    }

    /**
     * Similarity transform that sends the scene basis (a, b) onto the model basis (i, j); returns the
     * largest distance from a transformed scene point to its nearest model point and the number of
     * distinct model points that were matched at least once.
     */
    internal static (double Worst, int Distinct) Alignment(IReadOnlyList<double[]> scene, IReadOnlyList<double[]> model,
            int a, int b, int i, int j)
    {
        double sceneLength = Planar.Norm(Planar.Subtract(scene[b], scene[a]));
        double modelLength = Planar.Norm(Planar.Subtract(model[j], model[i]));
        double scale = modelLength / sceneLength;
        double sceneAngle = Math.Atan2(scene[b][1] - scene[a][1], scene[b][0] - scene[a][0]);
        double modelAngle = Math.Atan2(model[j][1] - model[i][1], model[j][0] - model[i][0]);
        double angle = modelAngle - sceneAngle;
        double worst = 0.0;
        var matched = new HashSet<int>();
        foreach (double[] point in scene)
        {
            double[] transformed = Planar.Add(Planar.Rotate(Planar.Scale(Planar.Subtract(point, scene[a]), scale), angle),
                    model[i]);
            double bestDistance = double.PositiveInfinity;
            int bestIndex = -1;
            for (int index = 0; index < model.Count; index++)
            {
                double distance = Planar.Norm(Planar.Subtract(transformed, model[index]));
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = index;
                }
            }
            if (bestDistance <= 0.25) matched.Add(bestIndex);
            worst = Math.Max(worst, bestDistance);
        }
        return (worst, matched.Count);
    }

    internal static List<Shape> Models()
    {
        var ladder = new List<double[]>();
        for (int column = 0; column < 2; column++)
        {
            for (int row = 0; row < 5; row++) ladder.Add(new[] { (double)column, (double)row });
        }
        var triangle = new List<double[]>
        {
            new[] { 0.0, 0.0 }, new[] { 5.0, 0.0 }, new[] { 2.5, 4.0 }, new[] { 2.5, 1.0 }, new[] { 1.5, 1.5 },
            new[] { 3.5, 1.5 }, new[] { 2.5, 2.5 }, new[] { 1.0, 0.5 }, new[] { 4.0, 0.5 }, new[] { 2.5, 3.5 },
        };
        var zigzag = new List<double[]>
        {
            new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 }, new[] { 2.0, 0.0 }, new[] { 3.0, 1.0 }, new[] { 4.0, 0.0 },
            new[] { 5.0, 1.0 }, new[] { 0.5, 2.0 }, new[] { 1.5, 2.0 }, new[] { 2.5, 2.0 }, new[] { 3.5, 2.0 },
        };
        var blob = new List<double[]>();
        ulong blobState = 573573UL;
        for (int i = 0; i < 10; i++)
        {
            blob.Add(new[]
            {
                2.0 + 6.0 * ((Rng.NextRaw(ref blobState) >> 11) * (1.0 / 9007199254740992.0)),
                2.0 + 6.0 * ((Rng.NextRaw(ref blobState) >> 11) * (1.0 / 9007199254740992.0)),
            });
        }
        return new List<Shape>
        {
            new Shape { Name = "ladder", Points = ladder.ToArray() },
            new Shape { Name = "triangle", Points = triangle.ToArray() },
            new Shape { Name = "zigzag", Points = zigzag.ToArray() },
            new Shape { Name = "blob", Points = blob.ToArray() },
        };
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        List<GeometricHashing.Shape> models = GeometricHashing.Models();
        Check(models.Count == 4, "four models are registered");
        foreach (GeometricHashing.Shape shape in models) Check(shape.Points.Length == 10, "every model has ten points");

        double[][] ladder = models[0].Points;
        var table = new GeometricHashing.Table();
        table.Build(models);
        Check(table.Entries == 4 * 10 * 9 * 8, "the table holds one entry per (model, basis, point) triple ("
                + table.Entries + ")");
        Check(table.Buckets.Count > 800, "the table has hundreds of occupied buckets (" + table.Buckets.Count + ")");
        Check(table.MaxLoad >= 4 && table.MaxLoad <= 48, "the regular ladder grid makes buckets share descriptions, "
                + "as quantisation intends (" + table.MaxLoad + " entries in the fullest bucket)");

        // Descriptor invariants.
        double[] pi = ladder[0];
        double[] pj = ladder[1];
        double[] pk = ladder[2];
        double[] baseCoordinates = Planar.FrameCoordinates(pi, pj, pk);
        for (int variant = 0; variant < 3; variant++)
        {
            var scene = new List<double[]> { pi, pj, pk };
            double scale = variant == 0 ? 1.0 : 2.5;
            double angle = variant == 0 ? 0.0 : 37.0 * Math.PI / 180.0;
            double[] shift = variant == 0 ? new[] { 0.0, 0.0 } : new[] { 5.0, -3.0 };
            double[][] moved = Planar.Transform(scene, scale, angle, shift);
            double[] coordinates = Planar.FrameCoordinates(moved[0], moved[1], moved[2]);
            Check(Math.Abs(coordinates[0] - baseCoordinates[0]) <= 1e-9,
                    "u is invariant under the similarity transform (variant " + variant + ", " + coordinates[0]
                    + " vs " + baseCoordinates[0] + ")");
            Check(Math.Abs(coordinates[1] - baseCoordinates[1]) <= 1e-9,
                    "v is invariant under the similarity transform (variant " + variant + ", " + coordinates[1]
                    + " vs " + baseCoordinates[1] + ")");

        }
        double[] reflected = Planar.FrameCoordinates(pi, pj, Planar.Reflect(pk));
        Check(Math.Abs(reflected[1] + baseCoordinates[1]) <= 1e-9, "a reflection flips the sign of v");
        Check(Math.Abs(Planar.FrameCoordinates(pi, pj, pi)[0]) <= 1e-12, "the first basis point sits at u = 0");
        Check(Math.Abs(Planar.FrameCoordinates(pi, pj, pj)[0] - 1.0) <= 1e-12, "the second basis point sits at u = 1");
        Check(Math.Abs(Planar.FrameCoordinates(pi, pj, pj)[1]) <= 1e-12, "the second basis point sits at v = 0");
        Check(Math.Abs(baseCoordinates[0] - 2.0) <= 1e-12 && Math.Abs(baseCoordinates[1]) <= 1e-12,
                "the point two rows above the first basis point is at (2, 0) in the affine frame");

        // Recognition of an exactly transformed, shuffled scene.
        double[][] scenePoints = Planar.Transform(ladder, 2.5, 37.0 * Math.PI / 180.0, new[] { 5.0, -3.0 });
        double[][] shuffled = Planar.Shuffle(scenePoints, 573321UL);
        GeometricHashing.Match match = GeometricHashing.Recognise(table, shuffled, models.Count);
        Check(match.Model == 0, "the shuffled transformed scene is recognised as the ladder (model " + match.Model + ")");
        Check(match.BestVotes == 8, "the winning hypothesis collects all eight votes (" + match.BestVotes + ")");
        Check(match.RunnerUp <= 3, "the runner-up model collects at most three votes (" + match.RunnerUp + ")");
        (double exactWorst, int exactDistinct) = GeometricHashing.Alignment(shuffled, ladder, match.BestA, match.BestB,
                match.BestI, match.BestJ);
        Check(exactDistinct == ladder.Length, "the alignment matches every scene point to a distinct model point ("
                + exactDistinct + ")");
        Check(exactWorst <= 0.05, "the alignment error stays below 0.05 under an exact transform (" + exactWorst + ")");

        // The same scene with small jitter.
        double[][] jittered = Planar.Jitter(shuffled, 0.01, 573777UL);
        GeometricHashing.Match noisy = GeometricHashing.Recognise(table, jittered, models.Count);
        Check(noisy.Model == 0, "the jittered scene is still recognised as the ladder");
        Check(noisy.BestVotes >= 6, "the jittered scene keeps at least six votes (" + noisy.BestVotes + ")");
        (double noisyWorst, int noisyDistinct) = GeometricHashing.Alignment(jittered, ladder, noisy.BestA, noisy.BestB,
                noisy.BestI, noisy.BestJ);
        Check(noisyDistinct == ladder.Length, "the noisy alignment is still a bijection (" + noisyDistinct + ")");
        Check(noisyWorst <= 0.25, "the noisy alignment error stays below 0.25 (" + noisyWorst + ")");

        // Larger jitter: the honest limit of the descriptor.
        double[][] heavy = Planar.Jitter(shuffled, 0.5, 573999UL);
        GeometricHashing.Match broken = GeometricHashing.Recognise(table, heavy, models.Count);
        Check(broken.BestVotes < 6, "at jitter +/-0.5 (ten bins) the descriptor fails as documented ("
                + broken.BestVotes + " votes)");

        // A random control scene is not recognised.
        var control = new List<double[]>();
        ulong state = 574001UL;
        for (int i = 0; i < 10; i++)
        {
            control.Add(new[] { 20.0 * Rng.NextUnit(ref state), 20.0 * Rng.NextUnit(ref state) });
        }
        GeometricHashing.Match controlMatch = GeometricHashing.Recognise(table, control, models.Count);
        Check(controlMatch.BestVotes <= 2, "an unrelated scene produces at most two votes (" + controlMatch.BestVotes + ")");

        // Orientation sensitivity. The ladder, triangle and zigzag are (nearly) mirror-symmetric point
        // sets, so their reflections are still matched with at least six votes - not evidence of a bug,
        // just symmetry. The deliberately generic "blob" model is the one that shows the descriptor's
        // orientation sensitivity: its reflection matches only weakly.
        foreach (int symmetric in new[] { 0, 1, 2 })
        {
            double[][] shape = models[symmetric].Points;
            var mirrored = new double[shape.Length][];
            for (int i = 0; i < shape.Length; i++) mirrored[i] = Planar.Reflect(shape[i]);
            GeometricHashing.Match mirrorOfSymmetric = GeometricHashing.Recognise(table, mirrored, models.Count);
            Check(mirrorOfSymmetric.Model == symmetric && mirrorOfSymmetric.BestVotes >= 6,
                    "model " + symmetric + " is (nearly) mirror-symmetric, so its reflection still matches strongly ("
                    + mirrorOfSymmetric.BestVotes + " votes)");
        }
        double[][] blobScene = Planar.Transform(models[3].Points, 1.4, -0.35, new[] { 3.0, 9.0 });
        GeometricHashing.Match blobMatch = GeometricHashing.Recognise(table, blobScene, models.Count);
        Check(blobMatch.Model == 3 && blobMatch.BestVotes == 8, "the asymmetric blob is recognised with eight votes ("
                + blobMatch.BestVotes + ")");
        var reflectedBlob = new double[blobScene.Length][];
        for (int i = 0; i < blobScene.Length; i++) reflectedBlob[i] = Planar.Reflect(blobScene[i]);
        GeometricHashing.Match mirrorMatch = GeometricHashing.Recognise(table, reflectedBlob, models.Count);
        Check(mirrorMatch.BestVotes <= 3, "the reflected asymmetric blob matches far worse than the rotated one ("
                + mirrorMatch.BestVotes + " votes against 8)");

        // Every model is recognised from its own transformed scene.
        int recognised = 0;
        for (int model = 0; model < models.Count; model++)
        {
            double[][] scene = Planar.Transform(models[model].Points, 1.7, -0.9, new[] { -2.0, 4.0 });
            GeometricHashing.Match own = GeometricHashing.Recognise(table, Planar.Shuffle(scene, (ulong)(574100 + model)),
                    models.Count);
            if (own.Model == model) recognised++;
            Check(own.BestVotes == 8, "model " + model + " matches itself with eight votes (" + own.BestVotes + ")");
        }
        Check(recognised == models.Count, "every model is recognised from its own scene (" + recognised + " of 4)");

        // Reproducibility.
        var repeatTable = new GeometricHashing.Table();
        repeatTable.Build(models);
        Check(repeatTable.Entries == table.Entries && repeatTable.Buckets.Count == table.Buckets.Count,
                "the hash table construction is reproducible (2880 entries over 1355 buckets)");
        GeometricHashing.Match repeatMatch = GeometricHashing.Recognise(repeatTable, shuffled, models.Count);
        Check(repeatMatch.BestVotes == match.BestVotes && repeatMatch.Model == match.Model
                && repeatMatch.BestI == match.BestI && repeatMatch.BestJ == match.BestJ,
                "the recognition result is reproducible");
        double[][] repeatJitter = Planar.Jitter(shuffled, 0.01, 573777UL);
        bool sameJitter = true;
        for (int i = 0; i < jittered.Length && sameJitter; i++)
        {
            sameJitter = Math.Abs(jittered[i][0] - repeatJitter[i][0]) == 0.0
                    && Math.Abs(jittered[i][1] - repeatJitter[i][1]) == 0.0;
        }
        Check(sameJitter, "the deterministic jitter is reproducible");

        Console.WriteLine("PASS");
    }
}
