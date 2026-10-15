// Task 372 - Polygon area via the shoelace formula (+ orientation). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
using System;

/// <summary>
/// The shoelace (Gauss) formula sums the cross products x_i*y_{i+1} - x_{i+1}*y_i over
/// the edges of the closed polygon.  Half of that sum is the SIGNED area: positive when
/// the vertices are listed counter-clockwise, negative when clockwise, and its absolute
/// value is the area for any *simple* polygon.  Because the vertices are integers the
/// doubled area is an exact integer and the area is an exact multiple of 0.5, so all
/// comparisons below are exact (no tolerance) except for the deliberately tolerant
/// rasterization cross-check of a non-axis-aligned triangle.
///
/// Self-intersecting / self-touching (figure-8, "bowtie") polygons: the formula is not an
/// area at all.  It returns the sum of the lobe areas weighted by their traversal
/// orientation, so a bowtie whose two lobes cancel reports 0 (or the signed difference
/// for unequal lobes).  Math.Abs does not repair that - only simple polygons are
/// guaranteed to yield the true area.  Two bowties below document this behaviour.
///
/// Independent oracle: RasterizedArea counts unit cells (or finer) of the bounding box
/// whose centres lie inside the polygon via ray casting.  For polygons whose edges are
/// axis-parallel with integer coordinates every cell lies entirely inside or outside, so
/// the count is EXACT and asserted equal to |shoelace area|.  For a general triangle the
/// comparison uses a documented tolerance (cells crossed by the boundary contribute at
/// most ~perimeter/samplesPerUnit).
/// </summary>
public static class Solution
{
    private sealed class Lcg
    {
        private int seed;

        public Lcg(int seed)
        {
            this.seed = seed;
        }

        public int Next(int modulus)
        {
            seed = (seed * 75 + 74) % 65537;
            return seed % modulus;
        }
    }

    /// <summary>Twice the signed area (an exact integer for integer vertices).</summary>
    public static long SignedDoubleArea(int[][] points)
    {
        long total = 0;
        for (int i = 0; i < points.Length; i++)
        {
            int[] p = points[i];
            int[] q = points[(i + 1) % points.Length];
            total += (long)p[0] * q[1] - (long)q[0] * p[1];
        }
        return total;
    }

    /// <summary>Signed area; exact (a multiple of 0.5) because vertices are integers.</summary>
    public static double PolygonArea(int[][] points)
    {
        double area = SignedDoubleArea(points) / 2.0;
        return area == 0 ? 0 : area;
    }

    /// <summary>+1 counter-clockwise, -1 clockwise, 0 degenerate (all collinear).</summary>
    public static int Orientation(int[][] points)
    {
        long doubleArea = SignedDoubleArea(points);
        return doubleArea > 0 ? 1 : (doubleArea < 0 ? -1 : 0);
    }

    /// <summary>Ray casting; sample points used below never lie on an edge.</summary>
    public static bool PointInPolygon(double x, double y, int[][] points)
    {
        bool inside = false;
        for (int i = 0; i < points.Length; i++)
        {
            int[] p = points[i];
            int[] q = points[(i + 1) % points.Length];
            if ((p[1] > y) != (q[1] > y))
            {
                double xCross = p[0] + (y - p[1]) * (q[0] - p[0]) / (double)(q[1] - p[1]);
                if (xCross > x) inside = !inside;
            }
        }
        return inside;
    }

    /// <summary>Count cell centres inside the polygon; exact for integer axis-aligned edges.</summary>
    public static double RasterizedArea(int[][] points, int samplesPerUnit)
    {
        if (points.Length < 3) return 0.0;
        int minX = points[0][0], maxX = points[0][0], minY = points[0][1], maxY = points[0][1];
        foreach (int[] p in points)
        {
            minX = Math.Min(minX, p[0]);
            maxX = Math.Max(maxX, p[0]);
            minY = Math.Min(minY, p[1]);
            maxY = Math.Max(maxY, p[1]);
        }
        double epsilon = 1e-6;                    // nudges samples off integer/diagonal lines
        long inside = 0;
        for (int i = minX * samplesPerUnit; i < maxX * samplesPerUnit; i++)
        {
            double x = (i + 0.5) / samplesPerUnit + epsilon;
            for (int j = minY * samplesPerUnit; j < maxY * samplesPerUnit; j++)
            {
                double y = (j + 0.5) / samplesPerUnit + epsilon;
                if (PointInPolygon(x, y, points)) inside++;
            }
        }
        return inside / (double)(samplesPerUnit * samplesPerUnit);
    }

    private static int[][] LcgRectangle(Lcg lcg)
    {
        int width = 1 + lcg.Next(6);
        int height = 1 + lcg.Next(6);
        return new[] { new[] { 0, 0 }, new[] { width, 0 }, new[] { width, height }, new[] { 0, height } };
    }

    private static int[][] LcgLShape(Lcg lcg, out int expected)
    {
        int a = 2 + lcg.Next(7);
        int b = 1 + lcg.Next(4);
        int c = 1 + lcg.Next(a - 1);
        int d = b + 1 + lcg.Next(5);
        expected = a * b + c * (d - b);
        return new[]
        {
            new[] { 0, 0 }, new[] { a, 0 }, new[] { a, b }, new[] { c, b }, new[] { c, d }, new[] { 0, d },
        };
    }

    private static int[][] Reversed(int[][] points)
    {
        var result = new int[points.Length][];
        for (int i = 0; i < points.Length; i++) result[i] = points[points.Length - 1 - i];
        return result;
    }

    public static void Main()
    {
        // --- hand-checked polygons -------------------------------------------
        int[][] square = { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } };
        Check(SignedDoubleArea(square) == 8, "square doubled area");
        Check(PolygonArea(square) == 4.0, "square area 4");
        Check(Orientation(square) == 1, "square is CCW");
        Check(RasterizedArea(square, 1) == 4.0, "square rasterized exactly");

        int[][] smallTriangle = { new[] { 0, 0 }, new[] { 1, 0 }, new[] { 0, 1 } };
        Check(PolygonArea(smallTriangle) == 0.5, "unit triangle 0.5");
        Check(Orientation(smallTriangle) == 1, "unit triangle CCW");

        int[][] bigTriangle = { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 0, 2 } };
        Check(PolygonArea(bigTriangle) == 2.0, "leg-2 triangle 2");

        int[][] clockwise = { new[] { 0, 0 }, new[] { 0, 2 }, new[] { 2, 2 }, new[] { 2, 0 } };
        Check(SignedDoubleArea(clockwise) == -8, "clockwise doubled area");
        Check(PolygonArea(clockwise) == -4.0, "clockwise negative area");
        Check(Math.Abs(PolygonArea(clockwise)) == 4.0, "clockwise absolute area 4");
        Check(Orientation(clockwise) == -1, "clockwise orientation");

        int[][] lShape = { new[] { 0, 0 }, new[] { 3, 0 }, new[] { 3, 1 }, new[] { 1, 1 }, new[] { 1, 3 }, new[] { 0, 3 } };
        Check(PolygonArea(lShape) == 5.0, "L-shape area 5");
        Check(RasterizedArea(lShape, 1) == 5.0, "L-shape rasterized exactly");

        // general triangle with diagonal edges: rasterized within the documented tolerance
        int[][] general = { new[] { 0, 0 }, new[] { 5, 1 }, new[] { 2, 4 } };
        Check(PolygonArea(general) == 9.0, "general triangle 9");
        double estimate = RasterizedArea(general, 32);
        Check(Math.Abs(estimate - 9.0) <= 0.5, "general triangle rasterized within 0.5, got " + estimate);

        // collinear vertices do not change the area
        int[][] collinearExtra = { new[] { 0, 0 }, new[] { 1, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } };
        Check(PolygonArea(collinearExtra) == 4.0, "collinear extra vertex");
        int[][] repeatedVertex = { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 }, new[] { 0, 0 } };
        Check(Orientation(repeatedVertex) == 1, "repeated vertex");

        // --- figure-8 (bowtie) documented behaviour --------------------------
        int[][] bowtie = { new[] { 0, 0 }, new[] { 2, 2 }, new[] { 2, 0 }, new[] { 0, 2 } };      // lobes +1/-1
        Check(SignedDoubleArea(bowtie) == 0, "bowtie lobes cancel");
        Check(PolygonArea(bowtie) == 0.0, "bowtie signed area 0");
        int[][] skewedBowtie = { new[] { 0, 0 }, new[] { 2, 4 }, new[] { 2, 0 }, new[] { 0, 2 } };  // 8/3 - 2/3 = 2
        Check(SignedDoubleArea(skewedBowtie) == -4, "skewed bowtie doubled area");
        Check(PolygonArea(skewedBowtie) == -2.0, "skewed bowtie -2");

        // --- LCG cross-checks against the exact rasterization oracle ---------
        var lcg = new Lcg(37211);
        for (int trial = 0; trial < 30; trial++)
        {
            int[][] rectangle = LcgRectangle(lcg);
            int width = 0, height = 0;
            foreach (int[] p in rectangle)
            {
                width = Math.Max(width, p[0]);
                height = Math.Max(height, p[1]);
            }
            Check(PolygonArea(rectangle) == width * height, "rect area " + trial);
            Check(RasterizedArea(rectangle, 1) == width * height, "rect raster " + trial);
            Check(PolygonArea(Reversed(rectangle)) == -PolygonArea(rectangle), "rect reversed " + trial);
            Check(Orientation(Reversed(rectangle)) == -Orientation(rectangle), "rect reversed orientation " + trial);
        }
        for (int trial = 0; trial < 30; trial++)
        {
            int[][] shape = LcgLShape(lcg, out int expected);
            Check(SignedDoubleArea(shape) == 2L * expected, "L doubled area " + trial);
            Check(PolygonArea(shape) == expected, "L area " + trial);
            Check(RasterizedArea(shape, 1) == expected, "L raster " + trial);
            Check(Orientation(shape) == 1, "L orientation " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
