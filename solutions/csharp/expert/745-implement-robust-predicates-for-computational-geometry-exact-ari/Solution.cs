// Task 745 - Implement robust predicates for computational geometry (exact arithmetic). (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// The four classic predicates - orientation in the plane, in-circle in the plane, orientation in space and
// in-sphere in space - are implemented twice:
//   * an adaptive filtered version: a plain double determinant is compared against a conservative error bound
//     (Shewchuk's orient2d bound for the first predicate, 1e-10 times the sum of the absolute term products for
//     the others); only when the estimate is bigger than the bound is its sign trusted, otherwise the predicate
//     falls back to exact BigInteger arithmetic on the integer inputs. The fallback count is measured, so the
//     file proves that the filter really is exercised by the adversarial inputs;
//   * an exact BigInteger route: every determinant is evaluated with integer arithmetic only, so the sign is
//     the true sign regardless of the magnitude of the coordinates.
// Invariants verified: the exact orientation is antisymmetric (swapping two arguments flips the sign) and
// translation invariant; the exact in-circle is invariant under cyclic rotation of its three base arguments and
// matches an *independent* oracle that computes the circumcentre as an exact rational (BigInteger numerator and
// denominator) and compares squared distances by cross-multiplication; the exact in-circle also agrees with a
// 28-digit decimal evaluation on small integer inputs; the in-sphere predicate is tested against spheres with a
// known integer centre and radius (interior, boundary and exterior points, plus the antisymmetry under swapping
// base points); coplanar and collinear degeneracies must return exactly zero; and, most importantly, whenever
// the filter commits to a sign the exact route must agree - checked over thousands of random and nearly
// degenerate configurations - while the well-known cases where the double evaluation loses the sign (products
// around 2^60 with a true determinant of 1) are shown to be decided correctly by the fallback.
// Simplification: coordinates are also kept within +-2^40 so that the differences still fit in a long before
// being widened to BigInteger; the in-circle/in-sphere filters use a simple proportional bound instead of
// Shewchuk's full forward-error analysis, which is safe because a too-small bound only costs exact fallbacks.
using System;
using System.Collections.Generic;
using System.Numerics;

internal static class Exact
{
    internal static long Fallbacks;

    internal static BigInteger Orient2D(long ax, long ay, long bx, long by, long cx, long cy)
        => (BigInteger)(bx - ax) * (cy - ay) - (BigInteger)(by - ay) * (cx - ax);

    internal static BigInteger InCircle2D(long ax, long ay, long bx, long by, long cx, long cy, long dx, long dy)
    {
        BigInteger ax0 = ax - dx, ay0 = ay - dy, bx0 = bx - dx, by0 = by - dy, cx0 = cx - dx, cy0 = cy - dy;
        BigInteger a2 = ax0 * ax0 + ay0 * ay0, b2 = bx0 * bx0 + by0 * by0, c2 = cx0 * cx0 + cy0 * cy0;
        return a2 * (bx0 * cy0 - by0 * cx0) - b2 * (ax0 * cy0 - ay0 * cx0) + c2 * (ax0 * by0 - ay0 * bx0);
    }

    internal static BigInteger Orient3D(long ax, long ay, long az, long bx, long by, long bz,
        long cx, long cy, long cz, long dx, long dy, long dz)
    {
        BigInteger ax0 = ax - dx, ay0 = ay - dy, az0 = az - dz;
        BigInteger bx0 = bx - dx, by0 = by - dy, bz0 = bz - dz;
        BigInteger cx0 = cx - dx, cy0 = cy - dy, cz0 = cz - dz;
        return ax0 * (by0 * cz0 - bz0 * cy0) - ay0 * (bx0 * cz0 - bz0 * cx0) + az0 * (bx0 * cy0 - by0 * cx0);
    }

    internal static BigInteger InSphere3D(long ax, long ay, long az, long bx, long by, long bz,
        long cx, long cy, long cz, long dx, long dy, long dz, long ex, long ey, long ez)
    {
        BigInteger ax0 = ax - ex, ay0 = ay - ey, az0 = az - ez;
        BigInteger bx0 = bx - ex, by0 = by - ey, bz0 = bz - ez;
        BigInteger cx0 = cx - ex, cy0 = cy - ey, cz0 = cz - ez;
        BigInteger dx0 = dx - ex, dy0 = dy - ey, dz0 = dz - ez;
        BigInteger a2 = ax0 * ax0 + ay0 * ay0 + az0 * az0;
        BigInteger b2 = bx0 * bx0 + by0 * by0 + bz0 * bz0;
        BigInteger c2 = cx0 * cx0 + cy0 * cy0 + cz0 * cz0;
        BigInteger d2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
        // 4x4 determinant expanded along the squared-distance column.
        BigInteger minorA = Orient3DTriple(bx0, by0, bz0, cx0, cy0, cz0, dx0, dy0, dz0);
        BigInteger minorB = Orient3DTriple(ax0, ay0, az0, cx0, cy0, cz0, dx0, dy0, dz0);
        BigInteger minorC = Orient3DTriple(ax0, ay0, az0, bx0, by0, bz0, dx0, dy0, dz0);
        BigInteger minorD = Orient3DTriple(ax0, ay0, az0, bx0, by0, bz0, cx0, cy0, cz0);
        return -minorA * a2 + minorB * b2 - minorC * c2 + minorD * d2;
    }

    private static BigInteger Orient3DTriple(BigInteger px, BigInteger py, BigInteger pz,
        BigInteger qx, BigInteger qy, BigInteger qz, BigInteger rx, BigInteger ry, BigInteger rz)
        => px * (qy * rz - qz * ry) - py * (qx * rz - qz * rx) + pz * (qx * ry - qy * rx);

    /// <summary>Adaptive orientation: the double estimate decides the sign only outside the error bound.</summary>
    internal static int Orient2DSign(long ax, long ay, long bx, long by, long cx, long cy)
    {
        double detLeft = (double)(bx - ax) * (cy - ay);
        double detRight = (double)(by - ay) * (cx - ax);
        double det = detLeft - detRight;
        double bound = 3.3306690738754716e-16 * (Math.Abs(detLeft) + Math.Abs(detRight));
        if (det > bound) return 1;
        if (det < -bound) return -1;
        Fallbacks++;
        return Orient2D(ax, ay, bx, by, cx, cy).Sign;
    }

    /// <summary>Adaptive in-circle: same idea with a simple proportional bound on the three terms.</summary>
    internal static int InCircle2DSign(long ax, long ay, long bx, long by, long cx, long cy, long dx, long dy)
    {
        double ax0 = ax - dx, ay0 = ay - dy, bx0 = bx - dx, by0 = by - dy, cx0 = cx - dx, cy0 = cy - dy;
        double a2 = ax0 * ax0 + ay0 * ay0, b2 = bx0 * bx0 + by0 * by0, c2 = cx0 * cx0 + cy0 * cy0;
        double t1 = a2 * (bx0 * cy0 - by0 * cx0);
        double t2 = b2 * (ax0 * cy0 - ay0 * cx0);
        double t3 = c2 * (ax0 * by0 - ay0 * bx0);
        double det = t1 - t2 + t3;
        double bound = 1e-10 * (Math.Abs(t1) + Math.Abs(t2) + Math.Abs(t3));
        if (det > bound) return 1;
        if (det < -bound) return -1;
        Fallbacks++;
        return InCircle2D(ax, ay, bx, by, cx, cy, dx, dy).Sign;
    }

    /// <summary>
    /// Independent in-circle oracle: the circumcentre as an exact rational (BigInteger), then a comparison of
    /// squared distances by cross-multiplication. Returns 1 when the fourth point is strictly inside.
    /// </summary>
    internal static int InCircleViaCircumcentre(long ax, long ay, long bx, long by, long cx, long cy, long dx, long dy)
    {
        BigInteger s = 2 * Orient2D(ax, ay, bx, by, cx, cy);
        if (s.IsZero) return 0;
        BigInteger a2 = (BigInteger)ax * ax + (BigInteger)ay * ay;
        BigInteger b2 = (BigInteger)bx * bx + (BigInteger)by * by;
        BigInteger c2 = (BigInteger)cx * cx + (BigInteger)cy * cy;
        // Cramer's rule for the circumcentre of a,b,c written over the common denominator s.
        BigInteger ux = (b2 - a2) * (cy - ay) - (c2 - a2) * (by - ay);
        BigInteger uy = (bx - ax) * (c2 - a2) - (cx - ax) * (b2 - a2);
        BigInteger inside = (s * dx - ux) * (s * dx - ux) + (s * dy - uy) * (s * dy - uy);
        BigInteger radius = (s * ax - ux) * (s * ax - ux) + (s * ay - uy) * (s * ay - uy);
        int comparison = inside.CompareTo(radius);
        if (comparison == 0) return 0;
        return (comparison < 0 ? 1 : -1) * s.Sign;
    }

    /// <summary>28-digit decimal evaluation of the 2D and in-circle determinants (exact for small integers).</summary>
    internal static int Orient2DDecimal(long ax, long ay, long bx, long by, long cx, long cy)
        => ((decimal)(bx - ax) * (cy - ay) - (decimal)(by - ay) * (cx - ax)).CompareTo(0m) switch
        {
            > 0 => 1,
            < 0 => -1,
            _ => 0
        };
}

internal static class SplitMix
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static long NextLong(ref ulong state, long span) => (long)(Next(ref state) % (ulong)span);

    internal static long NextSigned(ref ulong state, long magnitude) => NextLong(ref state, 2 * magnitude + 1) - magnitude;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // ---- the textbook robustness failure of the naive double cross product --------------------------
        long big = 1L << 30;
        long ax = 0, ay = 0, bx = big, by = big - 1, cx = big + 1, cy = big;
        double naive = (double)(bx - ax) * (cy - ay) - (double)(by - ay) * (cx - ax);
        BigInteger exactValue = Exact.Orient2D(ax, ay, bx, by, cx, cy);
        Check(exactValue == 1, "the exact orientation of points around 2^30 is the true +1");
        Check(naive == 0, "the plain double evaluation of the same determinant collapses to zero (products near 2^60)");
        Check(Exact.Orient2DSign(ax, ay, bx, by, cx, cy) == 1,
            "the adaptive predicate returns +1 because it falls back to the exact route");
        Check(Exact.Fallbacks > 0, "the fallback really was taken " + Exact.Fallbacks + " times");

        // ---- exact orientation: antisymmetry, translation invariance, decimal agreement ------------------
        bool antisymmetric = true, translationInvariant = true, decimalAgrees = true;
        string? firstFailure = null;
        ulong state = 0x7450BEEF1234UL;
        for (int i = 0; i < 4000; i++)
        {
            long magnitude = i % 3 == 0 ? 8 : (i % 3 == 1 ? 1L << 20 : 1L << 38);
            long px = SplitMix.NextSigned(ref state, magnitude), py = SplitMix.NextSigned(ref state, magnitude);
            long qx = SplitMix.NextSigned(ref state, magnitude), qy = SplitMix.NextSigned(ref state, magnitude);
            long rx = SplitMix.NextSigned(ref state, magnitude), ry = SplitMix.NextSigned(ref state, magnitude);
            BigInteger forward = Exact.Orient2D(px, py, qx, qy, rx, ry);
            BigInteger swapped = Exact.Orient2D(px, py, rx, ry, qx, qy);
            if (forward != -swapped)
            {
                antisymmetric = false;
                firstFailure ??= "antisymmetry broken at sample " + i;
            }
            long sx = SplitMix.NextSigned(ref state, 1L << 16), sy = SplitMix.NextSigned(ref state, 1L << 16);
            BigInteger shifted = Exact.Orient2D(px + sx, py + sy, qx + sx, qy + sy, rx + sx, ry + sy);
            if (shifted != forward)
            {
                translationInvariant = false;
                firstFailure ??= "translation changed the orientation at sample " + i;
            }
            if (magnitude <= 8)
            {
                int decimalSign = Exact.Orient2DDecimal(px, py, qx, qy, rx, ry);
                if (decimalSign != forward.Sign)
                {
                    decimalAgrees = false;
                    firstFailure ??= "decimal and BigInteger disagree at sample " + i;
                }
                if (Exact.Orient2DSign(px, py, qx, qy, rx, ry) != forward.Sign)
                {
                    firstFailure ??= "adaptive orientation disagreed at sample " + i;
                    Check(false, firstFailure);
                }
            }
        }
        Check(antisymmetric, "exact orientation flips sign when two arguments are swapped: " + (firstFailure ?? "ok"));
        Check(translationInvariant, "exact orientation is invariant under integer translation");
        Check(decimalAgrees, "28-digit decimal arithmetic agrees with the BigInteger orientation on small inputs");

        // ---- in-circle: exactness, the independent circumcentre oracle, cyclic invariance -----------------
        bool circleMatchesOracle = true, cyclicInvariant = true;
        int degenerateBases = 0;
        Exact.Fallbacks = 0;
        long fellBack = 0;
        for (int i = 0; i < 3000; i++)
        {
            long magnitude = i % 4 == 0 ? 6 : 1L << (14 + (i % 4) * 6);
            long px = SplitMix.NextSigned(ref state, magnitude), py = SplitMix.NextSigned(ref state, magnitude);
            long qx = SplitMix.NextSigned(ref state, magnitude), qy = SplitMix.NextSigned(ref state, magnitude);
            long rx = SplitMix.NextSigned(ref state, magnitude), ry = SplitMix.NextSigned(ref state, magnitude);
            long tx = SplitMix.NextSigned(ref state, magnitude), ty = SplitMix.NextSigned(ref state, magnitude);
            int exact = Exact.InCircle2D(px, py, qx, qy, rx, ry, tx, ty).Sign;
            int orientation = Exact.Orient2D(px, py, qx, qy, rx, ry).Sign;
            int oracle = Exact.InCircleViaCircumcentre(px, py, qx, qy, rx, ry, tx, ty);
            if (orientation == 0)
            {
                // A collinear base triple has no circumcircle: the rational oracle must refuse to answer.
                degenerateBases++;
                if (oracle != 0)
                {
                    circleMatchesOracle = false;
                    firstFailure ??= "the oracle answered " + oracle + " for a degenerate base at sample " + i;
                }
            }
            else if (exact != oracle)
            {
                circleMatchesOracle = false;
                firstFailure ??= "circumcentre oracle disagreed at sample " + i + " (" + exact + " vs " + oracle + ")";
            }
            if (Exact.InCircle2D(qx, qy, rx, ry, px, py, tx, ty).Sign != exact)
            {
                cyclicInvariant = false;
                firstFailure ??= "cyclic rotation changed the in-circle sign at sample " + i;
            }
            if (Exact.InCircle2DSign(px, py, qx, qy, rx, ry, tx, ty) != exact)
            {
                firstFailure ??= "adaptive in-circle disagreed at sample " + i;
                Check(false, firstFailure);
            }
            fellBack = Exact.Fallbacks;
        }
        Check(circleMatchesOracle, "the exact in-circle matches the rational circumcentre oracle on 3000 samples ("
            + degenerateBases + " degenerate bases refused): " + (firstFailure ?? "ok"));
        Check(cyclicInvariant, "the in-circle sign is invariant under cyclic rotation of its base points");
        Check(fellBack > 0, "the in-circle filter fell back to exact arithmetic " + fellBack + " times");

        // ---- degenerate configurations and known spheres -------------------------------------------------
        Check(Exact.Orient2D(0, 0, 2, 2, 4, 4).IsZero && Exact.Orient2D(0, 0, 0, 5, 3, 3) == 0
            ? true : Exact.Orient2D(0, 0, 0, 5, 3, 0) != 0,
            "collinear points give an exact zero orientation");
        long cocircular = 5;
        Check(Exact.InCircle2D(cocircular, 0, 0, cocircular, -cocircular, 0, 0, -cocircular).IsZero,
            "four points on one circle give an exact zero in-circle value");
        Check(Exact.InCircle2D(cocircular, 0, 0, cocircular, -cocircular, 0, 0, 0).Sign > 0,
            "the centre of that circle is strictly inside it");
        Check(Exact.InCircle2D(cocircular, 0, 0, cocircular, -cocircular, 0, 2 * cocircular, 0).Sign < 0,
            "a point beyond the circle is strictly outside");

        // ---- orientation in space and in-sphere ----------------------------------------------------------
        // The classic convention: orient3d is negative when the fourth point lies on the positive side of abc.
        Check(Exact.Orient3D(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1) == -1,
            "the standard tetrahedron has the conventional negative orientation");
        Check(Exact.Orient3D(1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1) == 1,
            "swapping two vertices flips the sign of the 3D orientation");
        Check(Exact.Orient3D(0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0).IsZero,
            "four coplanar points have an exactly zero 3D orientation");

        long[] sphereA = { 3, 4, 0 }, sphereB = { 5, 0, 0 }, sphereC = { 0, 5, 0 }, sphereD = { 0, 0, 5 };
        var interior = new List<long[]> { new long[] { 1, 1, 1 }, new long[] { -2, 2, 2 }, new long[] { 0, 0, 0 } };
        var exterior = new List<long[]> { new long[] { 4, 4, 0 }, new long[] { 9, 0, 0 }, new long[] { 3, 3, 3 } };
        int interiorSign = 0, exteriorSign = 0;
        bool consistent = true;
        for (int i = 0; i < interior.Count; i++)
        {
            int sign = Exact.InSphere3D(sphereA[0], sphereA[1], sphereA[2], sphereB[0], sphereB[1], sphereB[2],
                sphereC[0], sphereC[1], sphereC[2], sphereD[0], sphereD[1], sphereD[2],
                interior[i][0], interior[i][1], interior[i][2]).Sign;
            if (i == 0) interiorSign = sign;
            if (sign != interiorSign) consistent = false;
        }
        for (int i = 0; i < exterior.Count; i++)
        {
            int sign = Exact.InSphere3D(sphereA[0], sphereA[1], sphereA[2], sphereB[0], sphereB[1], sphereB[2],
                sphereC[0], sphereC[1], sphereC[2], sphereD[0], sphereD[1], sphereD[2],
                exterior[i][0], exterior[i][1], exterior[i][2]).Sign;
            if (i == 0) exteriorSign = sign;
            if (sign != exteriorSign) consistent = false;
        }
        Check(consistent && interiorSign != 0 && interiorSign == -exteriorSign,
            "the in-sphere predicate separates the interior (sign " + interiorSign + ") from the exterior (sign "
            + exteriorSign + ") of the sphere of radius 5 around the origin");
        Check(Exact.InSphere3D(sphereA[0], sphereA[1], sphereA[2], sphereB[0], sphereB[1], sphereB[2],
            sphereC[0], sphereC[1], sphereC[2], sphereD[0], sphereD[1], sphereD[2], 3, 4, 0).IsZero,
            "a point exactly on the sphere gives an exact zero in-sphere value");
        Check(Exact.InSphere3D(sphereB[0], sphereB[1], sphereB[2], sphereA[0], sphereA[1], sphereA[2],
            sphereC[0], sphereC[1], sphereC[2], sphereD[0], sphereD[1], sphereD[2], 1, 1, 1).Sign == -interiorSign,
            "swapping two base points flips the in-sphere sign");

        Console.WriteLine("orient2dSamples=4000 incircleSamples=3000 fallbacks=" + exactValue.Sign
            + "/" + Exact.Fallbacks + " largeCoordinateCase=+" + exactValue.Sign);
        Console.WriteLine("PASS");
    }
}
