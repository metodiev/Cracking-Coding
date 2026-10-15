// Task 386 - Implement elliptic curve point arithmetic (outline). (Tier: Intermediate, CTCI Beyond — Security & Privacy)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static long[] ExtendedGcd(long a, long b)
    {
        if (b == 0) return new long[] { a, 1, 0 };
        var prev = ExtendedGcd(b, a % b);
        return new long[] { prev[0], prev[2], prev[1] - (a / b) * prev[2] };
    }

    private static long ModInverse(long a, long modulus)
    {
        var eg = ExtendedGcd(((a % modulus) + modulus) % modulus, modulus);
        if (eg[0] != 1) throw new InvalidOperationException("modular inverse does not exist");
        return ((eg[1] % modulus) + modulus) % modulus;
    }

    private static long ModPower(long baseValue, long exponent, long modulus)
    {
        long result = 1 % modulus;
        long b = ((baseValue % modulus) + modulus) % modulus;
        long e = exponent;
        while (e > 0)
        {
            if ((e & 1) == 1) result = (long)((UInt128)result * (UInt128)b % (UInt128)modulus);
            b = (long)((UInt128)b * (UInt128)b % (UInt128)modulus);
            e >>= 1;
        }
        return result;
    }

    private static bool IsOnCurve(long[]? point, long a, long b, long p)
    {
        if (point == null) return true;               // O is on every curve by definition
        long x = point[0], y = point[1];
        return ((y * y - (x * x % p * x % p + a * x + b)) % p + p) % p == 0;
    }

    private static bool SamePoint(long[]? one, long[]? two)
    {
        if (one == null || two == null) return one == two;
        return one[0] == two[0] && one[1] == two[1];
    }

    /// <summary>Group addition on y^2 = x^3 + a*x + b over F_p; null is the identity O.</summary>
    private static long[]? Add(long[]? point1, long[]? point2, long a, long p)
    {
        if (point1 == null) return point2;
        if (point2 == null) return point1;
        long x1 = point1[0], y1 = point1[1], x2 = point2[0], y2 = point2[1];
        if (x1 == x2 && (y1 + y2) % p == 0) return null;   // P + (-P) = O (also y = 0 doubling)
        long slope;
        if (x1 == x2)
        {
            slope = (3 * x1 * x1 + a) % p * ModInverse(2 * y1, p) % p;
        }
        else
        {
            slope = (((y2 - y1) % p + p) % p) * ModInverse(x2 - x1, p) % p;
        }
        long x3 = ((slope * slope - x1 - x2) % p + p) % p;
        long y3 = ((slope * (x1 - x3) - y1) % p + p) % p;
        return new long[] { x3, y3 };
    }

    /// <summary>Double-and-add; k >= 0, k = 0 gives null (O).</summary>
    private static long[]? ScalarMultiply(long k, long[]? point, long a, long p)
    {
        long[]? result = null;
        long[]? addend = point;
        while (k > 0)
        {
            if ((k & 1) == 1) result = Add(result, addend, a, p);
            addend = Add(addend, addend, a, p);
            k >>= 1;
        }
        return result;
    }

    /// <summary>Repeated addition; reference oracle for small k.</summary>
    private static long[]? ScalarMultiplyNaive(long k, long[]? point, long a, long p)
    {
        long[]? result = null;
        for (long i = 0; i < k; i++) result = Add(result, point, a, p);
        return result;
    }

    /// <summary>All affine points in deterministic order.</summary>
    private static List<long[]> EnumeratePoints(long a, long b, long p)
    {
        var points = new List<long[]>();
        for (long x = 0; x < p; x++)
            for (long y = 0; y < p; y++)
                if ((y * y - (x * x % p * x % p + a * x + b)) % p == 0)
                    points.Add(new long[] { x, y });
        return points;
    }

    /// <summary>#E(F_p) = 1 + sum_x (1 + chi(f(x))) with chi the Legendre symbol.</summary>
    private static long CurveOrder(long a, long b, long p)
    {
        long count = 1;
        for (long x = 0; x < p; x++)
        {
            long rhs = (x * x % p * x % p + a * x + b) % p;
            if (rhs == 0) count += 1;
            else if (ModPower(rhs, (p - 1) / 2, p) == 1) count += 2;
        }
        return count;
    }

    private static string Label(long[]? point) => point == null ? "O" : $"({point[0]},{point[1]})";

    private static int seed = 3131;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        long a = 2, b = 3, p = 97;
        var point = new long[] { 3, 6 };
        Check(IsOnCurve(point, a, b, p), "P = (3, 6) is on the curve");
        Check((6 * 6 - (3 * 3 * 3 + 2 * 3 + 3)) % 97 == 0, "hand-check 36 = 27 + 6 + 3");

        var points = EnumeratePoints(a, b, p);
        Check(points.Count == 99, "99 affine points");
        foreach (var pt in points) Check(IsOnCurve(pt, a, b, p), "enumerated point on curve");
        int torsionCount = 0;
        foreach (var pt in points) if (pt[1] == 0) torsionCount++;
        Check(torsionCount == 3, "three points with y = 0 (order 2)");

        // independent group-order computation via Legendre symbols, and the Hasse bound
        long order = CurveOrder(a, b, p);
        Check(order == 100 && order == points.Count + 1, "group order 100");
        Check((order - (p + 1)) * (order - (p + 1)) <= 4 * p, "Hasse bound");

        // identity handling
        Check(SamePoint(Add(null, point, a, p), point), "O + P = P");
        Check(SamePoint(Add(point, null, a, p), point), "P + O = P");
        Check(Add(null, null, a, p) == null, "O + O = O");

        // hand-verified group law on P = (3, 6): 2P, 3P, 4P, 5P = O, 6P = P
        var p2 = Add(point, point, a, p);
        Check(SamePoint(p2, new long[] { 80, 10 }), "2P = (80, 10)");
        var p3 = Add(p2, point, a, p);
        Check(SamePoint(p3, new long[] { 80, 87 }), "3P = (80, 87)");
        var p4 = Add(p2, p2, a, p);
        Check(SamePoint(p4, new long[] { 3, 91 }), "4P = (3, 91) = -P");
        Check(Add(point, new long[] { 3, 91 }, a, p) == null, "P + (-P) = O");
        Check(ScalarMultiply(5, point, a, p) == null, "5P = O");
        Check(SamePoint(ScalarMultiply(6, point, a, p), point), "6P = P");
        Check(IsOnCurve(p2, a, b, p) && IsOnCurve(p3, a, b, p) && IsOnCurve(p4, a, b, p),
                "derived points on curve");

        // negation and an order-2 point (doubling must take the y1 + y2 = 0 branch)
        Check(Add(new long[] { 0, 10 }, new long[] { 0, 87 }, a, p) == null, "(0,10) + (0,-10) = O");
        Check(Add(new long[] { 30, 0 }, new long[] { 30, 0 }, a, p) == null, "2 * (30, 0) = O");

        // commutativity and associativity spot checks
        for (int i = 0; i < 12; i += 3)
        {
            var q = points[i];
            var r = points[(i + 4) % points.Count];
            Check(SamePoint(Add(point, q, a, p), Add(q, point, a, p)), "commutativity " + i);
            Check(SamePoint(Add(Add(point, q, a, p), r, a, p), Add(point, Add(q, r, a, p), a, p)),
                    "associativity " + i);
        }

        // double-and-add equals naive repeated addition, and results stay on the curve
        for (long k = 0; k <= 30; k++)
        {
            Check(SamePoint(ScalarMultiply(k, point, a, p), ScalarMultiplyNaive(k, point, a, p)),
                    "double-and-add P " + k);
            Check(SamePoint(ScalarMultiply(k, new long[] { 80, 10 }, a, p),
                    ScalarMultiplyNaive(k, new long[] { 80, 10 }, a, p)), "double-and-add 2P " + k);
        }
        for (int i = 0; i < 6; i++)
        {
            var pt = points[i];
            for (long k = 0; k <= 12; k++)
            {
                var got = ScalarMultiply(k, pt, a, p);
                Check(SamePoint(got, ScalarMultiplyNaive(k, pt, a, p)), "pt " + i + " k " + k);
                Check(got == null || IsOnCurve(got, a, b, p), "on curve " + i + " " + k);
            }
        }

        // the group order is 100, so 100 * P = O for every point
        foreach (var pt in points) Check(ScalarMultiply(100, pt, a, p) == null, "100P = O " + Label(pt));

        // LCG-driven keys: double-and-add and naive must agree
        for (int trial = 0; trial < 40; trial++)
        {
            long k = NextSeed() % 50;
            var basePoint = points[NextSeed() % points.Count];
            var got = ScalarMultiply(k, basePoint, a, p);
            Check(SamePoint(got, ScalarMultiplyNaive(k, basePoint, a, p)), "LCG " + trial);
            Check(got == null || IsOnCurve(got, a, b, p), "LCG on curve " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
