/** Task 386 - Implement elliptic curve point arithmetic (outline). (Tier: Intermediate, CTCI Beyond — Security & Privacy) */
import java.util.*;

public class Solution {
    static long[] extendedGcd(long a, long b) {
        if (b == 0) return new long[] { a, 1, 0 };
        long[] prev = extendedGcd(b, a % b);
        return new long[] { prev[0], prev[2], prev[1] - (a / b) * prev[2] };
    }

    static long modInverse(long a, long modulus) {
        long[] eg = extendedGcd(((a % modulus) + modulus) % modulus, modulus);
        if (eg[0] != 1) throw new IllegalArgumentException("modular inverse does not exist");
        return ((eg[1] % modulus) + modulus) % modulus;
    }

    static long modPower(long base, long exponent, long modulus) {
        long result = 1 % modulus;
        long b = base % modulus;
        long e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = result * b % modulus;
            b = b * b % modulus;
            e >>= 1;
        }
        return result;
    }

    static boolean isOnCurve(long[] point, long a, long b, long p) {
        if (point == null) return true;               // O is on every curve by definition
        long x = point[0], y = point[1];
        return ((y * y - (x * x % p * x % p + a * x + b)) % p + p) % p == 0;
    }

    static boolean samePoint(long[] one, long[] two) {
        if (one == null || two == null) return one == two;
        return Arrays.equals(one, two);
    }

    /** Group addition on y^2 = x^3 + a*x + b over F_p; null is the identity O. */
    static long[] add(long[] point1, long[] point2, long a, long p) {
        if (point1 == null) return point2;
        if (point2 == null) return point1;
        long x1 = point1[0], y1 = point1[1], x2 = point2[0], y2 = point2[1];
        if (x1 == x2 && (y1 + y2) % p == 0) return null;   // P + (-P) = O (also y = 0 doubling)
        long slope;
        if (x1 == x2) {
            slope = (3 * x1 * x1 + a) % p * modInverse(2 * y1, p) % p;
        } else {
            slope = (((y2 - y1) % p + p) % p) * modInverse(x2 - x1, p) % p;
        }
        long x3 = (slope * slope - x1 - x2) % p;
        x3 = (x3 + p) % p;
        long y3 = ((slope * (x1 - x3) - y1) % p + p) % p;
        return new long[] { x3, y3 };
    }

    /** Double-and-add; k >= 0, k = 0 gives null (O). */
    static long[] scalarMultiply(long k, long[] point, long a, long p) {
        long[] result = null;
        long[] addend = point;
        while (k > 0) {
            if ((k & 1) == 1) result = add(result, addend, a, p);
            addend = add(addend, addend, a, p);
            k >>= 1;
        }
        return result;
    }

    /** Repeated addition; reference oracle for small k. */
    static long[] scalarMultiplyNaive(long k, long[] point, long a, long p) {
        long[] result = null;
        for (long i = 0; i < k; i++) result = add(result, point, a, p);
        return result;
    }

    /** All affine points in deterministic order. */
    static List<long[]> enumeratePoints(long a, long b, long p) {
        List<long[]> points = new ArrayList<>();
        for (long x = 0; x < p; x++)
            for (long y = 0; y < p; y++)
                if ((y * y - (x * x % p * x % p + a * x + b)) % p == 0)
                    points.add(new long[] { x, y });
        return points;
    }

    /** #E(F_p) = 1 + sum_x (1 + chi(f(x))) with chi the Legendre symbol. */
    static long curveOrder(long a, long b, long p) {
        long count = 1;
        for (long x = 0; x < p; x++) {
            long rhs = (x * x % p * x % p + a * x + b) % p;
            if (rhs == 0) count += 1;
            else if (modPower(rhs, (p - 1) / 2, p) == 1) count += 2;
        }
        return count;
    }

    private static String label(long[] point) {
        return point == null ? "O" : "(" + point[0] + "," + point[1] + ")";
    }

    private static int seed = 3131;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        long a = 2, b = 3, p = 97;
        long[] point = { 3, 6 };
        check(isOnCurve(point, a, b, p), "P = (3, 6) is on the curve");
        check((6 * 6 - (3 * 3 * 3 + 2 * 3 + 3)) % 97 == 0, "hand-check 36 = 27 + 6 + 3");

        List<long[]> points = enumeratePoints(a, b, p);
        check(points.size() == 99, "99 affine points");
        for (long[] pt : points) check(isOnCurve(pt, a, b, p), "enumerated point on curve");
        int torsionCount = 0;
        for (long[] pt : points) if (pt[1] == 0) torsionCount++;
        check(torsionCount == 3, "three points with y = 0 (order 2)");

        // independent group-order computation via Legendre symbols, and the Hasse bound
        long order = curveOrder(a, b, p);
        check(order == 100 && order == points.size() + 1, "group order 100");
        check((order - (p + 1)) * (order - (p + 1)) <= 4 * p, "Hasse bound");

        // identity handling
        check(samePoint(add(null, point, a, p), point), "O + P = P");
        check(samePoint(add(point, null, a, p), point), "P + O = P");
        check(add(null, null, a, p) == null, "O + O = O");

        // hand-verified group law on P = (3, 6): 2P, 3P, 4P, 5P = O, 6P = P
        long[] p2 = add(point, point, a, p);
        check(samePoint(p2, new long[] { 80, 10 }), "2P = (80, 10)");
        long[] p3 = add(p2, point, a, p);
        check(samePoint(p3, new long[] { 80, 87 }), "3P = (80, 87)");
        long[] p4 = add(p2, p2, a, p);
        check(samePoint(p4, new long[] { 3, 91 }), "4P = (3, 91) = -P");
        check(add(point, new long[] { 3, 91 }, a, p) == null, "P + (-P) = O");
        check(scalarMultiply(5, point, a, p) == null, "5P = O");
        check(samePoint(scalarMultiply(6, point, a, p), point), "6P = P");
        check(isOnCurve(p2, a, b, p) && isOnCurve(p3, a, b, p) && isOnCurve(p4, a, b, p),
                "derived points on curve");

        // negation and an order-2 point (doubling must take the y1 + y2 = 0 branch)
        check(add(new long[] { 0, 10 }, new long[] { 0, 87 }, a, p) == null, "(0,10) + (0,-10) = O");
        check(add(new long[] { 30, 0 }, new long[] { 30, 0 }, a, p) == null, "2 * (30, 0) = O");

        // commutativity and associativity spot checks
        for (int i = 0; i < 12; i += 3) {
            long[] q = points.get(i);
            long[] r = points.get((i + 4) % points.size());
            check(samePoint(add(point, q, a, p), add(q, point, a, p)), "commutativity " + i);
            check(samePoint(add(add(point, q, a, p), r, a, p), add(point, add(q, r, a, p), a, p)),
                    "associativity " + i);
        }

        // double-and-add equals naive repeated addition, and results stay on the curve
        for (long k = 0; k <= 30; k++) {
            check(samePoint(scalarMultiply(k, point, a, p), scalarMultiplyNaive(k, point, a, p)),
                    "double-and-add P " + k);
            check(samePoint(scalarMultiply(k, new long[] { 80, 10 }, a, p),
                    scalarMultiplyNaive(k, new long[] { 80, 10 }, a, p)), "double-and-add 2P " + k);
        }
        for (int i = 0; i < 6; i++) {
            long[] pt = points.get(i);
            for (long k = 0; k <= 12; k++) {
                long[] got = scalarMultiply(k, pt, a, p);
                check(samePoint(got, scalarMultiplyNaive(k, pt, a, p)), "pt " + i + " k " + k);
                check(got == null || isOnCurve(got, a, b, p), "on curve " + i + " " + k);
            }
        }

        // the group order is 100, so 100 * P = O for every point
        for (long[] pt : points) check(scalarMultiply(100, pt, a, p) == null, "100P = O " + label(pt));

        // LCG-driven keys: double-and-add and naive must agree
        for (int trial = 0; trial < 40; trial++) {
            long k = nextSeed() % 50;
            long[] base = points.get(nextSeed() % points.size());
            long[] got = scalarMultiply(k, base, a, p);
            check(samePoint(got, scalarMultiplyNaive(k, base, a, p)), "LCG " + trial);
            check(got == null || isOnCurve(got, a, b, p), "LCG on curve " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
