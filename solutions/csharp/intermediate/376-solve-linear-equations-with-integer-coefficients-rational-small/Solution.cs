// Task 376 - Exact linear systems over rationals (integer coefficients). (Tier: Intermediate, CTCI Ch 6)
using System;

/// <summary>
/// The same Gaussian elimination as task 375, but every entry is an exact rational
/// number: a small Fraction type on top of gcd (implemented here per language) keeps
/// num/den normalised with den &gt; 0.  All arithmetic is exact, so:
///   * the systems of task 375 return exact rational solutions - every assertion is an
///     EQUALITY of rationals, NOT a tolerance check (contrast with 375);
///   * no pivoting strategy is required: the first non-zero pivot can be taken because
///     exact arithmetic never introduces rounding, so a zero pivot really is zero
///     (documented; 375's partial pivoting exists to limit rounding, which cannot
///     happen here);
///   * sentinels: status "unique"/"infinite"/"inconsistent", x = null unless unique.
///
/// Cross-check with floats: a compact double-precision Gaussian implemented HERE (same
/// file, duplicating 375's algorithm) must agree with the exact result to 1e-9 - this is
/// the documented link between the two tasks; with small integer inputs the doubles round
/// the rationals faithfully.
///
/// Inputs and expected solutions include genuine fractions (1/3, 1/6) whose float images
/// cannot be represented exactly - those are asserted exactly on the rationals.
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

    /// <summary>Exact fraction with normalised sign and gcd-reduced terms.</summary>
    private sealed class Rational : IEquatable<Rational>
    {
        public long Num { get; }
        public long Den { get; }

        public Rational(long num) : this(num, 1)
        {
        }

        public Rational(long num, long den)
        {
            if (den == 0) throw new DivideByZeroException("rational with zero denominator");
            if (den < 0)
            {
                num = -num;
                den = -den;
            }
            long g = Gcd(Math.Abs(num), den);
            if (g == 0)
            {
                num = 0;
                den = 1;
            }
            else
            {
                num /= g;
                den /= g;
            }
            Num = num;
            Den = den;
        }

        private static long Gcd(long a, long b)
        {
            while (b != 0)
            {
                long t = a % b;
                a = b;
                b = t;
            }
            return a;
        }

        public Rational Add(Rational o) => new Rational(Num * o.Den + o.Num * Den, Den * o.Den);

        public Rational Sub(Rational o) => new Rational(Num * o.Den - o.Num * Den, Den * o.Den);

        public Rational Mul(Rational o) => new Rational(Num * o.Num, Den * o.Den);

        public Rational Div(Rational o)
        {
            if (o.Num == 0) throw new DivideByZeroException("division by zero rational");
            return new Rational(Num * o.Den, Den * o.Num);
        }

        public bool IsZero => Num == 0;

        public bool Equals(Rational? other) => other != null && Num == other.Num && Den == other.Den;

        public override bool Equals(object? obj) => obj is Rational other && Equals(other);

        public override int GetHashCode() => HashCode.Combine(Num, Den);

        public override string ToString() => Den == 1 ? Num.ToString() : Num + "/" + Den;
    }

    /// <summary>status in {"unique", "infinite", "inconsistent"}; x is null unless unique.</summary>
    private sealed class RationalResult
    {
        public string Status { get; }
        public Rational[]? X { get; }

        public RationalResult(string status, Rational[]? x)
        {
            Status = status;
            X = x;
        }
    }

    private static RationalResult SolveRational(int[][] matrix, long[] rhs)
    {
        int n = matrix.Length;
        var augmented = new Rational[n][];
        for (int i = 0; i < n; i++)
        {
            augmented[i] = new Rational[n + 1];
            for (int j = 0; j < n; j++) augmented[i][j] = new Rational(matrix[i][j]);
            augmented[i][n] = new Rational(rhs[i]);
        }

        int rank = 0;
        for (int col = 0; col < n; col++)
        {
            int pivotRow = -1;
            for (int row = rank; row < n; row++)
            {
                if (!augmented[row][col].IsZero)
                {
                    pivotRow = row;
                    break;
                }
            }
            if (pivotRow == -1) continue;                     // genuinely free column
            Rational[] swap = augmented[rank];
            augmented[rank] = augmented[pivotRow];
            augmented[pivotRow] = swap;
            for (int row = rank + 1; row < n; row++)
            {
                if (augmented[row][col].IsZero) continue;
                Rational factor = augmented[row][col].Div(augmented[rank][col]);
                for (int k = col; k <= n; k++)
                {
                    augmented[row][k] = augmented[row][k].Sub(factor.Mul(augmented[rank][k]));
                }
            }
            rank++;
        }

        for (int row = rank; row < n; row++)
        {
            if (!augmented[row][n].IsZero) return new RationalResult("inconsistent", null);
        }
        if (rank < n) return new RationalResult("infinite", null);

        var x = new Rational[n];
        for (int i = 0; i < n; i++) x[i] = new Rational(0);
        for (int row = n - 1; row >= 0; row--)
        {
            Rational total = augmented[row][n];
            for (int col = row + 1; col < n; col++) total = total.Sub(augmented[row][col].Mul(x[col]));
            x[row] = total.Div(augmented[row][row]);
        }
        return new RationalResult("unique", x);
    }

    /// <summary>Compact double-precision copy of 375's elimination, used only as a cross-check.</summary>
    private static double[]? SolveDouble(int[][] matrix, long[] rhs)
    {
        double eps = 1e-12;
        int n = matrix.Length;
        var augmented = new double[n][];
        for (int i = 0; i < n; i++)
        {
            augmented[i] = new double[n + 1];
            for (int j = 0; j < n; j++) augmented[i][j] = matrix[i][j];
            augmented[i][n] = rhs[i];
        }
        int rank = 0;
        for (int col = 0; col < n; col++)
        {
            int pivotRow = -1;
            double pivotSize = 0.0;
            for (int row = rank; row < n; row++)
            {
                if (Math.Abs(augmented[row][col]) > pivotSize)
                {
                    pivotSize = Math.Abs(augmented[row][col]);
                    pivotRow = row;
                }
            }
            if (pivotRow == -1 || pivotSize <= eps) continue;
            double[] swap = augmented[rank];
            augmented[rank] = augmented[pivotRow];
            augmented[pivotRow] = swap;
            for (int row = rank + 1; row < n; row++)
            {
                double factor = augmented[row][col] / augmented[rank][col];
                for (int k = col; k <= n; k++) augmented[row][k] -= factor * augmented[rank][k];
            }
            rank++;
        }
        if (rank < n) return null;
        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double total = augmented[row][n];
            for (int col = row + 1; col < n; col++) total -= augmented[row][col] * x[col];
            x[row] = total / augmented[row][row];
        }
        return x;
    }

    private static bool SameRationals(Rational[] got, long[] expectedNum, long[] expectedDen)
    {
        if (got.Length != expectedNum.Length) return false;
        for (int i = 0; i < got.Length; i++)
        {
            if (!got[i].Equals(new Rational(expectedNum[i], expectedDen[i]))) return false;
        }
        return true;
    }

    private static (int[][] matrix, long[] rhs, int[] planted) LcgSystem(Lcg lcg)
    {
        int size = 2 + lcg.Next(3);
        var matrix = new int[size][];
        for (int i = 0; i < size; i++)
        {
            matrix[i] = new int[size];
            for (int j = 0; j < size; j++) matrix[i][j] = lcg.Next(11) - 5;
        }
        var planted = new int[size];
        for (int i = 0; i < size; i++) planted[i] = lcg.Next(11) - 5;
        var rhs = new long[size];
        for (int i = 0; i < size; i++)
        {
            long total = 0;
            for (int j = 0; j < size; j++) total += (long)matrix[i][j] * planted[j];
            rhs[i] = total;
        }
        return (matrix, rhs, planted);
    }

    public static void Main()
    {
        // 2x2 integer system: x = 2, y = 1, exactly
        RationalResult result = SolveRational(new[] { new[] { 1, 1 }, new[] { 1, -1 } }, new long[] { 3, 1 });
        Check(result.Status == "unique", "2x2 unique");
        Check(SameRationals(result.X!, new long[] { 2, 1 }, new long[] { 1, 1 }), "2x2 exact (2, 1)");

        // pivot swap: leading zero, x = (3, 2)
        result = SolveRational(new[] { new[] { 0, 1 }, new[] { 1, 0 } }, new long[] { 2, 3 });
        Check(result.Status == "unique", "pivot swap unique");
        Check(SameRationals(result.X!, new long[] { 3, 2 }, new long[] { 1, 1 }), "pivot swap (3, 2)");

        // classic 3x3, solution (5, 3, -2)
        result = SolveRational(
            new[] { new[] { 1, 1, 1 }, new[] { 0, 2, 5 }, new[] { 2, 5, -1 } }, new long[] { 6, -4, 27 });
        Check(result.Status == "unique", "3x3 unique");
        Check(SameRationals(result.X!, new long[] { 5, 3, -2 }, new long[] { 1, 1, 1 }), "3x3 exact");

        // 4x4 diagonally dominant, solution (1, 2, 3, 4)
        result = SolveRational(
            new[]
            {
                new[] { 2, 1, 1, 1 }, new[] { 1, 2, 1, 1 }, new[] { 1, 1, 2, 1 }, new[] { 1, 1, 1, 2 },
            },
            new long[] { 11, 12, 13, 14 });
        Check(result.Status == "unique", "4x4 unique");
        Check(SameRationals(result.X!, new long[] { 1, 2, 3, 4 }, new long[] { 1, 1, 1, 1 }), "4x4 exact");

        // genuine fractions: 3x = 1 -> x = 1/3
        result = SolveRational(new[] { new[] { 3 } }, new long[] { 1 });
        Check(result.Status == "unique", "1x1 unique");
        Check(SameRationals(result.X!, new long[] { 1 }, new long[] { 3 }), "1x1 exact 1/3");

        // 2x + 4y = 1, x - y = 0 -> x = y = 1/6
        result = SolveRational(new[] { new[] { 2, 4 }, new[] { 1, -1 } }, new long[] { 1, 0 });
        Check(result.Status == "unique", "1/6 system unique");
        Check(SameRationals(result.X!, new long[] { 1, 1 }, new long[] { 6, 6 }), "1/6 exact");

        // 5x + 2y = 1, x + y = 0 -> x = 1/3, y = -1/3
        result = SolveRational(new[] { new[] { 5, 2 }, new[] { 1, 1 } }, new long[] { 1, 0 });
        Check(result.Status == "unique", "third system unique");
        Check(SameRationals(result.X!, new long[] { 1, -1 }, new long[] { 3, 3 }), "1/3, -1/3 exact");

        // sentinels behave exactly like 375's
        result = SolveRational(new[] { new[] { 1, 1 }, new[] { 2, 2 } }, new long[] { 1, 2 });
        Check(result.Status == "infinite" && result.X == null, "infinite sentinel");
        result = SolveRational(new[] { new[] { 1, 1 }, new[] { 2, 2 } }, new long[] { 1, 5 });
        Check(result.Status == "inconsistent" && result.X == null, "inconsistent sentinel");
        result = SolveRational(
            new[] { new[] { 1, 0, 0 }, new[] { 0, 1, 0 }, new[] { 1, 1, 0 } }, new long[] { 1, 2, 3 });
        Check(result.Status == "infinite" && result.X == null, "3x3 dependent infinite");
        result = SolveRational(
            new[] { new[] { 1, 0, 0 }, new[] { 0, 1, 0 }, new[] { 1, 1, 0 } }, new long[] { 1, 2, 4 });
        Check(result.Status == "inconsistent" && result.X == null, "3x3 dependent inconsistent");

        // --- float cross-check: the double solver agrees to 1e-9 ---------------
        int[][][] matrices =
        {
            new[] { new[] { 1, 1 }, new[] { 1, -1 } },
            new[] { new[] { 0, 1 }, new[] { 1, 0 } },
            new[] { new[] { 1, 1, 1 }, new[] { 0, 2, 5 }, new[] { 2, 5, -1 } },
            new[] { new[] { 3 } },
            new[] { new[] { 2, 4 }, new[] { 1, -1 } },
            new[] { new[] { 5, 2 }, new[] { 1, 1 } },
        };
        long[][] rhss = { new long[] { 3, 1 }, new long[] { 2, 3 }, new long[] { 6, -4, 27 },
            new long[] { 1 }, new long[] { 1, 0 }, new long[] { 1, 0 } };
        long[][] expectedNums = { new long[] { 2, 1 }, new long[] { 3, 2 }, new long[] { 5, 3, -2 },
            new long[] { 1 }, new long[] { 1, 1 }, new long[] { 1, -1 } };
        long[][] expectedDens = { new long[] { 1, 1 }, new long[] { 1, 1 }, new long[] { 1, 1, 1 },
            new long[] { 3 }, new long[] { 6, 6 }, new long[] { 3, 3 } };
        for (int s = 0; s < matrices.Length; s++)
        {
            double[]? approx = SolveDouble(matrices[s], rhss[s]);
            Check(approx != null, "float cross-check solvable " + s);
            for (int i = 0; i < approx!.Length; i++)
            {
                double exact = (double)expectedNums[s][i] / expectedDens[s][i];
                Check(Math.Abs(approx[i] - exact) < 1e-9, "float cross-check " + s + " index " + i);
            }
        }

        // --- LCG cross-checks: exact recovery of the planted solution ----------
        var lcg = new Lcg(37613);
        int uniqueCount = 0;
        for (int trial = 0; trial < 80; trial++)
        {
            (int[][] matrix, long[] rhs, int[] planted) = LcgSystem(lcg);
            result = SolveRational(matrix, rhs);
            if (result.Status != "unique") continue;              // singular; skipped
            uniqueCount++;
            var plantedNum = new long[planted.Length];
            var plantedDen = new long[planted.Length];
            for (int i = 0; i < planted.Length; i++)
            {
                plantedNum[i] = planted[i];
                plantedDen[i] = 1;
            }
            Check(SameRationals(result.X!, plantedNum, plantedDen), "planted solution " + trial);
            // exact substitution A x == b, no tolerance involved
            for (int i = 0; i < matrix.Length; i++)
            {
                Rational total = new Rational(0);
                for (int j = 0; j < matrix[i].Length; j++)
                {
                    total = total.Add(new Rational(matrix[i][j]).Mul(result.X![j]));
                }
                Check(total.Equals(new Rational(rhs[i])), "exact substitution " + trial);
            }
            double[]? approx = SolveDouble(matrix, rhs);
            Check(approx != null, "float path solvable " + trial);
            for (int i = 0; i < planted.Length; i++)
            {
                Check(Math.Abs(approx![i] - planted[i]) < 1e-9, "float agrees " + trial);
            }
        }
        Check(uniqueCount >= 40, "enough unique systems, got " + uniqueCount);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
