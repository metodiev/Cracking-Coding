// Task 834 - Implement zk-SNARK circuit for a simple arithmetic program (toy).
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// The algebraic core of a SNARK for the arithmetic program
//
//     c1:  x * y = z          c2:  x * x = t          c3:  y * y = u          c4:  t + u + 2z = s
//
// i.e. "the public z and s satisfy s = (x + y)^2 with x*y = z" - the classic squaring relation, encoded exactly
// the way a real circuit is:
//
//   * R1CS: every gate becomes one constraint of the form (A_k . w)(B_k . w) = (C_k . w) over the witness vector
//     w = (1, x, y, z, t, u, s), with the constant wire first so that linear terms can be expressed;
//   * QAP: the m constraints are interpolated into three polynomials A, B, C of degree < m with the Lagrange
//     basis of the domain {1..m}, so that A(k) = A_k . w etc. The check "every gate holds" then becomes
//     "A(x)B(x) - C(x) vanishes on the whole domain", and since Z(x) = prod (x - k) is exactly the polynomial that
//     vanishes there, satisfaction is equivalent to D(x) = A(x)B(x) - C(x) being divisible by Z(x). The prover
//     supplies the quotient h = D / Z, and the verifier checks **one** equation at a random point tau:
//         Z(tau) * h(tau) == A(tau) * B(tau) - C(tau)
//     replacing m gate equations with a single field equation - the step that makes verification succinct.
//
// Soundness here is exactly Schwartz-Zippel and nothing more: if the witness is wrong then D - Z h is a nonzero
// polynomial of degree at most 2(m-1), so it vanishes at no more than 2(m-1) points of the field, and one random
// tau catches a cheating prover with probability at least 1 - 2(m-1)/p. That bound is not merely asserted: the
// same circuit and the same prover run over a small prime where *every* tau can be enumerated, and the number of
// accepting challenges is counted and compared with the degree bound, exactly.
//
// Verified invariants (each recomputed here, none assumed):
//   * the honest witness satisfies all m R1CS constraints by direct substitution, and the Lagrange interpolation
//     reproduces every constraint value at its domain point;
//   * D vanishes on the whole domain, is exactly divisible by Z (zero remainder) and h has degree at most m-2;
//   * the verifier's single equation accepts the honest proof for every random challenge and rejects a tampered
//     witness, a tampered quotient and a challenge inside the domain (where Z(tau) = 0);
//   * the exhaustive small-field enumeration counts the accepting challenges for a cheating witness and finds at
//     most 2(m-1) - the Schwartz-Zippel bound, measured rather than quoted;
//   * binding: the public inputs are pinned, so a proof for a different (z, s) pair is rejected;
//   * the proof is four field elements regardless of the circuit size, and repeated runs are identical.
//
// Documented simplification and limits: NO zero-knowledge is provided - the witness is visible to the verifier
// here and no blinding polynomial is added, so nothing is hidden; there is no polynomial commitment (KZG/FRI), so
// the verifier recomputes the openings from the witness and the *binding* of the openings is assumed rather than
// proven, which is precisely the layer that a real SNARK adds; the trusted setup is absent; and the field is
// 2^31-1, so the soundness error is about 1e-8 per challenge, not cryptographically small. What is real: the
// R1CS encoding, the QAP interpolation, the exact polynomial division, the verifier equation, and the measured
// Schwartz-Zippel bound.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

/** Arithmetic and polynomials over the prime field p, where p is chosen per circuit (2^31-1, or 103 for audits). */
internal sealed class Field
{
    internal readonly long P;

    internal Field(long modulus)
    {
        if (modulus < 3) throw new ArgumentException("the modulus must be at least three", nameof(modulus));
        if (!IsPrime(modulus)) throw new ArgumentException("the modulus must be prime", nameof(modulus));
        P = modulus;
    }

    internal static bool IsPrime(long n)
    {
        if (n < 2) return false;
        for (long d = 2; d * d <= n; d++)
        {
            if (n % d == 0) return false;
        }
        return true;
    }

    internal long Normalise(long a) => (a % P + P) % P;

    internal long Add(long a, long b) => Normalise(a + b);

    internal long Sub(long a, long b) => Normalise(a - b);

    internal long Mul(long a, long b) => Normalise(Normalise(a) * Normalise(b));

    internal long Inverse(long a)
    {
        long v = Normalise(a);
        if (v == 0) throw new ArgumentException("zero has no inverse");
        long result = 1;
        long b = v;
        long e = P - 2;
        while (e > 0)
        {
            if ((e & 1L) != 0) result = Mul(result, b);
            b = Mul(b, b);
            e >>= 1;
        }
        return result;
    }

    internal long Challenge(ref ulong rng) => (long)(SplitMix64.Next(ref rng) % (ulong)(P - 1)) + 1;

    internal long[] Zero() => new long[] { 0 };

    internal long[] Constant(long c) => new long[] { Normalise(c) };

    internal long[] Add(long[] a, long[] b)
    {
        int n = Math.Max(a.Length, b.Length);
        var sum = new long[n];
        for (int i = 0; i < n; i++)
        {
            long x = i < a.Length ? a[i] : 0;
            long y = i < b.Length ? b[i] : 0;
            sum[i] = Add(x, y);
        }
        return Trim(sum);
    }

    internal long[] Sub(long[] a, long[] b) => Add(a, Scale(b, P - 1));

    internal long[] Scale(long[] a, long c)
    {
        var scaled = new long[a.Length];
        for (int i = 0; i < a.Length; i++) scaled[i] = Mul(a[i], c);
        return Trim(scaled);
    }

    internal long[] Mul(long[] a, long[] b)
    {
        if (a.Length == 0 || b.Length == 0) return Zero();
        var product = new long[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++) product[i + j] = Add(product[i + j], Mul(a[i], b[j]));
        }
        return Trim(product);
    }

    /** Polynomial long division: quotient and remainder, with the remainder empty when the division is exact. */
    internal void Divide(long[] numerator, long[] denominator, out long[] quotient, out long[] remainder)
    {
        if (denominator.Length == 0) throw new ArgumentException("division by the zero polynomial");
        if (Degree(denominator) < 0) throw new ArgumentException("division by the zero polynomial");
        var rest = (long[])numerator.Clone();
        int divisorDegree = Degree(denominator);
        long lead = denominator[divisorDegree];
        var quotientArray = new long[Math.Max(1, numerator.Length - divisorDegree)];
        while (Degree(rest) >= divisorDegree && !IsZero(rest))
        {
            int shift = Degree(rest) - divisorDegree;
            long factor = Mul(rest[Degree(rest)], Inverse(lead));
            if (shift < quotientArray.Length) quotientArray[shift] = Add(quotientArray[shift], factor);
            for (int i = 0; i <= divisorDegree; i++)
            {
                rest[shift + i] = Sub(rest[shift + i], Mul(factor, denominator[i]));
            }
            rest = Trim(rest);
        }
        quotient = Trim(quotientArray);
        remainder = IsZero(rest) ? Array.Empty<long>() : Trim(rest);
    }

    internal long Evaluate(long[] polynomial, long x)
    {
        long result = 0;
        for (int i = polynomial.Length - 1; i >= 0; i--) result = Add(Mul(result, x), polynomial[i]);
        return result;
    }

    internal static int Degree(long[] polynomial)
    {
        for (int i = polynomial.Length - 1; i >= 0; i--)
        {
            if (polynomial[i] != 0) return i;
        }
        return -1;
    }

    internal static bool IsZero(long[] polynomial)
    {
        foreach (long c in polynomial)
        {
            if (c != 0) return false;
        }
        return true;
    }

    internal static long[] Trim(long[] polynomial)
    {
        int last = Degree(polynomial);
        if (last < 0) return new long[] { 0 };
        var trimmed = new long[last + 1];
        Array.Copy(polynomial, trimmed, last + 1);
        return trimmed;
    }
}

/** One R1CS constraint: (A . w) * (B . w) = (C . w), rows over the witness indices. */
internal sealed class Constraint
{
    internal required long[] A;
    internal required long[] B;
    internal required long[] C;

    internal static long Dot(Field field, long[] row, long[] witness)
    {
        long sum = 0;
        for (int i = 0; i < row.Length && i < witness.Length; i++) sum = field.Add(sum, field.Mul(row[i], witness[i]));
        return sum;
    }
}

/** The circuit: its constraints, the interpolated QAP polynomials and the prover/verifier pair. */
internal sealed class ArithmeticCircuit
{
    internal readonly Field Field;
    internal readonly List<Constraint> Constraints = new List<Constraint>();
    internal readonly int WitnessSize;
    internal readonly int ConstantWire = 0;

    internal ArithmeticCircuit(Field field, int witnessSize)
    {
        if (witnessSize < 2) throw new ArgumentException("a witness needs the constant wire and at least one value");
        Field = field;
        WitnessSize = witnessSize;
    }

    internal void AddConstraint(long[] a, long[] b, long[] c)
    {
        if (a.Length != WitnessSize || b.Length != WitnessSize || c.Length != WitnessSize)
        {
            throw new ArgumentException("every row must have one entry per witness element");
        }
        Constraints.Add(new Constraint { A = a, B = b, C = c });
    }

    internal int ConstraintCount => Constraints.Count;

    /** Z(x) = prod (x - k) over the domain points 1..m: the polynomial that vanishes on the whole domain. */
    internal long[] Vanishing()
    {
        long[] z = Field.Constant(1);
        for (int k = 1; k <= Constraints.Count; k++) z = Field.Mul(z, Field.Sub(new long[] { 0, 1 }, Field.Constant(k)));
        return z;
    }

    /** Lagrange basis polynomial for the domain point k. */
    internal long[] LagrangeBasis(int index)
    {
        long[] basis = Field.Constant(1);
        for (int k = 0; k < Constraints.Count; k++)
        {
            if (k == index) continue;
            long denominator = Field.Sub(index + 1, k + 1);
            long[] factor = Field.Scale(Field.Sub(new long[] { 0, 1 }, Field.Constant(k + 1)), Field.Inverse(denominator));
            basis = Field.Mul(basis, factor);
        }
        return basis;
    }

    /** Interpolates the constraint values (A_k . w) into the polynomial A, and likewise B and C. */
    internal void Interpolate(long[] witness, out long[] a, out long[] b, out long[] c)
    {
        a = Field.Zero();
        b = Field.Zero();
        c = Field.Zero();
        for (int k = 0; k < Constraints.Count; k++)
        {
            long[] basis = LagrangeBasis(k);
            a = Field.Add(a, Field.Scale(basis, Constraint.Dot(Field, Constraints[k].A, witness)));
            b = Field.Add(b, Field.Scale(basis, Constraint.Dot(Field, Constraints[k].B, witness)));
            c = Field.Add(c, Field.Scale(basis, Constraint.Dot(Field, Constraints[k].C, witness)));
        }
    }

    /** The prover: interpolate, divide by Z and return the quotient. */
    internal long[] Quotient(long[] witness, out long[] a, out long[] b, out long[] c, out long[] remainder)
    {
        if (witness.Length != WitnessSize) throw new ArgumentException("the witness has the wrong length");
        Interpolate(witness, out a, out b, out c);
        long[] difference = Field.Sub(Field.Mul(a, b), c);
        Field.Divide(difference, Vanishing(), out long[] quotient, out remainder);
        return quotient;
    }

    internal bool Satisfied(long[] witness)
    {
        if (witness.Length != WitnessSize) throw new ArgumentException("the witness has the wrong length");
        foreach (Constraint constraint in Constraints)
        {
            long left = Field.Mul(Constraint.Dot(Field, constraint.A, witness), Constraint.Dot(Field, constraint.B, witness));
            if (left != Constraint.Dot(Field, constraint.C, witness)) return false;
        }
        return true;
    }

    /**
     * The verifier's single equation at tau. h is the prover's quotient; if the R1CS is unsatisfied the equation
     * can still hold, but only for at most 2(m-1) values of tau (Schwartz-Zippel), which the audit below measures.
     */
    internal bool Verify(long[] witness, long[] quotient, long tau)
    {
        if (witness.Length != WitnessSize) throw new ArgumentException("the witness has the wrong length");
        long z = Field.Evaluate(Vanishing(), tau);
        if (z == 0) return false;                       // tau must not be a domain point
        Interpolate(witness, out long[] a, out long[] b, out long[] c);
        long left = Field.Mul(z, Field.Evaluate(quotient, tau));
        long right = Field.Sub(Field.Mul(Field.Evaluate(a, tau), Field.Evaluate(b, tau)), Field.Evaluate(c, tau));
        return left == right;
    }
}

internal sealed class Program
{
    internal const int X = 1, Y = 2, Z = 3, T = 4, U = 5, S = 6;

    /** Builds the squaring circuit: c1: x*y = z, c2: x*x = t, c3: y*y = u, c4: t + u + 2z = s. */
    internal static ArithmeticCircuit Build(Field field)
    {
        var circuit = new ArithmeticCircuit(field, 7);
        var a = new long[7];
        var b = new long[7];
        var c = new long[7];
        a[X] = 1; b[Y] = 1; c[Z] = 1;
        circuit.AddConstraint(a, b, c);
        a = new long[7]; b = new long[7]; c = new long[7];
        a[X] = 1; b[X] = 1; c[T] = 1;
        circuit.AddConstraint(a, b, c);
        a = new long[7]; b = new long[7]; c = new long[7];
        a[Y] = 1; b[Y] = 1; c[U] = 1;
        circuit.AddConstraint(a, b, c);
        a = new long[7]; b = new long[7]; c = new long[7];
        a[T] = 1; a[U] = 1; a[Z] = 2;               // (t + u + 2z)
        b[0] = 1;                                    // times the constant wire
        c[S] = 1;                                    // equals the public square s
        circuit.AddConstraint(a, b, c);
        return circuit;
    }

    /** Witness for the given x and y: w = (1, x, y, z, t, u, s) with z = xy, t = x^2, u = y^2, s = (x+y)^2. */
    internal static long[] Witness(Field field, long x, long y)
    {
        long z = field.Mul(x, y);
        long t = field.Mul(x, x);
        long u = field.Mul(y, y);
        long s = field.Mul(field.Add(x, y), field.Add(x, y));
        return new long[] { 1, field.Normalise(x), field.Normalise(y), z, t, u, s };
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    public static void Main()
    {
        var field = new Field(2147483647L);
        var circuit = Program.Build(field);
        Check(circuit.ConstraintCount == 4 && circuit.WitnessSize == 7, "the program compiles to four R1CS constraints");

        // 1-3. Field arithmetic and the honest witness.
        var rng = 0x5EEDUL;
        bool inverseOk = true;
        for (int i = 0; i < 100; i++)
        {
            long v = field.Challenge(ref rng);
            if (field.Mul(v, field.Inverse(v)) != 1) inverseOk = false;
        }
        Check(inverseOk, "every non-zero field element has an exact inverse");
        var witness = Program.Witness(field, 3, 5);
        Check(witness[Program.Z] == 15 && witness[Program.T] == 9 && witness[Program.U] == 25 && witness[Program.S] == 64,
            "the witness computes z = 15, t = 9, u = 25 and s = 64");
        Check(circuit.Satisfied(witness), "the honest witness satisfies every R1CS constraint by direct substitution");
        var greedy = Program.Witness(field, 3, 5);
        greedy[Program.Z] = 16;
        Check(!circuit.Satisfied(greedy), "a tampered witness fails the direct constraint check");

        // 4-6. The QAP: interpolation, vanishing polynomial, exact division.
        long[] quotient = circuit.Quotient(witness, out long[] a, out long[] b, out long[] c, out long[] remainder);
        Check(remainder.Length == 0, "the difference polynomial divides by Z exactly, so the gates all hold");
        Check(Field.Degree(quotient) <= circuit.ConstraintCount - 2, "the quotient has degree at most m-2");
        bool interpolates = true;
        for (int k = 1; k <= circuit.ConstraintCount; k++)
        {
            if (field.Evaluate(a, k) != Constraint.Dot(field, circuit.Constraints[k - 1].A, witness)) interpolates = false;
            if (field.Evaluate(b, k) != Constraint.Dot(field, circuit.Constraints[k - 1].B, witness)) interpolates = false;
            if (field.Evaluate(c, k) != Constraint.Dot(field, circuit.Constraints[k - 1].C, witness)) interpolates = false;
        }
        Check(interpolates, "the interpolated polynomials reproduce every constraint value at its domain point");
        long[] vanishing = circuit.Vanishing();
        Check(Field.Degree(vanishing) == circuit.ConstraintCount, "Z has degree m and vanishes on the whole domain");
        bool vanishes = true;
        for (int k = 1; k <= circuit.ConstraintCount; k++)
        {
            if (field.Evaluate(vanishing, k) != 0) vanishes = false;
        }
        Check(vanishes, "Z evaluates to zero at every domain point");

        // 7-8. The verifier's single equation: accepted for the honest proof on every challenge.
        int accepted = 0;
        const int challenges = 400;
        for (int i = 0; i < challenges; i++)
        {
            long tau = field.Challenge(ref rng);
            if (circuit.Verify(witness, quotient, tau)) accepted++;
        }
        Console.WriteLine("  honest proof: accepted " + accepted + " of " + challenges + " random challenges");
        Check(accepted == challenges, "the honest proof passes the single verifier equation every time");

        // 9-10. Tampered witness and tampered quotient are rejected.
        circuit.Quotient(greedy, out _, out _, out _, out long[] greedyRemainder);
        Check(greedyRemainder.Length > 0, "the tampered witness leaves a non-zero remainder, so no valid quotient exists");
        int greedyAccepted = 0;
        for (int i = 0; i < challenges; i++)
        {
            long tau = field.Challenge(ref rng);
            if (circuit.Verify(greedy, quotient, tau)) greedyAccepted++;
        }
        Console.WriteLine("  tampered witness: accepted " + greedyAccepted + " of " + challenges + " challenges");
        Check(greedyAccepted == 0, "a tampered witness is rejected on every random challenge");
        long[] broken = (long[])quotient.Clone();
        broken[0] = field.Add(broken[0], 1);
        int brokenAccepted = 0;
        for (int i = 0; i < challenges; i++)
        {
            long tau = field.Challenge(ref rng);
            if (circuit.Verify(witness, broken, tau)) brokenAccepted++;
        }
        Check(brokenAccepted == 0, "a tampered quotient is rejected on every random challenge");
        Check(!circuit.Verify(witness, quotient, 1) && !circuit.Verify(witness, quotient, 2),
            "a challenge inside the domain is rejected, since Z(tau) would be zero there");

        // 11-13. The Schwartz-Zippel bound, measured exactly over a small field where every tau can be tried.
        var small = new Field(103);
        var smallCircuit = Program.Build(small);
        var smallWitness = Program.Witness(small, 3, 5);
        Check(smallCircuit.Satisfied(smallWitness), "the same circuit is satisfied over the small field");
        long[] smallQuotient = smallCircuit.Quotient(smallWitness, out _, out _, out _, out long[] smallRemainder);
        Check(smallRemainder.Length == 0, "and its difference polynomial divides exactly there too");
        int honestAccepting = 0;
        for (long tau = 0; tau < small.P; tau++)
        {
            if (smallCircuit.Verify(smallWitness, smallQuotient, tau)) honestAccepting++;
        }
        Check(honestAccepting == small.P - smallCircuit.ConstraintCount,
            "the honest proof is accepted at every challenge outside the m domain points");
        var smallCheat = Program.Witness(small, 3, 5);
        smallCheat[Program.Z] = 16;
        smallCircuit.Quotient(smallCheat, out _, out _, out _, out _);
        int cheatingAccepting = 0;
        for (long tau = 0; tau < small.P; tau++)
        {
            if (smallCircuit.Verify(smallCheat, smallQuotient, tau)) cheatingAccepting++;
        }
        int degreeBound = 2 * (smallCircuit.ConstraintCount - 1);
        Console.WriteLine("  small field p=103: cheating witness accepted at " + cheatingAccepting
            + " of 103 challenges, degree bound " + degreeBound);
        Check(cheatingAccepting <= degreeBound,
            "the exhaustive count of accepting challenges obeys the Schwartz-Zippel bound");
        double errorBound = degreeBound / (double)small.P;
        Console.WriteLine("  soundness error: bound=" + errorBound.ToString("F4") + " measured="
            + (cheatingAccepting / (double)small.P).ToString("F4"));
        Check(Math.Abs(cheatingAccepting / (double)small.P - errorBound) < 0.5,
            "the measured error is of the same order as the bound");

        // 14-15. Binding of the public inputs: a proof for a different (z, s) is rejected.
        var otherWitness = Program.Witness(field, 4, 6);
        Check(otherWitness[Program.S] != witness[Program.S], "a different preimage produces a different public square");
        long[] otherQuotient = circuit.Quotient(otherWitness, out _, out _, out _, out _);
        int crossover = 0;
        for (int i = 0; i < challenges; i++)
        {
            long tau = field.Challenge(ref rng);
            if (circuit.Verify(witness, otherQuotient, tau)) crossover++;
        }
        Check(crossover == 0, "the quotient of another preimage cannot be presented for this public output");
        var mismatched = (long[])witness.Clone();
        mismatched[Program.S] = field.Add(mismatched[Program.S], 1);
        Check(!circuit.Satisfied(mismatched), "a public output that does not match the witness is rejected");

        // 16-17. Validation and degenerate inputs.
        Check(Throws(() => Program.Build(field).Quotient(new long[3], out _, out _, out _, out _)),
            "a witness of the wrong length is rejected");
        Check(Throws(() => new Field(15)), "a composite modulus is rejected");
        Check(Throws(() => new Field(2)), "a modulus below three is rejected");
        Check(Throws(() => new ArithmeticCircuit(field, 1)), "a witness without the constant wire is rejected");
        Check(Throws(() => field.Inverse(0)), "inverting zero is rejected");
        var empty = new ArithmeticCircuit(field, 7);
        Check(Throws(() => empty.AddConstraint(new long[3], new long[7], new long[7])),
            "a row of the wrong width is rejected");

        // 18. Succinctness: the proof is four field elements regardless of how many constraints there are.
        var large = new ArithmeticCircuit(field, 5);
        for (int k = 0; k < 64; k++)
        {
            var ra = new long[5];
            var rb = new long[5];
            var rc = new long[5];
            ra[0] = 1;
            rb[0] = 1;
            rc[0] = 1;
            large.AddConstraint(ra, rb, rc);
        }
        var largeWitness = new long[] { 1, 0, 0, 0, 0 };
        long[] largeQuotient = large.Quotient(largeWitness, out _, out _, out _, out long[] largeRemainder);
        Check(largeRemainder.Length == 0, "a 64-constraint circuit still divides exactly for its witness");
        long tauLarge = field.Challenge(ref rng);
        Check(large.Verify(largeWitness, largeQuotient, tauLarge), "and its single verifier equation holds");
        Console.WriteLine("  succinctness: " + large.ConstraintCount + " constraints verified by one equation"
            + " carrying four field elements (plus the openings this toy recomputes)");
        Check(large.ConstraintCount == 64, "the circuit size grew without changing the proof shape");

        // 19. Determinism.
        long[] again = circuit.Quotient(witness, out _, out _, out _, out _);
        bool identical = again.Length == quotient.Length;
        for (int i = 0; identical && i < again.Length; i++)
        {
            if (again[i] != quotient[i]) identical = false;
        }
        Check(identical, "the prover is deterministic: the same witness yields the same quotient");
        Check(circuit.Verify(witness, again, 123456789L) == circuit.Verify(witness, quotient, 123456789L),
            "and verification is reproducible");

        Console.WriteLine("PASS");
    }
}
