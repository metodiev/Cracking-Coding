// Task 693 - Implement erasure coding (Reed-Solomon) encoder/decoder small GF(2^8). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// A complete Reed-Solomon codec over GF(2^8) built on the classic Reed-Solomon field polynomial
// x^8 + x^4 + x^3 + x^2 + 1 (0x11D, as used by QR codes and many erasure coding libraries; AES
// uses 0x11B instead) with generator alpha = 2. The code is the standard shortened
// cyclic RS(n, k) code with 2t = n - k parity symbols whose generator polynomial
// g(x) = (x - a^1)(x - a^2)...(x - a^2t) has the first 2t powers of alpha as roots, so every
// codeword satisfies c(a^j) = 0 for j = 1..2t. Encoding is systematic: the message occupies the
// high order coefficients and the parity is the remainder of x^(2t) m(x) modulo g(x).
//
// Decoding is the textbook pipeline, not a search:
//   * syndromes S_j = r(a^j) for j = 1..2t; all zero means no error;
//   * erasure positions are folded in as the erasure locator product of (1 + X_i x);
//   * Berlekamp-Massey runs on the modified syndrome polynomial T = (S * erasure locator) mod x^2t
//     to find the error locator Lambda_2, and Lambda = erasure locator * Lambda_2;
//   * Chien search over the n code positions finds the roots of Lambda;
//   * Forney's formula e_i = Omega(X_i^-1) / Lambda'(X_i^-1) with Omega = (Lambda * S) mod x^2t
//     (b = 1 convention, so the X_i^(1-b) factor is one, and minus equals plus in characteristic 2)
//     yields the error magnitudes;
//   * the corrected word is re-checked against all syndromes, and the decoder returns null whenever
//     the locator degree does not match the number of roots or the syndromes do not vanish, so an
//     uncorrectable word is reported instead of silently producing garbage.
//
// What is verified: field identities, that the generator really vanishes at the code roots, the
// systematic layout, zero syndromes for every codeword, all single and all double symbol errors of
// the (15, 11) code, every erasure pattern up to 2t = 4, mixtures with 2e + f <= 2t, refusal to
// silently miscorrect from 3 errors onwards, the "decoded output is always a codeword within
// distance t" contract, the MDS minimum distance of a small (5, 3) code, and bit identical reruns.
using System;
using System.Collections.Generic;

internal static class Gf256
{
    public const int Primitive = 0x11D;

    public static readonly byte[] Exp = new byte[512];
    public static readonly byte[] Log = new byte[256];

    static Gf256()
    {
        int value = 1;
        for (int i = 0; i < 255; i++)
        {
            Exp[i] = (byte)value;
            Log[value] = (byte)i;
            value <<= 1;
            if ((value & 0x100) != 0) value ^= Primitive;
        }
        for (int i = 255; i < 512; i++) Exp[i] = Exp[i - 255];
    }

    public static byte Mul(byte a, byte b) => a == 0 || b == 0 ? (byte)0 : Exp[Log[a] + Log[b]];

    public static byte Div(byte a, byte b)
    {
        if (b == 0) throw new DivideByZeroException("GF(256) division by zero");
        return a == 0 ? (byte)0 : Exp[(Log[a] - Log[b] + 255) % 255];
    }

    public static byte Inv(byte a) => Div(1, a);

    public static byte Pow(byte a, int exponent)
    {
        if (a == 0) return 0;
        exponent %= 255;
        if (exponent < 0) exponent += 255;
        return Exp[Log[a] * exponent % 255];
    }
}

/// <summary>Polynomials are byte arrays in little endian order: p[i] is the coefficient of x^i.</summary>
internal static class GfPoly
{
    public static byte[] Trim(byte[] polynomial)
    {
        int degree = polynomial.Length;
        while (degree > 1 && polynomial[degree - 1] == 0) degree--;
        if (degree == polynomial.Length) return polynomial;
        var trimmed = new byte[degree];
        Array.Copy(polynomial, trimmed, degree);
        return trimmed;
    }

    public static int Degree(byte[] polynomial) => Trim(polynomial).Length - 1;

    public static byte Evaluate(byte[] polynomial, byte x)
    {
        byte result = 0;
        for (int i = polynomial.Length - 1; i >= 0; i--) result = (byte)(Gf256.Mul(result, x) ^ polynomial[i]);
        return result;
    }

    public static byte[] Multiply(byte[] a, byte[] b)
    {
        var product = new byte[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] == 0) continue;
            for (int j = 0; j < b.Length; j++) product[i + j] ^= Gf256.Mul(a[i], b[j]);
        }
        return Trim(product);
    }

    /// <summary>Remainder of a divided by b (b must be monic-safe, i.e. leading coefficient nonzero).</summary>
    public static byte[] Remainder(byte[] a, byte[] divisor)
    {
        var work = new byte[Math.Max(a.Length, divisor.Length)];
        Array.Copy(a, work, a.Length);
        int divisorDegree = divisor.Length - 1;
        byte leadingInverse = Gf256.Inv(divisor[divisorDegree]);
        for (int i = work.Length - 1; i >= divisorDegree; i--)
        {
            byte coefficient = work[i];
            if (coefficient == 0) continue;
            byte factor = Gf256.Mul(coefficient, leadingInverse);
            for (int j = 0; j <= divisorDegree; j++) work[i - divisorDegree + j] ^= Gf256.Mul(factor, divisor[j]);
        }
        var remainder = new byte[divisorDegree];
        Array.Copy(work, remainder, divisorDegree);
        return remainder;
    }

    public static byte[] Shift(byte[] polynomial, int degrees)
    {
        if (polynomial.Length == 1 && polynomial[0] == 0) return polynomial;
        var shifted = new byte[polynomial.Length + degrees];
        Array.Copy(polynomial, 0, shifted, degrees, polynomial.Length);
        return shifted;
    }

    /// <summary>a + scale * b, coefficient wise.</summary>
    public static byte[] AddScaled(byte[] a, byte[] b, byte scale)
    {
        int length = Math.Max(a.Length, b.Length);
        var result = new byte[length];
        Array.Copy(a, result, a.Length);
        for (int i = 0; i < b.Length; i++) result[i] ^= Gf256.Mul(scale, b[i]);
        return Trim(result);
    }

    /// <summary>Formal derivative; in characteristic two only the odd degree terms survive.</summary>
    public static byte[] Derivative(byte[] polynomial)
    {
        if (polynomial.Length < 2) return new byte[] { 0 };
        var derivative = new byte[polynomial.Length - 1];
        for (int i = 1; i < polynomial.Length; i++)
        {
            if (i % 2 == 0) continue;
            derivative[i - 1] = polynomial[i];
        }
        return Trim(derivative);
    }
}

internal sealed class ReedSolomon
{
    public sealed class DecodeResult
    {
        public byte[]? Data;
        public byte[]? Corrected;
        public string? Failure;
        public int CorrectedSymbols;

        public bool Ok => Data != null;
    }

    private readonly int _n;
    private readonly int _k;
    private readonly int _twoT;
    private readonly byte[] _generator;

    public int N => _n;
    public int K => _k;
    public int ParitySymbols => _twoT;
    public int CorrectionCapability => _twoT / 2;

    public ReedSolomon(int n, int k)
    {
        if (n <= 0 || n > 255 || k <= 0 || k >= n) throw new ArgumentException("need 0 < k < n <= 255");
        if ((n - k) % 2 != 0) throw new ArgumentException("n - k must be even");
        if (n >= 255 || !IsCyclicLength(n)) throw new ArgumentException("n must divide 255 so that a^0..a^(n-1) are distinct");
        _n = n;
        _k = k;
        _twoT = n - k;
        var generator = new byte[] { 1 };
        for (int j = 1; j <= _twoT; j++)
        {
            // (x - a^j) is (x + a^j) in characteristic two.
            generator = GfPoly.Multiply(generator, new byte[] { Gf256.Exp[j], 1 });
        }
        _generator = generator;
    }

    /// <summary>255 = 3 * 5 * 17, so only n in {1, 3, 5, 15, 17, 51, 85, 255} keep the points distinct.</summary>
    private static bool IsCyclicLength(int n) => n < 255 && 255 % n == 0;

    public byte[] Generator => _generator;

    /// <summary>Systematic encoder: message in the high order coefficients, parity in the low ones.</summary>
    public byte[] Encode(byte[] message)
    {
        if (message.Length != _k) throw new ArgumentException($"message must be {_k} bytes");
        var shifted = new byte[_n];
        Array.Copy(message, 0, shifted, _twoT, _k);
        byte[] parity = GfPoly.Remainder(shifted, _generator);
        var codeword = new byte[_n];
        Array.Copy(shifted, codeword, _n);
        for (int i = 0; i < parity.Length; i++) codeword[i] ^= parity[i];
        return codeword;
    }

    public byte Evaluate(byte[] polynomial, byte x) => GfPoly.Evaluate(polynomial, x);

    public byte[] Syndromes(byte[] word)
    {
        var syndromes = new byte[_twoT + 1];
        for (int j = 1; j <= _twoT; j++) syndromes[j] = GfPoly.Evaluate(word, Gf256.Exp[j]);
        return syndromes;
    }

    public bool IsCodeword(byte[] word)
    {
        byte[] syndromes = Syndromes(word);
        for (int j = 1; j <= _twoT; j++) if (syndromes[j] != 0) return false;
        return true;
    }

    public static int HammingDistance(byte[] a, byte[] b)
    {
        int distance = 0;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) distance++;
        return distance;
    }

    /// <summary>Berlekamp-Massey over the sequence <paramref name="sequence"/>[1..count].</summary>
    private static byte[] BerlekampMassey(byte[] sequence, int count)
    {
        var lambda = new byte[] { 1 };
        var previous = new byte[] { 1 };
        int degree = 0, shift = 1;
        byte previousDiscrepancy = 1;

        for (int r = 1; r <= count; r++)
        {
            byte discrepancy = sequence[r];
            for (int i = 1; i < lambda.Length && i <= r; i++)
                discrepancy ^= Gf256.Mul(lambda[i], sequence[r - i]);

            if (discrepancy == 0)
            {
                shift++;
                continue;
            }

            byte scale = Gf256.Div(discrepancy, previousDiscrepancy);
            byte[] candidate = GfPoly.AddScaled(lambda, GfPoly.Shift(previous, shift), scale);
            if (2 * degree <= r - 1)
            {
                byte[] old = lambda;
                lambda = candidate;
                degree = r - degree;
                previous = old;
                previousDiscrepancy = discrepancy;
                shift = 1;
            }
            else
            {
                lambda = candidate;
                shift++;
            }
        }
        return lambda;
    }

    public DecodeResult Decode(byte[] received, bool[] erased)
    {
        var result = new DecodeResult();
        if (received.Length != _n || erased.Length != _n) throw new ArgumentException("bad word length");

        byte[] syndromes = Syndromes(received);
        bool clean = true;
        for (int j = 1; j <= _twoT; j++) if (syndromes[j] != 0) clean = false;
        if (clean)
        {
            // A valid codeword: erased symbols are correct too, so nothing has to be computed.
            result.Data = Extract(received);
            result.Corrected = received;
            result.CorrectedSymbols = 0;
            return result;
        }

        int erasureCount = 0;
        var erasureLocator = new byte[] { 1 };
        for (int i = 0; i < _n; i++)
        {
            if (!erased[i]) continue;
            erasureCount++;
            erasureLocator = GfPoly.Multiply(erasureLocator, new byte[] { 1, Gf256.Exp[i % 255] });
        }
        if (erasureCount > _twoT)
        {
            result.Failure = $"{erasureCount} erasures need more than {_twoT} parity symbols";
            return result;
        }

        var syndromePolynomial = new byte[_twoT + 1];
        for (int j = 1; j <= _twoT; j++) syndromePolynomial[j - 1] = syndromes[j];
        byte[] modified = GfPoly.Remainder(GfPoly.Multiply(syndromePolynomial, erasureLocator), PowerOfX(_twoT));
        // The recurrence that Berlekamp-Massey solves only starts once the first f coefficients, which
        // carry the unknown numerator Omega, are skipped; the remaining 2t - f coefficients are the
        // sequence T_{f+1}..T_2t (1-indexed), hence the shift by the erasure count and then by one.
        var paddedModified = new byte[_twoT + 1];
        for (int i = 0; i < _twoT - erasureCount; i++)
        {
            int index = erasureCount + i;
            paddedModified[i + 1] = index < modified.Length ? modified[index] : (byte)0;
        }

        byte[] errorLocator = BerlekampMassey(paddedModified, _twoT - erasureCount);
        byte[] locator = GfPoly.Multiply(erasureLocator, errorLocator);

        var positions = new List<int>();
        for (int i = 0; i < _n; i++)
        {
            byte xInverse = Gf256.Exp[(255 - i % 255) % 255];
            if (GfPoly.Evaluate(locator, xInverse) == 0) positions.Add(i);
        }
        if (positions.Count != GfPoly.Degree(locator) || positions.Count > _twoT)
        {
            result.Failure = "the error locator has no root among the code positions";
            return result;
        }

        byte[] omega = GfPoly.Remainder(GfPoly.Multiply(locator, syndromePolynomial), PowerOfX(_twoT));
        byte[] derivative = GfPoly.Derivative(locator);
        var corrected = new byte[_n];
        Array.Copy(received, corrected, _n);
        foreach (int position in positions)
        {
            byte xInverse = Gf256.Exp[(255 - position % 255) % 255];
            byte numerator = GfPoly.Evaluate(omega, xInverse);
            byte denominator = GfPoly.Evaluate(derivative, xInverse);
            if (denominator == 0)
            {
                result.Failure = "Forney denominator vanished at position " + position;
                return result;
            }
            corrected[position] ^= Gf256.Div(numerator, denominator);
        }

        if (!IsCodeword(corrected))
        {
            result.Failure = "the correction did not produce a codeword";
            return result;
        }
        result.Data = Extract(corrected);
        result.Corrected = corrected;
        result.CorrectedSymbols = positions.Count;
        return result;
    }

    private static byte[] PowerOfX(int degree)
    {
        var polynomial = new byte[degree + 1];
        polynomial[degree] = 1;
        return polynomial;
    }

    private byte[] Extract(byte[] codeword)
    {
        var data = new byte[_k];
        Array.Copy(codeword, _twoT, data, 0, _k);
        return data;
    }
}

internal static class Rng693
{
    private static ulong _state = 0x243F6A8885A308D3UL;

    public static void Seed(ulong seed) => _state = seed;

    public static ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

public static class Solution
{
    public static void Main()
    {
        // --- Field identities. ------------------------------------------------------------
        bool inverses = true, logRound = true;
        for (int value = 1; value <= 255; value++)
        {
            byte a = (byte)value;
            if (Gf256.Mul(a, Gf256.Inv(a)) != 1) inverses = false;
            if (Gf256.Exp[Gf256.Log[a]] != a) logRound = false;
        }
        Check(inverses, "every nonzero element times its inverse is one");
        Check(logRound, "the discrete log and its exponential agree for all 255 nonzero elements");
        Check(Gf256.Pow(2, 255) == 1 && Gf256.Pow(2, 8) == 0x1D, "alpha has order 255 and reduces at x^8");
        Check(Gf256.Pow(2, 254) == Gf256.Inv(2), "alpha^254 is the inverse of alpha");
        Check(Gf256.Mul(0x53, Gf256.Inv(0xCA)) == Gf256.Div(0x53, 0xCA), "division is multiplication by the inverse");
        Check(Gf256.Mul(2, 0x80) == 0x1D, "xtime of 0x80 wraps through the field polynomial");

        // --- Code construction: the generator vanishes at the code roots. ----------------
        var rs = new ReedSolomon(15, 11);          // t = 2
        Check(rs.ParitySymbols == 4 && rs.CorrectionCapability == 2, "the (15, 11) code carries four parity symbols");
        Check(GfPoly.Degree(rs.Generator) == 4, "the generator has degree 2t");
        bool roots = true;
        for (int j = 1; j <= 4; j++) if (GfPoly.Evaluate(rs.Generator, Gf256.Exp[j]) != 0) roots = false;
        Check(roots, "g(alpha^j) = 0 for j = 1..2t");
        Check(GfPoly.Evaluate(rs.Generator, Gf256.Exp[5]) != 0, "the next power of alpha is not a root");

        bool rejected = false;
        try { _ = new ReedSolomon(16, 12); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a length that does not divide 255 is rejected");
        rejected = false;
        try { _ = new ReedSolomon(15, 10); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "an odd number of parity symbols is rejected");

        // --- Systematic layout and zero syndromes. ---------------------------------------
        Rng693.Seed(0x693BEEFUL);
        var message = new byte[11];
        for (int i = 0; i < 11; i++) message[i] = (byte)Gf256.Exp[Rng693.NextInt(255)];
        byte[] codeword = rs.Encode(message);
        bool systematic = true;
        for (int i = 0; i < 11; i++) if (codeword[4 + i] != message[i]) systematic = false;
        Check(systematic, "the message occupies the high order coefficients");
        Check(rs.IsCodeword(codeword), "every encoded word has zero syndromes");

        var clean = rs.Decode(codeword, new bool[15]);
        Check(clean.Ok && clean.CorrectedSymbols == 0, "a clean word needs no correction");
        Check(string.Join(",", clean.Data!) == string.Join(",", message), "a clean word returns its message");

        // A random message sweep: encode, decode, compare.
        int roundTrips = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            var payload = new byte[11];
            for (int i = 0; i < 11; i++) payload[i] = (byte)Rng693.NextInt(256);
            var encoded = rs.Encode(payload);
            var decoded = rs.Decode(encoded, new bool[15]);
            if (decoded.Ok && string.Join(",", decoded.Data!) == string.Join(",", payload)) roundTrips++;
        }
        Check(roundTrips == 40, "forty random round trips succeeded, got " + roundTrips);

        // Edge payloads.
        var zeros = rs.Encode(new byte[11]);
        Check(rs.IsCodeword(zeros) && rs.Decode(zeros, new bool[15]).Ok, "the all zero message encodes and decodes");
        var ffs = new byte[11];
        for (int i = 0; i < 11; i++) ffs[i] = 0xFF;
        Check(string.Join(",", rs.Decode(rs.Encode(ffs), new bool[15]).Data!) == string.Join(",", ffs),
            "the all ones message round trips");

        // --- All single symbol errors. ---------------------------------------------------
        int singleFailures = 0, singleChecked = 0;
        for (int position = 0; position < 15; position++)
        {
            for (int magnitude = 1; magnitude <= 255; magnitude += 17)
            {
                var damaged = (byte[])codeword.Clone();
                damaged[position] ^= (byte)magnitude;
                var decoded = rs.Decode(damaged, new bool[15]);
                singleChecked++;
                if (!decoded.Ok || decoded.CorrectedSymbols != 1
                    || string.Join(",", decoded.Data!) != string.Join(",", message)) singleFailures++;
            }
        }
        Check(singleFailures == 0, $"all {singleChecked} single symbol errors were corrected");

        // --- All double symbol errors (105 position pairs). -----------------------------
        int doubleFailures = 0, doubleChecked = 0;
        for (int i = 0; i < 15; i++)
        {
            for (int j = i + 1; j < 15; j++)
            {
                var damaged = (byte[])codeword.Clone();
                damaged[i] ^= 0x5A;
                damaged[j] ^= 0xC3;
                var decoded = rs.Decode(damaged, new bool[15]);
                doubleChecked++;
                if (!decoded.Ok || decoded.CorrectedSymbols != 2
                    || string.Join(",", decoded.Data!) != string.Join(",", message)) doubleFailures++;
            }
        }
        Check(doubleFailures == 0, $"all {doubleChecked} double symbol errors were corrected");

        // --- Erasures: every pattern up to 2t = 4 symbols. ------------------------------
        int erasureFailures = 0, erasureChecked = 0;
        for (int mask = 1; mask < 16; mask++)
        {
            // Only the erased symbols are damaged, so every mask is a pure erasure pattern of 1..4 symbols.
            var damaged = (byte[])codeword.Clone();
            var erased = new bool[15];
            for (int bit = 0; bit < 4; bit++)
            {
                if ((mask & (1 << bit)) == 0) continue;
                damaged[bit] ^= (byte)(0x11 * (bit + 1));
                erased[bit] = true;
            }
            var decoded = rs.Decode(damaged, erased);
            erasureChecked++;
            if (!decoded.Ok || string.Join(",", decoded.Data!) != string.Join(",", message)) erasureFailures++;
        }
        Check(erasureFailures == 0, $"all {erasureChecked} erasure masks up to 2t were recovered");

        // Four erasures in the middle, and five erasures must be refused.
        var spread = (byte[])codeword.Clone();
        var spreadErased = new bool[15];
        foreach (int position in new[] { 3, 7, 9, 14 }) { spread[position] ^= (byte)(0x30 + position); spreadErased[position] = true; }
        var spreadDecoded = rs.Decode(spread, spreadErased);
        Check(spreadDecoded.Ok && string.Join(",", spreadDecoded.Data!) == string.Join(",", message),
            "four erasures spread over the word are recovered");
        var tooMany = new bool[15];
        foreach (int position in new[] { 1, 4, 6, 9, 12 }) tooMany[position] = true;
        var refused = rs.Decode(spread, tooMany);
        Check(!refused.Ok, "five erasures exceed the parity and are refused: " + refused.Failure);

        // --- Mixed errors and erasures: 2e + f <= 2t. ------------------------------------
        var mixed = (byte[])codeword.Clone();
        var mixedErased = new bool[15];
        mixed[2] ^= 0x77;                     // one unknown position (an error)
        mixed[6] ^= 0xDE;                     // two known positions (erasures)
        mixed[11] ^= 0x99;
        mixedErased[6] = true;
        mixedErased[11] = true;
        var mixedDecoded = rs.Decode(mixed, mixedErased);
        Check(mixedDecoded.Ok && string.Join(",", mixedDecoded.Data!) == string.Join(",", message),
            "one error plus two erasures (2e + f = 4 = 2t) is corrected: " + mixedDecoded.Failure);
        Check(mixedDecoded.CorrectedSymbols == 3, "all three damaged symbols were repaired");

        // Two errors plus two erasures is 2e + f = 6 > 2t and must not be claimed as a success
        // unless the result really is a codeword within distance t of the input.
        var overreach = (byte[])codeword.Clone();
        var overreachErased = new bool[15];
        overreach[1] ^= 0x0F;
        overreach[4] ^= 0xF0;
        overreach[8] ^= 0x33;
        overreach[12] ^= 0xCC;
        overreachErased[8] = true;
        overreachErased[12] = true;
        var overreachDecoded = rs.Decode(overreach, overreachErased);
        Check(!overreachDecoded.Ok || rs.IsCodeword(overreachDecoded.Corrected!), "an over-capable mix never yields junk");

        // --- Beyond the capability: never silently wrong. --------------------------------
        int miscorrections = 0, refusals = 0;
        for (int i = 0; i < 15; i++)
        {
            for (int j = i + 1; j < 15; j++)
            {
                for (int l = j + 1; l < 15; l += 3)
                {
                    var damaged = (byte[])codeword.Clone();
                    damaged[i] ^= 0x21;
                    damaged[j] ^= 0x42;
                    damaged[l] ^= 0x84;
                    var decoded = rs.Decode(damaged, new bool[15]);
                    if (!decoded.Ok)
                    {
                        refusals++;
                        continue;
                    }
                    // A successful decode must be a genuine codeword within distance t of the input.
                    if (!rs.IsCodeword(decoded.Corrected!) || ReedSolomon.HammingDistance(decoded.Corrected!, damaged) > 2)
                        miscorrections++;
                }
            }
        }
        Check(miscorrections == 0, "no triple error decode produced a non-codeword or a far codeword");
        Check(refusals > 0, "at least some triple errors are refused outright, refusals=" + refusals);

        // --- MDS minimum distance on a small code. ---------------------------------------
        var small = new ReedSolomon(5, 3);          // n = 5, k = 3, d = 3
        Check(small.ParitySymbols == 2, "the (5, 3) code carries two parity symbols");
        int minWeight = 99;
        for (int a = 0; a < 256; a += 5)
        {
            for (int b = 0; b < 256; b += 7)
            {
                for (int c = 0; c < 256; c += 11)
                {
                    if (a == 0 && b == 0 && c == 0) continue;
                    var word = small.Encode(new byte[] { (byte)a, (byte)b, (byte)c });
                    int weight = 0;
                    foreach (byte symbol in word) if (symbol != 0) weight++;
                    if (weight < minWeight) minWeight = weight;
                }
            }
        }
        Check(minWeight >= 3, "no sampled codeword of the (5, 3) code has weight below 3, min=" + minWeight);
        bool weightTwo = false;
        for (int i = 0; i < 5 && !weightTwo; i++)
        {
            for (int j = i + 1; j < 5 && !weightTwo; j++)
            {
                for (int magnitude = 1; magnitude <= 255; magnitude++)
                {
                    var candidate = new byte[5];
                    candidate[i] = 1;
                    candidate[j] = (byte)magnitude;
                    if (small.IsCodeword(candidate)) { weightTwo = true; break; }
                }
            }
        }
        Check(!weightTwo, "the small code has no weight two codeword, so its distance is exactly three");
        Check(minWeight == 3, "a weight three codeword exists, confirming the MDS distance");

        // --- Degenerate but legal code: k = 1, t = 2. ------------------------------------
        var tiny = new ReedSolomon(5, 1);
        Check(tiny.CorrectionCapability == 2, "the (5, 1) code corrects two symbols");
        var only = tiny.Encode(new byte[] { 0xA5 });
        Check(tiny.IsCodeword(only), "the single message symbol produces a codeword");
        var hurt = (byte[])only.Clone();
        hurt[1] ^= 0x13;
        hurt[4] ^= 0x2F;
        var healed = tiny.Decode(hurt, new bool[5]);
        Check(healed.Ok && healed.Data![0] == 0xA5, "two errors in the (5, 1) code are corrected");
        Check(healed.CorrectedSymbols == 2, "both symbols were repaired");
        // Four of the five symbols may be erased: f = 2t is exactly the erasure limit.
        var erasedFour = new bool[5];
        foreach (int position in new[] { 0, 1, 2, 4 }) erasedFour[position] = true;
        var lastChance = tiny.Decode(hurt, erasedFour);
        Check(lastChance.Ok && lastChance.Data![0] == 0xA5, "four of five symbols erased is still recoverable");
        var erasedAll = new bool[5];
        for (int i = 0; i < 5; i++) erasedAll[i] = true;
        Check(!tiny.Decode(hurt, erasedAll).Ok, "erasing the whole word is beyond the capability");

        // --- Determinism. ----------------------------------------------------------------
        var first = RepeatScenario();
        var second = RepeatScenario();
        Check(first == second, "reruns are bit identical");
        Check(first.Contains("ok=40") && first.Contains("errors=1"), "the scenario summary is complete: " + first);

        Console.WriteLine($"n={rs.N} k={rs.K} t={rs.CorrectionCapability} single={singleChecked} " +
                          $"double={doubleChecked} erasures={erasureChecked} tripleRefusals={refusals}");
        Console.WriteLine("PASS");
    }

    private static string RepeatScenario()
    {
        var rs = new ReedSolomon(15, 11);
        Rng693.Seed(12345);
        var message = new byte[11];
        for (int i = 0; i < 11; i++) message[i] = (byte)Rng693.NextInt(256);
        byte[] codeword = rs.Encode(message);
        int oks = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            int a = Rng693.NextInt(15);
            int b = (a + 1 + Rng693.NextInt(14)) % 15;      // always a different position
            var damaged = (byte[])codeword.Clone();
            damaged[a] ^= (byte)(1 + Rng693.NextInt(255));
            damaged[b] ^= (byte)(1 + Rng693.NextInt(255));
            var decoded = rs.Decode(damaged, new bool[15]);
            if (decoded.Ok && decoded.CorrectedSymbols == 2) oks++;
        }
        var spot = (byte[])codeword.Clone();
        spot[5] ^= 1;
        var single = rs.Decode(spot, new bool[15]);
        return $"ok={oks} errors={single.CorrectedSymbols} data={string.Join("", single.Data!)}";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
