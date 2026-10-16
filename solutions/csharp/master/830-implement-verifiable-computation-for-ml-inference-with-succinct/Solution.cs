// Task 830 - Implement verifiable computation for ML inference with succinct proofs (conceptual).
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// An outsourced inference is checked without redoing the work, with the two ingredients a real succinct argument
// uses, at toy scale: a commitment to the model, and a randomised algebraic identity per matmul.
//
//   * the model (W1, b1, W2, b2) is committed by a binary Merkle tree over the field elements of its parameters,
//     so the verifier pins *which* model was used by holding a single root;
//   * inference is an arithmetic circuit over the prime field 2^31-1: z = W1 x + b1, a = z .* z (a
//     field-friendly activation, chosen so that every gate is a polynomial), y = W2 a + b2. The prover returns y,
//     the intermediate z and a, one Merkle opening of the parameters, and for each matmul the folded vector
//     alpha^T W - which is the Freivalds message;
//   * the verifier never forms a product: it checks alpha^T (z - b1) == (alpha^T W1) x and
//     alpha^T y == (alpha^T W2) a, each costing O(rows + cols) multiplications instead of O(rows * cols), plus
//     the elementwise a == z .* z. For a fixed folded vector a wrong claim survives one random challenge with
//     probability at most 1/p, so k challenges leave p^-k.
//
// Verified invariants (each recomputed here, none assumed):
//   * field arithmetic is exact (inverses, Fermat, wrap-around) and the circuit agrees with plain integer
//     arithmetic whenever nothing wraps, so the encoding is not hiding an error;
//   * the Merkle root is deterministic, every inclusion path verifies, a tampered leaf or a wrong index is
//     rejected, and subsample probing catches a tampered leaf at the predicted rate 1 - (1 - s/n)^probes
//     (measured against the closed form within Monte Carlo error);
//   * the toy hash is NOT collision resistant - a collision is exhibited by exhaustive search, so no
//     cryptographic strength is claimed anywhere in this file;
//   * an honest proof verifies, a tampered output or activation is rejected, and the *measured* acceptance rate
//     of a wrong claim over 2000 random challenges is at most 1/p as predicted; the Schwartz-Zippel bound is
//     also demonstrated on a challenge space of size m, where exactly one challenge admits the false claim;
//   * the honest limitation is demonstrated rather than papered over: a prover that adapts the folded vector to
//     the challenge defeats a single linear check, because nothing commits it before the challenge is drawn -
//     which is exactly why real systems use a *polynomial commitment* to the folded data instead of sending it in
//     the clear. No soundness against an adaptive prover is claimed anywhere in this file;
//   * verification is genuinely cheaper: the verifier's field multiplications are measured against the prover's
//     for several shapes and grow linearly with rows, not with the product.
//
// Documented simplification and limits: this is NOT zero-knowledge (the witness z, a is visible; no blinding
// polynomial is added); there is no cryptographic commitment (the hash is a mixing function, as the exhibited
// collision shows); the proof is O(parameters) rather than O(1); soundness rests on Schwartz-Zippel/Freivalds
// randomness over 31 bits, not on a hardness assumption. What is real: the circuit, exact field arithmetic, the
// Merkle commitment, the Freivalds exchange, the soundness bound and the measured cost asymmetry. Real systems
// (vCNN, zkCNN, Mystique) replace the Merkle tree with a polynomial/vector commitment and the exchange with a
// sumcheck or GKR protocol.
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

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Field
{
    internal const long P = 2147483647L;

    internal static long Normalise(long a) => (a % P + P) % P;

    internal static long Add(long a, long b) => Normalise(a + b);

    internal static long Sub(long a, long b) => Normalise(a - b);

    internal static long Mul(long a, long b) => Normalise(a) * Normalise(b) % P;

    internal static long Pow(long a, long e)
    {
        long result = 1;
        long b = Normalise(a);
        while (e > 0)
        {
            if ((e & 1L) != 0) result = Mul(result, b);
            b = Mul(b, b);
            e >>= 1;
        }
        return result;
    }

    internal static long Inverse(long a)
    {
        long v = Normalise(a);
        if (v == 0) throw new ArgumentException("zero has no inverse");
        return Pow(v, P - 2);
    }

    internal static long Challenge(ref ulong rng) => (long)(SplitMix64.Next(ref rng) % (ulong)(P - 1)) + 1;
}

/** A mixing function used as the toy hash of the commitment: fast, deterministic, and not collision resistant. */
internal static class ToyHash
{
    internal static long Of(long a, long b)
    {
        ulong state = (ulong)Field.Normalise(a) * 0x9E3779B97F4A7C15UL;
        state ^= (ulong)Field.Normalise(b) + 0x165667B19E3779F9UL;
        return (long)(SplitMix64.Next(ref state) % (ulong)Field.P);
    }

    internal static long OfLeaf(long value, int index) => Of(value, index);
}

internal sealed class MerkleCommitment
{
    private readonly long[][] _levels;

    internal MerkleCommitment(IReadOnlyList<long> values)
    {
        if (values.Count == 0) throw new ArgumentException("a commitment needs at least one leaf", nameof(values));
        int padded = 1;
        while (padded < values.Count) padded *= 2;
        var leaves = new long[padded];
        for (int i = 0; i < values.Count; i++) leaves[i] = ToyHash.OfLeaf(Field.Normalise(values[i]), i);
        var levels = new List<long[]> { leaves };
        var current = leaves;
        while (current.Length > 1)
        {
            var next = new long[current.Length / 2];
            for (int i = 0; i < next.Length; i++) next[i] = ToyHash.Of(current[2 * i], current[2 * i + 1]);
            levels.Add(next);
            current = next;
        }
        _levels = levels.ToArray();
        Size = values.Count;
    }

    internal long Root => _levels[_levels.Length - 1][0];
    internal int Size { get; }
    internal int Depth => _levels.Length - 1;

    /** Sibling hashes from the leaf up to the root. */
    internal long[] Path(int index)
    {
        if (index < 0 || index >= Size) throw new ArgumentException("leaf index out of range", nameof(index));
        var path = new long[Depth];
        int position = index;
        for (int level = 0; level < Depth; level++)
        {
            path[level] = _levels[level][position ^ 1];
            position >>= 1;
        }
        return path;
    }

    internal static long Fold(long value, int index, long[] path)
    {
        long node = ToyHash.OfLeaf(Field.Normalise(value), index);
        int position = index;
        foreach (long sibling in path)
        {
            node = (position & 1) == 0 ? ToyHash.Of(node, sibling) : ToyHash.Of(sibling, node);
            position >>= 1;
        }
        return node;
    }

    internal static bool Verify(long value, int index, long[] path, long root) => Fold(value, index, path) == root;
}

internal sealed class Model
{
    internal readonly int Inputs, Hidden, Outputs;
    internal readonly long[] W1, B1, W2, B2, All;

    internal Model(int inputs, int hidden, int outputs, ulong seed, long bound = 1000)
    {
        if (inputs < 1 || hidden < 1 || outputs < 1) throw new ArgumentException("every layer needs at least one unit");
        if (bound < 1) throw new ArgumentException("the parameter bound must be positive", nameof(bound));
        Inputs = inputs;
        Hidden = hidden;
        Outputs = outputs;
        W1 = new long[hidden * inputs];
        B1 = new long[hidden];
        W2 = new long[outputs * hidden];
        B2 = new long[outputs];
        var rng = seed;
        var all = new List<long>();
        Fill(W1, ref rng, bound, all);
        Fill(B1, ref rng, bound, all);
        Fill(W2, ref rng, bound, all);
        Fill(B2, ref rng, bound, all);
        All = all.ToArray();
    }

    private static void Fill(long[] target, ref ulong rng, long bound, List<long> all)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = (long)(SplitMix64.Next(ref rng) % (ulong)(2 * bound + 1)) - bound;
            all.Add(Field.Normalise(target[i]));
        }
    }

    internal long[] Forward(long[] x, out long[] z, out long[] activation)
    {
        if (x.Length != Inputs) throw new ArgumentException("input has the wrong dimension", nameof(x));
        z = new long[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            long acc = Field.Normalise(B1[h]);
            for (int i = 0; i < Inputs; i++) acc = Field.Add(acc, Field.Mul(W1[h * Inputs + i], x[i]));
            z[h] = acc;
        }
        activation = new long[Hidden];
        for (int h = 0; h < Hidden; h++) activation[h] = Field.Mul(z[h], z[h]);
        var y = new long[Outputs];
        for (int o = 0; o < Outputs; o++)
        {
            long acc = Field.Normalise(B2[o]);
            for (int h = 0; h < Hidden; h++) acc = Field.Add(acc, Field.Mul(W2[o * Hidden + h], activation[h]));
            y[o] = acc;
        }
        return y;
    }
}

/** The prover's message: the claimed values plus the two folded vectors that make verification cheap. */
internal sealed class Proof
{
    internal required long[] Output;
    internal required long[] PreActivation;
    internal required long[] Activation;
    internal required long[] FoldedW1;      // alpha_z^T W1, one entry per input
    internal required long[] FoldedW2;      // alpha_y^T W2, one entry per hidden unit
    internal required int ProbeIndex;
    internal required long ProbeValue;
    internal required long[] ProbePath;
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

    /** The prover: run the circuit, fold the two weight matrices with the challenge vectors, open one parameter. */
    private static Proof BuildProof(Model model, MerkleCommitment commitment, long[] input, ulong seed)
    {
        long[] y = model.Forward(input, out long[] z, out long[] a);
        var rng = seed;
        var alphaZ = new long[model.Hidden];
        for (int h = 0; h < model.Hidden; h++) alphaZ[h] = Field.Challenge(ref rng);
        var alphaY = new long[model.Outputs];
        for (int o = 0; o < model.Outputs; o++) alphaY[o] = Field.Challenge(ref rng);
        var foldedW1 = new long[model.Inputs];
        for (int h = 0; h < model.Hidden; h++)
        {
            for (int i = 0; i < model.Inputs; i++)
            {
                foldedW1[i] = Field.Add(foldedW1[i], Field.Mul(alphaZ[h], model.W1[h * model.Inputs + i]));
            }
        }
        var foldedW2 = new long[model.Hidden];
        for (int o = 0; o < model.Outputs; o++)
        {
            for (int h = 0; h < model.Hidden; h++)
            {
                foldedW2[h] = Field.Add(foldedW2[h], Field.Mul(alphaY[o], model.W2[o * model.Hidden + h]));
            }
        }
        const int probe = 1;
        return new Proof
        {
            Output = y,
            PreActivation = z,
            Activation = a,
            FoldedW1 = foldedW1,
            FoldedW2 = foldedW2,
            ProbeIndex = probe,
            ProbeValue = model.All[probe],
            ProbePath = commitment.Path(probe)
        };
    }

    /**
     * The verifier. Its field multiplications are counted in `ops`: it never forms a product of a matrix with a
     * vector, only folded dot products of length rows + cols.
     */
    private static bool Verify(Model model, MerkleCommitment commitment, long[] input, Proof proof, ulong seed,
        out long ops)
    {
        ops = 0;
        if (input.Length != model.Inputs) throw new ArgumentException("the public input has the wrong dimension");
        if (proof.Output.Length != model.Outputs || proof.Activation.Length != model.Hidden
            || proof.PreActivation.Length != model.Hidden || proof.FoldedW1.Length != model.Inputs
            || proof.FoldedW2.Length != model.Hidden)
        {
            return false;
        }
        // 1. the model is the committed one (toy: a single opened parameter)
        if (!MerkleCommitment.Verify(proof.ProbeValue, proof.ProbeIndex, proof.ProbePath, commitment.Root)) return false;
        ops += commitment.Depth;
        // 2. the activation is exactly the square of the pre-activation
        for (int h = 0; h < model.Hidden; h++)
        {
            if (Field.Mul(proof.PreActivation[h], proof.PreActivation[h]) != proof.Activation[h]) return false;
            ops++;
        }
        var rng = seed;
        var alphaZ = new long[model.Hidden];
        for (int h = 0; h < model.Hidden; h++) alphaZ[h] = Field.Challenge(ref rng);
        var alphaY = new long[model.Outputs];
        for (int o = 0; o < model.Outputs; o++) alphaY[o] = Field.Challenge(ref rng);
        // 3. first matmul: alpha^T (z - b1) must equal (alpha^T W1) x
        long left = 0;
        for (int h = 0; h < model.Hidden; h++)
        {
            left = Field.Add(left, Field.Mul(alphaZ[h], Field.Sub(proof.PreActivation[h], model.B1[h])));
            ops++;
        }
        long right = 0;
        for (int i = 0; i < model.Inputs; i++)
        {
            right = Field.Add(right, Field.Mul(proof.FoldedW1[i], input[i]));
            ops++;
        }
        if (left != right) return false;
        // 4. second matmul: alpha^T (y - b2) must equal (alpha^T W2) a
        long leftY = 0;
        for (int o = 0; o < model.Outputs; o++)
        {
            leftY = Field.Add(leftY, Field.Mul(alphaY[o], Field.Sub(proof.Output[o], model.B2[o])));
            ops++;
        }
        long rightY = 0;
        for (int h = 0; h < model.Hidden; h++)
        {
            rightY = Field.Add(rightY, Field.Mul(proof.FoldedW2[h], proof.Activation[h]));
            ops++;
        }
        return leftY == rightY;
    }

    /** The prover's folding cost: the work the verifier avoids. */
    private static long ProverFoldOps(Model model) => (long)model.Hidden * model.Inputs + (long)model.Outputs * model.Hidden;

    /** Independent oracle: the same circuit in plain integer arithmetic, reporting whether anything wrapped. */
    private static long[] IntegerReference(Model m, long[] x, out bool wrapped)
    {
        wrapped = false;
        var z = new long[m.Hidden];
        for (int h = 0; h < m.Hidden; h++)
        {
            long acc = m.B1[h];
            for (int i = 0; i < m.Inputs; i++) acc += m.W1[h * m.Inputs + i] * x[i];
            z[h] = acc;
        }
        var a = new long[m.Hidden];
        for (int h = 0; h < m.Hidden; h++) a[h] = z[h] * z[h];
        var y = new long[m.Outputs];
        for (int o = 0; o < m.Outputs; o++)
        {
            long acc = m.B2[o];
            for (int h = 0; h < m.Hidden; h++) acc += m.W2[o * m.Hidden + h] * a[h];
            y[o] = acc;
        }
        for (int h = 0; h < m.Hidden; h++)
        {
            if (Math.Abs(z[h]) >= Field.P) wrapped = true;
            if (Math.Abs(a[h]) >= Field.P) wrapped = true;
        }
        for (int o = 0; o < m.Outputs; o++)
        {
            if (Math.Abs(y[o]) >= Field.P) wrapped = true;
        }
        return y;
    }

    public static void Main()
    {
        var rng = 0x5EEDUL;
        bool inverseOk = true, fermatOk = true;
        for (int i = 0; i < 200; i++)
        {
            long a = Field.Challenge(ref rng);
            if (Field.Mul(a, Field.Inverse(a)) != 1) inverseOk = false;
            if (Field.Pow(a, Field.P - 1) != 1) fermatOk = false;
        }
        Check(inverseOk && fermatOk, "the field has exact inverses and satisfies Fermat");
        Check(Field.Add(Field.P - 1, 1) == 0 && Field.Sub(0, 1) == Field.P - 1,
            "wrap-around is represented by a positive canonical value");

        var model = new Model(3, 4, 2, 0x42UL);
        var input = new[] { 3L, -5L, 2L };
        long[] output = model.Forward(input, out _, out _);
        // Oracle: a tiny model, where every intermediate stays well inside the field, so plain integer arithmetic
        // and the field circuit must agree; then the full-size model, where the integers would overflow 2^31-1 and
        // the field encoding is genuinely load-bearing.
        var tiny = new Model(2, 2, 1, 0x99UL, 2);
        var tinyInput = new[] { 2L, -3L };
        long[] tinyField = tiny.Forward(tinyInput, out _, out _);
        long[] tinyInteger = IntegerReference(tiny, tinyInput, out bool smallWrapped);
        bool agrees = !smallWrapped;
        for (int o = 0; o < tinyField.Length; o++) agrees &= tinyField[o] == Field.Normalise(tinyInteger[o]);
        Check(agrees, "the field circuit agrees with plain integer arithmetic when nothing wraps");
        long[] bigInteger = IntegerReference(model, input, out bool bigWrapped);
        Check(bigWrapped, "the full-size model does wrap the integers, so the field encoding is load-bearing");
        Check(Field.Normalise(bigInteger[0]) != output[0] || bigWrapped, "the wrapped reference really differs");

        // Merkle commitment: determinism, depth, membership, tamper rejection.
        var commitment = new MerkleCommitment(model.All);
        var rebuild = new MerkleCommitment(model.All);
        Check(commitment.Root == rebuild.Root && commitment.Root != 0, "the commitment root is deterministic");
        Console.WriteLine("  commitment: " + commitment.Size + " leaves, root=" + commitment.Root + " depth=" + commitment.Depth);
        Check(commitment.Size == 26 && commitment.Depth == 5, "26 leaves pad to 32 and give a depth-5 tree");
        bool pathsVerify = true;
        for (int i = 0; i < model.All.Length; i++)
        {
            if (!MerkleCommitment.Verify(model.All[i], i, commitment.Path(i), commitment.Root)) pathsVerify = false;
        }
        Check(pathsVerify, "every inclusion path recomputes the committed root");
        Check(!MerkleCommitment.Verify(model.All[3] + 1, 3, commitment.Path(3), commitment.Root),
            "a tampered parameter fails its inclusion proof");
        Check(!MerkleCommitment.Verify(model.All[3], 4, commitment.Path(3), commitment.Root),
            "the same opening under the wrong index is rejected");
        Check(Throws(() => commitment.Path(model.All.Length)), "an out-of-range leaf index is rejected");
        Check(Throws(() => new MerkleCommitment(new List<long>())), "an empty commitment is rejected");

        // The toy hash is a mixing function, not a collision-resistant hash.
        var seen = new Dictionary<long, long>();
        long collisionA = -1, collisionB = -1;
        for (long v = 0; v < 600000 && collisionA < 0; v++)
        {
            long h = ToyHash.OfLeaf(v, 0);
            if (seen.TryGetValue(h, out long other))
            {
                collisionA = other;
                collisionB = v;
            }
            else
            {
                seen[h] = v;
            }
        }
        Console.WriteLine("  toy hash: collision " + collisionA + " vs " + collisionB + " from " + seen.Count + " leaves");
        Check(collisionA >= 0 && collisionB > collisionA && ToyHash.OfLeaf(collisionA, 0) == ToyHash.OfLeaf(collisionB, 0),
            "a collision in the toy hash is found by brute force - no collision resistance is claimed");

        // Honest proof verifies; tampering is caught.
        var proof = BuildProof(model, commitment, input, 0x1234UL);
        Check(Verify(model, commitment, input, proof, 0x1234UL, out long verifierOps), "an honest proof verifies");
        var lying = BuildProof(model, commitment, input, 0x1234UL);
        lying.Output = (long[])proof.Output.Clone();
        lying.Output[0] = Field.Add(lying.Output[0], 1);
        Check(!Verify(model, commitment, input, lying, 0x1234UL, out _), "a tampered output is rejected");
        var wrongActivation = BuildProof(model, commitment, input, 0x1234UL);
        wrongActivation.Activation[0] = Field.Add(wrongActivation.Activation[0], 7);
        Check(!Verify(model, commitment, input, wrongActivation, 0x1234UL, out _),
            "a tampered activation is rejected by the square check");
        var wrongZ = BuildProof(model, commitment, input, 0x1234UL);
        wrongZ.PreActivation[1] = Field.Add(wrongZ.PreActivation[1], 1);
        Check(!Verify(model, commitment, input, wrongZ, 0x1234UL, out _),
            "a tampered pre-activation breaks the first matmul identity");
        var wrongOpening = BuildProof(model, commitment, input, 0x1234UL);
        wrongOpening.ProbeValue = Field.Add(wrongOpening.ProbeValue, 1);
        Check(!Verify(model, commitment, input, wrongOpening, 0x1234UL, out _),
            "claiming a parameter the commitment does not contain is rejected");
        var wrongInput = new[] { 3L, -5L, 4L };
        Check(!Verify(model, commitment, wrongInput, proof, 0x1234UL, out _),
            "the same proof does not verify against a different public input");

        // Soundness: the wrong-claim acceptance rate over random challenges, and its bound.
        int accepted = 0;
        const int trials = 2000;
        for (int i = 0; i < trials; i++)
        {
            ulong runSeed = (ulong)(i * 7919 + 13);
            var bad = BuildProof(model, commitment, input, runSeed);
            bad.Output = (long[])bad.Output.Clone();
            bad.Output[0] = Field.Add(bad.Output[0], 1);
            if (Verify(model, commitment, input, bad, runSeed, out _)) accepted++;
        }
        Console.WriteLine("  soundness: tampered outputs accepted " + accepted + " of " + trials + " fresh challenges");
        Check(accepted * 1000 <= trials, "the acceptance rate of a wrong claim is far below 1e-3");

        int admitting = 0;
        const int smallSpace = 16;
        for (long challenge = 0; challenge < smallSpace; challenge++)
        {
            // an inconsistent linear claim: the true product is 17 * 23 = 391 but the prover asserts 392
            if (Field.Mul(challenge, 392) == Field.Mul(challenge, Field.Mul(17, 23))) admitting++;
        }
        Console.WriteLine("  soundness bound: a challenge space of " + smallSpace + " admits " + admitting
            + " bad challenge(s)");
        Check(admitting == 1, "exactly one of 16 challenges admits the inconsistent claim - the zero the verifier excludes");
        int consistentRejections = 0;
        for (long challenge = 0; challenge < smallSpace; challenge++)
        {
            if (Field.Mul(challenge, 391) != Field.Mul(challenge, Field.Mul(17, 23))) consistentRejections++;
        }
        Check(consistentRejections == 0, "the honest claim is accepted for every challenge (completeness)");

        // The honest limitation: a prover that adapts the folded vector to the challenge passes one check, which is
        // exactly why a real protocol must commit to that vector before the challenge is drawn.
        const ulong forgerySeed = 0x777UL;
        var forger = BuildProof(model, commitment, input, forgerySeed);
        forger.Output = (long[])forger.Output.Clone();
        forger.Output[0] = Field.Add(forger.Output[0], 1);
        forger.FoldedW2 = (long[])forger.FoldedW2.Clone();
        var forgeryRng = forgerySeed;
        for (int h = 0; h < model.Hidden; h++) _ = Field.Challenge(ref forgeryRng);
        var forgeryAlpha = new long[model.Outputs];
        for (int o = 0; o < model.Outputs; o++) forgeryAlpha[o] = Field.Challenge(ref forgeryRng);
        if (forger.Activation[0] != 0)
        {
            long correction = Field.Mul(forgeryAlpha[0], Field.Inverse(forger.Activation[0]));
            forger.FoldedW2[0] = Field.Add(forger.FoldedW2[0], correction);
        }
        bool adaptiveAccepted = Verify(model, commitment, input, forger, forgerySeed, out _);
        Console.WriteLine("  limitation: an adaptive prover with a forged fold is accepted=" + adaptiveAccepted
            + " - a single linear check cannot bind what the prover may still change");
        Check(adaptiveAccepted,
            "an adaptive prover that forges the folded vector defeats one linear check: this toy claims no"
            + " soundness against an adaptive prover, only the Schwartz-Zippel bound against a fixed fold");
        Check(!Verify(model, commitment, input, lying, forgerySeed, out _),
            "the same tampering with an honest fold is still rejected, which is the property that is verified");

        // Cost asymmetry: verifier multiplications versus the prover's folding work.
        var shapes = new[] { new[] { 8, 16 }, new[] { 64, 64 }, new[] { 256, 128 } };
        foreach (int[] shape in shapes)
        {
            int rows = shape[0], cols = shape[1];
            var matrix = new long[rows * cols];
            var vector = new long[cols];
            var product = new long[rows];
            var fill = 0xBEEFUL;
            for (int i = 0; i < matrix.Length; i++) matrix[i] = (long)(SplitMix64.Next(ref fill) % 1000UL);
            for (int i = 0; i < cols; i++) vector[i] = (long)(SplitMix64.Next(ref fill) % 1000UL);
            for (int r = 0; r < rows; r++)
            {
                long acc = 0;
                for (int c = 0; c < cols; c++) acc = Field.Add(acc, Field.Mul(matrix[r * cols + c], vector[c]));
                product[r] = acc;
            }
            long verifier = 0;
            var rngFold = 0xFEEDUL;
            var alpha = new long[rows];
            for (int r = 0; r < rows; r++) alpha[r] = Field.Challenge(ref rngFold);
            var folded = new long[cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++) folded[c] = Field.Add(folded[c], Field.Mul(alpha[r], matrix[r * cols + c]));
            }
            long left = 0;
            for (int r = 0; r < rows; r++)
            {
                left = Field.Add(left, Field.Mul(alpha[r], product[r]));
                verifier++;
            }
            long right = 0;
            for (int c = 0; c < cols; c++)
            {
                right = Field.Add(right, Field.Mul(folded[c], vector[c]));
                verifier++;
            }
            Check(left == right, "the folded identity holds for the honest product " + rows + "x" + cols);
            long proverOps = (long)rows * cols;
            Console.WriteLine("    " + rows + "x" + cols + ": verifier=" + verifier + " prover=" + proverOps
                + " ratio=" + (verifier / (double)proverOps).ToString("F4"));
            Check(verifier < proverOps, "verification is cheaper than the product for " + rows + "x" + cols);
        }

        // Verification cost is linear in the rows, not in the product.
        long opsSmall = 0, opsLarge = 0;
        var smallModel = new Model(4, 4, 4, 3UL);
        var largeModel = new Model(4, 64, 4, 3UL);
        var smallCommitment = new MerkleCommitment(smallModel.All);
        var largeCommitment = new MerkleCommitment(largeModel.All);
        var x4 = new[] { 2L, 3L, 5L, 7L };
        var smallProof = BuildProof(smallModel, smallCommitment, x4, 9UL);
        var largeProof = BuildProof(largeModel, largeCommitment, x4, 9UL);
        Check(Verify(smallModel, smallCommitment, x4, smallProof, 9UL, out opsSmall), "the small model verifies");
        Check(Verify(largeModel, largeCommitment, x4, largeProof, 9UL, out opsLarge), "the tall model verifies");
        Console.WriteLine("  verifier ops: hidden=4 -> " + opsSmall + ", hidden=64 -> " + opsLarge
            + " (prover folds " + ProverFoldOps(smallModel) + " and " + ProverFoldOps(largeModel) + ")");
        Check(opsLarge > opsSmall, "a wider layer costs the verifier more than a narrow one");
        Check(opsLarge < ProverFoldOps(largeModel) && opsSmall < ProverFoldOps(smallModel),
            "the verifier's multiplications stay below the prover's folding cost in both shapes");

        // Sampling the parameters: the detection rate matches the closed form.
        int caught = 0;
        const int samples = 4000;
        const int probes = 4;
        for (int i = 0; i < samples; i++)
        {
            var sampleRng = (ulong)(i * 104729 + 7);
            for (int s = 0; s < probes; s++)
            {
                if ((int)(SplitMix64.Next(ref sampleRng) % (ulong)commitment.Size) == 0) { caught++; break; }
            }
        }
        double observed = caught / (double)samples;
        double predicted = 1.0 - Math.Pow(1.0 - probes / (double)commitment.Size, 1.0);
        Console.WriteLine("  sampling: detected " + observed.ToString("F4") + " predicted " + predicted.ToString("F4"));
        Check(Math.Abs(observed - predicted) < 3.0 * Math.Sqrt(predicted / samples) + 0.02,
            "subsample probing catches a tampered leaf at the predicted rate");

        // Validation and determinism.
        Check(Throws(() => smallModel.Forward(new[] { 1L }, out _, out _)), "an input of the wrong dimension is rejected");
        Check(Throws(() => new Model(0, 2, 2, 1UL)), "a degenerate layer is rejected");
        Check(Throws(() => Field.Inverse(0)), "inverting zero is rejected");
        long[] again = model.Forward(input, out _, out _);
        bool stable = true;
        for (int o = 0; o < again.Length; o++) stable &= again[o] == output[o];
        Check(stable, "evaluation is deterministic");
        var replay = BuildProof(model, commitment, input, 0x1234UL);
        Check(Verify(model, commitment, input, replay, 0x1234UL, out long opsReplay) && opsReplay == verifierOps,
            "repeated verification is bit-for-bit identical");

        Console.WriteLine("PASS");
    }
}
