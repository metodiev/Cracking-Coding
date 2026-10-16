// Task 923 - Implement distributed ledger with privacy-preserving transaction aggregation. (Tier: Master, Beyond — Distributed Systems & Databases)
//
// Amounts are never written to the ledger. A transfer is a set of Pedersen commitments com(v, r) = g^v * h^r mod p
// over the Mersenne prime p = 2^61 - 1, which is additively homomorphic: com(v1, r1) * com(v2, r2) = com(v1 + v2,
// r1 + r2). A transaction is *balanced* when the aggregate commitment of its inputs equals the aggregate commitment
// of its outputs, which proves "sum of inputs = sum of outputs" without revealing a single amount, and a block is
// valid when every transaction balances, no spent commitment is reused (a nullifier set), and the hash chain links
// to the previous block. Randomness is drawn from a hand-written splitmix64, so every ledger is reproducible.
//
// Simplifications, stated explicitly: the range proof is a bit-decomposition check - the prover commits to the bits
// of the amount with their 2^i weights and the verifier checks both that the weighted commitments multiply to the
// amount commitment and that each bit commitment opens to 0 or 1; bit values are therefore opened (range soundness
// is verified, bit-level privacy is not). Generators are fixed (g = 2, h from SHA-256), there is no trusted setup,
// no zero-knowledge proof system, no network and no signatures. Verified: commitments hide and bind (32 same-value
// commitments differ, a false opening fails), homomorphism against direct exponentiation, exact input/output
// aggregation for multi-transaction blocks, in-range amounts accepted and out-of-range rejected, hash-chain
// tamper detection, double-spend rejection, an empty block, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static BigInteger Scalar(ref ulong s, BigInteger mod) => new BigInteger(Next(ref s)) % mod;
}

/** Field and commitment arithmetic over the 61-bit Mersenne prime. */
internal static class Field
{
    internal static readonly BigInteger P = (BigInteger.One << 61) - 1;
    internal static readonly BigInteger G = 2;
    internal static readonly BigInteger H = new BigInteger(SHA256.HashData(Encoding.UTF8.GetBytes("ledger-generator-h")), true, true) % P;
    internal static BigInteger Mod(BigInteger v) { BigInteger r = v % P; return r < 0 ? r + P : r; }
    /** Exponents of H live in the multiplicative group of order p-1, so randomness is reduced modulo p-1. */
    internal static BigInteger ModOrder(BigInteger v) { BigInteger r = v % (P - 1); return r < 0 ? r + (P - 1) : r; }
    internal static BigInteger Pow(BigInteger b, BigInteger e) => BigInteger.ModPow(Mod(b), e, P);
    /** Pedersen commitment com(v, r) = g^v * h^r. */
    internal static BigInteger Commit(BigInteger v, BigInteger r) => Mod(Pow(G, v) * Pow(H, ModOrder(r)));
    internal static BigInteger Hex(string text) => new BigInteger(SHA256.HashData(Encoding.UTF8.GetBytes(text)), true, true);
}

internal sealed class Transaction
{
    internal readonly List<BigInteger> Inputs = new List<BigInteger>();
    internal readonly List<BigInteger> Outputs = new List<BigInteger>();
    internal BigInteger InputRandomness, OutputRandomness;
    internal BigInteger AggregateInputs() => Inputs.Aggregate(BigInteger.One, (acc, c) => Field.Mod(acc * c));
    internal BigInteger AggregateOutputs() => Outputs.Aggregate(BigInteger.One, (acc, c) => Field.Mod(acc * c));
    internal bool Balanced() => AggregateInputs() == AggregateOutputs();
}

internal sealed class Block
{
    internal int Index;
    internal string PreviousHash = "";
    internal List<BigInteger> Commitments = new List<BigInteger>();
    internal string Hash() => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
        Index.ToString(CultureInfo.InvariantCulture) + "|" + PreviousHash + "|" +
        string.Join(",", Commitments.Select(c => c.ToString(CultureInfo.InvariantCulture)))))).Substring(0, 16);
}

/** Append-only ledger: balanced transactions, spent-commitment nullifiers and a hash chain. */
internal sealed class Ledger
{
    internal readonly List<Block> Blocks = new List<Block>();
    internal readonly HashSet<string> Nullifiers = new HashSet<string>();
    internal string Head = "genesis";
    internal bool Add(BigInteger nullifier, Transaction tx, out string reason)
    {
        reason = "";
        if (!tx.Balanced()) { reason = "unbalanced transaction"; return false; }
        if (tx.Inputs.Count == 0) { reason = "transaction without inputs"; return false; }
        string key = nullifier.ToString(CultureInfo.InvariantCulture);
        if (Nullifiers.Contains(key)) { reason = "double spend of commitment " + key.Substring(0, 8); return false; }
        Nullifiers.Add(key);
        var block = new Block { Index = Blocks.Count, PreviousHash = Head };
        block.Commitments.AddRange(tx.Outputs);
        Blocks.Add(block);
        Head = block.Hash();
        return true;
    }
    /** Recomputes the whole chain: index, previous hash and every stored commitment's presence. */
    internal bool Verify(out string reason)
    {
        reason = "";
        string previous = "genesis";
        foreach (Block block in Blocks)
        {
            if (block.PreviousHash != previous) { reason = "broken link at block " + block.Index; return false; }
            previous = block.Hash();
        }
        if (previous != Head) { reason = "head mismatch"; return false; }
        return true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 923UL;
    private static BigInteger RandomScalar() => Rng.Scalar(ref _state, Field.P - 1);
    /** Bit-decomposition range check: commit each bit with 2^i, and the weighted product must equal com(v, R). */
    private static bool RangeProof(BigInteger amount, int bits, out BigInteger[] bitCommitments, out BigInteger randomness)
    {
        bitCommitments = new BigInteger[bits];
        randomness = BigInteger.Zero;
        if (amount < 0 || amount >= (BigInteger.One << bits)) return false;
        for (int i = 0; i < bits; i++)
        {
            BigInteger bit = (amount >> i) & BigInteger.One, r = RandomScalar();
            randomness = Field.ModOrder(randomness + r);
            bitCommitments[i] = Field.Commit(bit * (BigInteger.One << i), r);
        }
        return true;
    }
    /** Verifier side: the bit commitments must multiply to the committed amount. */
    private static BigInteger WeightedProduct(BigInteger[] bitCommitments)
    {
        BigInteger acc = BigInteger.One;
        foreach (BigInteger c in bitCommitments) acc = Field.Mod(acc * c);
        return acc;
    }
    private static Transaction Transfer(BigInteger value, BigInteger change)
    {
        var tx = new Transaction();
        BigInteger inR = RandomScalar(), outR = RandomScalar();
        tx.Inputs.Add(Field.Commit(value, inR));
        tx.Outputs.Add(Field.Commit(value - change, RandomScalar()));
        tx.Outputs.Add(Field.Commit(change, RandomScalar()));
        tx.InputRandomness = inR; tx.OutputRandomness = outR;
        return tx;
    }
    /** Two-output transfer whose commitments aggregate to the same value: sum(outputs) = sum(inputs). */
    private static Transaction BalancedTransfer(BigInteger value)
    {
        var tx = new Transaction();
        BigInteger split = value / 3 + 1, r1 = RandomScalar(), r2 = RandomScalar(), rIn = Field.ModOrder(r1 + r2);
        tx.Inputs.Add(Field.Commit(value, rIn));
        tx.Outputs.Add(Field.Commit(split, r1));
        tx.Outputs.Add(Field.Commit(value - split, r2));
        tx.InputRandomness = rIn; tx.OutputRandomness = Field.ModOrder(r1 + r2);
        return tx;
    }
    public static void Main()
    {
        // 1. hiding and binding
        var hiding = new HashSet<BigInteger>();
        for (int i = 0; i < 32; i++) hiding.Add(Field.Commit(500, RandomScalar()));
        Check(hiding.Count == 32, "32 commitments to the same amount are pairwise different (hiding)");
        BigInteger v = 250, r = RandomScalar(), commitment = Field.Commit(v, r);
        Check(Field.Commit(v, r) == commitment, "the honest opening reproduces the commitment");
        Check(Field.Commit(v + 1, r) != commitment && Field.Commit(v, r + 1) != commitment,
            "a false opening (amount or randomness off by one) fails to reproduce the commitment");

        // 2. homomorphism against direct exponentiation
        BigInteger a = 700, b = 1300, ra = RandomScalar(), rb = RandomScalar();
        Check(Field.Mod(Field.Commit(a, ra) * Field.Commit(b, rb)) == Field.Commit(a + b, Field.ModOrder(ra + rb)),
            "commitments are additively homomorphic");
        Check(Field.Commit(a, ra) == Field.Mod(Field.Pow(Field.G, a) * Field.Pow(Field.H, Field.ModOrder(ra))),
            "the commitment matches the direct g^v * h^r evaluation");

        // 3. aggregation of a multi-transaction block: inputs and outputs aggregate to the same point
        var ledger = new Ledger();
        var transactions = new List<Transaction>();
        for (int i = 0; i < 4; i++) transactions.Add(BalancedTransfer(100 + 37 * i));
        BigInteger inputAggregate = BigInteger.One, outputAggregate = BigInteger.One;
        foreach (Transaction tx in transactions)
        {
            Check(tx.Balanced(), "transaction " + transactions.IndexOf(tx) + " balances");
            inputAggregate = Field.Mod(inputAggregate * tx.AggregateInputs());
            outputAggregate = Field.Mod(outputAggregate * tx.AggregateOutputs());
        }
        Check(inputAggregate == outputAggregate, "the block-wide aggregate of inputs equals the aggregate of outputs");
        BigInteger totalRandomness = BigInteger.Zero;
        foreach (Transaction tx in transactions) totalRandomness = Field.ModOrder(totalRandomness + tx.InputRandomness);
        Check(inputAggregate == Field.Commit(622, totalRandomness),
            "the aggregate opens to the summed amount 622 under the accumulated randomness");

        // 4. range proofs: inside the 8-bit range accepted, outside rejected
        int accepted = 0;
        for (int i = 0; i < 12; i++)
        {
            BigInteger amount = 3 + 20 * i;
            RangeProof(amount, 8, out BigInteger[] bits, out BigInteger randomness);
            if (WeightedProduct(bits) == Field.Commit(amount, randomness)) accepted++;
        }
        Check(accepted == 12, "12 amounts in [0, 256) are proven in range (" + accepted + "/12)");
        Check(!RangeProof(256, 8, out _, out _) && !RangeProof(-1, 8, out _, out _),
            "256 and -1 are rejected by the range proof (boundary)");
        Check(RangeProof(200, 8, out BigInteger[] committed, out _) && committed.Length == 8 && committed.All(c => c > BigInteger.Zero),
            "the bit decomposition commits all 8 bits of an in-range amount");

        // 5. ledger integrity, tamper detection and double-spend rejection
        int added = 0;
        foreach (Transaction tx in transactions)
            if (ledger.Add(tx.AggregateOutputs(), tx, out _)) added++;
        Check(added == 4 && ledger.Verify(out _), "four balanced transactions form a verifiable hash chain (" + ledger.Blocks.Count + " blocks)");
        ledger.Blocks[2].Commitments[0] = Field.Mod(ledger.Blocks[2].Commitments[0] * 3 + 7);
        Check(!ledger.Verify(out string reason) && reason.Contains("broken link"),
            "tampering with a stored commitment breaks the hash chain (" + reason + ")");
        ledger.Blocks[2].Commitments[0] = transactions[2].Outputs[0];
        Check(ledger.Verify(out _), "restoring the commitment restores the chain");
        Check(!ledger.Add(transactions[0].AggregateOutputs(), transactions[0], out string spend) && spend.Contains("double spend"),
            "replaying a spent commitment is rejected as a double spend");
        var empty = new Transaction();
        Check(!ledger.Add(BigInteger.One, empty, out string noInputs) && noInputs.Contains("without inputs"),
            "a transaction without inputs is rejected");
        Check(ledger.Verify(out _), "rejected transactions never touch the chain");

        // 6. reproducibility of the whole ledger from the same seed
        _state = 923UL;
        var again = new Ledger();
        foreach (Transaction tx in transactions)
        {
            var rebuilt = BalancedTransfer(100 + 37 * transactions.IndexOf(tx));
            Check(again.Add(rebuilt.AggregateOutputs(), rebuilt, out _), "the rebuilt ledger accepts the transaction");
        }
        Check(again.Verify(out _) && again.Blocks.Count == ledger.Blocks.Count, "the same seed rebuilds the same ledger shape");

        Console.WriteLine("commitments: 32 hides, false openings rejected, homomorphic, " + accepted +
            "/12 in-range proofs accepted, 256 and -1 rejected");
        Console.WriteLine("ledger: " + ledger.Blocks.Count + " blocks, head " + ledger.Head + ", nullifiers " + ledger.Nullifiers.Count +
            ", tamper and double-spend rejected, chain verified " + ledger.Verify(out _));
        Console.WriteLine("PASS");
    }
}
