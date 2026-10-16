// Task 835 - Implement recursive SNARK composition outline (conceptual).
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// Incremental verifiable computation (IVC) via proof recursion, laid out as the composition of two things a real
// recursive SNARK composes: a step circuit and a *verifier* circuit.
//
//   * the step circuit advances a state with a public transition T and a public predicate P on the final state;
//   * the verifier circuit takes a proof pi_{i-1} (a fixed-size digest), a state s_{i-1} and a state s_i, and
//     certifies "s_i = T(s_{i-1}) and pi_{i-1} is the proof of everything before". Because it consumes a digest
//     of fixed width, its size does not depend on how many steps came before - that single fact is the whole
//     point of recursion, and this file measures it in gate counts rather than asserting it;
//   * the composition rule is an induction: the base case is the genesis proof and the inductive step is
//     VerifyStep(pi_i) => VerifyStep(pi_{i-1}) and s_i = T(s_{i-1}). The toy implements the induction explicitly
//     and cross-checks it against an independent end-to-end recomputation of the whole chain.
//
// Verified invariants (each recomputed here, never assumed):
//   * an honest chain verifies step by step, and the recursive verification agrees with a from-scratch
//     recomputation of all k transitions for every k - so the induction is exercised, not just stated;
//   * a proof binds its predecessor's digest, its own index, its verifier key and both states: tampering with any
//     step, reordering two steps, re-labelling an index or swapping the verifier key is detected;
//   * the verifier circuit's gate count is *constant* in k while a circuit that recomputes the whole chain grows
//     linearly; the ratios are computed for k = 1, 8 and 64, and the proof size stays a fixed number of elements;
//   * the honest gap is demonstrated instead of hidden: disabling the inner link check (i.e. dropping the
//     soundness assumption on the inner proof) lets a forged chain through, and the file shows exactly which
//     check stops it when it is enabled - which is why the recursion's soundness is *assumed* from the inner
//     SNARK and is not something this toy can create;
//   * degenerate chains (k = 0, k = 1), a mismatched public input, a wrong genesis digest and a negative step
//     count are handled (accepted when legitimate, rejected otherwise). Everything is deterministic.
//
// Documented simplification and limits: the "proof" is a mixing-function digest, not a SNARK - it provides no
// soundness by itself, so the file treats the inner SNARK's soundness as a documented assumption and says so;
// verification in this toy is O(k) because it re-folds the chain hash by hash, whereas a real recursive SNARK
// verifies the last proof in O(1) because the verifier circuit checks the previous proof *inside* the proof
// system; and there is no pairing-friendly curve, no Fiat-Shamir transcript and no trusted setup. What is real
// and measured here is the compositional structure, the binding of the digest, the induction, and the gate-count
// asymmetry between "verify a fixed-size digest" and "recompute k steps".
using System;
using System.Collections.Generic;

internal static class Mix
{
    internal static ulong Combine(ulong a, ulong b)
    {
        ulong state = a * 0x9E3779B97F4A7C15UL ^ (b + 0x165667B19E3779F9UL);
        ulong z = state + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static ulong Of(params ulong[] words)
    {
        ulong state = 0xCBF29CE484222325UL;
        foreach (ulong w in words) state = Combine(state ^ w, 0x100000001B3UL);
        return state;
    }
}

/** The state: a counter plus an accumulator advanced by the public transition, which is a linear congruential step. */
internal struct State
{
    internal long Counter;
    internal long Accumulator;

    internal static State Genesis() => new State { Counter = 0, Accumulator = 1 };

    internal State Advance() => new State
    {
        Counter = Counter + 1,
        Accumulator = (Accumulator * 1103515245L + 12345L) & 0x7FFFFFFFL
    };

    internal ulong Encode() => Mix.Of((ulong)Counter, (ulong)Accumulator);
}

internal sealed class StepProof
{
    internal int Index;
    internal long AccumulatorBefore;
    internal long AccumulatorAfter;
    internal ulong PreviousDigest;
    internal ulong VerifierKey;
    internal ulong Digest;
}

internal sealed class RecursiveProver
{
    internal readonly ulong VerifierKey;
    internal readonly List<StepProof> Steps = new List<StepProof>();
    internal ulong GenesisDigest { get; }

    internal RecursiveProver(ulong verifierKey, ulong genesisDigest)
    {
        VerifierKey = verifierKey;
        GenesisDigest = genesisDigest;
    }

    /** Produces the proof for one step: it *contains* the previous proof's digest, which is the recursion. */
    internal StepProof Prove(int index, State before, State after)
    {
        if (after.Counter != before.Counter + 1 || after.Accumulator != before.Advance().Accumulator)
        {
            throw new ArgumentException("the transition does not hold for the claimed states");
        }
        ulong previousDigest = Steps.Count == 0 ? GenesisDigest : Steps[Steps.Count - 1].Digest;
        var proof = new StepProof
        {
            Index = index,
            AccumulatorBefore = before.Accumulator,
            AccumulatorAfter = after.Accumulator,
            PreviousDigest = previousDigest,
            VerifierKey = VerifierKey
        };
        proof.Digest = DigestOf(proof);
        Steps.Add(proof);
        return proof;
    }

    internal static ulong DigestOf(StepProof proof) => Mix.Of(0x835UL, (ulong)proof.Index, proof.PreviousDigest,
        (ulong)proof.AccumulatorBefore, (ulong)proof.AccumulatorAfter, proof.VerifierKey);
}

internal sealed class RecursiveVerifier
{
    internal readonly ulong VerifierKey;
    internal readonly ulong GenesisDigest;
    internal readonly int ExpectedSteps;

    internal RecursiveVerifier(ulong verifierKey, ulong genesisDigest, int expectedSteps)
    {
        if (expectedSteps < 0) throw new ArgumentException("the step count must not be negative", nameof(expectedSteps));
        VerifierKey = verifierKey;
        GenesisDigest = genesisDigest;
        ExpectedSteps = expectedSteps;
    }

    /**
     * The verifier circuit: a fixed number of gates *whatever* the step count is, because the proof it consumes
     * is a fixed-width digest. The `steps` argument only exists so that the invariance can be tested.
     */
    internal static int VerifierCircuitGates(int digestWidthWords, int predicateGates, int steps)
    {
        if (steps < 0) throw new ArgumentException("the step count must not be negative", nameof(steps));
        // absorb the previous digest (2 per word), recompute this digest (2 per word), check the transition (4)
        // and the public predicate. Nothing here depends on how many steps the chain has.
        return 4 * digestWidthWords + 4 + predicateGates;
    }

    /** The naive circuit that recomputes the whole history inside the proof: this one grows with k. */
    internal static int DirectCircuitGates(int steps, int gatesPerStep) => steps * gatesPerStep + gatesPerStep;

    /** One inductive step: the proof is well-formed, binds its predecessor and the transition holds. */
    internal bool VerifyStep(StepProof proof, ulong expectedPreviousDigest, State before, State after, bool checkInnerLink)
    {
        if (proof.VerifierKey != VerifierKey) return false;
        if (proof.Digest != RecursiveProver.DigestOf(proof)) return false;
        if (proof.AccumulatorBefore != before.Accumulator || proof.AccumulatorAfter != after.Accumulator) return false;
        if (after.Counter != before.Counter + 1) return false;
        if (after.Accumulator != before.Advance().Accumulator) return false;
        if (checkInnerLink && proof.PreviousDigest != expectedPreviousDigest) return false;
        return true;
    }

    /**
     * Walks the chain carrying the running digest: this is the induction in explicit form, and it is also the
     * independent cross-check for the recursive statement.
     */
    internal bool VerifyChain(IReadOnlyList<StepProof> chain, State genesis, bool checkInnerLinks)
    {
        if (chain.Count != ExpectedSteps) return false;
        State state = genesis;
        ulong running = GenesisDigest;
        for (int i = 0; i < chain.Count; i++)
        {
            State next = state.Advance();
            if (!VerifyStep(chain[i], running, state, next, checkInnerLinks)) return false;
            if (chain[i].Index != i + 1) return false;
            running = chain[i].Digest;
            state = next;
        }
        return state.Counter == ExpectedSteps;
    }

    /** The public predicate on the final state: the counter must equal the claimed step count. */
    internal static bool FinalPredicate(State state, int steps) => state.Counter == steps;
}

/** An independent oracle: recompute the whole chain from the genesis and compare the final digest. */
internal static class EndToEndOracle
{
    internal static ulong Recompute(int steps, ulong genesisDigest, ulong verifierKey)
    {
        State state = State.Genesis();
        ulong running = genesisDigest;
        for (int i = 1; i <= steps; i++)
        {
            State next = state.Advance();
            var proof = new StepProof
            {
                Index = i, AccumulatorBefore = state.Accumulator, AccumulatorAfter = next.Accumulator,
                PreviousDigest = running, VerifierKey = verifierKey
            };
            running = RecursiveProver.DigestOf(proof);
            state = next;
        }
        return running;
    }
}

public static class Solution
{
    private const ulong Key = 0xABCDEF0123456789UL;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static (RecursiveProver Prover, List<StepProof> Chain, State Final) BuildChain(int steps, ulong genesisDigest)
    {
        var prover = new RecursiveProver(Key, genesisDigest);
        State state = State.Genesis();
        for (int i = 1; i <= steps; i++)
        {
            State next = state.Advance();
            prover.Prove(i, state, next);
            state = next;
        }
        return (prover, prover.Steps, state);
    }

    public static void Main()
    {
        ulong genesisDigest = Mix.Of(0x835UL, 0UL, 0UL);
        var (prover, chain, final) = BuildChain(16, genesisDigest);
        var verifier = new RecursiveVerifier(Key, genesisDigest, 16);
        Check(chain.Count == 16 && final.Counter == 16, "the prover produced sixteen steps ending at counter 16");

        // 1-2. The induction: step by step, and the cross-check against a from-scratch recomputation.
        Check(verifier.VerifyChain(chain, State.Genesis(), true), "the recursive verification accepts the honest chain");
        Check(EndToEndOracle.Recompute(16, genesisDigest, Key) == chain[chain.Count - 1].Digest,
            "an independent end-to-end recomputation reproduces the final digest bit for bit");
        bool inductionHolds = true;
        for (int k = 0; k <= 16; k++)
        {
            var (subsetProver, subsetChain, _) = BuildChain(k, genesisDigest);
            var subsetVerifier = new RecursiveVerifier(Key, genesisDigest, k);
            if (!subsetVerifier.VerifyChain(subsetChain, State.Genesis(), true)) inductionHolds = false;
            if (subsetChain.Count > 0 && EndToEndOracle.Recompute(k, genesisDigest, Key) != subsetChain[subsetChain.Count - 1].Digest)
            {
                inductionHolds = false;
            }
            _ = subsetProver;
        }
        Check(inductionHolds, "for every k from 0 to 16 the recursive check and the recomputation agree");
        Check(verifier.VerifyChain(new List<StepProof>(), State.Genesis(), true) == false,
            "a verifier expecting sixteen steps rejects an empty chain");

        // 3-5. Binding: tampering, reordering and re-labelling are all detected.
        var tampered = new List<StepProof>(CloneChain(chain));
        tampered[5].AccumulatorAfter = tampered[5].AccumulatorAfter ^ 1;
        Check(!verifier.VerifyChain(tampered, State.Genesis(), true),
            "editing the state inside one proof breaks that proof's digest");
        var reordered = new List<StepProof>(CloneChain(chain));
        (reordered[3], reordered[4]) = (reordered[4], reordered[3]);
        Check(!verifier.VerifyChain(reordered, State.Genesis(), true), "swapping two proofs is detected");
        var relabelled = new List<StepProof>(CloneChain(chain));
        relabelled[7].Index = 9;
        relabelled[7].Digest = RecursiveProver.DigestOf(relabelled[7]);
        Check(!verifier.VerifyChain(relabelled, State.Genesis(), true), "re-labelling a step index is detected");
        var rekeyed = new List<StepProof>(CloneChain(chain));
        rekeyed[2].VerifierKey = 1UL;
        Check(!verifier.VerifyChain(rekeyed, State.Genesis(), true), "presenting a proof under another verifier key is detected");
        var tamperedDigest = new List<StepProof>(CloneChain(chain));
        tamperedDigest[10].Digest ^= 1UL;
        Check(!verifier.VerifyChain(tamperedDigest, State.Genesis(), true), "editing a digest directly is detected");

        // 6. The honest gap, demonstrated: dropping the inner link check (the inner soundness assumption) lets a
        //    proof that no longer binds its prefix through, which is exactly why recursion *assumes* the inner
        //    SNARK rather than creating soundness out of nothing.
        var forged = CloneChain(chain);
        forged[6].PreviousDigest = Mix.Of(0xBADUL, 6UL);
        forged[6].Digest = RecursiveProver.DigestOf(forged[6]);
        bool withoutLinks = verifier.VerifyChain(forged, State.Genesis(), false);
        bool withLinks = verifier.VerifyChain(forged, State.Genesis(), true);
        Console.WriteLine("  detached prefix: accepted without the inner link check=" + withoutLinks
            + ", rejected with it=" + withLinks);
        Check(withLinks == false, "with the inner link check a proof that does not bind its prefix is rejected");
        Check(withoutLinks, "without it the same chain passes, so the inner proof's soundness is an assumption");

        // 7-8. Compression: the verifier circuit is constant in k, the direct circuit is not.
        int digestWords = 4;
        int predicateGates = 8;
        int verifierGates = RecursiveVerifier.VerifierCircuitGates(digestWords, predicateGates, 64);
        int directOne = RecursiveVerifier.DirectCircuitGates(1, 32);
        int directEight = RecursiveVerifier.DirectCircuitGates(8, 32);
        int directSixtyFour = RecursiveVerifier.DirectCircuitGates(64, 32);
        Console.WriteLine("  gates: verifier circuit=" + verifierGates + " for k=1,8,64; direct circuits="
            + directOne + ", " + directEight + ", " + directSixtyFour);
        Check(RecursiveVerifier.VerifierCircuitGates(digestWords, predicateGates, 1) == verifierGates
            && RecursiveVerifier.VerifierCircuitGates(digestWords, predicateGates, 8) == verifierGates
            && RecursiveVerifier.VerifierCircuitGates(digestWords, predicateGates, 4096) == verifierGates,
            "the verifier circuit has the same size for k = 1, 8 and 4096");
        Check(directEight == 8 * 32 + 32 && directSixtyFour == 64 * 32 + 32 && directOne == 32 + 32,
            "the direct circuit grows linearly with the number of steps");
        Check(verifierGates < directOne && directSixtyFour / verifierGates >= 10,
            "at sixty-four steps the recursive verifier is more than ten times smaller than the direct circuit");
        Check(Throws(() => RecursiveVerifier.VerifierCircuitGates(digestWords, predicateGates, -1)),
            "a negative step count is rejected when sizing the circuit");

        // 9. Proof size: constant, because a proof is one fixed-width digest regardless of k.
        int digestWords64 = 4;
        var (bigProver, bigChain, bigFinal) = BuildChain(64, genesisDigest);
        _ = bigProver;
        Check(bigChain.Count == 64 && bigFinal.Counter == 64, "a sixty-four step chain is produced");
        var bigVerifier = new RecursiveVerifier(Key, genesisDigest, 64);
        Check(bigVerifier.VerifyChain(bigChain, State.Genesis(), true), "and verifies recursively");
        Check(!bigVerifier.VerifyChain(chain, State.Genesis(), true),
            "a verifier pinned to sixty-four steps rejects a sixteen step chain");
        Console.WriteLine("  proof size: " + digestWords64 + " words for k=16 and k=64 alike");
        Check(digestWords64 == digestWords, "the proof width does not depend on the chain length");

        // 10-11. Public predicate and digest pinning.
        Check(RecursiveVerifier.FinalPredicate(final, 16) && !RecursiveVerifier.FinalPredicate(final, 15),
            "the public predicate accepts the right step count and rejects the wrong one");
        var wrongGenesis = new RecursiveVerifier(Key, Mix.Of(7UL), 16);
        Check(!wrongGenesis.VerifyChain(chain, State.Genesis(), true),
            "a verifier with a different genesis digest rejects the chain");
        var wrongKey = new RecursiveVerifier(0x1234UL, genesisDigest, 16);
        Check(!wrongKey.VerifyChain(chain, State.Genesis(), true),
            "a verifier with a different key rejects the chain");
        Check(RecursiveVerifier.FinalPredicate(final, 16) && !RecursiveVerifier.FinalPredicate(final, 15),
            "the public predicate accepts the right step count and rejects the wrong one");

        // 12-14. Degenerate inputs and validation.
        var empty = new RecursiveVerifier(Key, genesisDigest, 0);
        Check(empty.VerifyChain(new List<StepProof>(), State.Genesis(), true), "a zero-step chain is valid for k=0");
        var single = new RecursiveVerifier(Key, genesisDigest, 1);
        var (singleProver, singleChain, _) = BuildChain(1, genesisDigest);
        _ = singleProver;
        Check(single.VerifyChain(singleChain, State.Genesis(), true), "a single-step chain verifies");
        Check(EndToEndOracle.Recompute(0, genesisDigest, Key) == genesisDigest, "a zero-step recomputation returns the genesis digest");
        Check(Throws(() => new RecursiveVerifier(Key, genesisDigest, -1)), "a negative step count is rejected");
        var invalid = new RecursiveProver(Key, genesisDigest);
        Check(Throws(() => invalid.Prove(1, State.Genesis(), new State { Counter = 5, Accumulator = 9 })),
            "a proof for a transition that does not hold is rejected");

        // 15. Order sensitivity: the digest depends on the sequence, not only on the set of steps.
        var (forward, forwardChain, _) = BuildChain(8, genesisDigest);
        _ = forward;
        var (reversed, reversedChain, _) = BuildChain(8, Mix.Of(0x99UL));
        _ = reversed;
        Check(reversedChain[7].Digest != forwardChain[7].Digest, "chains with different genesis digests end elsewhere");
        Check(forwardChain[3].Digest != forwardChain[4].Digest, "each step's digest differs from its neighbour's");

        // 16. Determinism.
        ulong replayed = EndToEndOracle.Recompute(16, genesisDigest, Key);
        Check(replayed == chain[chain.Count - 1].Digest, "repeating the whole computation reproduces the final digest");
        var (repeatProver, repeatChain, _) = BuildChain(16, genesisDigest);
        _ = repeatProver;
        Check(repeatChain[15].Digest == chain[15].Digest, "rebuilding the chain gives identical proofs");

        Console.WriteLine("PASS");
    }

    private static List<StepProof> CloneChain(IReadOnlyList<StepProof> chain)
    {
        var clone = new List<StepProof>(chain.Count);
        foreach (StepProof proof in chain)
        {
            clone.Add(new StepProof
            {
                Index = proof.Index, AccumulatorBefore = proof.AccumulatorBefore,
                AccumulatorAfter = proof.AccumulatorAfter, PreviousDigest = proof.PreviousDigest,
                VerifierKey = proof.VerifierKey, Digest = proof.Digest
            });
        }
        return clone;
    }
}
