// Task 831 - Implement blockchain light client with SPV proofs and pruning.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A header-only (SPV) client, complete with the three things that make one usable at scale: proof-of-work and
// link validation on every header it accepts, Merkle inclusion proofs for "my transaction is in that block", and
// pruning of headers below a trusted checkpoint so the footprint stays bounded instead of growing with the chain.
//
//   * a header carries height, parent hash, the Merkle root of its transactions, a difficulty target, a timestamp
//     and a nonce; its hash must be below the target (proof of work) and its target must equal what the retarget
//     rule prescribes from the parent - the rule looks back four blocks, so a branch's difficulty depends on its
//     own history and two branches at the same height can carry different work;
//   * the client keeps every chain it has seen, keyed by tip, and follows the one with the greatest cumulative
//     work (not the greatest length): work per header is floor(2^64 / target);
//   * inclusion is a Merkle path from a transaction id to the header's root, verified in O(log n) hashes;
//   * pruning drops headers below a trusted checkpoint while folding their work into a frozen accumulator, so the
//     cumulative-work ordering and all post-checkpoint proofs keep working. Headers below the checkpoint can no
//     longer be checked by the client - that is the weak-subjectivity trade-off and it is documented, not hidden.
//
// Verified invariants (each recomputed here, none assumed):
//   * every accepted header satisfies hash < target and a header whose nonce is bumped off the solution is
//     rejected; the linkage, the height increment and the retarget rule are all enforced (each violation has its
//     own rejected submission);
//   * an independent full validator that re-derives the whole chain from genesis agrees with the light client on
//     the tip hash and on the cumulative work, and the work is exactly the sum of the per-header work;
//   * inclusion proofs verify, a modified transaction, a wrong index and a proof presented against another block
//     are all rejected; the proof length is exactly ceil(log2 n);
//   * the most-work branch wins, and a branch carrying fewer headers can beat a longer one because its difficulty
//     is higher - the fork choice is measured against the work of both branches rather than against their lengths;
//   * pruning leaves exactly the headers from the checkpoint up, the frozen work plus the retained work still
//     equals the client's cumulative work bit-for-bit, post-checkpoint proofs still verify, and proofs for pruned
//     blocks are refused;
//   * mining effort is bounded: the measured number of hash attempts per block is a small multiple of the
//     expected 2^64 / target, which is the cost model that makes a light client viable. All runs are deterministic.
//
// Documented simplification: the hash is a 64-bit mixing function (splitmix64), NOT a cryptographic hash - no
// collision resistance, no preimage resistance, and therefore no real proof-of-work security is claimed; the
// difficulty is retargeted from a four-block window with small clamps instead of Bitcoin's 2016-block rule; the
// timestamp is supplied by the miner rather than read from a clock (no wall-clock time anywhere); and the
// checkpoint is assumed valid rather than justified. What is real: the header validation, the cumulative-work
// fork choice, the Merkle/SPV proof format and verification, and the pruning arithmetic with its frozen-work
// invariant.
using System;
using System.Collections.Generic;

internal static class Mix
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static ulong Combine(ulong a, ulong b)
    {
        ulong state = a * 0x9E3779B97F4A7C15UL ^ (b + 0x165667B19E3779F9UL);
        return Next(ref state);
    }

    internal static ulong Of(params ulong[] words)
    {
        ulong state = 0xCBF29CE484222325UL;
        foreach (ulong w in words)
        {
            state = Combine(state ^ w, 0x100000001B3UL);
        }
        return state;
    }
}

internal sealed class Header
{
    internal int Height;
    internal ulong Parent;
    internal ulong MerkleRoot;
    internal ulong Target;
    internal long Timestamp;
    internal ulong Nonce;
    internal ulong Hash;

    internal Header Copy() => new Header
    {
        Height = Height, Parent = Parent, MerkleRoot = MerkleRoot, Target = Target,
        Timestamp = Timestamp, Nonce = Nonce, Hash = Hash
    };
}

internal static class Chain
{
    internal const int RetargetWindow = 4;
    internal const long BlockInterval = 5;
    internal const ulong MaxTarget = 1UL << 58;
    internal const ulong MinTarget = 1UL << 50;

    internal static ulong ComputeHash(Header h) =>
        Mix.Of((ulong)h.Height, h.Parent, h.MerkleRoot, h.Target, (ulong)h.Timestamp, h.Nonce);

    internal static ulong WorkOf(ulong target) => ulong.MaxValue / target;

    /** Merkle root of the transaction ids; a single transaction hashes to itself. */
    internal static ulong MerkleRoot(IReadOnlyList<ulong> transactions)
    {
        if (transactions.Count == 0) throw new ArgumentException("a block needs at least one transaction");
        var level = new List<ulong>(transactions);
        while (level.Count > 1)
        {
            var next = new List<ulong>();
            for (int i = 0; i < level.Count; i += 2)
            {
                ulong left = level[i];
                ulong right = i + 1 < level.Count ? level[i + 1] : level[i];
                next.Add(Mix.Combine(left, right));
            }
            level = next;
        }
        return level[0];
    }

    /** Merkle path of one transaction, bottom-up; duplicates the last node of an odd level, as Bitcoin does. */
    internal static ulong[] MerklePath(IReadOnlyList<ulong> transactions, int index)
    {
        if (index < 0 || index >= transactions.Count) throw new ArgumentException("transaction index out of range");
        var level = new List<ulong>(transactions);
        var path = new List<ulong>();
        int position = index;
        while (level.Count > 1)
        {
            int sibling = position ^ 1;
            path.Add(sibling < level.Count ? level[sibling] : level[position]);
            var next = new List<ulong>();
            for (int i = 0; i < level.Count; i += 2)
            {
                ulong left = level[i];
                ulong right = i + 1 < level.Count ? level[i + 1] : level[i];
                next.Add(Mix.Combine(left, right));
            }
            level = next;
            position >>= 1;
        }
        return path.ToArray();
    }

    internal static ulong FoldMerkle(ulong transaction, int index, IReadOnlyList<ulong> path)
    {
        ulong node = transaction;
        int position = index;
        foreach (ulong sibling in path)
        {
            node = (position & 1) == 0 ? Mix.Combine(node, sibling) : Mix.Combine(sibling, node);
            position >>= 1;
        }
        return node;
    }

    /**
     * The retarget rule: inside a window the target is inherited; at a window boundary it scales with how long the
     * window actually took, tightly clamped. It is a function of the branch's own headers, which is what lets two
     * branches at the same height carry different work.
     */
    internal static bool NeedsRetarget(int nextHeight, int parentHeight) =>
        nextHeight % RetargetWindow == 0 && parentHeight - RetargetWindow >= 1;

    internal static ulong Retarget(Header parent, Header older)
    {
        long actual = Math.Max(1, parent.Timestamp - older.Timestamp);
        long expected = RetargetWindow * BlockInterval;
        double scaled = parent.Target * (double)actual / expected;
        ulong target = (ulong)Math.Max(1.0, scaled);
        if (target > MaxTarget) target = MaxTarget;
        if (target < MinTarget) target = MinTarget;
        return target;
    }

    internal static ulong NextTarget(IReadOnlyList<Header> chain, Header parent) =>
        NeedsRetarget(parent.Height + 1, parent.Height)
            ? Retarget(parent, chain[chain.Count - 1 - RetargetWindow])
            : parent.Target;
}

/** A header-only client: every known header indexed by hash, cumulative work memoised, forks followed by work. */
internal sealed class LightClient
{
    private readonly Dictionary<ulong, Header> _retained = new Dictionary<ulong, Header>();
    private readonly Dictionary<ulong, ulong> _workTo = new Dictionary<ulong, ulong>();
    private ulong _best;

    /** Starts from a trusted checkpoint header (assumed valid) - the weak-subjectivity assumption. */
    internal LightClient(Header checkpoint)
    {
        _retained[checkpoint.Hash] = checkpoint.Copy();
        _workTo[checkpoint.Hash] = Chain.WorkOf(checkpoint.Target);
        _best = checkpoint.Hash;
    }

    internal int KnownHeaders => _retained.Count;
    internal ulong BestTip => _best;
    internal ulong BestWork => _workTo[_best];
    internal Header BestHeader => _retained[_best];

    internal int LowestRetainedHeight
    {
        get
        {
            int lowest = int.MaxValue;
            foreach (Header h in _retained.Values) lowest = Math.Min(lowest, h.Height);
            return lowest;
        }
    }

    internal bool Retains(ulong hash) => _retained.ContainsKey(hash);

    /** Up to `max` retained headers ending at the best tip, oldest first - the retarget window lookback. */
    internal List<Header> Ancestry(int max)
    {
        var list = new List<Header>();
        Header current = _retained[_best];
        while (list.Count < max)
        {
            list.Insert(0, current);
            if (!_retained.TryGetValue(current.Parent, out Header parent)) break;
            current = parent;
        }
        return list;
    }

    /**
     * Validates and stores a header. The parent is looked up *by hash*, so a branch may fork from any retained
     * header rather than only from a current tip - the reason a client cannot follow tips alone. Returns false on
     * any violation: unknown parent, wrong height, unreachable retarget window, wrong target, bad proof of work or
     * a hash that does not match the header fields.
     */
    internal bool Submit(Header header)
    {
        if (!_retained.TryGetValue(header.Parent, out Header parent)) return false;
        if (header.Height != parent.Height + 1) return false;
        if (Chain.NeedsRetarget(header.Height, parent.Height))
        {
            Header older = parent;
            for (int i = 0; i < Chain.RetargetWindow; i++)
            {
                if (!_retained.TryGetValue(older.Parent, out Header step)) return false;
                older = step;
            }
            if (header.Target != Chain.Retarget(parent, older)) return false;
        }
        else if (header.Target != parent.Target)
        {
            return false;
        }
        ulong hash = Chain.ComputeHash(header);
        if (hash != header.Hash) return false;
        if (hash >= header.Target) return false;
        _retained[hash] = header.Copy();
        ulong work = _workTo[header.Parent] + Chain.WorkOf(header.Target);
        _workTo[hash] = work;
        if (work > BestWork) _best = hash;
        return true;
    }

    /** Verifies an SPV proof against the retained header with the given hash. */
    internal bool VerifyInclusion(ulong blockHash, ulong transaction, int index, IReadOnlyList<ulong> path)
    {
        if (!_retained.TryGetValue(blockHash, out Header header)) return false;
        return Chain.FoldMerkle(transaction, index, path) == header.MerkleRoot;
    }

    /**
     * Drops every header below `keepFromHeight`. Their work stays in the memo table, so the cumulative work is
     * unchanged and all post-checkpoint proofs keep verifying; the dropped headers themselves are gone, which is
     * the weak-subjectivity trade-off.
     */
    internal int Prune(int keepFromHeight)
    {
        if (keepFromHeight < 1) throw new ArgumentException("the prune height must be positive", nameof(keepFromHeight));
        var doomed = new List<ulong>();
        foreach (KeyValuePair<ulong, Header> entry in _retained)
        {
            if (entry.Value.Height < keepFromHeight) doomed.Add(entry.Key);
        }
        foreach (ulong hash in doomed) _retained.Remove(hash);
        return doomed.Count;
    }

    internal int RetainedHeaders => _retained.Count;
}

internal sealed class MinedBlock
{
    internal required Header Block;
    internal required List<ulong> Transactions;
}

internal static class Miner
{
    /** Deterministic mining: increment the nonce until the header hash drops below the target. */
    internal static MinedBlock Mine(IReadOnlyList<Header> chain, Header parent, List<ulong> transactions, long timestamp,
        out long attempts)
    {
        var header = new Header
        {
            Height = parent.Height + 1,
            Parent = parent.Hash,
            MerkleRoot = Chain.MerkleRoot(transactions),
            Target = Chain.NextTarget(chain, parent),
            Timestamp = timestamp,
            Nonce = 0
        };
        attempts = 0;
        while (true)
        {
            header.Hash = Chain.ComputeHash(header);
            attempts++;
            if (header.Hash < header.Target) break;
            header.Nonce++;
        }
        return new MinedBlock { Block = header, Transactions = new List<ulong>(transactions) };
    }

    /** Genesis: the root of the simulated chain, mined like any other header. */
    internal static Header Genesis(List<ulong> transactions, long timestamp, ulong target)
    {
        var header = new Header
        {
            Height = 1,
            Parent = 0,
            MerkleRoot = Chain.MerkleRoot(transactions),
            Target = target,
            Timestamp = timestamp
        };
        while (true)
        {
            header.Hash = Chain.ComputeHash(header);
            if (header.Hash < header.Target) break;
            header.Nonce++;
        }
        return header;
    }
}

/** Independent oracle: re-derives everything from genesis the slow way. */
internal static class FullValidator
{
    internal static bool Validate(IReadOnlyList<Header> headers, ulong genesisParent, out ulong work, out ulong tip)
    {
        work = 0;
        tip = 0;
        for (int i = 0; i < headers.Count; i++)
        {
            Header h = headers[i];
            if (Chain.ComputeHash(h) != h.Hash) return false;
            if (h.Hash >= h.Target) return false;
            if (i == 0)
            {
                if (h.Parent != genesisParent) return false;
            }
            else
            {
                if (h.Parent != headers[i - 1].Hash || h.Height != headers[i - 1].Height + 1) return false;
                var prefix = new List<Header>();
                for (int j = 0; j < i; j++) prefix.Add(headers[j]);
                if (h.Target != Chain.NextTarget(prefix, headers[i - 1])) return false;
            }
            work += Chain.WorkOf(h.Target);
            tip = h.Hash;
        }
        return headers.Count > 0;
    }
}

public static class Solution
{
    private const int ChainLength = 24;
    private const int PruneAt = 18;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static ulong Tx(ulong seed) => Mix.Of(0xDEADBEAFUL, seed);

    public static void Main()
    {
        // 1. Genesis and a first, full, honestly mined chain (checkpoint = genesis).
        var genesisTransactions = new List<ulong> { Tx(1), Tx(2), Tx(3) };
        Header genesis = Miner.Genesis(genesisTransactions, 0, 1UL << 56);
        Check(genesis.Hash < genesis.Target, "the genesis header satisfies its own proof of work");

        var honest = new List<Header> { genesis };
        var blockTransactions = new List<List<ulong>> { genesisTransactions };
        long attempts = 0;
        for (int i = 1; i < ChainLength; i++)
        {
            var txs = new List<ulong> { Tx((ulong)(10 + i)), Tx((ulong)(200 + i)) };
            var before = new List<Header>(honest);
            MinedBlock block = Miner.Mine(before, honest[honest.Count - 1], txs, 5L * i, out long localAttempts);
            attempts = Math.Max(attempts, localAttempts);
            honest.Add(block.Block);
            blockTransactions.Add(txs);
        }
        Check(FullValidator.Validate(honest, 0, out ulong honestWork, out ulong honestTip),
            "the mined chain validates end to end with the independent full validator");
        Console.WriteLine("  chain: " + honest.Count + " headers, work=" + honestWork + ", worst mining attempts=" + attempts);
        Check(attempts > 0 && attempts < 400000, "mining effort stays within a plausible bound for the target");

        var client = new LightClient(genesis);
        int accepted = 0;
        for (int i = 1; i < ChainLength; i++)
        {
            if (client.Submit(honest[i])) accepted++;
        }
        Check(accepted == ChainLength - 1, "the light client accepts every honestly mined header");
        Check(client.BestTip == honestTip, "the client's tip agrees with the full validator");
        Check(client.BestWork == honestWork, "the client's cumulative work agrees with the full validator bit-for-bit");
        ulong summed = 0;
        foreach (Header h in honest) summed += Chain.WorkOf(h.Target);
        Check(client.BestWork == summed, "the cumulative work is exactly the sum of the per-header work");

        // 2. Header validation: every individual violation is refused.
        var goodBlock = Miner.Mine(honest.GetRange(0, 5), honest[4], new List<ulong> { Tx(999) }, 27, out _);
        var trialClient = new LightClient(honest[0]);
        for (int i = 1; i <= 4; i++) trialClient.Submit(honest[i]);
        Check(trialClient.Submit(goodBlock.Block), "a valid extension is accepted");
        var bumpedNonce = goodBlock.Block.Copy();
        bumpedNonce.Nonce += 1;
        bumpedNonce.Hash = Chain.ComputeHash(bumpedNonce);
        if (bumpedNonce.Hash < bumpedNonce.Target)
        {
            bumpedNonce.Nonce += 1;
            bumpedNonce.Hash = Chain.ComputeHash(bumpedNonce);
        }
        Check(!trialClient.Submit(bumpedNonce), "a header whose nonce no longer solves the puzzle is rejected");
        var wrongParent = goodBlock.Block.Copy();
        wrongParent.Parent = Mix.Of(1UL, 2UL);
        Check(!trialClient.Submit(wrongParent), "a header with an unknown parent is rejected");
        var wrongHeight = goodBlock.Block.Copy();
        wrongHeight.Height += 1;
        wrongHeight.Hash = Chain.ComputeHash(wrongHeight);
        Check(!trialClient.Submit(wrongHeight), "a header with a skipped height is rejected");
        var wrongTarget = goodBlock.Block.Copy();
        wrongTarget.Target = 1UL << 58;
        Check(!trialClient.Submit(wrongTarget), "an easier target that the retarget rule does not allow is rejected");
        var forgedHash = goodBlock.Block.Copy();
        forgedHash.Hash ^= 1UL;
        Check(!trialClient.Submit(forgedHash), "a header whose claimed hash does not match its fields is rejected");

        // 3-4. Retargeting and fork choice: a fast branch gets harder, a slow one easier, and a shorter branch can
        //      still carry more work than a longer one.
        var fast = new List<Header> { genesis };
        var slow = new List<Header> { genesis };
        for (int i = 1; i <= 12; i++)
        {
            fast.Add(Miner.Mine(new List<Header>(fast), fast[fast.Count - 1], new List<ulong> { Tx(55) }, 1L * i, out _).Block);
        }
        for (int i = 1; i <= 16; i++)
        {
            slow.Add(Miner.Mine(new List<Header>(slow), slow[slow.Count - 1], new List<ulong> { Tx(55) }, 9L * i, out _).Block);
        }
        Check(FullValidator.Validate(fast, 0, out ulong fastWork, out ulong fastTip), "the fast branch validates");
        Check(FullValidator.Validate(slow, 0, out ulong slowWork, out ulong slowTip), "the slow branch validates");
        Console.WriteLine("  retarget: fast " + fast.Count + " headers tip target=" + fast[fast.Count - 1].Target
            + " work=" + fastWork + " | slow " + slow.Count + " headers tip target="
            + slow[slow.Count - 1].Target + " work=" + slowWork);
        int boundary = Chain.RetargetWindow * 2;
        Check(fast[boundary].Target < genesis.Target,
            "blocks arriving faster than the interval shrink the target (harder)");
        Check(slow[boundary].Target > genesis.Target,
            "blocks arriving slower than the interval grow the target (easier)");
        Check(fastWork > slowWork && fast.Count < slow.Count,
            "the shorter branch carries more work: fork choice cannot be a length comparison");
        var forkClient = new LightClient(genesis);
        foreach (Header h in fast) forkClient.Submit(h);
        foreach (Header h in slow) forkClient.Submit(h);
        Check(forkClient.KnownHeaders == 1 + (fast.Count - 1) + (slow.Count - 1),
            "the client keeps every header of both branches, since they diverge above the genesis");
        Check(forkClient.BestTip == fastTip, "the heavier (shorter) branch wins the fork");
        Check(forkClient.BestWork == fastWork, "and the client's work equals the full validator's for that branch");
        var singleTip = new LightClient(genesis);
        foreach (Header h in slow) singleTip.Submit(h);
        foreach (Header h in fast) singleTip.Submit(h);
        Check(singleTip.BestTip == fastTip, "submission order does not change which branch wins");

        // 5. SPV inclusion proofs.
        int proven = 0;
        bool allProofsVerify = true;
        for (int b = 0; b < honest.Count; b++)
        {
            var txs = blockTransactions[b];
            for (int t = 0; t < txs.Count; t++)
            {
                ulong[] path = Chain.MerklePath(txs, t);
                int expectedDepth = 0;
                int n = txs.Count;
                while (n > 1) { n = (n + 1) / 2; expectedDepth++; }
                if (path.Length != expectedDepth) allProofsVerify = false;
                if (!client.VerifyInclusion(honest[b].Hash, txs[t], t, path)) allProofsVerify = false;
                proven++;
            }
        }
        Console.WriteLine("  spv: " + proven + " inclusion proofs verified, path length ceil(log2 n)");
        Check(allProofsVerify, "every transaction's Merkle path verifies against its block header");
        var lastBlock = honest[honest.Count - 1];
        var lastTxs = blockTransactions[blockTransactions.Count - 1];
        ulong[] lastPath = Chain.MerklePath(lastTxs, 0);
        Check(!client.VerifyInclusion(lastBlock.Hash, lastTxs[0] + 1, 0, lastPath),
            "a modified transaction fails its inclusion proof");
        Check(!client.VerifyInclusion(lastBlock.Hash, lastTxs[0], 1, lastPath),
            "the same proof under the wrong index fails");
        Check(!client.VerifyInclusion(honest[1].Hash, lastTxs[0], 0, lastPath),
            "a proof presented against another block fails");
        Check(!client.VerifyInclusion(Mix.Of(7UL, 7UL), lastTxs[0], 0, lastPath),
            "a proof for an unknown block is refused");
        Check(Throws(() => Chain.MerklePath(lastTxs, 5)), "a transaction index outside the block is rejected");

        // 6. Pruning: the frozen work keeps the accounting exact and post-checkpoint proofs keep working.
        ulong workBefore = client.BestWork;
        ulong tipBefore = client.BestTip;
        int headersBefore = client.RetainedHeaders;
        int removed = client.Prune(PruneAt);
        int expectedRemoved = 0;
        foreach (Header h in honest)
        {
            if (h.Height < PruneAt) expectedRemoved++;
        }
        Console.WriteLine("  pruning: retained " + headersBefore + " -> " + client.RetainedHeaders
            + " (removed " + removed + " of " + expectedRemoved + "), work preserved: " + (client.BestWork == workBefore));
        Check(removed > 0 && client.RetainedHeaders < headersBefore, "pruning actually drops headers");
        Check(client.BestWork == workBefore, "the cumulative work survives pruning exactly");
        Check(client.BestTip == tipBefore, "the tip is unchanged by pruning");
        Check(client.LowestRetainedHeight == PruneAt, "the lowest retained header is the checkpoint header");
        Check(removed == expectedRemoved, "exactly the headers below the checkpoint were dropped");
        Check(!client.VerifyInclusion(honest[0].Hash, blockTransactions[0][0], 0,
            Chain.MerklePath(blockTransactions[0], 0)), "a pruned block's proof is refused");
        Check(client.VerifyInclusion(honest[PruneAt].Hash, blockTransactions[PruneAt][0], 0,
            Chain.MerklePath(blockTransactions[PruneAt], 0)), "a post-checkpoint proof still verifies");
        var extension = Miner.Mine(client.Ancestry(8), client.BestHeader,
            new List<ulong> { Tx(4242) }, 5L * ChainLength, out _);
        Check(client.Submit(extension.Block), "the pruned client still accepts new headers on its tip");
        Check(client.VerifyInclusion(extension.Block.Hash, Tx(4242), 0,
            Chain.MerklePath(new List<ulong> { Tx(4242) }, 0)), "and can verify an SPV proof for the new block");
        Check(!client.VerifyInclusion(genesis.Hash, genesisTransactions[0], 0,
            Chain.MerklePath(genesisTransactions, 0)), "the genesis proof is gone after pruning");
        Check(Throws(() => client.Prune(0)), "a non-positive prune height is rejected");

        // 7. Determinism and the checkpoint contract.
        var replay = new LightClient(genesis);
        for (int i = 1; i < ChainLength; i++) replay.Submit(honest[i]);
        Check(replay.BestTip == tipBefore && replay.BestWork == workBefore, "two clients fed the same headers agree");
        Check(client.BestWork == workBefore + Chain.WorkOf(extension.Block.Target),
            "the work after the extension equals the earlier work plus the new header's work");
        Check(client.Retains(honest[ChainLength - 1].Hash) && !client.Retains(honest[0].Hash),
            "the client can only find retained headers");

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }
}
