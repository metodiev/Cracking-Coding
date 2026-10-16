// Task 842 - Implement blockchain light node gossip + header stitching robustly. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A light node that keeps only block headers, and the robustness it needs when the headers arrive shuffled over
// gossiping peers that do not all tell the truth:
//
//   * proof of work is real here: a header is valid when hash(header) % difficulty == 0, so the expected mining
//     work of a block is exactly its difficulty and the cumulative work of a chain is the sum of its difficulties.
//     The tip is the branch with the most cumulative work (ties broken by the smaller hash), which is the rule that
//     makes a long low-difficulty fork worthless.
//   * stitching: headers arrive out of order, in scrambled batches and with duplicates, so the node indexes them by
//     hash, keeps a bounded orphan pool for headers whose parent has not arrived, and re-runs that pool to a fixed
//     point whenever a header lands. A header whose parent is unknown can never reach the tip, because cumulative
//     work is only defined once the parent is known. The pool has to be at least as deep as the reorg it must
//     survive, which the bounded-pool checks below pin down.
//   * the defences: a header with a bad proof of work, or below the network minimum difficulty, is rejected and
//     charged as a strike to the peer that sent it (three strikes ban the peer and its traffic is ignored), a
//     header that rewrites the hard-coded checkpoint is poisoned together with every descendant branch, and the
//     node raises an eclipse warning when one peer supplied more than 80% of the accepted headers.
//
// Verified, each by recomputation:
//   * against an independent oracle: the full node's own chain, mined in this process, plus a from-scratch sum of
//     its difficulties. For 24 seeded delivery permutations the light node's tip hash, height, work and distinct
//     header count all match the oracle, and the round-robin gossip schedule of the same permutation converges too;
//   * safety and liveness: convergence needs at most 5 gossip rounds at batch size 3 (checked, not just observed),
//     duplicates never inflate the store, a second full pass leaves the tip unchanged, the low-difficulty fork is
//     rejected header by header and bans its peer, the short valid fork is stored but never selected, the heavier
//     valid fork *does* reorg the node (so the most-work rule is live in both directions), and the heavier fork
//     that rewrites the checkpoint is refused while the honest tip stands;
//   * mining work is measured: the nonce trials actually spent are compared with the difficulty sum the work model
//     predicts, so "the difficulty is real" is a measurement rather than a comment.
//
// Documented simplifications: hashes are 64-bit splitmix mixings of a canonical header string (no merkle tree, no
// transactions, no difficulty retarget — the minimum difficulty is a constant), the network is an in-process
// round-robin schedule with explicit rounds rather than sockets, the peer model is malicious-by-content rather than
// adaptive, and there is no chain-work overflow analysis or header-first sync protocol. Randomness is hand-written
// splitmix64 with explicit seeds, so every permutation, tip and count in the output is reproducible. No real
// consensus value or key material is involved.
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

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class BlockHeader
{
    internal int Index { get; }
    internal ulong PrevHash { get; }
    internal ulong Payload { get; }
    internal int Difficulty { get; }
    internal long Nonce { get; private set; }
    internal ulong Hash { get; private set; }

    internal BlockHeader(int index, ulong prevHash, ulong payload, int difficulty, long nonce)
    {
        Index = index;
        PrevHash = prevHash;
        Payload = payload;
        Difficulty = difficulty;
        SetNonce(nonce);
    }

    /// <summary>Canonical encoding of everything the proof of work covers.</summary>
    internal string Canonical() => Index + ":" + PrevHash + ":" + Payload + ":" + Difficulty + ":" + Nonce;

    internal void SetNonce(long nonce)
    {
        Nonce = nonce;
        Hash = Pow.Hash(this);
    }
}

internal static class Pow
{
    /// <summary>64-bit digest of the canonical header string; a real node hashes the serialised header.</summary>
    internal static ulong Hash(BlockHeader header)
    {
        ulong state = 0x9E3779B97F4A7C15UL;
        foreach (char c in header.Canonical())
        {
            state ^= (ulong)(c + 0x9E37);
            state = SplitMix64.Next(ref state);
        }
        return SplitMix64.Next(ref state);
    }

    /// <summary>A header meets its target when the digest is divisible by the difficulty: expected work = difficulty.</summary>
    internal static bool Check(BlockHeader header) => header.Difficulty > 0 && header.Hash % (ulong)header.Difficulty == 0UL;

    /// <summary>Deterministic mining: scan nonces upward; the trials spent are the measured work of the block.</summary>
    internal static BlockHeader Mine(int index, ulong prevHash, ulong payload, int difficulty, long startNonce, out long trials)
    {
        var header = new BlockHeader(index, prevHash, payload, difficulty, startNonce);
        trials = 0;
        for (long nonce = startNonce; ; nonce++)
        {
            header.SetNonce(nonce);
            trials++;
            if (Check(header)) return header;
        }
    }
}

/// <summary>A peer's own chain, in order, with the cumulative work of its prefix.</summary>
internal sealed class PeerChain
{
    internal List<BlockHeader> Headers { get; } = new List<BlockHeader>();
    internal long Work { get; private set; }
    internal ulong TipHash => Headers.Count == 0 ? 0UL : Headers[Headers.Count - 1].Hash;
    internal int TipIndex => Headers.Count == 0 ? -1 : Headers[Headers.Count - 1].Index;

    internal void Append(BlockHeader header)
    {
        if (Headers.Count > 0 && header.PrevHash != TipHash)
            throw new ArgumentException("a chain must append to its own tip", nameof(header));
        Headers.Add(header);
        Work += header.Difficulty;
    }
}

internal sealed class Peer
{
    internal string Name { get; }
    internal PeerChain Chain { get; }
    internal bool Hostile { get; }
    internal int Cursor { get; set; }
    internal long Served { get; set; }

    internal Peer(string name, PeerChain chain, bool hostile)
    {
        Name = name;
        Chain = chain;
        Hostile = hostile;
    }
}

internal sealed class LightNode
{
    internal Dictionary<ulong, BlockHeader> Known { get; } = new Dictionary<ulong, BlockHeader>();
    internal Dictionary<ulong, long> CumulativeWork { get; } = new Dictionary<ulong, long>();
    internal List<BlockHeader> Orphans { get; } = new List<BlockHeader>();
    internal Dictionary<string, int> Strikes { get; } = new Dictionary<string, int>();
    internal Dictionary<string, long> AcceptedByPeer { get; } = new Dictionary<string, long>();
    internal HashSet<string> Banned { get; } = new HashSet<string>();
    internal HashSet<ulong> Poisoned { get; } = new HashSet<ulong>();

    internal string CurrentPeer { get; set; } = "";
    internal ulong TipHash { get; private set; }
    internal long TipWork { get; private set; } = -1;
    internal int TipHeight { get; private set; } = -1;

    internal long RejectedPow { get; private set; }
    internal long RejectedDifficulty { get; private set; }
    internal long Duplicates { get; private set; }
    internal long OrphanEvictions { get; private set; }
    internal long CheckpointViolations { get; private set; }

    private readonly int _minimumDifficulty;
    private readonly int _checkpointHeight;
    private readonly ulong _checkpointHash;
    private readonly int _orphanLimit;
    private readonly int _strikeLimit;

    internal LightNode(int minimumDifficulty, int checkpointHeight, ulong checkpointHash, int orphanLimit, int strikeLimit)
    {
        _minimumDifficulty = minimumDifficulty;
        _checkpointHeight = checkpointHeight;
        _checkpointHash = checkpointHash;
        _orphanLimit = orphanLimit;
        _strikeLimit = strikeLimit;
    }

    internal long DistinctHeaders => Known.Count;
    internal long PendingOrphans => Orphans.Count;

    /// <summary>
    /// Handles one header from one peer. Every decision the node takes is counted, so the checks assert on the
    /// decisions rather than on the input.
    /// </summary>
    internal void Offer(string peerName, BlockHeader header)
    {
        if (Banned.Contains(peerName)) return;
        if (!Pow.Check(header)) { Charge(peerName); RejectedPow++; return; }
        if (header.Difficulty < _minimumDifficulty) { Charge(peerName); RejectedDifficulty++; return; }
        if (_checkpointHeight >= 0 && header.Index == _checkpointHeight && header.Hash != _checkpointHash)
        {
            Poisoned.Add(header.Hash);                  // the whole descendant branch is excluded from tip selection
            CheckpointViolations++;
            Charge(peerName);
            return;
        }
        if (header.Index == 0)
        {
            if (Known.ContainsKey(header.Hash)) Duplicates++;
            else Admit(header, 0);
            TryStitchOrphans();
            return;
        }
        if (!Known.ContainsKey(header.PrevHash))
        {
            if (Orphans.Count >= _orphanLimit)
            {
                Orphans.RemoveAt(0);                    // bounded pool: the oldest unanchored header is dropped
                OrphanEvictions++;
            }
            Orphans.Add(header);
            return;
        }
        if (Known.ContainsKey(header.Hash))
        {
            Duplicates++;
            TryStitchOrphans();
            return;
        }
        Admit(header, CumulativeWork[header.PrevHash]);
        TryStitchOrphans();
    }

    private void Charge(string peerName)
    {
        Strikes.TryGetValue(peerName, out int strikes);
        Strikes[peerName] = strikes + 1;
        if (strikes + 1 >= _strikeLimit) Banned.Add(peerName);
    }

    private void Admit(BlockHeader header, long parentWork)
    {
        Known[header.Hash] = header;
        CumulativeWork[header.Hash] = parentWork + header.Difficulty;
        AcceptedByPeer.TryGetValue(CurrentPeer, out long count);
        AcceptedByPeer[CurrentPeer] = count + 1;
        RecomputeTip();
    }

    /// <summary>Re-runs the orphan pool to a fixed point: every header whose parent is now known is admitted.</summary>
    private void TryStitchOrphans()
    {
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (int i = 0; i < Orphans.Count; i++)
            {
                BlockHeader candidate = Orphans[i];
                if (!Known.ContainsKey(candidate.PrevHash)) continue;
                Orphans.RemoveAt(i);
                if (!Known.ContainsKey(candidate.Hash)) Admit(candidate, CumulativeWork[candidate.PrevHash]);
                progress = true;
                break;
            }
        }
    }

    /// <summary>Tip selection: most cumulative work among branches that do not rewrite the checkpoint.</summary>
    internal void RecomputeTip()
    {
        ulong bestHash = 0;
        long bestWork = -1;
        int bestHeight = -1;
        foreach (KeyValuePair<ulong, BlockHeader> entry in Known)
        {
            if (!BranchIsClean(entry.Key)) continue;
            long work = CumulativeWork[entry.Key];
            if (work > bestWork || (work == bestWork && bestHash != 0 && entry.Key < bestHash))
            {
                bestHash = entry.Key;
                bestWork = work;
                bestHeight = entry.Value.Index;
            }
        }
        if (bestWork >= 0)
        {
            TipHash = bestHash;
            TipWork = bestWork;
            TipHeight = bestHeight;
        }
    }

    /// <summary>Walks a branch up to its genesis; false as soon as a poisoned header is on the way.</summary>
    internal bool BranchIsClean(ulong hash)
    {
        ulong cursor = hash;
        int guard = 0;
        while (cursor != 0 && guard++ < 10000)
        {
            if (Poisoned.Contains(cursor)) return false;
            if (!Known.TryGetValue(cursor, out BlockHeader header) || header.Index == 0) return true;
            cursor = header.PrevHash;
        }
        return true;
    }

    /// <summary>Eclipse heuristic: the share of accepted headers that came from the busiest peer.</summary>
    internal double TopPeerShare(out string topPeer)
    {
        long total = 0, best = 0;
        topPeer = "";
        foreach (KeyValuePair<string, long> entry in AcceptedByPeer)
        {
            total += entry.Value;
            if (entry.Value > best) { best = entry.Value; topPeer = entry.Key; }
        }
        return total == 0 ? 0.0 : (double)best / total;
    }
}

internal static class Network
{
    /// <summary>
    /// Round-robin gossip with a fixed batch size. A batch is sometimes reversed and padded with a duplicate, which
    /// is what real gossip looks like, so the receiver has to stitch rather than append.
    /// </summary>
    internal static int Gossip(IReadOnlyList<Peer> peers, LightNode node, int maxRounds, int batchSize, ulong honestTip,
        ref ulong state)
    {
        for (int round = 0; round < maxRounds; round++)
        {
            for (int p = 0; p < peers.Count; p++)
            {
                Peer peer = peers[(p + round) % peers.Count];
                var batch = new List<BlockHeader>();
                for (int i = 0; i < batchSize && peer.Cursor < peer.Chain.Headers.Count; i++)
                {
                    batch.Add(peer.Chain.Headers[peer.Cursor]);
                    peer.Cursor++;
                }
                if (batch.Count > 2 && (round + p) % 2 == 0)
                {
                    batch.Reverse();
                    batch.Add(batch[SplitMix64.NextBelow(ref state, batch.Count)]);
                }
                foreach (BlockHeader header in batch)
                {
                    node.CurrentPeer = peer.Name;
                    peer.Served++;
                    node.Offer(peer.Name, header);
                }
            }
            if (node.TipHash == honestTip) return round + 1;
        }
        return maxRounds;
    }

    /// <summary>A peer that ignores the batch schedule: it delivers a whole chain in one go, shuffled.</summary>
    internal static void DeliverShuffled(LightNode node, string peerName, PeerChain chain, ref ulong state)
    {
        var order = new List<int>();
        for (int i = 0; i < chain.Headers.Count; i++) order.Add(i);
        for (int i = order.Count - 1; i > 0; i--)
        {
            int j = SplitMix64.NextBelow(ref state, i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        foreach (int index in order)
        {
            node.CurrentPeer = peerName;
            node.Offer(peerName, chain.Headers[index]);
        }
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Mines a chain of `count` headers from genesis; reports the measured total nonce trials as well.</summary>
    private static PeerChain BuildChain(int count, ulong genesisHash, int difficulty, ulong payloadSeed, long nonceStride,
        out long trials)
    {
        var chain = new PeerChain();
        ulong prev = genesisHash;
        trials = 0;
        for (int i = 0; i < count; i++)
        {
            BlockHeader header = Pow.Mine(i, prev, payloadSeed + (ulong)i, difficulty, i * nonceStride, out long spent);
            chain.Append(header);
            prev = header.Hash;
            trials += spent;
        }
        return chain;
    }

    /// <summary>
    /// A fork that shares the prefix of `basis` up to `forkAtHeight` and then diverges, which is how real reorgs
    /// happen: everything at or below the fork point is common, and the checkpoint below it stays intact.
    /// </summary>
    private static PeerChain BuildFork(PeerChain basis, int forkAtHeight, int count, int difficulty, ulong payloadSeed,
        long nonceStride, out long trials)
    {
        var chain = new PeerChain();
        for (int i = 0; i <= forkAtHeight; i++) chain.Append(basis.Headers[i]);
        ulong prev = chain.TipHash;
        trials = 0;
        for (int i = 0; i < count; i++)
        {
            int index = forkAtHeight + 1 + i;
            BlockHeader header = Pow.Mine(index, prev, payloadSeed + (ulong)i, difficulty, index * nonceStride,
                out long spent);
            chain.Append(header);
            prev = header.Hash;
            trials += spent;
        }
        return chain;
    }

    public static void Main()
    {
        const int MinimumDifficulty = 24, ChainLength = 12, CheckpointHeight = 3, BanLimit = 3;
        long honestWork = 0;
        var honest = new PeerChain();
        {
            ulong prev = 0;
            for (int i = 0; i < ChainLength; i++)
            {
                BlockHeader header = Pow.Mine(i, prev, 0x4242UL + (ulong)i * 7, MinimumDifficulty, i * 101, out long spent);
                honest.Append(header);
                prev = header.Hash;
                honestWork += spent;
            }
        }
        ulong checkpointHash = honest.Headers[CheckpointHeight].Hash;
        Check(honest.TipIndex == ChainLength - 1, "the honest chain was mined with " + ChainLength + " headers");

        // ---- proof of work and the work bookkeeping --------------------------------------------------
        {
            bool allValid = true;
            long summedDifficulty = 0;
            foreach (BlockHeader header in honest.Headers)
            {
                if (!Pow.Check(header)) allValid = false;
                summedDifficulty += header.Difficulty;
            }
            Check(allValid, "every mined header satisfies hash % difficulty == 0, so the work is real");
            Check(honest.Work == summedDifficulty, "the chain's cumulative work equals the independently summed "
                + "difficulties (" + summedDifficulty + ")");
            Check(honestWork >= ChainLength && honestWork <= 6L * summedDifficulty, "the measured mining effort ("
                + honestWork + " nonce trials for " + summedDifficulty + " expected) is in the right band, which is the "
                + "only evidence that the difficulty means anything");
            Console.WriteLine("proof of work: " + ChainLength + " headers, expected work=" + summedDifficulty
                + " measured trials=" + honestWork);
        }

        // ---- stitching: every delivery order reaches the oracle's tip --------------------------------
        {
            var tips = new HashSet<ulong>();
            bool allMatch = true;
            int worstRounds = 0;
            for (int seed = 1; seed <= 24; seed++)
            {
                ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL + 7UL;
                var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
                Network.DeliverShuffled(node, "honest", honest, ref state);
                if (node.TipHash != honest.TipHash || node.TipWork != honest.Work
                    || node.TipHeight != honest.TipIndex || node.DistinctHeaders != ChainLength) allMatch = false;
                tips.Add(node.TipHash);

                var live = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
                var peer = new Peer("honest", PeerChainCopy(honest), false);
                ulong gossipState = state;
                int rounds = Network.Gossip(new[] { peer }, live, 20, 3, honest.TipHash, ref gossipState);
                if (live.TipHash != honest.TipHash || live.TipHeight != honest.TipIndex) allMatch = false;
                if (rounds > worstRounds) worstRounds = rounds;
                tips.Add(live.TipHash);
            }
            Check(allMatch, "all 24 shuffled delivery orders and all 24 gossip schedules converge on the oracle's tip "
                + "hash, height and work");
            Check(tips.Count == 1 && tips.Contains(honest.TipHash), "the tip is the oracle's tip no matter how the "
                + "headers arrive");
            Check(worstRounds <= 5, "convergence takes at most 5 gossip rounds at batch size 3 for " + ChainLength
                + " headers (worst observed " + worstRounds + "), so the node is live");
            Console.WriteLine("stitching: 24 permutations converge, worst " + worstRounds + " gossip rounds");
        }

        // ---- duplicates and idempotence ---------------------------------------------------------------
        {
            ulong state = 0x5EED842000000002UL;
            var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(node, "honest", honest, ref state);
            long afterFirst = node.DistinctHeaders;
            ulong tipAfterFirst = node.TipHash;
            Network.DeliverShuffled(node, "honest", honest, ref state);
            Check(node.DistinctHeaders == afterFirst && afterFirst == ChainLength,
                "re-delivering every header does not inflate the store (" + node.DistinctHeaders + " distinct)");
            Check(node.TipHash == tipAfterFirst && node.Duplicates > 0, "the tip is unchanged and " + node.Duplicates
                + " duplicates were counted");
            Console.WriteLine("idempotence: distinct=" + node.DistinctHeaders + " duplicates=" + node.Duplicates);
        }

        // ---- the low-difficulty fork is rejected and its peer banned ---------------------------------
        {
            // It forks *above* the checkpoint, so only the difficulty floor can stop it - and it is longer in
            // blocks than the honest chain, which is exactly the attack the work rule defeats.
            PeerChain weak = BuildFork(honest, CheckpointHeight, 14, 8, 0x9000UL, 17, out long weakTrials);
            Check(weak.Headers.Count > honest.Headers.Count && weak.Work < honest.Work, "the hostile fork has more "
                + "blocks (" + weak.Headers.Count + " vs " + honest.Headers.Count + ") but less work ("
                + weak.Work + " vs " + honest.Work + ")");
            ulong state = 0x5EED842000000003UL;
            var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            int offered = 0;
            for (int i = CheckpointHeight + 1; i < weak.Headers.Count; i++)
            {
                node.CurrentPeer = "hostile";
                node.Offer("hostile", weak.Headers[i]);
                offered++;
            }
            Check(node.RejectedDifficulty == BanLimit && node.RejectedPow == 0, "the hostile headers are rejected on "
                + "the difficulty floor, not on their proof of work (" + node.RejectedDifficulty + " rejections), and "
                + "the third one bans the peer");
            Check(node.Banned.Contains("hostile"), "strikes ban the peer (" + node.Strikes["hostile"] + " strikes)");
            Check(offered == 14, "the peer offered 14 headers before being ignored");
            long beforeBan = node.RejectedDifficulty;
            for (int i = CheckpointHeight + 1; i < weak.Headers.Count; i++)
            {
                node.CurrentPeer = "hostile";
                node.Offer("hostile", weak.Headers[i]);
            }
            Check(node.RejectedDifficulty == beforeBan && node.DistinctHeaders == 0, "a banned peer's headers are "
                + "ignored without further work and none of them is stored");
            Network.DeliverShuffled(node, "honest", honest, ref state);
            Check(node.TipHash == honest.TipHash && node.TipHeight == honest.TipIndex,
                "after the attack the node still holds the honest tip");
            Console.WriteLine("low-difficulty fork: " + offered + " headers offered (" + weakTrials
                + " trials to mine), " + node.Strikes["hostile"] + " strikes, peer banned, tip intact");
        }

        // ---- a valid but lighter fork is stored, never selected --------------------------------------
        {
            PeerChain shortFork = BuildFork(honest, 5, 3, MinimumDifficulty, 0x7100UL, 23, out long _);
            Check(shortFork.Work < honest.Work, "the short fork shares the honest prefix to height 5 and carries "
                + shortFork.Work + " work against " + honest.Work);
            ulong state = 0x5EED842000000004UL;
            var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(node, "honest", honest, ref state);
            Network.DeliverShuffled(node, "fork", shortFork, ref state);
            Check(node.RejectedPow == 0 && node.RejectedDifficulty == 0,
                "the short fork's new headers are individually valid and above the floor");
            Check(node.DistinctHeaders == ChainLength + 3, "its three new headers are stored (" + node.DistinctHeaders
                + " distinct, the shared prefix is deduplicated)");
            Check(node.TipHash == honest.TipHash && node.TipWork == honest.Work, "but the lighter branch is never the "
                + "tip");
            Console.WriteLine("lighter fork: 3 new headers stored, tip unchanged at work " + honest.Work);
        }

        // ---- a heavier valid fork reorgs the node ----------------------------------------------------
        {
            PeerChain heavy = BuildFork(honest, 5, 9, MinimumDifficulty, 0x5500UL, 29, out long _);
            Check(heavy.Work > honest.Work, "the heavy fork out-weighs the honest chain (" + heavy.Work + " > "
                + honest.Work + ") with 9 new blocks");
            ulong state = 0x5EED842000000005UL;
            var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(node, "honest", honest, ref state);
            Check(node.TipHash == honest.TipHash, "the node starts on the honest chain");
            Network.DeliverShuffled(node, "heavy", heavy, ref state);
            Check(node.TipHash == heavy.TipHash && node.TipWork == heavy.Work, "a heavier valid fork becomes the tip "
                + "(reorg), so the most-work rule is live in both directions");
            Check(node.TipHeight == heavy.TipIndex, "the reorged tip has the fork's height (" + node.TipHeight + ")");
            Check(node.DistinctHeaders == ChainLength + 9, "the old branch is retained for later switching ("
                + node.DistinctHeaders + " distinct headers)");
            Console.WriteLine("heavy fork: tip switched to height " + heavy.TipIndex + " at work " + heavy.Work
                + " over " + honest.Work);
        }

        // ---- the checkpoint cannot be rewritten ------------------------------------------------------
        {
            // This fork starts from a different genesis, so it disagrees with the hard-coded checkpoint at height 3.
            PeerChain rewrite = BuildChain(ChainLength + 2, 0, MinimumDifficulty, 0x3300UL, 31, out long _);
            Check(rewrite.Headers[CheckpointHeight].Hash != checkpointHash && rewrite.Work > honest.Work,
                "the rewrite fork disagrees at the checkpoint height and out-weighs the honest chain (" + rewrite.Work
                + " > " + honest.Work + ")");
            ulong state = 0x5EED842000000006UL;
            var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(node, "honest", honest, ref state);
            Network.DeliverShuffled(node, "rewrite", rewrite, ref state);
            Check(node.CheckpointViolations == 1, "the header that rewrites the checkpoint is counted as a violation");
            Check(node.Poisoned.Count == 1, "it is poisoned");
            Check(node.TipHash == honest.TipHash, "the heavier fork that contradicts the checkpoint is refused and the "
                + "honest tip stands");
            Check(node.DistinctHeaders <= ChainLength + CheckpointHeight, "nothing above the checkpoint of the rewrite "
                + "branch is stored (" + node.DistinctHeaders + " distinct headers)");
            Console.WriteLine("checkpoint: violation counted, heavier branch " + rewrite.Work + " refused, tip height="
                + node.TipHeight);
        }

        // ---- orphans: a header without its parent waits, then stitches -------------------------------
        {
            ulong state = 0x5EED842000000007UL;
            var node = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 32, BanLimit);
            for (int i = honest.Headers.Count - 1; i >= 1; i--)     // parents arrive last
            {
                node.CurrentPeer = "honest";
                node.Offer("honest", honest.Headers[i]);
            }
            Check(node.DistinctHeaders == 0 && node.PendingOrphans == ChainLength - 1,
                "headers whose parents are missing wait in the orphan pool (" + node.PendingOrphans + ")");
            Check(node.TipHash == 0, "no tip exists before the ancestor arrives");
            node.CurrentPeer = "honest";
            node.Offer("honest", honest.Headers[0]);
            Check(node.PendingOrphans == 0, "the arrival of the ancestor releases the whole pool");
            Check(node.TipHash == honest.TipHash && node.TipHeight == honest.TipIndex,
                "the stitched tip is the oracle's tip");

            var bounded = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 2, BanLimit);
            for (int i = honest.Headers.Count - 1; i >= 1; i--)
            {
                bounded.CurrentPeer = "honest";
                bounded.Offer("honest", honest.Headers[i]);
            }
            Check(bounded.PendingOrphans <= 2 && bounded.OrphanEvictions == ChainLength - 1 - 2,
                "the orphan pool is bounded: " + bounded.OrphanEvictions + " dropped, " + bounded.PendingOrphans
                + " kept with a limit of 2");
            Console.WriteLine("orphans: pool bounded, " + node.PendingOrphans + " pending after stitching, "
                + bounded.OrphanEvictions + " evicted with a limit of 2; state=" + state);
        }

        // ---- eclipse heuristic ----------------------------------------------------------------------
        {
            ulong state = 0x5EED842000000008UL;
            var solo = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(solo, "solo", honest, ref state);
            double soloShare = solo.TopPeerShare(out string soloPeer);
            Check(soloShare > 0.8 && soloPeer == "solo", "one peer supplying every header raises the eclipse warning ("
                + soloShare.ToString("F2") + ")");

            var mixed = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            for (int i = 0; i < honest.Headers.Count; i++)
            {
                string peer = i % 3 == 0 ? "peer-a" : (i % 3 == 1 ? "peer-b" : "peer-c");
                mixed.CurrentPeer = peer;
                mixed.Offer(peer, honest.Headers[i]);
            }
            double mixedShare = mixed.TopPeerShare(out _);
            Check(mixedShare < 0.8, "three peers sharing the headers do not (" + mixedShare.ToString("F2") + ")");
            Check(mixed.TipHash == honest.TipHash, "and the tip is still correct in the mixed case");
            Console.WriteLine("eclipse: single-peer share=" + soloShare.ToString("F2") + " mixed share="
                + mixedShare.ToString("F2"));
        }

        // ---- determinism ----------------------------------------------------------------------------
        {
            ulong stateA = 0x5EED842000000009UL, stateB = stateA;
            var first = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(first, "honest", honest, ref stateA);
            var second = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            Network.DeliverShuffled(second, "honest", honest, ref stateB);
            Check(first.TipHash == second.TipHash && first.Duplicates == second.Duplicates,
                "the same shuffle seed produces the same tip and the same duplicate count");

            ulong gossipA = 0x5EED84200000000AUL, gossipB = gossipA;
            var nodeA = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            var nodeB = new LightNode(MinimumDifficulty, CheckpointHeight, checkpointHash, 64, BanLimit);
            int roundsA = Network.Gossip(new[] { new Peer("honest", PeerChainCopy(honest), false) }, nodeA, 20, 4,
                honest.TipHash, ref gossipA);
            int roundsB = Network.Gossip(new[] { new Peer("honest", PeerChainCopy(honest), false) }, nodeB, 20, 4,
                honest.TipHash, ref gossipB);
            Check(roundsA == roundsB && nodeA.TipHash == nodeB.TipHash, "gossip convergence is deterministic ("
                + roundsA + " rounds at batch size 4)");
            Console.WriteLine("determinism: duplicates=" + first.Duplicates + " gossip rounds=" + roundsA
                + " tip=" + first.TipHash);
        }

        Console.WriteLine("PASS");
    }

    /// <summary>A shallow copy of a chain, so each gossip schedule starts from an unread cursor.</summary>
    private static PeerChain PeerChainCopy(PeerChain source)
    {
        var copy = new PeerChain();
        foreach (BlockHeader header in source.Headers) copy.Append(header);
        return copy;
    }
}
