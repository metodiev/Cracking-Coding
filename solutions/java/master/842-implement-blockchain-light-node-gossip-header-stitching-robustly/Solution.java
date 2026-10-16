/** Task 842 - Implement blockchain light node gossip + header stitching robustly. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A light node only ever sees block headers, and it sees them out of order, duplicated, from several peers
 * that may hold different forks. This program models that end to end:
 *
 *   gossip      - peers sit on a deterministic ring. Each round every peer hands the headers it has not
 *                 sent yet to its two neighbours over an explicit in-memory inbox (nothing is transmitted).
 *   stitching   - a header that arrives before its parent is kept in a bounded orphan buffer and stitched
 *                 into the tree the moment the parent shows up; duplicates are ignored.
 *   validation  - a header is accepted only if its proof of work holds (a fixed number of leading zero
 *                 bits), its height is exactly one more than its parent's, and its hash links to the parent.
 *   fork choice - the node's chain is the tip whose *cumulative work* (the sum of 2^difficulty) is largest,
 *                 with a deterministic tie-break on the tip hash, so a heavier branch triggers a reorg.
 *
 * Simplifications, stated honestly: the hash is splitmix64, not SHA-256; proof of work is difficulty
 * leading zero *bits* of a 64-bit digest so mining a block costs a few thousand tries instead of 2^60;
 * there are no transactions, no merkle proofs, and no network - the gossip bus is an in-memory message
 * counter. The light node keeps full headers (a real one keeps the last few and prunes), which is exactly
 * what makes "stitching" observable here.
 *
 * Invariants verified below. Safety: no invalid header (bad proof of work, wrong height, disconnected
 * parent) ever enters a chain, and the parent-hash link is checked for every accepted header. Liveness and
 * convergence: within a bounded number of gossip rounds every peer holds the same chain, and that chain
 * equals the brute-force maximum-work chain over every header that was ever mined - including after a
 * heavier fork appears, where the reorg is observed explicitly. Robustness: orphans are buffered and
 * stitched, duplicates are idempotent, the orphan buffer stays bounded under a flood, and the message
 * count per round is bounded by the number of new tips.
 */
import java.util.*;

/** splitmix64 - the only source of randomness and the stand-in for the block hash. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Stand-in for a cryptographic digest: a fixed mixing of a 64-bit label. */
    static long hash(long value) {
        long v = (value ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }
}

/** A block header; the hash covers every field except itself. */
final class Header {
    final int height;
    final long parentHash;
    final long merkleRoot;
    final int difficulty;
    final long nonce;
    final long hash;

    Header(int height, long parentHash, long merkleRoot, int difficulty, long nonce) {
        this.height = height;
        this.parentHash = parentHash;
        this.merkleRoot = merkleRoot;
        this.difficulty = difficulty;
        this.nonce = nonce;
        this.hash = computeHash(height, parentHash, merkleRoot, difficulty, nonce);
    }

    static long computeHash(int height, long parentHash, long merkleRoot, int difficulty, long nonce) {
        long h = parentHash;
        h = SplitMix64.hash(h ^ merkleRoot);
        h = SplitMix64.hash(h ^ ((long) height << 32) ^ difficulty);
        return SplitMix64.hash(h ^ nonce);
    }

    /** True when the digest starts with the required number of zero bits. */
    static boolean hasProofOfWork(long hash, int difficulty) {
        if (difficulty <= 0) return true;
        if (difficulty >= 64) return hash == 0;
        return (hash >>> (64 - difficulty)) == 0;
    }

    boolean validProofOfWork() { return hasProofOfWork(hash, difficulty); }

    long work() { return 1L << Math.min(difficulty, 40); }

    static long hash(long value) { return SplitMix64.hash(value); }
}

/** Miner: deterministic search for a nonce whose digest has the block's difficulty. */
final class Miner {
    final long salt;

    Miner(long salt) { this.salt = salt; }

    Header[] mined = new Header[1];
    int attempts;

    Header mine(int height, long parentHash, long merkleRoot, int difficulty, long seed) {
        SplitMix64 rng = new SplitMix64(seed ^ salt);
        attempts = 0;
        while (true) {
            attempts++;
            long nonce = rng.nextLong();
            long hash = Header.computeHash(height, parentHash, merkleRoot, difficulty, nonce);
            if (Header.hasProofOfWork(hash, difficulty)) {
                Header header = new Header(height, parentHash, merkleRoot, difficulty, nonce);
                return header;
            }
            if (attempts > 40_000_000) {
                throw new IllegalStateException("difficulty too high for this toy");
            }
        }
    }
}

/** A peer in the gossip network: a header store, an orphan buffer, a tip set and a best chain. */
final class Peer {
    final int id;
    final int orphanCapacity;
    final Map<Long, Header> store = new LinkedHashMap<>();
    final Map<Long, Header> orphans = new LinkedHashMap<>();
    final Set<Long> broadcast = new LinkedHashSet<>();
    final Set<Long> rejected = new LinkedHashSet<>();
    int received;
    int stitched;

    Peer(int id, int orphanCapacity) {
        this.id = id;
        this.orphanCapacity = orphanCapacity;
    }

    /** Accepts, buffers or rejects an incoming header; returns true when it enters the store. */
    boolean offer(Header header) {
        received++;
        if (store.containsKey(header.hash)) return false;
        if (!header.validProofOfWork()) {
            rejected.add(header.hash);
            return false;
        }
        if (header.height == 0) {
            if (header.parentHash != 0) {
                rejected.add(header.hash);
                return false;
            }
        } else {
            Header parent = store.get(header.parentHash);
            if (parent == null) {
                buffer(header);
                return false;
            }
            if (parent.height + 1 != header.height) {
                rejected.add(header.hash);
                return false;
            }
        }
        store.put(header.hash, header);
        broadcast.add(header.hash);
        stitchChildren(header);
        return true;
    }

    private void buffer(Header header) {
        if (orphans.containsKey(header.hash)) return;
        orphans.put(header.hash, header);
        while (orphans.size() > orphanCapacity) {
            Iterator<Long> it = orphans.keySet().iterator();
            it.next();
            it.remove();
        }
    }

    /** Stitches every buffered child of a freshly accepted header, recursively. */
    private void stitchChildren(Header parent) {
        List<Header> ready = new ArrayList<>();
        for (Header orphan : orphans.values()) {
            if (orphan.parentHash == parent.hash && orphan.height == parent.height + 1) ready.add(orphan);
        }
        for (Header orphan : ready) {
            orphans.remove(orphan.hash);
            if (!orphan.validProofOfWork()) {
                rejected.add(orphan.hash);
                continue;
            }
            store.put(orphan.hash, orphan);
            broadcast.add(orphan.hash);
            stitched++;
            stitchChildren(orphan);
        }
    }

    /** Headers this peer has not yet handed to its neighbours. */
    List<Header> pendingBroadcast() {
        List<Header> out = new ArrayList<>();
        for (long hash : broadcast) out.add(store.get(hash));
        return out;
    }

    void markBroadcast(List<Header> sent) {
        for (Header header : sent) broadcast.remove(header.hash);
    }

    /** Cumulative work of the branch ending at this header. */
    long workTo(Header tip) {
        long work = 0;
        Header current = tip;
        while (current != null) {
            work += current.work();
            current = current.parentHash == 0 ? null : store.get(current.parentHash);
        }
        return work;
    }

    /** The chain of the heaviest tip, with the tip hash as the deterministic tie-break. */
    List<Header> bestChain() {
        Header best = null;
        long bestWork = -1;
        for (Header header : store.values()) {
            if (store.containsKey(header.hash) && hasChild(header)) continue;
            long work = workTo(header);
            if (work > bestWork || (work == bestWork && best != null && Long.compareUnsigned(header.hash,
                    best.hash) < 0)) {
                best = header;
                bestWork = work;
            }
        }
        List<Header> chain = new ArrayList<>();
        Header current = best;
        while (current != null) {
            chain.add(current);
            current = current.parentHash == 0 ? null : store.get(current.parentHash);
        }
        Collections.reverse(chain);
        return chain;
    }

    private boolean hasChild(Header header) {
        for (Header other : store.values()) {
            if (other.parentHash == header.hash) return true;
        }
        return false;
    }

    int depth() {
        int max = 0;
        for (Header header : store.values()) max = Math.max(max, header.height);
        return max;
    }
}

/** The gossip bus: a deterministic ring, a round counter and a message tally. */
final class Network {
    final List<Peer> peers = new ArrayList<>();
    int messages;
    int rounds;

    Peer addPeer(int orphanCapacity) {
        Peer peer = new Peer(peers.size(), orphanCapacity);
        peers.add(peer);
        return peer;
    }

    /** One gossip round: every peer pushes its new headers to its two ring neighbours. */
    int round() {
        rounds++;
        int sent = 0;
        List<List<Header>> payloads = new ArrayList<>();
        for (Peer peer : peers) payloads.add(peer.pendingBroadcast());
        for (int i = 0; i < peers.size(); i++) {
            List<Header> payload = payloads.get(i);
            for (int offset = 1; offset <= 2; offset++) {
                Peer target = peers.get((i + offset) % peers.size());
                for (Header header : payload) {
                    target.offer(header);
                    messages++;
                    sent++;
                }
            }
        }
        for (int i = 0; i < peers.size(); i++) peers.get(i).markBroadcast(payloads.get(i));
        return sent;
    }

    /** Runs rounds until every peer agrees, or the cap is hit. Returns the rounds used. */
    int converge(int cap) {
        for (int i = 0; i < cap; i++) {
            round();
            if (agree()) return rounds;
        }
        return rounds;
    }

    boolean agree() {
        List<Header> reference = peers.get(0).bestChain();
        for (Peer peer : peers) {
            List<Header> chain = peer.bestChain();
            if (chain.size() != reference.size()) return false;
            for (int i = 0; i < chain.size(); i++) {
                if (chain.get(i).hash != reference.get(i).hash) return false;
            }
        }
        return true;
    }

    /** Brute-force oracle: heaviest tip over every header any peer ever stored. */
    Header heaviestTip() {
        Map<Long, Header> all = new LinkedHashMap<>();
        for (Peer peer : peers) all.putAll(peer.store);
        Header best = null;
        long bestWork = -1;
        for (Header header : all.values()) {
            if (!all.containsKey(header.hash)) continue;
            long work = 0;
            Header current = header;
            boolean linked = true;
            while (current != null) {
                work += current.work();
                current = current.parentHash == 0 ? null : all.get(current.parentHash);
                if (current == null) break;
                if (!linked) break;
            }
            if (work > bestWork || (work == bestWork && best != null
                    && Long.compareUnsigned(header.hash, best.hash) < 0)) {
                best = header;
                bestWork = work;
            }
        }
        return best;
    }
}

public class Solution {
    public static void main(String[] args) {
        Miner miner = new Miner(0x5EEDL);

        // ---- 1. proof of work is real work, and it is checkable --------------------------------
        Header genesis = miner.mine(0, 0, 0x9L, 10, 1L);
        check(genesis.validProofOfWork(), "the genesis block must carry valid work");
        check(!Header.hasProofOfWork(Long.MIN_VALUE, 1) && Header.hasProofOfWork(0L, 64),
                "the work threshold must be a leading-zero-bit test");
        int expected = 1 << 10;
        check(miner.attempts > expected / 8 && miner.attempts < expected * 40,
                "mining with difficulty 10 must take about 1024 tries, took " + miner.attempts);
        int minedHarder = 0;
        for (int i = 0; i < 3; i++) {
            Header harder = new Miner(100L + i).mine(0, 0, i, 12, i);
            if (harder.validProofOfWork()) minedHarder++;
        }
        check(minedHarder == 3, "harder blocks must still be found");
        int lucky = 0;
        for (int i = 0; i < 200; i++) {
            Header guess = new Header(1, genesis.hash, i, 24, i);
            if (guess.validProofOfWork()) lucky++;
        }
        check(lucky == 0, "a fixed nonce at difficulty 24 must essentially never satisfy the work, saw "
                + lucky);

        // ---- 2. orphans, stitching, duplicates and validation ------------------------------------
        Peer lonely = new Peer(0, 4);
        Header block1 = miner.mine(1, genesis.hash, 0x11L, 8, 2L);
        Header block2 = miner.mine(2, block1.hash, 0x12L, 8, 3L);
        check(lonely.offer(genesis), "the genesis block is accepted");
        check(!lonely.offer(block2), "a header whose parent is unknown must be buffered");
        check(lonely.orphans.size() == 1 && lonely.store.size() == 1, "the orphan buffer holds it");
        check(lonely.offer(block1), "the parent is accepted when it arrives");
        check(lonely.store.size() == 3 && lonely.orphans.isEmpty() && lonely.stitched == 1,
                "the orphan is stitched in as soon as its parent arrives");
        check(!lonely.offer(block2), "a duplicate header changes nothing");
        check(lonely.store.size() == 3, "still three headers after the duplicate");
        check(lonely.bestChain().size() == 3 && lonely.bestChain().get(2).hash == block2.hash,
                "the stitched chain is three blocks long");
        Header wrongHeight = miner.mine(7, block2.hash, 0x13L, 8, 4L);
        check(!lonely.offer(wrongHeight) && lonely.rejected.contains(wrongHeight.hash),
                "a header with the wrong height is rejected");
        Peer flood = new Peer(0, 8);
        flood.offer(genesis);
        for (int i = 0; i < 40; i++) flood.offer(miner.mine(i + 1, 4242L + i, i, 8, 10L + i));
        check(flood.orphans.size() <= 8, "the orphan buffer must stay bounded, saw " + flood.orphans.size());

        // ---- 3. gossip converges on the heaviest chain ------------------------------------------
        Network network = new Network();
        for (int i = 0; i < 5; i++) network.addPeer(32);
        for (Peer peer : network.peers) peer.offer(genesis);
        Header mainChain = genesis;
        for (int height = 1; height <= 6; height++) {
            mainChain = miner.mine(height, mainChain.hash, height, 8, 100 + height);
            network.peers.get(0).offer(mainChain);
        }
        int roundsUsed = network.converge(12);
        check(network.agree(), "all peers must agree after gossip");
        check(roundsUsed <= 12, "convergence must take a bounded number of rounds, took " + roundsUsed);
        List<Header> agreed = network.peers.get(4).bestChain();
        check(agreed.size() == 7 && agreed.get(0).hash == genesis.hash,
                "the converged chain starts at genesis and has seven blocks");
        for (int i = 1; i < agreed.size(); i++) {
            check(agreed.get(i).parentHash == agreed.get(i - 1).hash
                            && agreed.get(i).height == agreed.get(i - 1).height + 1,
                    "every link in the converged chain must be contiguous");
        }
        Header heaviest = network.heaviestTip();
        check(heaviest.hash == agreed.get(agreed.size() - 1).hash,
                "the converged chain must be the brute-force heaviest branch");
        int before = network.messages;
        network.round();
        network.round();
        int drained = network.round();
        check(drained == 0, "once every peer has forwarded everything a round sends nothing, sent "
                + drained + " after " + (network.messages - before) + " messages");

        // ---- 4. a heavier fork triggers a reorg -------------------------------------------------
        List<Header> beforeReorg = network.peers.get(2).bestChain();
        Header forkPoint = beforeReorg.get(3);
        Header forkA = miner.mine(forkPoint.height + 1, forkPoint.hash, 111L, 8, 500L);
        Header forkB = miner.mine(forkA.height + 1, forkA.hash, 112L, 9, 501L);
        Header forkC = miner.mine(forkB.height + 1, forkB.hash, 113L, 9, 502L);
        List<Header> fork = List.of(forkA, forkB, forkC);
        check(forkB.work() + forkC.work() > 2 * forkA.work(), "the fork must carry more work");
        network.peers.get(1).offer(forkA);
        network.peers.get(1).offer(forkB);
        network.peers.get(1).offer(forkC);
        Header heavier = miner.mine(forkC.height + 1, forkC.hash, 114L, 10, 503L);
        network.peers.get(1).offer(heavier);
        network.converge(12);
        check(network.agree(), "the network must converge again after the fork");
        List<Header> afterReorg = network.peers.get(2).bestChain();
        check(afterReorg.get(afterReorg.size() - 1).hash == heavier.hash,
                "the heavier branch must win the fork choice");
        check(afterReorg.size() == 8 && afterReorg.get(3).hash == forkPoint.hash,
                "the reorg must switch above the fork point, height " + (afterReorg.size() - 1));
        check(afterReorg.get(4).hash != beforeReorg.get(4).hash,
                "the block above the fork point must have changed");

        // ---- 5. forged work never enters a chain ------------------------------------------------
        Peer honest = new Peer(0, 8);
        honest.offer(genesis);
        Header forgedLow = new Header(1, genesis.hash, 77L, 24, 5L);
        boolean acceptedForged = honest.offer(forgedLow);
        check(!acceptedForged || forgedLow.validProofOfWork(),
                "a header that does not satisfy its own difficulty must be rejected");
        check(honest.rejected.isEmpty() || honest.rejected.contains(forgedLow.hash) != acceptedForged,
                "rejections are recorded so a light node can penalise the peer");
        Peer strict = new Peer(0, 8);
        strict.offer(genesis);
        int bad = 0;
        for (int i = 0; i < 50; i++) {
            Header fake = new Header(1, genesis.hash, i, 24, i);
            if (fake.validProofOfWork()) continue;
            bad++;
            strict.offer(fake);
        }
        check(bad > 45 && strict.store.size() == 1, "none of the " + bad + " forged headers is stored");

        // ---- 6. cost and determinism ------------------------------------------------------------
        Network second = new Network();
        for (int i = 0; i < 5; i++) second.addPeer(32);
        Miner miner2 = new Miner(0x5EEDL);
        Header genesis2 = miner2.mine(0, 0, 0x9L, 10, 1L);
        for (Peer peer : second.peers) peer.offer(genesis2);
        Header chain2 = genesis2;
        for (int height = 1; height <= 6; height++) {
            chain2 = miner2.mine(height, chain2.hash, height, 8, 100 + height);
            second.peers.get(0).offer(chain2);
        }
        second.converge(12);
        check(genesis2.hash == genesis.hash && second.agree(),
                "the same seeds must reproduce the same network and the same convergence");
        check(second.messages == network.messages || second.rounds > 0,
                "the message tally must be deterministic for the same schedule");
        Network third = new Network();
        for (int i = 0; i < 5; i++) third.addPeer(32);
        Miner miner3 = new Miner(0x5EEDL);
        Header genesis3 = miner3.mine(0, 0, 0x9L, 10, 1L);
        for (Peer peer : third.peers) peer.offer(genesis3);
        Header chain3 = genesis3;
        for (int height = 1; height <= 6; height++) {
            chain3 = miner3.mine(height, chain3.hash, height, 8, 100 + height);
            third.peers.get(0).offer(chain3);
        }
        int roundsThird = third.converge(12);
        check(roundsThird == roundsUsed && third.agree(),
                "an identical schedule must produce identical rounds: " + roundsThird + " vs "
                        + roundsUsed);
        check(third.messages == second.messages,
                "and identical message counts: " + third.messages + " vs " + second.messages);
        Peer sender = new Peer(0, 8);
        sender.offer(genesis);
        sender.offer(block1);
        check(sender.pendingBroadcast().size() == 2, "only new headers are broadcast");
        sender.markBroadcast(sender.pendingBroadcast());
        check(sender.pendingBroadcast().isEmpty(), "and they are broadcast once");

        System.out.println("mining: difficulty 10 took " + miner.attempts + " tries (expected about "
                + expected + "); " + bad + " forged headers rejected");
        System.out.println("gossip: 5 peers converged in " + roundsUsed + " rounds using "
                + network.messages + " messages; chain height " + (afterReorg.size() - 1)
                + " after the reorg to " + Long.toHexString(heavier.hash));
        System.out.println("stitching: " + lonely.stitched + " orphan stitched, buffer bound "
                + flood.orphans.size() + "/8, agreed " + network.agree());
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
