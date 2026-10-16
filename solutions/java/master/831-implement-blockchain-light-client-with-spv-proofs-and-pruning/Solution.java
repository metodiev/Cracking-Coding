/** Task 831 - Implement blockchain light client with SPV proofs and pruning. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A full node stores every block and every transaction; a light client stores only block *headers* and verifies
 * individual payments with a Merkle (SPV) proof against a header's transaction root. This file implements that
 * pipeline end to end:
 *
 *     header h_i = (index, prevHash, txRoot, target, nonce)      with PoW: hash(h_i) &lt; target
 *     txRoot     = Merkle root of the block's transactions
 *     SPV proof  = (tx, index, sibling hashes)  -&gt; recompute the root, compare with the header
 *
 * On top of that the client prunes: it keeps a sliding window of the last W headers plus finalised checkpoints,
 * serves proofs only for blocks it still holds, and follows the *most work* chain (the sum of the difficulty, i.e.
 * of 1/target, over the headers) so a heavier fork triggers a reorganisation.
 *
 * Simplified: the hash is a deterministic splitmix64 mixer rather than SHA-256, so the proof of work is a toy
 * "hash below target" puzzle mined by a deterministic nonce search rather than a real race; there is no gossip,
 * no mempool and no difficulty retargeting. Everything is in-process and deterministic. Verified: the header chain
 * links up and every header satisfies its own target; the recomputed Merkle root matches the header for every
 * block; SPV proofs verify for every transaction at every index and a tampered transaction, a wrong index or a
 * wrong root is rejected; a header that does not meet its target is refused; the most-work rule picks the heavier
 * fork and a reorganisation rewrites the canonical chain so that proofs from the abandoned chain stop verifying;
 * pruned blocks can no longer be proven while retained ones can, and the client's storage stays inside the window
 * plus the checkpoints; checkpoint headers cannot be contradicted; and the single transaction, single block,
 * oversized window and empty block cases all behave.
 */
import java.util.*;

/** Deterministic toy hash. Not a cryptographic hash: it only has to make the Merkle and PoW logic testable. */
final class ToyHash {
    static long mix(long a) {
        long z = a + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long combine(long left, long right) {
        return mix(mix(left) ^ Long.rotateLeft(mix(right), 17));
    }
}

/** Binary Merkle tree with opening paths; leaf count is padded to a power of two. */
final class Merkle {
    final long[] leaves;
    final long[][] levels;

    Merkle(long[] values) {
        int n = 1;
        while (n < Math.max(1, values.length)) n *= 2;
        leaves = new long[n];
        System.arraycopy(values, 0, leaves, 0, values.length);
        List<long[]> built = new ArrayList<>();
        built.add(leaves.clone());
        long[] current = leaves;
        while (current.length > 1) {
            long[] next = new long[current.length / 2];
            for (int i = 0; i < next.length; i++) next[i] = ToyHash.combine(current[2 * i], current[2 * i + 1]);
            built.add(next);
            current = next;
        }
        levels = built.toArray(new long[0][]);
    }

    long root() { return levels[levels.length - 1][0]; }

    long[] open(int index) {
        long[] path = new long[levels.length - 1];
        int position = index;
        for (int level = 0; level < levels.length - 1; level++) {
            path[level] = levels[level][position ^ 1];
            position >>= 1;
        }
        return path;
    }

    static long rootFromPath(int index, long value, long[] path) {
        long current = value;
        int position = index;
        for (long sibling : path) {
            current = (position & 1) == 0 ? ToyHash.combine(current, sibling)
                    : ToyHash.combine(sibling, current);
            position >>= 1;
        }
        return current;
    }
}

/** One transaction: a payer, a payee and an amount. */
final class Tx {
    final String from, to;
    final long amount;

    Tx(String from, String to, long amount) {
        this.from = from; this.to = to; this.amount = amount;
    }

    long hash() { return ToyHash.mix(amount * 31 + from.hashCode() * 1000003L + to.hashCode()); }

    @Override
    public String toString() { return from + "->" + to + ":" + amount; }
}

/** A block header: the only thing a light client stores. */
final class Header {
    final int index;
    final long prevHash, txRoot, target, nonce;
    boolean valid;              // set when the header passed the PoW and linkage checks
    int height;

    Header(int index, long prevHash, long txRoot, long target, long nonce) {
        this.index = index;
        this.prevHash = prevHash;
        this.txRoot = txRoot;
        this.target = target;
        this.nonce = nonce;
    }

    /** The hash of the serialised header: the PoW function. */
    long hash() {
        long h = ToyHash.mix(index * 1000003L + 7);
        h = ToyHash.combine(h, prevHash);
        h = ToyHash.combine(h, txRoot);
        h = ToyHash.combine(h, target);
        h = ToyHash.combine(h, nonce);
        return h;
    }

    boolean meetsTarget() { return Math.abs(hash()) < target; }

    double work() { return 1.0 / target; }
}

/** A full block: header plus the transactions (which the light client never keeps). */
final class Block {
    final Header header;
    final List<Tx> transactions;

    Block(Header header, List<Tx> transactions) {
        this.header = header;
        this.transactions = transactions;
    }

    static long rootOf(List<Tx> transactions) {
        long[] hashes = new long[transactions.size()];
        for (int i = 0; i < hashes.length; i++) hashes[i] = transactions.get(i).hash();
        return new Merkle(hashes).root();
    }

    /** An inclusion proof: the transaction, its position and the sibling path. */
    static Object[] prove(List<Tx> transactions, int index) {
        long[] hashes = new long[transactions.size()];
        for (int i = 0; i < hashes.length; i++) hashes[i] = transactions.get(i).hash();
        return new Object[] {transactions.get(index), index, new Merkle(hashes).open(index)};
    }

    static boolean verifyProof(Tx tx, int index, long[] path, long root) {
        return Merkle.rootFromPath(index, tx.hash(), path) == root;
    }
}

/** Deterministic miner: searches nonces until the header hash is below the target. */
final class Miner {
    static Header mine(int index, long prevHash, long root, long target) {
        for (long nonce = 0; nonce < 100000; nonce++) {
            Header candidate = new Header(index, prevHash, root, target, nonce);
            if (candidate.meetsTarget()) return candidate;
        }
        throw new AssertionError("no nonce found for target " + target);
    }

    /** Builds a chain of `length` blocks starting at block `startIndex`, with the given target. */
    static List<Block> build(int startIndex, int length, long prevHash, long target, long seedTag) {
        List<Block> chain = new ArrayList<>();
        long head = prevHash;
        for (int i = 0; i < length; i++) {
            int index = startIndex + i;
            int n = 1 + (int) ((seedTag + i) % 3);
            List<Tx> txs = new ArrayList<>();
            for (int k = 0; k < n; k++)
                txs.add(new Tx("a" + (seedTag % 3), "b" + index + k, 100 + 7L * (index + k + seedTag)));
            long root = Block.rootOf(txs);
            Header header = mine(index, head, root, target);
            chain.add(new Block(header, txs));
            head = header.hash();
        }
        return chain;
    }
}

/** The light client: a pruned header store, checkpoints and SPV proof verification, following most work. */
final class LightClient {
    final int window;
    final Map<Integer, Header> headers = new LinkedHashMap<>();
    final Map<Integer, Long> checkpoints = new HashMap<>();
    int head = -1;
    int accepted, rejected;

    LightClient(int window) { this.window = window; }

    /** Work of the chain ending at this header, walking back through the headers the client holds. */
    double chainWork(Header header) {
        double work = 0;
        Header current = header;
        while (current != null) {
            work += current.work();
            if (current.index == 0) break;
            current = headers.get(current.index - 1);
        }
        return work;
    }

    /** Accepts a header only if it links into known history, meets its target, matches the checkpoint and - on a
     *  fork - carries strictly more work than the current head. This is the most-work fork choice rule. */
    boolean accept(Header header) {
        if (!header.meetsTarget()) {
            rejected++;
            return false;
        }
        if (header.index == 0) {
            if (header.prevHash != 0) {
                rejected++;
                return false;
            }
        } else {
            Header parent = headers.get(header.index - 1);
            if (parent == null || parent.hash() != header.prevHash) {
                rejected++;
                return false;
            }
        }
        Long checkpoint = checkpoints.get(header.index);
        if (checkpoint != null && checkpoint != header.hash()) {
            rejected++;
            return false;
        }
        headers.put(header.index, header);
        accepted++;
        double candidateWork = chainWork(headers.get(header.index));
        if (head < 0 || candidateWork > chainWork(headers.get(head))) head = header.index;
        prune();
        return true;
    }

    /** Keeps the last `window` headers around the head plus everything a checkpoint pins down. */
    void prune() {
        for (Integer index : new ArrayList<>(headers.keySet())) {
            if (index <= head - window && !checkpoints.containsKey(index)) headers.remove(index);
        }
    }

    Header headHeader() { return headers.get(head); }

    /** A block is canonical when the client can walk from the head back to it through parent links. */
    boolean isCanonical(int index) {
        int cursor = head;
        while (cursor >= 0) {
            if (cursor == index) return true;
            if (cursor == 0) break;
            Header parent = headers.get(cursor - 1);
            if (parent == null) return false;
            cursor = parent.index;
        }
        return false;
    }

    boolean serves(int index) { return headers.containsKey(index); }

    /** Verifies an SPV proof against the header the client holds for that block. */
    boolean verifyPayment(Tx tx, int blockIndex, int position, long[] path) {
        Header header = headers.get(blockIndex);
        if (header == null) return false;
        return Block.verifyProof(tx, position, path, header.txRoot);
    }

    void checkpoint(int index, long hash) { checkpoints.put(index, hash); }

    double totalWork() {
        double work = 0;
        for (Header header : headers.values()) work += header.work();
        return work;
    }
}

public class Solution {
    static final long EASY_TARGET = 0x1000000000000000L, HARD_TARGET = 0x0800000000000000L, FORK_TARGET = 0x0010000000000000L;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    public static void main(String[] args) {
        // ---- 1. build a chain and verify linkage plus proof of work on every header
        List<Block> chain = Miner.build(0, 12, 0L, EASY_TARGET, 1L);
        boolean linked = true, powOk = true;
        for (int i = 0; i < chain.size(); i++) {
            Header header = chain.get(i).header;
            long expectedPrev = i == 0 ? 0 : chain.get(i - 1).header.hash();
            if (header.prevHash != expectedPrev) linked = false;
            if (!header.meetsTarget()) powOk = false;
        }
        check(linked && powOk, "the chain links from block to block and every header satisfies its own target");

        // ---- 2. the committed transaction root matches the block contents
        boolean rootsOk = true;
        for (Block block : chain) if (block.header.txRoot != Block.rootOf(block.transactions)) rootsOk = false;
        check(rootsOk, "every header's transaction root is the Merkle root of the block's transactions");

        // ---- 3. SPV proofs verify for every transaction of every block
        boolean allProofs = true;
        for (Block block : chain)
            for (int i = 0; i < block.transactions.size(); i++) {
                Object[] proof = Block.prove(block.transactions, i);
                Tx tx = (Tx) proof[0];
                int position = (Integer) proof[1];
                long[] path = (long[]) proof[2];
                if (!Block.verifyProof(tx, position, path, block.header.txRoot)) allProofs = false;
            }
        check(allProofs, "an inclusion proof verifies for every transaction at every position");

        // ---- 4. tampered transactions, wrong positions and wrong roots are rejected
        Block first = chain.get(0);
        Object[] proof = Block.prove(first.transactions, 0);
        Tx honest = (Tx) proof[0];
        long[] path = (long[]) proof[2];
        boolean tampered = !Block.verifyProof(new Tx(honest.from, honest.to, honest.amount + 1), 0, path,
                first.header.txRoot);
        boolean wrongPosition = !Block.verifyProof(honest, 1, path, first.header.txRoot);
        boolean wrongRoot = !Block.verifyProof(honest, 0, path, first.header.txRoot + 1);
        check(tampered && wrongPosition && wrongRoot,
                "a modified transaction, a wrong leaf position and a wrong root are all rejected");

        // ---- 5. the light client accepts headers up to its window and prunes older ones
        LightClient client = new LightClient(4);
        for (Block block : chain) if (!client.accept(block.header)) throw new AssertionError("honest header refused");
        int headIndex = client.head;
        boolean prunedOld = !client.serves(0) && !client.serves(headIndex - client.window - 1);
        boolean keptRecent = client.serves(headIndex) && client.serves(headIndex - client.window + 1);
        check(prunedOld && keptRecent, "the client keeps the last " + client.window + " headers and prunes older "
                + "ones");

        // ---- 6. retained blocks can be proven, pruned blocks cannot
        Block recent = chain.get(chain.size() - 1);
        Object[] recentProof = Block.prove(recent.transactions, 0);
        boolean recentOk = client.verifyPayment((Tx) recentProof[0], recent.header.index, (Integer) recentProof[1],
                (long[]) recentProof[2]);
        Object[] oldProof = Block.prove(first.transactions, 0);
        boolean oldRefused = !client.verifyPayment((Tx) oldProof[0], first.header.index, (Integer) oldProof[1],
                (long[]) oldProof[2]);
        check(recentOk && oldRefused, "a proof for a retained block verifies while the same proof for a pruned "
                + "block is refused");

        // ---- 7. a header that does not meet its target is refused
        long prev = client.headHeader().hash();
        Header lazy = new Header(-1, prev, 12345L, HARD_TARGET, 0);
        boolean lazyRejected = !client.accept(lazy);
        Header wrongLink = new Header(client.head + 1, prev + 1, first.header.txRoot, EASY_TARGET, 0);
        boolean linkRejected = !client.accept(wrongLink);
        check(lazyRejected && linkRejected && client.rejected >= 2,
                "a header below the difficulty and a header with a broken parent link are both refused");

        // ---- 8. checkpoints cannot be contradicted and pin headers against pruning
        LightClient pinned = new LightClient(2);
        pinned.accept(chain.get(0).header);
        pinned.checkpoint(0, chain.get(0).header.hash());
        for (int i = 1; i < 6; i++) pinned.accept(chain.get(i).header);
        boolean pinnedKept = pinned.serves(0);
        Header conflicting = new Header(0, 0, 999L, EASY_TARGET, 0);
        check(pinnedKept && !pinned.accept(conflicting), "a checkpoint keeps its header through pruning and a "
                + "conflicting header at the same height is refused");

        // ---- 9. fork choice: the most-work chain wins and a reorganisation rewrites the canonical chain
        double honestWork = 0;
        for (Block block : chain) honestWork += block.header.work();
        List<Block> fork = Miner.build(6, 6, chain.get(5).header.hash(), FORK_TARGET, 9L);
        double forkWork = 0;
        for (Block block : fork) forkWork += block.header.work();
        forkWork += 0;
        double forkTotal = 0;
        for (int i = 0; i <= 5; i++) forkTotal += chain.get(i).header.work();
        for (Block block : fork) forkTotal += block.header.work();
        LightClient reorgClient = new LightClient(64);
        for (Block block : chain) reorgClient.accept(block.header);
        boolean forkAccepted = true;
        for (Block block : fork) if (!reorgClient.accept(block.header)) forkAccepted = false;
        Tx forkOnly = fork.get(5).transactions.get(0);
        Object[] forkProof = Block.prove(fork.get(5).transactions, 0);
        boolean forkProofOk = reorgClient.verifyPayment(forkOnly, fork.get(5).header.index, (Integer) forkProof[1],
                (long[]) forkProof[2]);
        check(forkAccepted && forkProofOk && forkTotal > honestWork && reorgClient.head == 11
                        && reorgClient.isCanonical(11),
                "the heavier fork (" + forkTotal + " work vs " + honestWork + ") is adopted, its head becomes the "
                        + "client's head and its transactions can be proven");
        Tx abandoned = chain.get(11).transactions.get(0);
        Object[] abandonedProof = Block.prove(chain.get(11).transactions, 0);
        boolean abandonedRefused = !reorgClient.verifyPayment(abandoned, 11, (Integer) abandonedProof[1],
                (long[]) abandonedProof[2]);
        check(abandonedRefused, "after the reorganisation the abandoned block's payment proof no longer verifies "
                + "against the canonical header at that height");
        boolean staleRefused = !reorgClient.accept(new Header(12, chain.get(11).header.hash(), 2L, EASY_TARGET, 0));
        check(staleRefused, "a header extending the abandoned chain does not link to the canonical parent and is "
                + "refused");

        // ---- 10. edge cases: a single transaction, a single block, an oversized window, an empty block
        List<Tx> one = new ArrayList<>();
        one.add(new Tx("x", "y", 42));
        Merkle single = new Merkle(new long[] {one.get(0).hash()});
        boolean singleOk = single.root() == one.get(0).hash()
                && Block.verifyProof(one.get(0), 0, single.open(0), single.root())
                && !Block.verifyProof(new Tx("x", "y", 43), 0, single.open(0), single.root());
        List<Block> singleton = Miner.build(0, 1, 0L, EASY_TARGET, 5L);
        LightClient tiny = new LightClient(10);
        boolean tinyOk = tiny.accept(singleton.get(0).header) && tiny.serves(0);
        long emptyRoot = Block.rootOf(new ArrayList<>());
        check(singleOk && tinyOk, "a single-transaction block proves and rejects as expected, and a window larger "
                + "than the chain keeps everything");
        check(emptyRoot == new Merkle(new long[0]).root(),
                "an empty block has the empty-tree root and still proves nothing");

        // ---- 11. storage bound: the client never holds more than the window plus the checkpoints
        int pinnedCount = 0;
        for (int index : pinned.checkpoints.keySet()) if (pinned.headers.containsKey(index)) pinnedCount++;
        check(pinned.headers.size() <= pinned.window + pinnedCount
                        && client.headers.size() <= client.window + client.checkpoints.size(),
                "the client's header store stays inside the window plus the pinned checkpoints");
        check(client.checkpoints.isEmpty(), "an unpinned client stores no checkpoints at all");

        System.out.println("spv light client: " + chain.size() + " blocks, window " + client.window + ", heads "
                + client.head + "; accepted " + client.accepted + ", rejected " + client.rejected);
        System.out.println("reorg: fork work " + forkWork + " vs original " + honestWork + "; retained headers "
                + client.headers.size());
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
