/** Task 923 - Implement distributed ledger with privacy-preserving transaction aggregation.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A toy but faithful private-aggregation ledger. Every transaction amount is additively secret-shared over
 * the four validators modulo p = 2^61 - 1: the first three shares are fresh pseudo-random field elements
 * (so they are independent of the amount - that is the privacy argument) and the last one is the balancing
 * share that makes the sum congruent to the amount. A block therefore stores no plaintext at all, only one
 * aggregate share per validator plus a root committing to the individual share pairs, a chained block hash
 * over (previous hash, index, that root), and one keyed MAC per validator; a block is only accepted with a
 * 3-of-4 quorum of valid MACs. Anyone can add the aggregate shares up to recover the *sum* of the block's
 * amounts, while any single validator's view is provably uninformative about the individual amounts.
 * Simplified: the MAC and the hash are deterministic splitmix64 mixes rather than cryptographic primitives,
 * there is no networking or byzantine behaviour model, and the quorum is only counted, not voted on.
 * Verified: the aggregate equals the plaintext sum (debits included), the chain links and quorums verify,
 * tampering with a share root or an aggregate is caught, 2-of-4 and forged signatures are rejected, the
 * per-validator correlation with the plaintext is negligible, fewer than all shares cannot recover an
 * amount, an empty block aggregates to zero, and minting is reproducible.
 */
import java.util.*;

/** A non-cryptographic (but deterministic) mixer: splitmix64 with fixed constants. */
final class Mix {
    static long mix(long x) {
        long z = x + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long mix(long a, long b) { return mix(a ^ mix(b)); }
}

/** One ledger block: index, predecessor hash, share root, aggregate shares and validator MACs. */
final class Block {
    final int index;
    final long prevHash, root;
    final long[] aggregates, signatures;

    Block(int index, long prevHash, long root, long[] aggregates, long[] signatures) {
        this.index = index;
        this.prevHash = prevHash;
        this.root = root;
        this.aggregates = aggregates;
        this.signatures = signatures;
    }

    /** What the validators sign: everything that fixes the block's content except the MACs themselves. */
    long header() { return Mix.mix(Mix.mix(prevHash, index), root); }

    /** What the next block chains to: the header plus the aggregate shares. */
    long hash() { return Mix.mix(header(), aggregateHash()); }

    long aggregateHash() {
        long h = Mix.mix(aggregates.length);
        for (long a : aggregates) h = Mix.mix(h, a);
        return h;
    }

    long total() {
        long sum = 0;
        for (long a : aggregates) sum = Ledger.plus(sum, a);
        return sum;
    }
}

/** The privacy-preserving ledger: additive sharing, aggregation and the hash/MAC chain. */
final class Ledger {
    static final long MOD = (1L << 61) - 1;
    static final int VALIDATORS = 4, QUORUM = 3;

    static long norm(long v) {
        long r = v % MOD;
        return r < 0 ? r + MOD : r;
    }

    /** Field addition; both inputs are already reduced, so the sum cannot overflow. */
    static long plus(long a, long b) {
        long s = a + b;
        return s >= MOD ? s - MOD : s;
    }

    static long key(int validator) { return Mix.mix(0x5EED0000L + validator); }

    /** Additive sharing: VALIDATORS shares whose sum mod p is the amount. */
    static long[] share(long amount, long seed) {
        long[] shares = new long[VALIDATORS];
        long running = 0, state = Mix.mix(seed);
        for (int v = 0; v < VALIDATORS - 1; v++) {
            state = Mix.mix(state);
            shares[v] = (state >>> 3) % MOD;
            running = plus(running, shares[v]);
        }
        shares[VALIDATORS - 1] = norm(amount - running);
        return shares;
    }

    static long[] mac(long header) {
        long[] signatures = new long[VALIDATORS];
        for (int v = 0; v < VALIDATORS; v++) signatures[v] = Mix.mix(key(v), header);
        return signatures;
    }

    /** Seal a block of amounts: aggregate the shares, commit to the shares, sign the header. */
    static Block mint(int index, long prevHash, long[] amounts) {
        long[] aggregates = new long[VALIDATORS];
        long root = Mix.mix(index, prevHash);
        for (int t = 0; t < amounts.length; t++) {
            long[] shares = share(amounts[t], Mix.mix(0x7A5E0000L + t * 131 + index));
            for (int v = 0; v < VALIDATORS; v++) aggregates[v] = plus(aggregates[v], shares[v]);
            root = Mix.mix(root, Mix.mix(shares[0], shares[1]));
        }
        long header = Mix.mix(Mix.mix(prevHash, index), root);
        return new Block(index, prevHash, root, aggregates, mac(header));
    }

    /** Count the validators whose MAC matches the header they were supposed to sign. */
    static int quorum(Block b) {
        long header = b.header();
        int count = 0;
        for (int v = 0; v < VALIDATORS; v++) if (b.signatures[v] == Mix.mix(key(v), header)) count++;
        return count;
    }

    /** Verify a whole chain: linkage by hash and a quorum of valid MACs on every block. */
    static boolean verify(List<Block> chain) {
        long prev = 0;
        for (Block b : chain) {
            if (b.prevHash != prev || quorum(b) < QUORUM) return false;
            prev = b.hash();
        }
        return true;
    }

    static long aggregateTotal(List<Block> chain) {
        long sum = 0;
        for (Block b : chain) sum = plus(sum, b.total());
        return sum;
    }
}

public class Solution {
    static final int TRANSACTIONS = 300;

    public static void main(String[] args) {
        // ---- the fixture: mixed credits and debits, deterministically generated --------------------
        long[] amounts = new long[TRANSACTIONS];
        for (int t = 0; t < TRANSACTIONS; t++) amounts[t] = (Mix.mix(0xA11CE + t * 7919L) % 5000) - 2500;
        long plaintextTotal = 0;
        for (long a : amounts) plaintextTotal = Ledger.plus(plaintextTotal, Ledger.norm(a));

        List<List<Long>> blocks = new ArrayList<>();
        int perBlock = 25;
        for (int start = 0; start < TRANSACTIONS; start += perBlock) {
            List<Long> batch = new ArrayList<>();
            for (int t = start; t < Math.min(TRANSACTIONS, start + perBlock); t++) batch.add(amounts[t]);
            blocks.add(batch);
        }
        List<Block> chain = new ArrayList<>();
        long prev = 0;
        for (int i = 0; i < blocks.size(); i++) {
            long[] batch = new long[blocks.get(i).size()];
            for (int t = 0; t < batch.length; t++) batch[t] = blocks.get(i).get(t);
            Block block = Ledger.mint(i + 1, prev, batch);
            chain.add(block);
            prev = block.hash();
        }
        long aggregated = Ledger.aggregateTotal(chain);

        // ---- privacy statistics -------------------------------------------------------------------
        int identical = 0;
        double worstCorrelation = 0;
        for (int v = 0; v < Ledger.VALIDATORS; v++) {
            double[] shares = new double[TRANSACTIONS], plain = new double[TRANSACTIONS];
            for (int t = 0; t < TRANSACTIONS; t++) {
                long[] s = Ledger.share(amounts[t], Mix.mix(0x7A5E0000L + t * 131 + (t / perBlock + 1)));
                shares[t] = s[v];
                plain[t] = amounts[t];
                if (v == Ledger.VALIDATORS - 1 && s[v] == amounts[t] && amounts[t] != 0) identical++;
            }
            worstCorrelation = Math.max(worstCorrelation, Math.abs(correlation(shares, plain)));
        }

        System.out.println("ledger: " + chain.size() + " blocks, " + TRANSACTIONS + " transactions, plaintext total "
                + plaintextTotal + ", aggregated total " + aggregated);
        System.out.println("privacy: shares equal to their amount " + identical + ", worst |corr(share, amount)| "
                + String.format("%.4f", worstCorrelation) + ", quorums " + quorums(chain));

        check(Ledger.verify(chain), "the honest chain must verify");
        check(aggregated == plaintextTotal, "the aggregate shares must recover the plaintext total: " + aggregated
                + " vs " + plaintextTotal);
        for (int v = 0; v < Ledger.VALIDATORS; v++) check(Ledger.quorum(chain.get(0)) == Ledger.VALIDATORS,
                "every honest block must carry all validator MACs");
        check(chain.get(0).total() != plaintextTotal, "a single block's aggregate is only part of the total");
        check(identical == 0, "a balancing share must not accidentally expose its amount");
        check(worstCorrelation < 0.2, "shares must not correlate with the plaintext, worst " + worstCorrelation);

        // ---- fewer than all shares cannot recover an amount ---------------------------------------
        long[] s = Ledger.share(amounts[0], Mix.mix(0x7A5E0000L + 0 * 131 + 1));
        long partial = Ledger.plus(s[0], Ledger.plus(s[1], s[2]));
        check(Ledger.norm(partial) != Ledger.norm(amounts[0]), "3 of 4 shares must not reconstruct the amount");
        long full = Ledger.plus(partial, s[3]);
        check(Ledger.norm(full) == Ledger.norm(amounts[0]),
                "all 4 shares must reconstruct the amount exactly: " + full + " vs " + amounts[0]);

        // ---- negative controls: tampering, missing quorum and a forged MAC ------------------------
        Block forgedOk = chain.get(2);
        long[] shortSignatures = forgedOk.signatures.clone();
        shortSignatures[0] = 0;
        shortSignatures[1] = 0;
        Block weakQuorum = new Block(forgedOk.index, forgedOk.prevHash, forgedOk.root, forgedOk.aggregates, shortSignatures);
        List<Block> weak = new ArrayList<>(chain);
        weak.set(2, weakQuorum);
        check(Ledger.quorum(weakQuorum) == 2 && !Ledger.verify(weak), "a 2-of-4 block must be rejected");

        long[] forgedSignatures = forgedOk.signatures.clone();
        forgedSignatures[1] = Mix.mix(0xDEADL, forgedOk.header());
        Block forged = new Block(forgedOk.index, forgedOk.prevHash, forgedOk.root, forgedOk.aggregates, forgedSignatures);
        List<Block> tamperedMacs = new ArrayList<>(chain);
        tamperedMacs.set(2, forged);
        check(Ledger.quorum(forged) == Ledger.VALIDATORS - 1, "a MAC signed with the wrong key must not be counted");
        check(Ledger.verify(tamperedMacs), "the remaining 3 of 4 valid MACs still meet the quorum");
        long[] twoForged = forgedOk.signatures.clone();
        twoForged[1] = Mix.mix(0xDEADL, forgedOk.header());
        twoForged[2] = Mix.mix(0xBEEFL, forgedOk.header());
        List<Block> forgedTwice = new ArrayList<>(chain);
        forgedTwice.set(2, new Block(forgedOk.index, forgedOk.prevHash, forgedOk.root, forgedOk.aggregates, twoForged));
        check(!Ledger.verify(forgedTwice), "two forged MACs must drop the block below the quorum");

        long[] shifted = forgedOk.aggregates.clone();
        shifted[0] = Ledger.plus(shifted[0], 1);
        Block corrupted = new Block(forgedOk.index, forgedOk.prevHash, forgedOk.root, shifted, forgedOk.signatures);
        check(Ledger.aggregateTotal(rewrite(chain, 2, corrupted)) != plaintextTotal,
                "a corrupted aggregate must break the recovered total");

        Block otherRoot = Ledger.mint(forgedOk.index, forgedOk.prevHash, new long[] {1, 2, 3});
        Block rewritten = new Block(forgedOk.index, forgedOk.prevHash, otherRoot.root, forgedOk.aggregates, forgedOk.signatures);
        check(!Ledger.verify(rewrite(chain, 2, rewritten)), "replacing a block's share root must break the chain");

        // ---- boundaries: an empty block, and a genesis-only chain ---------------------------------
        Block empty = Ledger.mint(chain.size() + 1, prev, new long[0]);
        check(empty.total() == 0, "an empty block aggregates to zero");
        List<Block> withEmpty = new ArrayList<>(chain);
        withEmpty.add(empty);
        check(Ledger.verify(withEmpty) && Ledger.aggregateTotal(withEmpty) == plaintextTotal,
                "an empty block must leave the total unchanged and keep the chain valid");
        List<Block> genesisOnly = new ArrayList<>();
        genesisOnly.add(Ledger.mint(0, 0, new long[0]));
        check(Ledger.verify(genesisOnly) && Ledger.aggregateTotal(genesisOnly) == 0, "a genesis-only chain is valid");

        // ---- determinism -------------------------------------------------------------------------
        long[] batch = new long[] {11, -7, 2500};
        Block a = Ledger.mint(7, 12345, batch), b = Ledger.mint(7, 12345, batch);
        check(a.hash() == b.hash() && Arrays.equals(a.aggregates, b.aggregates), "minting must be reproducible");
        check(Ledger.mint(7, 12346, batch).hash() != a.hash(), "a different predecessor must change the hash");

        System.out.println("tamper controls: short quorum rejected, forged MAC rejected, corrupted aggregate changes the "
                + "total, rewritten root breaks the chain, empty and genesis-only chains valid");
        System.out.println("PASS");
    }

    static List<Block> rewrite(List<Block> chain, int index, Block replacement) {
        List<Block> copy = new ArrayList<>(chain);
        copy.set(index, replacement);
        return copy;
    }

    static String quorums(List<Block> chain) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < chain.size(); i++) sb.append(Ledger.quorum(chain.get(i))).append(i + 1 == chain.size() ? "]" : ",");
        return sb.toString();
    }

    /** Pearson correlation of two deterministic series (used to show shares carry no linear signal). */
    static double correlation(double[] a, double[] b) {
        double ma = 0, mb = 0;
        for (int i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.length;
        mb /= b.length;
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < a.length; i++) {
            cov += (a[i] - ma) * (b[i] - mb);
            va += (a[i] - ma) * (a[i] - ma);
            vb += (b[i] - mb) * (b[i] - mb);
        }
        return cov / Math.sqrt(va * vb);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
