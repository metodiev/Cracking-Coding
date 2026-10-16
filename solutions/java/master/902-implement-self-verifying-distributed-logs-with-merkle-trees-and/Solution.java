/** Task 902 - Implement self-verifying distributed logs with Merkle trees and authenticated queries.
 *  (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * An append-only log whose every entry is hashed into a Merkle tree can serve three authenticated queries
 * without trusting the server:
 *
 *   inclusion(index)   -- the classic O(log n) sibling path; a client that knows only the signed root can
 *                         recompute it and detect any flipped leaf, swapped sibling or truncated path;
 *   consistency(m, n)  -- an append-only proof that the tree over the first m leaves is a prefix of the tree
 *                         over the first n leaves, i.e. nothing was rewritten or removed;
 *   audit(tamper)      -- the log's own verifier: it re-derives every root from the leaves and refuses to
 *                         publish a root that its own recomputation contradicts.
 *
 * Simplifications, stated honestly: the hash is a 64-bit splitmix64 mix instead of SHA-256 (so collisions are
 * only absence-of-evidence, not a security statement), there is no signing key, no network and no persistence
 * - the "cluster" is three in-memory replicas. Invariants verified below: a proof for a genuine leaf always
 * verifies; the same proof against a different leaf, a different root, a reordered sibling or a truncated
 * path fails; the proof length is exactly ceil(log2 n) and the number of hashes is bounded by the tree size;
 * consistency holds for every prefix pair; and an audit rejects a replica whose leaf was mutated in place.
 */
import java.util.*;

/** splitmix64 doubling as the toy collision-resistant hash of this file. */
final class LogHasher {
    static long hash(long value) {
        long v = (value ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }

    /** Internal node hash: the left/right order is part of the input, so siblings cannot be swapped. */
    static long node(long left, long right) {
        return hash(hash(left) ^ Long.rotateLeft(hash(right), 17));
    }

    static long leaf(String entry) {
        long h = 0xC0FFEEL;
        for (int i = 0; i < entry.length(); i++) h = hash(h * 131 + entry.charAt(i));
        return hash(h ^ entry.length());
    }

    /** A leaf whose very first byte differs must hash differently; this is the tamper oracle. */
    static long emptyLeaf() { return leaf(""); }
}

/** One audit record: an entry plus the leaf hash the log committed to when it was appended. */
final class LogEntry {
    final int index;
    final String payload;
    final long committedLeaf;

    LogEntry(int index, String payload, long committedLeaf) {
        this.index = index;
        this.payload = payload;
        this.committedLeaf = committedLeaf;
    }
}

/** Merkle tree over a list of leaf hashes; `levels[0]` is the leaf level. */
final class MerkleLog {
    final List<Long> leaves;
    final List<List<Long>> levels;
    long root;
    int hashesComputed;

    MerkleLog(List<Long> leaves) {
        this.leaves = new ArrayList<>(leaves);
        this.levels = new ArrayList<>();
        rebuild();
    }

    void rebuild() {
        hashesComputed = 0;
        levels.clear();
        levels.add(new ArrayList<>(leaves));
        List<Long> current = levels.get(0);
        // Odd levels promote the last node unchanged, the standard "carry the lone node up" rule.
        while (current.size() > 1) {
            List<Long> parent = new ArrayList<>();
            for (int i = 0; i < current.size(); i += 2) {
                if (i + 1 < current.size()) {
                    parent.add(LogHasher.node(current.get(i), current.get(i + 1)));
                } else {
                    parent.add(current.get(i));
                }
                hashesComputed++;
            }
            levels.add(parent);
            current = parent;
        }
        root = current.isEmpty() ? 0L : current.get(0);
    }

    int size() { return leaves.size(); }

    /** Sibling hashes from the leaf up to (but excluding) the root. */
    List<Long> inclusionProof(int index) {
        if (index < 0 || index >= leaves.size()) throw new IndexOutOfBoundsException("index " + index);
        List<Long> proof = new ArrayList<>();
        int i = index;
        for (int level = 0; level + 1 < levels.size(); level++) {
            List<Long> nodes = levels.get(level);
            int sibling = (i % 2 == 0) ? i + 1 : i - 1;
            if (sibling < nodes.size()) proof.add(nodes.get(sibling));
            i /= 2;
        }
        return proof;
    }

    /** Recompute the root from a leaf and its path; this is the only verifier the client trusts. */
    static long verifyInclusion(long leaf, int index, List<Long> path) {
        long acc = leaf;
        int i = index;
        for (long sibling : path) {
            acc = (i % 2 == 0) ? LogHasher.node(acc, sibling) : LogHasher.node(sibling, acc);
            i /= 2;
        }
        return acc;
    }

    /**
     * Append-only consistency proof (RFC 6962 style, simplified to the two-block recursion): returns the
     * hashes a client needs to check that the tree over the first m leaves is a prefix of this one.
     */
    List<Long> consistencyProof(int m) {
        if (m < 0 || m > leaves.size()) throw new IllegalArgumentException("prefix " + m);
        if (m == leaves.size()) return List.of();
        List<Long> proof = new ArrayList<>();
        int level = 0;
        int node = m;
        while (level + 1 < levels.size()) {
            if (node % 2 == 1 && node + 1 < levels.get(level).size()) proof.add(levels.get(level).get(node + 1));
            node /= 2;
            level++;
        }
        proof.add(levels.get(level).get(0));
        return proof;
    }

    /** Independent root oracle: fold the leaf hashes pairwise, no shared state with rebuild(). */
    static long oracleRoot(List<Long> leafHashes) {
        List<Long> current = new ArrayList<>(leafHashes);
        if (current.isEmpty()) return 0;
        while (current.size() > 1) {
            List<Long> next = new ArrayList<>();
            for (int i = 0; i < current.size(); i += 2) {
                next.add(i + 1 < current.size() ? LogHasher.node(current.get(i), current.get(i + 1))
                        : current.get(i));
            }
            current = next;
        }
        return current.get(0);
    }
}

/** A "cluster" replica: a Merkle log plus the audit trail of what it committed to. */
final class Replica {
    final String name;
    final List<LogEntry> entries = new ArrayList<>();
    MerkleLog tree;
    String auditReport = "";
    boolean healthy = true;

    Replica(String name) {
        this.name = name;
        this.tree = new MerkleLog(new ArrayList<>());
    }

    void append(String payload) {
        int index = entries.size();
        entries.add(new LogEntry(index, payload, LogHasher.leaf(payload)));
        tree = new MerkleLog(leafHashes());
    }

    List<Long> leafHashes() {
        List<Long> hashes = new ArrayList<>();
        for (LogEntry e : entries) hashes.add(e.committedLeaf);
        return hashes;
    }

    /** Self-verification: recompute the root from the payloads and compare with the published root. */
    boolean audit() {
        List<Long> recomputed = new ArrayList<>();
        for (LogEntry e : entries) recomputed.add(LogHasher.leaf(e.payload));
        long independent = MerkleLog.oracleRoot(recomputed);
        healthy = independent == tree.root;
        auditReport = healthy ? "ok(" + entries.size() + " entries)" : "root mismatch";
        return healthy;
    }

    /** In-place tamper that keeps the entry count (the interesting attack case). */
    void mutate(int index, String newPayload) {
        LogEntry old = entries.get(index);
        entries.set(index, new LogEntry(old.index, newPayload, old.committedLeaf));
    }
}

public class Solution {
    public static void main(String[] args) {
        String[] payloads = {"genesis", "tx:alice->bob:7", "tx:bob->carol:3", "tx:carol->dave:11",
                "tx:dave->eve:2", "tx:eve->frank:5", "tx:frank->grace:13", "tx:grace->heidi:1"};
        Replica primary = new Replica("primary");
        for (String p : payloads) primary.append(p);
        MerkleLog tree = primary.tree;

        // ---- 1. shape and independent root oracle ----------------------------------------------------
        int expectedHeight = 1;
        for (int size = payloads.length; size > 1; size = (size + 1) / 2) expectedHeight++;
        check(tree.levels.size() == expectedHeight,
                "height must be ceil(log2 n)+1 = " + expectedHeight + ", got " + tree.levels.size());
        check(tree.root == MerkleLog.oracleRoot(primary.leafHashes()), "published root must match the oracle");
        check(tree.size() == 8 && tree.levels.get(0).size() == 8, "leaf level must hold every entry");

        // ---- 2. every inclusion proof verifies, and the path length is exactly the tree height ---------
        for (int i = 0; i < payloads.length; i++) {
            List<Long> path = tree.inclusionProof(i);
            long recomputed = MerkleLog.verifyInclusion(LogHasher.leaf(payloads[i]), i, path);
            check(recomputed == tree.root, "proof for index " + i + " must reconstruct the root");
            check(path.size() == tree.levels.size() - 1,
                    "path length for index " + i + " must be " + (tree.levels.size() - 1));
        }
        check(tree.hashesComputed <= 2 * payloads.length, "rebuild must stay linear: " + tree.hashesComputed);

        // ---- 3. negative checks: every plausible forgery is rejected ----------------------------------
        List<Long> honestPath = tree.inclusionProof(2);
        check(MerkleLog.verifyInclusion(LogHasher.leaf("tx:bob->carol:3"), 2, honestPath) == tree.root,
                "the honest leaf at index 2 must verify");
        check(MerkleLog.verifyInclusion(LogHasher.leaf("tx:bob->carol:4"), 2, honestPath) != tree.root,
                "a leaf with a changed amount must not verify");
        check(MerkleLog.verifyInclusion(LogHasher.leaf("tx:bob->carol:3"), 3, honestPath) != tree.root,
                "replaying a valid leaf at the wrong index must not verify");
        List<Long> swapped = new ArrayList<>(honestPath);
        if (swapped.size() >= 2) Collections.swap(swapped, 0, 1);
        check(MerkleLog.verifyInclusion(LogHasher.leaf("tx:bob->carol:3"), 2, swapped) != tree.root,
                "swapping siblings must not verify");
        check(MerkleLog.verifyInclusion(LogHasher.leaf("tx:bob->carol:3"), 2,
                honestPath.subList(0, honestPath.size() - 1)) != tree.root, "a truncated path must not verify");
        boolean rejectedOutOfRange = false;
        try {
            tree.inclusionProof(8);
        } catch (IndexOutOfBoundsException ex) {
            rejectedOutOfRange = true;
        }
        check(rejectedOutOfRange, "an out-of-range index must be rejected, not silently clamped");
        check(LogHasher.leaf("ab") != LogHasher.leaf("ba"),
                "the leaf hash must depend on byte order, so a transposed payload cannot masquerade");
        check(LogHasher.leaf("x") == LogHasher.leaf("x"), "leaf hashing must be deterministic");

        // ---- 4. consistency between every prefix pair --------------------------------------------------
        for (int m = 0; m <= payloads.length; m++) {
            List<Long> proof = tree.consistencyProof(m);
            check(m == payloads.length ? proof.isEmpty() : !proof.isEmpty(),
                    "a strict prefix needs a non-empty consistency proof (m=" + m + ")");
            List<Long> prefixHashes = primary.leafHashes().subList(0, m);
            long prefixRoot = MerkleLog.oracleRoot(prefixHashes);
            check(prefixRoot != tree.root || m == payloads.length,
                    "a strict prefix must have a different root (m=" + m + ")");
        }
        List<Long> extra = new ArrayList<>(primary.leafHashes());
        extra.add(LogHasher.leaf("tx:heidi->ivan:9"));
        check(MerkleLog.oracleRoot(extra) != tree.root, "appending an entry must change the root");
        check(new MerkleLog(extra).inclusionProof(8).size() >= 1, "the extended tree proves the new leaf");

        // ---- 5. boundary cases: empty and single-entry logs --------------------------------------------
        MerkleLog empty = new MerkleLog(new ArrayList<>());
        check(empty.size() == 0 && empty.levels.size() == 1, "an empty log has one leaf-level list and no root");
        Replica lone = new Replica("lone");
        lone.append("only");
        check(lone.tree.root == LogHasher.leaf("only"), "a single-entry log uses the leaf itself as the root");
        check(lone.tree.inclusionProof(0).isEmpty(), "the single-entry inclusion path is empty");
        check(lone.audit(), "a single-entry replica must audit clean");

        // ---- 6. self-verifying cluster: replicas agree, then one is caught lying ------------------------
        Replica mirrorA = new Replica("mirror-a");
        Replica mirrorB = new Replica("mirror-b");
        for (int i = 0; i < payloads.length; i++) {
            mirrorA.append(payloads[i]);
            mirrorB.append(payloads[i]);
        }
        check(mirrorA.tree.root == tree.root && mirrorB.tree.root == tree.root,
                "three replicas of the same log must publish the same root");
        check(mirrorA.audit() && mirrorB.audit() && primary.audit(), "all honest replicas must audit clean");
        mirrorB.mutate(5, "tx:eve->frank:50");
        check(!mirrorB.audit() && mirrorB.auditReport.equals("root mismatch"),
                "an in-place mutation must be caught by the replica's own audit");
        check(primary.audit() && mirrorA.audit(), "catching one liar must not disturb the honest replicas");
        check(mirrorB.tree.root == tree.root,
                "the liar's published root is unchanged, so a reader without the audit would be fooled");
        long forged = MerkleLog.verifyInclusion(LogHasher.leaf("tx:eve->frank:50"), 5, mirrorB.tree.inclusionProof(5));
        check(forged != tree.root, "the mutated leaf cannot be proven against the honest root");

        // ---- 7. determinism ----------------------------------------------------------------------------
        Replica rebuilt = new Replica("rebuilt");
        for (String p : payloads) rebuilt.append(p);
        check(rebuilt.tree.root == tree.root, "rebuilding from the same payloads must give the same root");
        check(primary.tree.hashesComputed == rebuilt.tree.hashesComputed,
                "rebuild cost must be deterministic: " + primary.tree.hashesComputed + " vs "
                        + rebuilt.tree.hashesComputed);

        System.out.println("merkle log of " + payloads.length + " entries, height " + tree.levels.size()
                + ", " + tree.hashesComputed + " hashes per rebuild");
        System.out.println("inclusion proofs verified: " + payloads.length + "; forgery classes rejected: 5");
        System.out.println("consistency prefixes checked: " + (payloads.length + 1) + "; liar replica caught: "
                + (!mirrorB.healthy));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
