// Task 902 - Implement self-verifying distributed logs with Merkle trees and authenticated queries.
// (Tier: Master, Beyond - Distributed Systems)
//
// An append-only log whose entries are authenticated with an RFC 6962 style Merkle tree: leaves are
// H(0x00 || entry), internal nodes are H(0x01 || left || right), and a node without a right sibling is
// promoted unchanged.  Because the tree shape is the binary decomposition of the log size, the same
// entries always produce the same root and any edit changes it.
//
// What is implemented here:
//  * audit paths - the sibling hashes from a leaf to the root, plus the RFC 6962 verification where the
//    rightmost leaf of an unbalanced tree climbs by halving instead of folding (the subtle part of the
//    algorithm);
//  * authenticated queries - a range query returns the consecutive entries together with an audit path
//    per entry, so a client can verify the answer against a root it already trusts;
//  * verified appends - the log keeps the frontier of complete subtrees (a compact Merkle tree), so a
//    client that knows only the old root, the old frontier and the newly appended entries can recompute
//    both roots and confirm that the new history really extends the old one.
//
// Simplifications: entries are small strings, sizes stay below 64 leaves, SHA-256 comes from the BCL, and
// the "distributed" aspect is modelled as a server holding the log and clients verifying proofs in
// process.  Verified: the recursive root against an independent bottom-up level-by-level oracle, every
// audit path, tampered or truncated proofs, empty and single-entry boundaries, range queries, and
// append-consistency with forged roots/frontiers/leaves rejected.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private static byte[] LeafHash(string entry) =>
        SHA256.HashData(Concat(new byte[] { 0x00 }, Encoding.UTF8.GetBytes(entry)));

    private static byte[] NodeHash(byte[] left, byte[] right) =>
        SHA256.HashData(Concat(new byte[] { 0x01 }, Concat(left, right)));

    private static byte[] Concat(byte[] a, byte[] b)
    {
        byte[] result = new byte[a.Length + b.Length];
        Array.Copy(a, 0, result, 0, a.Length);
        Array.Copy(b, 0, result, a.Length, b.Length);
        return result;
    }

    private static bool Same(byte[] a, byte[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static bool AllEqual(IReadOnlyList<byte[]> a, IReadOnlyList<byte[]> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (!Same(a[i], b[i])) return false;
        }
        return true;
    }

    private static string Hex(byte[] value)
    {
        StringBuilder sb = new StringBuilder(value.Length * 2);
        foreach (byte b in value) sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
        return sb.ToString();
    }

    private static byte[] EmptyRoot() => SHA256.HashData(Array.Empty<byte>());

    internal sealed class MerkleLog
    {
        private readonly List<string> _entries = new List<string>();
        private readonly List<byte[]> _frontier = new List<byte[]>(); // one hash per set bit of the size

        internal int Size => _entries.Count;

        internal void Append(string entry)
        {
            _entries.Add(entry);
            int index = _entries.Count - 1;
            byte[] node = LeafHash(entry);
            while ((index & 1) == 1)
            {
                node = NodeHash(_frontier[_frontier.Count - 1], node);
                _frontier.RemoveAt(_frontier.Count - 1);
                index >>= 1;
            }
            _frontier.Add(node);
        }

        /** The hash of the complete subtree rooted at that node, memo-free recursive definition. */
        internal byte[] SubtreeRoot(int from, int count)
        {
            if (count == 1) return LeafHash(_entries[from]);
            int k = 1;
            while (k * 2 < count) k *= 2;
            return NodeHash(SubtreeRoot(from, k), SubtreeRoot(from + k, count - k));
        }

        internal byte[] Root() => Size == 0 ? EmptyRoot() : Fold(Frontier());

        internal List<byte[]> Frontier() => new List<byte[]>(_frontier);

        internal static byte[] Fold(List<byte[]> frontier)
        {
            if (frontier.Count == 0) return EmptyRoot();
            byte[] acc = frontier[frontier.Count - 1];
            for (int i = frontier.Count - 2; i >= 0; i--) acc = NodeHash(frontier[i], acc);
            return acc;
        }

        /** The RFC 6962 audit path: sibling hashes from the leaf up to the root. */
        internal List<byte[]> AuditPath(int index)
        {
            List<byte[]> path = new List<byte[]>();
            Fill(path, 0, Size, index);
            return path;
        }

        private void Fill(List<byte[]> path, int from, int count, int index)
        {
            if (count <= 1) return;
            int k = 1;
            while (k * 2 < count) k *= 2;
            if (index < k)
            {
                Fill(path, from, k, index);
                path.Add(SubtreeRoot(from + k, count - k));
            }
            else
            {
                Fill(path, from + k, count - k, index - k);
                path.Add(SubtreeRoot(from, k));
            }
        }

        internal string EntryAt(int index) => _entries[index];

        internal List<string> Range(int from, int count)
        {
            List<string> slice = new List<string>();
            for (int i = from; i < from + count; i++) slice.Add(_entries[i]);
            return slice;
        }
    }

    /** RFC 6962 verification, including the rightmost-leaf halving rule. */
    private static bool VerifyAuditPath(byte[] leafHash, int index, int size, List<byte[]> path, byte[] root)
    {
        if (index < 0 || index >= size || size <= 0) return false;
        if (size == 1) return path.Count == 0 && Same(leafHash, root);
        long fn = index;
        long sn = size - 1;
        byte[] r = leafHash;
        foreach (byte[] p in path)
        {
            if (sn == 0) return false; // the proof is longer than the tree is deep
            if ((fn & 1) == 1 || fn == sn)
            {
                r = NodeHash(p, r);
                if ((fn & 1) == 0)
                {
                    while ((fn & 1) == 0 && fn != 0)
                    {
                        fn >>= 1;
                        sn >>= 1;
                    }
                }
            }
            else
            {
                r = NodeHash(r, p);
            }
            fn >>= 1;
            sn >>= 1;
        }
        return sn == 0 && Same(r, root);
    }

    /** Independent oracle: the tree built level by level, promoting the odd node at every level. */
    private static byte[] LevelOracle(List<string> entries)
    {
        if (entries.Count == 0) return EmptyRoot();
        List<byte[]> level = new List<byte[]>();
        foreach (string entry in entries) level.Add(LeafHash(entry));
        while (level.Count > 1)
        {
            List<byte[]> next = new List<byte[]>();
            for (int i = 0; i + 1 < level.Count; i += 2) next.Add(NodeHash(level[i], level[i + 1]));
            if ((level.Count & 1) == 1) next.Add(level[level.Count - 1]);
            level = next;
        }
        return level[0];
    }

    private static string Entry(int i) => "tx:" + i + ":" + (i * 37 % 101);

    public static void Main()
    {
        // ---- 1. the recursive root against the independent level-by-level oracle --------------------
        int oracleSizes = 0;
        for (int size = 1; size <= 24; size++)
        {
            MerkleLog log = new MerkleLog();
            for (int i = 0; i < size; i++) log.Append(Entry(i));
            List<string> entries = log.Range(0, size);
            Check(Same(log.Root(), LevelOracle(entries)), "root matches the level oracle at size " + size);
            oracleSizes++;
        }
        Check(Same(new MerkleLog().Root(), EmptyRoot()), "an empty log exposes the empty-root constant");

        // ---- 2. audit paths verify for every leaf of every size ----------------------------------
        MerkleLog big = new MerkleLog();
        for (int i = 0; i < 24; i++) big.Append(Entry(i));
        int verified = 0;
        for (int size = 1; size <= 24; size++)
        {
            MerkleLog log = new MerkleLog();
            for (int i = 0; i < size; i++) log.Append(Entry(i));
            byte[] root = log.Root();
            for (int index = 0; index < size; index++)
            {
                bool ok = VerifyAuditPath(LeafHash(log.EntryAt(index)), index, size, log.AuditPath(index), root);
                if (!ok) Check(false, "audit path failed at size " + size + " index " + index);
                verified++;
            }
        }
        Check(verified == 24 * 25 / 2, "all " + verified + " audit paths verified");

        // ---- 3. tampered, truncated and misplaced proofs ------------------------------------------
        byte[] root24 = big.Root();
        Check(!VerifyAuditPath(LeafHash("tx:0:tampered"), 0, 24, big.AuditPath(0), root24),
            "a tampered leaf fails verification");
        Check(!VerifyAuditPath(LeafHash(big.EntryAt(1)), 1, 24, big.AuditPath(0), root24),
            "a proof presented with the wrong leaf fails");
        Check(!VerifyAuditPath(LeafHash(big.EntryAt(0)), 3, 24, big.AuditPath(0), root24),
            "a proof presented at the wrong index fails");
        List<byte[]> truncated = new List<byte[]>(big.AuditPath(0));
        truncated.RemoveAt(truncated.Count - 1);
        Check(!VerifyAuditPath(LeafHash(big.EntryAt(0)), 0, 24, truncated, root24), "a truncated path fails");
        List<byte[]> extended = new List<byte[]>(big.AuditPath(0));
        extended.Add(new byte[32]);
        Check(!VerifyAuditPath(LeafHash(big.EntryAt(0)), 0, 24, extended, root24), "an over-long path fails");

        // ---- 4. boundary shapes --------------------------------------------------------------------
        MerkleLog single = new MerkleLog();
        single.Append("only");
        Check(single.AuditPath(0).Count == 0 && Same(single.Root(), LeafHash("only")),
            "a single-entry log has an empty audit path and root = leaf hash");
        MerkleLog two = new MerkleLog();
        two.Append("a");
        two.Append("b");
        Check(Same(two.Root(), NodeHash(LeafHash("a"), LeafHash("b"))), "a two-entry log uses one internal node");
        Check(!VerifyAuditPath(LeafHash("a"), 1, 2, two.AuditPath(1), two.Root()), "swapping the leaves breaks the path");

        // ---- 5. authenticated range query -----------------------------------------------------------
        int from = 7;
        int count = 9;
        List<string> slice = big.Range(from, count);
        bool rangeOk = slice.Count == count;
        for (int i = 0; i < slice.Count; i++)
        {
            rangeOk &= slice[i] == Entry(from + i);
            rangeOk &= VerifyAuditPath(LeafHash(slice[i]), from + i, big.Size, big.AuditPath(from + i), root24);
        }
        Check(rangeOk, "every entry of the range query is contiguous and authenticated by the root");
        Check(!VerifyAuditPath(LeafHash(slice[0]), from + 1, big.Size, big.AuditPath(from + 1), root24),
            "a range entry cannot be claimed at a neighbouring index");

        // ---- 6. verified appends (frontier consistency) ------------------------------------------
        MerkleLog growing = new MerkleLog();
        for (int i = 0; i < 10; i++) growing.Append(Entry(i));
        byte[] oldRoot = growing.Root();
        List<byte[]> oldFrontier = growing.Frontier();
        for (int i = 10; i < 17; i++) growing.Append(Entry(i));
        byte[] newRoot = growing.Root();
        Check(MerkleLog.Fold(oldFrontier) is byte[] folded && Same(folded, oldRoot),
            "the published frontier folds back to the old root");
        Check(Same(growing.Root(), LevelOracle(growing.Range(0, 17))), "the extended log still matches the level oracle");

        List<byte[]> forgedFrontier = new List<byte[]>(oldFrontier);
        forgedFrontier[0] = LeafHash("forged");
        Check(!Same(MerkleLog.Fold(forgedFrontier), oldRoot), "a forged frontier does not fold to the old root");
        List<string> appended = new List<string>();
        for (int i = 10; i < 17; i++) appended.Add(Entry(i));
        Check(Same(Replay(oldFrontier, appended, 10), newRoot), "the honest replay reproduces the new root");
        List<string> forgedAppend = new List<string>(appended);
        forgedAppend[3] = "tx:13:replaced";
        Check(!Same(Replay(oldFrontier, forgedAppend, 10), newRoot), "replaying forged appended entries misses the new root");
        Check(!Same(Replay(oldFrontier, appended.GetRange(0, 6), 10), newRoot), "a shortened append list misses the new root");
        Check(Same(Replay(oldFrontier, appended.GetRange(0, 6), 10), LevelOracle(JoinAndPrefix(16))),
            "the shortened replay matches the independently built log of the same prefix");

        // ---- 7. order and content sensitivity, determinism ---------------------------------------
        MerkleLog swapped = new MerkleLog();
        swapped.Append(Entry(1));
        swapped.Append(Entry(0));
        MerkleLog three = new MerkleLog();
        three.Append("a");
        three.Append("b");
        three.Append("c");
        MerkleLog reversedThree = new MerkleLog();
        reversedThree.Append("c");
        reversedThree.Append("b");
        reversedThree.Append("a");
        Check(!Same(three.Root(), reversedThree.Root()), "reversing three entries changes the root");
        MerkleLog ordered = new MerkleLog();
        ordered.Append(Entry(0));
        ordered.Append(Entry(1));
        Check(!Same(swapped.Root(), ordered.Root()), "swapping two entries changes the root");
        MerkleLog rebuilt = new MerkleLog();
        for (int i = 0; i < 24; i++) rebuilt.Append(Entry(i));
        Check(Same(rebuilt.Root(), root24), "rebuilding from the same entries reproduces the root " + Hex(root24).Substring(0, 12));
        Check(MerkleLog.Fold(growing.Frontier()) is byte[] top && Same(top, newRoot), "the frontier always folds to the current root");

        Console.WriteLine("roots: recursive definition equals the level oracle for " + oracleSizes + " sizes; root(24) = " +
            Hex(root24).Substring(0, 16) + "...");
        Console.WriteLine("audit paths: " + verified + " verified, tampered/truncated/over-long/misplaced paths all rejected");
        Console.WriteLine("range query: " + count + " entries from index " + from + " authenticated; append frontier of " +
            oldFrontier.Count + " nodes extended the log to size " + growing.Size);
        Console.WriteLine("PASS");
    }

    private static List<string> JoinAndPrefix(int size)
    {
        List<string> entries = new List<string>();
        for (int i = 0; i < size; i++) entries.Add(Entry(i));
        return entries;
    }

    /** Client-side replay: fold the old frontier, append the claimed entries, return the resulting root. */
    private static byte[] Replay(List<byte[]> frontier, List<string> entries, int oldSize)
    {
        List<byte[]> work = new List<byte[]>(frontier);
        int index = oldSize;
        foreach (string entry in entries)
        {
            byte[] node = LeafHash(entry);
            int bit = index;
            while ((bit & 1) == 1)
            {
                node = NodeHash(work[work.Count - 1], node);
                work.RemoveAt(work.Count - 1);
                bit >>= 1;
            }
            work.Add(node);
            index++;
        }
        return MerkleLog.Fold(work);
    }
}
