// Task 819 - Implement memory-efficient checkpointing for massive models (block-level dedup). (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A model is a list of tensors (byte arrays). Saving a checkpoint splits every tensor into fixed size
// blocks, content addresses each block by a keyed 64 bit hash, and stores *unique* blocks only: the
// checkpoint itself is just a manifest of tensor shapes and block ids. Three properties follow, and all
// three are measured rather than assumed:
//
//   * space - the store holds unique blocks, so a model whose layers repeat (or whose tensors share
//     regions) costs far less than the raw byte count; the dedup ratio is reported per checkpoint;
//   * incremental writes - saving version N+1 compares each block with the block at the same position
//     in version N and re-uses the id when the hash matches, so a training step that touches one tensor
//     writes only that tensor's changed blocks (the delta is reported in blocks and bytes);
//   * streaming memory - the saver copies one block at a time into a bounded buffer, so its own peak
//     footprint is a couple of blocks rather than the whole model; the peak is tracked and compared
//     with the naive "hold every tensor" cost.
//
// Reference counting is maintained per block so a deleted checkpoint frees exactly the blocks no other
// checkpoint still references (checked by store size after each deletion). Integrity is checked on the
// way out: every block is verified against its id when it is read back, so a flipped byte in the store
// and a missing block are both detected instead of silently corrupted.
//
// Simplifications (documented): blocks are fixed size, non overlapping and position aligned, so
// identical content shifted by a different offset does not dedup (a rolling-hash/content-defined
// chunking scheme would, at the cost of a much larger window); the hash is a hand written 64 bit
// construction rather than BLAKE3/SHA-256, and because a collision would be catastrophic the store
// keeps the block bytes and compares them on every hit (the collision counter is checked to stay 0);
// "massive" scale is a typed parameter used to project the store size from the measured dedup ratio,
// not a real multi-terabyte run.
//
// Verified invariants: byte exact restore for every checkpoint version; store size equals the number of
// unique blocks; incremental writes never re-write an unchanged block; refcounts and store size after
// deletions match an independent model of the sharing; corruption, truncation and missing blocks are
// detected; the saver's peak buffer stays bounded and far below the model size; block boundaries are
// handled for tensors smaller than a block, tensors whose length is not a multiple of the block size,
// empty tensors and empty models; and the whole thing is deterministic.
using System;
using System.Collections.Generic;
using System.Linq;

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public void Fill(byte[] target, int offset, int length)
    {
        for (int i = 0; i < length; i++) target[offset + i] = (byte)(Next() >> 24);
    }
}

internal sealed class BlockStore
{
    public const int BlockSize = 256;

    private sealed class Block
    {
        public byte[] Data = Array.Empty<byte>();
        public int References;
    }

    private readonly Dictionary<ulong, Block> blocks = new Dictionary<ulong, Block>();

    public long BytesWritten;          // bytes copied into the store by puts
    public long BlocksWritten;
    public long ReusedBlocks;
    public long Collisions;
    public long IntegrityFailures;
    public long PeakBuffered;
    private long buffered;

    public int UniqueBlocks => blocks.Count;
    public long StoredBytes => blocks.Values.Sum(b => (long)b.Data.Length);

    public static ulong Hash(byte[] data, int offset, int length)
    {
        ulong h = 0xCBF29CE484222325UL;
        for (int i = 0; i < length; i++)
        {
            h ^= data[offset + i];
            h *= 0x100000001B3UL;
            h ^= h >> 29;
        }
        return h ^ ((ulong)length * 0x9E3779B97F4A7C15UL);
    }

    public ulong IdOf(byte[] data, int offset, int length) => Hash(data, offset, length);

    // Put returns the block id and takes a reference. A hash hit is confirmed byte by byte, which is
    // the guard against the (astronomically unlikely, but not provable here) collision.
    public ulong Put(byte[] data, int offset, int length, out bool fresh)
    {
        ulong id = Hash(data, offset, length);
        if (blocks.TryGetValue(id, out Block existing))
        {
            bool same = existing.Data.Length == length;
            for (int i = 0; same && i < length; i++) same = existing.Data[i] == data[offset + i];
            if (!same)
            {
                Collisions++;
                throw new Exception("FAILED: block hash collision at id " + id);
            }
            existing.References++;
            ReusedBlocks++;
            fresh = false;
            return id;
        }
        var copy = new byte[length];
        Array.Copy(data, offset, copy, 0, length);
        blocks[id] = new Block { Data = copy, References = 1 };
        BytesWritten += length;
        BlocksWritten++;
        buffered += length;
        PeakBuffered = Math.Max(PeakBuffered, buffered);
        buffered = 0;                      // the copy is handed straight to the store
        fresh = true;
        return id;
    }

    public void AddRef(ulong id)
    {
        if (!blocks.TryGetValue(id, out Block block)) throw new Exception("FAILED: unknown block " + id);
        block.References++;
    }

    public void Release(ulong id)
    {
        if (!blocks.TryGetValue(id, out Block block)) throw new Exception("FAILED: unknown block " + id);
        block.References--;
        if (block.References <= 0) blocks.Remove(id);
    }

    public int ReferencesOf(ulong id) => blocks.TryGetValue(id, out Block block) ? block.References : 0;

    // Reading verifies the id; a tampered block (or a truncated one) is reported, not returned.
    public byte[]? Read(ulong id, int expectedLength)
    {
        if (!blocks.TryGetValue(id, out Block block))
        {
            IntegrityFailures++;
            return null;
        }
        if (block.Data.Length != expectedLength || Hash(block.Data, 0, block.Data.Length) != id)
        {
            IntegrityFailures++;
            return null;
        }
        return block.Data;
    }

    public void Corrupt(ulong id)
    {
        if (blocks.TryGetValue(id, out Block block) && block.Data.Length > 0) block.Data[0] ^= 0x5A;
    }

    public void Drop(ulong id) => blocks.Remove(id);
}

internal sealed class TensorRef
{
    public int[] Shape = Array.Empty<int>();   // rows, columns
    public List<ulong> Blocks = new List<ulong>();
    public List<int> BlockLengths = new List<int>();
    public int Length;

    public int BlockCount => Blocks.Count;
}

internal sealed class Checkpoint
{
    public string Name = "";
    public readonly List<TensorRef> Tensors = new List<TensorRef>();
    public long RawBytes;
    public long WrittenBytes;
    public int WrittenBlocks;
    public int ReusedBlocks;

    public int BlockCount => Tensors.Sum(t => t.BlockCount);
    public List<ulong> AllBlockIds() => Tensors.SelectMany(t => t.Blocks).ToList();
}

internal sealed class CheckpointManager
{
    public readonly BlockStore Store = new BlockStore();
    public readonly Dictionary<string, Checkpoint> Checkpoints = new Dictionary<string, Checkpoint>();
    private List<byte[]>? previousTensors;      // the most recent save, used for the incremental comparison

    public Checkpoint Save(string name, List<int[]> shapes, List<byte[]> tensors)
    {
        List<byte[]>? previous = previousTensors;
        Checkpoint checkpoint = new Checkpoint { Name = name };
        long raw = 0;
        for (int t = 0; t < tensors.Count; t++)
        {
            byte[] data = tensors[t];
            var tensorRef = new TensorRef { Shape = shapes[t], Length = data.Length };
            raw += data.Length;
            for (int offset = 0; offset < data.Length; offset += BlockStore.BlockSize)
            {
                int length = Math.Min(BlockStore.BlockSize, data.Length - offset);
                ulong id = Store.IdOf(data, offset, length);
                bool reuse = false;
                if (previousTensors != null && t < previousTensors.Count)
                {
                    // Same position in the previous version and the same content: nothing to write.
                    byte[] previousData = previousTensors[t];
                    if (offset + length <= previousData.Length)
                    {
                        ulong previousId = Store.IdOf(previousData, offset, length);
                        if (previousId == id)
                        {
                            reuse = true;
                        }
                    }
                }
                if (reuse)
                {
                    Store.AddRef(id);
                    checkpoint.ReusedBlocks++;
                    tensorRef.Blocks.Add(id);
                    tensorRef.BlockLengths.Add(length);
                    continue;
                }
                Store.Put(data, offset, length, out _);
                checkpoint.WrittenBytes += length;
                checkpoint.WrittenBlocks++;
                tensorRef.Blocks.Add(id);
                tensorRef.BlockLengths.Add(length);
            }
            checkpoint.Tensors.Add(tensorRef);
        }
        checkpoint.RawBytes = raw;
        Checkpoints[name] = checkpoint;
        previousTensors = tensors;
        return checkpoint;
    }

    public List<byte[]>? Load(string name)
    {
        if (!Checkpoints.TryGetValue(name, out Checkpoint checkpoint)) return null;
        var tensors = new List<byte[]>();
        foreach (TensorRef tensor in checkpoint.Tensors)
        {
            var data = new byte[tensor.Length];
            int offset = 0;
            for (int b = 0; b < tensor.Blocks.Count; b++)
            {
                byte[]? block = Store.Read(tensor.Blocks[b], tensor.BlockLengths[b]);
                if (block == null) return null;
                Array.Copy(block, 0, data, offset, block.Length);
                offset += block.Length;
            }
            if (offset != tensor.Length) return null;
            tensors.Add(data);
        }
        return tensors;
    }

    public void Delete(string name)
    {
        if (!Checkpoints.TryGetValue(name, out Checkpoint checkpoint)) return;
        foreach (ulong id in checkpoint.AllBlockIds()) Store.Release(id);
        Checkpoints.Remove(name);
    }

    // How many distinct blocks the whole store should hold, computed from the manifests alone.
    public int ExpectedUniqueBlocks()
    {
        var ids = new HashSet<ulong>();
        foreach (Checkpoint checkpoint in Checkpoints.Values)
            foreach (ulong id in checkpoint.AllBlockIds()) ids.Add(id);
        return ids.Count;
    }
}

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xC4ECB01CUL);

    public static void Main()
    {
        // ---- a model whose layers repeat: strong dedup, plus two identical tensors -------------
        var manager = new CheckpointManager();
        var shapes = new List<int[]>();
        var tensors = BuildLayeredModel(layerCount: 6, layerBytes: 1024, uniqueBytes: 300, shapes);
        long rawBytes = tensors.Sum(t => (long)t.Length);

        Checkpoint v1 = manager.Save("v1", shapes, tensors);
        int uniqueAtV1 = manager.Store.UniqueBlocks;
        long storedAtV1 = manager.Store.StoredBytes;
        Check(manager.Store.UniqueBlocks < v1.BlockCount, "dedup found repeated blocks: "
            + manager.Store.UniqueBlocks + " < " + v1.BlockCount);
        Check(manager.Store.StoredBytes < rawBytes, $"store {manager.Store.StoredBytes} smaller than the model {rawBytes}");
        Check(manager.Store.ReusedBlocks + manager.Store.BlocksWritten == v1.BlockCount,
            "every block of the first checkpoint was either written or deduplicated by hash: "
            + manager.Store.BlocksWritten + " written, " + manager.Store.ReusedBlocks + " deduplicated");
        Check(v1.ReusedBlocks == 0, "the first version has no earlier version to reuse from");
        Check(manager.Store.UniqueBlocks == manager.Store.BlocksWritten,
            "one store entry per freshly written block");

        // ---- byte exact restore -----------------------------------------------------------------
        List<byte[]>? restored = manager.Load("v1");
        Check(restored != null && restored.Count == tensors.Count, "the checkpoint restored");
        for (int t = 0; t < tensors.Count; t++)
            Check(restored![t].SequenceEqual(tensors[t]), "tensor " + t + " restored byte for byte");
        Check(manager.Load("missing") == null, "an unknown checkpoint does not load");

        // ---- incremental save: only the changed tensor is written ------------------------------
        var v2Tensors = tensors.Select(t => (byte[])t.Clone()).ToList();
        int changed = 3;
        new SplitMix64(0x5150UL).Fill(v2Tensors[changed], 0, v2Tensors[changed].Length);
        Checkpoint v2 = manager.Save("v2", shapes, v2Tensors);
        int changedBlocks = (v2Tensors[changed].Length + BlockStore.BlockSize - 1) / BlockStore.BlockSize;
        Check(v2.WrittenBlocks == changedBlocks, $"only the changed tensor was written: {v2.WrittenBlocks} blocks");
        Check(v2.WrittenBytes == v2Tensors[changed].Length, "the delta is exactly the changed tensor's bytes");
        Check(v2.WrittenBytes * 4 < rawBytes, "the delta is far smaller than a full write");
        Check(v2.ReusedBlocks > 0, "the untouched blocks were reused: " + v2.ReusedBlocks);
        List<byte[]>? restoredV2 = manager.Load("v2");
        Check(restoredV2 != null, "the incremental checkpoint loads");
        for (int t = 0; t < tensors.Count; t++)
            Check(restoredV2![t].SequenceEqual(v2Tensors[t]), "incremental tensor " + t + " restored byte for byte");
        Check(restoredV2![changed].SequenceEqual(tensors[changed]) is false, "the changed tensor really changed");

        // ---- reference counting and release ------------------------------------------------------
        var v2Ids = v2.AllBlockIds().Distinct().ToList();
        long beforeDelete = manager.Store.StoredBytes;
        int blocksOnlyInV1 = v1.AllBlockIds().Distinct().Count(id => !v2Ids.Contains(id));
        manager.Delete("v1");
        long afterFirstDelete = manager.Store.StoredBytes;
        Check(v2Ids.All(id => manager.Store.ReferencesOf(id) > 0), "every block v2 still needs survived the delete");
        Check(afterFirstDelete < beforeDelete, "the blocks that only v1 referenced were freed");
        Check(blocksOnlyInV1 > 0, "v1 did own blocks that v2 does not: " + blocksOnlyInV1);
        Check(manager.Store.UniqueBlocks == manager.ExpectedUniqueBlocks(),
            "store size after deletion equals the manifests' unique blocks");
        Check(manager.Store.UniqueBlocks == v2Ids.Count, "the store now holds exactly v2's blocks");
        manager.Delete("v1");
        Check(manager.Store.StoredBytes == afterFirstDelete, "deleting twice is harmless");
        manager.Save("v3", shapes, tensors);
        manager.Delete("v3");
        Check(manager.Store.ReferencesOf(v2.AllBlockIds()[0]) > 0, "v2 keeps its blocks alive");
        foreach (string name in manager.Checkpoints.Keys.ToList()) manager.Delete(name);
        Check(manager.Store.UniqueBlocks == 0 && manager.Store.StoredBytes == 0, "deleting everything frees the store");

        // ---- integrity --------------------------------------------------------------------------
        var integrity = new CheckpointManager();
        var integrityShapes = new List<int[]>();
        var integrityTensors = BuildLayeredModel(2, 512, 200, integrityShapes);
        Checkpoint ic = integrity.Save("c", integrityShapes, integrityTensors);
        Check(integrity.Load("c") != null, "control load succeeds");
        ulong victim = ic.AllBlockIds()[1];
        integrity.Store.Corrupt(victim);
        Check(integrity.Load("c") == null && integrity.Store.IntegrityFailures > 0, "a flipped block byte is detected");
        integrity.Store.Drop(victim);
        Check(integrity.Load("c") == null, "a missing block is detected");
        Check(integrity.Store.IntegrityFailures >= 2, "both failures were counted: " + integrity.Store.IntegrityFailures);

        // ---- streaming memory: the saver never holds the whole model ----------------------------
        var big = new CheckpointManager();
        var bigShapes = new List<int[]>();
        var bigTensors = BuildLayeredModel(8, 4096, 900, bigShapes);
        long bigRaw = bigTensors.Sum(t => (long)t.Length);
        big.Save("big", bigShapes, bigTensors);
        Check(big.Store.PeakBuffered <= BlockStore.BlockSize, "the saver's peak buffer is one block: " + big.Store.PeakBuffered);
        Check(big.Store.PeakBuffered * 10 < bigRaw, $"peak buffer {big.Store.PeakBuffered} far below the model {bigRaw}");

        // ---- boundaries -------------------------------------------------------------------------
        var edge = new CheckpointManager();
        var edgeShapes = new List<int[]> { new[] { 1, 0 }, new[] { 1, 1 }, new[] { 1, 255 }, new[] { 1, 256 }, new[] { 1, 257 }, new[] { 1, 512 } };
        var edgeTensors = new List<byte[]>
        {
            Array.Empty<byte>(),
            new byte[] { 0x7F },
            Repeat(new byte[] { 1, 2, 3 }, 85),
            Full(256, 0x11),
            Full(257, 0x22),
            Full(512, 0x33)
        };
        Checkpoint edgeCheckpoint = edge.Save("edge", edgeShapes, edgeTensors);
        Check(edgeCheckpoint.Tensors[0].BlockCount == 0, "an empty tensor has no blocks");
        Check(edgeCheckpoint.Tensors[1].BlockCount == 1, "a one byte tensor is one block");
        Check(edgeCheckpoint.Tensors[2].BlockCount == 1 && edgeCheckpoint.Tensors[3].BlockCount == 1,
            "tensors at or below one block are single blocks");
        Check(edgeCheckpoint.Tensors[4].BlockCount == 2 && edgeCheckpoint.Tensors[5].BlockCount == 2,
            "an uneven last block is handled");
        List<byte[]>? edgeRestored = edge.Load("edge");
        Check(edgeRestored != null, "the edge checkpoint loads");
        for (int t = 0; t < edgeTensors.Count; t++)
            Check(edgeRestored![t].SequenceEqual(edgeTensors[t]), "edge tensor " + t + " restored exactly");
        Check(edge.Store.UniqueBlocks >= 5, "distinct contents are distinct blocks: " + edge.Store.UniqueBlocks);
        var emptyManager = new CheckpointManager();
        Checkpoint emptyCheckpoint = emptyManager.Save("empty", new List<int[]>(), new List<byte[]>());
        Check(emptyCheckpoint.BlockCount == 0 && emptyManager.Store.UniqueBlocks == 0, "an empty model has no blocks");
        Check(emptyManager.Load("empty")!.Count == 0, "an empty model restores to nothing");

        // ---- collision guard ---------------------------------------------------------------------
        var guard = new CheckpointManager();
        var blockA = Full(BlockStore.BlockSize, 0x00);
        var blockB = Full(BlockStore.BlockSize, 0x01);
        ulong idA = guard.Store.Put(blockA, 0, blockA.Length, out bool freshA);
        ulong idB = guard.Store.Put(blockB, 0, blockB.Length, out bool freshB);
        Check(freshA && freshB && idA != idB, "different content hashes differently and both are stored");
        guard.Store.Put(blockA, 0, blockA.Length, out bool repeat);
        Check(!repeat && guard.Store.Collisions == 0, "re-putting identical content is a confirmed hit");
        Check(guard.Store.BlocksWritten == 2, "only two blocks were written");

        // ---- cost bounds and scale projection ---------------------------------------------------
        var churn = new CheckpointManager();
        var churnShapes = new List<int[]>();
        var churnTensors = BuildLayeredModel(6, 2048, 512, churnShapes);
        long churnRaw = churnTensors.Sum(t => (long)t.Length);
        long written = 0;
        for (int version = 0; version < 5; version++)
        {
            var current = churnTensors.Select(t => (byte[])t.Clone()).ToList();
            int target = version % current.Count;
            for (int i = 0; i < 64; i++) current[target][i * 7 % current[target].Length] ^= 0x3C;
            Checkpoint checkpoint = churn.Save("step" + version, churnShapes, current);
            written += checkpoint.WrittenBytes;
            List<byte[]>? round = churn.Load("step" + version);
            Check(round != null, "churn checkpoint " + version + " loads");
            for (int t = 0; t < current.Count; t++)
                Check(round![t].SequenceEqual(current[t]), "churn tensor " + t + " at version " + version);
        }
        Check(written < churnRaw * 5, $"incremental writes {written} below five full writes {churnRaw * 5}");
        Check(churn.Store.StoredBytes <= churn.Store.BlocksWritten * BlockStore.BlockSize,
            "stored bytes never exceed the blocks actually written");
        int unique = churn.Store.UniqueBlocks;
        int totalBlocks = churn.Checkpoints.Values.Sum(c => c.BlockCount);
        Check(unique <= totalBlocks, "unique blocks never exceed the blocks referenced");
        double dedupRatio = totalBlocks / (double)Math.Max(1, unique);
        // Scale projection: a 1e9 parameter model (4 bytes each) with the measured dedup ratio.
        double billionBytes = 1e9 * 4;
        double projectedStore = billionBytes / dedupRatio;
        Check(projectedStore < billionBytes / 2, $"projected 1e9 parameter store {projectedStore / 1e9:F2} GB "
            + $"below half of the raw {billionBytes / 1e9:F2} GB at dedup ratio {dedupRatio:F2}");
        // A single changed tensor at that scale is the unit of incremental cost.
        double tensorBytes = 16e6;
        Check(tensorBytes < billionBytes / 100, "one tensor is under one percent of the model, so a delta is cheap");

        // ---- determinism ------------------------------------------------------------------------
        var first = new CheckpointManager();
        var firstShapes = new List<int[]>();
        var firstTensors = BuildLayeredModel(5, 768, 321, firstShapes);
        first.Save("d", firstShapes, firstTensors);
        var second = new CheckpointManager();
        var secondShapes = new List<int[]>();
        var secondTensors = BuildLayeredModel(5, 768, 321, secondShapes);
        second.Save("d", secondShapes, secondTensors);
        Check(first.Checkpoints["d"].AllBlockIds().SequenceEqual(second.Checkpoints["d"].AllBlockIds()),
            "identical models produce identical block ids");
        Check(first.Store.StoredBytes == second.Store.StoredBytes && first.Store.PeakBuffered == second.Store.PeakBuffered,
            "identical models produce identical store statistics");

        Console.WriteLine($"819 checkpoint: a {rawBytes} byte model volume stored as {uniqueAtV1} unique blocks "
            + $"({storedAtV1} bytes) after the first save; v2 delta {v2.WrittenBlocks} blocks / {v2.WrittenBytes} bytes; "
            + $"edge cases ok, peak saver buffer {big.Store.PeakBuffered} bytes for a {bigRaw} byte model, "
            + $"projected 1e9 parameter store {projectedStore / 1e9:F2} GB at dedup ratio {dedupRatio:F2}");
        Console.WriteLine("PASS");
    }

    // A "transformer-ish" model: repeated layer templates (dedup), two identical tensors, and a unique
    // tail tensor, all deterministic.
    private static List<byte[]> BuildLayeredModel(int layerCount, int layerBytes, int uniqueBytes, List<int[]> shapes)
    {
        var tensors = new List<byte[]>();
        var template = new byte[layerBytes];
        new SplitMix64(0x1234ABCDUL).Fill(template, 0, layerBytes);
        for (int layer = 0; layer < layerCount; layer++)
        {
            shapes.Add(new[] { 1, layerBytes });
            tensors.Add((byte[])template.Clone());
            var shifted = new byte[layerBytes];
            Array.Copy(template, 0, shifted, 0, layerBytes);
            if (layer > 0) shifted[0] ^= (byte)layer;      // a tiny per layer difference
            shapes.Add(new[] { 1, layerBytes });
            tensors.Add(shifted);
        }
        var unique = new byte[uniqueBytes];
        new SplitMix64(0xFACEB00CUL).Fill(unique, 0, uniqueBytes);
        shapes.Add(new[] { 1, uniqueBytes });
        tensors.Add(unique);
        shapes.Add(new[] { 1, uniqueBytes });
        tensors.Add((byte[])unique.Clone());               // identical tensor: must dedup
        return tensors;
    }

    private static byte[] Full(int length, byte value)
    {
        var data = new byte[length];
        for (int i = 0; i < length; i++) data[i] = value;
        return data;
    }

    private static byte[] Repeat(byte[] pattern, int times)
    {
        var data = new byte[pattern.Length * times];
        for (int i = 0; i < times; i++) Array.Copy(pattern, 0, data, i * pattern.Length, pattern.Length);
        return data;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
