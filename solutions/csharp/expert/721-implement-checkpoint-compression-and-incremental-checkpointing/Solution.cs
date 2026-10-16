// Task 721 - Implement checkpoint compression and incremental checkpointing. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// A training checkpoint is a large flat array of 32-bit parameter words, and consecutive checkpoints
// usually differ in a handful of blocks, so rewriting the whole image is wasteful. The store here has
// two encodings:
//
//   full   type byte 0 + element count + run-length-encoded zig-zag varints. Long runs of equal words
//          (zero padding, frozen layers, quantised weights that saturate) collapse to two varints.
//   delta  type byte 1 + block count + change bitmap + one XOR payload per *changed* block, again
//          zig-zag varints of the XOR difference. XOR of 32-bit words is exactly invertible, so a
//          delta reconstructs the previous checkpoint bit-for-bit, negative values included.
//
// The chain keeps payloads in order; Restore(i) walks back to the newest full image at or before i and
// replays deltas. When more than half of the blocks changed the encoder falls back to a full image,
// which bounds write amplification (a delta larger than an image is never written) and lets the store
// stop depending on the earlier prefix once a full image has been written.
//
// Verified: varint/zig-zag round-trips on every boundary value, RLE round-trips (constant, all-distinct,
// empty, single element, long runs), bit-exact restore of *every* checkpoint against the recorded
// states (an independent oracle: the store never sees those arrays again), the header-only delta of an
// unchanged state, the dense-workload fallback, the size advantage on the sparse workload, and a
// rejected truncated payload. All mutation indices come from splitmix64, so runs are reproducible.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal static class Varint
{
    internal static void Write(List<byte> sink, ulong value)
    {
        while (value >= 0x80)
        {
            sink.Add((byte)(value | 0x80));
            value >>= 7;
        }
        sink.Add((byte)value);
    }

    internal static ulong Read(IReadOnlyList<byte> buffer, ref int offset)
    {
        ulong value = 0;
        int shift = 0;
        while (true)
        {
            if (offset >= buffer.Count) throw new InvalidOperationException("truncated varint");
            byte b = buffer[offset++];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return value;
            shift += 7;
            if (shift > 63) throw new InvalidOperationException("varint too long");
        }
    }

    /// <summary>Zig-zag mapping: small magnitudes (of either sign) stay small.</summary>
    internal static ulong ZigZag(long value) => (ulong)((value << 1) ^ (value >> 63));

    internal static long UnZigZag(ulong value) => (long)(value >> 1) ^ -((long)(value & 1));
}

internal static class Rle
{
    internal static byte[] Encode(IReadOnlyList<int> values)
    {
        var sink = new List<byte>();
        Varint.Write(sink, (ulong)values.Count);
        int i = 0;
        while (i < values.Count)
        {
            int j = i;
            while (j < values.Count && values[j] == values[i]) j++;
            Varint.Write(sink, (ulong)(j - i));
            Varint.Write(sink, Varint.ZigZag(values[i]));
            i = j;
        }
        return sink.ToArray();
    }

    internal static int[] Decode(IReadOnlyList<byte> buffer)
    {
        int offset = 0;
        int count = checked((int)Varint.Read(buffer, ref offset));
        if (count < 0) throw new InvalidOperationException("negative element count");
        var values = new int[count];
        int position = 0;
        while (position < count)
        {
            int run = checked((int)Varint.Read(buffer, ref offset));
            if (run <= 0 || position + run > count) throw new InvalidOperationException("corrupt run length");
            int value = (int)Varint.UnZigZag(Varint.Read(buffer, ref offset));
            for (int k = 0; k < run; k++) values[position++] = value;
        }
        return values;
    }
}

internal sealed class Chunk
{
    internal readonly byte[] Payload;
    internal readonly bool Full;
    internal readonly int ChangedBlocks;

    internal Chunk(byte[] payload, bool full, int changedBlocks)
    {
        Payload = payload;
        Full = full;
        ChangedBlocks = changedBlocks;
    }

    internal int Bytes => Payload.Length;
}

/// <summary>Append-only checkpoint chain with full images and block-XOR deltas.</summary>
internal sealed class CheckpointStore
{
    internal const int BlockWords = 16;

    private readonly List<Chunk> _chunks = new List<Chunk>();
    private int[] _last = Array.Empty<int>();

    internal CheckpointStore(int words)
    {
        if (words < 0) throw new ArgumentOutOfRangeException(nameof(words));
        Words = words;
    }

    internal int Words { get; }
    internal int Blocks => (Words + BlockWords - 1) / BlockWords;
    internal int ChunkCount => _chunks.Count;
    internal long StoredBytes { get; private set; }
    internal long FullBytes { get; private set; }
    internal long DeltaBytes { get; private set; }
    internal int FullChunks { get; private set; }
    internal int DeltaChunks { get; private set; }

    internal Chunk ChunkAt(int index) => _chunks[index];

    internal Chunk Append(IReadOnlyList<int> state)
    {
        if (state.Count != Words) throw new ArgumentException("state length mismatch", nameof(state));
        int[] current = ToArray(state);
        Chunk chunk;
        if (_chunks.Count == 0)
        {
            chunk = EncodeFull(current);
        }
        else
        {
            bool[] changed = new bool[Blocks];
            int changedCount = 0;
            for (int b = 0; b < Blocks; b++)
            {
                int from = b * BlockWords;
                int to = Math.Min(Words, from + BlockWords);
                for (int i = from; i < to; i++)
                {
                    if (_last[i] != current[i]) { changed[b] = true; changedCount++; break; }
                }
            }
            chunk = changedCount * 2 > Blocks && Blocks > 0 ? EncodeFull(current) : EncodeDelta(current, changed, changedCount);
        }
        _chunks.Add(chunk);
        StoredBytes += chunk.Bytes;
        if (chunk.Full) { FullChunks++; FullBytes += chunk.Bytes; } else { DeltaChunks++; DeltaBytes += chunk.Bytes; }
        _last = current;
        return chunk;
    }

    internal int[] Restore(int index, out int deltasApplied, out int fullsDecoded)
    {
        if (index < 0 || index >= _chunks.Count) throw new ArgumentOutOfRangeException(nameof(index));
        int start = index;
        while (!_chunks[start].Full)
        {
            start--;
            if (start < 0) throw new InvalidOperationException("chain has no full image");
        }
        int[] state = DecodeFull(_chunks[start].Payload);
        deltasApplied = 0;
        fullsDecoded = 1;
        for (int i = start + 1; i <= index; i++)
        {
            state = ApplyDelta(state, _chunks[i]);
            deltasApplied++;
        }
        return state;
    }

    /// <summary>Test hook: applies a raw chunk to a state, so corrupt payloads can be exercised.</summary>
    internal int[] ApplyForTest(int[] state, Chunk chunk) => ApplyDelta(state, chunk);

    private Chunk EncodeFull(int[] state)
    {
        var sink = new List<byte> { 0 };
        Append(sink, Rle.Encode(state));
        return new Chunk(sink.ToArray(), true, Blocks);
    }

    private int[] DecodeFull(IReadOnlyList<byte> payload)
    {
        if (payload.Count < 1 || payload[0] != 0) throw new InvalidOperationException("not a full image");
        var body = new byte[payload.Count - 1];
        for (int i = 1; i < payload.Count; i++) body[i - 1] = payload[i];
        return Rle.Decode(body);
    }

    private Chunk EncodeDelta(int[] state, bool[] changed, int changedCount)
    {
        var sink = new List<byte> { 1 };
        Varint.Write(sink, (ulong)Blocks);
        var bitmap = new byte[(Blocks + 7) / 8];
        for (int b = 0; b < Blocks; b++) if (changed[b]) bitmap[b >> 3] |= (byte)(1 << (b & 7));
        Append(sink, bitmap);
        for (int b = 0; b < Blocks; b++)
        {
            if (!changed[b]) continue;
            int from = b * BlockWords;
            int to = Math.Min(Words, from + BlockWords);
            for (int i = from; i < to; i++) Varint.Write(sink, Varint.ZigZag(_last[i] ^ state[i]));
        }
        return new Chunk(sink.ToArray(), false, changedCount);
    }

    private int[] ApplyDelta(int[] previous, Chunk chunk)
    {
        IReadOnlyList<byte> payload = chunk.Payload;
        int offset = 1;
        if (payload.Count < 2 || payload[0] != 1) throw new InvalidOperationException("not a delta");
        int blocks = checked((int)Varint.Read(payload, ref offset));
        if (blocks != Blocks) throw new InvalidOperationException("block count mismatch");
        int bitmapBytes = (blocks + 7) / 8;
        if (offset + bitmapBytes > payload.Count) throw new InvalidOperationException("truncated bitmap");
        var state = (int[])previous.Clone();
        var bitmap = new bool[blocks];
        for (int b = 0; b < blocks; b++) bitmap[b] = ((payload[offset + (b >> 3)] >> (b & 7)) & 1) != 0;
        offset += bitmapBytes;
        for (int b = 0; b < blocks; b++)
        {
            if (!bitmap[b]) continue;
            int from = b * BlockWords;
            int to = Math.Min(Words, from + BlockWords);
            for (int i = from; i < to; i++)
            {
                int difference = (int)Varint.UnZigZag(Varint.Read(payload, ref offset));
                state[i] ^= difference;
            }
        }
        return state;
    }

    private static int[] ToArray(IReadOnlyList<int> values)
    {
        var array = new int[values.Count];
        for (int i = 0; i < array.Length; i++) array[i] = values[i];
        return array;
    }

    private static void Append(List<byte> sink, IReadOnlyList<byte> bytes)
    {
        for (int i = 0; i < bytes.Count; i++) sink.Add(bytes[i]);
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- varint and zig-zag round-trips over every interesting boundary ------------------------
        {
            ulong[] boundaries = { 0, 1, 127, 128, 16383, 16384, 1UL << 21, (1UL << 21) - 1, 1UL << 28,
                1UL << 31, 1UL << 35, ulong.MaxValue >> 1, ulong.MaxValue };
            foreach (ulong value in boundaries)
            {
                var sink = new List<byte>();
                Varint.Write(sink, value);
                int offset = 0;
                Check(Varint.Read(sink, ref offset) == value && offset == sink.Count, "varint round-trip " + value);
            }
            long[] signed = { 0, 1, -1, 127, -128, int.MaxValue, int.MinValue, long.MaxValue / 3, -(long.MaxValue / 3) };
            foreach (long value in signed)
                Check(Varint.UnZigZag(Varint.ZigZag(value)) == value, "zig-zag round-trip " + value);
            Check(Varint.ZigZag(-1) == 1 && Varint.ZigZag(1) == 2, "zig-zag interleaves signs");
        }

        // ---- RLE round-trips -----------------------------------------------------------------------
        {
            Check(Rle.Decode(Rle.Encode(Array.Empty<int>())).Length == 0, "the empty array encodes to a count only");
            Check(SameArray(Rle.Decode(Rle.Encode(new[] { 7 })), new[] { 7 }), "single element round-trip");
            var constant = new int[1000];
            for (int i = 0; i < constant.Length; i++) constant[i] = -3;
            byte[] constantBytes = Rle.Encode(constant);
            Check(SameArray(Rle.Decode(constantBytes), constant), "a constant array round-trips");
            Check(constantBytes.Length < 8, "a constant array compresses to a handful of bytes (" + constantBytes.Length + ")");
            var distinct = new int[64];
            for (int i = 0; i < distinct.Length; i++) distinct[i] = i * 37 - 20;
            Check(SameArray(Rle.Decode(Rle.Encode(distinct)), distinct), "an all-distinct array round-trips");
        }

        // ---- a single full image and the header-only delta of an unchanged state --------------------
        {
            var store = new CheckpointStore(500);       // 32 blocks, the last one short
            var state = new int[500];
            var rng = new SplitMix64(0xC0DE721UL);
            for (int i = 0; i < 360; i++) state[i] = rng.NextInt(9) - 4;
            Chunk first = store.Append(state);
            Check(first.Full && first.Bytes > 0, "the first checkpoint is a full image");
            var restored = store.Restore(0, out int deltas, out int fulls);
            Check(SameArray(restored, state) && deltas == 0 && fulls == 1, "the first image restores directly");

            Chunk unchanged = store.Append(state);
            Check(!unchanged.Full && unchanged.ChangedBlocks == 0, "an unchanged checkpoint is a header-only delta");
            Check(unchanged.Bytes <= 8, "the unchanged delta costs a type byte, a count and a bitmap");
            Check(SameArray(store.Restore(1, out _, out _), state), "an unchanged delta restores the same state");

            state[499] = 12345;                        // last word of the short trailing block
            Chunk tail = store.Append(state);
            Check(!tail.Full && tail.ChangedBlocks == 1, "touching the short trailing block marks one block");
            Check(SameArray(store.Restore(2, out _, out _), state), "the trailing block restores bit-exactly");

            state[0] ^= unchecked((int)0xFFFFFFFF);    // negative values survive XOR deltas
            store.Append(state);
            Check(SameArray(store.Restore(3, out _, out _), state), "negative words survive an XOR delta");
        }

        // ---- the sparse workload: 40 checkpoints, every restore cross-checked ----------------------
        const int words = 512, checkpoints = 40;
        var sparseStore = new CheckpointStore(words);
        var reference = new List<int[]>();
        var fullImageBytes = new long[checkpoints];
        {
            var rng = new SplitMix64(0x51A5E721UL);
            var state = new int[words];
            for (int i = 0; i < 96; i++) state[i] = rng.NextInt(200) - 100;   // a small frozen-ish prefix
            for (int step = 0; step < checkpoints; step++)
            {
                int block = rng.NextInt(words / CheckpointStore.BlockWords);
                for (int k = 0; k < CheckpointStore.BlockWords; k++) state[block * CheckpointStore.BlockWords + k] = rng.NextInt(50) - 25;
                if (step % 3 == 0) state[rng.NextInt(words)] = rng.NextInt(1000);
                reference.Add((int[])state.Clone());
                sparseStore.Append(state);
                var probe = new CheckpointStore(words);
                fullImageBytes[step] = probe.Append(state).Bytes;
            }
        }
        {
            bool everyRestoreExact = true;
            for (int i = 0; i < checkpoints; i++)
                if (!SameArray(sparseStore.Restore(i, out _, out _), reference[i])) everyRestoreExact = false;
            Check(everyRestoreExact, "every checkpoint in the chain restores bit-for-bit from its payloads");
            Check(sparseStore.DeltaChunks == checkpoints - 1 && sparseStore.FullChunks == 1,
                "the sparse workload writes one image and 39 deltas (" + sparseStore.DeltaChunks + " deltas)");
            long allFull = 0;
            foreach (long bytes in fullImageBytes) allFull += bytes;
            Console.WriteLine("sparseChainBytes=" + sparseStore.StoredBytes + " fullImagesBytes=" + allFull
                + " ratio=" + ((double)allFull / sparseStore.StoredBytes).ToString("G4")
                + " avgDeltaBytes=" + (sparseStore.DeltaBytes / (checkpoints - 1)));
            Check(sparseStore.StoredBytes * 5 < allFull,
                "incremental checkpointing writes under a fifth of the equivalent full images ("
                + sparseStore.StoredBytes + " vs " + allFull + ")");
            Check(sparseStore.StoredBytes == sparseStore.FullBytes + sparseStore.DeltaBytes,
                "the byte counters add up");

            int[] restoredAgain = sparseStore.Restore(17, out int deltas, out int fulls);
            Check(SameArray(restoredAgain, reference[17]) && deltas == 17 && fulls == 1,
                "restoring index 17 replays 17 deltas on top of the single image");
            Check(sparseStore.StoredBytes == sparseStore.FullBytes + sparseStore.DeltaBytes && sparseStore.ChunkCount == checkpoints,
                "restoring never mutates the store");
        }

        // ---- the dense workload falls back to full images ------------------------------------------
        {
            var denseStore = new CheckpointStore(words);
            var rng = new SplitMix64(0xDE5E721UL);
            var state = new int[words];
            for (int step = 0; step < 8; step++)
            {
                for (int i = 0; i < words; i++) state[i] = rng.NextInt(1000) - 500;
                denseStore.Append(state);
            }
            Check(denseStore.FullChunks == 8 && denseStore.DeltaChunks == 0,
                "a workload that rewrites every word never writes a delta");
            Check(denseStore.StoredBytes > sparseStore.StoredBytes,
                "the dense workload costs more bytes than the sparse one");

            var mixed = new CheckpointStore(words);
            var state2 = new int[words];
            mixed.Append(state2);
            for (int i = 0; i < 17 * CheckpointStore.BlockWords; i++) state2[i] = 3;   // 17 of 32 blocks
            Chunk heavy = mixed.Append(state2);
            Check(heavy.Full, "more than half the blocks changing triggers a full image");
            Check(SameArray(mixed.Restore(1, out _, out _), state2), "the fallback image restores correctly");
        }

        // ---- corrupt payloads are rejected, and encoding is deterministic --------------------------
        {
            var store = new CheckpointStore(64);
            var rng = new SplitMix64(0xBAD721UL);
            var state = new int[64];
            for (int i = 0; i < 64; i++) state[i] = rng.NextInt(20);
            store.Append(state);
            state[5] = 999;
            store.Append(state);
            Check(!store.ChunkAt(1).Full, "a one-word change is written as a delta");

            Chunk delta = store.ChunkAt(1);
            var truncated = new byte[delta.Bytes - 2];
            Array.Copy(delta.Payload, truncated, truncated.Length);
            bool threw = false;
            try { store.ApplyForTest(state, new Chunk(truncated, false, 1)); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "a truncated delta payload is rejected instead of decoding garbage");

            bool bogusRejected = false;
            try { Rle.Decode(new byte[] { 5, 3 }); }        // promises 5 elements and then the stream ends
            catch (InvalidOperationException) { bogusRejected = true; }
            Check(bogusRejected, "a corrupt RLE stream whose count is not filled is rejected");

            bool overflowRejected = false;
            try { Rle.Decode(new byte[] { 3, 7, 0 }); }     // a run longer than the element count
            catch (InvalidOperationException) { overflowRejected = true; }
            Check(overflowRejected, "a run longer than the declared element count is rejected");

            var a = new CheckpointStore(words);
            var b = new CheckpointStore(words);
            for (int i = 0; i < 3; i++)
            {
                var s = reference[checkpoints - 1 - i];
                a.Append(s);
                b.Append(s);
            }
            bool identicalPayloads = true;
            for (int i = 0; i < a.ChunkCount; i++)
                if (!SameArray(a.ChunkAt(i).Payload, b.ChunkAt(i).Payload)) identicalPayloads = false;
            Check(identicalPayloads, "the same states produce byte-identical payloads");

            var empty = new CheckpointStore(0);
            empty.Append(Array.Empty<int>());
            Check(empty.Restore(0, out _, out _).Length == 0, "a zero-word checkpoint round-trips");
        }

        Console.WriteLine("PASS");
    }

    private static bool SameArray(int[] a, int[] b)
    {
        if (a == null || b == null || a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static bool SameArray(byte[] a, byte[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
