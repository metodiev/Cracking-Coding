// Task 989 - Implement offline mode packaging so learners can download problem sets + tests.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Statements and tests go into a content-addressed store (identical bytes stored once, so logical and physical
// size differ) and become a sorted path -> (size, hash) manifest plus a fixed 64-byte chunk plan. A downloader
// pulls chunks over a simulated link and keeps one only after its hash verifies: a corrupted chunk is
// re-fetched, a transport hiccup consumes an attempt, and losing the connection aborts the session with every
// verified chunk retained, so the next session resumes without re-downloading anything; exhausting attempts
// fails the download and installs nothing. Verified: dedup accounting, manifest determinism against an
// independent rebuild, the 0/64/65-byte chunk boundaries, byte-exact reassembly with content-addressed hashes
// intact (oracle), recovery from a scheduled failure and corruption, atomic failure, resume transfer volume,
// the quota refusal and the single-byte / empty boundaries.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static string Error(Action action)
    {
        try { action(); return ""; }
        catch (Exception ex) { return ex.Message; }
    }
    private static ulong HashBytes(byte[] data)
    {
        unchecked
        {
            ulong h = 14695981039346656037UL;
            foreach (byte b in data) { h ^= b; h *= 1099511628211UL; }
            return h;
        }
    }
    private static ulong HashText(string text) => HashBytes(Encoding.UTF8.GetBytes(text));
    private static string Hex(ulong value) => value.ToString("x16", CultureInfo.InvariantCulture);
    /** Deterministic byte source (splitmix64): no unseeded randomness anywhere. */
    private static byte[] Bytes(int length, ulong seed)
    {
        var data = new byte[length];
        ulong state = seed;
        for (int i = 0; i < length; i++)
        {
            unchecked
            {
                state += 0x9E3779B97F4A7C15UL;
                state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9UL) ^ (state >> 27);
                data[i] = (byte)(state >> 24);
            }
        }
        return data;
    }
    /** Content-addressed blob store: identical bytes are stored once, so physical < logical size. */
    private sealed class Store
    {
        private readonly Dictionary<ulong, byte[]> _blobs = new Dictionary<ulong, byte[]>();
        public int PhysicalBytes;
        public ulong Put(byte[] data)
        {
            ulong hash = HashBytes(data);
            if (!_blobs.ContainsKey(hash)) { _blobs[hash] = (byte[])data.Clone(); PhysicalBytes += data.Length; }
            return hash;
        }
    }
    private sealed class Chunk
    {
        public readonly string Path;
        public readonly int Index, Offset, Length;
        public readonly ulong Hash;
        public Chunk(string path, int index, int offset, int length, ulong hash)
        { Path = path; Index = index; Offset = offset; Length = length; Hash = hash; }
        public string Id => Path + "#" + Index.ToString(CultureInfo.InvariantCulture);
    }
    private sealed class Bundle
    {
        public readonly List<string> Paths = new List<string>();
        public readonly Dictionary<string, byte[]> Data = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        public readonly Dictionary<string, ulong> Hash = new Dictionary<string, ulong>(StringComparer.Ordinal);
        public readonly List<Chunk> Chunks = new List<Chunk>();
        public int LogicalBytes;
        public string ManifestText = ""; public ulong ManifestHash;
    }
    private static Bundle Build(Store store, List<(string Path, byte[] Data)> files, int chunkSize, int quota)
    {
        var bundle = new Bundle();
        var ordered = new List<(string Path, byte[] Data)>(files);
        ordered.Sort((x, y) => string.CompareOrdinal(x.Path, y.Path));
        foreach ((string path, byte[] data) in ordered)
        {
            if (bundle.Data.ContainsKey(path)) throw new Exception("duplicate-path:" + path);
            bundle.Paths.Add(path);
            bundle.Data[path] = data;
            bundle.Hash[path] = store.Put(data);
            bundle.LogicalBytes += data.Length;
        }
        if (bundle.LogicalBytes > quota) throw new Exception("quota-exceeded:logical=" + bundle.LogicalBytes + ":limit=" + quota);
        var manifest = new StringBuilder();
        foreach (string path in bundle.Paths)
        {
            byte[] data = bundle.Data[path];
            manifest.Append(path).Append('|').Append(data.Length).Append('|').Append(Hex(bundle.Hash[path])).Append('\n');
            for (int offset = 0, index = 0; offset < data.Length; offset += chunkSize, index++)
            {
                int length = Math.Min(chunkSize, data.Length - offset);
                var slice = new byte[length];
                Array.Copy(data, offset, slice, 0, length);
                bundle.Chunks.Add(new Chunk(path, index, offset, length, HashBytes(slice)));
            }
        }
        bundle.ManifestText = manifest.ToString(); bundle.ManifestHash = HashText(bundle.ManifestText);
        return bundle;
    }
    /** Simulated link with a deterministic fault schedule: fail-first, corrupt-first, or go offline. */
    private sealed class Link
    {
        private readonly HashSet<string> _failFirst = new HashSet<string>(StringComparer.Ordinal);
        private readonly HashSet<string> _corruptFirst = new HashSet<string>(StringComparer.Ordinal);
        private readonly HashSet<string> _seen = new HashSet<string>(StringComparer.Ordinal);
        public readonly List<string> Attempts = new List<string>();
        public int BytesTransferred, StopAfter = int.MaxValue;
        public void FailFirst(string id) { _failFirst.Add(id); }
        public void CorruptFirst(string id) { _corruptFirst.Add(id); }
        public byte[] Request(string chunkId, byte[] payload)
        {
            if (Attempts.Count >= StopAfter) throw new Exception("network-offline");
            Attempts.Add(chunkId);
            BytesTransferred += payload.Length;
            bool first = _seen.Add(chunkId);
            if (first && _failFirst.Contains(chunkId)) throw new Exception("transport-error:" + chunkId);
            var copy = (byte[])payload.Clone();
            if (first && _corruptFirst.Contains(chunkId)) copy[0] ^= 0xFF;
            return copy;
        }
    }    private sealed class Downloader
    {
        private readonly Bundle _bundle;
        private readonly Link _link;
        private readonly int _maxAttempts;
        public readonly Dictionary<string, byte[]> Verified = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        public string Failure = "";
        public bool Complete;
        public Downloader(Bundle bundle, Link link, int maxAttempts)
        { _bundle = bundle; _link = link; _maxAttempts = maxAttempts; }
        public bool Run()
        {
            Complete = false; Failure = "";
            foreach (Chunk chunk in _bundle.Chunks)
            {
                if (Verified.ContainsKey(chunk.Id)) continue;
                byte[] payload = Slice(_bundle, chunk);
                int tries = 0;
                bool stored = false;
                while (tries < _maxAttempts && !stored)
                {
                    tries++;
                    byte[] received;
                    try { received = _link.Request(chunk.Id, payload); }
                    catch (Exception ex)
                    {
                        if (ex.Message == "network-offline") { Failure = "network-offline"; return false; }
                        if (tries >= _maxAttempts) { Failure = "transport-error-exhausted:" + chunk.Id; return false; }
                        continue;
                    }
                    if (HashBytes(received) == chunk.Hash) { Verified[chunk.Id] = received; stored = true; }
                    else if (tries >= _maxAttempts) { Failure = "chunk-verification-failed:" + chunk.Id; return false; }
                }
                if (!stored) return false;
            }
            Complete = true;
            return true;
        }
        public Dictionary<string, byte[]> Files()
        {
            var result = new Dictionary<string, byte[]>(StringComparer.Ordinal);
            foreach (string path in _bundle.Paths)
            {
                var bytes = new byte[_bundle.Data[path].Length];
                foreach (Chunk chunk in _bundle.Chunks)
                    if (chunk.Path == path && Verified.TryGetValue(chunk.Id, out byte[] data))
                        Array.Copy(data, 0, bytes, chunk.Offset, chunk.Length);
                result[path] = bytes;
            }
            return result;
        }    }    private static byte[] Slice(Bundle bundle, Chunk chunk)
    {
        var slice = new byte[chunk.Length];
        Array.Copy(bundle.Data[chunk.Path], chunk.Offset, slice, 0, chunk.Length);
        return slice;
    }
    private static List<(string Path, byte[] Data)> Files(params (string, byte[])[] items) => new List<(string, byte[])>(items);

    public static void Main()
    {
        byte[] alpha = Bytes(130, 7), beta = Bytes(10, 11), gamma = Bytes(64, 13);
        var store = new Store();
        var dedupStore = new Store();
        Bundle dedup = Build(dedupStore, Files(("x.txt", Bytes(200, 99)), ("y.txt", Bytes(200, 99))), 64, 4096);
        Bundle bundle = Build(store, Files(("b.txt", beta), ("c.txt", gamma), ("a.txt", alpha)), 64, 4096);
        string expected = "a.txt|130|" + Hex(HashBytes(alpha)) + "\nb.txt|10|" + Hex(HashBytes(beta))
                          + "\nc.txt|64|" + Hex(HashBytes(gamma)) + "\n";
        Bundle permuted = Build(new Store(), Files(("a.txt", alpha), ("c.txt", gamma), ("b.txt", beta)), 64, 4096);
        // Oracle: the manifest is rebuilt from literal metadata, not from the packager's loop.
        Check(dedup.LogicalBytes == 400 && dedupStore.PhysicalBytes == 200 && dedup.Hash["x.txt"] == dedup.Hash["y.txt"]
              && bundle.ManifestText == expected && permuted.ManifestText == expected && bundle.ManifestHash == HashText(expected)
              && bundle.Chunks.Count == 5 && Build(store, Files(("one", Bytes(64, 1))), 64, 4096).Chunks.Count == 1
              && Build(store, Files(("two", Bytes(65, 2))), 64, 4096).Chunks.Count == 2
              && Build(store, Files(("two", Bytes(65, 2))), 64, 4096).Chunks[1].Length == 1
              && Build(store, Files(("zero", new byte[0])), 64, 4096).Chunks.Count == 0,
              "identical blobs share one physical copy, the manifest matches an independent rebuild (oracle) and the"
              + " 0 / 64 / 65 byte chunk boundaries hold");
        var cleanLink = new Link();
        var happy = new Downloader(bundle, cleanLink, 3);
        bool happyOk = happy.Run();
        Dictionary<string, byte[]> got = happy.Files();
        Check(happyOk && happy.Complete && cleanLink.Attempts.Count == 5 && got["b.txt"].Length == beta.Length
              && HashBytes(got["a.txt"]) == HashBytes(alpha) && HashBytes(got["b.txt"]) == HashBytes(beta)
              && HashBytes(got["c.txt"]) == HashBytes(gamma) && HashBytes(got["a.txt"]) == bundle.Hash["a.txt"],
              "reassembly is byte-exact with content addressing intact and each chunk is fetched once (oracle)");
        var flakyLink = new Link();
        flakyLink.FailFirst("a.txt#1");
        flakyLink.CorruptFirst("a.txt#2");
        var flaky = new Downloader(bundle, flakyLink, 3);
        int retries = 0;
        bool flakyOk = flaky.Run();
        foreach (string attempt in flakyLink.Attempts) if (attempt == "a.txt#1" || attempt == "a.txt#2") retries++;
        var brokenLink = new Link();
        brokenLink.CorruptFirst("a.txt#0");
        var broken = new Downloader(bundle, brokenLink, 1);
        Check(flakyOk && flaky.Complete && HashBytes(flaky.Files()["a.txt"]) == HashBytes(alpha) && retries == 4
              && flakyLink.Attempts.Count == 7 && !broken.Run() && broken.Failure == "chunk-verification-failed:a.txt#0"
              && broken.Verified.Count == 0 && !broken.Complete,
              "a dropped transfer and a corrupted chunk are re-fetched, while exhausted attempts fail the download"
              + " and install nothing");
        var resumeLink = new Link();
        resumeLink.StopAfter = 2;
        var downloader = new Downloader(bundle, resumeLink, 3);
        bool firstRun = downloader.Run();
        int transferredFirst = resumeLink.BytesTransferred, verifiedFirst = downloader.Verified.Count;
        resumeLink.StopAfter = int.MaxValue;
        bool secondRun = downloader.Run();
        Check(!firstRun && verifiedFirst == 2 && secondRun && downloader.Complete && resumeLink.Attempts.Count == 5
              && resumeLink.BytesTransferred == transferredFirst + 2 + beta.Length + gamma.Length
              && HashBytes(downloader.Files()["c.txt"]) == HashBytes(gamma),
              "resume keeps verified chunks and transfers only what is still missing");
        Bundle oneByte = Build(store, Files(("tiny", Bytes(1, 5))), 64, 4096);
        var tinyLink = new Link();
        var tinyDownload = new Downloader(oneByte, tinyLink, 1);
        var emptyRun = new Downloader(Build(store, Files(), 64, 4096), new Link(), 1);
        Check(oneByte.Chunks.Count == 1 && oneByte.Chunks[0].Length == 1 && tinyDownload.Run() && tinyLink.Attempts.Count == 1
              && tinyDownload.Files()["tiny"].Length == 1 && HashBytes(tinyDownload.Files()["tiny"]) == oneByte.Hash["tiny"],
              "single-byte file boundary: one chunk of one byte, verified against the manifest hash");
        Check(emptyRun.Run() && emptyRun.Files().Count == 0 && emptyRun.Failure == "" && emptyRun.Complete
              && Error(() => Build(new Store(), Files(("big", Bytes(300, 3))), 64, 128)) == "quota-exceeded:logical=300:limit=128",
              "an empty bundle downloads nothing and a bundle over quota is refused before any chunk exists");
        Console.WriteLine("paths: " + bundle.Paths.Count + ", chunks: " + bundle.Chunks.Count + ", logical: "
                          + bundle.LogicalBytes + ", physical: " + store.PhysicalBytes + ", manifest: " + Hex(bundle.ManifestHash));
        Console.WriteLine("PASS");    }
}
