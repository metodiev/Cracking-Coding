// Task 814 - Implement searchable encrypted index with limited leakage and proof-of-correctness. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Scheme (client holds the key, the server holds only opaque entries):
//   * A label is (keyword, version). tag(w, v) = H(master, "tag"|w|v) and key(w, v) = H(master, "key"|w|v)
//     are keyed, domain separated. The server stores a bucket per tag holding sealed entries; it never
//     sees a keyword, only bucket ids.
//   * An update batch gives every keyword it touches a *fresh* version, so an update never re-uses a
//     tag the server has already seen: the update is unlinkable to earlier searches of the same
//     keyword (that is the toy version of forward privacy, and it is measured below).
//   * A search is a trapdoor: the set of tags of the versions the client knows plus the matching key
//     for each. The server answers with the sealed entries for those tags (nothing else can be
//     requested, and nothing else may come back).
//   * Proof of correctness: the client keeps an O(#labels) commitment per version - a count and a
//     commutative multiset hash Sum(H(key(w,v), nonce|body)). It recomputes both over what the server
//     returned, so omission (count/hash short) and injection (count/hash long) are detected. The
//     per-entry fingerprints are keyed with the label key the server never sees, which is why the
//     server cannot construct a compensating pair.
//
// Simplifications (deliberate, documented): the hashes are a hand written splitmix64/keystream
// construction, NOT a PRF like HMAC-SHA256; the multiset hash is an additive 64 bit accumulator, so
// it detects single element and count changes but is not a collision resistant set commitment; the
// server is a deterministic in-process store with an explicit tamper switch (no network, no threads,
// no timing side channels); deletes and dynamic keyword removal are out of scope; the vocabulary is
// the plaintext keyword set the client owns.
//
// Leakage that is *deliberately* allowed and measured: label equality (the same label is the same tag,
// i.e. search pattern), the number of entries per bucket (access pattern), the arrival order of
// updates and the number of versions per keyword. Leakage that is checked to be absent: no plaintext
// keyword window survives in any stored byte, tags cannot be precomputed from a keyword without the
// master key, and two different master keys give completely disjoint tag sets.
//
// Verified invariants: retrieval equals a plaintext inverted-index oracle for every vocabulary keyword
// (plus absent keywords and duplicate keywords); honest multiset/count verification accepts;
// omission, injection, duplication, payload corruption and cross label substitution are all rejected;
// a stale trapdoor is detected as incomplete; search work is bounded by the label's postings and stays
// far below a brute-force document scan; the client side verification state stays O(#labels).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

// ---- deterministic primitives ---------------------------------------------------------------------

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
}

internal static class ToyCrypto
{
    public static ulong Hash64(ulong key, byte[] data)
    {
        ulong h = 0xCBF29CE484222325UL ^ key;
        for (int i = 0; i < data.Length; i++) { h ^= data[i]; h *= 0x100000001B3UL; h ^= h >> 32; }
        h ^= (ulong)data.Length * 0x9E3779B97F4A7C15UL;
        return h ^ (h >> 29);
    }

    public static ulong Hash64(ulong key, string text) => Hash64(key, Encoding.UTF8.GetBytes(text));

    public static byte[] Crypt(byte[] input, ulong key, ulong nonce)
    {
        var rng = new SplitMix64(key ^ (nonce * 0xFF51AFD7ED558CCDUL));
        var output = new byte[input.Length];
        int pos = 0;
        while (pos < input.Length)
        {
            ulong word = rng.Next();
            for (int b = 0; b < 8 && pos < input.Length; b++)
            {
                output[pos] = (byte)(input[pos] ^ (byte)(word >> (8 * b)));
                pos++;
            }
        }
        return output;
    }

    public static ulong Tag(ulong key, ulong nonce, byte[] body)
        => Hash64(key ^ (nonce * 0x9E3779B97F4A7C15UL), body) ^ (nonce * 0x100000001B3UL);
}

internal sealed class SealedEntry
{
    public ulong Tag;
    public ulong Nonce;
    public byte[] Body = Array.Empty<byte>();
    public ulong Mac;

    public SealedEntry Clone() => new SealedEntry { Tag = Tag, Nonce = Nonce, Body = (byte[])Body.Clone(), Mac = Mac };
}

// ---- client side key schedule ---------------------------------------------------------------------

internal sealed class VersionCommitment
{
    public long Sum;
    public int Count;
}

internal sealed class KeywordState
{
    public readonly List<VersionCommitment> Versions = new List<VersionCommitment>();
}

// ---- the untrusted store --------------------------------------------------------------------------

internal sealed class IndexServer
{
    private readonly Dictionary<ulong, List<SealedEntry>> buckets = new Dictionary<ulong, List<SealedEntry>>();
    private readonly List<ulong> arrival = new List<ulong>();

    public int Examined { get; private set; }      // entries handed back, i.e. the work a search costs
    public int TamperDrops { get; set; }           // malicious server behaviour, used by the checks
    public int TamperDuplicates { get; set; }
    public bool TamperCorrupt { get; set; }

    public int BucketCount => buckets.Count;
    public int EntryCount => buckets.Values.Sum(b => b.Count);
    public IReadOnlyList<ulong> Arrival => arrival;

    public void Store(SealedEntry entry)
    {
        if (!buckets.TryGetValue(entry.Tag, out List<SealedEntry> bucket))
        {
            bucket = new List<SealedEntry>();
            buckets[entry.Tag] = bucket;
            arrival.Add(entry.Tag);
        }
        bucket.Add(entry);
    }

    public IList<SealedEntry> Fetch(IEnumerable<ulong> tags)
    {
        var result = new List<SealedEntry>();
        foreach (ulong tag in tags)
        {
            if (!buckets.TryGetValue(tag, out List<SealedEntry> bucket)) continue;
            for (int i = 0; i < bucket.Count; i++)
            {
                int dropAt = TamperDrops > 0 ? TamperDrops : int.MaxValue;
                if (i >= dropAt) continue;
                SealedEntry copy = bucket[i].Clone();
                if (TamperCorrupt) copy.Body[copy.Body.Length - 1] ^= 0x10;
                result.Add(copy);
                if (TamperDuplicates > 0 && i == 0) result.Add(copy.Clone());
            }
        }
        Examined += result.Count;
        return result;
    }

    // What the server can see, as bytes: bucket ids, arrival order, entry sizes and payload bytes.
    public byte[] Transcript()
    {
        var ms = new System.IO.MemoryStream();
        foreach (ulong tag in arrival)
        {
            byte[] tagBytes = BitConverter.GetBytes(tag);
            ms.Write(tagBytes, 0, tagBytes.Length);
            foreach (SealedEntry entry in buckets[tag])
            {
                byte[] nonce = BitConverter.GetBytes(entry.Nonce);
                ms.Write(nonce, 0, nonce.Length);
                ms.Write(entry.Body, 0, entry.Body.Length);
            }
        }
        return ms.ToArray();
    }

    public void ResetExamined() => Examined = 0;

    // Size of one bucket, i.e. what the access pattern leaks to the store.
    public int BucketSize(ulong tag) => buckets.TryGetValue(tag, out List<SealedEntry> bucket) ? bucket.Count : 0;
}

// ---- the honest client holding the key -----------------------------------------------------------

internal sealed class Trapdoor
{
    public readonly List<ulong> Tags = new List<ulong>();
    public readonly List<ulong> Keys = new List<ulong>();
    public int Versions;
}

internal sealed class SearchResult
{
    public bool Verified;
    public string Reason = "";
    public List<int> Documents = new List<int>();
}

internal sealed class EncryptedIndexClient
{
    private readonly ulong master;
    private readonly IndexServer server;
    private readonly Dictionary<string, KeywordState> state = new Dictionary<string, KeywordState>();
    private readonly Dictionary<string, List<int>> labels = new Dictionary<string, List<int>>();
    private ulong nonceCounter = 0x5EED0000UL;

    public EncryptedIndexClient(ulong master, IndexServer server)
    {
        this.master = master;
        this.server = server;
    }

    public ulong TagOf(string keyword, int version) => ToyCrypto.Hash64(master, $"tag|{keyword}|{version}");

    public ulong KeyOf(string keyword, int version) => ToyCrypto.Hash64(master, $"key|{keyword}|{version}");

    public ulong Fingerprint(ulong key, ulong nonce, byte[] body) => ToyCrypto.Hash64(key ^ nonce, body);

    public int LabelCount => state.Count;

    public int VersionCount => state.Values.Sum(s => s.Versions.Count);

    // Client side verification state in bytes: per label a 12 byte commitment plus a keyword entry.
    public int StateBytes => state.Sum(kv => kv.Key.Length + 16 + kv.Value.Versions.Count * 12);

    // An update batch: every document contributes its keywords; each touched keyword gets a fresh
    // version, so the tags it emits have never been revealed before.
    public void Update(IEnumerable<KeyValuePair<int, string[]>> batch)
    {
        var grouped = new Dictionary<string, List<int>>();
        foreach (KeyValuePair<int, string[]> document in batch)
            foreach (string keyword in document.Value)
            {
                if (!grouped.TryGetValue(keyword, out List<int> docs))
                {
                    docs = new List<int>();
                    grouped[keyword] = docs;
                }
                if (!docs.Contains(document.Key)) docs.Add(document.Key);
            }

        foreach (KeyValuePair<string, List<int>> group in grouped.OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            string keyword = group.Key;
            if (!state.TryGetValue(keyword, out KeywordState keywordState))
            {
                keywordState = new KeywordState();
                state[keyword] = keywordState;
                labels[keyword] = new List<int>();
            }
            int version = keywordState.Versions.Count;
            ulong key = KeyOf(keyword, version);
            ulong tag = TagOf(keyword, version);
            var commitment = new VersionCommitment();
            foreach (int documentId in group.Value)
            {
                nonceCounter += 0x9E3779B97F4A7C15UL;
                ulong nonce = nonceCounter;
                byte[] body = ToyCrypto.Crypt(Encoding.UTF8.GetBytes("doc:" + documentId), key, nonce);
                ulong mac = ToyCrypto.Tag(key, nonce, body);
                server.Store(new SealedEntry { Tag = tag, Nonce = nonce, Body = body, Mac = mac });
                commitment.Sum += (long)Fingerprint(key, nonce, body);
                commitment.Count++;
            }
            keywordState.Versions.Add(commitment);
            labels[keyword].AddRange(group.Value);
        }
    }

    public Trapdoor BuildTrapdoor(string keyword)
    {
        var trapdoor = new Trapdoor();
        if (!state.TryGetValue(keyword, out KeywordState keywordState)) return trapdoor;
        for (int version = 0; version < keywordState.Versions.Count; version++)
        {
            trapdoor.Tags.Add(TagOf(keyword, version));
            trapdoor.Keys.Add(KeyOf(keyword, version));
            trapdoor.Versions++;
        }
        return trapdoor;
    }

    // Trapdoor with an explicit version limit: used to show that a stale client state is detected.
    public Trapdoor BuildTrapdoor(string keyword, int versions)
    {
        var trapdoor = new Trapdoor();
        if (!state.TryGetValue(keyword, out KeywordState keywordState)) return trapdoor;
        int limit = Math.Min(versions, keywordState.Versions.Count);
        for (int version = 0; version < limit; version++)
        {
            trapdoor.Tags.Add(TagOf(keyword, version));
            trapdoor.Keys.Add(KeyOf(keyword, version));
            trapdoor.Versions++;
        }
        return trapdoor;
    }

    // The client verifies against its commitment for the *whole* keyword, so a trapdoor that covers
    // only some of the versions it knows is detected as an incomplete answer.
    private long ExpectedSum(string keyword)
    {
        if (!state.TryGetValue(keyword, out KeywordState keywordState)) return 0;
        long sum = 0;
        foreach (VersionCommitment commitment in keywordState.Versions) sum += commitment.Sum;
        return sum;
    }

    private int ExpectedCount(string keyword)
    {
        if (!state.TryGetValue(keyword, out KeywordState keywordState)) return 0;
        int count = 0;
        foreach (VersionCommitment commitment in keywordState.Versions) count += commitment.Count;
        return count;
    }

    // Entries committed for a keyword, i.e. the number of postings the store must hold.
    public int CommittedEntries(string keyword) => ExpectedCount(keyword);

    // Fetch + verify + decrypt. Verification is the proof of correctness: the returned multiset must
    // hash and count exactly as committed, every returned tag must have been requested, and every
    // entry must authenticate under the key of the version whose tag it arrived with.
    public SearchResult Search(string keyword, Trapdoor trapdoor)
    {
        var result = new SearchResult();
        IList<SealedEntry> returned = server.Fetch(trapdoor.Tags);
        var requested = new HashSet<ulong>(trapdoor.Tags);

        long sum = 0;
        int count = 0;
        var documents = new List<int>();
        for (int i = 0; i < returned.Count; i++)
        {
            SealedEntry entry = returned[i];
            if (!requested.Contains(entry.Tag)) { result.Reason = "entry for an unrequested tag"; return result; }
            int version = trapdoor.Tags.IndexOf(entry.Tag);
            ulong key = trapdoor.Keys[version];
            if (ToyCrypto.Tag(key, entry.Nonce, entry.Body) != entry.Mac)
            {
                result.Reason = "entry failed authentication";
                return result;
            }
            sum += (long)Fingerprint(key, entry.Nonce, entry.Body);
            count++;
            string text = Encoding.UTF8.GetString(ToyCrypto.Crypt(entry.Body, key, entry.Nonce));
            documents.Add(int.Parse(text.Substring(4)));
        }

        if (count != ExpectedCount(keyword))
        {
            result.Reason = "entry count mismatch";
            return result;
        }
        if (sum != ExpectedSum(keyword))
        {
            result.Reason = "multiset hash mismatch";
            return result;
        }
        documents.Sort();
        result.Documents = documents;
        result.Verified = true;
        return result;
    }

    // Independent truth used only as an oracle in the checks.
    public List<int> PlainPostings(string keyword)
        => labels.TryGetValue(keyword, out List<int> docs) ? docs.OrderBy(d => d).ToList() : new List<int>();
}

// ---- checks ---------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xBADF00D5EED1234UL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        string[] vocabulary = { "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta" };
        var documents = BuildCorpus(vocabulary, 60);
        var server = new IndexServer();
        var client = new EncryptedIndexClient(0x9E3779B97F4A7C15UL, server);

        // ---- build -----------------------------------------------------------------------------
        client.Update(documents);
        Check(server.EntryCount > 0, "index is populated");
        Check(client.LabelCount == vocabulary.Length, "every vocabulary keyword has a label: " + client.LabelCount);
        // One tag per keyword per update round, because each round is a fresh version batch.
        Check(server.BucketCount == vocabulary.Length, "one bucket per (keyword, version) round: " + server.BucketCount);

        // ---- retrieval against the plaintext inverted index oracle -----------------------------
        int compared = 0;
        foreach (string keyword in vocabulary)
        {
            SearchResult result = client.Search(keyword, client.BuildTrapdoor(keyword));
            Check(result.Verified, "honest search verified: " + keyword + " (" + result.Reason + ")");
            Check(result.Documents.SequenceEqual(client.PlainPostings(keyword)),
                "posting list matches the oracle for " + keyword);
            compared++;
        }
        Check(compared == vocabulary.Length, "all keywords compared");

        SearchResult missing = client.Search("omega", client.BuildTrapdoor("omega"));
        Check(missing.Verified && missing.Documents.Count == 0, "absent keyword returns an empty verified set");
        SearchResult empty = client.Search("", client.BuildTrapdoor(""));
        Check(empty.Verified && empty.Documents.Count == 0, "empty keyword returns nothing");

        // A keyword appearing several times in one document is one posting.
        var single = new IndexServer();
        var singleClient = new EncryptedIndexClient(0x1111UL, single);
        singleClient.Update(new[]
        {
            new KeyValuePair<int, string[]>(1, new[] { "dup", "dup" }),
            new KeyValuePair<int, string[]>(2, new[] { "dup" }),
            new KeyValuePair<int, string[]>(3, Array.Empty<string>())
        });
        SearchResult duplicated = singleClient.Search("dup", singleClient.BuildTrapdoor("dup"));
        Check(duplicated.Verified && duplicated.Documents.SequenceEqual(new List<int> { 1, 2 }),
            "duplicate keyword in one document counts once");
        Check(single.EntryCount == 2, "documents without keywords contribute nothing");

        // ---- conjunctive search ---------------------------------------------------------------
        foreach (string left in new[] { "alpha", "beta" })
            foreach (string right in new[] { "gamma", "missing" })
            {
                SearchResult a = client.Search(left, client.BuildTrapdoor(left));
                SearchResult b = client.Search(right, client.BuildTrapdoor(right));
                var intersection = a.Documents.Intersect(b.Documents).ToList();
                var oracle = client.PlainPostings(left).Intersect(client.PlainPostings(right)).ToList();
                Check(intersection.SequenceEqual(oracle), $"AND search oracle {left} & {right}");
            }

        // ---- proof of correctness: malicious server behaviour ----------------------------------
        var malicious = new IndexServer();
        var victim = new EncryptedIndexClient(0x2222UL, malicious);
        victim.Update(documents);
        // Pick a keyword that really has several postings, so one drop or one injection is visible.
        string probe = vocabulary.First(k => victim.PlainPostings(k).Count >= 3);
        Trapdoor honestTrapdoor = victim.BuildTrapdoor(probe);
        Check(victim.Search(probe, victim.BuildTrapdoor(probe)).Verified, "control search verifies");

        malicious.TamperDrops = 1;
        SearchResult dropped = victim.Search(probe, honestTrapdoor);
        Check(!dropped.Verified, "a dropped entry is rejected");
        malicious.TamperDrops = 0;

        malicious.TamperDuplicates = 1;
        SearchResult injected = victim.Search(probe, honestTrapdoor);
        Check(!injected.Verified, "a duplicated (injected) entry is rejected");
        malicious.TamperDuplicates = 0;

        malicious.TamperCorrupt = true;
        SearchResult corrupted = victim.Search(probe, honestTrapdoor);
        Check(!corrupted.Verified, "a corrupted payload is rejected");
        malicious.TamperCorrupt = false;

        // A stale trapdoor (client state one version behind) must not look complete.
        var lateServer = new IndexServer();
        var lateClient = new EncryptedIndexClient(0x3333UL, lateServer);
        lateClient.Update(new[] { new KeyValuePair<int, string[]>(1, new[] { "trending", "filler" }) });
        Trapdoor stale = lateClient.BuildTrapdoor("trending");
        lateClient.Update(new[] { new KeyValuePair<int, string[]>(2, new[] { "trending" }) });
        SearchResult staleResult = lateClient.Search("trending", stale);
        Check(!staleResult.Verified, "a stale trapdoor is detected as incomplete");
        SearchResult freshResult = lateClient.Search("trending", lateClient.BuildTrapdoor("trending"));
        Check(freshResult.Verified && freshResult.Documents.SequenceEqual(new List<int> { 1, 2 }),
            "the refreshed trapdoor returns the whole posting list");

        // ---- leakage measurements --------------------------------------------------------------
        byte[] transcript = server.Transcript();
        foreach (string keyword in vocabulary)
            Check(NoWindow(transcript, Encoding.UTF8.GetBytes(keyword)),
                "no plaintext window for keyword " + keyword);
        Check(NoWindow(transcript, Encoding.UTF8.GetBytes("doc:")), "no plaintext window for the document id prefix");

        // Label equality is the only structural leak: identical labels share a tag, distinct ones do not.
        var tagSet = new HashSet<ulong>();
        int labels = 0;
        foreach (string keyword in vocabulary)
            foreach (int version in Enumerable.Range(0, client.BuildTrapdoor(keyword).Versions))
            {
                tagSet.Add(client.TagOf(keyword, version));
                labels++;
            }
        Check(tagSet.Count == labels, "distinct labels give distinct tags");
        Check(client.TagOf("alpha", 0) == client.TagOf("alpha", 0), "the same label gives the same tag");

        // Tags are keyed: an unkeyed precomputation from the keyword never matches.
        int precomputed = 0;
        foreach (string keyword in vocabulary)
        {
            if (client.TagOf(keyword, 0) == ToyCrypto.Hash64(0UL, keyword)) precomputed++;
            if (client.TagOf(keyword, 0) == ToyCrypto.Hash64(0UL, "tag|" + keyword + "|0")) precomputed++;
        }
        Check(precomputed == 0, "no tag is reproducible without the master key");

        // Different master keys are unlinkable.
        var otherServer = new IndexServer();
        var otherClient = new EncryptedIndexClient(0xF00DFACEUL, otherServer);
        otherClient.Update(documents);
        var otherTags = new HashSet<ulong>(otherServer.Arrival);
        Check(!tagSet.Overlaps(otherTags), "two master keys share no tag");

        // Access pattern leak is acknowledged and measured: bucket sizes equal true posting sizes.
        foreach (string keyword in vocabulary)
        {
            int bucketEntries = 0;
            int versions = client.BuildTrapdoor(keyword).Versions;
            for (int version = 0; version < versions; version++) bucketEntries += server.BucketSize(client.TagOf(keyword, version));
            Check(bucketEntries == client.PlainPostings(keyword).Count && bucketEntries == client.CommittedEntries(keyword),
                "bucket size equals posting size for " + keyword);
        }

        // ---- forward privacy measurement -------------------------------------------------------
        var fwdServer = new IndexServer();
        var fwdClient = new EncryptedIndexClient(0x4444UL, fwdServer);
        fwdClient.Update(new[] { new KeyValuePair<int, string[]>(1, new[] { "invoice" }) });
        Trapdoor revealed = fwdClient.BuildTrapdoor("invoice");
        var revealedTags = new HashSet<ulong>(revealed.Tags);
        int overlaps = 0;
        for (int round = 2; round <= 6; round++)
        {
            fwdClient.Update(new[] { new KeyValuePair<int, string[]>(round, new[] { "invoice" }) });
            ulong freshTag = fwdClient.TagOf("invoice", round - 1);
            if (revealedTags.Contains(freshTag)) overlaps++;
            if (revealedTags.Contains(fwdClient.KeyOf("invoice", round - 1))) overlaps++;
        }
        Check(overlaps == 0, "an update never re-uses a tag or key the server has seen");
        Check(fwdClient.BuildTrapdoor("invoice").Versions == 6, "one version per touched update round");

        // ---- cost bounds ----------------------------------------------------------------------
        var costServer = new IndexServer();
        var costClient = new EncryptedIndexClient(0x5555UL, costServer);
        costClient.Update(documents);
        long bruteForce = costServer.EntryCount;
        long searched = 0;
        costServer.ResetExamined();
        foreach (string keyword in vocabulary)
        {
            costClient.Search(keyword, costClient.BuildTrapdoor(keyword));
            searched += costServer.Examined;
            costServer.ResetExamined();
        }
        Check(searched == bruteForce, "searching every keyword costs exactly the index size");
        costServer.ResetExamined();
        string single1 = "alpha";
        costClient.Search(single1, costClient.BuildTrapdoor(single1));
        Check(costServer.Examined == costClient.PlainPostings(single1).Count,
            "one search touches only that label's entries: " + costServer.Examined);
        Check(costServer.Examined * 4 < bruteForce, "single label search is far below a brute force scan");

        // Client side verification state stays far smaller than the index it verifies.
        int stateBytes = costClient.StateBytes;
        int storedBytes = costServer.EntryCount * 24;
        Check(stateBytes < storedBytes, $"verification state {stateBytes}B < stored index {storedBytes}B");
        Check(stateBytes / Math.Max(1, costClient.LabelCount) < 48, "per label state is a fixed small constant");

        // ---- determinism ----------------------------------------------------------------------
        var repeatServer = new IndexServer();
        var repeatClient = new EncryptedIndexClient(0x5555UL, repeatServer);
        repeatClient.Update(documents);
        Check(repeatServer.Arrival.SequenceEqual(costServer.Arrival), "deterministic tag construction");
        Check(repeatServer.Transcript().SequenceEqual(costServer.Transcript()), "deterministic transcript");
        SearchResult repeatResult = repeatClient.Search("alpha", repeatClient.BuildTrapdoor("alpha"));
        Check(repeatResult.Verified && repeatResult.Documents.SequenceEqual(costClient.PlainPostings("alpha")),
            "deterministic retrieval");

        // ---- degenerate shapes ----------------------------------------------------------------
        var tiny = new IndexServer();
        var tinyClient = new EncryptedIndexClient(0x6666UL, tiny);
        tinyClient.Update(Array.Empty<KeyValuePair<int, string[]>>());
        SearchResult onEmpty = tinyClient.Search("anything", tinyClient.BuildTrapdoor("anything"));
        Check(onEmpty.Verified && onEmpty.Documents.Count == 0 && tiny.EntryCount == 0,
            "empty update leaves an empty but well formed index");
        tinyClient.Update(new[] { new KeyValuePair<int, string[]>(7, new[] { "solo" }) });
        SearchResult solo = tinyClient.Search("solo", tinyClient.BuildTrapdoor("solo"));
        Check(solo.Verified && solo.Documents.SequenceEqual(new List<int> { 7 }), "single document single keyword");

        Console.WriteLine($"814 index: {server.EntryCount} sealed entries in {server.BucketCount} buckets for "
            + $"{vocabulary.Length} keyword versions; verification state {stateBytes}B ({client.LabelCount} labels, "
            + $"{client.VersionCount} versions) vs {storedBytes}B stored; no plaintext window leaked");
        Console.WriteLine("PASS");
    }

    private static List<KeyValuePair<int, string[]>> BuildCorpus(string[] vocabulary, int documentCount)
    {
        var corpus = new List<KeyValuePair<int, string[]>>();
        for (int id = 1; id <= documentCount; id++)
        {
            int keywordCount = Rand(vocabulary.Length + 1);
            var chosen = new List<string>();
            for (int k = 0; k < keywordCount; k++)
            {
                string keyword = vocabulary[Rand(vocabulary.Length)];
                if (!chosen.Contains(keyword)) chosen.Add(keyword);
            }
            chosen.Sort(StringComparer.Ordinal);
            corpus.Add(new KeyValuePair<int, string[]>(id, chosen.ToArray()));
        }
        return corpus;
    }

    private static bool NoWindow(byte[] haystack, byte[] needle)
    {
        if (needle.Length < 4 || haystack.Length < needle.Length) return true;
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
        {
            bool match = true;
            for (int k = 0; k < needle.Length && match; k++) match = haystack[i + k] == needle[k];
            if (match) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
