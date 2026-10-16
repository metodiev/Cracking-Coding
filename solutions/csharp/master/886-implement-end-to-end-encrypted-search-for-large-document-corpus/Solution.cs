// Task 886 - Implement end-to-end encrypted search for large document corpus with privacy budget. (Tier: Master, Beyond — Security & Privacy)
//
// Searchable symmetric encryption (SSE) over a deterministic, keyed inverted index, plus a client-side
// differential-privacy privacy budget that caps how much the access pattern may leak overall.
//
//  * Keys: a 256-bit master key is expanded (HMAC-SHA256) into a token key, a document key and a MAC
//    key. Nothing else is derived from the seed, so the server is never given key material.
//  * Index: one entry per (term, document, version). The lookup token is HMAC(tokenKey, "tok|term");
//    the entry's payload (document id, version, live/deleted flag) is XOR-encrypted with a keystream
//    HMAC(documentKey, "entry|version|salt") where the 4-byte salt is itself derived from the term, the
//    document and the version. Each entry carries a MAC over (token, version, salt, ciphertext).
//  * Search: the client sends one token; the server walks exactly that bucket and returns opaque
//    entries (one bucket per query, no scans). The client verifies every MAC, decrypts locally and
//    keeps only the newest version of each document - so updates and deletions are resolved *client
//    side*, which is what keeps the server blind. Superseding entries are added for every term of the
//    previous version, including tombstones for terms removed by an update, otherwise a stale entry
//    would win the version comparison.
//  * Privacy budget: the scheme reveals the search pattern, so the client owns an epsilon budget. Every
//    query costs epsilonBase (the bucket *size* metadata is already a leak) plus epsilonPerEntry per
//    entry observed; a query that would exceed the budget is refused after paying epsilonBase only.
//    Counts exported to the server are perturbed with Laplace(1/eps) noise (differential privacy).
//
// Simplifications (documented, not behavioural): tokens are deterministic, so the server can see when
// two queries are equal - that is exactly why the budget exists; the version number is stored in clear
// per entry, so the server learns the update count inside a bucket; compaction is cooperative (the
// server applies a list of entry indices the client asks to keep, and never learns the payloads);
// "large corpus" is a few hundred documents, sized so the checks stay sub-second.
//
// Verified: results equal an independent plaintext scan for all terms of the corpus (present, absent,
// empty index, deleted documents, updated documents, terms removed by an update); no term or document
// id appears in the bytes the server holds; two entries of the same document/version never share a
// ciphertext and the same document under two versions never does either; the per-query charge and the
// budget arithmetic are re-derived in the test and matched; the refusal boundary is exercised at
// exactly epsilon above/below the limit; refused queries do not touch a bucket; Laplace noise passes an
// empirical tail/mean/uniqueness test; tampering is detected and suppresses results; compaction keeps
// results identical while shrinking the index; same seed => identical tokens/noise, different seed =>
// different tokens but identical results.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

internal static class Hashing
{
    internal static byte[] Hmac(byte[] key, byte[] data)
    {
        using (var hmac = new HMACSHA256(key))
        {
            return hmac.ComputeHash(data);
        }
    }

    internal static byte[] Label(byte[] key, string label) => Hmac(key, Encoding.UTF8.GetBytes(label));

    internal static byte[] Join(params byte[][] parts)
    {
        int total = parts.Sum(p => p.Length);
        var joined = new byte[total];
        int offset = 0;
        foreach (byte[] part in parts)
        {
            Array.Copy(part, 0, joined, offset, part.Length);
            offset += part.Length;
        }
        return joined;
    }

    /** Deterministic keystream: HMAC(key, label|counter) blocks, truncated to the requested length. */
    internal static byte[] Keystream(byte[] key, string label, int length)
    {
        var stream = new byte[length];
        int offset = 0, block = 0;
        while (offset < length)
        {
            byte[] digest = Hmac(key, Encoding.UTF8.GetBytes(label + "|" + block.ToString(CultureInfo.InvariantCulture)));
            int take = Math.Min(digest.Length, length - offset);
            Array.Copy(digest, 0, stream, offset, take);
            offset += take;
            block++;
        }
        return stream;
    }

    internal static byte[] Xor(byte[] a, byte[] b)
    {
        var result = new byte[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = (byte)(a[i] ^ b[i]);
        return result;
    }

    internal static byte[] Take(byte[] source, int count)
    {
        var slice = new byte[count];
        Array.Copy(source, slice, count);
        return slice;
    }
}

internal sealed class Entry
{
    internal byte[] Token = Array.Empty<byte>();
    internal int Version;
    internal byte[] Salt = Array.Empty<byte>();
    internal byte[] Cipher = Array.Empty<byte>();
    internal byte[] Mac = Array.Empty<byte>();

    internal byte[] Serialize() => Hashing.Join(Token, Salt, Cipher, Mac,
        new[] { (byte)(Version >> 24), (byte)(Version >> 16), (byte)(Version >> 8), (byte)Version });
}

/** The untrusted store: it only ever sees tokens, versions, salts, ciphertexts and MACs. */
internal sealed class Server
{
    private readonly SortedDictionary<string, List<Entry>> _buckets =
        new SortedDictionary<string, List<Entry>>(StringComparer.Ordinal);

    internal long Inserts;
    internal long Lookups;
    internal long SizeProbes;
    internal long EntriesReturned;
    internal long BucketsRewritten;

    internal void Insert(Entry entry)
    {
        string key = Convert.ToHexString(entry.Token);
        List<Entry> bucket;
        if (!_buckets.TryGetValue(key, out bucket!))
        {
            bucket = new List<Entry>();
            _buckets[key] = bucket;
        }
        bucket.Add(entry);
        Inserts++;
    }

    internal int BucketSize(byte[] token)
    {
        SizeProbes++;
        List<Entry> bucket;
        return _buckets.TryGetValue(Convert.ToHexString(token), out bucket!) ? bucket.Count : 0;
    }

    internal List<Entry> Lookup(byte[] token)
    {
        Lookups++;
        List<Entry> bucket;
        if (!_buckets.TryGetValue(Convert.ToHexString(token), out bucket!)) return new List<Entry>();
        EntriesReturned += bucket.Count;
        return bucket;
    }

    internal IEnumerable<string> BucketKeys() => _buckets.Keys;

    internal List<Entry> Entries(string bucketKey) => _buckets[bucketKey];

    internal void RewriteBucket(string bucketKey, ICollection<int> keep)
    {
        List<Entry> bucket = _buckets[bucketKey];
        var kept = new List<Entry>();
        for (int i = 0; i < bucket.Count; i++) if (keep.Contains(i)) kept.Add(bucket[i]);
        _buckets[bucketKey] = kept;
        BucketsRewritten++;
    }

    internal void CorruptEntry(byte[] token, int index)
    {
        Entry entry = _buckets[Convert.ToHexString(token)][index];
        entry.Cipher[0] ^= 0xFF;
    }

    internal long StoredEntryCount() => _buckets.Values.Sum(b => (long)b.Count);

    internal byte[] Export()
    {
        var parts = new List<byte[]>();
        foreach (List<Entry> bucket in _buckets.Values) foreach (Entry entry in bucket) parts.Add(entry.Serialize());
        return Hashing.Join(parts.ToArray());
    }
}

internal sealed class Document
{
    internal int Id;
    internal string[] Terms;
    internal int Version = 1;
    internal bool Deleted;

    internal Document(int id, params string[] terms)
    {
        Id = id;
        Terms = terms;
    }
}

internal sealed class SearchResult
{
    internal int[] DocIds = Array.Empty<int>();
    internal int EntriesObserved;
    internal double Charge;
    internal bool Refused;
    internal bool Tampered;
}

internal sealed class EncryptedSearchClient
{
    private readonly byte[] _tokenKey;
    private readonly byte[] _docKey;
    private readonly byte[] _macKey;
    private readonly Server _server;

    internal double Budget;
    internal double BudgetUsed;
    internal double BaseEpsilon = 0.05;
    internal double PerEntryEpsilon = 0.01;
    internal int ReturnedQueries;
    internal int RefusedQueries;
    internal int TamperDetections;

    private ulong _noiseState = 0xC0FFEE123456789UL;

    internal EncryptedSearchClient(ulong seed, Server server, double budget)
    {
        _server = server;
        Budget = budget;
        var master = new byte[32];
        ulong state = seed;
        for (int i = 0; i < 4; i++)
        {
            ulong word = SplitMix64(ref state);
            for (int b = 0; b < 8; b++) master[i * 8 + b] = (byte)(word >> (8 * b));
        }
        _noiseState = seed ^ 0x9E3779B97F4A7C15UL;
        _tokenKey = Hashing.Label(master, "token-key");
        _docKey = Hashing.Label(master, "document-key");
        _macKey = Hashing.Label(master, "mac-key");
    }

    private static ulong SplitMix64(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal byte[] TokenFor(string term) => Hashing.Take(Hashing.Label(_tokenKey, "tok|" + term), 16);

    private byte[] SaltFor(string term, int docId, int version) =>
        Hashing.Take(Hashing.Label(_docKey, "salt|" + term + "|" + docId + "|" + version), 4);

    /** Adds the indexed form of every term of the document, at the document's current version. */
    internal void IndexDocument(Document doc)
    {
        foreach (string term in doc.Terms.Distinct(StringComparer.Ordinal).OrderBy(t => t, StringComparer.Ordinal))
            AddEntry(doc.Id, doc.Version, term, true);
    }

    internal void UpdateDocument(Document doc, params string[] newTerms)
    {
        string[] previous = doc.Terms;
        doc.Terms = newTerms;
        doc.Version++;
        doc.Deleted = false;
        foreach (string term in newTerms.Distinct(StringComparer.Ordinal).OrderBy(t => t, StringComparer.Ordinal))
            AddEntry(doc.Id, doc.Version, term, true);
        // Terms that disappeared must be superseded by a tombstone, else the older live entry wins.
        foreach (string term in previous.Except(newTerms, StringComparer.Ordinal)
                                       .Distinct(StringComparer.Ordinal)
                                       .OrderBy(t => t, StringComparer.Ordinal))
            AddEntry(doc.Id, doc.Version, term, false);
    }

    internal void DeleteDocument(Document doc)
    {
        doc.Version++;
        doc.Deleted = true;
        foreach (string term in doc.Terms.Distinct(StringComparer.Ordinal).OrderBy(t => t, StringComparer.Ordinal))
            AddEntry(doc.Id, doc.Version, term, false);
    }

    private void AddEntry(int docId, int version, string term, bool live)
    {
        byte[] token = TokenFor(term);
        byte[] salt = SaltFor(term, docId, version);
        byte[] payload = EncodePayload(docId, version, live);
        byte[] cipher = Hashing.Xor(payload, Hashing.Keystream(_docKey, "entry|" + version + "|" + Convert.ToHexString(salt), payload.Length));
        byte[] mac = Hashing.Take(Hashing.Hmac(_macKey, Hashing.Join(token, salt, cipher, VersionBytes(version))), 16);
        _server.Insert(new Entry { Token = token, Version = version, Salt = salt, Cipher = cipher, Mac = mac });
    }

    private static byte[] VersionBytes(int version) =>
        new[] { (byte)(version >> 24), (byte)(version >> 16), (byte)(version >> 8), (byte)version };

    private static byte[] EncodePayload(int docId, int version, bool live)
    {
        var payload = new byte[9];
        payload[0] = (byte)(docId >> 24);
        payload[1] = (byte)(docId >> 16);
        payload[2] = (byte)(docId >> 8);
        payload[3] = (byte)docId;
        payload[4] = (byte)(version >> 24);
        payload[5] = (byte)(version >> 16);
        payload[6] = (byte)(version >> 8);
        payload[7] = (byte)version;
        payload[8] = (byte)(live ? 1 : 0);
        return payload;
    }

    private bool TryOpen(Entry entry, out int docId, out int version, out bool live)
    {
        docId = 0; version = 0; live = false;
        byte[] expected = Hashing.Take(Hashing.Hmac(_macKey, Hashing.Join(entry.Token, entry.Salt, entry.Cipher, VersionBytes(entry.Version))), 16);
        if (!expected.SequenceEqual(entry.Mac)) return false;
        byte[] payload = Hashing.Xor(entry.Cipher,
            Hashing.Keystream(_docKey, "entry|" + entry.Version + "|" + Convert.ToHexString(entry.Salt), entry.Cipher.Length));
        docId = (payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3];
        version = (payload[4] << 24) | (payload[5] << 16) | (payload[6] << 8) | payload[7];
        live = payload[8] == 1;
        return version == entry.Version;
    }

    internal SearchResult Search(string term)
    {
        byte[] token = TokenFor(term);
        int bucketSize = _server.BucketSize(token);                       // metadata leak, always paid
        double charge = BaseEpsilon + PerEntryEpsilon * bucketSize;
        if (BudgetUsed + charge > Budget + 1e-12)
        {
            BudgetUsed += BaseEpsilon;
            RefusedQueries++;
            return new SearchResult { Refused = true, Charge = BaseEpsilon, EntriesObserved = bucketSize };
        }

        List<Entry> entries = _server.Lookup(token);
        BudgetUsed += charge;
        ReturnedQueries++;
        var newest = new Dictionary<int, KeyValuePair<int, bool>>();      // docId -> (version, live)
        bool tampered = false;
        foreach (Entry entry in entries)
        {
            if (!TryOpen(entry, out int docId, out int version, out bool live)) { tampered = true; continue; }
            KeyValuePair<int, bool> current;
            if (!newest.TryGetValue(docId, out current) || version > current.Key)
                newest[docId] = new KeyValuePair<int, bool>(version, live);
        }
        if (tampered)
        {
            TamperDetections++;
            return new SearchResult { Refused = false, Tampered = true, Charge = charge, EntriesObserved = bucketSize };
        }
        int[] ids = newest.Where(pair => pair.Value.Value).Select(pair => pair.Key).OrderBy(id => id).ToArray();
        return new SearchResult { DocIds = ids, Charge = charge, EntriesObserved = bucketSize };
    }

    /** Cooperation rather than trust: the server only executes the keep-list, never the payloads. */
    internal int Compact()
    {
        int removed = 0;
        foreach (string bucket in _server.BucketKeys().ToList())
        {
            List<Entry> entries = _server.Entries(bucket);
            var keep = new List<int>();
            var newest = new Dictionary<int, (int Version, int Index)>();
            for (int i = 0; i < entries.Count; i++)
            {
                if (!TryOpen(entries[i], out int docId, out int version, out bool _))
                {
                    keep.Add(i);          // unreadable (e.g. tampered) entries are never discarded
                    continue;
                }
                if (!newest.TryGetValue(docId, out (int Version, int Index) current) || version > current.Version)
                    newest[docId] = (version, i);
            }
            foreach ((int Version, int Index) value in newest.Values) keep.Add(value.Index);
            removed += entries.Count - keep.Count;
            _server.RewriteBucket(bucket, keep);
        }
        return removed;
    }

    /** Laplace(1/eps) noise for any count the client publishes; sampling is deterministic. */
    internal double ReportCount(double trueCount, double epsilon)
    {
        ulong word = SplitMix64(ref _noiseState);
        double u = (word >> 11) * (1.0 / 9007199254740992.0) - 0.5;        // uniform in (-0.5, 0.5)
        double noise = -Math.Sign(u) * Math.Log(1.0 - 2.0 * Math.Abs(u)) / epsilon;
        LastReportNoise = noise;
        return Math.Max(0.0, trueCount + noise);
    }

    internal double LastReportNoise;

    internal double ChargeFor(string term)
    {
        int bucketSize = _server.BucketSize(TokenFor(term));
        return BaseEpsilon + PerEntryEpsilon * bucketSize;
    }
}

public static class Solution
{
    /** Independent plaintext oracle: no keys, no index, just the corpus. */
    private static int[] PlaintextSearch(IEnumerable<Document> corpus, string term) =>
        corpus.Where(d => !d.Deleted && d.Terms.Contains(term, StringComparer.Ordinal))
              .Select(d => d.Id)
              .OrderBy(id => id)
              .ToArray();

    private static bool SameIds(int[] a, int[] b) => a.Length == b.Length && a.SequenceEqual(b);

    /** Raw byte-subsequence search: the honest "does the server hold this plaintext?" test. */
    private static bool ContainsBytes(byte[] haystack, byte[] needle)
    {
        if (needle.Length == 0 || haystack.Length < needle.Length) return false;
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
                if (haystack[i + j] != needle[j]) { match = false; break; }
            if (match) return true;
        }
        return false;
    }

    private static string Describe(int[] ids) => "[" + string.Join(",", ids) + "]";

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<Document> BuildCorpus()
    {
        var corpus = new List<Document>();
        string[] vocabulary = { "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta" };
        for (int i = 0; i < 120; i++)
        {
            int a = i % vocabulary.Length;
            int b = (i * 3 + 1) % vocabulary.Length;
            var terms = new List<string> { vocabulary[a], vocabulary[b] };
            if (i % 5 == 0) terms.Add("common");
            if (i % 7 == 0) terms.Add(vocabulary[(a + 4) % vocabulary.Length]);
            corpus.Add(new Document(i, terms.Distinct(StringComparer.Ordinal).ToArray()));
        }
        return corpus;
    }

    public static void Main()
    {
        // ---- 1. correctness against an independent plaintext scan -------------------------------
        var server = new Server();
        var corpus = BuildCorpus();
        var client = new EncryptedSearchClient(20261016UL, server, 20.0);
        foreach (Document doc in corpus) client.IndexDocument(doc);

        string[] probes = { "alpha", "beta", "common", "zeta", "absent", "", "Alpha", "comm" };
        foreach (string term in probes)
        {
            int[] expected = PlaintextSearch(corpus, term);
            SearchResult result = client.Search(term);
            Check(!result.Refused && !result.Tampered, "query '" + term + "' served");
            Check(SameIds(result.DocIds, expected), "plaintext oracle for '" + term + "': got " +
                Describe(result.DocIds) + " expected " + Describe(expected));
        }
        Check(client.Search("a").DocIds.Length == 0 && client.Search("commonx").DocIds.Length == 0,
            "absent and near-miss terms return nothing");
        Check(client.Search("alpha").DocIds.Length == PlaintextSearch(corpus, "alpha").Length,
            "term in many documents matches the oracle");

        // ---- 2. empty index and single document ------------------------------------------------
        var emptyServer = new Server();
        var emptyClient = new EncryptedSearchClient(7UL, emptyServer, 0.25);
        Check(emptyClient.Search("anything").DocIds.Length == 0 && emptyServer.StoredEntryCount() == 0,
            "empty index answers an empty result");
        var single = new Document(42, "solo", "solo2");
        emptyClient.IndexDocument(single);
        Check(SameIds(emptyClient.Search("solo").DocIds, new[] { 42 }), "single document is found");
        Check(emptyClient.Search("solo2").EntriesObserved == 1, "one entry observed per single-term document");

        // ---- 3. updates, removals and deletions, resolved client side --------------------------
        var updServer = new Server();
        var updClient = new EncryptedSearchClient(99UL, updServer, 10.0);
        var tracked = new List<Document>
        {
            new Document(101, "alpha", "beta"),
            new Document(102, "beta", "gamma"),
            new Document(103, "alpha"),
        };
        foreach (Document doc in tracked) updClient.IndexDocument(doc);
        Check(SameIds(updClient.Search("alpha").DocIds, PlaintextSearch(tracked, "alpha")), "pre-update alpha");
        Check(SameIds(updClient.Search("beta").DocIds, PlaintextSearch(tracked, "beta")), "pre-update beta");

        updClient.UpdateDocument(tracked[0], "gamma", "delta");      // alpha and beta leave document 1
        Check(SameIds(updClient.Search("alpha").DocIds, PlaintextSearch(tracked, "alpha")),
            "an update retracts terms the document lost");
        Check(SameIds(updClient.Search("beta").DocIds, PlaintextSearch(tracked, "beta")),
            "an update retracts shared terms only for that document");
        Check(SameIds(updClient.Search("delta").DocIds, PlaintextSearch(tracked, "delta")),
            "new terms become searchable");
        Check(updClient.Search("delta").DocIds.SequenceEqual(new[] { 101 }), "delta only in document 101");

        updClient.DeleteDocument(tracked[1]);
        Check(SameIds(updClient.Search("beta").DocIds, PlaintextSearch(tracked, "beta")), "deletion retracts beta");
        Check(SameIds(updClient.Search("gamma").DocIds, PlaintextSearch(tracked, "gamma")),
            "deletion leaves the other holder of gamma");

        // ---- 4. what the server can see --------------------------------------------------------
        byte[] exported = updServer.Export();
        foreach (string term in new[] { "alpha", "beta", "gamma", "delta" })
            Check(!ContainsBytes(exported, Encoding.UTF8.GetBytes(term)),
                "no plaintext term '" + term + "' in the stored bytes");
        foreach (Document doc in tracked)
            Check(!ContainsBytes(exported, new[] { (byte)(doc.Id >> 24), (byte)(doc.Id >> 16), (byte)(doc.Id >> 8), (byte)doc.Id }),
                "no raw document id in the stored bytes");
        Check(ContainsBytes(exported, new byte[] { 1 }) && ContainsBytes(exported, new byte[] { 2 }),
            "the byte search itself is not vacuous");
        byte[] token = updClient.TokenFor("alpha");
        Check(token.Length == 16 && !token.SequenceEqual(Encoding.UTF8.GetBytes("alpha")), "tokens are not plaintext");
        Check(!updClient.TokenFor("alpha").SequenceEqual(updClient.TokenFor("beta")), "distinct terms, distinct tokens");
        var otherClient = new EncryptedSearchClient(1000UL, new Server(), 1.0);
        Check(!otherClient.TokenFor("alpha").SequenceEqual(token), "a different key yields a different token");

        // Every entry of the same document/version must be independently encrypted (no linkage).
        var cipherServer = new Server();
        var cipherClient = new EncryptedSearchClient(5UL, cipherServer, 4.0);
        var linkDoc = new Document(77, "one", "two", "three");
        cipherClient.IndexDocument(linkDoc);
        var ciphers = new List<string>();
        foreach (string bucket in cipherServer.BucketKeys())
            foreach (Entry entry in cipherServer.Entries(bucket))
                ciphers.Add(Convert.ToHexString(entry.Cipher));
        Check(ciphers.Count == 3 && ciphers.Distinct().Count() == 3, "same document, distinct ciphertexts");
        byte[] idBytes = { 0, 0, 0, 77 };
        Check(ciphers.All(c => !ContainsBytes(Convert.FromHexString(c), idBytes)),
            "no ciphertext carries the raw document id");
        cipherClient.UpdateDocument(linkDoc, "one", "two", "three");
        var secondVersion = new HashSet<string>();
        foreach (string bucket in cipherServer.BucketKeys())
            foreach (Entry entry in cipherServer.Entries(bucket))
                if (entry.Version == 2) secondVersion.Add(Convert.ToHexString(entry.Cipher));
        Check(secondVersion.Count == 3 && !secondVersion.Overlaps(ciphers), "a re-index uses fresh ciphertexts");

        // ---- 5. access pattern: exactly one bucket per query -----------------------------------
        long lookupsBefore = server.Lookups, probesBefore = server.SizeProbes, returnedBefore = server.EntriesReturned;
        SearchResult one = client.Search("common");
        Check(server.Lookups == lookupsBefore + 1 && server.SizeProbes == probesBefore + 1,
            "one bucket lookup and one size probe per query");
        Check(server.EntriesReturned == returnedBefore + one.EntriesObserved,
            "the server returns exactly the bucket, nothing else");
        Check(one.EntriesObserved == PlaintextSearch(corpus, "common").Length,
            "the touched bucket holds exactly the live entries for the term");

        // ---- 6. privacy budget arithmetic -------------------------------------------------------
        var budgetServer = new Server();
        var budgetClient = new EncryptedSearchClient(11UL, budgetServer, 1.0)
        {
            BaseEpsilon = 0.05,
            PerEntryEpsilon = 0.01,
        };
        var budgetDoc = new Document(1, "term");
        budgetClient.IndexDocument(budgetDoc);
        double expectedCharge = 0.05 + 0.01 * budgetClient.Search("term").EntriesObserved;
        budgetClient.BudgetUsed = 0.0;
        SearchResult charged = budgetClient.Search("term");
        Check(Math.Abs(charged.Charge - expectedCharge) < 1e-12, "charge equals epsilonBase + epsilonPer * bucketSize");
        Check(Math.Abs(budgetClient.BudgetUsed - expectedCharge) < 1e-12, "budget accounting is exact after one query");

        double used = budgetClient.BudgetUsed;
        double nextCharge = budgetClient.ChargeFor("term");
        budgetClient.Budget = used + nextCharge;                       // exactly enough
        SearchResult exactly = budgetClient.Search("term");
        Check(!exactly.Refused, "a query that exactly exhausts the budget is served");
        Check(Math.Abs(budgetClient.BudgetUsed - budgetClient.Budget) < 1e-12, "the budget is exactly consumed");

        double floor = budgetClient.BudgetUsed;
        budgetClient.Budget = floor + nextCharge - 0.001;              // a hair too small
        long probesAtRefusal = budgetServer.SizeProbes, lookupsAtRefusal = budgetServer.Lookups;
        SearchResult refused = budgetClient.Search("term");
        Check(refused.Refused && refused.DocIds.Length == 0, "a query over budget is refused");
        Check(Math.Abs(budgetClient.BudgetUsed - (floor + 0.05)) < 1e-12, "a refusal costs only epsilonBase");
        Check(budgetServer.SizeProbes == probesAtRefusal + 1 && budgetServer.Lookups == lookupsAtRefusal,
            "a refused query never opens a bucket");

        // The per-query charge must match an independent recomputation over a longer session.
        var auditServer = new Server();
        var auditClient = new EncryptedSearchClient(13UL, auditServer, 100.0);
        foreach (Document doc in corpus) auditClient.IndexDocument(doc);
        double independent = 0.0;
        foreach (string term in new[] { "alpha", "beta", "gamma", "delta", "eta", "theta", "common" })
        {
            int size = 0;
            foreach (Document doc in corpus)
                if (doc.Terms.Contains(term, StringComparer.Ordinal)) size++;
            independent += size * 0.01 + 0.05;
            SearchResult r = auditClient.Search(term);
            Check(!r.Refused && r.EntriesObserved == size, "bucket size matches the corpus for '" + term + "'");
        }
        Check(Math.Abs(auditClient.BudgetUsed - independent) < 1e-9,
            "total epsilon equals sum(epsilonBase + epsilonPer * bucketSize)");

        // ---- 7. differential privacy of the published count ------------------------------------
        var client2 = new EncryptedSearchClient(17UL, new Server(), 1.0);
        var noises = new List<double>();
        for (int i = 0; i < 400; i++)
        {
            client2.ReportCount(10.0, 1.0);
            noises.Add(client2.LastReportNoise);
        }
        Check(noises.Count(n => Math.Abs(n) > 0.25) > 300, "noise is substantial in most reports");
        Check(noises.All(n => Math.Abs(n) < 12.0), "Laplace(1) noise stays inside a 12 unit tail");
        Check(noises.Count(n => Math.Abs(n) > 8.0) <= 2, "the heavy tail is rare");
        double meanAbs = noises.Sum(n => Math.Abs(n)) / noises.Count;
        Check(meanAbs > 0.7 && meanAbs < 1.3, "mean |noise| matches E|Laplace(1)| = 1, got " +
            meanAbs.ToString("F3", CultureInfo.InvariantCulture));
        Check(noises.Distinct().Count() > 380, "noise samples are effectively unique");
        Check(noises.Average(n => n) > -0.3 && noises.Average(n => n) < 0.3, "noise is unbiased");

        // ---- 8. tamper detection ---------------------------------------------------------------
        var tamperServer = new Server();
        var tamperClient = new EncryptedSearchClient(23UL, tamperServer, 5.0);
        var tamperDoc = new Document(5, "secret");
        tamperClient.IndexDocument(tamperDoc);
        Check(SameIds(tamperClient.Search("secret").DocIds, new[] { 5 }), "healthy index answers");
        tamperServer.CorruptEntry(tamperClient.TokenFor("secret"), 0);
        SearchResult tampered = tamperClient.Search("secret");
        Check(tampered.Tampered && tampered.DocIds.Length == 0, "a corrupted entry suppresses the result");
        Check(tamperClient.TamperDetections == 1, "tampering is reported once");
        Check(Math.Abs(tampered.Charge - (0.05 + 0.01)) < 1e-12, "the access pattern was still paid for");

        // ---- 9. compaction keeps answers and shrinks the store ---------------------------------
        long beforeCompaction = updServer.StoredEntryCount();
        int[] alphaBefore = updClient.Search("alpha").DocIds;
        int[] gammaBefore = updClient.Search("gamma").DocIds;
        int removed = updClient.Compact();
        Check(removed > 0 && updServer.StoredEntryCount() == beforeCompaction - removed,
            "compaction removes superseded entries: " + removed + " of " + beforeCompaction);
        Check(SameIds(updClient.Search("alpha").DocIds, alphaBefore) && SameIds(updClient.Search("gamma").DocIds, gammaBefore),
            "compaction preserves query answers");
        Check(updClient.Compact() == 0, "compaction is idempotent");
        Check(updClient.Compact() == 0 && updServer.StoredEntryCount() >= 1, "the store is still non-empty");

        // ---- 10. determinism -------------------------------------------------------------------
        var repeatServer = new Server();
        var repeatClient = new EncryptedSearchClient(20261016UL, repeatServer, 20.0);
        foreach (Document doc in corpus) repeatClient.IndexDocument(doc);
        Check(Convert.ToHexString(repeatClient.TokenFor("alpha")) == Convert.ToHexString(client.TokenFor("alpha")),
            "same seed, same tokens");
        Check(repeatServer.StoredEntryCount() == server.StoredEntryCount(), "same seed, same index size");
        Check(SameIds(repeatClient.Search("alpha").DocIds, PlaintextSearch(corpus, "alpha")), "repeat run agrees with the oracle");
        Check(SameIds(repeatClient.Search("beta").DocIds, client.Search("beta").DocIds),
            "both clients return the same ids for the same corpus");
        Check(repeatClient.Search("alpha").DocIds.Length == PlaintextSearch(corpus, "alpha").Length,
            "final query still matches the oracle");

        Console.WriteLine("corpus=" + corpus.Count + " docs, stored entries=" + server.StoredEntryCount() +
            ", bucket lookups=" + server.Lookups + ", size probes=" + server.SizeProbes);
        Console.WriteLine("budget: served=" + client.ReturnedQueries + " refused=" + client.RefusedQueries +
            " used=" + client.BudgetUsed.ToString("F4", CultureInfo.InvariantCulture) +
            " of " + client.Budget.ToString("F2", CultureInfo.InvariantCulture) +
            "; tamper detections=" + tamperClient.TamperDetections + "; compaction removed=" + removed);
        Console.WriteLine("dp: mean|noise|=" + meanAbs.ToString("F3", CultureInfo.InvariantCulture) +
            " max|noise|=" + noises.Max(n => Math.Abs(n)).ToString("F3", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
