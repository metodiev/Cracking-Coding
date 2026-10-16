/** Task 814 - Implement searchable encrypted index with limited leakage and proof-of-correctness. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The scheme is a forward-private searchable symmetric encryption (SSE) index in the shape of Sophos/Diana:
 *
 *   - The client derives, for every keyword w, a state key  K_w = PRF(K_master, w)  and for the i-th update
 *     to w a one-time search token  t_{w,i} = PRF(K_w, i).  Only the token reaches the server, so the server
 *     learns neither w nor which earlier update a later token belongs to (forward privacy).
 *   - The server holds an append-only dictionary  token -> entry,  the entry being an authenticated
 *     ciphertext of (add|delete, document id).  A search is a list of tokens; the server answers with the
 *     entries stored under them.  Deletes are tombstones and the client resolves them.
 *   - Proof of correctness: the client keeps a commitment (leaf hash) for every token it uploaded, and the
 *     server must answer each token with the entry plus a Merkle inclusion path up to the index root.  An
 *     omitted entry, an altered ciphertext, or an entry under a token that was never issued all fail.
 *
 * Leakage is *measured* rather than assumed.  The server sees exactly three things: how many entries exist,
 * which tokens a search returns (access pattern) and whether two searches used the same token set (search
 * pattern).  Each check below quantifies one of those, or shows that no plaintext reaches the wire.
 *
 * Deliberate simplifications (fully deterministic -- java.util only, no threads, sockets, clocks or real
 * crypto): the PRF is splitmix64 with a fixed key schedule, the cipher is a keystream XOR with a mix-fold
 * tag, and the index holds tens of entries.  These stand-ins exist so that leakage, forward privacy and
 * proof soundness can be measured; they are not cryptographically secure and no key material is real.
 *
 * Verified invariants:
 *   correctness - searches equal a plaintext brute-force oracle over the same workload, including tombstoned
 *                 and re-added documents, duplicated updates, an empty index and an absent keyword.
 *   forward priv- the tokens of successive updates to one keyword are pairwise distinct while the state key
 *   acy           is stable, so a new token cannot be matched to earlier ones.
 *   leakage     - stored entries == number of updates (no amplification); a search costs exactly the number
 *                 of updates to that keyword, independent of index size; equal searches reuse equal tokens
 *                 (the documented search pattern); distinct keywords never share a token.
 *   proof       - honest proofs verify against the root the client recomputes from its own commitments;
 *                 omission, ciphertext alteration and a swapped root are all detected.
 *   confidential- no keyword or document id appears in any server-visible byte string, and no stored token
 *   ity           equals the PRF of the keyword under the master key.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** Toy PRF and stream cipher: deterministic and invertible, but NOT cryptographically secure. */
class Prf {
    static long mix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** PRF(k, i): the value an adversary must not be able to compute without k. */
    static long prf(long k, long i) { return mix(k ^ mix(i + 0x2545F4914F6CDD1DL)); }

    static long prfString(long k, String s) {
        long h = mix(k ^ (s.length() * 0x9E3779B1L));
        for (int i = 0; i < s.length(); i++) h = mix(h ^ s.charAt(i));
        return h;
    }

    static byte[] xorStream(long key, long nonce, byte[] in) {
        SplitMix64 rng = new SplitMix64(mix(key ^ nonce) ^ 0x2545F4914F6CDD1DL);
        byte[] out = new byte[in.length];
        int i = 0;
        boolean fresh = true;
        long v = 0;
        while (i < in.length) {
            if (fresh) v = rng.nextLong();
            out[i] = (byte) (in[i] ^ (byte) (v >>> (8 * (i % 8))));
            fresh = (i % 8) == 7;
            i++;
        }
        return out;
    }

    static long tag(long key, long nonce, byte[] ct) {
        long h = mix(key ^ mix(nonce) ^ ct.length);
        for (byte b : ct) h = mix(h ^ (b & 0xFFL));
        return h;
    }
}

/** Merkle commitments over the index leaves; padding to a power of two keeps every proof binary. */
class Merkle {
    static final long PAD = 0x5A5A5A5A5A5A5A5AL;

    static long leaf(long token, byte[] ct, long tag) {
        long h = Prf.mix(token * 0x9E3779B97F4A7C15L ^ tag);
        for (byte b : ct) h = Prf.mix(h ^ (b & 0xFFL));
        return h;
    }

    static long[] padded(long[] leaves) {
        int n = 1;
        while (n < leaves.length) n <<= 1;
        long[] out = new long[n];
        Arrays.fill(out, PAD);
        System.arraycopy(leaves, 0, out, 0, leaves.length);
        return out;
    }

    static long[] levelUp(long[] cur) {
        long[] next = new long[cur.length / 2];
        for (int i = 0; i < next.length; i++) next[i] = Prf.mix(cur[2 * i] ^ Prf.mix(cur[2 * i + 1]));
        return next;
    }

    static long root(long[] leaves) {
        long[] cur = padded(leaves);
        while (cur.length > 1) cur = levelUp(cur);
        return cur[0];
    }

    /** Sibling hashes from the leaf level upwards; the path length is log2(padded size). */
    static List<Long> proof(long[] leaves, int index) {
        List<Long> path = new ArrayList<>();
        long[] cur = padded(leaves);
        int idx = index;
        while (cur.length > 1) {
            path.add(cur[idx ^ 1]);
            cur = levelUp(cur);
            idx >>= 1;
        }
        return path;
    }

    /** Recomputes the root from a leaf and its path; the index bits decide the child order. */
    static long fold(long leaf, int index, List<Long> path) {
        long h = leaf;
        int idx = index;
        for (long sibling : path) {
            h = (idx & 1) == 0 ? Prf.mix(h ^ Prf.mix(sibling)) : Prf.mix(sibling ^ Prf.mix(h));
            idx >>= 1;
        }
        return h;
    }
}

/** One opaque entry as stored by the server: the token, the ciphertext and its tag. */
class Entry {
    final long token;
    final long tag;
    final byte[] ct;

    Entry(long token, long tag, byte[] ct) { this.token = token; this.tag = tag; this.ct = ct; }

    Entry tampered() { byte[] bad = ct.clone(); bad[0] ^= 1; return new Entry(token, tag, bad); }
}

/** What the server sends back for one token: the entry, its inclusion path and the index root. */
class Response {
    final Entry entry;
    final List<Long> path;
    final long root;
    final int index;

    Response(Entry entry, List<Long> path, long root, int index) {
        this.entry = entry; this.path = path; this.root = root; this.index = index;
    }
}

/** The untrusted server: an append-only token dictionary answering searches with proofs. */
class IndexServer {
    final TreeMap<Long, Entry> store = new TreeMap<>();
    long lookups;

    void put(Entry e) { store.put(e.token, e); }

    int size() { return store.size(); }

    long[] leaves() {
        long[] out = new long[store.size()];
        int i = 0;
        for (Entry e : store.values()) out[i++] = Merkle.leaf(e.token, e.ct, e.tag);
        return out;
    }

    long root() { return Merkle.root(leaves()); }

    int indexOf(long token) {
        int i = 0;
        for (long t : store.keySet()) { if (t == token) return i; i++; }
        return -1;
    }

    /** Honest answer for one token; `null` when nothing is stored under it. */
    Response fetch(long token) {
        lookups++;
        Entry e = store.get(token);
        if (e == null) return null;
        long[] leaves = leaves();
        int idx = indexOf(token);
        return new Response(e, Merkle.proof(leaves, idx), Merkle.root(leaves), idx);
    }
}

class SearchResult {
    boolean ok;
    String reason = "";
    final SortedSet<String> docs = new TreeSet<>();
    final List<Long> tokens = new ArrayList<>();
    long lookups;

    static SearchResult failed(String reason) {
        SearchResult r = new SearchResult();
        r.ok = false;
        r.reason = reason;
        return r;
    }
}

/** The client: owns the keys, issues one-time tokens and verifies every answer against its commitments. */
class SseClient {
    final long masterKey;
    final IndexServer server;
    final Map<String, Integer> counter = new TreeMap<>();
    final Map<String, Long> stateKey = new TreeMap<>();
    final Map<Long, Long> commitment = new TreeMap<>();
    final Map<Long, Long> entryKey = new TreeMap<>();
    final Map<Long, Long> entryNonce = new TreeMap<>();
    final List<Long> issued = new ArrayList<>();

    SseClient(long masterKey, IndexServer server) { this.masterKey = masterKey; this.server = server; }

    long stateKeyOf(String keyword) {
        return stateKey.computeIfAbsent(keyword, k -> Prf.prfString(masterKey, "state|" + k));
    }

    int updates(String keyword) { return counter.getOrDefault(keyword, 0); }

    static byte[] encode(boolean add, String docId) {
        byte[] id = docId.getBytes();
        byte[] out = new byte[id.length + 1];
        out[0] = (byte) (add ? 1 : 0);
        System.arraycopy(id, 0, out, 1, id.length);
        return out;
    }

    static String decodeDoc(byte[] plain) { return new String(plain, 1, plain.length - 1); }

    /** Issue the next token for `keyword` and upload one encrypted entry. */
    long update(String keyword, String docId, boolean add) {
        long k = stateKeyOf(keyword);
        int i = updates(keyword);
        long token = Prf.prf(k, i);
        long key = Prf.prf(k, 0x4000_0000L + i);
        long nonce = Prf.prf(k, 0x8000_0000L + i);
        byte[] ct = Prf.xorStream(key, nonce, encode(add, docId));
        long tag = Prf.tag(key, nonce, ct);
        server.put(new Entry(token, tag, ct));
        commitment.put(token, Merkle.leaf(token, ct, tag));
        entryKey.put(token, key);
        entryNonce.put(token, nonce);
        issued.add(token);
        counter.put(keyword, i + 1);
        return token;
    }

    /** The client's own view of the index root, recomputed from the commitments it uploaded. */
    long expectedRoot() {
        long[] leaves = new long[commitment.size()];
        int i = 0;
        for (long c : commitment.values()) leaves[i++] = c;
        return Merkle.root(leaves);
    }

    int indexOfToken(long token) {
        int i = 0;
        for (long t : commitment.keySet()) { if (t == token) return i; i++; }
        return -1;
    }

    /** Search = enumerate the keyword's tokens, fetch, verify proofs, then resolve tombstones. */
    SearchResult search(String keyword) {
        SearchResult r = new SearchResult();
        long k = stateKeyOf(keyword);
        int u = updates(keyword);
        long root = expectedRoot();
        TreeMap<String, Boolean> live = new TreeMap<>();
        for (int i = 0; i < u; i++) {
            long token = Prf.prf(k, i);
            r.tokens.add(token);
            Response resp = server.fetch(token);
            if (resp == null) { r.lookups = server.lookups; return SearchResult.failed("entry " + i + " is missing"); }
            Long want = commitment.get(token);
            if (want == null) { r.lookups = server.lookups; return SearchResult.failed("token " + i + " was never issued"); }
            long leaf = Merkle.leaf(resp.entry.token, resp.entry.ct, resp.entry.tag);
            if (leaf != want) { r.lookups = server.lookups; return SearchResult.failed("entry " + i + " was altered"); }
            if (resp.root != root || Merkle.fold(leaf, resp.index, resp.path) != root) {
                r.lookups = server.lookups;
                return SearchResult.failed("inclusion proof for entry " + i + " does not fold to the root");
            }
            long key = entryKey.get(token), nonce = entryNonce.get(token);
            if (Prf.tag(key, nonce, resp.entry.ct) != resp.entry.tag) {
                r.lookups = server.lookups;
                return SearchResult.failed("tag mismatch for entry " + i);
            }
            byte[] plain = Prf.xorStream(key, nonce, resp.entry.ct);
            live.put(decodeDoc(plain), plain[0] != 0);
        }
        r.lookups = server.lookups;
        for (Map.Entry<String, Boolean> e : live.entrySet()) if (e.getValue()) r.docs.add(e.getKey());
        r.ok = true;
        return r;
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static final long MASTER = 0x0123456789ABCDEFL;

    /** Plaintext oracle: replay the workload with no encryption at all. */
    static SortedSet<String> oracle(List<Object[]> workload, String keyword) {
        TreeMap<String, Boolean> live = new TreeMap<>();
        for (Object[] u : workload) {
            if (!u[0].equals(keyword)) continue;
            live.put((String) u[1], (Boolean) u[2]);
        }
        SortedSet<String> out = new TreeSet<>();
        for (Map.Entry<String, Boolean> e : live.entrySet()) if (e.getValue()) out.add(e.getKey());
        return out;
    }

    static int indexOf(byte[] hay, byte[] needle) {
        if (needle.length == 0 || needle.length > hay.length) return -1;
        outer:
        for (int i = 0; i + needle.length <= hay.length; i++) {
            for (int j = 0; j < needle.length; j++) if (hay[i + j] != needle[j]) continue outer;
            return i;
        }
        return -1;
    }

    /** The token sequence must be exactly PRF(K_w, 0..n-1): reproducible with no hidden state. */
    static boolean tokensArePrfSequence(SseClient client, String keyword) {
        long k = client.stateKeyOf(keyword);
        for (int i = 0; i < client.updates(keyword); i++) if (!client.issued.contains(Prf.prf(k, i))) return false;
        return true;
    }

    public static void main(String[] args) {
        // ---- 1. round trip and brute-force agreement ----------------------------------------------
        IndexServer server = new IndexServer();
        SseClient client = new SseClient(MASTER, server);
        client.update("alpha", "doc1", true);
        client.update("alpha", "doc2", true);
        client.update("alpha", "doc3", true);
        SearchResult r1 = client.search("alpha");
        check(r1.ok, "honest search must verify: " + r1.reason);
        check(r1.docs.toString().equals("[doc1, doc2, doc3]"), "search must return the three added docs: " + r1.docs);
        check(r1.lookups == 3, "a search must cost one lookup per update, saw " + r1.lookups);
        check(server.size() == 3, "the server stores exactly one entry per update");

        SplitMix64 rng = new SplitMix64(20260814L);
        String[] keywords = {"k0", "k1", "k2", "k3", "k4", "k5"};
        List<Object[]> workload = new ArrayList<>();
        IndexServer s2 = new IndexServer();
        SseClient c2 = new SseClient(MASTER, s2);
        for (int i = 0; i < 60; i++) {
            String kw = keywords[rng.nextInt(keywords.length)];
            String doc = "d" + rng.nextInt(12);
            boolean add = rng.nextInt(4) != 0;
            c2.update(kw, doc, add);
            workload.add(new Object[] {kw, doc, add});
        }
        for (String kw : keywords) {
            SearchResult r = c2.search(kw);
            check(r.ok, "search for " + kw + " failed: " + r.reason);
            check(r.docs.equals(oracle(workload, kw)), "search for " + kw + " disagrees with the oracle: "
                    + r.docs + " vs " + oracle(workload, kw));
        }
        check(s2.size() == workload.size(), "stored entries must equal updates: " + s2.size() + " vs " + workload.size());
        check(c2.commitment.size() == workload.size(), "there must be one commitment per update");

        // ---- 2. tombstones, re-adds and duplicates -------------------------------------------------
        IndexServer s3 = new IndexServer();
        SseClient c3 = new SseClient(MASTER, s3);
        c3.update("w", "a", true);
        c3.update("w", "b", true);
        c3.update("w", "a", false);
        check(c3.search("w").docs.toString().equals("[b]"), "a tombstone must drop the document");
        c3.update("w", "a", true);
        check(c3.search("w").docs.toString().equals("[a, b]"), "re-adding after a delete must restore the document");
        c3.update("w", "b", false);
        c3.update("w", "b", false);
        SearchResult afterDeletes = c3.search("w");
        check(afterDeletes.ok && afterDeletes.docs.toString().equals("[a]"), "a repeated delete is idempotent");
        check(afterDeletes.tokens.size() == 6, "six updates means six tokens are probed");

        // ---- 3. boundaries: empty index, absent keyword, single entry ------------------------------
        IndexServer s4 = new IndexServer();
        SseClient c4 = new SseClient(MASTER, s4);
        SearchResult empty = c4.search("nothing");
        check(empty.ok && empty.docs.isEmpty(), "an empty index must answer an empty result");
        check(s4.lookups == 0, "an unknown keyword must cost zero lookups, saw " + s4.lookups);
        c4.update("only", "x", true);
        SearchResult single = c4.search("only");
        check(single.ok && single.docs.toString().equals("[x]"), "a single-entry keyword must resolve");
        check(single.lookups == 1, "the single entry must cost one lookup");
        check(c4.search("other").docs.isEmpty(), "an absent keyword next to a present one must be empty");

        // ---- 4. forward privacy and the leakage budget ---------------------------------------------
        IndexServer s5 = new IndexServer();
        SseClient c5 = new SseClient(MASTER, s5);
        List<Long> tokens = new ArrayList<>();
        for (int i = 0; i < 6; i++) tokens.add(c5.update("fwd", "d" + i, true));
        check(new HashSet<>(tokens).size() == tokens.size(), "every update token must be distinct (forward privacy)");
        check(tokensArePrfSequence(c5, "fwd"), "the token sequence must be reproducible from the state key");
        check(c5.stateKeyOf("fwd") == c5.stateKeyOf("fwd"), "the keyword state key must be stable across calls");
        check(c5.stateKeyOf("fwd") != c5.stateKeyOf("other"), "distinct keywords must have distinct state keys");

        Set<Long> patternA = new TreeSet<>(c5.search("fwd").tokens);
        Set<Long> patternB = new TreeSet<>(c5.search("fwd").tokens);
        check(patternA.equals(patternB), "the search pattern of a repeated keyword is (documented) identical");
        c5.update("other", "z", true);
        Set<Long> patternC = new TreeSet<>(c5.search("other").tokens);
        Set<Long> shared = new TreeSet<>(patternA);
        shared.retainAll(patternC);
        check(shared.isEmpty(), "tokens from different keywords must never be shared");

        IndexServer s6 = new IndexServer();
        SseClient c6 = new SseClient(MASTER, s6);
        for (int i = 0; i < 150; i++) c6.update("bulk", "doc" + i, true);
        c6.update("rare", "target", true);
        long before = s6.lookups;
        SearchResult rare = c6.search("rare");
        check(rare.ok && rare.docs.toString().equals("[target]"), "the rare keyword must resolve inside a large index");
        check(s6.lookups - before == 1, "search cost must follow the keyword, not the index: " + (s6.lookups - before));
        check(s6.size() == 151, "the bulk index must hold one entry per update");

        // ---- 5. proof of correctness: honest, omitted, altered, injected --------------------------
        IndexServer s7 = new IndexServer();
        SseClient c7 = new SseClient(MASTER, s7);
        for (int i = 0; i < 5; i++) c7.update("proof", "p" + i, true);
        check(c7.search("proof").ok, "the honest five-entry index must verify");
        long victim = c7.search("proof").tokens.get(2);

        IndexServer s8 = new IndexServer();
        SseClient c8 = new SseClient(MASTER, s8);
        for (int i = 0; i < 5; i++) c8.update("proof", "p" + i, true);
        long honestRoot = s8.root();
        s8.store.remove(victim);
        check(s8.root() != honestRoot && c8.expectedRoot() != s8.root(),
                "dropping an entry must change the root the client recomputes");
        SearchResult omitted = c8.search("proof");
        check(!omitted.ok, "omitting an entry must break the search");
        check(omitted.reason.contains("missing") || omitted.reason.contains("root"),
                "omission must be attributed (missing entry or root mismatch): " + omitted.reason);

        IndexServer s9 = new IndexServer();
        SseClient c9 = new SseClient(MASTER, s9);
        for (int i = 0; i < 5; i++) c9.update("proof", "p" + i, true);
        Entry tamperedEntry = s9.store.get(victim).tampered();
        check(Merkle.leaf(tamperedEntry.token, tamperedEntry.ct, tamperedEntry.tag) != c9.commitment.get(victim),
                "the stored commitment must reject an altered ciphertext");
        s9.store.put(victim, tamperedEntry);
        SearchResult altered = c9.search("proof");
        check(!altered.ok, "altering a ciphertext must break the search");
        check(altered.reason.contains("altered") || altered.reason.contains("root"),
                "alteration must be attributed: " + altered.reason);

        IndexServer s10 = new IndexServer();
        SseClient c10 = new SseClient(MASTER, s10);
        for (int i = 0; i < 3; i++) c10.update("proof", "p" + i, true);
        long injectedToken = Prf.prf(0x1234L, 99L);
        s10.put(new Entry(injectedToken, 0L, new byte[] {9, 9}));
        SearchResult injected = c10.search("proof");
        check(!injected.ok, "an entry injected under a foreign token must be detected by the committed root");
        check(s10.size() == 4 && c10.commitment.size() == 3 && !c10.commitment.containsKey(injectedToken),
                "the injected entry must never enter the client's commitments");

        IndexServer s11 = new IndexServer();
        SseClient c11 = new SseClient(MASTER, s11);
        for (int i = 0; i < 4; i++) c11.update("proof", "p" + i, true);
        check(c11.expectedRoot() == s11.root(), "the client must recompute the honest server root from its commitments");
        Entry saved = s11.store.get(c11.search("proof").tokens.get(0));
        s11.store.put(saved.token, new Entry(saved.token, saved.tag + 1L, saved.ct.clone()));
        SearchResult tagBroken = c11.search("proof");
        check(!tagBroken.ok, "a forged tag must break the search: " + tagBroken.reason);

        // ---- 6. confidentiality of the server-visible bytes ---------------------------------------
        IndexServer s12 = new IndexServer();
        SseClient c12 = new SseClient(MASTER, s12);
        c12.update("secretkeyword", "docsecret", true);
        c12.update("secretkeyword", "docsecret2", true);
        long exposed = 0;
        int plaintextHits = 0;
        for (Entry e : s12.store.values()) {
            exposed += e.ct.length;
            if (indexOf(e.ct, "secretkeyword".getBytes()) >= 0) plaintextHits++;
            if (indexOf(e.ct, "docsecret".getBytes()) >= 0) plaintextHits++;
        }
        check(exposed > 0 && plaintextHits == 0, "no keyword or document id may appear in a stored ciphertext");
        boolean tokenIsPrfOfKeyword = false;
        for (long t : s12.store.keySet()) if (t == Prf.prfString(MASTER, "secretkeyword")) tokenIsPrfOfKeyword = true;
        check(!tokenIsPrfOfKeyword, "the stored token must not be the raw PRF of the keyword");

        IndexServer s13 = new IndexServer();
        SseClient c13 = new SseClient(MASTER, s13);
        long token13 = c13.update("q", "d9", true);
        Entry e13 = s13.store.get(token13);
        long rightKey = c13.entryKey.get(token13);
        long nonce13 = c13.entryNonce.get(token13);
        check(Prf.tag(rightKey ^ 1L, nonce13, e13.ct) != e13.tag, "a foreign entry key must fail the tag check");
        byte[] withRight = Prf.xorStream(rightKey, nonce13, e13.ct);
        check(withRight[0] == 1 && SseClient.decodeDoc(withRight).equals("d9"), "the owner recovers the exact payload");
        check(c13.stateKeyOf("q") != 0, "the keyword state key must be a real derived value");

        // ---- 7. determinism ------------------------------------------------------------------------
        IndexServer d1 = new IndexServer();
        SseClient dc1 = new SseClient(MASTER, d1);
        IndexServer d2 = new IndexServer();
        SseClient dc2 = new SseClient(MASTER, d2);
        for (int i = 0; i < 8; i++) { dc1.update("det", "doc" + (i % 3), true); dc2.update("det", "doc" + (i % 3), true); }
        check(d1.store.keySet().equals(d2.store.keySet()), "the same workload must issue the same tokens");
        check(d1.root() == d2.root(), "the same workload must produce the same index root");
        check(dc1.search("det").docs.equals(dc2.search("det").docs), "the same workload must produce the same hits");

        System.out.println("index entries / updates : " + s2.size() + " / " + workload.size());
        System.out.println("keywords oracle-checked : " + keywords.length);
        System.out.println("rare-keyword search cost: 1 lookup in an index of " + s6.size());
        System.out.println("plaintext bytes on server: " + plaintextHits + " (of " + exposed + " ciphertext bytes)");
        System.out.println("proof tampering detected: omission, alteration, forged tag, injection");
        System.out.println("PASS");
    }
}
