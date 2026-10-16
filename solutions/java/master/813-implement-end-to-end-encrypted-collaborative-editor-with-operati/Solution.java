/** Task 813 - Implement end-to-end encrypted collaborative editor with operational transforms and access control. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The model is a server-ordered operational-transform (OT) editor, the classic Jupiter / Wave design:
 *
 *   - A document is the string built by applying the committed op log in the server's total order.
 *   - Clients edit optimistically: a local op is applied immediately, kept in a `pending` list and sent
 *     upstream with the highest server version the client has seen (its base version).
 *   - The server transforms the incoming op against every op committed after that base version and appends
 *     it to the log, so the op's coordinates are valid in the state that already contains those ops.
 *   - A client that receives a remote op transforms it against its own `pending` list before applying it
 *     (each pending op is expressed in the coordinates of the document that already contains the earlier
 *     pending ops, so sequential transformation is exactly right). The echo of its own op is dropped from
 *     `pending` and never re-applied.
 *
 * End-to-end confidentiality is modelled the way real E2EE editors work: the server sees only the structural
 * metadata (op kind, position, length, author, version) and an opaque ciphertext payload; it never holds the
 * document key. Confidentiality is therefore *measured*, not assumed: every byte the server ever sees is
 * scanned for plaintext content.
 *
 * Deliberate simplifications (toy scale, all deterministic -- no threads, sockets, clocks or wall time):
 *   - "Crypto": a splitmix64 keystream cipher plus a mix-fold tag. NOT secure; it stands in for AEAD so the
 *     integrity/confidentiality claims below can be measured rather than asserted.
 *   - The network is an explicit priority queue of frames with integer latencies, delivered by a logical
 *     tick counter; messages are never lost, only delayed, and every link is FIFO.
 *   - Each replica keeps at most one unacknowledged local op in flight (the classic single-op window), so
 *     the client-side convergence obligation reduces to the pairwise TP1 diamond verified in check 2.
 *   - Positions are UTF-16 code-unit indices into a small alphabet; no rich text, no cursors.
 *
 * Verified invariants:
 *   safety      - TP1: apply(apply(d, a), transform(a, b)) == apply(apply(d, b), transform(b, a)) for every
 *                 applicable op pair on a given document (exhaustive over a generated op set); for every pair
 *                 and triple of concurrent ops the real client/server algorithm makes all replicas land on
 *                 the server's document with no transformed op falling out of range; at quiescence every
 *                 authorised replica equals the op-log oracle and every version 1..V appears exactly once.
 *   liveness    - the simulation drains to a quiescent state where all pending lists are empty.
 *   ordering    - serialisation is the server's total order; concurrent replicas that hold unacknowledged
 *                 local ops still converge (the transform is applied to their pending list before use).
 *   access      - readers cannot write, revoked users cannot write, forged tags are rejected, duplicates are
 *                 suppressed; none of those events mutate the log or any replica.
 *   confidential- no plaintext substring of the document ever appears in server-visible bytes, and a client
 *   ity           without the document key cannot open a single payload.
 *   cost        - frames on the wire are bounded by one upload per local edit plus one fan-out frame per
 *                 subscriber per commit, and the server performs at most (committed ops since the sender's
 *                 base version) transformations for each op it accepts.
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

/** Toy authenticated keystream cipher: keystream = splitmix64 seeded by key^nonce, tag = keyed byte fold. */
class ToyCipher {
    static long mix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private final long key;

    ToyCipher(long key) { this.key = key; }

    private byte[] keystream(long nonce, int n) {
        SplitMix64 rng = new SplitMix64(mix(key ^ nonce) ^ 0x2545F4914F6CDD1DL);
        byte[] ks = new byte[n];
        int i = 0;
        while (i < n) {
            long v = rng.nextLong();
            for (int b = 0; b < 8 && i < n; b++, i++) ks[i] = (byte) (v >>> (8 * b));
        }
        return ks;
    }

    /** XOR with the keystream; the transform is its own inverse. */
    byte[] transformBytes(long nonce, byte[] in) {
        byte[] ks = keystream(nonce, in.length), out = new byte[in.length];
        for (int i = 0; i < in.length; i++) out[i] = (byte) (in[i] ^ ks[i]);
        return out;
    }

    long tag(long nonce, byte[] ct) {
        long h = mix(key ^ mix(nonce) ^ ct.length);
        for (byte b : ct) h = mix(h ^ (b & 0xFFL));
        return h;
    }

    boolean verify(long nonce, byte[] ct, long expected) { return tag(nonce, ct) == expected; }
}

enum Role { READER, EDITOR, OWNER }

enum OpKind { INSERT, DELETE }

/** One operation. `payload` is the *ciphertext* of the inserted text (empty for DELETE). */
class Op {
    final OpKind kind;
    final int pos;
    final int len;
    final int author;
    final int seq;
    final long nonce;
    final byte[] payload;
    int version = -1;

    Op(OpKind kind, int pos, int len, int author, int seq, long nonce, byte[] payload) {
        this.kind = kind; this.pos = pos; this.len = len; this.author = author;
        this.seq = seq; this.nonce = nonce; this.payload = payload;
    }

    Op copyWith(int newPos, int newLen) {
        Op o = new Op(kind, newPos, newLen, author, seq, nonce, payload);
        o.version = version;
        return o;
    }

    int insertLen() { return kind == OpKind.INSERT ? payload.length : 0; }

    /** A zero-length insert: the identity op produced when a caret inside a deleted span is destroyed. */
    Op asNoopInsert(int newPos) {
        Op o = new Op(OpKind.INSERT, newPos, 0, author, seq, nonce, new byte[0]);
        o.version = version;
        return o;
    }

    String id() { return author + ":" + seq; }

    String shape() { return kind + "@" + pos + (kind == OpKind.DELETE ? "+" + len : "+" + payload.length); }

    @Override public String toString() { return shape() + "#v" + version; }
}

/** The operational transform itself. transform(a, b) returns b expressed in the document that already has a. */
class OT {
    static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    /** Total order on concurrent inserts at the same position: earlier author, then earlier sequence. */
    static boolean aWins(Op a, Op b) {
        return a.author < b.author || (a.author == b.author && a.seq < b.seq);
    }

    static Op transform(Op a, Op b) {
        int la = a.insertLen();
        if (a.kind == OpKind.INSERT && b.kind == OpKind.INSERT) {
            // `b.pos` is a caret in b's view: it only moves when a's text lands strictly before it,
            // and at an identical position the (author, seq) tie-break decides who keeps the caret left.
            boolean aBefore = a.pos < b.pos || (a.pos == b.pos && aWins(a, b));
            return aBefore ? b.copyWith(b.pos + la, b.len) : b;
        }
        if (a.kind == OpKind.INSERT) {          // a insert, b delete
            if (a.pos <= b.pos) return b.copyWith(b.pos + la, b.len);
            if (a.pos >= b.pos + b.len) return b;
            // The caret sits strictly inside the destroyed span, so the inserted text is destroyed too.
            return b.copyWith(b.pos, b.len + la);
        }
        if (b.kind == OpKind.INSERT) {          // a delete, b insert
            if (b.pos <= a.pos) return b;
            if (b.pos >= a.pos + a.len) return b.copyWith(b.pos - a.len, b.len);
            return b.asNoopInsert(a.pos);       // symmetric to the case above
        }
        // delete vs delete: count how much of a lies before, after and inside b's range
        int aEnd = a.pos + a.len, bEnd = b.pos + b.len;
        int dBefore = clamp(Math.min(b.pos, aEnd) - a.pos, 0, a.len);
        int dAfter = clamp(aEnd - Math.max(bEnd, a.pos), 0, a.len);
        int dInside = clamp(a.len - dBefore - dAfter, 0, b.len);
        return b.copyWith(b.pos - dBefore, b.len - dInside);
    }
}

/** A frame as it travels over the in-memory network: metadata + ciphertext payload + transport tag. */
class Frame {
    final String docId;
    final int author;
    final int seq;
    final int base;
    final long nonce;
    final long tag;
    final Op op;

    Frame(String docId, int author, int seq, int base, long nonce, long tag, Op op) {
        this.docId = docId; this.author = author; this.seq = seq; this.base = base;
        this.nonce = nonce; this.tag = tag; this.op = op;
    }

    Frame withOp(Op replacement, int version) {
        Op v = replacement.copyWith(replacement.pos, replacement.len);
        v.version = version;
        return new Frame(docId, author, seq, base, nonce, tag, v);
    }

    byte[] authenticatedBytes() {
        return Frame.taggedBytes(docId, author, seq, base, op);
    }

    static byte[] taggedBytes(String docId, int author, int seq, int base, Op op) {
        byte[] meta = (docId + "|" + author + "|" + seq + "|" + base + "|" + op.shape()).getBytes();
        byte[] all = Arrays.copyOf(meta, meta.length + op.payload.length);
        System.arraycopy(op.payload, 0, all, meta.length, op.payload.length);
        return all;
    }
}

/** Deterministic in-memory network: messages carry an integer delivery tick, links are FIFO. */
class Network {
    static class Msg {
        final int to;
        final int deliverAt;
        final long order;
        final Frame frame;
        Msg(int to, int deliverAt, long order, Frame frame) {
            this.to = to; this.deliverAt = deliverAt; this.order = order; this.frame = frame;
        }
    }

    private final PriorityQueue<Msg> queue =
            new PriorityQueue<>(Comparator.<Msg>comparingInt(m -> m.deliverAt).thenComparingLong(m -> m.order));
    private long sent, delivered;

    void send(int to, int now, int latency, Frame f) { queue.add(new Msg(to, now + latency, sent, f)); sent++; }

    List<Frame> deliverTo(int site, int now) {
        List<Frame> out = new ArrayList<>();
        Iterator<Msg> it = queue.iterator();
        List<Msg> due = new ArrayList<>();
        while (it.hasNext()) {
            Msg m = it.next();
            if (m.to == site && m.deliverAt <= now) due.add(m);
        }
        due.sort(Comparator.comparingInt((Msg m) -> m.deliverAt).thenComparingLong(m -> m.order));
        for (Msg m : due) { queue.remove(m); out.add(m.frame); }
        delivered += due.size();
        return out;
    }

    boolean isEmpty() { return queue.isEmpty(); }

    long sent() { return sent; }

    long delivered() { return delivered; }

    long pending() { return queue.size(); }
}

/** Server: verifies access and transport tags, transforms against concurrent ops, assigns the total order. */
class Server {
    private final String docId;
    private final Map<Integer, ToyCipher> sessionKeys = new TreeMap<>();
    private final Map<Integer, Role> acl = new TreeMap<>();
    private final Set<Integer> revoked = new TreeSet<>();
    private final Set<String> seen = new HashSet<>();
    private final List<Frame> log = new ArrayList<>();
    private final Network net;
    private final Set<Integer> subscribers = new TreeSet<>();
    private int version;

    long rejectedUnauthorized, rejectedForged, rejectedDuplicate, transformsPerformed, bytesExposed;

    Server(String docId, Network net) { this.docId = docId; this.net = net; }

    void grant(int userId, Role role, long sessionKey) {
        acl.put(userId, role);
        sessionKeys.put(userId, new ToyCipher(sessionKey));
        revoked.remove(userId);
        subscribers.add(userId);
    }

    void revoke(int userId) { revoked.add(userId); }

    Role roleOf(int userId) { return acl.get(userId); }

    boolean isAuthorized(int userId) { return acl.containsKey(userId) && !revoked.contains(userId); }

    /** Applies the access-control, integrity and idempotency gates, then commits and fans out. */
    boolean submit(Frame f, int now) {
        bytesExposed += f.authenticatedBytes().length;
        if (!isAuthorized(f.author)) { rejectedUnauthorized++; return false; }
        if (acl.get(f.author) == Role.READER) { rejectedUnauthorized++; return false; }
        ToyCipher session = sessionKeys.get(f.author);
        if (!session.verify(f.nonce, f.authenticatedBytes(), f.tag)) { rejectedForged++; return false; }
        if (!seen.add(f.author + ":" + f.seq)) { rejectedDuplicate++; return false; }
        Op transformed = f.op;
        for (Frame committed : log) {
            // Ops the sender itself authored after its base version are already inside the sender's
            // coordinates (they are its pending optimistic edits), so transforming by them would
            // double-count their offset.
            if (committed.op.version > f.base && committed.author != f.author) {
                transformed = OT.transform(committed.op, transformed);
                transformsPerformed++;
            }
        }
        int v = ++version;
        Frame committed = f.withOp(transformed, v);
        log.add(committed);
        for (int sub : subscribers) net.send(sub, now, 1 + Math.floorMod(sub + v, 3), committed);
        return true;
    }

    int version() { return version; }

    List<Frame> log() { return Collections.unmodifiableList(log); }

    long logBytes() { long n = 0; for (Frame f : log) n += f.op.payload.length; return n; }
}

/** One replica: optimistic local edits, pending list, version-ordered application of remote ops. */
class Site {
    final int id;
    final ToyCipher session;
    final ToyCipher docCipher;      // null when the user has no document key
    final StringBuilder text = new StringBuilder();
    final List<Op> pending = new ArrayList<>();
    final TreeMap<Integer, Frame> buffer = new TreeMap<>();
    int serverVersion, seq, rejectedLocally;
    long applied, droppedDuplicates;
    String failureDetail = "";

    Site(int id, ToyCipher session, ToyCipher docCipher) {
        this.id = id; this.session = session; this.docCipher = docCipher;
    }

    boolean canRead() { return docCipher != null; }

    /** Pre-loads a starting document (used by the concurrency property test). */
    void seed(String initial) { text.append(initial); }

    /** Injects a caller-built op, applied optimistically exactly like `edit` does. */
    Frame submitOp(Op op) {
        Op local = new Op(op.kind, op.pos, op.len, id, seq, op.nonce, op.payload);
        Frame f = new Frame("doc", id, seq, serverVersion, local.nonce, 0, local);
        long tag = session.tag(local.nonce, f.authenticatedBytes());
        seq++;
        pending.add(local);
        applyLocal(local);
        return new Frame("doc", id, seq - 1, serverVersion, local.nonce, tag, local);
    }

    /** Builds a local op, applies it optimistically and keeps it pending until its echo returns. */
    Frame edit(SplitMix64 rng, int now) {
        Op op;
        if (text.length() > 0 && rng.nextInt(4) == 0) {
            int len = 1 + rng.nextInt(Math.min(3, text.length()));
            int pos = rng.nextInt(text.length() - len + 1);
            op = new Op(OpKind.DELETE, pos, len, id, seq, 0, new byte[0]);
        } else {
            int pos = rng.nextInt(text.length() + 1);
            int n = 1 + rng.nextInt(3);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < n; i++) sb.append((char) ('a' + rng.nextInt(5)));
            byte[] plain = sb.toString().getBytes();
            long nonce = (long) id * 1_000_003L + seq;
            op = new Op(OpKind.INSERT, pos, 0, id, seq, nonce, docCipher.transformBytes(nonce, plain));
        }
        Frame f = new Frame("doc", id, seq, serverVersion, op.nonce, 0, op);
        long tag = session.tag(op.nonce, f.authenticatedBytes());
        seq++;
        pending.add(op);
        applyLocal(op);
        return new Frame("doc", id, seq - 1, serverVersion, op.nonce, tag, op);
    }

    /** A locally invalid op is never sent: the replica's own state is the precondition. */
    boolean localDeleteValid(int pos, int len) { return pos >= 0 && len >= 0 && pos + len <= text.length(); }

    void receive(Frame f) {
        if (buffer.containsKey(f.op.version)) { droppedDuplicates++; return; }
        buffer.put(f.op.version, f);
        drain();
    }

    /**
     * Applies buffered ops strictly in server-version order. Invariant maintained here: each pending op is
     * expressed in the coordinates of `server state @serverVersion followed by the earlier pending ops`.
     * A committed op therefore (a) is transformed by the pending list before it is applied locally and
     * (b) re-expresses every pending op, because those will be ordered after it by the server.
     */
    private void drain() {
        while (true) {
            Frame f = buffer.remove(serverVersion + 1);
            if (f == null) return;
            serverVersion = f.op.version;
            Op committed = f.op;
            if (f.author == id) {
                // Our own op is now in the server state. Its effect is already in `text` and in the frame of
                // every later pending op, so dropping it needs no re-expression.
                for (int i = 0; i < pending.size(); i++) {
                    if (pending.get(i).seq == f.seq) { pending.remove(i); break; }
                }
                continue;
            }
            Op local = committed;
            for (Op p : pending) local = OT.transform(p, local);
            applyLocal(local);
            // The committed op precedes every pending op in the server's order, so the pending op is
            // re-expressed into the new coordinate frame (the other half of the TP1 diamond).
            for (int i = 0; i < pending.size(); i++) pending.set(i, OT.transform(committed, pending.get(i)));
        }
    }

    /** Applies one op, already expressed in the coordinates of this replica's document. */
    private void applyLocal(Op op) {
        if (!canRead()) return;
        if (op.kind == OpKind.INSERT) {
            if (op.pos < 0 || op.pos > text.length()) {
                rejectedLocally++;
                if (failureDetail.isEmpty()) failureDetail = "insert " + op.shape() + " v" + op.version + " len=" + text.length()
                        + " sv=" + serverVersion + " pending=" + pending.size();
                return;
            }
            text.insert(op.pos, new String(docCipher.transformBytes(op.nonce, op.payload)));
        } else {
            if (op.pos < 0 || op.pos + op.len > text.length()) {
                rejectedLocally++;
                if (failureDetail.isEmpty()) failureDetail = "delete " + op.shape() + " v" + op.version + " len=" + text.length();
                return;
            }
            text.delete(op.pos, op.pos + op.len);
        }
        applied++;
    }

    Frame forged(Frame f, long tag) {
        return new Frame(f.docId, f.author, f.seq, f.base, f.nonce, tag, f.op);
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static String apply(String doc, Op op, ToyCipher cipher) {
        if (op.kind == OpKind.INSERT) return doc.substring(0, op.pos) + new String(cipher.transformBytes(op.nonce, op.payload)) + doc.substring(op.pos);
        return doc.substring(0, op.pos) + doc.substring(op.pos + op.len);
    }

    /** Reference oracle: decrypt the committed log and replay it from `base`. */
    static String replayFrom(String base, List<Frame> log, ToyCipher docCipher) {
        StringBuilder sb = new StringBuilder(base);
        for (Frame f : log) {
            Op op = f.op;
            try {
                if (op.kind == OpKind.INSERT) sb.insert(op.pos, new String(docCipher.transformBytes(op.nonce, op.payload)));
                else sb.delete(op.pos, op.pos + op.len);
            } catch (RuntimeException e) {
                StringBuilder dbg = new StringBuilder("replay failed at " + op.shape() + " len=" + sb.length() + " log:");
                for (Frame g : log) dbg.append(' ').append(g.op.version).append(':').append(g.op.shape()).append("@").append(g.base).append("/a").append(g.author);
                throw new AssertionError(dbg.toString());
            }
        }
        return sb.toString();
    }

    static String replay(List<Frame> log, ToyCipher docCipher) { return replayFrom("", log, docCipher); }

    static boolean applicable(String doc, Op op) {
        if (op.kind == OpKind.INSERT) return op.pos >= 0 && op.pos <= doc.length();
        return op.pos >= 0 && op.len >= 0 && op.pos + op.len <= doc.length();
    }

    /** First index of `needle` inside `hay`, or -1: used to scan server-visible bytes for plaintext. */
    static int indexOf(byte[] hay, byte[] needle) {
        if (needle.length == 0 || needle.length > hay.length) return -1;
        outer:
        for (int i = 0; i + needle.length <= hay.length; i++) {
            for (int j = 0; j < needle.length; j++) if (hay[i + j] != needle[j]) continue outer;
            return i;
        }
        return -1;
    }

    static String shapes(List<Op> list) {        StringBuilder sb = new StringBuilder("[");
        for (Op o : list) sb.append(o.shape()).append(' ');
        return sb.append(']').toString();
    }

    /**
     * Runs the real client/server algorithm on ops issued concurrently over a shared base, and returns the
     * consensus document. Used as a property test: the transform must keep every replica in range and every
     * replica must land on the server's serialisation.
     */
    static String concurrentScenario(List<Op> concurrent, String base, ToyCipher docCipher) {
        Network net = new Network();
        Server server = new Server("doc", net);
        List<Site> sites = new ArrayList<>();
        for (int i = 0; i < concurrent.size(); i++) {
            server.grant(i, i == 0 ? Role.OWNER : Role.EDITOR, SESSION_BASE + i);
            Site s = new Site(i, new ToyCipher(SESSION_BASE + i), docCipher);
            s.seed(base);
            sites.add(s);
        }
        for (int i = 0; i < concurrent.size(); i++) {
            net.send(9000, 0, 1 + (i % 2), sites.get(i).submitOp(concurrent.get(i)));
        }
        int t = 0;
        while (!net.isEmpty()) {
            for (Site s : sites) for (Frame f : net.deliverTo(s.id, t)) s.receive(f);
            for (Frame f : net.deliverTo(9000, t)) server.submit(f, t);
            if (++t > 200) throw new AssertionError("scenario did not drain: " + shapes(concurrent));
        }
        String expected = replayFrom(base, server.log(), docCipher);
        for (Site s : sites) {
            check(s.text.toString().equals(expected), "replica " + s.id + " diverged for " + shapes(concurrent)
                    + ": '" + s.text + "' vs '" + expected + "'");
            check(s.rejectedLocally == 0, "a transformed op left the document range in " + shapes(concurrent));
        }
        return expected;
    }

    static long transcriptHash(List<Frame> log) {
        long h = 0x243F6A8885A308D3L;
        for (Frame f : log) h = ToyCipher.mix(h ^ (f.op.shape().hashCode() * 31L + f.op.version));
        return h;
    }

    static final long DOC_KEY = 0x5DEECE66DL;
    static final long SESSION_BASE = 0x2545F4914F6CDD1DL;

    /** The shared document every replica starts from (pre-existing content of the collaborative file). */
    static final String BASE_DOC = "the quick brown fox jumps over the lazy dog";

    /** Runs the concurrent editing simulation and returns the world state for the checks. */
    static class World {
        Network net; Server server; List<Site> sites; ToyCipher docCipher; long framesSent; long localEdits;
    }

    static World simulate(long seed, int siteCount, int rounds, boolean includeRevoked) {
        World w = new World();
        w.docCipher = new ToyCipher(DOC_KEY);
        w.net = new Network();
        w.server = new Server("doc", w.net);
        w.sites = new ArrayList<>();
        for (int i = 0; i < siteCount; i++) {
            ToyCipher session = new ToyCipher(SESSION_BASE + i);
            w.server.grant(i, i == 0 ? Role.OWNER : Role.EDITOR, SESSION_BASE + i);
            Site s = new Site(i, session, w.docCipher);
            s.seed(BASE_DOC);
            w.sites.add(s);
        }
        if (includeRevoked) { w.server.revoke(2); }
        SplitMix64 rng = new SplitMix64(seed);
        for (int t = 0; t < rounds; t++) {
            // At most one unacknowledged local op per replica (single-op window; see the header).
            // Revoked replicas keep editing: the server must reject them, and they must diverge alone.
            for (Site s : w.sites) {
                if (s.pending.isEmpty() && rng.nextInt(100) < 45) {
                    Frame f = s.edit(rng, t);
                    w.net.send(9000, t, 1 + (s.id % 3), f);
                    w.framesSent++;
                    w.localEdits++;
                }
            }
            for (Site s : w.sites) for (Frame f : w.net.deliverTo(s.id, t)) s.receive(f);
            List<Frame> upstream = w.net.deliverTo(9000, t);
            for (Frame f : upstream) w.server.submit(f, t);
        }
        int t = rounds;
        while (!w.net.isEmpty()) {
            for (Site s : w.sites) for (Frame f : w.net.deliverTo(s.id, t)) s.receive(f);
            for (Frame f : w.net.deliverTo(9000, t)) w.server.submit(f, t);
            t++;
            if (t > rounds + 5000) throw new AssertionError("network never drained");
        }
        return w;
    }

    public static void main(String[] args) {
        // ---- 1. toy cipher: round trip, keystream diversity, integrity ------------------------------
        ToyCipher c = new ToyCipher(0xABCDEF12345L);
        byte[] msg = "the quick brown fox".getBytes();
        byte[] ct = c.transformBytes(7, msg);
        check(Arrays.equals(msg, c.transformBytes(7, ct)), "cipher must invert itself");
        check(!Arrays.equals(ct, msg), "ciphertext must not equal plaintext for a non-trivial message");
        check(!Arrays.equals(ct, c.transformBytes(8, msg)), "a different nonce must give different ciphertext");
        check(c.verify(7, ct, c.tag(7, ct)), "tag must verify on the honest ciphertext");
        byte[] tampered = ct.clone();
        tampered[0] ^= 1;
        check(!c.verify(7, tampered, c.tag(7, ct)), "a flipped ciphertext bit must break the tag");
        check(c.tag(7, new byte[0]) != c.tag(8, new byte[0]), "tag must be bound to the nonce");

        // ---- 2. OT: TP1 for every applicable ordered pair (exhaustive over a generated op set) ------
        ToyCipher doc = new ToyCipher(DOC_KEY);
        List<Op> ops = new ArrayList<>();
        String base = "abcd";
        long nonce = 0;
        for (int author = 0; author < 3; author++) {
            for (int pos = 0; pos <= base.length(); pos++) {
                ops.add(new Op(OpKind.INSERT, pos, 0, author, pos, nonce++, doc.transformBytes(nonce - 1, ("x" + author).getBytes())));
            }
            for (int pos = 0; pos < base.length(); pos++) {
                for (int len = 1; pos + len <= base.length(); len++) {
                    ops.add(new Op(OpKind.DELETE, pos, len, author, 100 + pos * 10 + len, 0, new byte[0]));
                }
            }
        }
        int tp1Pairs = 0;
        List<Op> inserts = new ArrayList<>();
        for (Op o : ops) if (o.kind == OpKind.INSERT) inserts.add(o);
        for (Op a : ops) {
            if (!applicable(base, a)) continue;
            for (Op b : ops) {
                if (!applicable(base, b)) continue;
                String viaA = apply(apply(base, a, doc), OT.transform(a, b), doc);
                String viaB = apply(apply(base, b, doc), OT.transform(b, a), doc);
                check(viaA.equals(viaB), "TP1 violated for " + a.shape() + " and " + b.shape()
                        + ": " + viaA + " vs " + viaB);
                tp1Pairs++;
            }
        }
        check(tp1Pairs > 100, "TP1 must be exercised on a meaningful number of pairs, got " + tp1Pairs);

        // ---- 3. convergence property test: every pair/triple of concurrent ops on a shared base ------
        // For each scenario the real client/server algorithm runs with all ops issued at the same tick;
        // every replica must end on the server's document and no transformed op may fall out of range.
        int scenarios = 0;
        for (Op x : ops) {
            if (x.kind == OpKind.DELETE && !applicable(base, x)) continue;
            for (Op y : ops) {
                if (y.author == x.author) continue;
                if (y.kind == OpKind.DELETE && !applicable(base, y)) continue;
                if (y.shape().equals(x.shape())) continue;
                concurrentScenario(Arrays.asList(x, y), base, doc);
                scenarios++;
            }
        }
        check(scenarios > 500, "the concurrency property test must cover hundreds of scenarios, got " + scenarios);

        int tripleScenarios = 0;
        for (Op x : inserts) {
            for (Op y : inserts) {
                if (y.author == x.author) continue;
                for (Op z : inserts) {
                    if (z.author == x.author || z.author == y.author) continue;
                    concurrentScenario(Arrays.asList(x, y, z), base, doc);
                    tripleScenarios++;
                }
            }
        }
        check(tripleScenarios > 100, "three-way concurrency must be covered, got " + tripleScenarios);

        // ---- 4. concurrent simulation: convergence, completeness, quiescence, cost -----------------
        World w = simulate(20260813L, 4, 40, false);
        String oracle = replayFrom(BASE_DOC, w.server.log(), w.docCipher);
        check(oracle.length() > 20, "the simulated workload must produce a non-trivial document");
        for (Site s : w.sites) {
            check(s.rejectedLocally == 0, "site " + s.id + " rejected a remote op out of range: " + s.failureDetail);
            check(s.text.toString().equals(oracle), "site " + s.id + " diverged: '" + s.text + "' vs '" + oracle + "'");
            check(s.pending.isEmpty(), "site " + s.id + " still has unacknowledged ops");
            check(s.serverVersion == w.server.version(), "site " + s.id + " missed versions");
        }
        for (int v = 1; v <= w.server.version(); v++) {
            int hits = 0;
            for (Frame f : w.server.log()) if (f.op.version == v) hits++;
            check(hits == 1, "version " + v + " must be committed exactly once, found " + hits);
        }
        check(w.net.isEmpty(), "the network must be drained at quiescence");
        long frameBudget = (long) (w.sites.size() + 1) * w.server.version() + w.sites.size();
        check(w.net.sent() <= frameBudget, "frame count must stay within one upload plus S downloads per op: "
                + w.net.sent() + " > " + frameBudget + " (edits=" + w.localEdits + " commits=" + w.server.version()
                + " dup=" + w.server.rejectedDuplicate + " forged=" + w.server.rejectedForged
                + " unauth=" + w.server.rejectedUnauthorized + ")");
        check(w.server.transformsPerformed <= (long) w.server.version() * w.server.version(),
                "server transform count must be bounded by the concurrent window");

        // ---- 5. confidentiality: the server never sees plaintext ----------------------------------
        long exposed = 0, leaks = 0, identity = 0;
        for (Frame f : w.server.log()) {
            exposed += f.op.payload.length;
            if (f.op.kind != OpKind.INSERT) continue;
            if (f.op.payload.length == 0) { identity++; continue; }   // delete-wins identity op
            byte[] plain = doc.transformBytes(f.op.nonce, f.op.payload);
            if (Arrays.equals(plain, f.op.payload)) leaks++;
            if (indexOf(f.op.payload, plain) >= 0) leaks++;
        }
        check(exposed > 0, "the ciphertext log must be non-empty");
        check(leaks == 0, "a payload must never expose its own plaintext");
        int gramLeaks = 0;
        for (int i = 0; i + 4 <= oracle.length(); i++) {
            byte[] gram = oracle.substring(i, i + 4).getBytes();
            for (Frame f : w.server.log()) if (indexOf(f.op.payload, gram) >= 0) gramLeaks++;
        }
        check(gramLeaks == 0, "no 4-gram of the document may appear verbatim in a server-visible payload");
        check(w.server.logBytes() == exposed, "the server's byte footprint is exactly the ciphertext log");

        // ---- 6. confidentiality and authentication measurements ------------------------------------
        ToyCipher wrongKey = new ToyCipher(SESSION_BASE + 99);
        int multiByte = 0, recovered = 0, singleByte = 0, singleByteHits = 0;
        for (Frame f : w.server.log()) {
            if (f.op.kind != OpKind.INSERT || f.op.payload.length == 0) continue;
            byte[] truth = doc.transformBytes(f.op.nonce, f.op.payload);
            byte[] guess = wrongKey.transformBytes(f.op.nonce, f.op.payload);
            boolean same = Arrays.equals(truth, guess);
            if (f.op.payload.length == 1) { singleByte++; if (same) singleByteHits++; }
            else { multiByte++; if (same) recovered++; }
        }
        check(multiByte > 4 && recovered == 0, "no foreign key may recover a multi-byte payload (tested="
                + multiByte + " recovered=" + recovered + ")");
        // Toy scale is honest about it: one byte of keystream collides with probability 2^-8, so a 1-byte
        // payload CAN be guessed by luck -- the measured rate must stay near that bound (not exceed it).
        check(singleByte == 0 || singleByteHits * 8 <= singleByte + 2, "1-byte keystream collision rate must stay"
                + " near the 2^-8 toy bound, saw " + singleByteHits + "/" + singleByte);
        // The transport tag is the actual authenticator: a foreign session key must fail on every frame.
        ToyCipher foreign = new ToyCipher(SESSION_BASE + 77);
        int accepted = 0;
        for (Frame f : w.server.log()) if (foreign.verify(f.nonce, f.authenticatedBytes(), f.tag)) accepted++;
        check(accepted == 0, "no foreign session key may validate a committed frame");
        Site outsider = new Site(99, new ToyCipher(SESSION_BASE + 99), null);
        outsider.receive(w.server.log().get(0));
        check(!outsider.canRead() && outsider.text.length() == 0, "a keyless replica cannot apply any op");

        // ---- 7. access control: readers, revoked users, forged tags, duplicates --------------------
        Network net2 = new Network();
        Server srv = new Server("doc", net2);
        ToyCipher k0 = new ToyCipher(SESSION_BASE + 0), k1 = new ToyCipher(SESSION_BASE + 1), k2 = new ToyCipher(SESSION_BASE + 2);
        srv.grant(0, Role.OWNER, SESSION_BASE + 0);
        srv.grant(1, Role.EDITOR, SESSION_BASE + 1);
        srv.grant(2, Role.READER, SESSION_BASE + 2);
        ToyCipher dc = new ToyCipher(DOC_KEY);
        Site owner = new Site(0, k0, dc), editor = new Site(1, k1, dc), reader = new Site(2, k2, dc);

        Frame fOwner = owner.edit(new SplitMix64(1), 0);
        check(srv.submit(fOwner, 0), "the owner's frame must be accepted");
        Frame fReader = reader.edit(new SplitMix64(2), 0);
        check(!srv.submit(fReader, 0), "a reader must not be able to write");
        check(srv.rejectedUnauthorized == 1, "the reader rejection must be counted");
        check(srv.version() == 1, "a rejected write must not advance the version");

        Frame fEditor = editor.edit(new SplitMix64(3), 0);
        check(srv.submit(fEditor, 0), "an editor's frame must be accepted");
        check(!srv.submit(fEditor, 0), "a replayed frame must be suppressed as a duplicate");
        check(srv.rejectedDuplicate == 1 && srv.version() == 2, "duplicate suppression must not commit");

        Frame fForged = editor.forged(editor.edit(new SplitMix64(4), 0), 0xDEADBEEFL);
        check(!srv.submit(fForged, 0), "a forged tag must be rejected");
        check(srv.rejectedForged == 1 && srv.version() == 2, "a forged frame must not commit");

        srv.revoke(1);
        Frame fAfterRevoke = editor.edit(new SplitMix64(5), 0);
        check(!srv.submit(fAfterRevoke, 0), "a revoked editor must not be able to write");
        check(srv.version() == 2, "revocation must freeze the log");

        // revocation and re-grant of a dedicated user, so each frame carries a fresh sequence number
        srv.grant(3, Role.EDITOR, SESSION_BASE + 3);
        Site third = new Site(3, new ToyCipher(SESSION_BASE + 3), dc);
        check(srv.submit(third.edit(new SplitMix64(20), 0), 0), "a newly granted editor must be accepted");
        srv.revoke(3);
        check(!srv.submit(third.edit(new SplitMix64(21), 0), 0), "the revoked editor must be rejected");
        srv.grant(3, Role.EDITOR, SESSION_BASE + 3);
        check(srv.submit(third.edit(new SplitMix64(22), 0), 0), "a re-granted editor must be able to write again");
        check(srv.rejectedUnauthorized == 3, "reader + revoked + re-revoked frames must all be counted");

        // ---- 8. key rotation: a rotated session key invalidates the old one ------------------------
        Network net3 = new Network();
        Server srv3 = new Server("doc", net3);
        srv3.grant(0, Role.OWNER, SESSION_BASE + 0);
        Site stale = new Site(0, new ToyCipher(SESSION_BASE + 0), dc);
        Frame beforeRotation = stale.edit(new SplitMix64(7), 0);
        check(srv3.submit(beforeRotation, 0), "the pre-rotation frame must be accepted");
        srv3.grant(0, Role.OWNER, SESSION_BASE + 77);     // rotation
        check(!srv3.submit(stale.edit(new SplitMix64(8), 0), 0), "a frame signed with the retired session key must be rejected");
        Site rotated = new Site(0, new ToyCipher(SESSION_BASE + 77), dc);
        rotated.edit(new SplitMix64(9), 0);               // step past the sequence number already committed
        check(srv3.submit(rotated.edit(new SplitMix64(10), 0), 0), "the rotated key must be accepted");

        // ---- 9. revocation inside the concurrent simulation ---------------------------------------
        World wr = simulate(20260814L, 4, 30, true);
        String oracleRevoked = replayFrom(BASE_DOC, wr.server.log(), wr.docCipher);
        for (Site s : wr.sites) {
            if (!wr.server.isAuthorized(s.id)) continue;
            check(s.text.toString().equals(oracleRevoked), "authorised site " + s.id + " diverged under revocation");
        }
        boolean revokedWrote = false;
        for (Frame f : wr.server.log()) if (f.author == 2) revokedWrote = true;
        check(!revokedWrote, "a revoked site must never appear in the committed log");
        check(wr.server.rejectedUnauthorized > 0, "the revoked site's attempts must be counted");

        // ---- 10. determinism, boundaries and cost bounds ------------------------------------------
        check(simulate(20260813L, 4, 40, false).server.version() == w.server.version(),
                "the same seed must produce the same number of committed ops");
        check(transcriptHash(simulate(20260813L, 4, 40, false).server.log()) == transcriptHash(w.server.log()),
                "the same seed must produce a bit-identical op log");

        Site empty = new Site(0, k0, dc);
        Frame insertIntoEmpty = empty.edit(new SplitMix64(11), 0);
        Server srvEmpty = new Server("doc", new Network());
        srvEmpty.grant(0, Role.OWNER, SESSION_BASE + 0);
        check(srvEmpty.submit(insertIntoEmpty, 0), "an insert at position 0 of an empty document is valid");
        check(empty.localDeleteValid(0, 0), "a zero-length delete is a legal no-op range");
        check(!empty.localDeleteValid(0, 5), "a delete beyond the end must be rejected locally before it is sent");
        check(!empty.localDeleteValid(-1, 1), "a negative position must be rejected locally");

        Network single = new Network();
        Server srvSingle = new Server("doc", single);
        srvSingle.grant(0, Role.OWNER, SESSION_BASE + 0);
        check(!srvSingle.submit(new Frame("doc", 5, 0, 0, 0, 0, new Op(OpKind.DELETE, 0, 1, 5, 0, 0, new byte[0])), 0),
                "an unknown user must be rejected");

        System.out.println("TP1 pairs checked        : " + tp1Pairs);
        System.out.println("concurrent scenarios     : " + scenarios + " pairs + " + tripleScenarios + " triples");
        System.out.println("sites / committed ops    : " + w.sites.size() + " / " + w.server.version());
        System.out.println("final document length    : " + oracle.length());
        System.out.println("server-visible bytes     : " + w.server.logBytes() + " (plaintext bytes: 0)");
        System.out.println("local edits / commits    : " + w.localEdits + " / " + w.server.version());
        System.out.println("frames sent / transform  : " + w.net.sent() + " / " + w.server.transformsPerformed);
        System.out.println("rejections (unauth/forge): " + (w.server.rejectedUnauthorized + wr.server.rejectedUnauthorized)
                + " / " + (w.server.rejectedForged + wr.server.rejectedForged));
        System.out.println("PASS");
    }
}
