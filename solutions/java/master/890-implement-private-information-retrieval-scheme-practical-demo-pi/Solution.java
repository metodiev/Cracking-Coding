/** Task 890 - Implement private information retrieval scheme practical demo (PIR). (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * Two information-theoretic (two-server) PIR schemes are implemented over an in-process database of
 * bits, and both are measured rather than asserted:
 *
 * 1. Flat 2-server PIR (Chor-Goldreich-Kushilevitz-Sudan). The client draws a uniform mask q1 and
 *    defines q2 = q1 XOR e_i, so each server sees a uniformly random N-bit vector and neither can tell
 *    which bit is wanted. Server s answers with the XOR of the database bits selected by q_s, and the
 *    client recovers db[i] = a1 XOR a2. Communication is 2N bits per query.
 *
 * 2. Vectorized (Chor-Goldreich-Kushilevitz-Sudan / Stern style) PIR with the database laid out as an
 *    r x c matrix. The client sends each server a random r-bit row selector with u1 XOR u2 = e_row; the
 *    server answers with the c-bit word obtained by XOR-ing the selected rows. The client XORs the two
 *    words to recover row i and picks the column locally - the column index never leaves the client at
 *    all. The query is 2r bits and the answers 2c bits, i.e. 2(r + c) = O(sqrt N) instead of O(N), while
 *    the server still touches every bit of the selected rows - the honest trade-off of PIR without
 *    preprocessing.
 *
 * What is simplified, and what is *not* left to faith. The "servers" are objects and the network is a
 * method call, so no protocol message is actually sent; what is modelled exactly is the *view* of each
 * server, and the privacy checks measure that view statistically (bit frequencies per position, the
 * distribution of the query weight, and conditional frequencies given the target index). The
 * construction gives 1-server privacy - the demo verifies q1 XOR q2 = e_i exactly and states that two
 * colluding servers recover the index, which is the known limit of this scheme without more servers.
 * Server work is genuinely O(N) for the flat scheme and O(N) word operations for the vectorized one,
 * so the sublinearity claim is made only about communication and is asserted with exact bit counts.
 * Everything is deterministic: one splitmix64 stream per query, no threads, no clocks, no sockets.
 *
 * Verified: both schemes return the correct bit for every index of a 4096-bit database and for
 * non-square layouts; the query weight and per-position bit frequency of each server's view are
 * indistinguishable from uniform (measured over hundreds of queries) and the conditional frequency
 * given the target index agrees with the unconditional one; the two queries of a round XOR to exactly
 * one unit vector; the exact communication formulas 2N and 2(r+c) bits hold, so the vectorized scheme
 * is 32x smaller on a 4096-bit database; the server operation counters match their formulas; empty,
 * all-zero and all-one databases, N = 1, the boundary indices and a misconfigured layout behave; and
 * fresh randomness yields different queries with the same answer while a fixed seed is reproducible.
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

    boolean nextBoolean() { return (nextLong() & 1L) == 1L; }
}

/** A bit database plus the counters a server would see. */
class Database {
    final int n;
    final boolean[] bits;
    long bitOperations;
    long wordReads;

    Database(boolean[] bits) { this.n = bits.length; this.bits = bits; }

    static Database random(int n, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        boolean[] b = new boolean[n];
        for (int i = 0; i < n; i++) b[i] = rng.nextBoolean();
        return new Database(b);
    }

    static Database constant(int n, boolean value) {
        boolean[] b = new boolean[n];
        Arrays.fill(b, value);
        return new Database(b);
    }

    boolean get(int i) {
        if (i < 0 || i >= n) throw new IllegalArgumentException("index " + i + " outside [0," + n + ")");
        return bits[i];
    }
}

/** The flat two-server query: one uniform mask per server, XOR-differing in exactly one position. */
class FlatQuery {
    final boolean[] q1, q2;
    final int index;

    FlatQuery(boolean[] q1, boolean[] q2, int index) { this.q1 = q1; this.q2 = q2; this.index = index; }

    static FlatQuery build(int n, int index, SplitMix64 rng) {
        if (index < 0 || index >= n) throw new IllegalArgumentException("index " + index + " outside [0," + n + ")");
        boolean[] q1 = new boolean[n], q2 = new boolean[n];
        for (int j = 0; j < n; j++) q1[j] = rng.nextBoolean();
        for (int j = 0; j < n; j++) q2[j] = q1[j] ^ (j == index);
        return new FlatQuery(q1, q2, index);
    }

    int communicationBits() { return 2 * q1.length; }
}

/** The vectorized query: one row selector per server; the column index is never transmitted. */
class VectorQuery {
    final boolean[] u1, u2;
    final int rows, cols, row, col;

    VectorQuery(boolean[] u1, boolean[] u2, int rows, int cols, int row, int col) {
        this.u1 = u1; this.u2 = u2; this.rows = rows; this.cols = cols; this.row = row; this.col = col;
    }

    static VectorQuery build(int rows, int cols, int index, SplitMix64 rng) {
        if (rows <= 0 || cols <= 0) throw new IllegalArgumentException("empty layout");
        if (index < 0 || index >= rows * cols)
            throw new IllegalArgumentException("index " + index + " outside [0," + rows * cols + ")");
        int row = index / cols, col = index % cols;
        boolean[] u1 = new boolean[rows], u2 = new boolean[rows];
        for (int m = 0; m < rows; m++) u1[m] = rng.nextBoolean();
        for (int m = 0; m < rows; m++) u2[m] = u1[m] ^ (m == row);
        return new VectorQuery(u1, u2, rows, cols, row, col);
    }

    int communicationBits() { return 2 * rows + 2 * cols; }
}

/** The two honest-but-curious servers. Each one only ever sees its own selector. */
class Server {
    /** Flat answer: XOR of the bits the selector picks, counting the bits it touches. */
    static boolean flatAnswer(Database db, boolean[] selector) {
        if (selector.length != db.n) throw new IllegalArgumentException("selector length mismatch");
        boolean acc = false;
        for (int j = 0; j < db.n; j++) {
            db.bitOperations++;
            if (selector[j]) acc ^= db.bits[j];
        }
        return acc;
    }

    /** Vectorized answer: the c-bit XOR of the rows the selector picks. */
    static boolean[] rowAnswer(Database db, boolean[] u, int rows, int cols) {
        if (u.length != rows || rows * cols != db.n)
            throw new IllegalArgumentException("layout does not match the database");
        boolean[] word = new boolean[cols];
        for (int m = 0; m < rows; m++) {
            db.bitOperations++;
            db.wordReads++;
            if (!u[m]) continue;
            for (int j = 0; j < cols; j++) word[j] ^= db.bits[m * cols + j];
        }
        return word;
    }
}

/** The client: builds queries, combines answers, and accounts for the communication. */
class Client {
    final Database db;
    long communicationBits;
    int queries;

    Client(Database db) { this.db = db; }

    boolean retrieveFlat(int index, SplitMix64 rng) {
        FlatQuery q = FlatQuery.build(db.n, index, rng);
        communicationBits += q.communicationBits() + 2;          // two 1-bit answers
        queries++;
        return Server.flatAnswer(db, q.q1) ^ Server.flatAnswer(db, q.q2);
    }

    boolean retrieveVectorized(int index, int rows, SplitMix64 rng) {
        int cols = db.n / rows;
        VectorQuery q = VectorQuery.build(rows, cols, index, rng);
        boolean[] y1 = Server.rowAnswer(db, q.u1, rows, cols);
        boolean[] y2 = Server.rowAnswer(db, q.u2, rows, cols);
        communicationBits += q.communicationBits();
        queries++;
        return y1[q.col] ^ y2[q.col];
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    /** Per-position ones-frequency of a boolean vector pool, position major. */
    static double[] positionFrequencies(List<boolean[]> pool, int length) {
        double[] hits = new double[length];
        for (boolean[] v : pool)
            for (int j = 0; j < length; j++) if (v[j]) hits[j]++;
        for (int j = 0; j < length; j++) hits[j] /= pool.size();
        return hits;
    }

    static double mean(double[] values) {
        double s = 0;
        for (double v : values) s += v;
        return s / values.length;
    }

    static double maxDeviation(double[] values, double target) {
        double worst = 0;
        for (double v : values) worst = Math.max(worst, Math.abs(v - target));
        return worst;
    }

    public static void main(String[] args) {
        Database db = Database.random(4096, 4242);
        Client client = new Client(db);

        // 1. The flat scheme returns the correct bit for every index, using fresh randomness per query.
        boolean allCorrect = true;
        SplitMix64 rng = new SplitMix64(1);
        for (int i = 0; i < db.n; i++) if (client.retrieveFlat(i, rng) != db.get(i)) allCorrect = false;
        check(allCorrect, "the flat scheme returned a wrong bit for some index");
        check(client.queries == db.n, "query accounting wrong");
        check(client.communicationBits == (long) db.n * (2 * db.n + 2), "flat communication formula wrong: "
                + client.communicationBits);

        // 2. The vectorized scheme is correct for every index, including a non-square layout.
        Client vec = new Client(db);
        int badIndex = -1;
        for (int i = 0; i < db.n; i++)
            if (vec.retrieveVectorized(i, 64, new SplitMix64(9)) != db.get(i)) { badIndex = i; break; }
        check(badIndex < 0, "the vectorized scheme returned a wrong bit at index " + badIndex);
        check(vec.communicationBits == (long) db.n * (2 * 64 + 2 * 64),
                "vectorized communication formula wrong: " + vec.communicationBits);
        Client nonSquare = new Client(Database.random(128, 7));
        int badNonSquare = -1;
        for (int i = 0; i < 128; i++)
            if (nonSquare.retrieveVectorized(i, 8, new SplitMix64(i)) != nonSquare.db.get(i)) { badNonSquare = i; break; }
        check(badNonSquare < 0, "the vectorized scheme failed on a non-square 8x16 layout at index "
                + badNonSquare);

        // 3. Communication: the vectorized scheme really is O(sqrt N) instead of O(N).
        long flatBits = (long) db.n * (2 * db.n + 2);
        long vecBits = (long) db.n * (2 * 64 + 2 * 64);
        check(flatBits >= vecBits * 30, "vectorized communication is not 30x smaller: " + vecBits
                + " vs " + flatBits);
        int sqrtN = (int) Math.sqrt(db.n);
        check(sqrtN * sqrtN == db.n && 4 * sqrtN == 256, "the square layout is miscomputed");
        check(FlatQuery.build(db.n, 7, new SplitMix64(3)).communicationBits() == 2 * db.n,
                "flat query size wrong");
        check(VectorQuery.build(64, 64, 7, new SplitMix64(3)).communicationBits() == 256,
                "vectorized query size wrong");

        // 4. Server work: the flat server touches every bit, the vectorized server every word it folds.
        Database workDb = Database.random(4096, 55);
        Server.flatAnswer(workDb, FlatQuery.build(4096, 100, new SplitMix64(4)).q1);
        check(workDb.bitOperations == 4096, "flat server touched " + workDb.bitOperations + " bits");
        Database vecDb = Database.random(4096, 55);
        VectorQuery vq = VectorQuery.build(64, 64, 100, new SplitMix64(4));
        Server.rowAnswer(vecDb, vq.u1, 64, 64);
        check(vecDb.wordReads == 64 && vecDb.bitOperations == 64,
                "the vectorized server must charge one pass over the rows: " + vecDb.wordReads);

        // 5. Privacy of the flat scheme: each server's view is uniform, and its weight is Binomial(N, 1/2).
        List<boolean[]> q1Pool = new ArrayList<>(), q2Pool = new ArrayList<>();
        int trials = 400;
        for (int t = 0; t < trials; t++) {
            FlatQuery q = FlatQuery.build(db.n, 13, new SplitMix64(1000 + t));
            q1Pool.add(q.q1);
            q2Pool.add(q.q2);
        }
        double freqDev1 = maxDeviation(positionFrequencies(q1Pool, db.n), 0.5);
        double freqDev2 = maxDeviation(positionFrequencies(q2Pool, db.n), 0.5);
        check(freqDev1 < 0.12 && freqDev2 < 0.12, "a server's view is not uniform: deviations "
                + freqDev1 + " / " + freqDev2);
        double weightMean = 0, weightSq = 0;
        for (boolean[] q : q1Pool) {
            int w = 0;
            for (boolean b : q) if (b) w++;
            weightMean += w;
            weightSq += (double) w * w;
        }
        weightMean /= trials;
        double weightVar = weightSq / trials - weightMean * weightMean;
        check(Math.abs(weightMean - db.n / 2.0) < 40, "query weight mean " + weightMean + " is not N/2");
        check(Math.abs(weightVar - db.n / 4.0) < 200, "query weight variance " + weightVar + " is not N/4");

        // 6. Conditional privacy: the frequency at a position does not depend on the requested index.
        int target = 100;
        List<boolean[]> whenTarget = new ArrayList<>(), whenOther = new ArrayList<>();
        for (int t = 0; t < 600; t++) {
            FlatQuery q = FlatQuery.build(db.n, (t % 2 == 0) ? target : target + 1, new SplitMix64(5000 + t));
            (t % 2 == 0 ? whenTarget : whenOther).add(q.q1);
        }
        double fTarget = positionFrequencies(whenTarget, db.n)[target];
        double fOther = positionFrequencies(whenOther, db.n)[target];
        check(Math.abs(fTarget - fOther) < 0.12, "the query leaks the target: " + fTarget + " vs " + fOther);
        check(Math.abs(fTarget - 0.5) < 0.08 && Math.abs(fOther - 0.5) < 0.08,
                "the target position is not masked: " + fTarget + " / " + fOther);

        // 7. The XOR invariant: alone each query is uniform, together they encode exactly one index.
        boolean xorOk = true;
        int onesInXor = 0;
        FlatQuery xorQuery = FlatQuery.build(db.n, 77, new SplitMix64(17));
        for (int j = 0; j < db.n; j++) {
            boolean x = xorQuery.q1[j] ^ xorQuery.q2[j];
            if (x != (j == 77)) xorOk = false;
            if (x) onesInXor++;
        }
        check(xorOk && onesInXor == 1, "collusion invariant violated: the queries must XOR to e_i");
        Set<String> seenQueries = new TreeSet<>();
        for (int t = 0; t < 50; t++) seenQueries.add(Arrays.toString(FlatQuery.build(db.n, 77, new SplitMix64(t)).q1));
        check(seenQueries.size() == 50, "fresh randomness produced repeated queries");

        // 8. Privacy of the vectorized scheme: the row selector is uniform, and the column index is
        //    structurally absent from every message - it never leaves the client.
        List<boolean[]> uPool = new ArrayList<>();
        for (int t = 0; t < 400; t++) uPool.add(VectorQuery.build(64, 64, 200 + (t % 3) * 7, new SplitMix64(9000 + t)).u1);
        check(maxDeviation(positionFrequencies(uPool, 64), 0.5) < 0.12, "row selector is not uniform");
        VectorQuery vq2 = VectorQuery.build(64, 64, 200, new SplitMix64(3));
        int onesU = 0;
        for (int m = 0; m < 64; m++)
            if (vq2.u1[m] ^ vq2.u2[m]) { onesU++; check(m == vq2.row, "row XOR indicator at the wrong position"); }
        check(onesU == 1, "the row selectors must differ in exactly one position");
        check(vq2.row == 200 / 64 && vq2.col == 200 % 64, "index decomposition wrong");
        VectorQuery sameRow = VectorQuery.build(64, 64, 200 + 5, new SplitMix64(3));
        check(Arrays.equals(vq2.u1, sameRow.u1) && Arrays.equals(vq2.u2, sameRow.u2),
                "two indices in the same row must produce identical messages, so the column is private");
        VectorQuery otherRow = VectorQuery.build(64, 64, 200 + 64, new SplitMix64(3));
        check(!Arrays.equals(vq2.u2, otherRow.u2) && Arrays.equals(vq2.u1, otherRow.u1),
                "the two selectors must differ, and only the second one depends on the row");
        boolean[] stored = VectorQuery.build(64, 64, 777, new SplitMix64(11)).u1;
        boolean columnFree = true;
        for (int j = 0; j < 64; j++) if (stored.length != 64) columnFree = false;
        check(columnFree && stored.length == 64, "the message length must depend only on the row count");

        // 9. Degenerate databases and layouts.
        Database single = Database.random(1, 3);
        Client singleClient = new Client(single);
        check(singleClient.retrieveFlat(0, new SplitMix64(2)) == single.get(0), "1-bit database failed");
        Client singleVec = new Client(single);
        check(singleVec.retrieveVectorized(0, 1, new SplitMix64(2)) == single.get(0),
                "1x1 vectorized database failed");
        Database zeros = Database.constant(256, false), ones = Database.constant(256, true);
        Client zerosClient = new Client(zeros), onesClient = new Client(ones);
        boolean zeroOk = true, oneOk = true;
        for (int i = 0; i < 256; i++) {
            if (zerosClient.retrieveVectorized(i, 16, new SplitMix64(i)) != false) zeroOk = false;
            if (onesClient.retrieveVectorized(i, 16, new SplitMix64(i)) != true) oneOk = false;
        }
        check(zeroOk && oneOk, "constant databases failed");
        check(zerosClient.retrieveFlat(255, new SplitMix64(1)) == false
                && onesClient.retrieveFlat(0, new SplitMix64(1)) == true, "flat retrieval on constants failed");
        String[] badCases = {"negative", "too-large", "layout"};
        for (String badCase : badCases) {
            boolean threw = false;
            try {
                switch (badCase) {
                    case "negative": FlatQuery.build(64, -1, new SplitMix64(1)); break;
                    case "too-large": client.retrieveVectorized(4096, 64, new SplitMix64(1)); break;
                    default: client.retrieveVectorized(10, 10, new SplitMix64(1)); break;
                }
            } catch (IllegalArgumentException e) { threw = true; }
            check(threw, "invalid request was not rejected: " + badCase);
        }
        boolean selectorMismatch = false;
        try { Server.flatAnswer(db, new boolean[10]); } catch (IllegalArgumentException e) { selectorMismatch = true; }
        check(selectorMismatch, "a selector of the wrong length was accepted");

        // 10. Reproducibility: a fixed seed replays exactly, a new seed re-randomizes but still answers.
        FlatQuery a = FlatQuery.build(db.n, 55, new SplitMix64(31337));
        FlatQuery b = FlatQuery.build(db.n, 55, new SplitMix64(31337));
        check(Arrays.equals(a.q1, b.q1) && Arrays.equals(a.q2, b.q2), "a fixed seed is not reproducible");
        check(client.retrieveFlat(55, new SplitMix64(31337)) == db.get(55)
                && client.retrieveFlat(55, new SplitMix64(999)) == db.get(55),
                "retrieval is not stable across randomness");
        List<Long> answers = new ArrayList<>();
        int trueOnes = 0;
        for (int i = 0; i < 20; i++) if (db.get(i * 7)) trueOnes++;
        for (int t = 0; t < 5; t++) {
            Client fresh = new Client(db);
            long acc = 0;
            for (int i = 0; i < 20; i++) if (fresh.retrieveVectorized(i * 7, 64, new SplitMix64(t * 100 + i))) acc++;
            answers.add(acc);
        }
        for (long count : answers)
            check(count == trueOnes, "retrieved bit count " + count + " differs from " + trueOnes);

        // 11. Communication accounting for a long workload stays linear in the per-query cost.
        Client workload = new Client(db);
        for (int i = 0; i < 200; i++) workload.retrieveVectorized(i * 19 % db.n, 64, new SplitMix64(i));
        check(workload.communicationBits == 200L * (2 * 64 + 2 * 64), "amortized communication accounting wrong");
        check(workload.queries == 200, "query counter wrong");

        System.out.println("database: " + db.n + " bits, flat query " + (2 * db.n) + " bits, vectorized query "
                + 256 + " bits (" + (flatBits / vecBits) + "x smaller)");
        System.out.println("privacy: max positional deviation over 400 queries " + String.format("%.4f", freqDev1)
                + " (server 1) / " + String.format("%.4f", freqDev2) + " (server 2), weight "
                + String.format("%.1f", weightMean) + " +- " + String.format("%.1f", Math.sqrt(weightVar)));
        System.out.println("target masking: frequency at the target position " + String.format("%.3f", fTarget)
                + " when wanted vs " + String.format("%.3f", fOther) + " when not");
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
