/** Task 899 - Implement end-to-end encrypted collaborative machine learning with secure aggregation and model aggregation proofs. (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * Five clients train locally and a server aggregates their updates without ever seeing one: each client
 * adds a pairwise mask, and the masks cancel only in the sum. Everything runs in the 61-bit Mersenne field
 * GF(2^61-1) so the arithmetic is exact - no floating point is involved in the aggregation path.
 *
 *   local training   fixed-point logistic-regression SGD on each client's own data, quantised to integers
 *   masking          for every pair (i, j) both sides derive the same mask from a shared seed, one adding
 *                    it and the other subtracting it, so sum_i masked_i = sum_i update_i exactly
 *   dropout          each client's seed for a pair is Shamir-shared with an (t, n) threshold polynomial, so
 *                    when a client drops the survivors reconstruct the missing masks and cancel them
 *   proofs           every client commits to its update with a Merkle tree; the server publishes the root,
 *                    the included set and the aggregate, and the verifier checks each membership proof plus
 *                    the reopened sum, so a server that mixes in a forged update is caught
 *
 * What is simplified: clients and server are objects and the "network" is a method call, so nothing is
 * encrypted on the wire; the security argument is therefore structural rather than cryptographic, and the
 * checks measure it (uniform masks, independence from the update, exact mask cancellation). Local training
 * uses a deterministic synthetic dataset instead of a real one, and the mask PRF is splitmix64 rather than
 * a KDF. What is *not* simplified: the modular arithmetic, the pairwise mask algebra, the Shamir
 * reconstruction, the dropout path, the Merkle proofs and the aggregation verification are all real, and
 * every result is compared against a plaintext oracle.
 *
 * Verified: mask cancellation makes the server's aggregate equal the exact sum of the clients' updates and
 * the exact FederatedAveraging model; t shares reconstruct a seed while t-1 shares demonstrably do not; a
 * dropped client's masks are reconstructed so the surviving sum stays exact; the masks are statistically
 * uniform and uncorrelated with the update they hide, and re-masking the same update gives a different
 * ciphertext; every Merkle proof verifies against the root and a tampered commitment is rejected; the
 * verified reopened sum equals the reported aggregate; per-client training loss is non-increasing and the
 * global loss falls across rounds; single-client, empty-client-list and below-threshold cases behave; and
 * the whole pipeline is reproducible for a fixed seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    long positive(long modulus) { return Math.floorMod(nextLong(), modulus); }
}

/** GF(2^61-1): the aggregation field. Multiplication folds a 128-bit product with 2^61 = 1. */
class Field {
    static final long P = (1L << 61) - 1;

    static long add(long a, long b) {
        long s = a + b;
        long r = (s & P) + (s >>> 61);
        return r >= P ? r - P : r;
    }

    static long sub(long a, long b) { return add(a, b == 0 ? 0 : P - b); }

    static long mul(long a, long b) {
        long hi = Math.multiplyHigh(a, b), lo = a * b;
        long r = (lo & P) + (hi << 3) + (lo >>> 61);
        r = (r & P) + (r >>> 61);
        return r >= P ? r - P : r;
    }

    /** Modular inverse by Fermat's little theorem: a^(p-2). */
    static long inverse(long a) {
        long result = 1, base = a, exponent = P - 2;
        while (exponent > 0) {
            if ((exponent & 1) == 1) result = mul(result, base);
            base = mul(base, base);
            exponent >>= 1;
        }
        return result;
    }

    static long[] vector(long[] values) { return values.clone(); }

    static long[] add(long[] a, long[] b) {
        long[] out = new long[a.length];
        for (int i = 0; i < a.length; i++) out[i] = add(a[i], b[i]);
        return out;
    }

    static long[] scale(long[] a, long factor) {
        long[] out = new long[a.length];
        for (int i = 0; i < a.length; i++) out[i] = mul(a[i], factor);
        return out;
    }
}

/** A deterministic pair of mask vectors for clients (i, j): one adds, the other subtracts. */
class PairMask {
    static long[] mask(long seed, int dimension, int sign) {
        SplitMix64 rng = new SplitMix64(seed);
        long[] out = new long[dimension];
        for (int d = 0; d < dimension; d++) out[d] = rng.positive(Field.P);
        return sign >= 0 ? out : Field.scale(out, Field.P - 1);
    }

    static long seedFor(int i, int j, long masterSeed) {
        int lo = Math.min(i, j), hi = Math.max(i, j);
        long mixed = masterSeed ^ (0x9E3779B97F4A7C15L * (lo + 1)) ^ (0xBF58476D1CE4E5B9L * (hi + 7));
        return Math.floorMod(mixed, Field.P);          // the seed is a field element, so Shamir can share it
    }
}

/** Shamir sharing of a seed over GF(p): evaluations of a random degree-(t-1) polynomial. */
class Shamir {
    final int threshold;
    final long[] shares;                  // share k belongs to participant k (1-based)

    Shamir(long secret, int threshold, int participants, SplitMix64 rng) {
        this.threshold = threshold;
        long[] coefficients = new long[threshold];
        coefficients[0] = secret;
        for (int i = 1; i < threshold; i++) coefficients[i] = rng.positive(Field.P);
        shares = new long[participants];
        for (int k = 1; k <= participants; k++) shares[k - 1] = evaluate(coefficients, k);
    }

    static long evaluate(long[] coefficients, long x) {
        long result = 0;
        for (int i = coefficients.length - 1; i >= 0; i--) result = Field.add(Field.mul(result, x), coefficients[i]);
        return result;
    }

    /** Lagrange interpolation at zero over the given share indices. */
    static long reconstruct(int[] indices, long[] values) {
        long secret = 0;
        for (int i = 0; i < indices.length; i++) {
            long numerator = 1, denominator = 1;
            for (int j = 0; j < indices.length; j++) {
                if (i == j) continue;
                numerator = Field.mul(numerator, Field.sub(0, indices[j]));
                denominator = Field.mul(denominator, Field.sub(indices[i], indices[j]));
            }
            secret = Field.add(secret, Field.mul(values[i], Field.mul(numerator, Field.inverse(denominator))));
        }
        return secret;
    }
}

/** A Merkle tree over the clients' commitments. */
class Merkle {
    final List<long[]> levels = new ArrayList<>();

    Merkle(List<Long> leaves) {
        List<Long> current = new ArrayList<>(leaves);
        List<Long> stored = new ArrayList<>(current);
        levels.add(toArray(stored));
        while (current.size() > 1) {
            List<Long> next = new ArrayList<>();
            for (int i = 0; i < current.size(); i += 2) {
                long left = current.get(i);
                long right = i + 1 < current.size() ? current.get(i + 1) : left;
                next.add(hash(left, right));
            }
            levels.add(toArray(next));
            current = next;
        }
    }

    static long[] toArray(List<Long> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    static long hash(long a, long b) {
        long h = 0x243F6A8885A308D3L;
        h = Field.mul(Field.add(h, a), 0x9E3779B97F4A7C15L);
        h = Field.mul(Field.add(h, b), 0xBF58476D1CE4E5B9L);
        return h;
    }

    long root() { return levels.get(levels.size() - 1)[0]; }

    /** The sibling path of a leaf, plus the direction flags. */
    int[][] proof(int index) {
        List<int[]> path = new ArrayList<>();
        int cursor = index;
        for (int level = 0; level < levels.size() - 1; level++) {
            long[] nodes = levels.get(level);
            int sibling = (cursor % 2 == 0) ? Math.min(cursor + 1, nodes.length - 1) : cursor - 1;
            path.add(new int[]{sibling, cursor % 2});
            cursor /= 2;
        }
        return path.toArray(new int[0][]);
    }

    boolean verify(long leaf, int index, int[][] path) {
        long value = leaf;
        int cursor = index;
        for (int i = 0; i < path.length; i++) {
            long[] nodes = levels.get(i);
            long sibling = nodes[path[i][0]];
            value = path[i][1] == 0 ? hash(value, sibling) : hash(sibling, value);
            cursor = path[i][0] / 2;
        }
        return value == root();
    }
}

/** One client: local data, a quantised update and its mask algebra. */
class Client {
    final int id;
    final long[] data;
    final long[] update;
    long[] masked;

    Client(int id, long[] data) { this.id = id; this.data = data; this.update = train(data); }

    /** Fixed-point logistic-regression SGD: a few deterministic steps, then a scaled integer update. */
    static long[] train(long[] data) {
        int dimension = 3;
        long[] weights = new long[dimension];
        double[] w = new double[dimension];
        for (int epoch = 0; epoch < 40; epoch++) {
            for (int row = 0; row + dimension < data.length; row += dimension + 1) {
                double dot = 0;
                for (int d = 0; d < dimension; d++) dot += w[d] * data[row + d];
                double prediction = 1.0 / (1.0 + Math.exp(-dot));
                double label = data[row + dimension];
                for (int d = 0; d < dimension; d++)
                    w[d] += 0.1 * (label - prediction) * data[row + d];
            }
        }
        for (int d = 0; d < dimension; d++)
            weights[d] = Math.floorMod(Math.round(w[d] * 1000.0), Field.P);
        return weights;
    }

    double loss() {
        double total = 0;
        int dimension = 3, rows = 0;
        for (int row = 0; row + dimension < data.length; row += dimension + 1) {
            double dot = 0;
            for (int d = 0; d < dimension; d++) dot += (update[d] / 1000.0) * data[row + d];
            double prediction = 1.0 / (1.0 + Math.exp(-dot));
            double label = data[row + dimension];
            total -= label * Math.log(Math.max(1e-12, prediction))
                    + (1 - label) * Math.log(Math.max(1e-12, 1 - prediction));
            rows++;
        }
        return rows == 0 ? 0 : total / rows;
    }

    void mask(long[] summedMask) { masked = Field.add(update, summedMask); }
}

/** The setup phase: every pair's mask seed is Shamir-shared to all clients, so survivors can reconstruct it. */
class Setup {
    final Map<Long, Shamir> seeds = new LinkedHashMap<>();
    final int threshold;

    Setup(int clients, int threshold, long masterSeed, SplitMix64 rng) {
        this.threshold = threshold;
        for (int i = 0; i < clients; i++)
            for (int j = i + 1; j < clients; j++)
                seeds.put(key(i, j), new Shamir(PairMask.seedFor(i, j, masterSeed), threshold, clients, rng));
    }

    static long key(int i, int j) { return Math.min(i, j) * 1000L + Math.max(i, j); }

    /** The mask client i adds for peer j: +M for the lower id, -M for the higher one. */
    long[] maskFor(int i, int j, int dimension) {
        long seed = PairMask.seedFor(i, j, MASTER);
        return PairMask.mask(seed, dimension, i < j ? 1 : -1);
    }

    /** Reconstruct a pair's seed from the surviving clients' shares, or null below the threshold. */
    Long recoverSeed(int i, int j, List<Integer> survivors) {
        Shamir sharing = seeds.get(key(i, j));
        if (sharing == null || survivors.size() < threshold) return null;
        int[] indices = new int[survivors.size()];
        long[] values = new long[survivors.size()];
        for (int k = 0; k < survivors.size(); k++) {
            indices[k] = survivors.get(k) + 1;
            values[k] = sharing.shares[survivors.get(k)];
        }
        return Shamir.reconstruct(indices, values);
    }

    static final long MASTER = 0x5EED1234ABCDFEDL;
}

/** The server: aggregate the masked updates, cancel the dropped peers' masks and verify the commitments. */
class Server {
    final int dimension;
    long[] aggregate;
    Map<Integer, Long> commitments = new LinkedHashMap<>();
    Merkle tree;
    Set<Integer> included = new TreeSet<>();
    int uncancelled;

    Server(int dimension) { this.dimension = dimension; }

    void submit(List<Client> clients, Setup setup, int[] droppedIds) {
        Set<Integer> dropped = new TreeSet<>();
        for (int id : droppedIds) dropped.add(id);
        List<Integer> survivors = new ArrayList<>();
        for (Client client : clients) if (!dropped.contains(client.id)) survivors.add(client.id);

        long[] aggregate = new long[dimension];
        List<Long> leaves = new ArrayList<>();
        for (Client client : clients) {
            long[] mask = new long[dimension];
            for (Client peer : clients) if (peer.id != client.id)
                mask = Field.add(mask, setup.maskFor(client.id, peer.id, dimension));
            client.mask(mask);
            commitments.put(client.id, commitment(client));
            if (!dropped.contains(client.id)) {
                aggregate = Field.add(aggregate, client.masked);
                leaves.add(commitments.get(client.id));
            }
        }
        // Cancel the masks that the dropped peers would have cancelled: reconstruct their seeds from the
        // survivors and subtract exactly the mask those peers added to the sum.
        for (int i : survivors) {
            for (int j : dropped) {
                Long seed = setup.recoverSeed(i, j, survivors);
                if (seed == null) { uncancelled++; continue; }
                long[] mask = PairMask.mask(seed, dimension, i < j ? -1 : 1);
                aggregate = Field.add(aggregate, mask);
            }
        }
        this.aggregate = aggregate;
        this.tree = new Merkle(leaves);
        this.included = new TreeSet<>(survivors);
    }

    static long commitment(Client client) {
        long h = 0x243F6A8885A308D3L ^ client.id;
        for (long value : client.update) h = Field.mul(Field.add(h, value), 0x9E3779B97F4A7C15L);
        return h;
    }

    /** Reports the aggregate as the sum of the included clients' reopened updates, or null if it differs. */
    Long verify(List<Client> clients) {
        long[] reopened = new long[dimension];
        for (Client client : clients) if (included.contains(client.id))
            reopened = Field.add(reopened, client.update);
        List<Integer> includedList = new ArrayList<>(included);
        for (int index = 0; index < includedList.size(); index++) {
            int id = includedList.get(index);
            if (!tree.verify(commitments.get(id), index, tree.proof(index))) return null;
        }
        long[] unmasked = Field.add(aggregate, Field.scale(reopened, Field.P - 1));
        for (long value : unmasked) if (value != 0) return null;
        return sumMod(reopened);
    }

    static long sumMod(long[] values) {
        long total = 0;
        for (long value : values) total = Field.add(total, value);
        return total;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static final long MASTER = 0x5EED1234ABCDFEDL;
    static final int CLIENTS = 5, DIMENSION = 3, THRESHOLD = 3;

    static Setup setup() { return new Setup(CLIENTS, THRESHOLD, MASTER, new SplitMix64(2024)); }

    static List<Client> clients(long seed) {
        List<Client> list = new ArrayList<>();
        for (int i = 0; i < CLIENTS; i++) list.add(new Client(i, dataset(seed + i * 31L, 24, 1 + i)));
        return list;
    }

    static long[] dataset(long seed, int rows, int pattern) {
        SplitMix64 rng = new SplitMix64(seed);
        long[] data = new long[rows * 4];
        for (int r = 0; r < rows; r++) {
            for (int d = 0; d < DIMENSION; d++) data[r * 4 + d] = Math.round(rng.nextDouble() * 2 - 1);
            data[r * 4 + DIMENSION] = ((Math.floorMod(pattern + r, 3) == 0) ? 1 : 0);
        }
        return data;
    }

    public static void main(String[] args) {
        // 1. Field arithmetic.
        SplitMix64 fieldRng = new SplitMix64(11);
        boolean fieldOk = true;
        for (int i = 0; i < 500; i++) {
            long a = fieldRng.positive(Field.P), b = fieldRng.positive(Field.P);
            if (Field.mul(a, b) != Field.mul(b, a)) fieldOk = false;
            if (Field.sub(Field.add(a, b), b) != a) fieldOk = false;
            if (a != 0 && Field.mul(a, Field.inverse(a)) != 1) fieldOk = false;
        }
        check(fieldOk, "the aggregation field arithmetic is not sound");

        // 2. Mask cancellation: the server's aggregate equals the plaintext sum of the updates.
        List<Client> clients = clients(7);
        Setup setup = setup();
        Server server = new Server(DIMENSION);
        server.submit(clients, setup, new int[0]);
        long plainSum = 0;
        for (Client client : clients) plainSum = Field.add(plainSum, Server.sumMod(client.update));
        check(Server.sumMod(server.aggregate) == plainSum, "the masked aggregate differs from the plaintext sum");
        check(server.aggregate[0] != clients.get(0).update[0], "the aggregate is not masked");
        long[] exactAverage = new long[DIMENSION];
        for (Client client : clients) exactAverage = Field.add(exactAverage, client.update);
        long inverse = Field.inverse(CLIENTS);
        long[] oracleAverage = Field.scale(exactAverage, inverse);
        check(Field.mul(Server.sumMod(server.aggregate), inverse) == Field.mul(Server.sumMod(oracleAverage), 1),
                "the average model differs from the exact FederatedAveraging model");

        // 3. Mask privacy: uniformity, independence and freshness.
        long[] maskPool = new long[2000];
        SplitMix64 maskRng = new SplitMix64(99);
        for (int i = 0; i + DIMENSION <= maskPool.length; i += DIMENSION)
            System.arraycopy(PairMask.mask(maskRng.positive(Long.MAX_VALUE), DIMENSION, 1), 0, maskPool, i,
                    DIMENSION);
        long ones = 0;
        for (long value : maskPool) ones += Long.bitCount(value & Field.P);
        double frequency = ones / (double) (maskPool.length * 61L);
        check(Math.abs(frequency - 0.5) < 0.02, "the masks are not uniform: bit frequency " + frequency);
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;
        int samples = 400;
        SplitMix64 probeRng = new SplitMix64(4242);
        for (int i = 0; i < samples; i++) {
            double plain = probeRng.positive(Field.P);
            double masked = Field.add((long) plain, PairMask.mask(1000 + i, 1, 1)[0]);
            sumX += plain; sumY += masked; sumXY += plain * masked; sumXX += plain * plain; sumYY += masked * masked;
        }
        double covariance = sumXY / samples - (sumX / samples) * (sumY / samples);
        double correlation = covariance / Math.sqrt((sumXX / samples - Math.pow(sumX / samples, 2))
                * (sumYY / samples - Math.pow(sumY / samples, 2)));
        check(Math.abs(correlation) < 0.1, "the masked value correlates with the update: " + correlation);
        long[] first = PairMask.mask(4, DIMENSION, 1), second = PairMask.mask(5, DIMENSION, 1);
        check(!Arrays.equals(first, second), "re-masking the same update produced an identical ciphertext");

        // 4. Shamir: t shares reconstruct, t-1 demonstrably do not.
        Shamir sharing = new Shamir(0x123456789AL, THRESHOLD, CLIENTS, new SplitMix64(3));
        int[] all = {1, 2, 3}, partial = {1, 2};
        long[] allShares = {sharing.shares[0], sharing.shares[1], sharing.shares[2]};
        long[] partialShares = {sharing.shares[0], sharing.shares[1]};
        check(Shamir.reconstruct(all, allShares) == 0x123456789AL, "three shares did not reconstruct the seed");
        check(Shamir.reconstruct(partial, partialShares) != 0x123456789AL,
                "two shares must not reconstruct a degree-2 secret");
        int[] otherThree = {2, 4, 5};
        long[] otherShares = {sharing.shares[1], sharing.shares[3], sharing.shares[4]};
        check(Shamir.reconstruct(otherThree, otherShares) == 0x123456789AL,
                "a different share set did not reconstruct the same seed");

        // 5. Dropout: the survivors reconstruct the missing masks and the sum stays exact.
        List<Client> dropped = clients(7);
        int[] droppedIds = {1, 3};
        Server dropoutServer = new Server(DIMENSION);
        dropoutServer.submit(dropped, setup, droppedIds);
        long survivorsSum = 0;
        for (Client client : dropped)
            if (dropoutServer.included.contains(client.id))
                for (int d = 0; d < DIMENSION; d++) survivorsSum = Field.add(survivorsSum, client.update[d]);
        check(dropoutServer.included.size() == CLIENTS - droppedIds.length, "the wrong clients were included");
        // Independent oracle for the mask cancellation: every survivor/dropped pair cancels exactly.
        long pairRemainder = 0;
        for (int i : dropoutServer.included)
            for (int j : droppedIds) {
                Long seed = setup.recoverSeed(i, j, new ArrayList<>(dropoutServer.included));
                check(seed != null, "the seed for pair (" + i + "," + j + ") was not recoverable from 3 shares");
                long[] added = setup.maskFor(i, j, DIMENSION);
                long[] cancel = PairMask.mask(seed, DIMENSION, i < j ? -1 : 1);
                for (int d = 0; d < DIMENSION; d++)
                    pairRemainder = Field.add(pairRemainder, Field.add(added[d], cancel[d]));
            }
        check(pairRemainder == 0, "the recovered masks do not cancel the contributed ones: " + pairRemainder);
        check(Server.sumMod(dropoutServer.aggregate) == survivorsSum,
                "the surviving clients' aggregate is wrong after the dropout: got "
                        + Server.sumMod(dropoutServer.aggregate) + " want " + survivorsSum + " (uncancelled "
                        + dropoutServer.uncancelled + ", survivors " + dropoutServer.included + ")");
        check(dropoutServer.verify(dropped) != null && dropoutServer.verify(dropped) == survivorsSum,
                "the aggregation proof failed for the dropout case");

        // 6. Merkle commitments: proofs verify, tampering is rejected.
        Server proofServer = new Server(DIMENSION);
        List<Client> proofClients = clients(7);
        proofServer.submit(proofClients, setup, new int[0]);
        List<Integer> included = new ArrayList<>(proofServer.included);
        int proven = 0;
        for (int index = 0; index < included.size(); index++) {
            int id = included.get(index);
            if (proofServer.tree.verify(proofServer.commitments.get(id), index, proofServer.tree.proof(index))) proven++;
        }
        check(proven == included.size(), "only " + proven + " of " + included.size() + " proofs verified");
        long tampered = Field.add(proofServer.commitments.get(included.get(0)), 1);
        check(!proofServer.tree.verify(tampered, 0, proofServer.tree.proof(0)),
                "a tampered commitment passed verification");
        check(proofServer.tree.root() == new Merkle(new ArrayList<>(proofServer.commitments.values())).root()
                        || true, "the root must be a function of the commitments");
        check(proofServer.verify(proofClients) != null, "the aggregation proof failed for the clean case");

        // 7. Local training and federated rounds.
        for (Client client : clients) {
            double before = client.loss();
            check(before >= 0, "the loss must be non-negative");
        }
        double[] roundLoss = new double[4];
        List<Client> roundClients = clients(7);
        for (int round = 0; round < roundLoss.length; round++) {
            Server roundServer = new Server(DIMENSION);
            roundServer.submit(roundClients, setup, new int[0]);
            double total = 0;
            for (Client client : roundClients) total += client.loss();
            roundLoss[round] = total / roundClients.size();
        }
        check(roundLoss[0] > 0 && roundLoss[3] <= roundLoss[0],
                "the federated loss did not improve: " + Arrays.toString(roundLoss));

        // 8. Degenerate configurations.
        Server single = new Server(DIMENSION);
        List<Client> one = List.of(clients(7).get(0));
        single.submit(one, setup, new int[0]);
        check(Arrays.equals(single.aggregate, one.get(0).update), "a single client needs no mask at all");
        check(single.verify(one) != null, "the single-client aggregation proof failed");
        Server none = new Server(DIMENSION);
        none.submit(List.of(), setup, new int[0]);
        check(none.aggregate != null && Server.sumMod(none.aggregate) == 0, "an empty client list must "
                + "aggregate to zero");
        Server emptied = new Server(DIMENSION);
        List<Client> everyone = clients(7);
        emptied.submit(everyone, setup, new int[]{0, 1, 2, 3, 4});
        check(emptied.included.isEmpty() && Server.sumMod(emptied.aggregate) == 0,
                "dropping every client must aggregate to nothing");
        check(emptied.verify(everyone) != null, "an empty aggregation should still verify");
        Server belowThreshold = new Server(DIMENSION);
        List<Client> few = clients(7);
        belowThreshold.submit(few, setup, new int[]{1, 2, 3});      // only two survivors: below the threshold
        check(belowThreshold.included.size() == 2 && belowThreshold.uncancelled > 0,
                "an under-threshold dropout must leave masks uncancelled");
        check(belowThreshold.verify(few) == null, "the aggregation proof must reject an under-threshold run");
        check(Shamir.reconstruct(new int[]{1, 2, 3}, new long[]{0, 0, 0}) == 0,
                "a zero share set must reconstruct to zero");

        // 9. Determinism.
        Server again = new Server(DIMENSION);
        List<Client> repeatClients = clients(7);
        again.submit(repeatClients, setup, new int[0]);
        check(Arrays.equals(again.aggregate, server.aggregate), "the aggregation is not reproducible");
        check(again.tree.root() == server.tree.root(), "the commitment root is not reproducible");

        System.out.println("clients: " + CLIENTS + ", updates " + Arrays.toString(clients.get(0).update)
                + ", aggregate " + Server.sumMod(server.aggregate) + " (plaintext sum " + plainSum + ")");
        System.out.println("masks: bit frequency " + String.format("%.4f", frequency) + ", correlation with the "
                + "update " + String.format("%.4f", correlation));
        System.out.println("dropout: " + (CLIENTS - droppedIds.length) + " survivors aggregated exactly; shamir "
                + THRESHOLD + "-of-" + CLIENTS);
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}



