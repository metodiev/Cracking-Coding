/** Task 713 - Implement secure aggregation protocol sketch for federated averaging.
 * (Tier: Expert, CTCI Beyond -- Security & Privacy)
 *
 * Secure aggregation (Bonawitz et al.) lets a server learn the *sum* of the clients' updates while
 * learning nothing about any single update. The mechanism is pairwise additive masking over a finite
 * field: every pair of clients (u, v) shares a seed s_{uv} and derives the same pseudorandom vector
 * from it, so client v adds the vector for every u < v and subtracts it for every u > v. In the total
 * sum every pairwise mask appears twice with opposite signs and cancels exactly, which is why the sum
 * is all the server ever sees.
 *
 * Dropouts are what makes the protocol interesting: if a client disappears before uploading, its masks
 * no longer cancel and the server would get a garbage sum. The standard fix, implemented here, is that
 * every surviving client v reveals the pairwise seed s_{vu} it shares with each dropped client u (only
 * that one seed, not its own update); the server then re-derives +/- PRG(s_{vu}) and removes exactly
 * the residual the dropped client left behind. Revealing those seeds is harmless because the dropped
 * client's contribution is gone anyway.
 *
 * Simplified: the seeds are produced by a deterministic splitmix64 stream (a real deployment would use
 * a key-agreement plus a PRG with its own security analysis), the field is the Mersenne prime 2^31 - 1
 * so that products fit in a long, and no authentication is modelled - it is an honest-but-curious
 * sketch. Signed updates must be shifted into [0, modulus) before masking; the quantise helper refuses
 * negative entries instead of silently wrapping them, and the precondition that the true sum stays
 * below the modulus is checked explicitly.
 *
 * Verified: the recovered sum equals the brute-force sum for every dropout pattern (nobody, one client,
 * half, everybody), the naive sum without the dropout correction is demonstrably wrong, masked values
 * differ from the plaintext and look uniform over the field, the no-wrap precondition is checked, the
 * quantise/secure-sum/dequantise round trip reproduces a federated average within the quantisation
 * error, malformed input is rejected and the protocol is deterministic.
 */
import java.util.*;

class SecureAggregator {
    static final long MODULUS = 2147483647L;           // 2^31 - 1, prime

    private final int clients;
    private final int dimension;
    private final long seedBase;
    private long seedLookups;

    SecureAggregator(int clients, int dimension, long seedBase) {
        if (clients < 2) throw new IllegalArgumentException("need at least two clients");
        if (dimension < 1) throw new IllegalArgumentException("dimension must be positive");
        this.clients = clients;
        this.dimension = dimension;
        this.seedBase = seedBase;
    }

    int clients() {
        return clients;
    }

    int dimension() {
        return dimension;
    }

    long seedLookups() {
        return seedLookups;
    }

    static long add(long a, long b) {
        return (a + b) % MODULUS;
    }

    static long subtract(long a, long b) {
        return ((a - b) % MODULUS + MODULUS) % MODULUS;
    }

    /** The pairwise seed shared by two clients; both derive the same value. */
    private long pairSeed(int a, int b) {
        int low = Math.min(a, b);
        int high = Math.max(a, b);
        long z = seedBase + 0x9E3779B97F4A7C15L * (low * 1048576L + high + 1);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** PRG(s_ab, index): a uniform field element derived from the pair seed. */
    long maskValue(int a, int b, int index) {
        seedLookups++;
        long z = pairSeed(a, b) + 0x9E3779B97F4A7C15L * (index + 1);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return Math.floorMod(z, MODULUS);
    }

    /** Client `client` masks its quantised update: +PRG for lower-indexed peers, -PRG for higher. */
    long[] mask(int client, long[] update) {
        if (client < 0 || client >= clients) throw new IllegalArgumentException("unknown client");
        if (update.length != dimension) throw new IllegalArgumentException("dimension mismatch");
        long[] masked = new long[dimension];
        for (int k = 0; k < dimension; k++) {
            if (update[k] < 0 || update[k] >= MODULUS) {
                throw new IllegalArgumentException("update entry out of the field");
            }
            masked[k] = update[k];
        }
        for (int other = 0; other < clients; other++) {
            if (other == client) continue;
            boolean lower = other < client;
            for (int k = 0; k < dimension; k++) {
                long value = maskValue(client, other, k);
                masked[k] = lower ? add(masked[k], value) : subtract(masked[k], value);
            }
        }
        return masked;
    }

    /**
     * Server side: sums the surviving masked vectors and removes, for every dropped client, the
     * pairwise masks that no longer cancel. The seed material the survivors reveal for a dropped client
     * is modelled directly by the pair seed, which is exactly what a real survivor sends.
     */
    long[] recover(List<long[]> maskedContributions, boolean[] alive) {
        if (maskedContributions.size() != clients) throw new IllegalArgumentException("need one vector per client");
        if (alive.length != clients) throw new IllegalArgumentException("alive flags mismatch");
        for (long[] contribution : maskedContributions) {
            if (contribution.length != dimension) throw new IllegalArgumentException("dimension mismatch");
        }
        long[] sum = new long[dimension];
        for (int client = 0; client < clients; client++) {
            if (!alive[client]) continue;
            for (int k = 0; k < dimension; k++) sum[k] = add(sum[k], maskedContributions.get(client)[k]);
        }
        for (int dropped = 0; dropped < clients; dropped++) {
            if (alive[dropped]) continue;
            for (int survivor = 0; survivor < clients; survivor++) {
                if (!alive[survivor]) continue;
                for (int k = 0; k < dimension; k++) {
                    long residue = maskValue(dropped, survivor, k);
                    // in the survivor's masked vector the term was +residue when dropped < survivor
                    sum[k] = dropped < survivor ? subtract(sum[k], residue) : add(sum[k], residue);
                }
            }
        }
        return sum;
    }

    /** True sum over the surviving clients, computed the slow way: the oracle for the checks. */
    static long[] bruteForceSum(List<long[]> updates, boolean[] alive) {
        int dimension = updates.get(0).length;
        long[] sum = new long[dimension];
        for (int client = 0; client < updates.size(); client++) {
            if (!alive[client]) continue;
            for (int k = 0; k < dimension; k++) sum[k] = add(sum[k], updates.get(client)[k]);
        }
        return sum;
    }

    static long[] quantize(double[] update, int scale) {
        if (scale < 1) throw new IllegalArgumentException("scale must be positive");
        long[] quantized = new long[update.length];
        for (int k = 0; k < update.length; k++) {
            long value = Math.round(update[k] * scale);
            if (value < 0 || value >= MODULUS) throw new IllegalArgumentException("value out of field range");
            quantized[k] = value;
        }
        return quantized;
    }

    static double[] dequantize(long[] quantized, int scale) {
        double[] update = new double[quantized.length];
        for (int k = 0; k < quantized.length; k++) update[k] = quantized[k] / (double) scale;
        return update;
    }
}

public class Solution {
    private static long state = 0xA66A6A6A6AL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    public static void main(String[] args) {
        int clients = 8;
        int dimension = 5;
        SecureAggregator aggregator = new SecureAggregator(clients, dimension, 0x51DE55EEDL);
        List<long[]> updates = new ArrayList<>();
        List<long[]> masked = new ArrayList<>();
        for (int client = 0; client < clients; client++) {
            long[] update = new long[dimension];
            for (int k = 0; k < dimension; k++) update[k] = 100 + client * 37 + k * 11 + nextInt(50);
            updates.add(update);
            masked.add(aggregator.mask(client, update));
        }

        // --- every client present: masks cancel exactly --------------------------
        boolean[] allAlive = new boolean[clients];
        Arrays.fill(allAlive, true);
        long[] recovered = aggregator.recover(masked, allAlive);
        long[] expected = SecureAggregator.bruteForceSum(updates, allAlive);
        check(Arrays.equals(recovered, expected), "the full sum must be recovered exactly");
        long total = 0;
        for (long value : expected) total += value;
        check(total < SecureAggregator.MODULUS, "the quantised total must stay below the modulus: " + total);

        // --- masking really hides the update ------------------------------------
        int identical = 0;
        int maskPositive = 0;
        for (int client = 0; client < clients; client++) {
            long[] update = updates.get(client);
            long[] hidden = masked.get(client);
            boolean same = true;
            for (int k = 0; k < dimension; k++) {
                if (update[k] != hidden[k]) same = false;
                if (hidden[k] > 0) maskPositive++;
            }
            if (same) identical++;
        }
        check(identical == 0, "no masked vector may equal its plaintext");
        check(maskPositive > 0, "the masked values really do change the vector");
        Set<Long> distinctMasks = new HashSet<>();
        for (int client = 0; client < clients; client++) {
            distinctMasks.add(masked.get(client)[0]);
        }
        check(distinctMasks.size() == clients, "each client gets a different mask");
        double maskMean = 0;
        int maskSamples = 0;
        for (int a = 0; a < clients; a++) {
            for (int b = a + 1; b < clients; b++) {
                for (int k = 0; k < 500; k++) {
                    maskMean += aggregator.maskValue(a, b, k) / (double) SecureAggregator.MODULUS;
                    maskSamples++;
                }
            }
        }
        maskMean /= maskSamples;
        check(maskSamples > 5000, "the uniformity sample must be large: " + maskSamples);
        check(Math.abs(maskMean - 0.5) < 0.02, "pairwise masks look uniform: mean " + maskMean);
        check(aggregator.maskValue(1, 2, 0) == aggregator.maskValue(2, 1, 0),
                "both peers derive the same mask value");

        // --- dropout patterns -----------------------------------------------------
        for (int droppedClient = 0; droppedClient < clients; droppedClient++) {
            boolean[] alive = new boolean[clients];
            Arrays.fill(alive, true);
            alive[droppedClient] = false;
            long[] withDropout = aggregator.recover(masked, alive);
            check(Arrays.equals(withDropout, SecureAggregator.bruteForceSum(updates, alive)),
                    "dropping client " + droppedClient + " must still give the survivors' sum");
            long[] naive = new long[dimension];
            for (int client = 0; client < clients; client++) {
                if (!alive[client]) continue;
                for (int k = 0; k < dimension; k++) {
                    naive[k] = SecureAggregator.add(naive[k], masked.get(client)[k]);
                }
            }
            check(!Arrays.equals(naive, SecureAggregator.bruteForceSum(updates, alive)),
                    "without the revealed seeds the sum must be wrong");
        }

        boolean[] halfAlive = new boolean[clients];
        for (int client = 0; client < clients; client++) halfAlive[client] = client % 2 == 0;
        check(Arrays.equals(aggregator.recover(masked, halfAlive),
                        SecureAggregator.bruteForceSum(updates, halfAlive)),
                "half the clients dropping must still work");
        boolean[] noneAlive = new boolean[clients];
        long[] nothing = aggregator.recover(masked, noneAlive);
        for (long value : nothing) check(value == 0, "with no survivors the sum is zero");

        // --- the federated averaging use case -----------------------------------
        int scale = 1000;
        double[][] rawUpdates = new double[clients][dimension];
        List<long[]> quantizedUpdates = new ArrayList<>();
        List<long[]> quantizedMasked = new ArrayList<>();
        for (int client = 0; client < clients; client++) {
            for (int k = 0; k < dimension; k++) rawUpdates[client][k] = nextInt(2000) / 1000.0;
            long[] quantized = SecureAggregator.quantize(rawUpdates[client], scale);
            quantizedUpdates.add(quantized);
            quantizedMasked.add(aggregator.mask(client, quantized));
        }
        double[] plainAverage = new double[dimension];
        for (int client = 0; client < clients; client++) {
            for (int k = 0; k < dimension; k++) plainAverage[k] += rawUpdates[client][k] / clients;
        }
        boolean[] federatedAlive = new boolean[clients];
        Arrays.fill(federatedAlive, true);
        federatedAlive[3] = false;
        int surviving = 0;
        for (int client = 0; client < clients; client++) {
            if (federatedAlive[client]) surviving++;
        }
        long[] secureSum = aggregator.recover(quantizedMasked, federatedAlive);
        double[] secureAverage = new double[dimension];
        for (int k = 0; k < dimension; k++) secureAverage[k] = secureSum[k] / (double) scale / surviving;
        double[] expectedAverage = new double[dimension];
        for (int client = 0; client < clients; client++) {
            if (!federatedAlive[client]) continue;
            for (int k = 0; k < dimension; k++) expectedAverage[k] += rawUpdates[client][k];
        }
        for (int k = 0; k < dimension; k++) expectedAverage[k] /= surviving;
        for (int k = 0; k < dimension; k++) {
            // each entry is off by at most half a quantum, so the average is off by at most 0.5/scale
            check(Math.abs(secureAverage[k] - expectedAverage[k]) <= 0.5 / scale + 1e-9,
                    "secure average element " + k + ": " + secureAverage[k] + " vs "
                            + expectedAverage[k]);
        }
        check(Math.abs(plainAverage[0] - expectedAverage[0]) > 1e-12,
                "who participates changes the federated average");

        // --- rejected input --------------------------------------------------------
        check(rejects(() -> new SecureAggregator(1, 3, 1L)), "a single client is rejected");
        check(rejects(() -> new SecureAggregator(3, 0, 1L)), "zero dimension rejected");
        check(rejects(() -> aggregator.mask(9, new long[dimension])), "unknown client rejected");
        check(rejects(() -> aggregator.mask(0, new long[dimension + 1])), "dimension mismatch rejected");
        check(rejects(() -> aggregator.recover(masked.subList(0, 3), allAlive)),
                "a short contribution list is rejected");
        check(rejects(() -> aggregator.recover(masked, new boolean[2])), "wrong alive array rejected");
        check(rejects(() -> SecureAggregator.quantize(new double[]{1.0}, 0)), "zero scale rejected");
        check(rejects(() -> SecureAggregator.quantize(new double[]{1e9}, 1000)),
                "out-of-field quantised value rejected");

        // --- determinism -----------------------------------------------------------
        SecureAggregator rebuilt = new SecureAggregator(clients, dimension, 0x51DE55EEDL);
        long[] again = rebuilt.mask(0, updates.get(0));
        check(Arrays.equals(again, masked.get(0)), "the protocol must be deterministic");
        check(aggregator.seedLookups() > 0, "the PRG was exercised: " + aggregator.seedLookups());

        System.out.println("recovered " + dimension + " coordinates from " + clients
                + " clients with dropout recovery, seed lookups = " + aggregator.seedLookups());
        System.out.println("PASS");
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
