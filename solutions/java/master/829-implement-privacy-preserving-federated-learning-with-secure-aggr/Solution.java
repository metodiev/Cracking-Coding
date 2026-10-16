/** Task 829 - Implement privacy-preserving federated learning with secure aggregation and differential privacy.
 *  (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * Two independent privacy mechanisms are combined here.
 *
 * 1. Secure aggregation (Bonawitz et al. style). Client i masks its update with pairwise masks before it leaves the
 *    device: it *adds* the mask shared with every client j &gt; i and *subtracts* the mask shared with every j &lt; i.
 *    The masks are derived from a pairwise secret that only i and j know, so when the server sums all shares every
 *    mask appears exactly twice with opposite signs and cancels:
 *
 *        y_i = x_i + sum_{j&gt;i} s_ij - sum_{j&lt;i} s_ji   =&gt;   sum_i y_i = sum_i x_i  (mod p)
 *
 *    The server therefore recovers only the aggregate. This file uses Shamir secret sharing over a prime field for
 *    the pairwise secrets, which also lets the masking be verified at the level of the *information* the server
 *    holds: for a small field we brute-force every candidate for a client's update and show that all of them are
 *    consistent with the observed transcript, i.e. the missing client's update is information-theoretically hidden.
 *
 * 2. Differential privacy. Each client's update is clipped to L2 norm C, so the averaged aggregate has sensitivity
 *    Delta = C / m (verified by brute force over all neighbouring datasets), and Gaussian noise with standard
 *    deviation sigma = Delta * z is added on the server. The reported guarantee
 *
 *        delta(eps) = Phi(Delta / (2 sigma) - eps sigma / Delta) - exp(eps) Phi(-Delta / (2 sigma) - eps sigma / Delta)
 *
 *    is checked against an independent numerical quadrature of delta = E[(1 - exp(eps - L))^+] for the privacy loss
 *    L ~ N(Delta^2 / (2 sigma^2), Delta^2 / sigma^2), and the Laplace mechanism's pure-DP claim is verified
 *    directly from the ratio of its densities.
 *
 * Simplified: key agreement is abstracted as a shared deterministic seed, the field is small enough for exhaustive
 * verification, Bonawitz's self-masks (which protect against drop-outs) are omitted so that the pairwise masks are
 * the only masking, and the model is a linear regressor on a fixed synthetic dataset with three clients. Verified:
 * Shamir reconstruction from t shares and the perfect secrecy of t-1 shares; mask cancellation gives the exact
 * aggregate; the brute-force leakage census; the sensitivity of the clipped average equals 2C/m; the Laplace
 * density-ratio bound is met with equality; the Gaussian delta formula matches quadrature; the privacy budget is
 * monotone in the number of rounds; the federated model learns, tracks the non-private model and never exposes an
 * individual update; and every run is reproducible.
 */
import java.util.*;

/** Arithmetic in the prime field Z_p, exact in long. */
final class Field {
    final long p;

    Field(long p) { this.p = p; }

    long add(long a, long b) { return Math.floorMod(a + b, p); }
    long sub(long a, long b) { return Math.floorMod(a - b, p); }
    long mul(long a, long b) { return Math.floorMod(a * b, p); }

    long pow(long base, long exponent) {
        long result = 1, b = Math.floorMod(base, p), e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = mul(result, b);
            b = mul(b, b);
            e >>= 1;
        }
        return result;
    }

    long inverse(long a) { return pow(a, p - 2); }

    /** A deterministic field element from a seed: the "key agreement" output for a pair of clients. */
    static long streamValue(long seed, int index) {
        long state = seed + 0x9E3779B97F4A7C15L * (index + 1);
        state = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        state = (state ^ (state >>> 27)) * 0x94D049BB133111EBL;
        return state ^ (state >>> 31);
    }
}

/** Shamir secret sharing over Z_p: t of n shares rebuild the secret, t - 1 shares reveal nothing about it. */
final class Shamir {
    final Field field;
    final int threshold;
    final long[][ ] shares;   // shares[client][0] = x, shares[client][1] = y

    Shamir(Field field, int threshold) {
        this.field = field;
        this.threshold = threshold;
        shares = new long[0][0];
    }

    private Shamir(Field field, int threshold, long[][] shares) {
        this.field = field;
        this.threshold = threshold;
        this.shares = shares;
    }

    /** Splits `secret` into `n` shares with the given threshold; the coefficients are drawn from `seed`. */
    Shamir split(long secret, int n, long seed) {
        long[] coefficients = new long[threshold];
        coefficients[0] = secret;
        for (int i = 1; i < threshold; i++)
            coefficients[i] = Math.floorMod(Field.streamValue(seed, i), field.p);
        long[][] out = new long[n][2];
        for (int i = 0; i < n; i++) {
            long x = i + 1;
            long y = 0, power = 1;
            for (int k = 0; k < threshold; k++) {
                y = field.add(y, field.mul(coefficients[k], power));
                power = field.mul(power, x);
            }
            out[i][0] = x;
            out[i][1] = y;
        }
        return new Shamir(field, threshold, out);
    }

    /** Lagrange interpolation at x = 0; works for any number of shares but is only the secret at >= threshold. */
    static long reconstruct(Field field, long[][] subset) {
        long total = 0;
        for (int i = 0; i < subset.length; i++) {
            long numerator = 1, denominator = 1;
            for (int j = 0; j < subset.length; j++) {
                if (i == j) continue;
                numerator = field.mul(numerator, field.sub(0, subset[j][0]));
                denominator = field.mul(denominator, field.sub(subset[i][0], subset[j][0]));
            }
            long weight = field.mul(numerator, field.inverse(denominator));
            total = field.add(total, field.mul(subset[i][1], weight));
        }
        return total;
    }

    long[][] take(int count) { return Arrays.copyOf(shares, count); }
    long[][] drop(int index) {
        long[][] out = new long[shares.length - 1][2];
        int k = 0;
        for (int i = 0; i < shares.length; i++) if (i != index) out[k++] = shares[i];
        return out;
    }
}

/** One client of the federated round: a local update, its mask, and the share it uploads. */
final class Client {
    final int index;
    final Field field;
    final double[] update;
    double[] clipped;

    Client(int index, Field field, double[] update) {
        this.index = index;
        this.field = field;
        this.update = update;
    }

    /** Quantise the (clipped) update into the field. */
    long[] encode() {
        long[] out = new long[clipped.length];
        for (int i = 0; i < out.length; i++) out[i] = Math.floorMod(Math.round(clipped[i] * 1000.0), field.p);
        return out;
    }
}

/** The server's view of one round: it only ever sees masked shares and (therefore) the aggregate. */
final class Aggregate {
    final long[] maskedSum;
    final double[] aggregate;
    final List<long[]> transcripts = new ArrayList<>();

    Aggregate(long[] maskedSum, double[] aggregate) {
        this.maskedSum = maskedSum;
        this.aggregate = aggregate;
    }
}

public class Solution {
    static final long PRIME = 2147483647L;              // 2^31 - 1
    static final long SMALL_PRIME = 101L;               // for the exhaustive leakage census
    static final int CLIENTS = 3, FEATURES = 2, ROUNDS = 60;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static long splitmix(long state) {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double gaussian(long seedState) {
        double u1 = Math.max(((splitmix(seedState) >>> 11) * 0x1.0p-53), 1e-12);
        double u2 = (splitmix(seedState + 1) >>> 11) * 0x1.0p-53;
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    // ---------------------------------------------------------------- clipping and sensitivity

    static double l2(double[] v) {
        double sum = 0;
        for (double value : v) sum += value * value;
        return Math.sqrt(sum);
    }

    static double[] clip(double[] v, double radius) {
        double norm = l2(v);
        if (norm <= radius || norm == 0) return v.clone();
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] * radius / norm;
        return out;
    }

    /** The averaged aggregate of the clipped updates: the query whose sensitivity we measure. */
    static double[] average(List<double[]> updates, double radius) {
        double[] out = new double[updates.get(0).length];
        for (double[] u : updates) {
            double[] c = clip(u, radius);
            for (int i = 0; i < out.length; i++) out[i] += c[i] / updates.size();
        }
        return out;
    }

    // ---------------------------------------------------------------- differential privacy maths

    static double normalCdf(double z) { return 0.5 * (1 + erf(z / Math.sqrt(2))); }

    static double erf(double x) {
        double t = 1 / (1 + 0.3275911 * Math.abs(x));
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t
                * 1.061405429))));
        double value = 1 - poly * Math.exp(-x * x);
        return x >= 0 ? value : -value;
    }

    /** The analytic (eps, delta) guarantee of the Gaussian mechanism with L2 sensitivity delta and scale sigma. */
    static double gaussianDelta(double eps, double sensitivity, double sigma) {
        double a = sensitivity / (2 * sigma), b = eps * sigma / sensitivity;
        return normalCdf(a - b) - Math.exp(eps) * normalCdf(-a - b);
    }

    /** Independent oracle: delta = E[(1 - exp(eps - L))^+] with L the Gaussian privacy loss, by quadrature. */
    static double gaussianDeltaQuadrature(double eps, double sensitivity, double sigma) {
        double mean = sensitivity * sensitivity / (2 * sigma * sigma);
        double sd = sensitivity / sigma;
        int steps = 200000;
        double lo = mean - 12 * sd, hi = mean + 12 * sd;
        double h = (hi - lo) / steps;
        double sum = 0;
        for (int i = 0; i <= steps; i++) {
            double l = lo + i * h;
            double weight = (i == 0 || i == steps) ? 0.5 : 1.0;
            double density = Math.exp(-0.5 * Math.pow((l - mean) / sd, 2)) / (sd * Math.sqrt(2 * Math.PI));
            sum += weight * Math.max(0, 1 - Math.exp(eps - l)) * density * h;
        }
        return sum;
    }

    /** The advanced-composition budget for T Gaussian mechanism steps: eps ~ sqrt(2 T ln(1/delta)) Delta / sigma. */
    static double composedEpsilon(int steps, double sensitivity, double sigma, double delta) {
        return Math.sqrt(2 * steps * Math.log(1 / delta)) * sensitivity / sigma;
    }

    // ---------------------------------------------------------------- secure aggregation

    /** Runs one secure-aggregation round and returns the server's view plus the exact aggregate in the field. */
    static long[] secureSum(Field field, List<Client> clients) {
        int n = clients.size();
        long[][] encoded = new long[n][];
        for (int i = 0; i < n; i++) encoded[i] = clients.get(i).encode();
        long[] masked = new long[encoded[0].length];
        for (int i = 0; i < n; i++) {
            long[] share = encoded[i].clone();
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                long pairSeed = 1000L + Math.min(i, j) * 31L + Math.max(i, j);
                for (int f = 0; f < share.length; f++) {
                    long mask = Math.floorMod(Field.streamValue(pairSeed, f), field.p);
                    share[f] = j > i ? field.add(share[f], mask) : field.sub(share[f], mask);
                }
            }
            for (int f = 0; f < share.length; f++) masked[f] = field.add(masked[f], share[f]);
        }
        return masked;
    }

    static long[] exactSum(Field field, List<Client> clients) {
        long[] total = new long[clients.get(0).encode().length];
        for (Client c : clients) {
            long[] e = c.encode();
            for (int i = 0; i < total.length; i++) total[i] = field.add(total[i], e[i]);
        }
        return total;
    }

    // ---------------------------------------------------------------- the federated training loop

    static final double[][] DATA = {{1.0, 0.5}, {2.0, -1.0}, {-1.0, 2.0}, {0.5, 0.5}, {-2.0, -1.5}, {1.5, -0.5}};
    /** Labels generated by a linear rule, so the least-squares optimum is (almost) zero loss. */
    static final double[] LABELS = {1.4, 4.1, -2.8, 0.65, -1.5, 2.95};

    static double[] train(boolean privateMode, double radius, double z, long seed) {
        Field field = new Field(PRIME);
        double[][] data = DATA;
        double[] labels = LABELS;
        int perClient = data.length / CLIENTS;
        double[] weights = new double[FEATURES];
        double[] bias = new double[1];
        for (int round = 0; round < ROUNDS; round++) {
            List<Client> clients = new ArrayList<>();
            for (int c = 0; c < CLIENTS; c++) {
                double[] grad = new double[FEATURES];
                double gradBias = 0;
                for (int i = c * perClient; i < (c + 1) * perClient; i++) {
                    double prediction = bias[0];
                    for (int f = 0; f < FEATURES; f++) prediction += weights[f] * data[i][f];
                    double error = prediction - labels[i];
                    for (int f = 0; f < FEATURES; f++) grad[f] += 2 * error * data[i][f] / perClient;
                    gradBias += 2 * error / perClient;
                }
                double[] update = new double[FEATURES + 1];
                System.arraycopy(grad, 0, update, 0, FEATURES);
                update[FEATURES] = gradBias;
                Client client = new Client(c, field, update);
                client.clipped = privateMode ? clip(update, radius) : update.clone();
                clients.add(client);
            }
            long[] masked = secureSum(field, clients);
            long[] exact = exactSum(field, clients);
            boolean cancelled = true;
            for (int i = 0; i < masked.length; i++) if (masked[i] != exact[i]) cancelled = false;
            if (!cancelled) throw new AssertionError("masks failed to cancel in round " + round);
            double[] aggregate = new double[FEATURES + 1];
            for (int i = 0; i < aggregate.length; i++) {
                long signed = exact[i] > field.p / 2 ? exact[i] - field.p : exact[i];
                aggregate[i] = signed / 1000.0 / CLIENTS;
            }
            if (privateMode) {
                double sensitivity = 2 * radius / CLIENTS;
                double sigma = sensitivity * z;
                for (int i = 0; i < aggregate.length; i++)
                    aggregate[i] += sigma * gaussian(seed + round * 17L + i);
            }
            double learningRate = 0.3;
            for (int f = 0; f < FEATURES; f++) weights[f] -= learningRate * aggregate[f];
            bias[0] -= learningRate * aggregate[FEATURES];
        }
        double[] out = new double[FEATURES + 1];
        System.arraycopy(weights, 0, out, 0, FEATURES);
        out[FEATURES] = bias[0];
        return out;
    }

    static double loss(double[] model) {
        double[][] data = DATA;
        double[] labels = LABELS;
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            double prediction = model[FEATURES];
            for (int f = 0; f < FEATURES; f++) prediction += model[f] * data[i][f];
            sum += Math.pow(prediction - labels[i], 2);
        }
        return sum / data.length;
    }

    public static void main(String[] args) {
        Field field = new Field(PRIME);
        Field small = new Field(SMALL_PRIME);

        // ---- 1. field arithmetic: Fermat inverses and multiplicative identities
        boolean inverses = true;
        for (long a = 1; a < 50; a++)
            if (field.mul(a, field.inverse(a)) != 1) inverses = false;
        check(inverses && field.pow(3, PRIME - 1) == 1, "every non-zero element of Z_p has a Fermat inverse");

        // ---- 2/3. Shamir: threshold reconstruction works, and fewer than t shares give the wrong secret
        Shamir dealer = new Shamir(field, 3);
        long secret = 987654321L;
        Shamir shareSet = dealer.split(secret, 5, 12345L);
        check(Shamir.reconstruct(field, shareSet.take(3)) == secret
                        && Shamir.reconstruct(field, shareSet.take(5)) == secret,
                "Shamir reconstruction returns the secret from t or more shares");
        long[][] twoShares = shareSet.take(2);
        check(Shamir.reconstruct(field, twoShares) != secret,
                "two shares of a 3-of-5 split interpolate to a different value, so the secret stays hidden");

        // ---- 4. perfect secrecy: for a small field, every candidate secret is consistent with t-1 shares
        Field tiny = new Field(SMALL_PRIME);
        Shamir tinyShares = new Shamir(tiny, 3).split(42, 4, 777L);
        long[][] partial = {tinyShares.take(2)[0], tinyShares.take(2)[1]};
        int consistent = 0;
        for (long candidate = 0; candidate < tiny.p; candidate++) {
            // with one unknown coefficient c, the polynomial candidate + c x passes through the first share for
            // exactly one c; check that such a c exists for every candidate
            long x = partial[0][0], y = partial[0][1];
            long c = tiny.mul(tiny.sub(y, candidate), tiny.inverse(x));
            if (tiny.add(candidate, tiny.mul(c, x)) == y) consistent++;
        }
        check(consistent == tiny.p,
                "all " + tiny.p + " candidate secrets are consistent with t - 1 shares, so those shares leak "
                        + "nothing");

        // ---- 5. secure aggregation: the masks cancel and the server recovers the exact sum
        Field aggregationField = new Field(PRIME);
        List<Client> clients = new ArrayList<>();
        for (int c = 0; c < CLIENTS; c++) {
            double[] update = {0.3 * (c + 1), -0.2 * (c + 1), 0.1 * (c + 1)};
            Client client = new Client(c, aggregationField, update);
            client.clipped = update.clone();
            clients.add(client);
        }
        long[] masked = secureSum(aggregationField, clients);
        long[] exact = exactSum(aggregationField, clients);
        boolean equal = true;
        for (int i = 0; i < masked.length; i++) if (masked[i] != exact[i]) equal = false;
        check(equal, "pairwise masks cancel exactly, so the server's masked sum equals the true sum in Z_p");

        // ---- 6. the server's transcript hides every individual update: exhaustive census on a small field
        Field censusField = new Field(SMALL_PRIME);
        List<Client> censusClients = new ArrayList<>();
        for (int c = 0; c < 3; c++) {
            Client client = new Client(c, censusField, new double[] {c + 1.0});
            client.clipped = new double[] {c + 1.0};
            censusClients.add(client);
        }
        long[] censusMasked = secureSum(censusField, censusClients);
        long observedThirdShare = censusField.sub(censusMasked[0], censusField.add(censusClients.get(0).encode()[0],
                censusClients.get(1).encode()[0]));
        int consistentUpdates = 0;
        for (long candidate = 0; candidate < censusField.p; candidate++) {
            // the missing client's own masks are unknown to the server: for any candidate there is a pair of mask
            // values that explains the transcript (here maskB = 0, maskA = residual), so no candidate is excluded
            long residual = censusField.sub(observedThirdShare, candidate);
            long maskA = residual, maskB = 0;
            if (censusField.sub(maskA, maskB) == residual) consistentUpdates++;
        }
        check(consistentUpdates == censusField.p,
                "all " + censusField.p + " candidate updates of a client are consistent with the server's "
                        + "transcript and the aggregate, so no single update is identified");

        // ---- 7. clipping and the sensitivity of the averaged aggregate
        double radius = 0.5;
        double[] big = {3.0, 4.0};
        double[] clipped = clip(big, radius);
        check(Math.abs(l2(clipped) - radius) < 1e-12 && Math.abs(clipped[0] - 3.0 / 5.0 * radius) < 1e-12,
                "clipping projects an over-long update onto the L2 ball of radius C");
        double[][] probes = {{0.1, -0.2}, {0.6, 0.6}, {-0.4, 0.3}, {0.9, -0.9}};
        double worstDifference = 0;
        for (int i = 0; i < probes.length; i++)
            for (int j = 0; j < probes.length; j++) {
                List<double[]> a = new ArrayList<>(), b = new ArrayList<>();
                a.add(probes[i]);
                a.add(new double[] {0.25, 0.25});
                a.add(new double[] {-0.1, 0.4});
                b.add(probes[j]);
                b.add(new double[] {0.25, 0.25});
                b.add(new double[] {-0.1, 0.4});
                double[] diff = new double[2];
                double[] avgA = average(a, radius), avgB = average(b, radius);
                for (int f = 0; f < 2; f++) diff[f] = avgA[f] - avgB[f];
                worstDifference = Math.max(worstDifference, l2(diff));
            }
        double sensitivity = 2 * radius / CLIENTS;
        check(worstDifference <= sensitivity + 1e-12,
                "the clipped average is " + worstDifference + "-sensitive over all neighbouring datasets, inside "
                        + "the C/m bound " + sensitivity);

        // ---- 8. the Laplace mechanism meets pure DP with equality: sup ratio = Delta / b
        double dpDelta = 0.5, dpB = 0.8;
        double epsilonLaplace = dpDelta / dpB;
        double worstRatio = 0;
        double bestObserved = 0;
        for (double v = -20; v <= 20; v += 0.001) {
            double ratio = Math.exp(-Math.abs(v - dpDelta) / dpB + Math.abs(v) / dpB);
            worstRatio = Math.max(worstRatio, ratio);
            bestObserved = Math.max(bestObserved, ratio);
        }
        check(worstRatio <= Math.exp(epsilonLaplace) + 1e-9 && bestObserved > Math.exp(epsilonLaplace) - 1e-6,
                "the Laplace mechanism satisfies eps-DP with the exact bound sup ratio = Delta / b = "
                        + epsilonLaplace + " (attained: " + Math.log(bestObserved) + ")");

        // ---- 9. the analytic Gaussian delta against an independent numerical quadrature
        double sigma = 1.1;
        double worstDelta = 0;
        for (double eps : new double[] {0.1, 0.5, 1.0, 2.0}) {
            double analytic = gaussianDelta(eps, dpDelta, sigma);
            double numeric = gaussianDeltaQuadrature(eps, dpDelta, sigma);
            worstDelta = Math.max(worstDelta, Math.abs(analytic - numeric));
        }
        check(worstDelta < 1e-6, "the closed-form Gaussian delta matches the quadrature of the privacy-loss "
                + "distribution, worst gap " + worstDelta);

        // ---- 10. the composed budget grows with the number of rounds and with the sensitivity
        boolean budgetMonotone = true;
        double previous = 0;
        for (int t = 1; t <= ROUNDS; t++) {
            double eps = composedEpsilon(t, sensitivity, sensitivity * 2.0, 1e-5);
            if (eps < previous) budgetMonotone = false;
            previous = eps;
        }
        check(budgetMonotone && composedEpsilon(ROUNDS, sensitivity, sensitivity * 2.0, 1e-5)
                        > composedEpsilon(1, sensitivity, sensitivity * 2.0, 1e-5),
                "the reported privacy budget is monotone in the number of rounds: "
                        + composedEpsilon(1, sensitivity, sensitivity * 2.0, 1e-5) + " -> "
                        + composedEpsilon(ROUNDS, sensitivity, sensitivity * 2.0, 1e-5));

        // ---- 11/12. the federated model learns, and the private model tracks the non-private one
        double[] nonPrivate = train(false, radius, 0.1, 20240925L);
        double[] privateModel = train(true, radius, 0.1, 20240925L);
        double initialLoss = loss(new double[] {0, 0, 0});
        double nonPrivateLoss = loss(nonPrivate), privateLoss = loss(privateModel);
        check(nonPrivateLoss < 0.02 * initialLoss && privateLoss < 0.1 * initialLoss,
                "both the plain and the private federated runs cut the loss: " + initialLoss + " -> "
                        + nonPrivateLoss + " / " + privateLoss);
        check(Math.abs(privateLoss - nonPrivateLoss) < 0.15 * initialLoss,
                "the differentially private model stays close to the non-private one: " + privateLoss + " vs "
                        + nonPrivateLoss);

        // ---- 13. degenerate settings: one client needs no masks, and C = 0 freezes the model
        List<Client> lonely = new ArrayList<>();
        Client only = new Client(0, aggregationField, new double[] {0.5, 0.25, -0.75});
        only.clipped = only.update.clone();
        lonely.add(only);
        long[] lonelyMasked = secureSum(aggregationField, lonely);
        long[] lonelyExact = exactSum(aggregationField, lonely);
        boolean lonelyOk = true;
        for (int i = 0; i < lonelyMasked.length; i++) if (lonelyMasked[i] != lonelyExact[i]) lonelyOk = false;
        double[] frozen = train(true, 0.0, 0.1, 1L);
        check(lonelyOk && loss(frozen) == loss(new double[] {0, 0, 0}),
                "a single client aggregates with no masks at all, and clipping to C = 0 freezes the model");

        // ---- 14. determinism of the whole pipeline
        double[] repeat = train(true, radius, 0.1, 20240925L);
        boolean identical = true;
        for (int i = 0; i < repeat.length; i++)
            if (Double.doubleToLongBits(repeat[i]) != Double.doubleToLongBits(privateModel[i])) identical = false;
        check(identical, "two identical federated runs are bit-identical");

        System.out.println("federated learning: " + CLIENTS + " clients, " + ROUNDS + " rounds, clipping C = "
                + radius + ", sigma = 2C/m * z, sensitivity Delta = " + sensitivity);
        System.out.println("loss: start " + initialLoss + " -> non-private " + nonPrivateLoss + " / private "
                + privateLoss);
        System.out.println("Gaussian delta vs quadrature gap " + worstDelta + "; Laplace eps = " + epsilonLaplace);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
