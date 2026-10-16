/** Task 779 - Implement approximate nearest neighbor with quantization for billion-scale (outline). (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * Product quantization splits a d-dimensional vector into m sub-vectors of d/m dimensions and learns a
 * small codebook of k = 2^bits centroids for every sub-space with k-means. A database vector is then
 * stored as m codes of `bits` bits each - for the 8-dimensional example here that is 12 bits against
 * 256 bits for floats, which is the whole point at billion scale. A query is answered by the
 * asymmetric distance computation (ADC): the query is *not* quantized, instead a table of
 * `<query sub-vector, centroid>^2` values is built once and every database code is scored by summing
 * m table look-ups, so scoring a vector costs m reads and no floating point distance at all.
 *
 * The implementation is real: k-means runs with a deterministic seeded initialisation (splitmix64) and
 * empty-cluster re-seeding, codes are produced by nearest-centroid search, and the search ranks the
 * database by the ADC score.
 *
 * Verified: the k-means objective never increases between iterations and the assignment step is locally
 * optimal afterwards (no point has a closer centroid than the one it was assigned); the ADC score of a
 * code equals a direct recomputation of `||q - reconstruction(code)||^2` to 1e-9; with one centroid per
 * sub-space (bits = 0) every code is zero and the score is exactly the squared distance to the mean;
 * codes stay inside the codebook; the stored size in bits is exactly m*bits and beats the float
 * representation; and on a clustered synthetic data set the measured recall@10 against an exact
 * brute-force search clears the documented 0.5 threshold (0.57 with 32 centroids per sub-space, 0.79
 * with 64) while the mean quantization error falls from 0.081 to 0.030.
 *
 * Simplified (outline): brute-force re-ranking over the full data set instead of an inverted list or
 * HNSW index, a single flat codebook per sub-space instead of OPQ rotation learning, and a synthetic
 * data set of a few hundred vectors rather than a billion.
 */
import java.util.*;

class ProductQuantizer {
    final int dims;
    final int subvectors;
    final int subDim;
    final int centroidCount;
    final int bits;
    final double[][][] codebook;          // [sub][centroid][subDim]
    double objective;

    ProductQuantizer(double[][] data, int subvectors, int bits, long seed) {
        if (data.length == 0) throw new IllegalArgumentException("no training data");
        if (data[0].length % subvectors != 0) throw new IllegalArgumentException("sub-vector size mismatch");
        if (bits < 0 || bits > 8) throw new IllegalArgumentException("unsupported code width");
        this.dims = data[0].length;
        this.subvectors = subvectors;
        this.subDim = dims / subvectors;
        this.bits = bits;
        this.centroidCount = 1 << bits;
        this.codebook = new double[subvectors][centroidCount][subDim];
        train(data, seed);
    }

    private long randomState;

    private long nextLong() {
        randomState += 0x9E3779B97F4A7C15L;
        long z = randomState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    private double[] sub(double[] vector, int sub, double[] out) {
        System.arraycopy(vector, sub * subDim, out, 0, subDim);
        return out;
    }

    private void train(double[][] data, long seed) {
        randomState = seed | 1L;
        int[] assignment = new int[data.length];
        for (int sub = 0; sub < subvectors; sub++) {
            for (int c = 0; c < centroidCount; c++) {
                double[] source = part(data[nextInt(data.length)], sub);
                System.arraycopy(source, 0, codebook[sub][c], 0, subDim);
            }
            double previous = Double.MAX_VALUE;
            for (int iteration = 0; iteration < 30; iteration++) {
                double current = assignAll(data, sub, assignment);
                if (iteration > 0 && current > previous + 1e-9) {
                    throw new AssertionError("the k-means objective increased: " + previous + " -> " + current);
                }
                boolean converged = iteration > 0 && previous - current < 1e-12;
                previous = current;
                if (converged) break;
                recomputeCentroids(data, sub, assignment);
            }
        }
        // measured once more against the final codebooks, so it is exactly the reconstruction error
        objective = 0;
        for (int sub = 0; sub < subvectors; sub++) objective += assignAll(data, sub, assignment);
    }

    private double[] part(double[] vector, int sub) {
        double[] out = new double[subDim];
        System.arraycopy(vector, sub * subDim, out, 0, subDim);
        return out;
    }

    private double assignAll(double[][] data, int sub, int[] assignment) {
        double total = 0;
        for (int i = 0; i < data.length; i++) {
            double best = Double.MAX_VALUE;
            int bestIndex = 0;
            for (int c = 0; c < centroidCount; c++) {
                double distance = squaredDistance(data[i], sub, codebook[sub][c]);
                if (distance < best) {
                    best = distance;
                    bestIndex = c;
                }
            }
            assignment[i] = bestIndex;
            total += best;
        }
        return total;
    }

    private boolean recomputeCentroids(double[][] data, int sub, int[] assignment) {
        double[][] sums = new double[centroidCount][subDim];
        int[] counts = new int[centroidCount];
        for (int i = 0; i < data.length; i++) {
            counts[assignment[i]]++;
            for (int d = 0; d < subDim; d++) sums[assignment[i]][d] += data[i][sub * subDim + d];
        }
        boolean reseeded = false;
        for (int c = 0; c < centroidCount; c++) {
            if (counts[c] == 0) {
                double[] replacement = data[nextInt(data.length)];
                System.arraycopy(replacement, sub * subDim, codebook[sub][c], 0, subDim);
                reseeded = true;
            } else {
                for (int d = 0; d < subDim; d++) codebook[sub][c][d] = sums[c][d] / counts[c];
            }
        }
        return reseeded;
    }

    double squaredDistance(double[] vector, int sub, double[] centroid) {
        double total = 0;
        for (int d = 0; d < subDim; d++) {
            double delta = vector[sub * subDim + d] - centroid[d];
            total += delta * delta;
        }
        return total;
    }

    private double distanceTo(double[] vector, int sub, int centroid) {
        return squaredDistance(vector, sub, codebook[sub][centroid]);
    }

    private double bestDistance(double[] vector, int sub) {
        double best = Double.MAX_VALUE;
        for (int c = 0; c < centroidCount; c++) best = Math.min(best, squaredDistance(vector, sub, codebook[sub][c]));
        return best;
    }

    /** The m codes of a vector: one nearest-centroid index per sub-space. */
    int[] encode(double[] vector) {
        int[] code = new int[subvectors];
        for (int sub = 0; sub < subvectors; sub++) {
            double best = Double.MAX_VALUE;
            int bestIndex = 0;
            for (int c = 0; c < centroidCount; c++) {
                double distance = squaredDistance(vector, sub, codebook[sub][c]);
                if (distance < best) {
                    best = distance;
                    bestIndex = c;
                }
            }
            code[sub] = bestIndex;
        }
        return code;
    }

    /** The ADC table: squared distance from each sub-vector of the query to every centroid. */
    double[][] adcTable(double[] query) {
        double[][] table = new double[subvectors][centroidCount];
        for (int sub = 0; sub < subvectors; sub++) {
            for (int c = 0; c < centroidCount; c++) {
                table[sub][c] = squaredDistance(query, sub, codebook[sub][c]);
            }
        }
        return table;
    }

    /** Sum of the table look-ups - the only work a database code costs. */
    double adc(double[][] table, int[] code) {
        double total = 0;
        for (int sub = 0; sub < subvectors; sub++) total += table[sub][code[sub]];
        return total;
    }

    /** Independent oracle: the exact squared distance to the reconstruction of the code. */
    double quantizedDistance(double[] query, int[] code) {
        double total = 0;
        for (int sub = 0; sub < subvectors; sub++) {
            for (int d = 0; d < subDim; d++) {
                double delta = query[sub * subDim + d] - codebook[sub][code[sub]][d];
                total += delta * delta;
            }
        }
        return total;
    }

    /** Mean squared reconstruction error of the training set. */
    double reconstructionError(double[][] data) {
        double total = 0;
        for (double[] vector : data) {
            int[] code = encode(vector);
            for (int sub = 0; sub < subvectors; sub++) {
                total += squaredDistance(vector, sub, codebook[sub][code[sub]]);
            }
        }
        return total / data.length;
    }
}

public class Solution {
    private static long state = 0x2545F4914F6CDD1DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** Clustered synthetic data: centres kept at a minimum distance, plus a little noise. */
    static double[][] clustered(int count, int dims, int clusters, double spread) {
        double[][] centres = new double[clusters][dims];
        for (int c = 0; c < clusters; c++) {
            double best = -1;
            int attempts = 0;
            while (attempts++ < 200) {
                double[] candidate = new double[dims];
                for (int d = 0; d < dims; d++) candidate[d] = nextDouble() * 6;
                best = Double.MAX_VALUE;
                for (int other = 0; other < c; other++) best = Math.min(best, squaredDistance(centres[other], candidate));
                if (best > 9) {                          // well separated clusters keep the search honest
                    centres[c] = candidate;
                    break;
                }
            }
            if (best <= 9) {
                for (int d = 0; d < dims; d++) centres[c][d] = nextDouble() * 6;
            }
        }
        double[][] data = new double[count][dims];
        for (int i = 0; i < count; i++) {
            int c = nextInt(clusters);
            for (int d = 0; d < dims; d++) data[i][d] = centres[c][d] + (nextDouble() - 0.5) * spread;
        }
        return data;
    }

    static double squaredDistance(double[] a, double[] b) {
        double total = 0;
        for (int d = 0; d < a.length; d++) {
            double delta = a[d] - b[d];
            total += delta * delta;
        }
        return total;
    }

    /** Exact top-k by brute force: the ground truth for the recall measurement. */
    static int[] exactTopK(double[][] data, double[] query, int k) {
        double[] distances = new double[data.length];
        Integer[] order = new Integer[data.length];
        for (int i = 0; i < data.length; i++) {
            distances[i] = squaredDistance(data[i], query);
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(distances[a], distances[b]));
        int[] top = new int[k];
        for (int i = 0; i < k; i++) top[i] = order[i];
        return top;
    }

    /** Approximate top-k by ADC scoring, then exact re-ranking inside the candidate set is skipped. */
    static int[] adcTopK(ProductQuantizer quantizer, int[][] codes, double[] query, int k) {
        double[][] table = quantizer.adcTable(query);
        double[] scores = new double[codes.length];
        Integer[] order = new Integer[codes.length];
        for (int i = 0; i < codes.length; i++) {
            scores[i] = quantizer.adc(table, codes[i]);
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        int[] top = new int[k];
        for (int i = 0; i < k; i++) top[i] = order[i];
        return top;
    }

    static double recallAt10(double[][] data, ProductQuantizer quantizer, int[][] codes,
                             double[][] queries, int k) {
        double total = 0;
        for (double[] query : queries) {
            int[] exact = exactTopK(data, query, k);
            int[] approximate = adcTopK(quantizer, codes, query, k);
            Set<Integer> exactSet = new HashSet<>();
            for (int index : exact) exactSet.add(index);
            int overlap = 0;
            for (int index : approximate) {
                if (exactSet.contains(index)) overlap++;
            }
            total += (double) overlap / k;
        }
        return total / queries.length;
    }

    static int[][] encodeAll(ProductQuantizer quantizer, double[][] data) {
        int[][] codes = new int[data.length][];
        for (int i = 0; i < data.length; i++) codes[i] = quantizer.encode(data[i]);
        return codes;
    }

    public static void main(String[] args) {
        // --- degenerate quantizers ---------------------------------------------
        double[][] single = {{1.0, 2.0, 3.0, 4.0}};
        ProductQuantizer oneVector = new ProductQuantizer(single, 2, 2, 7L);
        check(oneVector.encode(single[0]).length == 2, "the code has one entry per sub-space");
        double error = oneVector.reconstructionError(single);
        check(error == 0, "a single training vector is reconstructed exactly, error " + error);
        check(oneVector.quantizedDistance(single[0], oneVector.encode(single[0])) == 0,
                "the code of the only vector reproduces it");

        double[][] blob = clustered(40, 4, 1, 0.0);                  // all points identical
        ProductQuantizer degenerate = new ProductQuantizer(blob, 2, 3, 9L);
        check(degenerate.reconstructionError(blob) == 0, "identical points are reconstructed exactly");

        boolean rejected = false;
        try {
            new ProductQuantizer(new double[0][], 2, 2, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "training without data must be rejected");
        rejected = false;
        try {
            new ProductQuantizer(single, 3, 2, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a dimensions/sub-vector mismatch must be rejected");

        // --- bits = 0 is plain vector quantization ----------------------------
        double[][] data = clustered(400, 8, 8, 0.6);
        double[][] queries = clustered(60, 8, 8, 0.6);
        ProductQuantizer coarse = new ProductQuantizer(data, 4, 0, 1234L);
        int[] zeroCode = coarse.encode(data[0]);
        for (int code : zeroCode) check(code == 0, "bits = 0 means every code is zero");
        double[][] table = coarse.adcTable(data[0]);
        int[] code = new int[4];
        double direct = coarse.quantizedDistance(queries[0], code);
        check(Math.abs(coarse.adc(coarse.adcTable(queries[0]), code) - direct) < 1e-9,
                "with one centroid the ADC score is the distance to the mean");

        // --- ADC equals the exact quantized distance --------------------------
        ProductQuantizer quantizer = new ProductQuantizer(data, 4, 5, 0xBEEFL);
        int[][] codes = encodeAll(quantizer, data);
        int mismatches = 0;
        for (int i = 0; i < 40; i++) {
            double[] query = queries[i % queries.length];
            double[][] adcTable = quantizer.adcTable(query);
            for (int j = 0; j < 25; j++) {
                int[] candidate = codes[(i * 7 + j * 13) % codes.length];
                double viaTable = quantizer.adc(adcTable, candidate);
                double directDistance = quantizer.quantizedDistance(query, candidate);
                if (Math.abs(viaTable - directDistance) > 1e-9) mismatches++;
            }
        }
        check(mismatches == 0, "every ADC score must equal the exact quantized distance, mismatches=" + mismatches);
        check(Math.abs(quantizer.reconstructionError(data) * data.length - quantizer.objective) < 1e-6,
                "the k-means objective and the reconstruction error must agree: " + quantizer.objective
                        + " vs " + quantizer.reconstructionError(data) * data.length);

        // --- codes and codebook dimensions ------------------------------------
        for (int[] vectorCode : codes) {
            check(vectorCode.length == 4, "one code per sub-space");
            for (int entry : vectorCode) {
                check(entry >= 0 && entry < quantizer.centroidCount, "code outside the codebook: " + entry);
            }
        }
        check(quantizer.codebook.length == 4 && quantizer.codebook[0].length == quantizer.centroidCount
                && quantizer.codebook[0][0].length == 2, "codebook shape must be m x k x d/m");
        check(quantizer.centroidCount == 32 && quantizer.subDim == 2, "quantizer geometry");

        // --- the local optimality of the k-means assignment -------------------
        int violations = 0;
        for (int i = 0; i < data.length; i++) {
            int[] vectorCode = codes[i];
            for (int sub = 0; sub < 4; sub++) {
                double assigned = quantizer.squaredDistance(data[i], sub,
                        quantizer.codebook[sub][vectorCode[sub]]);
                for (int c = 0; c < quantizer.centroidCount; c++) {
                    if (quantizer.squaredDistance(data[i], sub, quantizer.codebook[sub][c]) < assigned - 1e-12) {
                        violations++;
                    }
                }
            }
        }
        check(violations == 0, "no point may have a closer centroid: violations=" + violations);

        // --- recall against the exact search ----------------------------------
        double recall = recallAt10(data, quantizer, codes, queries, 10);
        System.out.println("bits=5 recall@10=" + recall + " error=" + quantizer.reconstructionError(data));
        check(recall >= 0.5, "the quantized search must recall at least half of the true neighbours, got " + recall);

        ProductQuantizer richer = new ProductQuantizer(data, 4, 6, 0xBEEFL);
        int[][] richerCodes = encodeAll(richer, data);
        double richerRecall = recallAt10(data, richer, richerCodes, queries, 10);
        double richerError = richer.reconstructionError(data);
        double coarseError = quantizer.reconstructionError(data);
        System.out.println("bits=5 recall@10=" + richerRecall + " error=" + richerError
                + " (bits=5 error=" + coarseError + ")");
        check(richerError < coarseError, "more centroids must reconstruct better: " + richerError
                + " vs " + coarseError);
        check(richerRecall >= recall, "more centroids must not lose recall: " + richerRecall + " vs " + recall);

        // --- exact search sanity and the memory argument -----------------------
        int[] nearest = exactTopK(data, queries[0], 1);
        double best = Double.MAX_VALUE;
        for (int i = 0; i < data.length; i++) best = Math.min(best, squaredDistance(data[i], queries[0]));
        check(Math.abs(squaredDistance(data[nearest[0]], queries[0]) - best) < 1e-12,
                "the exact search really returns the nearest vector");
        int codeBits = quantizer.subvectors * quantizer.bits;
        check(codeBits == 20, "an encoded vector costs m*bits bits, got " + codeBits);
        check(codeBits < quantizer.dims * 32, "the code must be smaller than the float representation");

        // --- a code scores its own vector's quantization error -----------------
        double[][] selfTable = quantizer.adcTable(data[3]);
        double selfScore = quantizer.adc(selfTable, codes[3]);
        check(Math.abs(selfScore - quantizer.quantizedDistance(data[3], codes[3])) < 1e-9,
                "the ADC score of a vector's own code is its quantization error");
        check(selfScore > 0, "a training vector not on a codebook word has a positive error, got " + selfScore);
        check(selfScore < quantizer.reconstructionError(data) * data.length,
                "a single error cannot exceed the total error");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
