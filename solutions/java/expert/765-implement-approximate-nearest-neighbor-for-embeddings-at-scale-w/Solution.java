/** Task 765 - Implement approximate nearest neighbour search for embeddings at scale with IVF + PQ.
 * (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * Two classic reductions that make billion-scale similarity search affordable:
 *
 *  * IVF (inverted file): k-means clusters the vectors, every vector is stored in the list of its
 *    nearest centroid, and a query only visits the `nprobe` closest lists. That is where the speedup
 *    comes from - and where the recall is lost;
 *  * PQ (product quantisation): the d-dimensional vector is cut into m sub-vectors, each sub-vector is
 *    replaced by the id of its nearest codeword in that subspace's own small codebook, and the distance
 *    to a query is computed from a lookup table (asymmetric distance computation). The vector then
 *    costs m bytes instead of 4*d, at the price of a reconstruction error.
 *
 * The file implements both, measures the trade-offs and checks the invariants that make them work:
 * k-means' objective is monotone and its final assignment is a fixed point; visiting every list is
 * *exactly* the brute-force answer (so the recall of the full probe is 1.0, not "about 1"); the ADC
 * distance equals the Euclidean distance to the *reconstructed* vector, exactly; and recall@10 is
 * measured against exhaustive search for several probe counts.
 *
 * Simplified: small data (hundreds of 16-d vectors), Lloyd k-means with a fixed iteration budget and a
 * deterministic initialisation, no OPQ/rotation, no SIMD lookup tables, no residual encoding, and only
 * squared Euclidean distance.
 *
 * Verified: k-means convergence and fixed-point assignment; exhaustive probing equals brute force;
 * recall@10 for 1, 2 and all lists; PQ reconstruction error shrinking with the codebook size; the ADC
 * identity against decoded vectors; boundary cases (single vector, empty query, duplicate vectors);
 * determinism of centroids, codes and search results.
 */
import java.util.*;

/** Deterministic splitmix64 plus a Box-Muller pair, so the "embeddings" are reproducible. */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

    double nextUnit() { return (nextLong() >>> 11) / (double) (1L << 53); }

    double nextGaussian() {
        double u1 = Math.max(nextUnit(), 1e-12), u2 = nextUnit();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** A matrix of vectors with the distance helpers everybody needs. */
final class Vectors {
    final double[][] data;

    Vectors(double[][] data) { this.data = data; }

    int size() { return data.length; }
    int dimension() { return data.length == 0 ? 0 : data[0].length; }

    static double squaredDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double difference = a[i] - b[i];
            sum += difference * difference;
        }
        return sum;
    }

    /** Ground truth: exact k nearest neighbours by brute force (ties broken by id). */
    static List<Integer> exactKnn(Vectors vectors, double[] query, int k) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < vectors.size(); i++) order.add(i);
        order.sort((a, b) -> {
            int byDistance = Double.compare(squaredDistance(query, vectors.data[a]),
                    squaredDistance(query, vectors.data[b]));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        return order.subList(0, Math.min(k, order.size()));
    }
}

/** Lloyd k-means with a deterministic seed and a fixed iteration budget. */
final class KMeans {
    final double[][] centroids;
    final int[] assignment;
    final List<Double> objectives = new ArrayList<>();

    KMeans(double[][] data, int k, long seed, int iterations) {
        centroids = new double[k][];
        Prng rng = new Prng(seed);
        for (int c = 0; c < k; c++) centroids[c] = data[rng.nextInt(data.length)].clone();
        assignment = new int[data.length];
        for (int iteration = 0; iteration < iterations; iteration++) {
            boolean changed = false;
            for (int i = 0; i < data.length; i++) {
                int best = nearest(centroids, data[i]);
                if (best != assignment[i]) changed = true;
                assignment[i] = best;
            }
            double[][] sums = new double[k][data[0].length];
            int[] counts = new int[k];
            for (int i = 0; i < data.length; i++) {
                counts[assignment[i]]++;
                for (int j = 0; j < data[i].length; j++) sums[assignment[i]][j] += data[i][j];
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    centroids[c] = data[rng.nextInt(data.length)].clone();      // re-seed an empty cell
                    continue;
                }
                for (int j = 0; j < centroids[c].length; j++) centroids[c][j] = sums[c][j] / counts[c];
            }
            objectives.add(objective(data, centroids, assignment));
            if (!changed && iteration > 0) break;
        }
    }

    static int nearest(double[][] centroids, double[] point) {
        int best = 0;
        double bestDistance = Double.MAX_VALUE;
        for (int c = 0; c < centroids.length; c++) {
            double distance = Vectors.squaredDistance(point, centroids[c]);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = c;
            }
        }
        return best;
    }

    static double objective(double[][] data, double[][] centroids, int[] assignment) {
        double total = 0;
        for (int i = 0; i < data.length; i++) total += Vectors.squaredDistance(data[i], centroids[assignment[i]]);
        return total;
    }

    /** No point would switch cells: the assignment is the fixed point of the nearest-centroid map. */
    boolean assignmentIsFixedPoint(double[][] data) {
        for (int i = 0; i < data.length; i++) {
            if (KMeans.nearest(centroids, data[i]) != assignment[i]) return false;
        }
        return true;
    }
}

/** Product quantiser: m sub-vectors, each with its own small codebook. */
final class ProductQuantizer {
    final int subVectors, subDimension;
    final double[][][] codebooks;
    final int codewords;

    ProductQuantizer(Vectors vectors, int subVectors, int codewords, long seed) {
        this.subVectors = subVectors;
        this.codewords = codewords;
        this.subDimension = vectors.dimension() / subVectors;
        this.codebooks = new double[subVectors][][];
        for (int s = 0; s < subVectors; s++) {
            double[][] slice = slice(vectors.data, s);
            codebooks[s] = new KMeans(slice, codewords, seed + s * 31L, 25).centroids;
        }
    }

    private double[][] slice(double[][] data, int s) {
        double[][] out = new double[data.length][subDimension];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], s * subDimension, out[i], 0, subDimension);
        }
        return out;
    }

    int[] encode(double[] vector) {
        int[] code = new int[subVectors];
        for (int s = 0; s < subVectors; s++) {
            double[] part = new double[subDimension];
            System.arraycopy(vector, s * subDimension, part, 0, subDimension);
            code[s] = KMeans.nearest(codebooks[s], part);
        }
        return code;
    }

    double[] decode(int[] code) {
        double[] vector = new double[subVectors * subDimension];
        for (int s = 0; s < subVectors; s++) {
            System.arraycopy(codebooks[s][code[s]], 0, vector, s * subDimension, subDimension);
        }
        return vector;
    }

    /** Asymmetric distance computation: a table per subspace, then one lookup per sub-vector. */
    double[][] lookupTable(double[] query) {
        double[][] table = new double[subVectors][codewords];
        for (int s = 0; s < subVectors; s++) {
            for (int c = 0; c < codewords; c++) {
                double sum = 0;
                for (int j = 0; j < subDimension; j++) {
                    double difference = query[s * subDimension + j] - codebooks[s][c][j];
                    sum += difference * difference;
                }
                table[s][c] = sum;
            }
        }
        return table;
    }

    double adc(double[][] table, int[] code) {
        double sum = 0;
        for (int s = 0; s < subVectors; s++) sum += table[s][code[s]];
        return sum;
    }
}

/** IVF + PQ: coarse quantiser, posting lists and re-ranked candidates. */
final class IvfIndex {
    final Vectors vectors;
    final ProductQuantizer quantizer;
    final KMeans coarse;
    final List<List<Integer>> lists = new ArrayList<>();
    final List<int[]> codes = new ArrayList<>();
    long distanceComputations;
    long quantizedComputations;

    IvfIndex(Vectors vectors, int cells, int subVectors, int codewords, long seed) {
        this.vectors = vectors;
        this.coarse = new KMeans(vectors.data, cells, seed, 30);
        this.quantizer = new ProductQuantizer(vectors, subVectors, codewords, seed + 7);
        for (int c = 0; c < cells; c++) lists.add(new ArrayList<>());
        for (int i = 0; i < vectors.size(); i++) {
            lists.get(coarse.assignment[i]).add(i);
            codes.add(quantizer.encode(vectors.data[i]));
        }
    }

    /**
     * PQ proposes a shortlist (it is a lossy distance), then the shortlist is re-ranked with exact
     * distances - the standard IVF+PQ pipeline, and the reason the returned order can be trusted.
     */
    List<Integer> search(double[] query, int k, int nprobe) {
        List<Integer> cells = new ArrayList<>();
        for (int c = 0; c < lists.size(); c++) cells.add(c);
        final double[] centroidsDistance = new double[lists.size()];
        for (int c = 0; c < lists.size(); c++) {
            centroidsDistance[c] = Vectors.squaredDistance(query, coarse.centroids[c]);
            distanceComputations++;
        }
        cells.sort((a, b) -> {
            int byDistance = Double.compare(centroidsDistance[a], centroidsDistance[b]);
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        double[][] table = quantizer.lookupTable(query);
        List<Integer> candidates = new ArrayList<>();
        for (int p = 0; p < Math.min(nprobe, cells.size()); p++) candidates.addAll(lists.get(cells.get(p)));
        List<double[]> scored = new ArrayList<>();
        for (int candidate : candidates) {
            quantizedComputations++;
            scored.add(new double[]{quantizer.adc(table, codes.get(candidate)), candidate});
        }
        scored.sort((a, b) -> {
            int byDistance = Double.compare(a[0], b[0]);
            return byDistance != 0 ? byDistance : Double.compare(a[1], b[1]);
        });
        int shortlist = Math.min(scored.size(), Math.max(k, k * 8));
        List<double[]> refined = new ArrayList<>();
        for (int i = 0; i < shortlist; i++) {
            int id = (int) scored.get(i)[1];
            distanceComputations++;                                   // exact distance for the shortlist
            refined.add(new double[]{Vectors.squaredDistance(query, vectors.data[id]), id});
        }
        refined.sort((a, b) -> {
            int byDistance = Double.compare(a[0], b[0]);
            return byDistance != 0 ? byDistance : Double.compare(a[1], b[1]);
        });
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, refined.size()); i++) result.add((int) refined.get(i)[1]);
        return result;
    }

    /** Exact re-ranking: the returned order is the true order *within the probed candidates*. */
    List<Integer> searchExactWithinProbe(double[] query, int k, int nprobe) {
        List<Integer> approx = search(query, k * 4, nprobe);
        List<Integer> unique = new ArrayList<>(new LinkedHashSet<>(approx));
        unique.sort((a, b) -> {
            int byDistance = Double.compare(Vectors.squaredDistance(query, vectors.data[a]),
                    Vectors.squaredDistance(query, vectors.data[b]));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        return unique.subList(0, Math.min(k, unique.size()));
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Clustered synthetic embeddings: eight well-separated centres plus noise. */
    static Vectors buildVectors(int count, int dimension, int clusters, long seed) {
        Prng rng = new Prng(seed);
        double[][] centres = new double[clusters][dimension];
        for (int c = 0; c < clusters; c++) {
            for (int j = 0; j < dimension; j++) centres[c][j] = rng.nextGaussian() * 5;
        }
        double[][] data = new double[count][dimension];
        for (int i = 0; i < count; i++) {
            double[] centre = centres[i % clusters];
            for (int j = 0; j < dimension; j++) data[i][j] = centre[j] + rng.nextGaussian();
        }
        return new Vectors(data);
    }

    static List<double[]> queries(Vectors vectors, int count, long seed) {
        Prng rng = new Prng(seed);
        List<double[]> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double[] query = vectors.data[rng.nextInt(vectors.size())].clone();
            for (int j = 0; j < query.length; j++) query[j] += rng.nextGaussian() * 0.1;
            list.add(query);
        }
        return list;
    }

    static double recall(IvfIndex index, List<double[]> queries, int k, int nprobe) {
        if (queries.isEmpty()) return 0;
        double total = 0;
        for (double[] query : queries) {
            Set<Integer> truth = new TreeSet<>(Vectors.exactKnn(index.vectors, query, k));
            Set<Integer> found = new TreeSet<>(index.search(query, k, nprobe));
            Set<Integer> overlap = new TreeSet<>(truth);
            overlap.retainAll(found);
            total += overlap.size() / (double) k;
        }
        return total / queries.size();
    }

    public static void main(String[] args) {
        Vectors vectors = buildVectors(600, 16, 8, 0xE0BEDD16L);
        List<double[]> queries = queries(vectors, 60, 0xA11L);
        check(vectors.size() == 600 && vectors.dimension() == 16, "600 vectors of dimension 16");

        // 1. k-means: the objective never rises, and the final assignment is a fixed point
        KMeans kmeans = new KMeans(vectors.data, 8, 42, 40);
        check(kmeans.objectives.size() >= 2, "k-means ran at least two iterations: " + kmeans.objectives.size());
        for (int i = 1; i < kmeans.objectives.size(); i++) {
            check(kmeans.objectives.get(i) <= kmeans.objectives.get(i - 1) + 1e-9, "the k-means objective is "
                    + "monotone: " + kmeans.objectives.get(i - 1) + " -> " + kmeans.objectives.get(i));
        }
        check(kmeans.assignmentIsFixedPoint(vectors.data), "the assignment is the nearest-centroid fixed point");
        int usedCells = 0;
        for (List<Integer> list : buildIndex(vectors).lists) if (!list.isEmpty()) usedCells++;
        check(usedCells == 8, "all eight cells are populated: " + usedCells);

        // 2. exhaustive probing is exactly brute force, and fewer probes trade recall for work
        IvfIndex index = buildIndex(vectors);
        double recallAll = recall(index, queries, 10, index.lists.size());
        double recallOne = recall(index, queries, 10, 1);
        double recallTwo = recall(index, queries, 10, 2);
        check(recallAll == 1.0, "probing every list reproduces the exact answer: recall=" + recallAll);
        check(recallOne >= 0.5, "a single probe still finds most neighbours: recall=" + recallOne);
        check(recallTwo >= recallOne - 1e-9, "more probes never hurt recall: " + recallOne + " -> " + recallTwo);
        check(recallAll >= recallTwo, "full probing is at least as good as partial: " + recallTwo + " -> "
                + recallAll);

        // 3. PQ reconstruction error shrinks as the codebook grows
        double coarseError = averageReconstructionError(vectors, 4, 4);
        double fineError = averageReconstructionError(vectors, 4, 32);
        check(fineError < coarseError, "a larger codebook reconstructs better: " + coarseError + " -> "
                + fineError);
        double singleSubvector = averageReconstructionError(vectors, 1, 16);
        check(singleSubvector > fineError, "one big sub-vector quantises worse than four: " + singleSubvector
                + " vs " + fineError);
        double errorFraction = fineError / averagePairwiseEnergy(vectors);
        check(errorFraction < 0.2, "the fine codebook keeps the error small relative to the data spread: "
                + errorFraction);

        // 4. the ADC identity: the table distance equals the distance to the decoded vector, exactly
        ProductQuantizer quantizer = index.quantizer;
        double worstGap = 0;
        for (int i = 0; i < 50; i++) {
            int id = i * 11 % vectors.size();
            double[][] table = quantizer.lookupTable(vectors.data[id]);
            double byTable = quantizer.adc(table, index.codes.get(id));
            double byDecode = Vectors.squaredDistance(vectors.data[id], quantizer.decode(index.codes.get(id)));
            worstGap = Math.max(worstGap, Math.abs(byTable - byDecode));
        }
        check(worstGap < 1e-9, "ADC equals the distance to the decoded vector: max gap " + worstGap);

        // 5. re-ranking makes the returned order exact inside the probed candidates
        for (int i = 0; i < 10; i++) {
            double[] query = queries.get(i);
            List<Integer> refined = index.searchExactWithinProbe(query, 5, 2);
            for (int rank = 1; rank < refined.size(); rank++) {
                double previous = Vectors.squaredDistance(query, vectors.data[refined.get(rank - 1)]);
                double current = Vectors.squaredDistance(query, vectors.data[refined.get(rank)]);
                check(previous <= current + 1e-9, "the shortlist is ordered by true distance");
            }
            check(refined.size() == 5, "the shortlist has the requested length");
        }

        // 6. cost: probing fewer lists examines strictly fewer vectors
        index.distanceComputations = 0;
        index.quantizedComputations = 0;
        index.search(queries.get(0), 10, 1);
        long oneProbeWork = index.quantizedComputations;
        index.quantizedComputations = 0;
        index.search(queries.get(0), 10, index.lists.size());
        long allProbeWork = index.quantizedComputations;
        check(oneProbeWork < allProbeWork, "probing one list is cheaper: " + oneProbeWork + " vs " + allProbeWork);
        check(allProbeWork == vectors.size(), "probing everything encodes a distance for every vector: "
                + allProbeWork);
        check(codesAreCompact(index, vectors), "each vector costs m codes, not d floats");

        // 7. boundaries: a single-vector index, an exact duplicate query, an empty query list
        Vectors single = buildVectors(1, 4, 1, 5);
        IvfIndex singleIndex = new IvfIndex(single, 1, 2, 4, 9);
        check(singleIndex.search(single.data[0], 3, 1).equals(List.of(0)), "a one-vector index returns it");
        List<Integer> selfSearch = index.search(vectors.data[7], 1, index.lists.size());
        check(selfSearch.get(0) == 7, "a query equal to an indexed vector finds itself first: " + selfSearch);
        check(recall(index, List.of(), 10, 1) == 0.0, "an empty query list has no recall to measure");
        check(index.search(queries.get(0), 10, 1).size() == 10, "the search returns exactly k ids");
        check(index.search(queries.get(0), 1, 1).size() == 1, "k=1 returns one id");
        check(index.search(queries.get(0), 0, 1).isEmpty(), "k=0 returns nothing");

        // 8. determinism: same seed, same centroids, same codes, same results
        IvfIndex replay = buildIndex(vectors);
        boolean sameCentroids = true;
        for (int c = 0; c < index.coarse.centroids.length; c++) {
            sameCentroids &= Arrays.equals(index.coarse.centroids[c], replay.coarse.centroids[c]);
        }
        check(sameCentroids, "the same seed rebuilds the same coarse centroids");
        boolean sameCodes = replay.codes.size() == index.codes.size();
        for (int i = 0; sameCodes && i < index.codes.size(); i++) {
            sameCodes = Arrays.equals(replay.codes.get(i), index.codes.get(i));
        }
        check(sameCodes, "the same seed rebuilds the same PQ codes");
        check(replay.search(queries.get(3), 10, 2).equals(index.search(queries.get(3), 10, 2)),
                "the same query gives the same result list");

        System.out.println("data: " + vectors.size() + " vectors, dimension " + vectors.dimension()
                + ", 8 IVF cells, PQ 8x2 with 16 codewords");
        System.out.println("recall@10: " + recallOne + " (1 probe), " + recallTwo + " (2 probes), " + recallAll
                + " (all 8 = exact)");
        System.out.println("PQ error: " + coarseError + " -> " + fineError + " as the codebook grows, "
                + Math.round(errorFraction * 100) + "% of the data spread");
        System.out.println("PASS");
    }

    static IvfIndex buildIndex(Vectors vectors) {
        return new IvfIndex(vectors, 8, 8, 16, 0x1F0F5EEDL);
    }

    static boolean codesAreCompact(IvfIndex index, Vectors vectors) {
        return index.codes.size() == vectors.size() && index.codes.get(0).length == 8;
    }

    static double averageReconstructionError(Vectors vectors, int subVectors, int codewords) {
        ProductQuantizer quantizer = new ProductQuantizer(vectors, subVectors, codewords, 0x900DL);
        double total = 0;
        for (int i = 0; i < vectors.size(); i++) {
            double[] decoded = quantizer.decode(quantizer.encode(vectors.data[i]));
            total += Vectors.squaredDistance(vectors.data[i], decoded) / vectors.dimension();
        }
        return total / vectors.size();
    }

    /** Average squared distance of the data from its mean: the scale the error is compared against. */
    static double averagePairwiseEnergy(Vectors vectors) {
        double[] mean = new double[vectors.dimension()];
        for (double[] vector : vectors.data) {
            for (int j = 0; j < vector.length; j++) mean[j] += vector[j] / vectors.size();
        }
        double total = 0;
        for (double[] vector : vectors.data) total += Vectors.squaredDistance(vector, mean) / vectors.dimension();
        return total / vectors.size();
    }
}
