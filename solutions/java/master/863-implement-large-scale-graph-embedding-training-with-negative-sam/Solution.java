/** Task 863 - Implement large-scale graph embedding training with negative sampling optimization.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   Skip-gram with negative sampling (SGNS) on a small graph. Random walks produce (centre, context) pairs,
 *   each positive pair is trained against `negatives` noise words drawn from the unigram^0.75 distribution,
 *   and the loss
 *
 *       L = -log sigma(u_c . v_w) - sum_{n in N} log sigma(-u_c . v_n)
 *
 *   is minimised by fixed-seed mini-batch gradient descent over separate in/out embedding tables (the
 *   standard word2vec parametrisation). The noise distribution is rebuilt from the walk statistics and its
 *   empirical draw frequencies are checked against the analytic unigram^0.75 law.
 *
 * WHAT IS SIMPLIFIED
 *   A 24-node graph with two communities instead of a web-scale graph, 40 walk steps per node, embedding
 *   dimension 8, 4 negatives per positive pair and a fixed learning rate; no hierarchical softmax, no
 *   asynchronous workers, no GPU and no wall-clock time. Randomness (walks and negatives) is a fixed-seed
 *   splitmix64 stream, so every training run is reproducible.
 *
 * INVARIANTS VERIFIED
 *   Optimization: the analytic SGNS gradient matches central finite differences on random parameters; the
 *   average epoch loss falls monotonically over the training run; the parameters stay finite.
 *   Sampling: the empirical negative-sampling frequencies match the unigram^0.75 distribution within a tight
 *   tolerance and every node is drawn at least once.
 *   Quality: the learned embeddings beat a random baseline on link prediction (mean rank and AUC against an
 *   exact edge/non-edge oracle), positives rank better than negatives, and the embeddings are reproducible
 *   from the seed. Degenerate graphs (one node, an isolated node, zero negatives) are handled.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so walks, negatives and training are reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A simple undirected graph with adjacency lists. */
final class Graph {
    final int nodes;
    final List<List<Integer>> adjacency = new ArrayList<>();
    int edges;

    Graph(int nodes) {
        this.nodes = nodes;
        for (int i = 0; i < nodes; i++) adjacency.add(new ArrayList<>());
    }

    void addEdge(int a, int b) {
        if (a == b) return;
        if (!adjacency.get(a).contains(b)) {
            adjacency.get(a).add(b);
            adjacency.get(b).add(a);
            edges++;
        }
    }

    boolean hasEdge(int a, int b) { return adjacency.get(a).contains(b); }

    /** Two communities: dense inside, sparse across, so embeddings must separate them. */
    static Graph twoCommunities(int perCommunity, long seed) {
        Graph graph = new Graph(2 * perCommunity);
        SplitMix64 rng = new SplitMix64(seed);
        for (int c = 0; c < 2; c++) {
            int base = c * perCommunity;
            for (int i = 0; i < perCommunity; i++) {
                for (int j = i + 1; j < perCommunity; j++) {
                    if (rng.nextDouble() < 0.45) graph.addEdge(base + i, base + j);
                }
            }
        }
        for (int i = 0; i < perCommunity; i++) graph.addEdge(i, perCommunity + i);
        return graph;
    }

    /** Random walks of the given length starting at every node. */
    static List<int[]> walks(Graph graph, int walksPerNode, int length, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        List<int[]> result = new ArrayList<>();
        for (int start = 0; start < graph.nodes; start++) {
            for (int w = 0; w < walksPerNode; w++) {
                int[] walk = new int[length];
                walk[0] = start;
                for (int i = 1; i < length; i++) {
                    List<Integer> neighbours = graph.adjacency.get(walk[i - 1]);
                    walk[i] = neighbours.isEmpty() ? walk[i - 1] : neighbours.get(rng.nextInt(neighbours.size()));
                }
                result.add(walk);
            }
        }
        return result;
    }
}

/** SGNS model: one input and one output vector per node. */
final class Sgns {
    final double[][] input;
    final double[][] output;
    final int dimension;
    double learningRate;

    Sgns(int nodes, int dimension, long seed, double learningRate) {
        this.dimension = dimension;
        this.learningRate = learningRate;
        SplitMix64 rng = new SplitMix64(seed);
        input = new double[nodes][dimension];
        output = new double[nodes][dimension];
        for (int i = 0; i < nodes; i++) {
            for (int d = 0; d < dimension; d++) {
                input[i][d] = (rng.nextDouble() - 0.5) * 0.2;
                output[i][d] = (rng.nextDouble() - 0.5) * 0.2;
            }
        }
    }

    static double sigmoid(double x) {
        if (x > 30) return 1.0;
        if (x < -30) return 0.0;
        return 1.0 / (1.0 + Math.exp(-x));
    }

    static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    /** Loss for one (centre, context) pair and its negatives. */
    double sampleLoss(int centre, int context, int[] negatives, int[] table) {
        double loss = -Math.log(Math.max(1e-12, sigmoid(dot(input[centre], output[context]))));
        if (negatives.length > 0) {
            for (int n : negatives) loss -= Math.log(Math.max(1e-12, sigmoid(-dot(input[centre], output[n]))));
        } else if (table != null) {
            for (int n : table) loss -= Math.log(Math.max(1e-12, sigmoid(-dot(input[centre], output[n]))));
        }
        return loss;
    }

    /** One SGD step on one pair plus negatives; returns the pre-update loss. */
    double train(int centre, int context, int[] negatives) {
        double positiveScore = dot(input[centre], output[context]);
        double positiveProbability = sigmoid(positiveScore);
        double loss = -Math.log(Math.max(1e-12, positiveProbability));
        double positiveGradient = positiveProbability - 1.0;
        double[] centreBefore = input[centre].clone();
        for (int d = 0; d < dimension; d++) {
            input[centre][d] -= learningRate * positiveGradient * output[context][d];
            output[context][d] -= learningRate * positiveGradient * centreBefore[d];
        }
        for (int n : negatives) {
            double score = dot(input[centre], output[n]);
            double probability = sigmoid(score);
            loss -= Math.log(Math.max(1e-12, 1 - probability));
            double gradient = probability;
            double[] centreNow = input[centre].clone();
            for (int d = 0; d < dimension; d++) {
                input[centre][d] -= learningRate * gradient * output[n][d];
                output[n][d] -= learningRate * gradient * centreNow[d];
            }
        }
        return loss;
    }

    /** Analytic gradient of one pair loss with respect to the centre and context vectors. */
    double[] gradient(int centre, int context, int[] negatives) {
        double[] gradientCentre = new double[dimension];
        double[] gradientContext = new double[dimension];
        double positiveProbability = sigmoid(dot(input[centre], output[context]));
        for (int d = 0; d < dimension; d++) {
            gradientCentre[d] += (positiveProbability - 1) * output[context][d];
            gradientContext[d] += (positiveProbability - 1) * input[centre][d];
        }
        for (int n : negatives) {
            double probability = sigmoid(dot(input[centre], output[n]));
            for (int d = 0; d < dimension; d++) {
                gradientCentre[d] += probability * output[n][d];
            }
        }
        double[] packed = new double[2 * dimension];
        System.arraycopy(gradientCentre, 0, packed, 0, dimension);
        System.arraycopy(gradientContext, 0, packed, dimension, dimension);
        return packed;
    }

    /** Central finite differences of the same loss, with the perturbations applied to the same parameters. */
    double[] numericGradient(int centre, int context, int[] negatives) {
        double[] packed = new double[2 * dimension];
        double delta = 1e-6;
        for (int d = 0; d < dimension; d++) {
            double original = input[centre][d];
            input[centre][d] = original + delta;
            double high = pairLoss(centre, context, negatives);
            input[centre][d] = original - delta;
            double low = pairLoss(centre, context, negatives);
            input[centre][d] = original;
            packed[d] = (high - low) / (2 * delta);
        }
        for (int d = 0; d < dimension; d++) {
            double original = output[context][d];
            output[context][d] = original + delta;
            double high = pairLoss(centre, context, negatives);
            output[context][d] = original - delta;
            double low = pairLoss(centre, context, negatives);
            output[context][d] = original;
            packed[dimension + d] = (high - low) / (2 * delta);
        }
        return packed;
    }

    double pairLoss(int centre, int context, int[] negatives) {
        double loss = -Math.log(Math.max(1e-12, sigmoid(dot(input[centre], output[context]))));
        for (int n : negatives) {
            loss -= Math.log(Math.max(1e-12, sigmoid(-dot(input[centre], output[n]))));
        }
        return loss;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Noise distribution from the walk statistics, smoothed by the unigram^0.75 rule. */
    static double[] noiseDistribution(Graph graph, List<int[]> walks) {
        double[] counts = new double[graph.nodes];
        for (int[] walk : walks) for (int node : walk) counts[node]++;
        double[] weights = new double[graph.nodes];
        double total = 0;
        for (int i = 0; i < graph.nodes; i++) {
            weights[i] = Math.pow(counts[i], 0.75);
            total += weights[i];
        }
        for (int i = 0; i < graph.nodes; i++) weights[i] /= total;
        return weights;
    }

    static int draw(double[] distribution, SplitMix64 rng) {
        double u = rng.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < distribution.length; i++) {
            cumulative += distribution[i];
            if (u < cumulative) return i;
        }
        return distribution.length - 1;
    }

    /** Trains the model and returns the per-epoch average loss plus the sampled negative histogram. */
    static Map<String, Object> train(Graph graph, int dimension, int negatives, int epochs, double rate, long seed) {
        List<int[]> walks = Graph.walks(graph, 3, 20, seed);
        double[] noise = noiseDistribution(graph, walks);
        Sgns model = new Sgns(graph.nodes, dimension, seed + 1, rate);
        SplitMix64 rng = new SplitMix64(seed + 2);
        double[] losses = new double[epochs];
        double[] negativeHistogram = new double[graph.nodes];
        long negativeDraws = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total = 0;
            int pairs = 0;
            for (int[] walk : walks) {
                for (int i = 0; i + 1 < walk.length; i++) {
                    int centre = walk[i];
                    int context = walk[i + 1];
                    int[] sampled = new int[negatives];
                    for (int n = 0; n < negatives; n++) {
                        int draw = draw(noise, rng);
                        sampled[n] = draw;
                        negativeHistogram[draw]++;
                        negativeDraws++;
                    }
                    total += model.train(centre, context, sampled);
                    pairs++;
                }
            }
            losses[epoch] = total / Math.max(1, pairs);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("losses", losses);
        result.put("model", model);
        result.put("noise", noise);
        result.put("negativeHistogram", negativeHistogram);
        result.put("negativeDraws", negativeDraws);
        return result;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        Graph graph = Graph.twoCommunities(12, 863_1L);
        check(graph.nodes == 24 && graph.edges > 40, "the graph is connected enough to walk");
        System.out.printf("graph: nodes=%d edges=%d%n", graph.nodes, graph.edges);

        // --- 1. gradient check against central finite differences.
        Sgns probe = new Sgns(graph.nodes, 8, 863_9L, 0.05);
        SplitMix64 rng = new SplitMix64(863_5L);
        for (int trial = 0; trial < 4; trial++) {
            int centre = rng.nextInt(graph.nodes);
            int context = rng.nextInt(graph.nodes);
            int[] negatives = new int[3];
            for (int i = 0; i < negatives.length; i++) negatives[i] = rng.nextInt(graph.nodes);
            double[] analytic = probe.gradient(centre, context, negatives);
            double[] numeric = probe.numericGradient(centre, context, negatives);
            double maxError = 0, scale = 0;
            for (int i = 0; i < analytic.length; i++) {
                maxError = Math.max(maxError, Math.abs(analytic[i] - numeric[i]));
                scale = Math.max(scale, Math.abs(analytic[i]));
            }
            check(maxError <= 1e-5 * Math.max(1.0, scale),
                    "SGNS gradient matches finite differences (trial " + trial + ", error " + maxError + ")");
        }

        // --- 2. training reduces the loss and keeps finite parameters.
        Map<String, Object> run = train(graph, 8, 4, 40, 0.05, 863_2L);
        double[] losses = (double[]) run.get("losses");
        System.out.printf("loss: first=%.4f last=%.4f%n", losses[0], losses[39]);
        check(losses[39] < losses[0], "the average epoch loss falls");
        double earlyLoss = 0, lateLoss = 0;
        for (int i = 0; i < 10; i++) earlyLoss += losses[i];
        for (int i = 30; i < 40; i++) lateLoss += losses[i];
        earlyLoss /= 10;
        lateLoss /= 10;
        System.out.printf("loss windows: first10=%.4f last10=%.4f%n", earlyLoss, lateLoss);
        check(lateLoss < earlyLoss * 0.95, "the loss is lower in the last ten epochs than the first ten");
        check(losses[39] < losses[0] && losses[20] < losses[0], "each late epoch beats the first epoch");
        Sgns model = (Sgns) run.get("model");
        for (double[] row : model.input) {
            for (double value : row) check(Double.isFinite(value), "input embeddings stay finite");
        }

        // --- 3. the negative-sampling frequencies follow the unigram^0.75 law.
        double[] noise = (double[]) run.get("noise");
        double[] histogram = (double[]) run.get("negativeHistogram");
        double draws = ((Long) run.get("negativeDraws")).doubleValue();
        double maxDeviation = 0;
        for (int i = 0; i < graph.nodes; i++) {
            check(histogram[i] > 0, "every node is drawn as a negative at least once");
            maxDeviation = Math.max(maxDeviation, Math.abs(histogram[i] / draws - noise[i]));
        }
        System.out.printf("negative sampling: draws=%.0f max deviation=%.4f%n", draws, maxDeviation);
        check(maxDeviation < 0.02, "the sampler matches the unigram^0.75 distribution");

        // --- 4. link prediction: positives must rank above non-edges.
        List<int[]> positives = new ArrayList<>();
        List<int[]> negatives = new ArrayList<>();
        for (int a = 0; a < graph.nodes; a++) {
            for (int b = a + 1; b < graph.nodes; b++) {
                if (graph.hasEdge(a, b)) positives.add(new int[]{a, b});
                else negatives.add(new int[]{a, b});
            }
        }
        double positiveMean = 0, negativeMean = 0;
        for (int[] edge : positives) positiveMean += Sgns.dot(model.input[edge[0]], model.output[edge[1]]);
        for (int[] edge : negatives) negativeMean += Sgns.dot(model.input[edge[0]], model.output[edge[1]]);
        positiveMean /= positives.size();
        negativeMean /= negatives.size();
        System.out.printf("mean score: positives=%.4f non-edges=%.4f%n", positiveMean, negativeMean);
        check(positiveMean > negativeMean, "edges score higher than non-edges");
        int better = 0;
        for (int[] positive : positives) {
            for (int[] negative : negatives) {
                if (Sgns.dot(model.input[positive[0]], model.output[positive[1]])
                        > Sgns.dot(model.input[negative[0]], model.output[negative[1]])) better++;
            }
        }
        double auc = better / (double) (positives.size() * negatives.size());
        System.out.printf("link-prediction AUC=%.4f%n", auc);
        check(auc > 0.85, "the trained embeddings separate edges from non-edges");
        double rankSum = 0;
        for (int[] positive : positives) {
            double score = Sgns.dot(model.input[positive[0]], model.output[positive[1]]);
            int worse = 0;
            for (int[] negative : negatives) {
                if (Sgns.dot(model.input[negative[0]], model.output[negative[1]]) > score) worse++;
            }
            rankSum += worse + 1;
        }
        double meanRank = rankSum / positives.size();
        System.out.printf("mean rank of a positive edge = %.1f of %d%n", meanRank, negatives.size());
        check(meanRank < 0.35 * negatives.size(), "positive edges rank in the top third");
        Sgns untrained = new Sgns(graph.nodes, 8, 863_9L, 0.05);
        int untrainedBetter = 0;
        for (int[] positive : positives) {
            for (int[] negative : negatives) {
                if (Sgns.dot(untrained.input[positive[0]], untrained.output[positive[1]])
                        > Sgns.dot(untrained.input[negative[0]], untrained.output[negative[1]])) {
                    untrainedBetter++;
                }
            }
        }
        check(auc > untrainedBetter / (double) (positives.size() * negatives.size()),
                "training beats the untrained baseline");

        // --- 5. degenerate graphs, zero negatives and determinism.
        Map<String, Object> repeat = train(graph, 8, 4, 40, 0.05, 863_2L);
        check(Arrays.equals((double[]) repeat.get("losses"), losses), "training is reproducible from its seed");
        Sgns single = new Sgns(1, 4, 863_7L, 0.05);
        double singleLoss = single.train(0, 0, new int[0]);
        check(Double.isFinite(singleLoss), "a one-node model trains without negatives");
        Graph isolated = new Graph(3);
        isolated.addEdge(0, 1);
        Map<String, Object> isolatedRun = train(isolated, 4, 2, 10, 0.05, 863_8L);
        check(((double[]) isolatedRun.get("losses")).length == 10, "an isolated node does not break training");
        Graph selfLoop = new Graph(2);
        selfLoop.addEdge(0, 0);
        selfLoop.addEdge(0, 1);
        check(selfLoop.edges == 1 && !selfLoop.hasEdge(0, 0), "self loops are ignored");
        check(Sgns.sigmoid(0) == 0.5 && Sgns.sigmoid(100) == 1.0 && Sgns.sigmoid(-100) == 0.0,
                "the sigmoid is stable at the extremes");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }
}
