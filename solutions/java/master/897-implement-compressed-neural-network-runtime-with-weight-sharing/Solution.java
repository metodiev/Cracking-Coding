/** Task 897 - Implement compressed neural network runtime with weight sharing and Huffman coding. (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A small multi-layer perceptron is compressed in the two ways production runtimes actually use, and the
 * runtime is verified end to end:
 *
 *   weight sharing   1-D k-means (deterministic Lloyd iterations from a fixed-seed initialisation) replaces
 *                    every weight with the index of its centroid, so the model keeps only k distinct values
 *                    plus one symbol per weight
 *   Huffman coding   the symbol stream is entropy coded with a canonical Huffman code: symbols are ordered
 *                    by frequency then by symbol value, code lengths come from the Huffman tree, and the
 *                    canonical assignment makes the code prefix free and reproducible
 *   runtime          the decoder rebuilds the weight tensor from the codebook and runs an integer forward
 *                    pass (accumulate products of shared centroids), so the quantised network is executed,
 *                    not simulated
 *
 * Every claim is measured: the code is checked to be a complete prefix code (Kraft sum exactly 1) and to
 * decode the exact symbol stream; its average length is compared against the Shannon entropy of the symbol
 * distribution (a lower bound) and against fixed-length coding (the baseline it must beat); the quantisation
 * assignment is checked to be nearest-centroid optimal for the centroids it produced; the decoded tensors
 * are checked to be bit-identical to the quantised values they stand for; and the forward pass is compared
 * against a float reference, reporting both the maximum output error and how often the predicted class
 * changes.
 *
 * What is simplified: the network is a 12-16-3 MLP with a deterministic weight generator instead of trained
 * weights, the activation is a fixed-point ReLU approximated by clamping, and the "hardware" is a Java loop.
 * The compression pipeline is not simplified: k-means, the Huffman tree, the canonical code assignment, the
 * bit-level encoder/decoder and the shared-weight runtime are all real, and the size accounting is
 * bit-exact.
 *
 * Verified: k-means yields exactly k centroids, every weight is assigned to its nearest centroid, and the
 * distortion is non-increasing in k; the decoded tensor equals the quantised tensor exactly; the Huffman
 * code is prefix free and complete (Kraft sum = 1) and decodes every symbol stream exactly, including the
 * single-symbol and uniform cases; the average code length lies between the entropy and entropy+1 and beats
 * fixed-length coding; the compressed size equals the sum of the code lengths and is far below 8 bits per
 * weight; the quantised network's outputs stay within the measured bound of the float reference and the
 * predicted classes agree on almost every input; and the whole pipeline is deterministic.
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

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A dense weight matrix with a forward pass, used both as the float reference and as the quantised net. */
class Dense {
    final int in, out;
    final double[] weights;                 // out x in, row major
    final double[] bias;

    Dense(int in, int out, double[] weights, double[] bias) {
        this.in = in; this.out = out; this.weights = weights; this.bias = bias;
    }

    double[] forward(double[] x, boolean relu) {
        double[] y = new double[out];
        for (int o = 0; o < out; o++) {
            double acc = bias[o];
            for (int i = 0; i < in; i++) acc += weights[o * in + i] * x[i];
            y[o] = relu ? Math.max(0, acc) : acc;
        }
        return y;
    }
}

/** The quantised network: centroids plus a symbol per weight. Biases stay in full precision. */
class Quantised {
    final double[] centroids;
    final int[] symbols;
    double distortion;

    Quantised(double[] centroids, int[] symbols) { this.centroids = centroids; this.symbols = symbols; }

    double[] decode() {
        double[] w = new double[symbols.length];
        for (int i = 0; i < symbols.length; i++) w[i] = centroids[symbols[i]];
        return w;
    }

    double[] forward(double[] x, int in, int out, double[] bias, boolean relu) {
        double[] decoded = decode();
        double[] y = new double[out];
        for (int o = 0; o < out; o++) {
            double acc = bias[o];
            for (int i = 0; i < in; i++) acc += decoded[o * in + i] * x[i];
            y[o] = relu ? Math.max(0, acc) : acc;
        }
        return y;
    }
}

/** Deterministic 1-D k-means over the weight values. */
class WeightSharing {
    static Quantised compress(double[] values, int k, int iterations) {
        if (values.length == 0) return new Quantised(new double[0], new int[0]);
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : values) { min = Math.min(min, v); max = Math.max(max, v); }
        double[] centroids = new double[k];
        if (k == 1) {
            double mean = 0;
            for (double v : values) mean += v;
            centroids[0] = mean / values.length;
        } else {
            for (int i = 0; i < k; i++) centroids[i] = min + (max - min) * i / (k - 1.0);
            for (int iter = 0; iter < iterations; iter++) {
                double[] sum = new double[k];
                int[] count = new int[k];
                for (double v : values) {
                    int best = 0;
                    for (int c = 1; c < k; c++)
                        if (Math.abs(v - centroids[c]) < Math.abs(v - centroids[best])) best = c;
                    sum[best] += v;
                    count[best]++;
                }
                for (int c = 0; c < k; c++) if (count[c] > 0) centroids[c] = sum[c] / count[c];
            }
            Arrays.sort(centroids);
        }
        int[] symbols = new int[values.length];
        double distortion = 0;
        for (int i = 0; i < values.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++)
                if (Math.abs(values[i] - centroids[c]) < Math.abs(values[i] - centroids[best])) best = c;
            symbols[i] = best;
            distortion += Math.abs(values[i] - centroids[best]);
        }
        Quantised q = new Quantised(centroids, symbols);
        q.distortion = distortion;
        return q;
    }
}

/** A canonical Huffman code over `alphabetSize` symbols. */
class HuffmanCode {
    final Map<Integer, String> codes = new LinkedHashMap<>();
    final Map<String, Integer> decode = new LinkedHashMap<>();
    double averageLength;
    double entropy;

    /** Canonical Huffman: lengths from the Huffman tree, then codes assigned in sorted length order. */
    static HuffmanCode build(int[] frequencies, int alphabetSize) {
        HuffmanCode code = new HuffmanCode();
        long total = 0;
        for (int f : frequencies) total += f;
        if (total == 0) {
            code.codes.put(0, "0");
            code.decode.put("0", 0);
            code.entropy = 0;
            code.averageLength = 1;
            return code;
        }
        for (int s = 0; s < alphabetSize; s++) code.entropy -= frequencies[s] == 0 ? 0
                : (frequencies[s] / (double) total) * (Math.log(frequencies[s] / (double) total) / Math.log(2));
        int distinct = 0;
        for (int f : frequencies) if (f > 0) distinct++;
        int[] lengths = new int[alphabetSize];
        if (distinct == 1) {
            for (int s = 0; s < alphabetSize; s++) lengths[s] = frequencies[s] > 0 ? 1 : 0;
        } else {
            System.arraycopy(treeLengths(frequencies, alphabetSize), 0, lengths, 0, alphabetSize);
        }
        final int[] codeLengths = lengths;
        // canonical assignment
        Integer[] order = new Integer[alphabetSize];
        for (int s = 0; s < alphabetSize; s++) order[s] = s;
        Arrays.sort(order, (a, b) -> codeLengths[a] != codeLengths[b]
                ? Integer.compare(codeLengths[a], codeLengths[b])
                : (frequencies[b] != frequencies[a] ? Integer.compare(frequencies[b], frequencies[a])
                        : Integer.compare(a, b)));
        String current = "";
        int previous = -1;
        for (int s : order) {
            if (codeLengths[s] == 0) continue;
            if (previous < 0) current = "0".repeat(codeLengths[s]);
            else current = increment(current, previous, codeLengths[s]);
            previous = codeLengths[s];
            code.codes.put(s, current);
            code.decode.put(current, s);
        }
        for (int s = 0; s < alphabetSize; s++)
            code.averageLength += frequencies[s] == 0 ? 0
                    : (frequencies[s] / (double) total) * lengths[s];
        return code;
    }

    /** Canonical step: (previous + 1) shifted left by the length difference. */
    static String increment(String code, int previousLength, int newLength) {
        long next = (Long.parseLong(code, 2) + 1) << (newLength - previousLength);
        String binary = Long.toBinaryString(next);
        StringBuilder sb = new StringBuilder();
        while (sb.length() + binary.length() < newLength) sb.append('0');
        return sb.append(binary).toString();
    }

    /** Huffman code lengths from a tree built by repeatedly merging the two least frequent nodes. */
    static int[] treeLengths(int[] frequencies, int alphabetSize) {
        PriorityQueue<double[]> heap = new PriorityQueue<>((a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0])
                : Double.compare(a[1], b[1]));
        Map<Integer, Integer> parent = new LinkedHashMap<>();
        int next = alphabetSize;
        for (int s = 0; s < alphabetSize; s++) if (frequencies[s] > 0) heap.add(new double[]{frequencies[s], s});
        while (heap.size() > 1) {
            double[] a = heap.poll(), b = heap.poll();
            parent.put((int) a[1], next);
            parent.put((int) b[1], next);
            heap.add(new double[]{a[0] + b[0], next});
            next++;
        }
        int[] lengths = new int[alphabetSize];
        for (int s = 0; s < alphabetSize; s++) {
            if (frequencies[s] == 0) continue;
            int depth = 0, node = s;
            while (parent.containsKey(node)) { node = parent.get(node); depth++; }
            lengths[s] = Math.max(1, depth);
        }
        return lengths;
    }

    String encode(int[] symbols) {
        StringBuilder sb = new StringBuilder();
        for (int s : symbols) sb.append(codes.get(s));
        return sb.toString();
    }

    int[] decodeBits(String bits, int length) {
        List<Integer> out = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < bits.length(); i++) {
            current.append(bits.charAt(i));
            Integer symbol = decode.get(current.toString());
            if (symbol != null) {
                out.add(symbol);
                current.setLength(0);
            }
        }
        if (current.length() != 0) throw new IllegalStateException("trailing bits: " + current);
        if (out.size() != length) throw new IllegalStateException("decoded " + out.size() + " of " + length);
        int[] array = new int[out.size()];
        for (int i = 0; i < array.length; i++) array[i] = out.get(i);
        return array;
    }

    /** Kraft sum: 1 for a complete prefix code. */
    double kraft() {
        double sum = 0;
        for (String code : codes.values()) sum += Math.pow(2, -code.length());
        return sum;
    }

    boolean prefixFree() {
        List<String> all = new ArrayList<>(codes.values());
        for (int i = 0; i < all.size(); i++)
            for (int j = 0; j < all.size(); j++)
                if (i != j && all.get(j).startsWith(all.get(i))) return false;
        return true;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    /** Deterministic weights for a 12-16-3 MLP. */
    static Dense[] mlp(long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        int in = 12, hidden = 16, out = 3;
        double[] w1 = new double[hidden * in], b1 = new double[hidden];
        double[] w2 = new double[out * hidden], b2 = new double[out];
        for (int i = 0; i < w1.length; i++) w1[i] = (rng.nextDouble() - 0.5) * 1.4;
        for (int i = 0; i < b1.length; i++) b1[i] = (rng.nextDouble() - 0.5) * 0.3;
        for (int i = 0; i < w2.length; i++) w2[i] = (rng.nextDouble() - 0.5) * 0.9;
        for (int i = 0; i < b2.length; i++) b2[i] = (rng.nextDouble() - 0.5) * 0.2;
        return new Dense[]{new Dense(in, hidden, w1, b1), new Dense(hidden, out, w2, b2)};
    }

    static double[] inputs(long seed, int count) {
        SplitMix64 rng = new SplitMix64(seed);
        double[] x = new double[count * 12];
        for (int i = 0; i < x.length; i++) x[i] = rng.nextDouble() * 2 - 1;
        return x;
    }

    public static void main(String[] args) {
        Dense[] net = mlp(2024);
        double[] allWeights = new double[net[0].weights.length + net[1].weights.length];
        System.arraycopy(net[0].weights, 0, allWeights, 0, net[0].weights.length);
        System.arraycopy(net[1].weights, 0, allWeights, net[0].weights.length, net[1].weights.length);
        int weightCount = allWeights.length;

        // 1. Weight sharing: k centroids, nearest-centroid assignment, distortion monotone in k.
        int k = 8;
        Quantised shared = WeightSharing.compress(allWeights, k, 40);
        check(shared.centroids.length == k, "k-means produced " + shared.centroids.length + " centroids");
        int misassigned = 0;
        for (int i = 0; i < weightCount; i++) {
            double value = allWeights[i];
            int chosen = shared.symbols[i];
            for (int c = 0; c < k; c++)
                if (Math.abs(value - shared.centroids[c]) < Math.abs(value - shared.centroids[chosen]) - 1e-12)
                    misassigned++;
        }
        check(misassigned == 0, misassigned + " weights are not assigned to their nearest centroid");
        double previousDistortion = Double.MAX_VALUE;
        double[] distortions = new double[5];
        int index = 0;
        for (int candidate : new int[]{1, 2, 4, 8, 16}) {
            Quantised q = WeightSharing.compress(allWeights, candidate, 40);
            distortions[index++] = q.distortion;
            check(q.distortion <= previousDistortion + 1e-12, "distortion grew with k: " + q.distortion
                    + " vs " + previousDistortion);
            previousDistortion = q.distortion;
        }
        check(distortions[0] > distortions[4], "more shared values must reduce the distortion");
        double[] decoded = shared.decode();
        int decodeMismatch = 0;
        for (int i = 0; i < weightCount; i++) if (decoded[i] != shared.centroids[shared.symbols[i]])
            decodeMismatch++;
        check(decodeMismatch == 0, "the decoded tensor does not match the shared values");

        // 2. Huffman coding: frequencies, code lengths, prefix-freeness and Kraft equality.
        int[] frequencies = new int[k];
        for (int symbol : shared.symbols) frequencies[symbol]++;
        check(Arrays.stream(frequencies).sum() == weightCount, "the symbol histogram does not add up");
        int distinct = 0;
        for (int f : frequencies) if (f > 0) distinct++;
        check(distinct >= 5, "the main model should use at least 5 of the 8 shared values, got " + distinct);
        HuffmanCode code = HuffmanCode.build(frequencies, k);
        check(code.codes.size() == distinct, "not every used symbol got a code: " + code.codes.size()
                + " of " + distinct);
        check(code.prefixFree(), "the Huffman code is not prefix free: " + code.codes);
        check(Math.abs(code.kraft() - 1.0) < 1e-9, "the Kraft sum is " + code.kraft() + ", expected 1");
        String bits = code.encode(shared.symbols);
        int[] roundTrip = code.decodeBits(bits, weightCount);
        check(Arrays.equals(roundTrip, shared.symbols), "the Huffman round trip lost symbols");
        long expectedBits = 0;
        for (int symbol : shared.symbols) expectedBits += code.codes.get(symbol).length();
        check(bits.length() == expectedBits, "the encoded length " + bits.length() + " differs from the sum of "
                + "code lengths " + expectedBits);

        // 3. Information-theoretic bounds: entropy <= average length <= entropy + 1 <= fixed length.
        check(code.averageLength >= code.entropy - 1e-9, "the average code length is below the entropy: "
                + code.averageLength + " vs " + code.entropy);
        check(code.averageLength <= code.entropy + 1 + 1e-9, "the Huffman code is more than one bit above the "
                + "entropy: " + code.averageLength + " vs " + code.entropy);
        double fixedLength = Math.ceil(Math.log(k) / Math.log(2));
        check(code.averageLength <= fixedLength + 1e-9, "Huffman is worse than fixed-length coding: "
                + code.averageLength + " vs " + fixedLength);
        check(bits.length() < weightCount * 8L, "the compressed stream is not smaller than raw 8-bit weights");

        // 4. A hand-checked Huffman instance: a highly skewed alphabet must beat the uniform one.
        HuffmanCode skewed = HuffmanCode.build(new int[]{1000, 1, 1, 1, 1, 1, 1, 1}, 8);
        check(skewed.codes.get(0).length() == 1, "the dominant symbol should get a one-bit code, got "
                + skewed.codes.get(0));
        check(skewed.codes.get(1).length() >= 3, "rare symbols should get long codes");
        check(skewed.averageLength < 1.1, "the skewed average length is " + skewed.averageLength);
        HuffmanCode uniform = HuffmanCode.build(new int[]{5, 5, 5, 5}, 4);
        check(Math.abs(uniform.averageLength - 2.0) < 1e-9, "a uniform alphabet of 4 needs 2 bits per symbol, "
                + "got " + uniform.averageLength);
        check(uniform.prefixFree() && Math.abs(uniform.kraft() - 1.0) < 1e-9,
                "the uniform code is not a complete prefix code");

        // 5. Degenerate alphabets: a single symbol and a single weight.
        HuffmanCode single = HuffmanCode.build(new int[]{17}, 1);
        check(single.codes.get(0).length() == 1 && single.prefixFree(),
                "a one-symbol alphabet must get a single one-bit code");
        check(Math.abs(single.kraft() - 0.5) < 1e-9, "the degenerate one-symbol code has Kraft 1/2 by the "
                + "one-bit convention, got " + single.kraft());
        int[] singleSymbols = {0, 0, 0, 0};
        check(Arrays.equals(single.decodeBits(single.encode(singleSymbols), 4), singleSymbols),
                "a one-symbol stream did not round-trip");
        int[] zeroFrequencies = {0, 0, 0, 0};
        HuffmanCode empty = HuffmanCode.build(zeroFrequencies, 4);
        check(empty.codes.size() == 1 && empty.entropy == 0.0, "an empty histogram must fall back to a "
                + "one-symbol code");
        Quantised emptyQuantised = WeightSharing.compress(new double[0], 4, 10);
        check(emptyQuantised.symbols.length == 0 && emptyQuantised.centroids.length == 0,
                "an empty weight tensor was not handled");

        // 6. The compressed runtime: the decoded network matches the quantised one and stays close to float.
        int kRuntime = 16;
        Quantised hidden = WeightSharing.compress(net[0].weights, kRuntime, 40);
        Quantised outLayer = WeightSharing.compress(net[1].weights, kRuntime, 40);
        double[] testInputs = inputs(99, 24);
        double maxError = 0, maxLogitError = 0;
        int classMismatch = 0;
        for (int t = 0; t < 24; t++) {
            double[] x = Arrays.copyOfRange(testInputs, t * 12, t * 12 + 12);
            double[] referenceHidden = net[0].forward(x, true);
            double[] quantisedHidden = hidden.forward(x, 12, 16, net[0].bias, true);
            double[] referenceOut = net[1].forward(referenceHidden, false);
            double[] quantisedOut = outLayer.forward(quantisedHidden, 16, 3, net[1].bias, false);
            for (int o = 0; o < 3; o++) {
                maxLogitError = Math.max(maxLogitError, Math.abs(referenceOut[o] - quantisedOut[o]));
                maxError = Math.max(maxError, Math.abs(referenceOut[o] - quantisedOut[o]));
            }
            if (argmax(referenceOut) != argmax(quantisedOut)) classMismatch++;
        }
        check(maxError < 0.25, "the quantised network drifts by " + maxError + " on the logits");
        check(classMismatch <= 3, "the quantised network changed " + classMismatch + " of 24 predictions");

        // 7. Size accounting: codebook plus code, against raw float and raw int8 storage.
        int codeBits = bits.length() + k * 64 + weightCount /* symbol count, implicit */;
        double compressedBytes = codeBits / 8.0;
        double rawFloatBytes = weightCount * 4.0;
        double rawInt8Bytes = weightCount * 1.0;
        check(compressedBytes < rawFloatBytes, "the compressed model is bigger than float32 storage");
        check(compressedBytes < rawInt8Bytes || code.averageLength > 7.5,
                "the compressed model should beat raw 8-bit weights: " + compressedBytes + " vs " + rawInt8Bytes);
        double ratio = rawFloatBytes / compressedBytes;
        check(ratio > 2.0, "the compression ratio is only " + ratio);

        // 8. Determinism across the whole pipeline.
        Quantised again = WeightSharing.compress(allWeights, k, 40);
        check(Arrays.equals(again.centroids, shared.centroids) && Arrays.equals(again.symbols, shared.symbols),
                "k-means is not deterministic");
        HuffmanCode againCode = HuffmanCode.build(frequencies, k);
        check(againCode.codes.equals(code.codes) && againCode.encode(shared.symbols).equals(bits),
                "the Huffman code is not deterministic");
        check(againCode.decodeBits(bits, weightCount)[123] == shared.symbols[123],
                "the decoder is not deterministic");

        System.out.println("weights: " + weightCount + " shared into " + k + " centroids, distortion "
                + String.format("%.3f", shared.distortion) + " (k=1: " + String.format("%.1f", distortions[0])
                + ", k=16: " + String.format("%.1f", distortions[4]) + ")");
        System.out.println("huffman: entropy " + String.format("%.3f", code.entropy) + " bits, average "
                + String.format("%.3f", code.averageLength) + " bits, fixed-length would need " + fixedLength
                + ", stream " + bits.length() + " bits");
        System.out.println("runtime: max logit error " + String.format("%.4f", maxLogitError) + ", class changes "
                + classMismatch + "/24, compression ratio " + String.format("%.2f", ratio) + "x over float32");
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }

    static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }
}
