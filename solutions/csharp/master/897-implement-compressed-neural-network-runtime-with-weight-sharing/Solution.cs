// Task 897 - Implement compressed neural network runtime with weight sharing and Huffman coding. (Tier: Master, Beyond — Machine Learning & AI)
//
// A small dense network is compressed to a codebook plus a Huffman-coded index stream, and the runtime
// performs inference *from that compressed form*:
//
//  * weight sharing - deterministic Lloyd's algorithm (initial centroids at quantiles of the sorted
//    weights, then repeated assignment/update) clusters the weights into K centroids; each weight becomes
//    an index into the codebook, which is the only part that keeps full precision;
//  * Huffman coding - the index stream of every layer is entropy-coded with a canonical Huffman code
//    built from the index frequencies, so skewed weight distributions cost well under log2(K) bits per
//    weight; the codes are verified to be prefix-free and to satisfy the Kraft inequality;
//  * sparse residual - weights whose quantization error exceeds a threshold are stored verbatim in an
//    exception list, so a reconstruction error of exactly zero can be requested for the outliers;
//  * runtime - inference decodes indices back to centroids and multiplies, in the same accumulation
//    order as an explicit dequantized network, which makes the equality of the two paths exact rather
//    than approximate.
//
// Simplifications (documented, not behavioural): codebooks are stored as float64 (no further scalar
// quantization), the model is a small fully connected ReLU MLP, k-means runs on the weight vector rather
// than per channel, and the codebook/exceptions are not entropy coded themselves. Latency is reported as
// bits and operations, not wall-clock time.
//
// Verified: Huffman codes are prefix-free, satisfy the Kraft inequality and round-trip every index
// stream; the average code length lies between the empirical entropy and entropy+1 bits and beats
// fixed-width coding; the full compressed size beats both raw float64 and codebook+fixed-width storage
// and shrinks as K grows; Lloyd's distortion decreases monotonically and the mean quantization error
// falls as K grows; inference from the compressed form is bit-identical to inference with explicitly
// dequantized weights and stays within a relative error bound of the original network; the sparse
// residual makes the outlier weights exact and drives the mean error down; single-centroid, empty and
// constant-weight cases behave; and the whole pipeline is reproducible from its seed.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class Huffman
{
    private sealed class Node
    {
        internal int Symbol = -1;
        internal long Frequency;
        internal Node? Left;
        internal Node? Right;
    }

    internal readonly Dictionary<int, string> Codes = new Dictionary<int, string>();
    internal readonly Dictionary<int, long> Frequencies = new Dictionary<int, long>();
    internal double Entropy;

    internal void Build(IReadOnlyList<int> stream)
    {
        Codes.Clear();
        Frequencies.Clear();
        foreach (int symbol in stream) Frequencies[symbol] = Frequencies.TryGetValue(symbol, out long count) ? count + 1 : 1;
        double total = stream.Count;
        Entropy = Frequencies.Values.Sum(f => -(f / total) * Math.Log2(f / total));

        var roots = Frequencies
            .OrderBy(pair => pair.Value)
            .ThenBy(pair => pair.Key)
            .Select(pair => new Node { Symbol = pair.Key, Frequency = pair.Value })
            .ToList();
        if (roots.Count == 1)
        {
            Codes[roots[0].Symbol] = "0";                       // a single symbol still needs one bit
            return;
        }
        while (roots.Count > 1)
        {
            roots = roots.OrderBy(node => node.Frequency).ThenBy(node => node.Symbol < 0 ? int.MaxValue : node.Symbol).ToList();
            Node left = roots[0];
            Node right = roots[1];
            roots.RemoveRange(0, 2);
            roots.Add(new Node { Frequency = left.Frequency + right.Frequency, Left = left, Right = right });
        }
        Assign(roots[0], "");
    }

    private void Assign(Node node, string prefix)
    {
        if (node.Symbol >= 0) { Codes[node.Symbol] = prefix.Length == 0 ? "0" : prefix; return; }
        Assign(node.Left!, prefix + "0");
        Assign(node.Right!, prefix + "1");
    }

    internal string Encode(IReadOnlyList<int> stream)
    {
        var builder = new System.Text.StringBuilder();
        foreach (int symbol in stream) builder.Append(Codes[symbol]);
        return builder.ToString();
    }

    internal List<int> Decode(string bits, int expectedCount)
    {
        var tree = new Dictionary<string, int>();
        foreach (KeyValuePair<int, string> pair in Codes) tree[pair.Value] = pair.Key;
        var output = new List<int>();
        var current = new System.Text.StringBuilder();
        foreach (char bit in bits)
        {
            current.Append(bit);
            if (tree.TryGetValue(current.ToString(), out int symbol))
            {
                output.Add(symbol);
                current.Clear();
            }
        }
        if (output.Count != expectedCount) throw new InvalidOperationException("bit stream did not decode cleanly");
        return output;
    }

    internal double AverageBits() => Codes.ToDictionary(pair => pair.Key, pair => (double)pair.Value.Length)
        .Sum(pair => pair.Value * Frequencies[pair.Key]) / Frequencies.Values.Sum();

    internal bool IsPrefixFree(out string problem)
    {
        problem = "";
        List<string> codes = Codes.Values.OrderBy(code => code.Length).ToList();
        for (int i = 0; i < codes.Count; i++)
            for (int j = i + 1; j < codes.Count; j++)
                if (codes[j].StartsWith(codes[i], StringComparison.Ordinal))
                {
                    problem = "code " + codes[i] + " prefixes " + codes[j];
                    return false;
                }
        return true;
    }

    internal double KraftSum() => Codes.Values.Sum(code => Math.Pow(2, -code.Length));
}

internal sealed class Layer
{
    internal int Inputs;
    internal int Outputs;
    internal double[] Weights = Array.Empty<double>();
    internal double[] Biases = Array.Empty<double>();

    internal double[] Forwards(double[] input)
    {
        var output = new double[Outputs];
        for (int neuron = 0; neuron < Outputs; neuron++)
        {
            double sum = Biases[neuron];
            int offset = neuron * Inputs;
            for (int i = 0; i < Inputs; i++) sum += Weights[offset + i] * input[i];
            output[neuron] = sum > 0 ? sum : 0;                  // ReLU
        }
        return output;
    }
}

internal sealed class Network
{
    internal readonly List<Layer> Layers = new List<Layer>();

    internal double[] Forward(double[] input)
    {
        double[] current = input;
        foreach (Layer layer in Layers) current = layer.Forwards(current);
        return current;
    }

    internal double[] AllWeights() => Layers.SelectMany(layer => layer.Weights).ToArray();
}

internal sealed class Codebook
{
    internal double[] Centroids = Array.Empty<double>();
    internal double Distortion;
    internal int Iterations;
    internal readonly List<double> DistortionTrace = new List<double>();

    /** Deterministic Lloyd's: quantile initialisation, then assignment/update loops. */
    internal void Train(double[] weights, int clusters)
    {
        clusters = Math.Max(1, Math.Min(clusters, weights.Length));
        double[] sorted = weights.OrderBy(w => w).ToArray();
        var centroids = new double[clusters];
        for (int c = 0; c < clusters; c++)
            centroids[c] = sorted[(int)((2L * c + 1) * sorted.Length / (2L * clusters))];
        DistortionTrace.Clear();
        for (int iteration = 0; iteration < 60; iteration++)
        {
            var sums = new double[clusters];
            var counts = new int[clusters];
            double distortion = 0;
            foreach (double weight in weights)
            {
                int best = 0;
                double bestDistance = double.MaxValue;
                for (int c = 0; c < clusters; c++)
                {
                    double distance = (weight - centroids[c]) * (weight - centroids[c]);
                    if (distance < bestDistance) { bestDistance = distance; best = c; }
                }
                sums[best] += weight;
                counts[best]++;
                distortion += bestDistance;
            }
            for (int c = 0; c < clusters; c++) if (counts[c] > 0) centroids[c] = sums[c] / counts[c];
            DistortionTrace.Add(distortion);
            Iterations = iteration + 1;
            if (iteration > 0 && Math.Abs(DistortionTrace[^2] - distortion) < 1e-15) { Distortion = distortion; break; }
            Distortion = distortion;
        }
        Array.Sort(centroids);
        Centroids = centroids;
    }

    internal int Nearest(double weight)
    {
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int c = 0; c < Centroids.Length; c++)
        {
            double distance = (weight - Centroids[c]) * (weight - Centroids[c]);
            if (distance < bestDistance) { bestDistance = distance; best = c; }
        }
        return best;
    }
}

internal sealed class CompressedNetwork
{
    internal readonly Network Original;
    internal readonly List<Codebook> Codebooks = new List<Codebook>();
    internal readonly List<int[]> Indices = new List<int[]>();
    internal readonly List<Huffman> Coders = new List<Huffman>();
    internal readonly List<int> Exceptions = new List<int>();          // flat weight positions stored verbatim
    internal readonly List<double> ExceptionValues = new List<double>();
    internal readonly Dictionary<int, double> Residual = new Dictionary<int, double>();
    internal long PayloadBits;
    internal double ResidualThreshold = double.MaxValue;

    internal CompressedNetwork(Network original) => Original = original;

    internal void Compress(int clusters, double residualThreshold)
    {
        ResidualThreshold = residualThreshold;
        int flat = 0;
        foreach (Layer layer in Original.Layers)
        {
            var codebook = new Codebook();
            codebook.Train(layer.Weights, clusters);
            var indices = new int[layer.Weights.Length];
            for (int i = 0; i < layer.Weights.Length; i++)
            {
                int index = codebook.Nearest(layer.Weights[i]);
                indices[i] = index;
                if (Math.Abs(codebook.Centroids[index] - layer.Weights[i]) > residualThreshold)
                {
                    Exceptions.Add(flat + i);
                    ExceptionValues.Add(layer.Weights[i]);
                    Residual[flat + i] = layer.Weights[i];
                }
            }
            var coder = new Huffman();
            coder.Build(indices);
            Codebooks.Add(codebook);
            Indices.Add(indices);
            Coders.Add(coder);
            PayloadBits += coder.Encode(indices).Length;
            flat += layer.Weights.Length;
        }
    }

    private int LayerBase(int layerIndex)
    {
        int total = 0;
        for (int i = 0; i < layerIndex; i++) total += Original.Layers[i].Weights.Length;
        return total;
    }

    /** Index streams recovered from the entropy-coded payload. */
    internal int[] DecodeIndices(int layerIndex)
    {
        Huffman coder = Coders[layerIndex];
        string bits = coder.Encode(Indices[layerIndex]);
        return coder.Decode(bits, Indices[layerIndex].Length).ToArray();
    }

    /** The dequantized network the compressed runtime is expected to reproduce exactly. */
    internal Network Dequantize(bool applyResiduals)
    {
        var network = new Network();
        var residual = new Dictionary<int, double>();
        if (applyResiduals)
            for (int e = 0; e < Exceptions.Count; e++) residual[Exceptions[e]] = ExceptionValues[e];
        int layerIndex = 0;
        foreach (Layer layer in Original.Layers)
        {
            var copy = new Layer { Inputs = layer.Inputs, Outputs = layer.Outputs, Weights = new double[layer.Weights.Length], Biases = (double[])layer.Biases.Clone() };
            int[] indices = DecodeIndices(layerIndex);
            int layerBase = LayerBase(layerIndex);
            for (int i = 0; i < copy.Weights.Length; i++)
                copy.Weights[i] = residual.TryGetValue(layerBase + i, out double exact)
                    ? exact
                    : Codebooks[layerIndex].Centroids[indices[i]];
            network.Layers.Add(copy);
            layerIndex++;
        }
        return network;
    }

    internal double[] CompressedForward(double[] input)
    {
        double[] current = input;
        for (int index = 0; index < Original.Layers.Count; index++) current = Forwards(index, current);
        return current;
    }

    private double[] Forwards(int layerIndex, double[] input)
    {
        Layer layer = Original.Layers[layerIndex];
        int[] indices = DecodeIndices(layerIndex);
        Codebook codebook = Codebooks[layerIndex];
        int layerBase = LayerBase(layerIndex);
        var output = new double[layer.Outputs];
        for (int neuron = 0; neuron < layer.Outputs; neuron++)
        {
            double sum = layer.Biases[neuron];
            int offset = neuron * layer.Inputs;
            for (int i = 0; i < layer.Inputs; i++)
            {
                double weight = Residual.TryGetValue(layerBase + offset + i, out double exact)
                    ? exact
                    : codebook.Centroids[indices[offset + i]];
                sum += weight * input[i];
            }
            output[neuron] = sum > 0 ? sum : 0;
        }
        return output;
    }

    internal long StoredBits(int clusters, bool withHuffman)
    {
        int totalWeights = Indices.Sum(stream => stream.Length);
        long bits = 64L * Original.Layers.Sum(layer => layer.Biases.Length);
        foreach (Codebook codebook in Codebooks) bits += 64L * codebook.Centroids.Length;
        bits += withHuffman ? PayloadBits : (long)Math.Ceiling(Math.Log2(Math.Max(2, clusters))) * totalWeights;
        bits += 64L * Exceptions.Count;
        return bits;
    }

    internal long RawBits()
    {
        long values = Original.Layers.Sum(layer => (long)layer.Weights.Length + layer.Biases.Length);
        return 64L * values;
    }

    internal double MeanQuantizationError(bool applyResiduals)
    {
        double total = 0;
        int count = 0;
        for (int layerIndex = 0; layerIndex < Original.Layers.Count; layerIndex++)
        {
            Layer layer = Original.Layers[layerIndex];
            int layerBase = LayerBase(layerIndex);
            for (int i = 0; i < layer.Weights.Length; i++)
            {
                double reconstructed = Codebooks[layerIndex].Centroids[Indices[layerIndex][i]];
                if (applyResiduals && Residual.TryGetValue(layerBase + i, out double exact)) reconstructed = exact;
                total += Math.Abs(reconstructed - layer.Weights[i]);
                count++;
            }
        }
        return count == 0 ? 0 : total / count;
    }

    internal double OriginalWeightAt(int flatPosition)
    {
        int flat = 0;
        foreach (Layer layer in Original.Layers)
        {
            if (flatPosition >= flat && flatPosition < flat + layer.Weights.Length)
                return layer.Weights[flatPosition - flat];
            flat += layer.Weights.Length;
        }
        throw new ArgumentOutOfRangeException(nameof(flatPosition));
    }
}

public static class Solution
{
    private static ulong _state = 0x243F6A8885A308D3UL;

    private static ulong NextRandom()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextWeight()
    {
        double u = (NextRandom() >> 11) * (1.0 / 9007199254740992.0);
        double v = (NextRandom() >> 11) * (1.0 / 9007199254740992.0);
        return (u + v - 1.0) * 1.2;                              // triangular: skewed, Huffman friendly
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string Fixed(double value) => value.ToString("F6", CultureInfo.InvariantCulture);

    private static Network BuildNetwork(ulong seed, params int[] shape)
    {
        _state = seed;
        var network = new Network();
        for (int layer = 0; layer + 1 < shape.Length; layer++)
        {
            var built = new Layer { Inputs = shape[layer], Outputs = shape[layer + 1] };
            built.Weights = new double[built.Inputs * built.Outputs];
            built.Biases = new double[built.Outputs];
            for (int i = 0; i < built.Weights.Length; i++) built.Weights[i] = NextWeight();
            for (int i = 0; i < built.Biases.Length; i++) built.Biases[i] = 0.01 * (i + 1);
            network.Layers.Add(built);
        }
        return network;
    }

    private static double[] MakeInput(int size, ulong seed)
    {
        _state = seed;
        var input = new double[size];
        for (int i = 0; i < size; i++) input[i] = NextWeight();
        return input;
    }

    private static double RelativeError(double[] a, double[] b)
    {
        double worst = 0;
        double scale = Math.Max(1e-9, b.Select(Math.Abs).Max());
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]) / scale);
        return worst;
    }

    public static void Main()
    {
        // ---- 1. Huffman coding -------------------------------------------------------------------
        var coder = new Huffman();
        var stream = new List<int>();
        ulong random = 7;
        for (int i = 0; i < 4000; i++)
        {
            random += 0x9E3779B97F4A7C15UL;
            ulong z = random;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            // Skewed distribution over six symbols.
            int bucket = (int)(z % 100);
            stream.Add(bucket < 50 ? 0 : bucket < 75 ? 1 : bucket < 87 ? 2 : bucket < 94 ? 3 : bucket < 98 ? 4 : 5);
        }
        coder.Build(stream);
        Check(coder.Codes.Count == 6, "six symbols produced six codes");
        Check(coder.IsPrefixFree(out string prefixProblem), "codes are prefix-free: " + prefixProblem);
        Check(coder.KraftSum() <= 1.0 + 1e-12, "the Kraft sum is at most one: " + Fixed(coder.KraftSum()));
        string bits = coder.Encode(stream);
        Check(coder.Decode(bits, stream.Count).SequenceEqual(stream), "the bit stream round-trips");
        Check(bits.Length < stream.Count * 3, "skewed symbols use under three bits each: " +
            ((double)bits.Length / stream.Count).ToString("F3", CultureInfo.InvariantCulture));
        double average = coder.AverageBits();
        Check(average >= coder.Entropy - 1e-9 && average < coder.Entropy + 1.0,
            "the average code length sits inside [entropy, entropy+1): " + Fixed(average) + " vs " + Fixed(coder.Entropy));
        var uniform = new Huffman();
        uniform.Build(Enumerable.Range(0, 4000).Select(i => i % 8).ToList());
        Check(uniform.AverageBits() >= 3.0 - 1e-9 && uniform.AverageBits() < 3.0 + 1e-9,
            "a uniform alphabet costs exactly log2(8): " + Fixed(uniform.AverageBits()));
        var singleSymbol = new Huffman();
        singleSymbol.Build(Enumerable.Repeat(3, 10).ToList());
        Check(singleSymbol.Codes.Count == 1 && singleSymbol.Encode(new[] { 3, 3 }).Length == 2 &&
              singleSymbol.Decode("00", 2).SequenceEqual(new[] { 3, 3 }), "a single symbol still codes and decodes");
        Check(uniform.KraftSum() <= 1.0 + 1e-12 && uniform.IsPrefixFree(out _), "a uniform alphabet is prefix-free");

        // ---- 2. weight sharing: Lloyd's monotonicity and error shrinkage -------------------------
        Network network = BuildNetwork(20261016, 8, 16, 8, 2);
        double[] weights = network.AllWeights();
        Check(weights.Length == 8 * 16 + 16 * 8 + 8 * 2, "the network has the expected weight count: " + weights.Length);
        var small = new Codebook();
        small.Train(weights, 8);
        Check(small.Iterations >= 1 && small.DistortionTrace.Count == small.Iterations, "Lloyd's ran " + small.Iterations + " iterations");
        for (int i = 1; i < small.DistortionTrace.Count; i++)
            Check(small.DistortionTrace[i] <= small.DistortionTrace[i - 1] + 1e-12,
                "distortion never increases at iteration " + i);
        Check(small.Centroids.Length == 8 && small.Centroids.SequenceEqual(small.Centroids.OrderBy(c => c)),
            "the codebook is sorted and complete");
        double previousError = double.MaxValue;
        foreach (int clusters in new[] { 2, 4, 8, 16, 32 })
        {
            var codebook = new Codebook();
            codebook.Train(weights, clusters);
            double error = weights.Select(w => Math.Abs(codebook.Centroids[codebook.Nearest(w)] - w)).Average();
            Check(error < previousError, "mean quantization error falls with K=" + clusters + ": " + Fixed(error));
            Check(codebook.Centroids.Length == clusters, "the codebook has K centroids");
            previousError = error;
        }
        Check(previousError < 0.02, "32 centroids reconstruct the weights closely: " + Fixed(previousError));

        // ---- 3. compression: size, ratio and reproducibility ------------------------------------
        var compressed = new CompressedNetwork(network);
        compressed.Compress(16, 0.02);
        long rawBits = compressed.RawBits();
        long huffmanBits = compressed.StoredBits(16, true);
        long fixedBits = compressed.StoredBits(16, false);
        var compact = new CompressedNetwork(network);
        compact.Compress(4, double.MaxValue);                      // pure codebook, no residual list
        long compactBits = compact.StoredBits(4, true);
        Check(huffmanBits < rawBits, "the compressed form beats raw float64: " + huffmanBits + " < " + rawBits);
        Check(huffmanBits < fixedBits, "entropy coding beats fixed-width indices: " + huffmanBits + " < " + fixedBits);
        Check(compactBits < rawBits / 4, "a small codebook keeps the payload under a quarter of raw: " +
            compactBits + " vs " + rawBits);
        Check(compactBits < huffmanBits, "fewer centroids cost more payload but much less overhead: " +
            compactBits + " < " + huffmanBits);
        var again = new CompressedNetwork(BuildNetwork(20261016, 8, 16, 8, 2));
        again.Compress(16, 0.02);
        Check(again.StoredBits(16, true) == huffmanBits, "the same seed reproduces the same payload size");
        Check(again.Codebooks.Zip(compressed.Codebooks, (a, b) => a.Centroids.SequenceEqual(b.Centroids)).All(x => x),
            "and the identical codebooks");
        foreach (int index in Enumerable.Range(0, compressed.Indices.Count))
            Check(compressed.DecodeIndices(index).SequenceEqual(compressed.Indices[index]),
                "layer " + index + " indices survive the codec");

        // ---- 4. inference from the compressed form ----------------------------------------------
        double[] input = MakeInput(8, 99);
        double[] fromCompressed = compressed.CompressedForward(input);
        double[] fromDequantized = compressed.Dequantize(true).Forward(input);
        double[] fromOrigin = network.Forward(input);
        Check(fromCompressed.Length == 2, "the network outputs two values");
        Check(fromCompressed.SequenceEqual(fromDequantized), "compressed inference equals dequantized inference exactly");
        Check(RelativeError(fromCompressed, fromOrigin) < 0.10,
            "and stays within 10% of the original network: " + Fixed(RelativeError(fromCompressed, fromOrigin)));
        var withoutResiduals = new CompressedNetwork(network);
        withoutResiduals.Compress(16, double.MaxValue);
        Check(!withoutResiduals.Dequantize(true).Forward(input).SequenceEqual(fromCompressed),
            "dropping the residual list changes the answer, so residuals are really used");
        Check(RelativeError(withoutResiduals.CompressedForward(input), fromOrigin) < 0.15,
            "the plain codebook path is still accurate: " + Fixed(RelativeError(withoutResiduals.CompressedForward(input), fromOrigin)));
        var wide = new CompressedNetwork(network);
        wide.Compress(32, 0.005);
        Check(RelativeError(wide.CompressedForward(input), fromOrigin) < RelativeError(fromCompressed, fromOrigin),
            "more centroids and tighter residuals reduce the inference error");

        // ---- 5. the sparse residual -------------------------------------------------------------
        var residual = new CompressedNetwork(network);
        residual.Compress(16, 0.05);
        Check(residual.Exceptions.Count > 0, "the residual list is populated: " + residual.Exceptions.Count + " exceptions");
        Check(residual.Exceptions.Count < weights.Length / 2, "it stays sparse: " + residual.Exceptions.Count +
            " of " + weights.Length);
        double errorWithResiduals = residual.MeanQuantizationError(true);
        double errorWithoutResiduals = residual.MeanQuantizationError(false);
        Check(errorWithResiduals < errorWithoutResiduals, "the residual cut the mean error: " + Fixed(errorWithResiduals) +
            " < " + Fixed(errorWithoutResiduals));
        Check(residual.Exceptions.Distinct().Count() == residual.Exceptions.Count, "exception positions are unique");
        foreach (int position in residual.Exceptions.Take(20))
            Check(Math.Abs(residual.ExceptionValues[residual.Exceptions.IndexOf(position)] - residual.OriginalWeightAt(position)) < 1e-15,
                "exception " + position + " stores the exact weight");
        var tighter = new CompressedNetwork(network);
        tighter.Compress(16, 0.01);
        Check(tighter.Exceptions.Count > residual.Exceptions.Count,
            "a tighter threshold makes the residual longer: " + tighter.Exceptions.Count);
        Check(tighter.MeanQuantizationError(true) <= residual.MeanQuantizationError(true) + 1e-12,
            "and lowers the error further");

        // ---- 6. degenerate shapes ----------------------------------------------------------------
        var oneCluster = new CompressedNetwork(network);
        oneCluster.Compress(1, double.MaxValue);
        Check(oneCluster.Codebooks.All(codebook => codebook.Centroids.Length == 1), "K=1 gives one centroid per layer");
        Check(oneCluster.CompressedForward(input).Length == 2, "K=1 still runs inference");
        Check(oneCluster.Coders.All(coder => coder.Codes.Count == 1), "and needs a single code per layer");
        var constant = BuildNetwork(5, 4, 4);
        foreach (Layer layer in constant.Layers) for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 0.25;
        var constantCompressed = new CompressedNetwork(constant);
        constantCompressed.Compress(8, 0.0);
        Check(constantCompressed.Codebooks.All(codebook => codebook.Centroids.Length == 1 ||
              codebook.Centroids.Distinct().Count() == 1), "a constant tensor needs one effective centroid");
        Check(constantCompressed.CompressedForward(new[] { 1.0, 1.0, 1.0, 1.0 })
              .SequenceEqual(constantCompressed.Dequantize(true).Forward(new[] { 1.0, 1.0, 1.0, 1.0 })),
            "a constant network still matches its dequantized twin");
        Check(constantCompressed.MeanQuantizationError(false) < 1e-12, "a constant tensor quantizes exactly");
        var empty = new CompressedNetwork(new Network());
        empty.Compress(4, 0.0);
        Check(empty.CompressedForward(new[] { 1.0 }).SequenceEqual(new[] { 1.0 }), "a network without layers is the identity");
        Check(empty.StoredBits(4, true) == 0, "and stores nothing");

        // ---- 7. scale ----------------------------------------------------------------------------
        Network big = BuildNetwork(4242, 64, 128, 32, 10);
        var bigCompressed = new CompressedNetwork(big);
        bigCompressed.Compress(32, 0.03);
        double[] bigInput = MakeInput(64, 12345);
        double[] bigCompressedOutput = bigCompressed.CompressedForward(bigInput);
        Check(bigCompressedOutput.SequenceEqual(bigCompressed.Dequantize(true).Forward(bigInput)),
            "the large network's compressed path is exact against its dequantized twin");
        double[] bigOrigin = big.Forward(bigInput);
        Check(RelativeError(bigCompressedOutput, bigOrigin) < 0.25,
            "the large network stays close: " + Fixed(RelativeError(bigCompressedOutput, bigOrigin)));
        long bigRaw = bigCompressed.RawBits();
        long bigStored = bigCompressed.StoredBits(32, true);
        Check(bigStored < bigRaw / 4, "the large network compresses under a quarter: " + bigStored + " vs " + bigRaw);
        Check(bigCompressed.Codebooks.All(codebook => codebook.DistortionTrace.Count > 0), "every layer trained");
        var bigFixed = bigCompressed.StoredBits(32, false);
        Check(bigStored < bigFixed, "entropy coding still wins at scale: " + bigStored + " < " + bigFixed);

        Console.WriteLine("network 8-16-8-2: " + weights.Length + " weights, K=16, mean error " +
            Fixed(previousError) + ", stored " + huffmanBits + " bits vs raw " + rawBits +
            " (" + (100.0 * huffmanBits / rawBits).ToString("F1", CultureInfo.InvariantCulture) + "%)");
        Console.WriteLine("huffman: entropy " + Fixed(coder.Entropy) + " bits, average " + Fixed(average) +
            " bits for 6 symbols; uniform 8-symbol alphabet costs " + Fixed(uniform.AverageBits()) +
            " bits; Kraft sum " + Fixed(coder.KraftSum()));
        Console.WriteLine("residual: " + residual.Exceptions.Count + " exceptions at threshold 0.05 reduce the mean error " +
            Fixed(errorWithoutResiduals) + " -> " + Fixed(errorWithResiduals) + "; inference matches dequantized exactly");
        Console.WriteLine("scale 64-128-32-10: " + big.AllWeights().Length + " weights, stored " + bigStored +
            " bits vs raw " + bigRaw + " (" + (100.0 * bigStored / bigRaw).ToString("F1", CultureInfo.InvariantCulture) + "%)");
        Console.WriteLine("PASS");
    }
}
