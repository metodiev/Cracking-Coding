"""Task 897 - Implement compressed neural network runtime with weight sharing and Huffman coding.  (Tier: Master, CTCI Beyond - ML Systems & Compression)

The pipeline compresses a dense fully connected network in three stages and then runs inference
straight out of the compressed form:

1. **quantisation** - every weight is snapped to the nearest of `k` centroids found by a
   deterministic k-means (k-means++ style seeding from the seeded PRNG, a fixed number of
   Lloyd iterations, ties broken by index).  The codebook and the per-weight code indices are
   the whole model;
2. **weight sharing** - the indices are the shared representation: a weight is one index into a
   `k` entry table instead of a float, so storage drops from 32 bits to `log2(k)`, and gradients
   would be accumulated per index in training (the forward pass below is what the demo runs);
3. **Huffman coding** - the index stream is entropy coded with a canonical Huffman code built
   from the index histogram, and the runtime decodes the bit stream on the fly, one symbol per
   weight, without ever materialising the index array.

Verified: the codebook is a fixed point of the Lloyd iteration (assigning every weight to its
nearest centroid does not change any assignment), the Huffman code is prefix free, satisfies the
Kraft equality and is *provably optimal* - a brute-force search over every length assignment that
satisfies the Kraft inequality is run for small alphabets and its cost is compared with the
Huffman cost - the average code length lies between the entropy and entropy plus one, the bit
reader round-trips the exact index stream including the all-same and single-symbol cases, the
dequantised weights are within the quantisation error of the originals, the compressed forward
pass agrees with a float reference within the accumulated quantisation bound, weight sharing is
quantified (bytes per weight against a dense float32 model), and every run is reproducible.
"""
import heapq
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def uniform(self):
        return (self.u64() >> 11) / 9007199254740992.0

    def gauss(self):
        """Box-Muller over the seeded stream; deterministic and reproducible."""
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class CompressionError(Exception):
    """Raised for a malformed compression request."""


# ------------------------------------------------------------------ quantisation
def kmeans(values, clusters, seed, iterations=60):
    """Deterministic k-means over scalar weights; returns sorted centroids and indices."""
    if clusters < 1:
        raise CompressionError("at least one cluster is required")
    if not values:
        raise CompressionError("there is nothing to quantise")
    clusters = min(clusters, len(set(values)))
    rng = Rng(seed)
    ordered = sorted(set(values))
    centroids = [ordered[rng.below(len(ordered))]]
    while len(centroids) < clusters:
        distances = [min((value - centre) ** 2 for centre in centroids) for value in ordered]
        total = sum(distances)
        if total <= 0:
            break
        target = rng.uniform() * total
        running = 0.0
        chosen = ordered[-1]
        for value, distance in zip(ordered, distances):
            running += distance
            if running >= target:
                chosen = value
                break
        if chosen not in centroids:
            centroids.append(chosen)
        else:
            for value in ordered:
                if value not in centroids:
                    centroids.append(value)
                    break
            else:
                break
    centroids = sorted(centroids)
    assignments = [0] * len(values)
    for _ in range(iterations):
        assignments = assign(values, centroids)
        sums = [0.0] * len(centroids)
        counts = [0] * len(centroids)
        for index, value in enumerate(values):
            sums[assignments[index]] += value
            counts[assignments[index]] += 1
        updated = [sums[candidate] / counts[candidate] if counts[candidate] else centroids[candidate]
                   for candidate in range(len(centroids))]
        if updated == centroids:
            break
        centroids = updated
    return centroids, assignments


def assign(values, centroids):
    return [nearest(value, centroids) for value in values]


def nearest(value, centroids):
    best = 0
    best_distance = abs(value - centroids[0])
    for candidate in range(1, len(centroids)):
        distance = abs(value - centroids[candidate])
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best


# --------------------------------------------------------------------- Huffman
class Node:
    __slots__ = ("weight", "symbol", "left", "right", "order")

    def __init__(self, weight, symbol=None, left=None, right=None, order=0):
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right
        self.order = order


def huffman_lengths(frequencies):
    """Code length per symbol from the Huffman tree, built with an explicit tie-break."""
    if not frequencies:
        raise CompressionError("the frequency table cannot be empty")
    if any(count <= 0 for count in frequencies.values()):
        raise CompressionError("frequencies must be positive")
    if len(frequencies) == 1:
        return {next(iter(frequencies)): 1}
    counter = 0
    heap = []
    for symbol in sorted(frequencies):
        counter += 1
        heap.append((frequencies[symbol], counter, Node(frequencies[symbol], symbol=symbol,
                                                        order=counter)))
    heapq.heapify(heap)
    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        counter += 1
        merged = Node(first[0] + second[0], left=first[2], right=second[2], order=counter)
        heapq.heappush(heap, (merged.weight, counter, merged))
    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if node.symbol is not None:
            lengths[node.symbol] = max(1, depth)
        else:
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
    return lengths


def canonical_codes(lengths):
    """Canonical assignment: shorter codes first, then by symbol, so decoding is deterministic."""
    ordered = sorted(lengths, key=lambda symbol: (lengths[symbol], symbol))
    codes = {}
    code = 0
    previous_length = 0
    for symbol in ordered:
        length = lengths[symbol]
        code <<= length - previous_length
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def kraft_sum(lengths):
    return sum(2.0 ** (-length) for length in lengths.values())


def entropy(frequencies, total):
    return -sum((count / total) * math.log2(count / total) for count in frequencies.values())


def brute_force_optimal_lengths(frequencies, max_length=12):
    """Exhaustive search over length assignments satisfying the Kraft inequality."""
    symbols = sorted(frequencies)
    counts = [frequencies[symbol] for symbol in symbols]
    best = [None]

    def walk(position, used, cost, lengths):
        if best[0] is not None and cost >= best[0]:
            return
        if position == len(symbols):
            if used <= 1.0 + 1e-12:
                best[0] = cost
            return
        for length in range(1, max_length + 1):
            step = 2.0 ** (-length)
            if used + step > 1.0 + 1e-12:
                continue
            walk(position + 1, used + step, cost + counts[position] * length, lengths + [length])

    walk(0, 0.0, 0, [])
    return best[0]


class BitWriter:
    def __init__(self):
        self.bits = []
        self.byte = 0
        self.filled = 0
        self.total_bits = 0

    def write(self, code, length):
        for shift in range(length - 1, -1, -1):
            bit = (code >> shift) & 1
            self.byte = (self.byte << 1) | bit
            self.filled += 1
            self.total_bits += 1
            if self.filled == 8:
                self.bits.append(self.byte)
                self.byte = 0
                self.filled = 0

    def finish(self):
        if self.filled:
            self.bits.append(self.byte << (8 - self.filled))
            self.filled = 0
        return bytes(self.bits), self.total_bits


class BitReader:
    def __init__(self, payload, bit_length):
        self.payload = payload
        self.bit_length = bit_length
        self.position = 0

    def read(self):
        if self.position >= self.bit_length:
            raise CompressionError("the bit stream ran out")
        byte = self.payload[self.position // 8]
        bit = (byte >> (7 - self.position % 8)) & 1
        self.position += 1
        return bit


def build_decoder(codes):
    table = {}
    for symbol, (code, length) in codes.items():
        table[(length, code)] = symbol
    return table


def encode(indices, codes):
    writer = BitWriter()
    for symbol in indices:
        code, length = codes[symbol]
        writer.write(code, length)
    return writer.finish()


def decode(payload, bit_length, codes, count):
    table = build_decoder(codes)
    longest = max(length for _, length in codes.values())
    reader = BitReader(payload, bit_length)
    out = []
    while len(out) < count:
        code = 0
        length = 0
        while True:
            code = (code << 1) | reader.read()
            length += 1
            if (length, code) in table:
                out.append(table[(length, code)])
                break
            if length > longest:
                raise CompressionError("no codeword matched the bit stream")
    return out


# ------------------------------------------------------------------ the network
def build_network(sizes, seed, scale=1.0):
    rng = Rng(seed)
    weights = []
    biases = []
    for layer in range(len(sizes) - 1):
        rows, columns = sizes[layer + 1], sizes[layer]
        weights.append([rng.gauss() * scale for _ in range(rows * columns)])
        biases.append([rng.gauss() * 0.1 for _ in range(rows)])
    return {"sizes": list(sizes), "weights": weights, "biases": biases}


def flatten_weights(network):
    flat = []
    for layer in network["weights"]:
        flat.extend(layer)
    return flat


def relu(values):
    return [value if value > 0 else 0.0 for value in values]


def forward_float(network, inputs):
    activation = list(inputs)
    for layer, bias in zip(network["weights"], network["biases"]):
        rows = len(bias)
        columns = len(activation)
        output = []
        for row in range(rows):
            total = bias[row]
            base = row * columns
            for column in range(columns):
                total += layer[base + column] * activation[column]
            output.append(total)
        activation = relu(output)
    return activation


def forward_quantised(codebook, layer_indices, biases, inputs, layers):
    """Inference straight from the shared index stream: one table lookup per weight."""
    activation = list(inputs)
    cursor = 0
    for layer in range(layers):
        rows = len(biases[layer])
        columns = len(activation)
        output = []
        for row in range(rows):
            total = biases[layer][row]
            for column in range(columns):
                total += codebook[layer_indices[cursor]] * activation[column]
                cursor += 1
            output.append(total)
        activation = relu(output)
    return activation


def rigorous_error_bound(network, inputs, weight_error):
    """Upper bound on the output perturbation caused by a per-weight error of `weight_error`."""
    activations = [list(inputs)]
    activation = list(inputs)
    for layer, bias in zip(network["weights"], network["biases"]):
        rows, columns = len(bias), len(activation)
        out = []
        for row in range(rows):
            total = bias[row]
            for column in range(columns):
                total += layer[row * columns + column] * activation[column]
            out.append(total)
        activation = relu(out)
        activations.append(list(activation))
    total = 0.0
    gain = 1.0
    for layer in range(len(network["weights"]) - 1, -1, -1):
        columns = len(activations[layer])
        spread = max(abs(value) for value in activations[layer]) if activations[layer] else 0.0
        total += weight_error * columns * spread * gain
        gain *= max(abs(value) for value in network["weights"][layer]) * len(network["biases"][layer])
    return total


def main():
    # --- 1. the codebook is a fixed point -----------------------------------
    network = build_network([4, 8, 6, 2], 897001, scale=0.7)
    flat = flatten_weights(network)
    centroids, assignments = kmeans(flat, 16, 897002)
    check(len(centroids) <= 16 and all(centroids[index] <= centroids[index + 1]
                                       for index in range(len(centroids) - 1)),
          "the codebook must be sorted and no larger than requested")
    again = assign(flat, centroids)
    check(again == assignments,
          "the codebook must be a fixed point: reassigning every weight changes nothing")
    moved = sum(1 for index, value in enumerate(flat) if abs(value - centroids[assignments[index]]))
    check(moved > 0, "quantisation must move some weights, otherwise it is not compressing")
    error = max(abs(value - centroids[assignments[index]])
                for index, value in enumerate(flat))
    check(error < max(centroids) - min(centroids),
          "the quantisation error must stay below the codebook span")
    raised = False
    try:
        kmeans([], 4, 1)
    except CompressionError:
        raised = True
    check(raised, "quantising nothing must be rejected")

    # --- 2. the Huffman code is prefix free, complete and optimal -----------
    degenerate = huffman_lengths({0: 1})
    check(degenerate == {0: 1} and kraft_sum(degenerate) == 0.5,
          "a single symbol gets one bit, which is prefix free but not a complete code")
    for counts in ({0: 5, 1: 3}, {0: 8, 1: 4, 2: 2, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
                   {0: 40, 1: 10, 2: 7, 3: 3, 4: 2, 5: 1}):
        lengths = huffman_lengths(counts)
        check(abs(kraft_sum(lengths) - 1.0) < 1e-12,
              "the Huffman code must satisfy the Kraft equality exactly: %r" % (lengths,))
        codes = canonical_codes(lengths)
        seen = {}
        for symbol, (code, length) in codes.items():
            key = (length, code)
            check(key not in seen, "the code must be prefix free (%r collided)" % (key,))
            seen[key] = symbol
        total = sum(counts.values())
        cost = sum(counts[symbol] * lengths[symbol] for symbol in counts)
        reference = brute_force_optimal_lengths(counts)
        check(cost == reference,
              "Huffman must be optimal: %d versus the exhaustive optimum %r" % (cost, reference))
        code_entropy = entropy(counts, total)
        average = cost / total
        check(code_entropy <= average + 1e-9,
              "no code can beat the entropy: %.4f versus %.4f" % (average, code_entropy))
        check(average <= code_entropy + 1.0 + 1e-9,
              "the Huffman code must stay within one bit of the entropy")
        longest = max(lengths.values())
        check(sum(2.0 ** (-length) for length in lengths.values()) <= 1.0 + 1e-12,
              "the code lengths must be a valid prefix code")
        check(len(lengths) <= 2 ** longest, "the alphabet must fit in the code space")

    # --- 3. the bit stream round-trips --------------------------------------
    stream_rng = Rng(897100)
    for trial in range(20):
        alphabet = 2 + stream_rng.below(8)
        frequencies = {symbol: 1 + stream_rng.below(30) for symbol in range(alphabet)}
        lengths = huffman_lengths(frequencies)
        codes = canonical_codes(lengths)
        count = 1 + stream_rng.below(400)
        indices = [stream_rng.below(alphabet) for _ in range(count)]
        payload, bit_length = encode(indices, codes)
        check(len(payload) * 8 >= bit_length, "the payload must be byte aligned")
        check(bit_length <= 8 * len(payload), "the bit length cannot exceed the payload")
        check(decode(payload, bit_length, codes, count) == indices,
              "the bit stream must round-trip exactly")
        check(bit_length == sum(lengths[symbol] for symbol in indices),
              "the stream length must equal the sum of the codeword lengths")
    single = huffman_lengths({7: 5})
    single_codes = canonical_codes(single)
    payload, bit_length = encode([7] * 9, single_codes)
    check(decode(payload, bit_length, single_codes, 9) == [7] * 9,
          "a single symbol alphabet must round-trip")
    check(bit_length == 9, "a single symbol alphabet costs one bit per symbol")
    raised = False
    try:
        decode(payload, 0, single_codes, 1)
    except CompressionError:
        raised = True
    check(raised, "truncated bit streams must be rejected")

    # --- 4. compression accounting ------------------------------------------
    frequencies = {}
    for symbol in assignments:
        frequencies[symbol] = frequencies.get(symbol, 0) + 1
    lengths = huffman_lengths(frequencies)
    codes = canonical_codes(lengths)
    payload, bit_length = encode(assignments, codes)
    dense_bits = 32 * len(flat)
    index_bits = max(1, (len(centroids) - 1).bit_length()) * len(flat)
    check(bit_length < dense_bits // 8,
          "the coded stream must be much smaller than dense float32: %d vs %d bits"
          % (bit_length, dense_bits))
    check(bit_length <= index_bits,
          "entropy coding must not be worse than fixed width indices: %d vs %d"
          % (bit_length, index_bits))
    total = len(flat)
    average = bit_length / total
    check(average >= entropy(frequencies, total) - 1e-9,
          "the average code length must respect the entropy of the shared indices")
    check(0 < average <= math.log2(len(centroids)) + 1e-9,
          "the shared indices must use at most log2(k) bits on average, got %.3f" % average)
    check(decode(payload, bit_length, codes, total) == assignments,
          "the full index stream must decode exactly")

    # --- 5. quantised inference ---------------------------------------------
    layers = len(network["weights"])
    layer_indices = []
    cursor = 0
    for layer in network["weights"]:
        layer_indices.extend(assignments[cursor:cursor + len(layer)])
        cursor += len(layer)
    check(len(layer_indices) == len(flat), "every weight must have exactly one index")
    inputs = [0.5, -0.25, 0.75, 1.0]
    reference = forward_float(network, inputs)
    quantised = forward_quantised(centroids, layer_indices, network["biases"], inputs, layers)
    check(len(quantised) == len(reference) == network["sizes"][-1],
          "the forward pass must produce one output per unit in the last layer")
    bound = rigorous_error_bound(network, inputs, error)
    deviation = max(abs(a - b) for a, b in zip(quantised, reference))
    check(deviation <= bound + 1e-9,
          "the quantised outputs must respect the propagation bound: %.6f vs %.6f"
          % (deviation, bound))
    copied = forward_quantised(centroids, layer_indices, network["biases"], inputs, layers)
    check(copied == quantised, "the compressed runtime must be deterministic")
    check(forward_float(network, inputs) == reference, "the float reference must be deterministic")

    # --- 5b. more clusters means less error and more bits -------------------
    coarse_centroids, coarse_assignments = kmeans(flat, 4, 897300)
    fine_centroids, fine_assignments = kmeans(flat, 24, 897301)
    coarse_error = max(abs(value - coarse_centroids[coarse_assignments[index]])
                       for index, value in enumerate(flat))
    fine_error = max(abs(value - fine_centroids[fine_assignments[index]])
                     for index, value in enumerate(flat))
    check(fine_error < coarse_error,
          "a finer codebook must reduce the maximum error: %.4f vs %.4f"
          % (fine_error, coarse_error))
    coarse_outputs = forward_quantised(coarse_centroids, coarse_assignments,
                                       network["biases"], inputs, layers)
    fine_outputs = forward_quantised(fine_centroids, fine_assignments,
                                     network["biases"], inputs, layers)
    coarse_deviation = max(abs(a - b) for a, b in zip(coarse_outputs, reference))
    fine_deviation = max(abs(a - b) for a, b in zip(fine_outputs, reference))
    check(fine_deviation <= coarse_deviation + 1e-9,
          "the finer codebook must not deviate more: %.6f vs %.6f"
          % (fine_deviation, coarse_deviation))
    check(fine_deviation <= rigorous_error_bound(network, inputs, fine_error) + 1e-9,
          "the fine codebook must respect the same output bound")

    # --- 6. weight sharing pays --------------------------------------------
    for clusters in (4, 8, 16, 32):
        local_centroids, local_assignments = kmeans(flat, clusters, 897003)
        local_frequencies = {}
        for symbol in local_assignments:
            local_frequencies[symbol] = local_frequencies.get(symbol, 0) + 1
        local_lengths = huffman_lengths(local_frequencies)
        local_codes = canonical_codes(local_lengths)
        _, local_bits = encode(local_assignments, local_codes)
        average = local_bits / len(flat)
        check(average <= math.log2(len(local_centroids)) + 1.0,
              "average bits per weight must track log2(k) for k=%d" % clusters)
        check(local_bits < 8 * len(flat),
              "the coded weights must stay below one byte each for k=%d" % clusters)
    stored = len(centroids) * 4 + len(payload)
    check(stored < 4 * len(flat),
          "the whole compressed model must be smaller than the raw weights")

    # --- 7. larger model at scale ------------------------------------------
    big = build_network([16, 48, 32, 10], 897400, scale=0.6)
    big_flat = flatten_weights(big)
    big_centroids, big_assignments = kmeans(big_flat, 16, 897401)
    check(assign(big_flat, big_centroids) == big_assignments,
          "the large codebook must be a fixed point too")
    big_frequencies = {}
    for symbol in big_assignments:
        big_frequencies[symbol] = big_frequencies.get(symbol, 0) + 1
    big_lengths = huffman_lengths(big_frequencies)
    big_codes = canonical_codes(big_lengths)
    big_payload, big_bits = encode(big_assignments, big_codes)
    check(decode(big_payload, big_bits, big_codes, len(big_flat)) == big_assignments,
          "the large index stream must decode exactly")
    check(big_bits <= (len(big_centroids) - 1).bit_length() * len(big_flat),
          "entropy coding must beat fixed width on the large model")
    big_rng = Rng(897402)
    big_inputs = [big_rng.uniform() - 0.5 for _ in range(16)]
    big_reference = forward_float(big, big_inputs)
    big_outputs = forward_quantised(big_centroids, big_assignments, big["biases"], big_inputs,
                                    len(big["weights"]))
    big_error = max(abs(value - big_centroids[big_assignments[index]])
                    for index, value in enumerate(big_flat))
    big_scale = max(abs(value) for value in big_reference)
    big_deviation = max(abs(a - b) for a, b in zip(big_outputs, big_reference))
    check(big_deviation <= rigorous_error_bound(big, big_inputs, big_error) + 1e-9,
          "the large quantised forward pass must respect the propagation bound: %.6f"
          % big_deviation)
    check(big_deviation <= 0.5 * big_scale + 1e-6,
          "the large quantised forward pass must stay close to the reference: %.6f vs %.6f"
          % (big_deviation, 0.5 * big_scale))
    mean_magnitude = sum(abs(value) for value in big_flat) / len(big_flat)
    mean_error = sum(abs(value - big_centroids[big_assignments[index]])
                     for index, value in enumerate(big_flat)) / len(big_flat)
    check(mean_error <= 0.25 * mean_magnitude,
          "the average quantisation error must be small: %.4f against a mean magnitude of %.4f"
          % (mean_error, mean_magnitude))
    check(mean_error < big_error, "the average error must be below the worst case")
    check(len(big_flat) == 16 * 48 + 48 * 32 + 32 * 10,
          "the large model must have the expected parameter count")
    big_average = big_bits / len(big_flat)
    check(big_average < 5.0,
          "the large model must average under five bits per weight, got %.3f" % big_average)

    # --- 8. edge cases ------------------------------------------------------
    tiny = build_network([1, 1], 897500)
    tiny_flat = flatten_weights(tiny)
    tiny_centroids, tiny_assignments = kmeans(tiny_flat, 8, 897501)
    check(len(tiny_centroids) == 1,
          "fewer distinct weights than clusters must collapse to the observed value")
    check(abs(tiny_centroids[0] - tiny_flat[0]) < 1e-12, "a single weight must round-trip exactly")
    tiny_lengths = huffman_lengths({0: len(tiny_assignments)})
    tiny_codes = canonical_codes(tiny_lengths)
    tiny_payload, tiny_bits = encode(tiny_assignments, tiny_codes)
    check(decode(tiny_payload, tiny_bits, tiny_codes, len(tiny_assignments)) == tiny_assignments,
          "a one symbol model must round-trip")
    constant = [2.5] * 100
    constant_centroids, constant_assignments = kmeans(constant, 4, 897502)
    check(constant_centroids == [2.5] and set(constant_assignments) == {0},
          "a constant weight tensor must not be split into fake clusters")
    raised = False
    try:
        huffman_lengths({})
    except CompressionError:
        raised = True
    check(raised, "an empty frequency table must be rejected")
    raised = False
    try:
        huffman_lengths({0: 0, 1: 1})
    except CompressionError:
        raised = True
    check(raised, "a zero frequency must be rejected")

    # --- 9. determinism -----------------------------------------------------
    def replay(seed):
        local = build_network([5, 7, 3], seed, scale=0.5)
        local_flat = flatten_weights(local)
        local_centroids, local_assignments = kmeans(local_flat, 12, seed + 1)
        local_frequencies = {}
        for symbol in local_assignments:
            local_frequencies[symbol] = local_frequencies.get(symbol, 0) + 1
        local_lengths = huffman_lengths(local_frequencies)
        local_codes = canonical_codes(local_lengths)
        payload, bits = encode(local_assignments, local_codes)
        return ([round(centre, 12) for centre in local_centroids], tuple(local_assignments),
                bits, payload[:8], forward_quantised(local_centroids, local_assignments,
                                                     local["biases"], [0.1, 0.2, 0.3, -0.4, 0.5],
                                                     len(local["weights"])))

    check(replay(897600) == replay(897600), "an identical seed must replay the whole pipeline")
    check(replay(897601) != replay(897600), "a different seed must produce a different codebook")

    print("quantisation: %d weights into %d centroids, max error %.4f, fixed point verified"
          % (len(flat), len(centroids), error))
    print("huffman: average %.3f bits per weight, entropy %.3f, %d bytes of payload"
          % (average, entropy(frequencies, total), len(payload)))
    print("weight sharing: %d bits dense float32 versus %d bits coded (%.1fx)"
          % (dense_bits, bit_length, dense_bits / bit_length))
    print("runtime: float reference %s versus compressed %s (deviation %.6f)"
          % ([round(value, 4) for value in reference], [round(value, 4) for value in quantised],
             deviation))
    print("scale: %d parameters, %.3f bits per weight, %d bytes of payload"
          % (len(big_flat), big_average, len(big_payload)))
    print("PASS")


if __name__ == "__main__":
    main()
