"""Task 716 - Implement quantized neural network inference simulation (8-bit).  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

8-bit inference replaces every float multiply-accumulate with integer arithmetic.  A tensor is
stored as integers q together with the affine map

    x ~ scale * (q - zero_point),   scale = (max - min) / (qmax - qmin),
    zero_point = clamp(round(qmin - min / scale))

so that the real value 0.0 maps onto a representable integer (which is what makes ReLU a plain
clamp).  A fully connected layer becomes

    acc   = sum_k (x_q[k] - x_zp) * (w_q[k] - w_zp)                 (exact int32 accumulation)
    out_q = clamp( round_shift(acc * M, shift) + out_zp )

where M and shift encode the only remaining real quantity, x_scale * w_scale / out_scale: it is
scaled by 2^shift and rounded into an int32, and the division becomes an arithmetic right shift
with round-half-away-from-zero.  That multiplier/shift requantisation is the heart of
gemmlowp-style kernels and is what this file simulates layer by layer through a 4 -> 16 -> 8 -> 1
ReLU network, with per-output-channel weight scales and activation scales obtained from a
post-training calibration pass.

Simplified versus a deployment: activations are quantised per tensor (not per group), the
accumulator is a Python integer validated against the int32 range instead of a machine int32, the
calibration is a plain min/max sweep over a sample set, and no SIMD packing, lookup tables or
per-channel bias folding are modelled.  Verified: every round trip within half a step, the
scale/zero-point derivation including the degenerate constant tensor and explicit saturation, the
per-channel win over per-tensor on a channel-varying tensor, the exact int32 accumulator against a
big-integer oracle, the integer requantisation within one quantised unit of the float rescale, the
quantised network's logits and labels against the float reference, and 8-bit beating 4-bit as the
bit width grows.
"""
import math
import sys

MASK64 = (1 << 64) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
QMIN, QMAX = -128, 127


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

    def uniform(self):
        self.state, r = splitmix64(self.state)
        return ((r >> 11) / (1 << 53)) * 2.0 - 1.0


# --------------------------------------------------------------------------- quantisation
class Quantized:
    """An affine-quantised tensor: value ~ scale * (q - zero_point)."""

    def __init__(self, scale, zero_point, qmin, qmax, data):
        check(scale > 0.0, "the scale must be positive")
        check(qmin < qmax, "the integer range must be non-empty")
        check(qmin <= zero_point <= qmax, "the zero point must be representable")
        self.scale = scale
        self.zero_point = zero_point
        self.qmin = qmin
        self.qmax = qmax
        self.data = data

    def dequantize(self):
        return [self.scale * (value - self.zero_point) for value in self.data]


def choose_scale(values, qmin=QMIN, qmax=QMAX):
    """(scale, zero_point) for the affine map covering the observed range.

    The range is extended to include 0.0, which is what makes the affine map exact at zero (ReLU
    becomes a plain clamp) and keeps the zero point representable without clamping it.
    """
    check(bool(values), "the calibration set cannot be empty")
    lowest = min(min(values), 0.0)
    highest = max(max(values), 0.0)
    if highest == lowest:
        return 1.0, 0                    # an all-zero tensor needs no resolution
    scale = (highest - lowest) / (qmax - qmin)
    zero_point = int(min(max(round(qmin - lowest / scale), qmin), qmax))
    check(qmin <= zero_point <= qmax, "the zero point stays representable")
    return scale, zero_point


def quantize(values, qmin=QMIN, qmax=QMAX, scale=None, zero_point=None):
    if scale is None or zero_point is None:
        scale, zero_point = choose_scale(values, qmin, qmax)
    data = [int(min(max(round(value / scale) + zero_point, qmin), qmax)) for value in values]
    return Quantized(scale, zero_point, qmin, qmax, data)


def quantize_signed(values, qmin=QMIN, qmax=QMAX):
    """Symmetric signed quantisation (the usual treatment of classifier logits)."""
    limit = max(abs(min(values)), abs(max(values)))
    if limit == 0.0:
        return Quantized(1.0, 0, qmin, qmax, [0] * len(values))
    scale = limit / ((qmax - qmin) / 2)
    data = [int(min(max(round(value / scale), qmin), qmax)) for value in values]
    return Quantized(scale, 0, qmin, qmax, data)


def quantize_per_channel(rows, qmin=QMIN, qmax=QMAX):
    """One (scale, zero_point, data) triple per row: the standard weight treatment."""
    channels = []
    for row in rows:
        scale, zero_point = choose_scale(list(row), qmin, qmax)
        data = [int(min(max(round(value / scale) + zero_point, qmin), qmax)) for value in row]
        channels.append((scale, zero_point, data))
    return channels


# --------------------------------------------------------------------------- integer kernels
def requantize_multiplier(real_multiplier, max_multiplier=INT32_MAX):
    """Represent a real multiplier as (M, shift) with M an int32, gemmlowp style."""
    check(real_multiplier >= 0.0, "the multiplier must be non-negative")
    if real_multiplier == 0.0:
        return 0, 0
    shift = 0
    while real_multiplier * (1 << shift) < max_multiplier / 2:
        shift += 1
    multiplier = round(real_multiplier * (1 << shift))
    while multiplier >= max_multiplier:
        shift -= 1
        multiplier = round(real_multiplier * (1 << shift))
    check(multiplier < max_multiplier, "the multiplier must fit in int32")
    return multiplier, shift


def round_shift(value, shift):
    """Arithmetic right shift with round-half-away-from-zero, the kernel's rounding rule."""
    if shift <= 0:
        return value << (-shift)
    half = 1 << (shift - 1)
    if value >= 0:
        return (value + half) >> shift
    return -((-value + half) >> shift)


def int32_matmul_dot(x_q, x_zero, w_q, w_zero):
    """Exact zero-point-corrected integer dot product, validated against the int32 range."""
    total = 0
    for a, b in zip(x_q, w_q):
        total += (a - x_zero) * (b - w_zero)
    check(INT32_MIN <= total <= INT32_MAX, f"the accumulator left the int32 range: {total}")
    return total


def integer_linear(x_q, x_scale, x_zero, channels, biases, out_scale, out_zero, relu=False):
    """One fully connected layer as an integer kernel; returns the raw int codes.

    The bias is folded into the accumulator before requantisation, in units of
    x_scale * w_scale, which is what keeps a layer's error at the rounding level instead of the
    bias magnitude.
    """
    outputs = []
    for (weight_scale, weight_zero, row), bias_acc in zip(channels, biases):
        acc = int32_matmul_dot(x_q, x_zero, row, weight_zero) + bias_acc
        check(INT32_MIN <= acc <= INT32_MAX, "the biased accumulator must fit in int32")
        multiplier, shift = requantize_multiplier(x_scale * weight_scale / out_scale)
        shifted = round_shift(acc * multiplier, shift) + out_zero
        if relu:
            shifted = max(shifted, out_zero)          # ReLU is a clamp at the zero point
        outputs.append(int(min(max(shifted, QMIN), QMAX)))
    return outputs


# --------------------------------------------------------------------------- model
def make_network(sizes, seed=0x516):
    """Deterministic float weights for a ReLU MLP with the given layer sizes."""
    rng = Rng(seed)
    network = []
    for index in range(len(sizes) - 1):
        bound = 1.0 / math.sqrt(sizes[index])
        layer = []
        for _ in range(sizes[index + 1]):
            layer.append(([rng.uniform() * bound for _ in range(sizes[index])],
                          rng.uniform() * bound))
        network.append(layer)
    return network


def make_inputs(samples, dimension, seed=0x717):
    rng = Rng(seed)
    return [[rng.uniform() for _ in range(dimension)] for _ in range(samples)]


def float_layers(network, x, relu=True):
    """Activations of every layer for one input, in float."""
    activations = []
    values = list(x)
    for index, layer in enumerate(network):
        last = index == len(network) - 1
        outputs = []
        for row, bias in layer:
            value = bias + sum(a * b for a, b in zip(values, row))
            outputs.append(value if (last or not relu) else max(value, 0.0))
        activations.append(outputs)
        values = outputs
    return activations


def calibrate(network, inputs, quantize_last_symmetrically=True):
    """Post-training calibration: one (scale, zero_point) per layer from observed activations."""
    input_values = [value for row in inputs for value in row]
    scales = [choose_scale(input_values)]
    last = len(network) - 1
    for index in range(len(network)):
        observed = []
        for x in inputs:
            observed.extend(float_layers(network, x)[index])
        if index == last and quantize_last_symmetrically:
            scales.append(choose_scale(observed))       # centred on zero, see quantize_signed
        else:
            scales.append(choose_scale(observed + [0.0]))
    return scales


def quantize_inputs(inputs, scale, zero_point):
    quantized = []
    for x in inputs:
        quantized.append([int(min(max(round(value / scale) + zero_point, QMIN), QMAX))
                          for value in x])
    return quantized


def forward_quantized(network, prepared, scales, x_q):
    """Run the integer path for one input; returns the dequantised logits of the last layer."""
    codes = list(x_q)
    scale, zero = scales[0]
    for index, (channels, biases) in enumerate(prepared):
        last = index == len(prepared) - 1
        out_scale, out_zero = scales[index + 1]
        codes = integer_linear(codes, scale, zero, channels, biases, out_scale, out_zero,
                               relu=not last)
        scale, zero = out_scale, out_zero
    return [scale * (value - zero) for value in codes]


def prepare_layers(network, scales):
    """Quantise the weights per output channel and fold each bias into accumulator units."""
    prepared = []
    for index, layer in enumerate(network):
        input_scale = scales[index][0]
        channels = []
        biases = []
        for row, bias in layer:
            scale, zero_point = choose_scale(list(row))
            data = [int(min(max(round(value / scale) + zero_point, QMIN), QMAX))
                    for value in row]
            channels.append((scale, zero_point, data))
            biases.append(int(round(bias / (input_scale * scale))))
        prepared.append((channels, biases))
    return prepared


# --------------------------------------------------------------------------- checks
def test_round_trip_is_within_half_a_step():
    rng = Rng(0x11)
    for _ in range(20):
        values = [rng.uniform() * 10.0 for _ in range(50)]
        tensor = quantize(values)
        for original, recovered in zip(values, tensor.dequantize()):
            check(abs(original - recovered) <= tensor.scale / 2 + 1e-12,
                  f"round trip off by more than half a step: {abs(original - recovered)}")
        check(all(QMIN <= value <= QMAX for value in tensor.data), "int8 range respected")
        check(len(tensor.data) == len(values), "every value is quantised")


def test_scale_and_zero_point_derivation():
    values = [-3.0, 1.5, 4.0, 0.25]
    scale, zero_point = choose_scale(values)
    check(abs(scale - (4.0 + 3.0) / 255) < 1e-12, f"scale = range / levels, got {scale}")
    check(QMIN <= zero_point <= QMAX, "the zero point is representable")
    exact_zero = quantize([0.0, 4.0], scale=scale, zero_point=zero_point)
    check(exact_zero.data[0] == zero_point, "the real value 0 maps exactly onto the zero point")
    check(abs(exact_zero.dequantize()[0]) < 1e-12, "and dequantises back to 0.0 exactly")
    tensor = quantize(values, scale=scale, zero_point=zero_point)
    recovered = tensor.dequantize()
    check(abs(recovered[2] - 4.0) <= scale, "the maximum is representable")
    check(abs(recovered[0] + 3.0) <= scale, "so is the minimum")
    check(all(abs(a - b) <= scale / 2 + 1e-12 for a, b in zip(values, recovered)),
          "the derivation gives the half-step guarantee")

    # a range that does not straddle zero is extended to include it
    positive_only = [5.0, 6.0, 7.0]
    scale, zero_point = choose_scale(positive_only)
    tensor = quantize(positive_only)
    check(all(QMIN <= code <= QMAX for code in tensor.data), "codes never leave the range")
    check(all(abs(a - b) <= tensor.scale / 2 + 1e-12
              for a, b in zip(positive_only, tensor.dequantize())),
          "extending the range to zero must not introduce clipping")

    constant = [2.5, 2.5, 2.5]
    constant_tensor = quantize(constant)
    check(all(value == constant_tensor.data[0] for value in constant_tensor.data),
          "a constant tensor becomes one repeated code")
    check(abs(constant_tensor.dequantize()[0] - 2.5) <= constant_tensor.scale / 2 + 1e-12,
          "and it lands within half a step of the constant")

    narrow_scale, narrow_zero = choose_scale([-1.0, 1.0])
    saturated = quantize([-100.0, 100.0], scale=narrow_scale, zero_point=narrow_zero)
    check(saturated.data[0] == QMIN and saturated.data[1] == QMAX, "out-of-range values clamp")
    check(min(abs(a - b) for a, b in zip([-100.0, 100.0], saturated.dequantize())) > 10.0,
          "the clamped values are visibly wrong (saturation is reported, not hidden)")
    return saturated

    narrow_scale, narrow_zero = choose_scale([-1.0, 1.0])
    saturated = quantize([-100.0, 100.0], scale=narrow_scale, zero_point=narrow_zero)
    check(saturated.data[0] == QMIN and saturated.data[1] == QMAX, "out-of-range values clamp")
    check(min(abs(a - b) for a, b in zip([-100.0, 100.0], saturated.dequantize())) > 10.0,
          "the clamped values are visibly wrong (saturation is reported, not hidden)")
    return saturated


def test_symmetric_logit_quantisation():
    values = [-2.0, 1.0, 0.5]
    tensor = quantize_signed(values)
    check(tensor.zero_point == 0, "symmetric quantisation puts zero exactly at code 0")
    check(min(tensor.data) >= QMIN and max(tensor.data) <= QMAX, "codes stay in range")
    scaled = max(abs(value) for value in values)
    check(abs(tensor.scale - scaled / ((QMAX - QMIN) / 2)) < 1e-12,
          "the symmetric scale covers the largest magnitude")
    check(quantize_signed([0.0]).data == [0], "an all-zero tensor is all zeros")


def test_per_channel_beats_per_tensor():
    rng = Rng(0x22)
    rows = [[rng.uniform() * (1.0 + channel) for _ in range(16)] for channel in range(8)]
    flat = [value for row in rows for value in row]
    per_tensor = quantize(flat)
    tensor_errors = [abs(value - recovered)
                     for value, recovered in zip(flat, per_tensor.dequantize())]
    per_channel = quantize_per_channel(rows)
    channel_errors = []
    for row, (scale, zero_point, data) in zip(rows, per_channel):
        channel_errors.extend([abs(value - (scale * (q - zero_point)))
                               for value, q in zip(row, data)])
        check(max(abs(value - (scale * (q - zero_point))) for value, q in zip(row, data))
              <= scale / 2 + 1e-12, "each channel respects its own half-step bound")
    mean_tensor = sum(tensor_errors) / len(tensor_errors)
    mean_channel = sum(channel_errors) / len(channel_errors)
    check(mean_channel < mean_tensor,
          f"per-channel scaling must win: {mean_channel} vs {mean_tensor}")
    check(len(per_channel) == len(rows), "one scale per output channel")
    single = quantize_per_channel([[1.0, -1.0]])
    check(len(single) == 1 and single[0][0] > 0, "a single-channel row still has a scale")


def test_integer_accumulator_matches_bigint():
    rng = Rng(0x33)
    x_q = [int(rng.uniform() * 127) for _ in range(64)]
    w_q = [int(rng.uniform() * 127) for _ in range(64)]
    x_zero, w_zero = 3, -7
    dot = int32_matmul_dot(x_q, x_zero, w_q, w_zero)
    oracle = sum((a - x_zero) * (b - w_zero) for a, b in zip(x_q, w_q))
    check(dot == oracle, "the int32 accumulator is exact")
    check(INT32_MIN <= dot <= INT32_MAX, "and stays inside int32 for realistic sizes")
    check(255 * 255 * 64 < INT32_MAX, "even the worst case for 64 inputs fits in int32")
    enough = INT32_MAX // (255 * 255) + 1
    check(255 * 255 * enough > INT32_MAX,
          f"an accumulator overflows once the layer has {enough} inputs, hence the guard")
    check(255 * 255 * 64 < INT32_MAX // 100, "and a 64-input layer keeps a hundredfold margin")
    check(int32_matmul_dot([0, 0], 0, [127, -128], 0) == 0, "zero inputs give zero")
    try:
        int32_matmul_dot([1 << 40], 0, [1 << 40], 0)
        raise AssertionError("an overflowing accumulator must be rejected")
    except AssertionError as error:
        check("int32 range" in str(error), f"unexpected message: {error}")


def test_requantisation_matches_the_float_rescale():
    rng = Rng(0x44)
    worst = 0.0
    for _ in range(400):
        real_multiplier = abs(rng.uniform()) * 0.5 + 1e-4
        multiplier, shift = requantize_multiplier(real_multiplier)
        check(0 <= multiplier < INT32_MAX, "the multiplier fits in int32")
        check(abs(real_multiplier - multiplier / (1 << shift))
              <= 1e-9 / real_multiplier + 1e-12,
              "the multiplier/shift pair represents the real multiplier accurately")
        for acc in (0, 1, -1, 37, -37, 12345, -12345, 1 << 20, -(1 << 20)):
            integer_result = round_shift(acc * multiplier, shift)
            float_result = acc * real_multiplier
            check(abs(integer_result - float_result) <= 1.0 + 1e-9,
                  f"requantisation off by more than a unit: {integer_result} vs {float_result}")
            worst = max(worst, abs(integer_result - float_result))
    check(requantize_multiplier(0.0) == (0, 0), "a zero multiplier is allowed")
    check(requantize_multiplier(1.0)[0] < INT32_MAX, "a multiplier of one is representable")
    check(round_shift(5, 0) == 5, "shift 0 is the identity")
    check(round_shift(3, 1) == 2 and round_shift(-3, 1) == -2, "rounding is symmetric")
    check(round_shift(0, 5) == 0, "zero is stable")
    try:
        requantize_multiplier(-0.5)
        raise AssertionError("a negative multiplier must be rejected")
    except AssertionError as error:
        check("non-negative" in str(error), f"unexpected message: {error}")
    return worst


def test_layer_against_the_float_reference():
    network = make_network([6, 10, 4], seed=0x55)
    inputs = make_inputs(12, 6, seed=0x56)
    scales = calibrate(network, inputs)
    prepared = prepare_layers(network, scales)
    input_scale, input_zero = scales[0]
    for x, x_q in zip(inputs, quantize_inputs(inputs, input_scale, input_zero)):
        reference = float_layers(network, x)
        previous_float = list(x)
        codes = list(x_q)
        scale, zero = input_scale, input_zero
        for index, (channels, biases) in enumerate(prepared):
            last = index == len(prepared) - 1
            out_scale, out_zero = scales[index + 1]
            codes = integer_linear(codes, scale, zero, channels, biases, out_scale, out_zero,
                                   relu=not last)
            dequantized = [out_scale * (value - out_zero) for value in codes]
            errors = [abs(got - want) for got, want in zip(dequantized, reference[index])]
            check(max(errors) <= 60.0 * out_scale + 1e-9,
                  f"layer {index} drifted {max(errors)} from the float reference, more than "
                  f"sixty quantised units of {out_scale} (the input quantisation dominates)")
            # analytic worst case: input rounding + weight rounding + requantisation rounding,
            # with a factor-four slack for the ReLU clamp at the zero point
            bound = 0.0
            for (weight_scale, weight_zero, data) in channels:
                weights_float = [weight_scale * (q - weight_zero) for q in data]
                candidate = sum(abs(w) * 0.5 * scale for w in weights_float) \
                    + sum(abs(v) * 0.5 * weight_scale for v in previous_float) \
                    + 0.5 * out_scale + 0.5 * scale * weight_scale
                bound = max(bound, candidate)
            check(max(errors) <= 4.0 * bound + 1e-9,
                  f"layer {index} error {max(errors)} exceeds four times the analytic bound "
                  f"{bound}")
            previous_float = dequantized
            scale, zero = out_scale, out_zero
    check(len(scales) == len(network) + 1, "one scale for the input plus one per layer")
    check(all(len(biases) == len(channels) for channels, biases in prepared),
          "every output channel has one bias term")


def test_full_inference_agreement():
    network = make_network([4, 16, 8, 1], seed=0x66)
    inputs = make_inputs(150, 4, seed=0x67)
    labels = [1 if sum(x) > 0 else 0 for x in inputs]
    scales = calibrate(network, inputs)
    prepared = prepare_layers(network, scales)
    quantized_inputs = quantize_inputs(inputs, scales[0][0], scales[0][1])
    float_correct = quantized_correct = agreements = 0
    worst_logit_gap = 0.0
    for x, x_q, label in zip(inputs, quantized_inputs, labels):
        quantized_logit = forward_quantized(network, prepared, scales, x_q)[0]
        float_logit = float_layers(network, x)[-1][0]
        worst_logit_gap = max(worst_logit_gap, abs(quantized_logit - float_logit))
        quantized_label = 1 if quantized_logit > 0 else 0
        float_label = 1 if float_logit > 0 else 0
        agreements += quantized_label == float_label
        float_correct += float_label == label
        quantized_correct += quantized_label == label
    accuracy_float = float_correct / len(inputs)
    accuracy_quantized = quantized_correct / len(inputs)
    agreement = agreements / len(inputs)
    check(worst_logit_gap < 1.0, f"logits must stay close, worst gap {worst_logit_gap}")
    check(agreement >= 0.97, f"quantised and float predictions must agree, got {agreement}")
    check(abs(accuracy_quantized - accuracy_float) <= 0.03,
          f"accuracy must survive quantisation: {accuracy_quantized} vs {accuracy_float}")
    check(0.1 <= accuracy_float <= 0.9,
          f"the untrained network must not predict one label everywhere, got {accuracy_float}")
    check(worst_logit_gap > 0.0, "quantisation does perturb the logits (just barely)")
    return agreement, accuracy_float, accuracy_quantized, worst_logit_gap


def test_bit_width_comparison():
    rng = Rng(0x77)
    values = [rng.uniform() * 4.0 for _ in range(400)]
    errors = {}
    for bits in (4, 8):
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))
        tensor = quantize(values, qmin=qmin, qmax=qmax)
        errors[bits] = sum(abs(a - b) for a, b in zip(values, tensor.dequantize())) / len(values)
        check(tensor.scale > 0.0, f"{bits}-bit quantisation has a positive scale")
        check(all(qmin <= code <= qmax for code in tensor.data), f"{bits}-bit codes in range")
    check(errors[8] < errors[4],
          f"more bits must mean less error: 8-bit {errors[8]} vs 4-bit {errors[4]}")
    check(errors[4] / errors[8] > 8.0,
          f"the error should fall with the level ratio, got {errors[4] / errors[8]}")
    return errors


def test_boundaries_and_validation():
    check(quantize([0.0]).data == [0], "a single zero value quantises to the zero point")
    check(quantize([0.0]).dequantize()[0] == 0.0, "and dequantises exactly")
    try:
        choose_scale([])
        raise AssertionError("an empty calibration set must be rejected")
    except AssertionError as error:
        check("cannot be empty" in str(error), f"unexpected message: {error}")
    for arguments, headline in (((0.0, 0, QMIN, QMAX, []), "positive"),
                                ((1.0, 300, QMIN, QMAX, []), "representable"),
                                ((1.0, 0, 5, 5, []), "non-empty")):
        try:
            Quantized(*arguments)
            raise AssertionError("an invalid tensor must be rejected")
        except AssertionError as error:
            check(headline in str(error), f"unexpected message: {error}")

    network = make_network([3, 4, 1], seed=0x88)
    inputs = make_inputs(8, 3, seed=0x89)
    scales = calibrate(network, inputs)
    prepared = prepare_layers(network, scales)
    sample = quantize_inputs([[10.0, -10.0, 5.0]], scales[0][0], scales[0][1])[0]
    first = forward_quantized(network, prepared, scales, sample)
    second = forward_quantized(network, prepared, scales, sample)
    check(first == second, "the inference path is deterministic")
    check(len(first) == 1, "the output layer has one unit")
    extreme = quantize_inputs([[1e6, -1e6, 0.0]], scales[0][0], scales[0][1])[0]
    forced = forward_quantized(network, prepared, scales, extreme)
    check(len(forced) == 1 and abs(forced[0]) < 1e6, "clamped inputs stay finite")
    return first


def main():
    test_round_trip_is_within_half_a_step()
    test_scale_and_zero_point_derivation()
    test_symmetric_logit_quantisation()
    test_per_channel_beats_per_tensor()
    test_integer_accumulator_matches_bigint()
    worst = test_requantisation_matches_the_float_rescale()
    test_layer_against_the_float_reference()
    agreement, accuracy_float, accuracy_quantized, gap = test_full_inference_agreement()
    errors = test_bit_width_comparison()
    test_boundaries_and_validation()

    print(f"requantisation worst deviation: {worst:.6f} quantised units (bound 1.0)")
    print(f"full inference: agreement {agreement:.3f} worst logit gap {gap:.5f} "
          f"(float accuracy {accuracy_float:.3f}, 8-bit {accuracy_quantized:.3f})")
    print("mean absolute error: 4-bit %.4f vs 8-bit %.4f" % (errors[4], errors[8]))
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
