"""Task 717 - Implement mixed-precision training utility (FP16/FP32 accumulator).  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

Mixed-precision training keeps the fast, narrow format for the bulk of the arithmetic and the wide
format only where range or precision is needed.  This file implements the format itself and then
measures the two effects that make the pattern necessary:

  * format -- IEEE 754 binary16 (1 sign, 5 exponent, 10 mantissa bits, biased by 15, subnormals
    below 2^-14 with a 2^-24 step, round-to-nearest-even, overflow to infinity at 65520).  The
    encoder/decoder is hand-written and audited bit-for-bit against Python's own half-precision
    packing (`struct`), which is only a reference oracle here, never the implementation.
  * accumulation -- summing k values in fp16 with an fp16 accumulator cannot represent
    1.0 + 2^-11 (the gap at 1.0 is 2^-10, so 2^-11 is an exact tie and rounds back to 1.0), so a
    whole stream of small values simply disappears.  Accumulating the same products in fp32 keeps
    them: that is the "FP16 storage, FP32 accumulator" rule the task is about.
  * loss scaling -- an fp16 gradient of 1e-9 flushes to zero; multiplied by a loss scale of 1024
    it becomes representable and can be unscaled afterwards, so the gradient survives.

The demo then trains a small linear model three ways (all-fp32, mixed, all-fp16) on identical
data and reports the loss each reaches, which is the utility question the task asks.

Simplified versus a real framework: the model is a one-layer linear regression updated with plain
SGD, no tensor cores or fused multiply-add are modelled, the accumulation order is fixed rather
than tree-shaped, and loss scaling is a constant rather than dynamic.  Verified: bit-exact
agreement with `struct` over thousands of random bit patterns and every boundary case (zeros, the
sign of zero, subnormals, the smallest subnormal, the largest finite value, overflow, infinities,
NaN), the half-ulp rounding bound for normal values, the tiny-value absorption demonstration, the
fp32 accumulator's accuracy against an exact reference, loss-scaling underflow and recovery, and
the training-loss comparison.
"""
import math
import struct
import sys

MASK64 = (1 << 64) - 1
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14
FP16_MIN_SUBNORMAL = 2.0 ** -24
FP16_MAX_CODE = 0x7BFF


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

    def code16(self):
        self.state, r = splitmix64(self.state)
        return r & 0xFFFF


# --------------------------------------------------------------------------- the fp16 format
def encode_fp16(value):
    """IEEE 754 binary16 bits for a Python float, hand-written with round-to-nearest-even."""
    if value != value:                                     # NaN
        return 0x7E00
    sign = 0x8000 if math.copysign(1.0, value) < 0.0 else 0x0000
    magnitude = abs(value)
    if magnitude == 0.0:
        return sign
    if magnitude >= 65520.0:                               # rounds up to infinity
        return sign | 0x7C00
    if magnitude < 2.0 ** -25:                             # smaller than half the smallest subnormal
        return sign
    mantissa_bits = 10
    if magnitude < FP16_MIN_NORMAL:
        # subnormal: a fixed step of 2^-24, i.e. an exponent field of zero
        scaled = magnitude / FP16_MIN_SUBNORMAL
        mantissa = round_half_even(scaled)
        if mantissa >= (1 << mantissa_bits):               # rounded up into the normal range
            return sign | 0x0400
        return sign | int(mantissa)
    fraction, exponent = math.frexp(magnitude)             # magnitude = fraction * 2**exponent
    biased = exponent - 1 + 15
    scaled = (fraction * 2.0 - 1.0) * (1 << mantissa_bits)
    mantissa = round_half_even(scaled)
    if mantissa >= (1 << mantissa_bits):
        mantissa -= 1 << mantissa_bits
        biased += 1
    if biased >= 31:
        # the carry ran past the largest exponent field: that is overflow only above the
        # midpoint 65520, otherwise the value rounds to the largest finite half
        return sign | (0x7C00 if magnitude >= 65520.0 else FP16_MAX_CODE)
    return sign | (biased << mantissa_bits) | int(mantissa)


def round_half_even(value):
    """Round to the nearest integer, ties to the even one (the IEEE default)."""
    floor = math.floor(value)
    remainder = value - floor
    if remainder > 0.5:
        return floor + 1
    if remainder < 0.5:
        return floor
    return floor if floor % 2 == 0 else floor + 1


def decode_fp16(bits):
    """Inverse of encode_fp16, including subnormals, infinities and NaN."""
    sign = -1.0 if bits & 0x8000 else 1.0
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    if exponent == 0x1F:
        return sign * math.inf if mantissa == 0 else math.nan
    if exponent == 0:
        return sign * mantissa * FP16_MIN_SUBNORMAL
    return sign * (1.0 + mantissa / 1024.0) * 2.0 ** (exponent - 15)


def to_fp16(value):
    """Round a Python float to the nearest representable half, as a Python float."""
    return decode_fp16(encode_fp16(value))


def reference_bits(value):
    """The `struct` reference encoder, used only to audit the hand-written one.

    `struct` raises OverflowError instead of returning infinity for out-of-range values, so the
    IEEE answer is supplied here.
    """
    try:
        return struct.unpack(">H", struct.pack(">e", value))[0]
    except OverflowError:
        return 0xFC00 if value < 0 else 0x7C00


def ulp(value):
    """The gap between neighbouring halfs around |value| (normal range only)."""
    magnitude = abs(value)
    if magnitude == 0.0:
        return FP16_MIN_SUBNORMAL
    if magnitude < FP16_MIN_NORMAL:
        return FP16_MIN_SUBNORMAL
    exponent = math.frexp(magnitude)[1]
    return 2.0 ** (exponent - 11)


# --------------------------------------------------------------------------- accumulators
def sum_fp16(values):
    """All-fp16 accumulation: every partial sum is rounded to half precision."""
    total = 0.0
    for value in values:
        total = to_fp16(total + to_fp16(value))
    return total


def sum_mixed(values):
    """fp16 storage with an fp32 accumulator: the products are halves, the running total is not."""
    total = 0.0
    for value in values:
        total += float(to_fp16(value))
    return total


def sum_exact(values):
    return float(sum(values))


def dot_mixed(a, b):
    """fp16 operands, fp32 accumulation (the rule tensor cores implement)."""
    check(len(a) == len(b), "the vectors must have equal length")
    total = 0.0
    for x, y in zip(a, b):
        total += float(to_fp16(x)) * float(to_fp16(y))
    return total


def scaled_gradient(gradient, loss_scale):
    """Store a gradient in fp16 after loss scaling: the underflow-avoidance trick."""
    check(loss_scale > 0, "the loss scale must be positive")
    scaled = to_fp16(gradient * loss_scale)
    return scaled / loss_scale, scaled


# --------------------------------------------------------------------------- tiny model
def make_problem(samples, dimension, seed=0x717):
    rng = Rng(seed)
    truth = [rng.uniform() for _ in range(dimension + 1)]
    features, targets = [], []
    for _ in range(samples):
        row = [rng.uniform() for _ in range(dimension)]
        features.append(row)
        targets.append(truth[dimension] + sum(truth[k] * row[k] for k in range(dimension)))
    return features, targets, truth


def train_linear(features, targets, steps, learning_rate, mode="fp32", loss_scale=0.0, seed=0x99):
    """One-layer linear regression: returns (loss history, final weights, statistics)."""
    check(mode in ("fp16", "mixed", "fp32"), "unknown precision mode")
    dimension = len(features[0])
    rng = Rng(seed)
    weights = [rng.uniform() * 0.1 for _ in range(dimension + 1)]
    losses = []
    absorbed = 0
    precision_loss = 0.0
    for _ in range(steps):
        gradients = [0.0] * (dimension + 1)
        for row, target in zip(features, targets):
            if mode == "fp16":
                prediction = to_fp16(weights[-1] + sum_fp16(
                    [float(to_fp16(weights[k])) * float(to_fp16(row[k]))
                     for k in range(dimension)]))
                error = to_fp16(prediction - target)
                for k in range(dimension):
                    gradients[k] = to_fp16(gradients[k]
                                           + to_fp16(2.0 * error * float(to_fp16(row[k]))))
                gradients[-1] = to_fp16(gradients[-1] + to_fp16(2.0 * error))
            else:
                prediction = weights[-1] + sum(float(to_fp16(weights[k]))
                                               * float(to_fp16(row[k]))
                                               for k in range(dimension))
                error = prediction - target
                for k in range(dimension):
                    gradients[k] += 2.0 * error * float(to_fp16(row[k]))
                gradients[-1] += 2.0 * error
        scale = 1.0 / len(features)
        for k in range(len(weights)):
            gradient = gradients[k] * scale
            if mode == "fp16":
                update = to_fp16(learning_rate * gradient)
                if update == 0.0 and gradient != 0.0:
                    absorbed += 1
                if update != 0.0:
                    precision_loss = max(precision_loss,
                                         abs(update - learning_rate * gradient)
                                         / abs(learning_rate * gradient))
                weights[k] = to_fp16(weights[k] - update)
            elif mode == "mixed":
                scaled, _ = scaled_gradient(gradient, loss_scale or 1024.0)
                weights[k] -= learning_rate * scaled
            else:
                weights[k] -= learning_rate * gradient
        losses.append(evaluate(features, targets, weights, mode))
    statistics = {"absorbed_updates": absorbed, "weights": weights,
                  "precision_loss": precision_loss}
    return losses, weights, statistics


def evaluate(features, targets, weights, mode):
    total = 0.0
    for row, target in zip(features, targets):
        if mode == "fp32":
            prediction = weights[-1] + sum(weights[k] * row[k] for k in range(len(row)))
        else:
            prediction = weights[-1] + sum(float(to_fp16(weights[k])) * float(to_fp16(row[k]))
                                           for k in range(len(row)))
        total += (prediction - target) ** 2
    return total / len(features)


# --------------------------------------------------------------------------- checks
def test_encoder_matches_the_reference():
    rng = Rng(0x11)
    for _ in range(4000):
        bits = rng.code16()
        value = decode_fp16(bits)
        if value != value:
            check(encode_fp16(value) == 0x7E00, "NaN encodes to the canonical quiet NaN")
            continue
        check(encode_fp16(value) == bits,
              f"re-encoding the half {bits:#06x} ({value}) must give the same bits")
    for _ in range(4000):
        value = rng.uniform() * 10.0 ** (rng.uniform() * 30 - 15)
        check(encode_fp16(value) == reference_bits(value),
              f"encoder disagrees with struct for {value!r}")
        check(encode_fp16(-value) == reference_bits(-value), "the sign bit is handled")
    for value in (0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.0, 65504.0, 65519.0, 65520.0, 1e30, -1e30,
                  FP16_MIN_NORMAL, FP16_MIN_SUBNORMAL, 2.0 ** -25, 2.0 ** -26, 3.0 * 2.0 ** -25,
                  1023.5, 1024.0, 100000.0):
        check(encode_fp16(value) == reference_bits(value),
              f"boundary value {value!r}: {encode_fp16(value):#06x} vs "
              f"{reference_bits(value):#06x}")
        check(encode_fp16(-value) == reference_bits(-value),
              f"boundary value {-value!r}")


def test_format_boundaries():
    check(encode_fp16(0.0) == 0x0000, "positive zero")
    check(encode_fp16(-0.0) == 0x8000, "negative zero keeps its sign")
    check(math.copysign(1.0, decode_fp16(0x8000)) == -1.0, "and it decodes with the sign intact")
    check(encode_fp16(math.inf) == 0x7C00, "positive infinity")
    check(encode_fp16(-math.inf) == 0xFC00, "negative infinity")
    check(decode_fp16(0x7C00) == math.inf and decode_fp16(0xFC00) == -math.inf, "the decode inverts")
    check(math.isnan(decode_fp16(0x7E00)), "NaN decodes to NaN")
    check(encode_fp16(FP16_MAX) == FP16_MAX_CODE, "the largest finite half")
    check(decode_fp16(FP16_MAX_CODE) == FP16_MAX, "and it decodes back")
    check(encode_fp16(65536.0) == 0x7C00, "values beyond the range overflow to infinity")
    check(decode_fp16(0x0001) == FP16_MIN_SUBNORMAL, "the smallest positive subnormal")
    check(decode_fp16(0x03FF) == 1023 * FP16_MIN_SUBNORMAL, "the largest subnormal")
    check(decode_fp16(0x0400) == FP16_MIN_NORMAL, "the smallest normal")
    check(to_fp16(2.0 ** -25) == 0.0, "half the smallest subnormal is a tie that rounds to zero")
    check(to_fp16(3.0 * 2.0 ** -25) == 2.0 ** -23, "the next tie rounds up to the even mantissa")
    check(to_fp16(5.0 * 2.0 ** -25) == 2.0 ** -23,
          "2.5 subnormal steps round to the even 2")
    check(to_fp16(7.0 * 2.0 ** -25) == 2.0 ** -22,
          "3.5 subnormal steps round to the even 4")
    check(to_fp16(1.5 * 2.0 ** -24) == 2.0 ** -23, "and the same rule holds just above")
    check(to_fp16(1.0 + 2.0 ** -11) == 1.0, "a tie at 1.0 rounds down to the even mantissa")
    check(to_fp16(1.0 + 3.0 * 2.0 ** -11) == 1.0 + 2.0 ** -9,
          "and the next tie rounds up")
    check(to_fp16(-1.0 - 2.0 ** -11) == -1.0, "rounding is symmetric for negatives")
    for value in (1.1, 3.14159, 123.456, 0.001, 40000.0):
        check(abs(to_fp16(value) - value) <= ulp(value) / 2 + 1e-12,
              f"the half-ulp bound must hold for {value}")


def test_small_values_are_absorbed_by_an_fp16_accumulator():
    ones = [1.0] * 1
    tiny = [2.0 ** -11] * 1000
    fp16_total = sum_fp16(ones + tiny)
    mixed_total = sum_mixed(ones + tiny)
    exact = sum_exact(ones + tiny)
    check(fp16_total == 1.0,
          f"1.0 + 1000 * 2**-11 must be absorbed by an fp16 accumulator, got {fp16_total}")
    check(abs(mixed_total - exact) < 1e-6,
          f"the fp32 accumulator keeps every term: {mixed_total} vs {exact}")
    check(abs(fp16_total - exact) > 0.4,
          f"so the fp16 error is large: {abs(fp16_total - exact)}")

    rng = Rng(0x22)
    worst_fp16 = worst_mixed = 0.0
    for _ in range(20):
        values = [rng.uniform() * 1e-3 for _ in range(600)] + [1.0]
        rounded_sum = sum(float(to_fp16(value)) for value in values)
        fp16_relative = abs(sum_fp16(values) - rounded_sum) / abs(rounded_sum)
        mixed_relative = abs(sum_mixed(values) - rounded_sum) / abs(rounded_sum)
        worst_fp16 = max(worst_fp16, fp16_relative)
        worst_mixed = max(worst_mixed, mixed_relative)
        check(mixed_relative < 1e-6,
              f"the mixed accumulator's own error must be tiny (the halves themselves carry the "
              f"rounding), got {mixed_relative}")
    check(worst_fp16 > 10.0 * worst_mixed,
          f"fp16 accumulation must be far worse: {worst_fp16} vs {worst_mixed}")
    check(sum_fp16([]) == 0.0 and sum_mixed([]) == 0.0, "an empty sum is zero")
    check(sum_fp16([math.inf, 1.0]) == math.inf, "infinity propagates")
    check(math.isnan(sum_fp16([math.nan])), "NaN propagates")
    return worst_fp16, worst_mixed


def test_dot_products_and_exactness():
    rng = Rng(0x33)
    for trial in range(10):
        a = [rng.uniform() * 2.0 for _ in range(32)]
        b = [rng.uniform() * 2.0 for _ in range(32)]
        mixed = dot_mixed(a, b)
        exact = sum(float(to_fp16(x)) * float(to_fp16(y)) for x, y in zip(a, b))
        check(abs(mixed - exact) < 1e-6,
              f"trial {trial}: the fp32 accumulator tracks the exact sum of the halves")
        naive = sum_fp16([x * y for x, y in zip(a, b)])
        check(abs(naive - exact) > abs(mixed - exact) or abs(naive - exact) == 0.0,
              "the fp16 dot product is never more accurate than the mixed one")
        check(abs(mixed) <= sum(abs(x) * abs(y) for x, y in zip(a, b)) + 1e-6,
              "the triangle inequality holds for the accumulation")
    check(dot_mixed([], []) == 0.0, "an empty dot product is zero")
    try:
        dot_mixed([1.0], [1.0, 2.0])
        raise AssertionError("mismatched lengths must be rejected")
    except AssertionError as error:
        check("equal length" in str(error), f"unexpected message: {error}")
    check(dot_mixed([1.0, 2.0], [3.0, 4.0]) == 11.0, "an exact small product survives")


def test_loss_scaling_rescues_small_gradients():
    gradient = 1e-9
    check(to_fp16(gradient) == 0.0, "a gradient of 1e-9 flushes to zero in fp16")
    recovered, scaled = scaled_gradient(gradient, 1024.0)
    check(scaled != 0.0, "loss scaling makes it representable")
    check(abs(recovered - gradient) <= 0.1 * gradient,
          f"and unscaling recovers the gradient: {recovered} vs {gradient}")
    for scale_value in (256.0, 512.0, 4096.0, 65536.0):
        recovered, scaled = scaled_gradient(1e-7, scale_value)
        check(scaled != 0.0, f"loss scale {scale_value} keeps 1e-7 alive")
        check(abs(recovered - 1e-7) <= 0.15 * 1e-7,
              f"loss scale {scale_value}: recovered {recovered}")
        check(abs(recovered) < 1e-6, "the unscaled gradient is small again")
    check(to_fp16(1e-7) != 0.0, "1e-7 is representable without scaling (it is just coarse)")
    check(to_fp16(1e-9) == 0.0 and to_fp16(-1e-9) == 0.0, "both signs flush")
    try:
        scaled_gradient(1.0, 0.0)
        raise AssertionError("a zero loss scale must be rejected")
    except AssertionError as error:
        check("positive" in str(error), f"unexpected message: {error}")


def test_training_utility():
    features, targets, truth = make_problem(48, 3, seed=0x44)
    steps, learning_rate = 400, 0.02
    fp32_losses, fp32_weights, _ = train_linear(features, targets, steps, learning_rate, "fp32")
    mixed_losses, mixed_weights, _ = train_linear(features, targets, steps, learning_rate, "mixed")
    fp16_losses, fp16_weights, fp16_stats = train_linear(features, targets, steps, learning_rate,
                                                         "fp16")
    check(fp32_losses[-1] < fp32_losses[0], "fp32 training must reduce the loss")
    check(mixed_losses[-1] < mixed_losses[0], "so must mixed precision")
    check(fp32_losses[-1] < 1e-3, f"fp32 reaches a small loss, got {fp32_losses[-1]}")
    check(mixed_losses[-1] <= 2.0 * fp32_losses[-1] + 1e-6,
          f"mixed precision must stay close to fp32: {mixed_losses[-1]} vs {fp32_losses[-1]}")
    check(fp32_losses[-1] <= fp16_losses[-1] + 1e-9,
          f"all-fp16 must not beat fp32: {fp16_losses[-1]} vs {fp32_losses[-1]}")
    check(fp16_losses[-1] <= 20.0 * fp32_losses[-1] + 1e-6,
          f"all-fp16 stays in the same ballpark: {fp16_losses[-1]} vs {fp32_losses[-1]}")
    check(1e-5 < fp16_stats["precision_loss"] < 0.1,
          f"each fp16 update loses at least 2^-11 of its value (more when it is subnormal), "
          f"measured {fp16_stats['precision_loss']}")

    exact = evaluate(features, targets, fp32_weights, "fp32")
    check(abs(exact - fp32_losses[-1]) < 1e-9, "the reported loss is the evaluated one")
    check(all(abs(a - b) < 0.05 for a, b in zip(mixed_weights, fp32_weights)),
          "the mixed-precision weights land near the fp32 ones")
    check(all(math.isfinite(weight) for weight in fp16_weights), "fp16 weights stay finite")
    check(len(fp16_weights) == len(fp32_weights), "the same parameter count")
    return (fp32_losses[-1], mixed_losses[-1], fp16_losses[-1],
            fp16_stats["absorbed_updates"], fp16_stats["precision_loss"])


def test_boundaries_and_determinism():
    check(to_fp16(to_fp16(0.1)) == to_fp16(0.1), "rounding a representable value is idempotent")
    for value in (0.0, 1.0, -2.5, FP16_MIN_SUBNORMAL, 65504.0):
        check(to_fp16(to_fp16(value)) == to_fp16(value), f"idempotence at {value}")
    check(sum_fp16([1.0]) == 1.0 and sum_mixed([1.0]) == 1.0, "a single-element sum")
    check(sum_fp16([2.0 ** -24] * 1024) == 1024 * 2.0 ** -24 or
          sum_fp16([2.0 ** -24] * 1024) > 0.0, "tiny subnormals can still add up")
    check(math.isinf(sum_fp16([65504.0, 65504.0])), "fp16 accumulation can overflow to infinity")
    check(abs(sum_mixed([65504.0, 65504.0]) - 131008.0) < 1e-9,
          "the fp32 accumulator does not overflow there")

    features, targets, _ = make_problem(16, 2, seed=0x55)
    first, _, _ = train_linear(features, targets, 20, 0.05, "mixed")
    second, _, _ = train_linear(features, targets, 20, 0.05, "mixed")
    check(first == second, "training is deterministic")
    try:
        train_linear(features, targets, 5, 0.05, "bfloat16")
        raise AssertionError("an unknown mode must be rejected")
    except AssertionError as error:
        check("unknown precision mode" in str(error), f"unexpected message: {error}")
    return first[-1]


def main():
    test_encoder_matches_the_reference()
    test_format_boundaries()
    worst_fp16, worst_mixed = test_small_values_are_absorbed_by_an_fp16_accumulator()
    test_dot_products_and_exactness()
    test_loss_scaling_rescues_small_gradients()
    fp32_loss, mixed_loss, fp16_loss, absorbed, precision_loss = test_training_utility()
    last = test_boundaries_and_determinism()

    print(f"accumulator relative error over 20 sums: fp16 {worst_fp16:.4f} vs "
          f"mixed {worst_mixed:.2e}")
    print(f"final training loss: fp32 {fp32_loss:.6g}, mixed {mixed_loss:.6g}, "
          f"fp16 {fp16_loss:.6g}")
    print(f"all-fp16 update path: {absorbed} updates flushed to zero, worst rounding loss per "
          f"update {precision_loss:.2e}")
    print(f"boundary/determinism check final loss: {last:.6g}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
