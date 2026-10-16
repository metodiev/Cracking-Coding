"""Task 932 - Implement differential testing between CPU/GPU implementations for numerical ops.  (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)

A differential-testing harness that compares a float64 CPU reference against a float32, blocked
"device" implementation of the same numerical kernels, and - the point of the exercise - catches
injected kernel bugs without flagging correct kernels.

* Kernels: batched matmul, row softmax and layer normalisation.  The device versions round every
  intermediate to float32 and accumulate in a different order (blocked tiles, reversed reductions),
  which is exactly how a GPU kernel differs from a scalar CPU loop.
* Metrics: maximum absolute and relative error, plus the ULP distance between the two float32
  results, compared against per-op tolerances calibrated on the well-scaled cases.
* Mutation testing: four plausible kernel bugs are injected one at a time - a dropped tail tile, a
  softmax normalised by the element count instead of the sum, a layer norm that divides by n
  instead of n - 1 for the sample variance, and an accumulator that overwrites instead of adding.
  The harness must flag all four and accept every correct kernel.

Verified: the float64 reference agrees with an exact fsum-based oracle; all correct kernels pass
their tolerances with small ULP distances; every injected bug is detected with an error far above
tolerance and a large ULP distance; softmax rows sum to one, the identity matmul is the identity and
layer-norm output has zero mean and unit variance; zero-sized, single-element and k = 1 shapes are
handled; a cancellation-heavy case shows the expected growth of the float32 error; and the harness
is deterministic.  Simplified: pure Python arithmetic with explicit float32 rounding, dense shapes.
"""
import math
import struct
from itertools import product

TOLERANCE = {"matmul": (2e-5, 2e-4), "softmax": (1e-6, 1e-5), "layernorm": (2e-5, 2e-4)}
BAD_TILE, BAD_SOFTMAX, BAD_NORM, BAD_ACCUMULATOR = "tile", "softmax", "norm", "accumulator"

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def f32(value):
    """Round a Python float to the nearest float32, the way a device register would."""
    return struct.unpack("f", struct.pack("f", value))[0]

def ulp_distance(left, right):
    """Distance in float32 bit patterns: the natural metric for values that should be identical."""
    def bits(value):
        return struct.unpack("I", struct.pack("f", f32(value)))[0]
    low, high = sorted((bits(left), bits(right)))
    return high - low


# ---------------------------------------------------------------- CPU references (float64)
def cpu_matmul(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]

def exact_matmul(left, right):
    """Oracle: exact accumulation with math.fsum, independent of the reference's own ordering."""
    return [[math.fsum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]

def cpu_softmax(matrix):
    out = []
    for row in matrix:
        peak = max(row)
        exponentials = [math.exp(value - peak) for value in row]
        total = sum(exponentials)
        out.append([value / total for value in exponentials])
    return out

def cpu_layernorm(matrix):
    out = []
    for row in matrix:
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / (len(row) - 1)
        out.append([(value - mean) / math.sqrt(variance) for value in row])
    return out


# ------------------------------------------------------------- device kernels (float32)
def device_matmul(left, right, block=2, bug=None):
    if not left or not right or not right[0]:
        return []
    rows, inner, columns = len(left), len(right), len(right[0])
    out = [[0.0] * columns for _ in range(rows)]
    for i0 in range(0, rows, block):
        for j0 in range(0, columns, block):
            limit = inner - (1 if bug == BAD_TILE else 0)      # bug: the tail tile is dropped
            for i in range(i0, min(i0 + block, rows)):
                for j in range(j0, min(j0 + block, columns)):
                    accumulator = 0.0
                    for k in range(limit):
                        term = f32(left[i][k]) * f32(right[k][j])
                        accumulator = (f32(term) if bug == BAD_ACCUMULATOR
                                       else f32(accumulator + term))
                    out[i][j] = f32(accumulator)
    return out

def device_softmax(matrix, bug=None):
    out = []
    for row in matrix:
        peak = max(row)
        exponentials = [f32(math.exp(f32(value - peak))) for value in row]
        total = 0.0
        for value in reversed(exponentials):                   # different reduction order
            total = f32(total + value)
        divisor = float(len(row)) if bug == BAD_SOFTMAX else total
        out.append([f32(value / divisor) for value in exponentials])
    return out

def device_layernorm(matrix, bug=None):
    out = []
    for row in matrix:
        total = 0.0
        for value in row:                                      # different pass order
            total = f32(total + f32(value))
        mean = f32(total / len(row))
        squared = 0.0
        for value in reversed(row):
            squared = f32(squared + f32(f32(value - mean) ** 2))
        divisor = float(len(row)) if bug == BAD_NORM else float(len(row) - 1)
        deviation = f32(math.sqrt(f32(squared / divisor)))
        out.append([f32(f32(value - mean) / deviation) for value in row])
    return out


# ------------------------------------------------------------------------- the harness
def compare(name, expected, actual):
    """Return (max absolute error, max relative error, max ULP distance) for two tensors."""
    absolute = relative = 0.0
    ulps = 0
    for expected_row, actual_row in zip(expected, actual):
        for a, b in zip(expected_row, actual_row):
            absolute = max(absolute, abs(a - b))
            relative = max(relative, abs(a - b) / max(abs(a), 1e-12))
            ulps = max(ulps, ulp_distance(a, b))
    return absolute, relative, ulps

def within(name, expected, actual):
    absolute, relative, ulps = compare(name, expected, actual)
    limit_abs, limit_rel = TOLERANCE[name]
    return absolute <= limit_abs and relative <= limit_rel, (absolute, relative, ulps)

def make_matrix(rows, columns, scale=1.0, state=0x4242):
    values, state = [], state
    for _ in range(rows):
        row = []
        for _ in range(columns):
            state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
            row.append(round(((state >> 11) / 9007199254740992.0 - 0.5) * 2.0 * scale, 6))
        values.append(row)
    return values

def main():
    left = make_matrix(5, 3, 1.5)
    right = make_matrix(3, 4, 1.2)
    batch = make_matrix(4, 5, 1.0)

    # 1. The float64 reference agrees with the exact fsum oracle (the harness's ground truth).
    reference = cpu_matmul(left, right)
    exact = exact_matmul(left, right)
    check(compare("matmul", exact, reference)[0] < 1e-12,
          "the float64 reference must match the exact accumulation")

    # 2. Correct device kernels pass: no false positives in the differential harness.
    accepted = {
        "matmul": within("matmul", reference, device_matmul(left, right))[0],
        "softmax": within("softmax", cpu_softmax(batch), device_softmax(batch))[0],
        "layernorm": within("layernorm", cpu_layernorm(batch), device_layernorm(batch))[0],
    }
    check(all(accepted.values()), "every correct kernel must the pass the differential harness")

    # 3. Mutation testing: all four injected bugs are caught.
    mutations = {
        BAD_TILE: within("matmul", reference, device_matmul(left, right, bug=BAD_TILE))[0],
        BAD_SOFTMAX: within("softmax", cpu_softmax(batch), device_softmax(batch, bug=BAD_SOFTMAX))[0],
        BAD_NORM: within("layernorm", cpu_layernorm(batch), device_layernorm(batch, bug=BAD_NORM))[0],
        BAD_ACCUMULATOR: within("matmul", reference,
                                device_matmul(left, right, bug=BAD_ACCUMULATOR))[0],
    }
    check(not any(mutations.values()), "every injected kernel bug must be flagged as a divergence")
    check(len(mutations) == 4 and not mutations[BAD_TILE],
          "the dropped tail tile is the mutation the harness must never miss")

    # 4. The ULP metric: tiny for correct kernels, enormous for bugs.
    _, _, good_ulps = compare("matmul", reference, device_matmul(left, right))
    _, _, bad_ulps = compare("matmul", reference, device_matmul(left, right, bug=BAD_TILE))
    check(good_ulps < 1 << 20 and bad_ulps > good_ulps * 100,
          "the ULP distance must separate a rounding difference from a real bug")

    # 5. Algebraic properties of the kernels: softmax rows sum to one, norms have zero mean.
    softmax = device_softmax(batch)
    check(all(abs(sum(row) - 1.0) < 1e-6 for row in softmax), "every softmax row must sum to one")
    normed = device_layernorm(batch)
    check(all(abs(sum(row) / len(row)) < 1e-6 for row in normed)
          and all(abs(sum(value * value for value in row) / (len(row) - 1) - 1.0) < 1e-4
                  for row in normed),
          "layer-normed rows must have zero mean and (sample) unit variance")
    identity = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    check(compare("matmul", left, device_matmul(left, identity))[0] < 1e-6,
          "multiplying by the identity must return the input")

    # 6. Boundary shapes: k = 1, single element and empty operands.
    check(compare("matmul", cpu_matmul(left, [[1.0], [2.0], [3.0]]),
                  device_matmul(left, [[1.0], [2.0], [3.0]]))[0] < 1e-5,
          "a rank-one contraction must agree")
    check(device_matmul([], []) == [] and device_softmax([]) == [] and device_layernorm([]) == [],
          "empty operands must return empty results instead of raising")
    single = [[0.25]]
    check(within("softmax", cpu_softmax(single), device_softmax(single))[0]
          and within("layernorm", cpu_layernorm(batch[:1]), device_layernorm(batch[:1]))[0],
          "single-element and single-row shapes must be handled")

    # 7. Cancellation: a badly scaled case loses low-order bits in float32 and is flagged for
    #    triage, while the same kernel on well-scaled data stays comfortably inside tolerance.
    big = [[1e6, 0.01, -1e6, 0.02], [2.0, 3.0, 4.0, 5.0]]
    small = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
    weights = [[1.0], [1.0], [1.0], [1.0]]
    scaled_error = compare("matmul", cpu_matmul(big, weights), device_matmul(big, weights))[0]
    tight_error = compare("matmul", cpu_matmul(small, weights), device_matmul(small, weights))[0]
    check(scaled_error > tight_error and scaled_error < 0.1,
          "cancellation must inflate the float32 error by orders of magnitude")
    check(not within("matmul", cpu_matmul(big, weights), device_matmul(big, weights))[0]
          and within("matmul", cpu_matmul(small, weights), device_matmul(small, weights))[0],
          "the harness must flag the cancellation case for triage and pass the well-scaled one")

    # 8. Determinism and a stable report across runs.
    report = {name: compare(name, expected, actual) for name, expected, actual in (
        ("matmul", reference, device_matmul(left, right)),
        ("softmax", cpu_softmax(batch), device_softmax(batch)),
        ("layernorm", cpu_layernorm(batch), device_layernorm(batch)))}
    again = {name: compare(name, expected, actual) for name, expected, actual in (
        ("matmul", reference, device_matmul(left, right)),
        ("softmax", cpu_softmax(batch), device_softmax(batch)),
        ("layernorm", cpu_layernorm(batch), device_layernorm(batch)))}
    check(report == again, "the harness must report identical numbers on every run")

    print("correct kernels (abs, rel, ulp): " + ", ".join(
        f"{name}={tuple(round(value, 8) if isinstance(value, float) else value for value in data)}"
        for name, data in report.items()))
    print(f"mutations caught {sum(1 for hit in mutations.values() if not hit)}/4, "
          f"tile-bug ULP distance {bad_ulps}, cancellation error {scaled_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
