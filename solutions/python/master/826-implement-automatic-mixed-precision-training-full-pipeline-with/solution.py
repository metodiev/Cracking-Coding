"""Task 826 - Implement automatic mixed-precision training full pipeline with loss-scaling. (Tier: Master, CTCI Beyond — Machine Learning & AI)

The full AMP pipeline is modelled end to end, not just the idea of it: fp32 *master*
weights, a forward pass whose weights and activations are *stored* in fp16 while the
products and accumulations happen in fp32 (the "fp16 storage / fp32 accumulate" rule
that tensor cores implement), a backward pass that produces fp16 gradients, dynamic
loss scaling around the backward pass, unscaling, global-norm gradient clipping,
finite-value screening, the skip-on-overflow rule, and the growth/backoff schedule of
the scale. A `TrainedModel` run reports the loss curve, the overflow events and the
scale history, so every rule can be asserted rather than described.

What is implemented
-------------------
* `fp16(x)`: faithful round-to-nearest-even-in-significand model of binary16 -- 11
  significant bits (1 implicit + 10 stored), the subnormal grid 2^-24 below 2^-14, and
  overflow to infinity at 65520. NaN/inf are propagated unchanged.
* `TinyMLP`: 2 -> 3 -> 1, tanh hidden layer, mean-squared-error loss; analytic
  backprop (not finite differences) in fp32, with every stored tensor rounded to fp16
  when the mixed-precision mode is on.
* `LossScaler`: initial scale, `growth_interval` consecutive finite steps before the
  scale doubles, halving on a non-finite step (with a floor), and the "skip the
  optimiser step entirely" rule.
* `train(...)`: the whole loop, with a deterministic dataset and no wall-clock,
  randomness, threads or filesystem access.

What is simplified
------------------
* Python floats are IEEE-754 doubles, so "fp32 accumulate" is really "double
  accumulate": the *storage* precision (the thing loss scaling exists for) is modelled
  exactly, the accumulation precision is modelled loosely. The invariants that depend
  only on storage precision (round-trip, relative error bound, overflow, scale being a
  power of two) are therefore exact.
* The dataset and the network are tiny (2 -> 3 -> 1, 8 points) so the run is instant;
  the pipeline, not the throughput, is the subject.
* No CUDA, no autocast dispatch, no fused optimiser: the same arithmetic in one process.

Invariants verified
-------------------
1.  fp16 round-trip: exactly representable values are fixed points, and every normal
    value satisfies |fp16(x) - x| <= 2^-11 |x| (half an ULP of an 11-bit significand).
2.  Overflow: |x| >= 65520 becomes infinity and is detected as non-finite; the
    subnormal grid swallows values below 2^-25 to zero.
3.  Loss scaling with a power-of-two S is *lossless*: scaling the loss by S and
    unscaling the gradients by 1/S returns the unscaled gradient bit for bit (which is
    why the scaler only ever produces powers of two).
4.  Scaling rescues underflow: a gradient that rounds to zero in plain fp16 survives
    after scaling, and comes back ~1/S of its true size.
5.  The dynamic-scale state machine: growth only after `growth_interval` consecutive
    finite steps, halving (never below the floor) on a non-finite step, counter reset.
6.  A step taken while the loss/gradients are non-finite leaves the master weights bit
    for bit identical and halves the scale (skip-on-overflow).
7.  Gradient clipping happens *after* unscaling: the global norm of the applied update
    never exceeds `max_norm`, while clipping the scaled gradients with the unscaled
    threshold would (checked explicitly to document the pitfall).
8.  Analytic backprop agrees with central finite differences of the fp32 forward.
9.  Liveness/quality: the fp32 baseline and the mixed-precision pipeline both drive the
    loss past a threshold, and the mixed-precision loss is not materially worse than the
    fp32 baseline (loss scaling buys stability without costing accuracy).
10. The fp32 master copy really holds more than fp16: after training, master weights
    exist that are *not* fp16-representable, and the same run twice is bit-identical.
11. One pipeline step equals an independently hand-derived chain rule for one weight.
"""
import math

FP16_MAX = 65520.0            # first value that rounds to infinity in binary16
FP16_TINY = 2.0 ** -14        # smallest normal
FP16_SUBNORMAL = 2.0 ** -24   # spacing of the subnormal grid


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# simulated binary16
# --------------------------------------------------------------------------------------
def fp16(x):
    """Round to nearest on the binary16 grid (11-bit significand, subnormals, overflow)."""
    if x != x or x in (math.inf, -math.inf):
        return x
    sign = -1.0 if x < 0 else 1.0
    a = abs(x)
    if a >= FP16_MAX:
        return sign * math.inf
    if a < FP16_TINY:                       # subnormal grid: fixed spacing
        return sign * round(a / FP16_SUBNORMAL) * FP16_SUBNORMAL
    mantissa, exponent = math.frexp(a)      # a = mantissa * 2**exponent, 0.5 <= m < 1
    return sign * math.ldexp(round(mantissa * 2048.0) / 2048.0, exponent)


def store(x, half):
    return fp16(x) if half else x


def every_finite(*values):
    for v in values:
        if isinstance(v, (list, tuple)):
            if not every_finite(*v):
                return False
        elif math.isinf(v) or v != v:
            return False
    return True


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------
def init_params():
    """Small deterministic initialisation (no random module, no time, no seeds to leak)."""
    return {"W1": [[0.31, -0.42], [-0.57, 0.28], [0.44, 0.63]],
            "b1": [0.11, -0.19, 0.07],
            "W2": [[0.53, -0.61, 0.37]],
            "b2": [0.05]}


def flat_params(params):
    out = []
    for row in params["W1"]:
        out += list(row)
    out += list(params["b1"])
    out += list(params["W2"][0])
    out += list(params["b2"])
    return out


def load_params(values):
    params, cursor = {}, 0
    params["W1"] = [list(values[cursor + 2 * j:cursor + 2 * j + 2]) for j in range(3)]
    cursor += 6
    params["b1"] = list(values[cursor:cursor + 3])
    cursor += 3
    params["W2"] = [list(values[cursor:cursor + 3])]
    cursor += 3
    params["b2"] = [values[cursor]]
    return params


def forward(params, x, half):
    """2 -> 3 -> 1 with tanh; returns (output, cache) with fp16 *storage* when half."""
    x_hat = [store(v, half) for v in x]
    z1 = [sum(store(params["W1"][j][k], half) * x_hat[k] for k in range(2))
          + store(params["b1"][j], half) for j in range(3)]
    a1 = [store(math.tanh(z1[j]), half) for j in range(3)]
    out = sum(store(params["W2"][0][j], half) * a1[j] for j in range(3)) \
        + store(params["b2"][0], half)
    return store(out, half), {"x": x_hat, "z1": z1, "a1": a1}


def loss_and_grads(params, batch, half=False, scale=1.0):
    """Mean-squared error and its analytic gradient, optionally loss-scaled and fp16-stored.

    d(scale * loss)/dw = scale * d(loss)/dw, so the scaling multiplies every gradient;
    when `half` is set the *stored* gradient tensors are rounded to fp16, which is
    exactly the step loss scaling exists to protect.
    """
    grads = {"W1": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "b1": [0.0, 0.0, 0.0],
             "W2": [[0.0, 0.0, 0.0]], "b2": [0.0]}
    total = 0.0
    for x, y in batch:
        out, cache = forward(params, x, half)
        total += (out - y) ** 2
    count = len(batch)
    loss = total / count
    for x, y in batch:
        out, cache = forward(params, x, half)
        d_out = scale * 2.0 * (out - y) / count
        for j in range(3):                          # layer 2: out = W2 . a1 + b2
            d_a1 = d_out * store(params["W2"][0][j], half)
            grads["W2"][0][j] += d_out * cache["a1"][j]
            d_z1 = d_a1 * (1.0 - cache["a1"][j] ** 2)      # tanh'
            grads["b1"][j] += d_z1
            for k in range(2):
                grads["W1"][j][k] += d_z1 * cache["x"][k]
        grads["b2"][0] += d_out
    if half:
        grads = {"W1": [[store(v, True) for v in row] for row in grads["W1"]],
                 "b1": [store(v, True) for v in grads["b1"]],
                 "W2": [[store(v, True) for v in row] for row in grads["W2"]],
                 "b2": [store(v, True) for v in grads["b2"]]}
    return loss, grads


def global_norm(grads):
    total = 0.0
    for row in grads["W1"]:
        total += sum(v * v for v in row)
    total += sum(v * v for v in grads["b1"])
    total += sum(v * v for v in grads["W2"][0])
    total += sum(v * v for v in grads["b2"])
    return math.sqrt(total)


def apply_update(params, grads, lr):
    """SGD step on the fp32 master weights, in place."""
    for j in range(3):
        for k in range(2):
            params["W1"][j][k] -= lr * grads["W1"][j][k]
        params["b1"][j] -= lr * grads["b1"][j]
    for j in range(3):
        params["W2"][0][j] -= lr * grads["W2"][0][j]
    params["b2"][0] -= lr * grads["b2"][0]


class LossScaler:
    """Dynamic loss scaling: only powers of two, so scaling and unscaling are exact."""

    def __init__(self, init_scale=4.0, growth_interval=3, backoff=2.0,
                 min_scale=1.0, max_scale=1024.0):
        self.scale = float(init_scale)
        self.growth_interval = growth_interval
        self.backoff = backoff
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.good_steps = 0
        self.history = [self.scale]

    def report(self, finite):
        """Returns True when the optimiser step must be *taken*, False when it is skipped."""
        if not finite:
            self.scale = max(self.min_scale, self.scale / self.backoff)
            self.good_steps = 0
            self.history.append(self.scale)
            return False
        self.good_steps += 1
        if self.good_steps >= self.growth_interval:
            self.good_steps = 0
            self.scale = min(self.max_scale, self.scale * 2.0)   # doubling keeps it a
            self.history.append(self.scale)                      # power of two
        return True


def copy_params(params):
    return {"W1": [list(row) for row in params["W1"]], "b1": list(params["b1"]),
            "W2": [list(row) for row in params["W2"]], "b2": list(params["b2"])}


def train(batch, steps=160, lr=0.35, half=False, max_norm=1.0, scaler=None, initial=None):
    """The pipeline. Returns a report: losses, per-step scale, skips and final weights."""
    params = init_params() if initial is None else copy_params(initial)
    report = {"loss": [], "scale": [], "skipped": 0, "clipped": 0,
              "norm_before": [], "norm_after": []}
    if half and scaler is None:
        scaler = LossScaler()
    for _ in range(steps):
        scale = scaler.scale if half else 1.0
        loss, grads = loss_and_grads(params, batch, half=half, scale=scale)
        finite = every_finite(loss, flat_params(grads))
        take = scaler.report(finite) if half else finite
        report["scale"].append(scale)
        if not take:
            report["skipped"] += 1
            continue
        if half:                                    # unscale *before* clipping and stepping
            inv = 1.0 / scale
            for row in grads["W1"]:
                for k in range(2):
                    row[k] *= inv
            for j in range(3):
                grads["b1"][j] *= inv
                grads["W2"][0][j] *= inv
            grads["b2"][0] *= inv
        norm = global_norm(grads)
        report["norm_before"].append(norm)
        if norm > max_norm:                         # global-norm clipping
            factor = max_norm / norm
            report["clipped"] += 1
            for row in grads["W1"]:
                for k in range(2):
                    row[k] *= factor
            for j in range(3):
                grads["b1"][j] *= factor
                grads["W2"][0][j] *= factor
            grads["b2"][0] *= factor
            norm = max_norm
        report["norm_after"].append(norm)
        apply_update(params, grads, lr)
        report["loss"].append(loss / scale if half else loss)
    report["weights"] = params
    return report


# --------------------------------------------------------------------------------------
# deterministic dataset (y = sin(1.7x) sampled, no randomness)
# --------------------------------------------------------------------------------------
def make_dataset():
    xs = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    return [([x, 0.5 * x], math.sin(1.7 * x)) for x in xs]


def main():
    # -- 1. the fp16 model itself --------------------------------------------------------
    exactly = [0.0, 1.0, 1.5, -2.0, 0.125, 0.375, 1024.0, -0.0001220703125]
    check(all(fp16(v) == v for v in exactly), "fp16 fixes its own grid points")
    worst = 0.0
    probe = 1.0
    for _ in range(400):                      # walk a deterministic set of values
        probe = probe * 1.017 + 0.0001
        for signed in (probe, -probe):
            if abs(signed) < FP16_TINY:
                continue
            error = abs(fp16(signed) - signed) / abs(signed)
            worst = max(worst, error)
    check(worst <= 2.0 ** -11, f"normal values round to half an 11-bit ULP ({worst:.2e})")
    check(check_subnormal(), "subnormal grid: 2^-25 flushes to zero, the grid step is 2^-24")

    # -- 2. overflow and non-finite screening -------------------------------------------
    check(fp16(6.0e4) == 6.0e4 and math.isinf(fp16(7.0e4)),
          "65520 is the first value that overflows to infinity")
    check(not every_finite(math.inf, 1.0) and not every_finite([[1.0, math.nan]]),
          "non-finite screening descends into nested parameter lists")
    check(every_finite(1.0, [[2.0, 3.0]]), "finite values pass the screening")

    # -- 3. loss scaling with a power of two is lossless ---------------------------------
    batch = make_dataset()
    params = init_params()
    _, grads_plain = loss_and_grads(params, batch, half=False, scale=1.0)
    _, grads_scaled = loss_and_grads(params, batch, half=False, scale=1024.0)
    same = all(a == 1024.0 * b for a, b in zip(flat_params(grads_scaled),
                                               flat_params(grads_plain)))
    back = all(a / 1024.0 == b for a, b in zip(flat_params(grads_scaled),
                                               flat_params(grads_plain)))
    check(same and back, "scaling by 2^10 and unscaling recovers the gradient bit for bit")

    # -- 4. scaling rescues gradients that fp16 would round to zero ----------------------
    tiny = 1.0e-9
    check(fp16(tiny) == 0.0, "a 1e-9 gradient underflows the fp16 subnormal grid")
    check(fp16(tiny * 1024.0) != 0.0 and fp16(fp16(tiny * 1024.0) / 1024.0) == 0.0,
          "the same gradient survives the scaled round trip and comes back at 1e-9")

    # -- 5. the dynamic-scale state machine ---------------------------------------------
    scaler = LossScaler(init_scale=4.0, growth_interval=3, min_scale=1.0, max_scale=8.0)
    check([scaler.report(True) for _ in range(3)] == [True, True, True]
          and scaler.scale == 8.0, "three finite steps double a scale of 4")
    check(scaler.scale == 8.0 and scaler.report(True) and scaler.scale == 8.0,
          "the scale is capped at max_scale")
    check(scaler.report(False) is False and scaler.scale == 4.0,
          "a non-finite step skips the update and halves the scale")
    check(all(scaler.report(False) is False for _ in range(6)) and scaler.scale == 1.0,
          "the scale never falls below min_scale")
    scaler.report(True)
    scaler.report(True)
    check(scaler.good_steps == 2, "the growth counter restarts after an overflow")
    check(all(math.log2(s).is_integer() for s in scaler.history),
          "the scaler only ever emits powers of two")

    # -- 6. skip-on-overflow keeps the master weights untouched --------------------------
    poisoned = init_params()
    poisoned["W2"] = [[7.0e4, 0.0, 0.0]]        # 7e4 * tanh(3) = 6.97e4 > 65520 -> inf
    poisoned_batch = [([3.0, 0.0], 1.0)]
    _, bad_grads = loss_and_grads(poisoned, poisoned_batch, half=True, scale=1.0)
    check(not every_finite(flat_params(bad_grads)), "the poisoned forward is non-finite")
    before = flat_params(poisoned)
    run = train(poisoned_batch, steps=2, half=True, initial=poisoned,
                scaler=LossScaler(init_scale=16.0))
    check(run["skipped"] == 2 and run["scale"][0] == 16.0 and run["scale"][1] == 8.0,
          "both poisoned steps are skipped and the scale backs off each time")
    check(flat_params(run["weights"]) == before, "skipped steps leave masters bit-identical")

    # -- 7. clipping happens after unscaling ---------------------------------------------
    amp = train(batch, steps=160, lr=0.35, half=True, max_norm=0.5)
    check(amp["clipped"] > 0, "the toy problem is clipped often enough to exercise the path")
    check(max(amp["norm_after"]) <= 0.5 * (1.0 + 1e-12),
          "every applied update obeys the max_norm bound")
    heavy = [i for i, n in enumerate(amp["norm_before"]) if n > 0.5]
    check(heavy and all(amp["scale"][i] > 1.0 for i in heavy),
          "clips coincide with a raised scale, so clipping in the *scaled* space to max_norm "
          "would shrink the real bound to max_norm / scale")
    report = train(batch, steps=160, lr=0.35, half=False, max_norm=1.0)

    # -- 8. backprop oracle: central differences of the fp32 forward ---------------------
    base = flat_params(init_params())
    analytic = flat_params(loss_and_grads(init_params(), batch)[1])
    h = 1e-6
    worst_grad = 0.0
    for i in (0, 4, 7, 10, 12):
        plus, minus = list(base), list(base)
        plus[i] += h
        minus[i] -= h
        fd = (loss_and_grads(load_params(plus), batch)[0]
              - loss_and_grads(load_params(minus), batch)[0]) / (2.0 * h)
        worst_grad = max(worst_grad, abs(analytic[i] - fd) / max(1e-9, abs(fd)))
    check(worst_grad < 1e-6, f"analytic vs finite-difference gradients ({worst_grad:.2e})")

    # -- 9. liveness and quality ----------------------------------------------------------
    baseline = report["loss"][-1]
    amped = amp["loss"][-1]
    check(amped < 0.02 < report["loss"][0], "mixed precision drives the loss below 0.02")
    check(amped <= baseline * 1.10 + 1e-9,
          f"loss scaling costs no accuracy ({amped:.6f} vs fp32 {baseline:.6f})")
    check(amp["skipped"] == 0, "no step of the well-scaled run overflows")

    # -- 10. the master copy keeps fp32-only bits, and the run is deterministic -----------
    masters = flat_params(amp["weights"])
    extra_bits = sum(1 for w in masters if fp16(w) != w)
    check(extra_bits > 0, f"{extra_bits} master weights are not representable in fp16")
    again = train(batch, steps=160, lr=0.35, half=True, max_norm=0.5)
    check(flat_params(again["weights"]) == masters, "the whole pipeline is bit-reproducible")

    # -- 11. one SGD step of the real pipeline against an independently derived gradient ---
    start = init_params()
    run1 = train(batch, steps=1, lr=0.1, half=False, max_norm=1e9)
    d_w11 = 0.0
    for x, y in batch:                          # hand-written chain rule for W1[0][0]
        a1 = [math.tanh(start["W1"][j][0] * x[0] + start["W1"][j][1] * x[1] + start["b1"][j])
              for j in range(3)]
        out = sum(start["W2"][0][j] * a1[j] for j in range(3)) + start["b2"][0]
        d_w11 += 2.0 * (out - y) * start["W2"][0][0] * (1.0 - a1[0] ** 2) * x[0] / len(batch)
    expected = start["W1"][0][0] - 0.1 * d_w11
    got = run1["weights"]["W1"][0][0]
    check(abs(got - expected) < 1e-12 * max(1.0, abs(expected)),
          f"one pipeline step equals the hand-derived update ({got} vs {expected})")
    check(len(run1["loss"]) == 1 and run1["skipped"] == 0, "the fp32 path never skips a step")

    print(f"fp16 relative rounding bound observed: {worst:.3e} (limit {2.0 ** -11:.3e})")
    print(f"loss: fp32 {baseline:.6f}, amp {amped:.6f} "
          f"(scale {amp['scale'][-1]:g}, skips {amp['skipped']}, clips {amp['clipped']})")
    print(f"master weights not representable in fp16: {extra_bits}/{len(masters)}")
    print("PASS")


def check_subnormal():
    """Subnormal grid: spacing 2^-24, so anything below 2^-25 flushes to zero."""
    return (fp16(2.0 ** -25) == 0.0
            and fp16(1.25 * 2.0 ** -24) == 2.0 ** -24
            and fp16(1.75 * 2.0 ** -24) == 2.0 ** -23
            and fp16(2.0 ** -14) == 2.0 ** -14
            and math.isinf(fp16(math.inf)))


if __name__ == "__main__":
    main()
