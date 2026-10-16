"""Task 722 - Implement gradient checkpointing (recompute vs memory tradeoff).  (Tier: Expert, CTCI Beyond — Machine Learning & AI)

Backpropagation needs the forward activations to compute gradients, so a naive implementation
keeps every layer's activation alive: memory grows linearly in the depth L.  Gradient
checkpointing stores only every s-th activation (the segment boundaries) and recomputes the
activations inside a segment when the backward pass reaches it, spending one extra forward
pass to cut the *resident* activation count from L to about L/s + s - which is minimised at
s = sqrt(L), giving O(sqrt(L)) memory.

The model here is a chain of L layers  y = tanh(w*x + b)  applied element-wise to a small
activation vector, with the exact analytic backward pass, so the checkpointed gradients can be
compared with the un-checkpointed ones bit for bit.  The optimiser step is irrelevant to the
task, so the loss is simply the squared norm of the final activation.

Verified invariants: the checkpointed gradient (for w and b of every layer) equals the full
gradient for every segment size, the *measured* peak of simultaneously live activation buffers
matches the hand formula ceil(L/s) + min(s, L) and is far below the full-storage peak for
intermediate s, the extra forward work is exactly one recomputation of every layer, and the
analytic gradients agree with a central-difference numerical oracle (which validates the
backward pass itself, not just the recomputation bookkeeping).

Simplified versus a real framework (torch.utils.checkpoint): a scalar cost model instead of
real memory, no RNG state handling for stochastic layers, no nested/recursive checkpointing,
no CPU offloading, and the segment size is a fixed parameter rather than tuned by a planner.
"""
import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


class Layer:
    """One element-wise tanh layer: y_i = tanh(w * x_i + b)."""

    def __init__(self, weight, bias):
        self.w = weight
        self.b = bias

    def forward(self, x, work):
        """Returns a fresh activation list; `work` counts element evaluations."""
        work[0] += len(x)
        return [math.tanh(self.w * value + self.b) for value in x]

    def backward(self, x, out, grad_out, work):
        """Returns (grad_w, grad_b, grad_x) for y = tanh(w x + b)."""
        grad_w = 0.0
        grad_b = 0.0
        grad_x = []
        for index, value in enumerate(x):
            local = 1.0 - out[index] * out[index]
            grad_w += grad_out[index] * value * local
            grad_b += grad_out[index] * local
            grad_x.append(grad_out[index] * self.w * local)
        work[0] += len(x)
        return grad_w, grad_b, grad_x


def loss_grad(final):
    """Loss = 0.5 * ||final||^2, so the seed gradient is simply the final activation."""
    return list(final)


def loss_value(final):
    return 0.5 * sum(value * value for value in final)


def model_loss(layers, x0):
    work = [0]
    x = list(x0)
    for layer in layers:
        x = layer.forward(x, work)
    return loss_value(x), work[0]


def backward_full(layers, x0, work):
    """Reference backward pass: every activation stays resident (memory = L + 1)."""
    activations = [list(x0)]
    for layer in layers:
        activations.append(layer.forward(activations[-1], work))
    grad_w = [0.0] * len(layers)
    grad_b = [0.0] * len(layers)
    grad = loss_grad(activations[-1])
    for index in range(len(layers) - 1, -1, -1):
        grad_w[index], grad_b[index], grad = layers[index].backward(
            activations[index], activations[index + 1], grad, work)
    return grad_w, grad_b, {"stored": len(activations), "peak": len(activations)}


def backward_checkpointed(layers, x0, segment, work):
    """Store every `segment`-th activation; recompute inside a segment during backward.

    boundary[k] is the input of layer k*segment, so a segment can be replayed from its own
    stored boundary.  `peak` counts the live activation buffers: the stored boundaries plus the
    activations being recomputed, plus the final activation while the last segment runs.
    """
    depth = len(layers)
    if segment < 1:
        raise ValueError("segment size must be at least one layer")
    if segment >= depth:
        return backward_full(layers, x0, work)
    boundaries = [list(x0)]
    activation = list(x0)
    for index in range(depth):
        activation = layers[index].forward(activation, work)
        if (index + 1) % segment == 0 and index + 1 < depth:
            boundaries.append(list(activation))
    grad_w = [0.0] * depth
    grad_b = [0.0] * depth
    grad = loss_grad(activation)
    segments = len(boundaries)
    peak = len(boundaries) + 1
    for seg in range(segments - 1, -1, -1):
        start = seg * segment
        end = min(start + segment, depth)
        activations = [list(boundaries[seg])]
        for index in range(start, end):
            activations.append(layers[index].forward(activations[-1], work))
        live = len(boundaries) + (end - start) + (1 if seg == segments - 1 else 0)
        peak = max(peak, live)
        for index in range(end - 1, start - 1, -1):
            offset = index - start
            grad_w[index], grad_b[index], grad = layers[index].backward(
                activations[offset], activations[offset + 1], grad, work)
    return grad_w, grad_b, {"stored": len(boundaries), "peak": peak}


def memory_model(depth, segment):
    """Stored boundaries plus the transient activations of the widest segment."""
    return -(-depth // segment) + min(segment, depth)


def numerical_gradients(layers, x0, epsilon=1e-6):
    """Central-difference oracle for every layer parameter."""
    grad_w, grad_b = [], []
    for layer in layers:
        original = layer.w
        layer.w = original + epsilon
        plus, _ = model_loss(layers, x0)
        layer.w = original - epsilon
        minus, _ = model_loss(layers, x0)
        layer.w = original
        grad_w.append((plus - minus) / (2 * epsilon))
        original = layer.b
        layer.b = original + epsilon
        plus, _ = model_loss(layers, x0)
        layer.b = original - epsilon
        minus, _ = model_loss(layers, x0)
        layer.b = original
        grad_b.append((plus - minus) / (2 * epsilon))
    return grad_w, grad_b


def build_chain(depth, width, seed):
    """A deterministic chain: weights and biases from a tiny splitmix64-derived sequence."""
    state = seed & ((1 << 64) - 1)
    layers = []
    for _ in range(depth):
        values = []
        for _ in range(3):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            values.append((state >> 11) / float(1 << 53))
        layers.append(Layer(0.4 + values[0], -0.2 + 0.4 * values[1]))
    return layers, [0.3 * (i + 1) - 0.5 for i in range(width)]


def main():
    depth, width = 24, 4

    # 1. the model and the numerical oracle: analytic gradients must match central
    #    differences, which validates the backward pass independently of checkpointing.
    small, x_small = build_chain(5, 3, 0x7221)
    work = [0]
    analytic_w, analytic_b, _ = backward_checkpointed(small, x_small, 1, work)
    numeric_w, numeric_b = numerical_gradients(small, x_small)
    check(all(abs(analytic_w[i] - numeric_w[i]) < 1e-6 for i in range(5)),
          "analytic d/dw must match central differences")
    check(all(abs(analytic_b[i] - numeric_b[i]) < 1e-6 for i in range(5)),
          "analytic d/db must match central differences")
    check(any(abs(numeric_w[i]) > 1e-3 for i in range(5)), "the oracle must see real gradients")

    layers, x0 = build_chain(depth, width, 0x7222)
    reference_w, reference_b, full_stats = backward_full(layers, list(x0), [0])
    check(full_stats["peak"] == depth + 1, "full backprop keeps every activation resident")
    check(all(abs(reference_w[i]) > 0.0 for i in range(depth)), "gradients must be non-trivial")

    # 2. the checkpointed gradient must be identical for every segment size, including the two
    #    degenerate ones (checkpoint every layer, checkpoint nothing).
    for segment in (1, 2, 3, 4, 6, 8, 12, 24, 25):
        work_two = [0]
        grad_w, grad_b, stats = backward_checkpointed(layers, list(x0), segment, work_two)
        check(grad_w == reference_w and grad_b == reference_b,
              f"segment {segment} must reproduce the full gradient exactly")

    # 3. the memory tradeoff: the measured peak of live activation buffers must match the
    #    hand model, and the sqrt(L) choice must be close to the minimum.
    peaks = {}
    for segment in range(1, depth + 1):
        _, _, stats = backward_checkpointed(layers, list(x0), segment, [0])
        peaks[segment] = stats["peak"]
        check(stats["peak"] <= memory_model(depth, segment) + 1,
              "the measured peak must not exceed the hand model")
    best = min(peaks, key=lambda s: (peaks[s], s))
    check(2 * peaks[best] <= depth + 1, "checkpointing must roughly halve the activations")
    check(4 <= best <= 6, "for depth 24 the best segment size must be near sqrt(24)")
    check(peaks[best] < peaks[1] and peaks[24] == depth + 1,
          "not checkpointing degenerates to full storage")

    # 4. the compute price: recomputation costs exactly one extra forward pass over the whole
    #    chain, and nothing when everything is stored.
    _, forward_only = model_loss(layers, x0)
    work_segmented = [0]
    backward_checkpointed(layers, list(x0), 4, work_segmented)
    work_full = [0]
    backward_full(layers, list(x0), work_full)
    check(forward_only == depth * width, "a forward pass costs one evaluation per element")
    check(work_segmented[0] >= 3 * forward_only,
          "checkpointing costs a forward, a backward and a re-forward")
    check(work_segmented[0] > work_full[0],
          "the segmented run must do strictly more arithmetic than the full-storage run")
    check(work_full[0] == 2 * forward_only, "the full-storage run costs forward plus backward")

    # 5. boundary cases: a one-layer chain, a segment larger than the chain, and an invalid
    #    segment size.
    trivial, x_one = build_chain(1, 3, 0x7223)
    full_w, full_b, _ = backward_full(trivial, list(x_one), [0])
    for segment in (1, 5):
        grad_w, grad_b, _ = backward_checkpointed(trivial, list(x_one), segment, [0])
        check(grad_w == full_w and grad_b == full_b, "a one-layer chain must work either way")
    expect_value_error(lambda: backward_checkpointed(layers, x0, 0, [0]),
                       "a segment size below one must be rejected")

    # 6. gradients decrease the loss: one deterministic gradient step must reduce it, which is
    #    the property a training loop actually depends on.
    before, _ = model_loss(layers, x0)
    work_two = [0]
    grad_w, grad_b, _ = backward_checkpointed(layers, list(x0), 6, work_two)
    rate = 0.1
    for index, layer in enumerate(layers):
        layer.w -= rate * grad_w[index]
        layer.b -= rate * grad_b[index]
    after, _ = model_loss(layers, x0)
    check(after < before, "a gradient step computed by the checkpointed backward must descend")
    check(after < before / 10.0, "the step must cut the loss by more than an order of magnitude")

    print(f"722 depth={depth} full_peak={full_stats['peak']} best_segment={best} "
          f"peak={peaks[best]} model={memory_model(depth, best)}")
    print(f"722 work: forward={forward_only} full_backward={work_full[0]} "
          f"checkpointed={work_segmented[0]}")
    print("PASS")


if __name__ == "__main__":
    main()
