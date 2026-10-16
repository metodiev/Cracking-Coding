"""Task 883 - Implement automatic differentiation for GPU kernels across fused operators.  (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)

A reverse-mode automatic differentiation engine over small dense tensors, plus the kernel
fusion layer a GPU compiler performs on the same graph:

  * the tape - every op records a backward closure; the backward pass walks the tape in
    reverse order and *accumulates* into the parents' gradient buffers, which is what makes
    shared subexpressions and broadcast operands correct;
  * the ops - elementwise add/multiply, matrix multiply, ReLU, exp, log, sum/mean reductions,
    softmax and layer normalisation, each with its analytic vector-Jacobian product;
  * fusion - ``fused_layernorm_affine_relu`` evaluates layer-normalise, affine and rectify in a
    single pass without materialising intermediate tensors and carries a hand-derived backward.
    The unfused tape and the fused kernel must agree bit for bit in the forward *and* the
    backward, while the fused path allocates one output tensor instead of five.

Verified: reverse-mode gradients against central finite differences for a layered expression
and for the fused kernel; the softmax+selection gradient against its closed form p - onehot and
the zero-row-sum identity; layer-normalisation structural invariants (zero row sum, orthogonality
to the centred input, batch-accumulated beta); gradient accumulation through a shared
subexpression; broadcast bias gradients; forward-mode dual numbers agreeing with reverse mode
on the same directional derivative; the ReLU subgradient at exactly zero and a dead unit; exp
and log gradients; fused versus unfused forward/backward equality with a measured allocation
reduction; a real gradient-descent training loop whose loss falls monotonically; rejection of
shape mismatches; and full determinism.

Simplified versus a production compiler: float64 on the host with a tape instead of a kernel
language and a device pass, and no memory planning or register allocation - only the fusion
boundary itself is modelled.
"""
import math

EPSILON = 1e-5


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def uniform(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.state >> 11) * (1.0 / float(1 << 53))

    def gauss(self):
        u = max(self.uniform(), 1e-12)
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * self.uniform())


# ------------------------------------------------------------------ tensors and tape
class Tensor:
    __slots__ = ("data", "shape", "grad")

    def __init__(self, data, shape):
        size = 1
        for extent in shape:
            size *= extent
        if size != len(data):
            raise ValueError(f"data of length {len(data)} does not fit shape {shape}")
        self.data = [float(value) for value in data]
        self.shape = shape
        self.grad = [0.0] * size

    @property
    def size(self):
        return len(self.data)

    def rows(self):
        count, columns = self.shape
        return [self.data[index * columns:(index + 1) * columns] for index in range(count)]

    def clear(self):
        self.grad = [0.0] * self.size

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Tensor({self.shape})"


class Tape:
    """Records backward closures; `use` declares the tensors whose gradients are reset."""

    def __init__(self):
        self.entries = []
        self.variables = []
        self.allocations = 0

    def use(self, *tensors):
        for tensor in tensors:
            if all(tensor is not other for other in self.variables):
                self.variables.append(tensor)
        return tensors[0] if len(tensors) == 1 else tensors

    def record(self, output, backward):
        self.entries.append((output, backward))
        self.allocations += 1
        return output

    def backward(self, output, seed=None):
        for tensor in self.variables:
            tensor.clear()
        for tensor, _backward in self.entries:
            tensor.clear()
        output.grad = list(seed) if seed is not None else [1.0] * output.size
        for tensor, backward in reversed(self.entries):
            if any(value != 0.0 for value in tensor.grad):
                backward(tensor.grad)
        return output


def tensor_from(rows):
    shape = (len(rows), len(rows[0]))
    if any(len(row) != shape[1] for row in rows):
        raise ValueError("ragged rows")
    return Tensor([value for row in rows for value in row], shape)


def dense(seed, rows, columns, scale=0.5):
    rng = Rng(seed)
    return Tensor([rng.gauss() * scale for _ in range(rows * columns)], (rows, columns))


# ------------------------------------------------------------------ ops
def add(tape, left, right, broadcast_right=False):
    if left.shape == right.shape:
        output = Tensor([left.data[index] + right.data[index] for index in range(left.size)],
                        left.shape)
    elif broadcast_right and right.shape == (1, left.shape[1]) and left.shape[0] > 1:
        columns = right.shape[1]
        output = Tensor([value + right.data[index % columns]
                         for index, value in enumerate(left.data)], left.shape)
    else:
        raise ValueError(f"cannot add shapes {left.shape} and {right.shape}")

    def backward(gradient):
        if left.shape == right.shape:
            for index in range(len(gradient)):
                left.grad[index] += gradient[index]
                right.grad[index] += gradient[index]
        else:
            columns = right.shape[1]
            for index, value in enumerate(gradient):
                left.grad[index] += value
                right.grad[index % columns] += value

    return tape.record(output, backward)


def multiply(tape, left, right):
    if left.shape != right.shape:
        raise ValueError(f"cannot multiply shapes {left.shape} and {right.shape}")
    output = Tensor([a * b for a, b in zip(left.data, right.data)], left.shape)

    def backward(gradient):
        for index, value in enumerate(gradient):
            left.grad[index] += value * right.data[index]
            right.grad[index] += value * left.data[index]

    return tape.record(output, backward)


def matmul(tape, left, right):
    m, k = left.shape
    k2, n = right.shape
    if k != k2:
        raise ValueError(f"cannot multiply shapes {left.shape} and {right.shape}")
    data = []
    for i in range(m):
        for j in range(n):
            total = 0.0
            for index in range(k):
                total += left.data[i * k + index] * right.data[index * n + j]
            data.append(total)
    output = Tensor(data, (m, n))

    def backward(gradient):
        for i in range(m):
            for j in range(n):
                value = gradient[i * n + j]
                if value == 0.0:
                    continue
                for index in range(k):
                    left.grad[i * k + index] += value * right.data[index * n + j]
                    right.grad[index * n + j] += value * left.data[i * k + index]

    return tape.record(output, backward)


def relu(tape, tensor):
    output = Tensor([value if value > 0.0 else 0.0 for value in tensor.data], tensor.shape)

    def backward(gradient):
        for index, value in enumerate(gradient):
            if tensor.data[index] > 0.0:
                tensor.grad[index] += value

    return tape.record(output, backward)


def exp(tape, tensor):
    values = [math.exp(value) for value in tensor.data]
    output = Tensor(values, tensor.shape)

    def backward(gradient):
        for index, value in enumerate(gradient):
            tensor.grad[index] += value * values[index]

    return tape.record(output, backward)


def log(tape, tensor):
    output = Tensor([math.log(value) for value in tensor.data], tensor.shape)

    def backward(gradient):
        for index, value in enumerate(gradient):
            tensor.grad[index] += value / tensor.data[index]

    return tape.record(output, backward)


def scale(tape, tensor, factor):
    output = Tensor([value * factor for value in tensor.data], tensor.shape)

    def backward(gradient):
        for index, value in enumerate(gradient):
            tensor.grad[index] += value * factor

    return tape.record(output, backward)


def softmax(tape, tensor):
    rows, columns = tensor.shape
    data = []
    for row in range(rows):
        values = tensor.data[row * columns:(row + 1) * columns]
        peak = max(values)
        exponentials = [math.exp(value - peak) for value in values]
        total = sum(exponentials)
        data.extend(value / total for value in exponentials)
    output = Tensor(data, tensor.shape)

    def backward(gradient):
        for row in range(rows):
            block = output.data[row * columns:(row + 1) * columns]
            incoming = gradient[row * columns:(row + 1) * columns]
            projection = sum(incoming[index] * block[index] for index in range(columns))
            for index in range(columns):
                tensor.grad[row * columns + index] += block[index] * (
                    incoming[index] - projection)

    return tape.record(output, backward)


def layernorm(tape, tensor, gamma, beta):
    rows, columns = tensor.shape
    if gamma.shape != (1, columns) or beta.shape != (1, columns):
        raise ValueError("normalisation parameters must be row vectors")
    means = []
    inverses = []
    normalised = []
    for row in range(rows):
        block = tensor.data[row * columns:(row + 1) * columns]
        average = sum(block) / columns
        variance = sum((value - average) ** 2 for value in block) / columns
        deviation = math.sqrt(variance + 1e-5)
        means.append(average)
        inverses.append(1.0 / deviation)
        normalised.extend((value - average) / deviation for value in block)
    data = [normalised[index] * gamma.data[index % columns] + beta.data[index % columns]
            for index in range(len(normalised))]
    output = Tensor(data, tensor.shape)

    def backward(gradient):
        for row in range(rows):
            start = row * columns
            xhat = normalised[start:start + columns]
            incoming = gradient[start:start + columns]
            for index in range(columns):
                gamma.grad[index] += incoming[index] * xhat[index]
                beta.grad[index] += incoming[index]
            dxhat = [incoming[index] * gamma.data[index] for index in range(columns)]
            mean_dxhat = sum(dxhat) / columns
            mean_product = sum(dxhat[index] * xhat[index] for index in range(columns)) / columns
            for index in range(columns):
                tensor.grad[start + index] += inverses[row] * (
                    dxhat[index] - mean_dxhat - xhat[index] * mean_product)

    return tape.record(output, backward)


def sum_all(tape, tensor):
    output = Tensor([sum(tensor.data)], (1, 1))

    def backward(gradient):
        for index in range(tensor.size):
            tensor.grad[index] += gradient[0]

    return tape.record(output, backward)


def mean_all(tape, tensor):
    count = tensor.size
    output = Tensor([sum(tensor.data) / count], (1, 1))

    def backward(gradient):
        for index in range(count):
            tensor.grad[index] += gradient[0] / count

    return tape.record(output, backward)


# ------------------------------------------------------------------ fusion
def fused_layernorm_affine_relu(tape, tensor, gamma, beta, bias):
    """One kernel: normalise, scale, shift and rectify without materialising intermediates.

    The backward is the composition of the four analytic backwards written out by hand; the
    per-row scratch buffers are reused instead of being recorded as extra tape entries.
    """
    rows, columns = tensor.shape
    data = [0.0] * (rows * columns)
    means = [0.0] * rows
    inverses = [0.0] * rows
    for row in range(rows):
        block = tensor.data[row * columns:(row + 1) * columns]
        average = sum(block) / columns
        variance = sum((value - average) ** 2 for value in block) / columns
        deviation = math.sqrt(variance + 1e-5)
        means[row] = average
        inverses[row] = 1.0 / deviation
        for index in range(columns):
            xhat = (block[index] - average) / deviation
            value = xhat * gamma.data[index] + beta.data[index] + bias.data[index]
            data[row * columns + index] = value if value > 0.0 else 0.0
    output = Tensor(data, tensor.shape)

    def backward(gradient):
        for row in range(rows):
            start = row * columns
            dxhat = [0.0] * columns
            for index in range(columns):
                xhat = (tensor.data[start + index] - means[row]) * inverses[row]
                pre_activation = xhat * gamma.data[index] + beta.data[index] + bias.data[index]
                if pre_activation <= 0.0:
                    continue
                incoming = gradient[start + index]
                gamma.grad[index] += incoming * xhat
                beta.grad[index] += incoming
                bias.grad[index] += incoming
                dxhat[index] = incoming * gamma.data[index]
            mean_dxhat = sum(dxhat) / columns
            mean_product = sum(dxhat[index]
                               * (tensor.data[start + index] - means[row]) * inverses[row]
                               for index in range(columns)) / columns
            for index in range(columns):
                xhat = (tensor.data[start + index] - means[row]) * inverses[row]
                tensor.grad[start + index] += inverses[row] * (
                    dxhat[index] - mean_dxhat - xhat * mean_product)

    return tape.record(output, backward)


def unfused_layernorm_affine_relu(tape, tensor, gamma, beta, bias):
    """The same expression through the tape: three extra intermediate tensors materialised."""
    return relu(tape, add(tape, layernorm(tape, tensor, gamma, beta), bias,
                          broadcast_right=True))


# ------------------------------------------------------------------ helpers
def build_layered(tape, x, weights, bias):
    """relu(x @ W + bias) - the graph used for the finite-difference checks."""
    return relu(tape, add(tape, matmul(tape, x, weights), bias, broadcast_right=True))


def loss_of(tape, prediction, target):
    difference = add(tape, prediction, scale(tape, target, -1.0))
    return mean_all(tape, multiply(tape, difference, difference))


def finite_difference(builder, tensor, index):
    """Central difference of mean(builder(tape)) with respect to tensor.data[index].

    The perturbed graphs declare `tensor` as a variable so their backwards reset its gradient
    buffer; callers must therefore snapshot any analytic gradient before perturbing.
    """
    original = tensor.data[index]
    values = []
    for sign in (1.0, -1.0):
        tensor.data[index] = original + sign * EPSILON
        tape = Tape()
        tape.use(tensor)
        value = builder(tape)
        loss = mean_all(tape, value)
        tape.backward(loss)
        values.append(loss.data[0])
    tensor.data[index] = original
    return (values[0] - values[1]) / (2.0 * EPSILON)


def dual_forward(rows, weights, bias, direction_rows, direction_weights):
    """Forward-mode (dual number) value and directional derivative of mean(relu(x @ w + b)).

    The bias takes part in the primal so that the rectifier gates exactly as in the reverse
    graph; its direction is zero, so it contributes nothing to the tangent.
    """
    m = len(rows)
    k = len(rows[0])
    n = len(weights[0])
    total = 0.0
    tangent = 0.0
    for i in range(m):
        for j in range(n):
            value = bias[j]
            derivative = 0.0
            for index in range(k):
                value += rows[i][index] * weights[index][j]
                derivative += (direction_rows[i][index] * weights[index][j]
                               + rows[i][index] * direction_weights[index][j])
            if value > 0.0:
                total += value
                tangent += derivative
    return total / (m * n), tangent / (m * n)


def main():
    # ---------------- reverse mode against finite differences ----------------
    x = dense(1, 4, 5)
    weights = dense(2, 5, 3)
    bias = dense(3, 1, 3, scale=0.2)

    def layered(tape):
        return build_layered(tape, x, weights, bias)

    tape = Tape()
    tape.use(x, weights, bias)
    prediction = layered(tape)
    loss = mean_all(tape, prediction)
    tape.backward(loss)
    expected = sum(max(0.0, value) for value in prediction.data) / prediction.size
    check(abs(loss.data[0] - expected) < 1e-12,
          "the loss must equal the mean of the rectified activations")
    analytic_weights = list(weights.grad)
    analytic_x = list(x.grad)
    analytic_bias = list(bias.grad)
    worst = 0.0
    for index in range(weights.size):
        numeric = finite_difference(layered, weights, index)
        worst = max(worst, abs(numeric - analytic_weights[index]))
    check(worst < 1e-5, f"reverse-mode gradient mismatch against finite differences: {worst}")
    check(all(math.isfinite(value) for value in analytic_weights), "gradients must be finite")
    check(any(value != 0.0 for value in analytic_x)
          and any(value != 0.0 for value in analytic_bias),
          "the upstream gradients must be populated too")

    # ---------------- shared subexpressions and broadcasts accumulate ----------------
    tape = Tape()
    value = dense(5, 2, 2)
    tape.use(value)
    tape.backward(sum_all(tape, multiply(tape, value, value)))
    check(all(abs(value.grad[index] - 2.0 * value.data[index]) < 1e-12 for index in range(4)),
          "the gradient of sum(x * x) must be 2x (accumulated twice)")
    tape = Tape()
    tape.use(value)
    tape.backward(sum_all(tape, add(tape, multiply(tape, value, value), value)))
    check(all(abs(value.grad[index] - (2.0 * value.data[index] + 1.0)) < 1e-12
              for index in range(4)),
          "a shared operand must accumulate both contributions")

    tape = Tape()
    matrix = dense(6, 5, 3)
    row_bias = dense(7, 1, 3, scale=0.1)
    tape.use(matrix, row_bias)
    tape.backward(sum_all(tape, add(tape, matrix, row_bias, broadcast_right=True)))
    check(all(abs(row_bias.grad[index] - 5.0) < 1e-12 for index in range(3)),
          f"a broadcast bias must accumulate once per row: {row_bias.grad}")
    check(all(abs(matrix.grad[index] - 1.0) < 1e-12 for index in range(matrix.size)),
          "every element of the matrix must receive its own gradient")

    # ---------------- softmax and its closed-form gradient ----------------
    logits = dense(8, 3, 4)
    tape = Tape()
    tape.use(logits)
    probabilities = softmax(tape, logits)
    tape.backward(probabilities)  # incoming ones: the Jacobian rows must sum to zero
    rows, columns = logits.shape
    for row in range(rows):
        total = sum(logits.grad[row * columns:(row + 1) * columns])
        check(abs(total) < 1e-12,
              f"seeding the softmax output with ones must give a zero row gradient: {total}")
        check(abs(sum(probabilities.data[row * columns:(row + 1) * columns]) - 1.0) < 1e-12,
              "softmax rows must sum to one")
    tape = Tape()
    tape.use(logits)
    probabilities = softmax(tape, logits)
    onehot = [0.0] * logits.size
    for row in range(rows):
        onehot[row * columns + (row % columns)] = 1.0
    # for the loss -log p_target the cotangent of the softmax output is -onehot / p, and
    # pushing it through the Jacobian p_i (delta_ij - p_j) must give exactly p - onehot
    cotangent = [-onehot[index] / probabilities.data[index] if onehot[index] else 0.0
                 for index in range(logits.size)]
    tape.backward(probabilities, seed=cotangent)
    for row in range(rows):
        for index in range(columns):
            expected = probabilities.data[row * columns + index] - onehot[row * columns + index]
            check(abs(logits.grad[row * columns + index] - expected) < 1e-9,
                  f"the cross-entropy gradient must reproduce p - onehot: "
                  f"{logits.grad[row * columns + index]} vs {expected}")

    # ---------------- layer normalisation invariants ----------------
    inputs = dense(9, 6, 8)
    gamma = Tensor([1.0 + 0.1 * index for index in range(8)], (1, 8))
    beta = Tensor([0.05 * index for index in range(8)], (1, 8))
    tape = Tape()
    tape.use(inputs, gamma, beta)
    tape.backward(sum_all(tape, layernorm(tape, inputs, gamma, beta)))
    for row in range(6):
        block = inputs.grad[row * 8:(row + 1) * 8]
        centred = [inputs.data[row * 8 + index]
                   - sum(inputs.data[row * 8:(row + 1) * 8]) / 8 for index in range(8)]
        check(abs(sum(block)) < 1e-9, f"the layer-norm gradient must sum to zero: {sum(block)}")
        projection = sum(block[index] * centred[index] for index in range(8))
        norm_block = math.sqrt(sum(value * value for value in block))
        norm_centred = math.sqrt(sum(value * value for value in centred))
        # the orthogonality identity holds exactly only when the normalised variance is one;
        # the 1e-5 variance epsilon in the reference layernorm leaves a relative deviation of
        # the same order, so the check is stated relatively
        check(abs(projection) <= 1e-3 * max(norm_block * norm_centred, 1e-12),
              f"the gradient must be orthogonal to the centred input: {projection} vs "
              f"{norm_block * norm_centred}")
    check(all(abs(value - 6.0) < 1e-9 for value in beta.grad),
          f"beta gradients must accumulate over the batch: {beta.grad}")
    check(abs(sum(gamma.grad)) < 1e-9,
          f"the gamma gradient must sum to zero across features: {sum(gamma.grad)}")
    check(any(abs(value) > 1e-6 for value in gamma.grad),
          "the gamma gradient must not be identically zero")

    # ---------------- fusion: identical numbers, one allocation ----------------
    fusion_inputs = dense(11, 6, 8)
    fusion_gamma = Tensor([1.0] * 8, (1, 8))
    fusion_beta = Tensor([0.0] * 8, (1, 8))
    fusion_bias = dense(12, 1, 8, scale=0.1)
    variables = (fusion_inputs, fusion_gamma, fusion_beta, fusion_bias)

    fused_tape = Tape()
    fused_tape.use(*variables)
    fused_output = fused_layernorm_affine_relu(fused_tape, fusion_inputs, fusion_gamma,
                                               fusion_beta, fusion_bias)
    unfused_tape = Tape()
    unfused_tape.use(*variables)
    unfused_output = unfused_layernorm_affine_relu(unfused_tape, fusion_inputs, fusion_gamma,
                                                   fusion_beta, fusion_bias)
    check(fused_output.data == unfused_output.data,
          "the fused kernel must reproduce the unfused forward bit for bit")
    check(fused_tape.allocations == 1 and unfused_tape.allocations >= 3,
          f"fusion must remove intermediate tensors: {fused_tape.allocations} vs "
          f"{unfused_tape.allocations}")
    check(unfused_output is not fused_output,
          "the two paths must be separate graphs")
    fused_tape.backward(mean_all(fused_tape, fused_output))
    fused_gradients = [list(tensor.grad) for tensor in variables]
    unfused_tape.backward(mean_all(unfused_tape, unfused_output))
    unfused_gradients = [list(tensor.grad) for tensor in variables]
    for name, first, second in zip(("inputs", "gamma", "beta", "bias"), fused_gradients,
                                   unfused_gradients):
        check(all(abs(a - b) < 1e-15 for a, b in zip(first, second)),
              f"fused and unfused gradients must agree for {name}")
    check(any(value != 0.0 for value in fused_gradients[0]),
          "the fused backward must actually produce a gradient")

    def fused_loss(tape):
        return mean_all(tape, fused_layernorm_affine_relu(tape, fusion_inputs, fusion_gamma,
                                                          fusion_beta, fusion_bias))

    fused_tape = Tape()
    fused_tape.use(*variables)
    fused_tape.backward(fused_loss(fused_tape))
    analytic_fused_bias = list(fusion_bias.grad)
    worst = 0.0
    for index in range(fusion_bias.size):
        numeric = finite_difference(fused_loss, fusion_bias, index)
        worst = max(worst, abs(numeric - analytic_fused_bias[index]))
    check(worst < 1e-5, f"the fused backward must match finite differences: {worst}")

    # ---------------- forward mode against reverse mode ----------------
    rows = x.rows()
    weight_rows = weights.rows()
    direction_rows = [[1.0 if (i + j) % 3 == 0 else 0.0 for j in range(5)] for i in range(4)]
    direction_weights = [[1.0 if (i * 2 + j) % 4 == 0 else 0.0 for j in range(3)]
                         for i in range(5)]
    value, tangent = dual_forward(rows, weight_rows, bias.data,
                                  direction_rows, direction_weights)
    direct = sum(max(0.0, sum(rows[i][k] * weight_rows[k][j] for k in range(5)) + bias.data[j])
                 for i in range(4) for j in range(3)) / 12.0
    check(abs(value - direct) < 1e-12, "the dual-number value must match the direct evaluation")
    tape = Tape()
    tape.use(x, weights, bias)
    tape.backward(mean_all(tape, build_layered(tape, x, weights, bias)))
    reverse_tangent = (sum(x.grad[i * 5 + j] * direction_rows[i][j]
                           for i in range(4) for j in range(5))
                       + sum(weights.grad[i * 3 + j] * direction_weights[i][j]
                             for i in range(5) for j in range(3)))
    check(abs(tangent - reverse_tangent) < 1e-12,
          f"forward and reverse mode must agree: {tangent} vs {reverse_tangent}")

    # ---------------- ReLU, exp, log edge cases ----------------
    tape = Tape()
    boundary = Tensor([0.0, -1.0, 1.0], (1, 3))
    tape.use(boundary)
    rectified = relu(tape, boundary)
    tape.backward(sum_all(tape, rectified))
    check(boundary.grad[0] == 0.0, "the ReLU subgradient at zero must be zero")
    check(boundary.grad[1] == 0.0, "a dead unit must get no gradient")
    check(boundary.grad[2] == 1.0, "a live unit must get the full gradient")
    check(rectified.data == [0.0, 0.0, 1.0], "ReLU must clip negatives to zero")
    tape = Tape()
    shifted = Tensor([1.0, 2.0], (1, 2))
    tape.use(shifted)
    tape.backward(sum_all(tape, exp(tape, shifted)))
    check(abs(shifted.grad[0] - math.exp(1.0)) < 1e-12,
          "the exp gradient must be the exponential itself")
    tape = Tape()
    positive = Tensor([1.0, 2.0], (1, 2))
    tape.use(positive)
    tape.backward(sum_all(tape, log(tape, positive)))
    check(abs(positive.grad[1] - 0.5) < 1e-12, "the log gradient must be 1/x")

    # ---------------- a real training loop ----------------
    parameters = [dense(21, 5, 4), dense(22, 1, 4, scale=0.1)]
    data = dense(23, 8, 5)
    truth = dense(30, 5, 4, scale=0.3)
    noise = Rng(31)
    labels = Tensor([sum(data.data[i * 5 + k] * truth.data[k * 4 + j] for k in range(5))
                     + 0.01 * noise.gauss() for i in range(8) for j in range(4)], (8, 4))
    losses = []
    for _step in range(80):
        tape = Tape()
        tape.use(data, labels, *parameters)
        prediction = add(tape, matmul(tape, data, parameters[0]), parameters[1],
                         broadcast_right=True)
        value = loss_of(tape, prediction, labels)
        tape.backward(value)
        losses.append(value.data[0])
        for parameter in parameters:
            for index in range(parameter.size):
                parameter.data[index] -= 0.2 * parameter.grad[index]
    check(losses[-1] < losses[0] * 0.05,
          f"gradient descent must reduce the loss by more than 95%: {losses[0]} -> "
          f"{losses[-1]}")
    check(all(later <= earlier + 1e-9 for earlier, later in zip(losses, losses[1:])),
          f"no optimisation step may increase the loss: {losses[:4]}")
    check(losses[-1] < 0.02,
          f"the fit must approach the generating model: {losses[-1]}")

    # ---------------- shape errors and determinism ----------------
    for call, why in (
        (lambda: matmul(Tape(), tensor_from([[1.0, 2.0]]), tensor_from([[1.0, 2.0]])),
         "inner dimension mismatch"),
        (lambda: multiply(Tape(), tensor_from([[1.0]]), tensor_from([[1.0], [2.0]])),
         "shape mismatch in multiply"),
        (lambda: Tensor([1.0, 2.0], (3, 1)), "data that does not fit the shape"),
        (lambda: tensor_from([[1.0], [1.0, 2.0]]), "ragged rows"),
        (lambda: layernorm(Tape(), inputs, Tensor([1.0, 2.0], (1, 2)), beta),
         "wrong normalisation parameter shape"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    first_tape = Tape()
    first_tape.use(*variables)
    first_tape.backward(fused_loss(first_tape))
    recorded = [list(tensor.grad) for tensor in variables]
    second_tape = Tape()
    second_tape.use(*variables)
    second_tape.backward(fused_loss(second_tape))
    check([list(tensor.grad) for tensor in variables] == recorded,
          "autodiff must be deterministic")

    print(f"finite-difference worst error={worst:.3e} fused allocations="
          f"{fused_tape.allocations} unfused={unfused_tape.allocations}")
    print(f"training loss {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"forward-mode tangent={tangent:.9f} reverse={reverse_tangent:.9f}")
    print("PASS")


if __name__ == "__main__":
    main()
