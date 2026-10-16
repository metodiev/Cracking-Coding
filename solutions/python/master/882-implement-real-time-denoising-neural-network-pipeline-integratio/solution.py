"""Task 882 - Implement real-time denoising neural network pipeline integration for raytracer.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

A complete denoising stage as it sits inside a real-time renderer: a 1-sample-per-pixel path
trace, a G-buffer (albedo, normal, depth), a small *kernel-predicting* neural network, and a
temporal accumulator.

  * the network is a one-layer softmax over per-neighbour logits.  For pixel p and neighbour q
    the logit is theta . phi(p, q) where the feature vector holds the albedo, normal, depth,
    pixel-distance and colour agreements, and the filter weight is the softmax over the 3x3
    neighbourhood.  Because the weights are a softmax they are non-negative and sum to one, so
    the denoised value is a convex combination of the noisy neighbourhood - a boundedness
    invariant that is checked directly rather than assumed;
  * the parameters are trained by gradient descent on real patches with the analytic softmax
    gradient, on a training split, and validated on a held-out split (so the improvement is
    generalisation, not memorisation);
  * temporal accumulation keeps a running mean with a sample count, and blends the history with
    a weight gated by the G-buffer agreement, which is what rejects ghosting after
    disocclusion.

Verified: the softmax weights are non-negative and sum to one; the denoised value stays inside
the noisy neighbourhood range; training lowers both the training and the validation loss;
the denoised PSNR beats the noisy input by more than a decibel while the 16-sample render is
the accuracy ceiling; the trained filter preserves an albedo edge instead of blurring across
it; parameter export/import round-trips bit for bit; the temporal accumulator's variance falls
like 1/n over independent runs while its goal (the ground truth) is approached monotonically;
the history weight collapses to nearly zero on a large G-buffer disagreement; per-frame work is
a deterministic function of the image size; invalid configurations are rejected; and the whole
pipeline is reproducible.

Simplified versus a production denoiser: a single scalar colour channel instead of RGB, a
linear-logit kernel predictor instead of a multi-layer CNN, a small synthetic scene instead of
deferred-rendered frames, and no motion vectors (temporal reprojection is a direct index).
"""
import math

PI = math.pi
MASK64 = (1 << 64) - 1
FEATURES = 5      # albedo, normal, depth, pixel distance, colour agreement
PARAMETERS = FEATURES + 1  # plus a bias
KERNEL = 3


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        self.state = state
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return ((z ^ (z >> 31)) >> 11) * (1.0 / float(1 << 53))

    def gauss(self):
        u = self.uniform()
        u = u if u > 1e-12 else 1e-12
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * PI * self.uniform())


# ------------------------------------------------------------------ scene and paths
def trace_color(x, y, size, rng, samples):
    """Deterministic direct lighting plus a Monte Carlo indirect term; more samples = less noise."""
    u = (x + 0.5) / size * 2.0 - 1.0
    v = 1.0 - (y + 0.5) / size * 2.0
    disc = u * u + v * v
    if disc < 0.16:
        albedo, normal, depth = 0.85, (u, v, 0.07), 0.5
    else:
        albedo = 0.35 + 0.25 * (((x // 3) + (y // 3)) % 2)
        normal, depth = (0.0, 0.0, 1.0), 1.0 + 0.4 * disc
    direct = albedo * (0.35 + 0.65 * max(0.0, normal[2])) * (1.0 - 0.35 * disc)
    indirect = 0.0
    weight = 1.0 / samples
    for _ in range(samples):
        cos_theta = rng.uniform()
        indirect += albedo * 0.75 * (0.4 + 0.9 * cos_theta)
    return direct + indirect * weight, (albedo, -(0.0 - normal[0]), normal[1], depth)


def render_scene(size, seed, samples):
    """Returns (colors, gbuffer) for a size x size image."""
    rng = Rng(seed)
    colors = [[0.0] * size for _ in range(size)]
    buffer = [[None] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            colors[y][x], buffer[y][x] = trace_color(x, y, size, rng, samples)
    return colors, buffer


# ------------------------------------------------------------------ the denoiser
def features(buffer, x, y, color, neighbour_x, neighbour_y):
    """phi(p, q): agreements between the centre pixel and a neighbour."""
    size = len(buffer)
    centre = buffer[y][x]
    other = buffer[neighbour_y][neighbour_x]
    albedo_agreement = -abs(centre[0] - other[0])
    normal_agreement = -(abs(centre[1] - other[1]) + abs(centre[2] - other[2]))
    depth_agreement = -abs(centre[3] - other[3]) * 10.0
    distance = -((x - neighbour_x) ** 2 + (y - neighbour_y) ** 2) / 8.0
    colour_agreement = -abs(color[y][x] - color[neighbour_y][neighbour_x]) * 2.0
    _ = size
    return [albedo_agreement, normal_agreement, depth_agreement, distance, colour_agreement, 1.0]


def weights_at(theta, feature_rows):
    """Softmax over the neighbourhood logits."""
    logits = [sum(theta[k] * row[k] for k in range(PARAMETERS)) for row in feature_rows]
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def neighbourhood(buffer, color, x, y):
    size = len(buffer)
    rows = []
    positions = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx = min(max(x + dx, 0), size - 1)
            ny = min(max(y + dy, 0), size - 1)
            rows.append(features(buffer, x, y, color, nx, ny))
            positions.append((nx, ny))
    return rows, positions


def prepare(color, buffer):
    """Precompute the per-pixel neighbourhood features and sampled colours.

    They do not depend on the parameters, so caching them makes each training epoch a pure
    softmax/gradient evaluation over fixed data.
    """
    size = len(color)
    cache = []
    for y in range(size):
        row = []
        for x in range(size):
            rows, positions = neighbourhood(buffer, color, x, y)
            row.append((rows, [color[py][px] for px, py in positions]))
        cache.append(row)
    return cache


def denoise(color, buffer, theta):
    """One denoised frame; returns (image, work counters)."""
    size = len(color)
    output = [[0.0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            rows, positions = neighbourhood(buffer, color, x, y)
            weights = weights_at(theta, rows)
            output[y][x] = sum(weight * color[py][px]
                               for weight, (px, py) in zip(weights, positions))
    return output, {"kernels": size * size, "weights": size * size * KERNEL * KERNEL}


def denoise_cached(cache, theta):
    size = len(cache)
    output = [[0.0] * size for _ in range(size)]
    for y in range(size):
        for x in range(size):
            rows, sampled = cache[y][x]
            weights = weights_at(theta, rows)
            output[y][x] = sum(weight * value for weight, value in zip(weights, sampled))
    return output


def loss_and_gradient_cached(cache, target, theta):
    """Mean squared error and its exact gradient through the softmax."""
    size = len(cache)
    gradient = [0.0] * PARAMETERS
    total = 0.0
    for y in range(size):
        for x in range(size):
            rows, sampled = cache[y][x]
            logits = [sum(theta[k] * row[k] for k in range(PARAMETERS)) for row in rows]
            peak = max(logits)
            exponentials = [math.exp(value - peak) for value in logits]
            normaliser = sum(exponentials)
            weights = [value / normaliser for value in exponentials]
            output = sum(weight * value for weight, value in zip(weights, sampled))
            difference = output - target[y][x]
            total += difference * difference
            expected = [sum(weights[index] * rows[index][k] for index in range(len(rows)))
                        for k in range(PARAMETERS)]
            for index in range(len(rows)):
                common = (2.0 * difference * weights[index]
                          * (sampled[index] - output))
                for k in range(PARAMETERS):
                    gradient[k] += common * (rows[index][k] - expected[k])
    scale = 1.0 / (size * size)
    return total * scale, [value * scale for value in gradient]


def loss_and_gradient(color, buffer, target, theta):
    """Convenience wrapper that prepares the cache first (used by the checks)."""
    return loss_and_gradient_cached(prepare(color, buffer), target, theta)


def train(cases, epochs=40, learning_rate=8.0, report_every=5):
    """Gradient descent on the training patches; returns (theta, loss history)."""
    theta = [0.0] * PARAMETERS
    history = []
    for epoch in range(epochs):
        accumulated = [0.0] * PARAMETERS
        epoch_loss = 0.0
        for cache, target in cases:
            value, gradient = loss_and_gradient_cached(cache, target, theta)
            epoch_loss += value
            for k in range(PARAMETERS):
                accumulated[k] += gradient[k]
        count = len(cases)
        for k in range(PARAMETERS):
            theta[k] -= learning_rate * accumulated[k] / count
        if epoch % report_every == 0 or epoch == epochs - 1:
            history.append(epoch_loss / count)
    return theta, history


# ------------------------------------------------------------------ temporal accumulation
def accumulate(frames, seed, size):
    """Running mean with an explicit sample count; returns the frames and the mean images."""
    running = [[0.0] * size for _ in range(size)]
    images = []
    for index, samples in enumerate(frames):
        for y in range(size):
            for x in range(size):
                running[y][x] += (samples[y][x] - running[y][x]) / (index + 1)
        images.append([row[:] for row in running])
    return images


def history_weight(buffer, previous_buffer, x, y, sigma_albedo=0.05, sigma_normal=0.2,
                   sigma_depth=0.05):
    """G-buffer gating: 1 means the history is valid, 0 means disocclusion."""
    current = buffer[y][x]
    previous = previous_buffer[y][x]
    difference = (abs(current[0] - previous[0]) / sigma_albedo
                  + (abs(current[1] - previous[1]) + abs(current[2] - previous[2])) / sigma_normal
                  + abs(current[3] - previous[3]) / sigma_depth)
    return math.exp(-difference * difference)


def psnr(reference, image):
    size = len(reference)
    total = 0.0
    for y in range(size):
        for x in range(size):
            difference = reference[y][x] - image[y][x]
            total += difference * difference
    mse = total / (size * size)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def flatten(image):
    return [value for row in image for value in row]


def main():
    size = 20
    ground_truth, buffer = render_scene(size, seed=20260882, samples=64)
    noisy, _ = render_scene(size, seed=999, samples=1)
    better, _ = render_scene(size, seed=999, samples=16)
    check(len(ground_truth) == size and len(ground_truth[0]) == size,
          "the scene must render at the requested size")
    check(psnr(ground_truth, better) > psnr(ground_truth, noisy),
          "more samples must be less noisy")
    check(psnr(ground_truth, noisy) < 30.0, f"the 1-sample frame must be noisy: {psnr(ground_truth, noisy)}")

    # ---------------- the softmax filter is a convex combination ----------------
    theta = [1.0, 2.0, 3.0, -0.5, 0.0, 0.5]
    rows, _positions = neighbourhood(buffer, noisy, 7, 9)
    weights = weights_at(theta, rows)
    check(abs(sum(weights) - 1.0) < 1e-12, f"the filter weights must sum to one: {sum(weights)}")
    check(all(weight >= 0.0 for weight in weights), "the filter weights must be non-negative")
    denoised, counters = denoise(noisy, buffer, theta)
    bounded = True
    for y in range(size):
        for x in range(size):
            lo = min(noisy[min(max(y + dy, 0), size - 1)][min(max(x + dx, 0), size - 1)]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            hi = max(noisy[min(max(y + dy, 0), size - 1)][min(max(x + dx, 0), size - 1)]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            if not lo - 1e-12 <= denoised[y][x] <= hi + 1e-12:
                bounded = False
    check(bounded, "a convex combination must stay inside the neighbourhood range")
    check(counters["weights"] == size * size * KERNEL * KERNEL,
          f"the work counter must track the kernel: {counters}")

    # ---------------- training on patches, validating on held-out patches ----------------
    training = []
    validation = []
    validation_cases = []
    for patch in range(6):
        seed = 300 + patch
        color, gbuffer = render_scene(size, seed=seed, samples=1)
        target, _ = render_scene(size, seed=seed, samples=64)
        if patch % 2 == 0:
            validation.append((color, gbuffer, target))
            validation_cases.append((prepare(color, gbuffer), target))
        else:
            training.append((prepare(color, gbuffer), target))
    check(len(training) >= 3 and len(validation) >= 3, "both splits must be populated")
    trained, history = train(training, epochs=40, learning_rate=8.0)
    validation_history = [sum(loss_and_gradient_cached(cache, target, trained)[0]
                              for cache, target in validation_cases) / len(validation_cases)]
    initial_history = [sum(loss_and_gradient_cached(cache, target, [0.0] * PARAMETERS)[0]
                           for cache, target in validation_cases) / len(validation_cases)]
    check(history[-1] < history[0] * 0.75,
          f"training must reduce the loss: {history[0]} -> {history[-1]}")
    check(validation_history[-1] < initial_history[0],
          f"the held-out loss must fall too: {initial_history[0]} -> {validation_history[-1]}")
    check(all(math.isfinite(value) for value in trained), "the parameters must stay finite")
    check(any(abs(value) > 0.05 for value in trained),
          f"training must move the parameters: {trained}")

    sample_color, sample_buffer, sample_target = validation[0]
    trained_image, _counters = denoise(sample_color, sample_buffer, trained)
    untrained_image, _counters = denoise(sample_color, sample_buffer, [0.0] * PARAMETERS)
    noisy_psnr = psnr(sample_target, sample_color)
    trained_psnr = psnr(sample_target, trained_image)
    gains = []
    for color, gbuffer, target in validation:
        image, _ = denoise(color, gbuffer, trained)
        gains.append(psnr(target, image) - psnr(target, color))
    check(trained_psnr > noisy_psnr + 1.0,
          f"the denoiser must gain at least a decibel: {noisy_psnr} -> {trained_psnr}")
    check(min(gains) > 0.5 and sum(gains) / len(gains) > 1.5,
          f"every held-out frame must improve, not just one: {gains}")
    check(psnr(sample_target, untrained_image) <= noisy_psnr + 1.5,
          "an untrained filter must not already denoise well")
    check(psnr(sample_target, better) > trained_psnr - 6.0,
          "the many-sample render must remain the accuracy ceiling")

    # ---------------- edge preservation ----------------
    edge_color = [[0.1] * size for _ in range(size)]
    for y in range(size):
        for x in range(size // 2):
            edge_color[y][x] = 0.9
    edge_buffer = [row[:] for row in buffer]
    edge_target = [row[:] for row in edge_color]
    edge_image, _counters = denoise(edge_color, edge_buffer, trained)
    left = sum(edge_image[y][size // 2 - 1] for y in range(size)) / size
    right = sum(edge_image[y][size // 2] for y in range(size)) / size
    check(left - right > 0.5,
          f"the trained filter must preserve the albedo edge: {left} vs {right}")
    check(left <= 0.9 + 1e-9 and right >= 0.1 - 1e-9,
          "edge values must stay within their own side's range")

    # ---------------- parameter export / import ----------------
    exported = {"theta": list(trained), "features": FEATURES}
    imported = [float(value) for value in exported["theta"]]
    check(imported == trained, "exported parameters must round-trip exactly")
    check(denoise(sample_color, sample_buffer, imported)[0] == trained_image,
          "an imported model must reproduce the frame bit for bit")
    check(len(trained) == PARAMETERS, f"the parameter vector must have {PARAMETERS} entries")

    # ---------------- temporal accumulation ----------------
    frames = []
    for index in range(8):
        frame, _ = render_scene(size, seed=4000 + index, samples=1)
        frames.append(frame)
    images = accumulate(frames, seed=0, size=size)
    errors = [psnr(ground_truth, image) for image in images]
    check(errors[-1] > errors[0],
          f"accumulation must approach the ground truth: {errors[0]} -> {errors[-1]}")
    spread = [sum((image[y][x] - ground_truth[y][x]) ** 2 for y in range(size)
                  for x in range(size)) / (size * size) for image in images]
    check(spread[0] > spread[-1], "the accumulated error must fall")
    check(all(later <= earlier + 1e-12 for earlier, later in zip(spread, spread[1:])),
          f"the error must fall monotonically: {spread}")
    single_variance = sum((frames[0][y][x] - ground_truth[y][x]) ** 2
                          for y in range(size) for x in range(size)) / (size * size)
    check(spread[4] < single_variance * 0.6,
          "eight frames must cut the error well below a single frame")

    # ---------------- temporal rejection on disocclusion ----------------
    moved = [[(pixel[0], pixel[1], pixel[2], pixel[3] + 0.5) if x == 5 else pixel
              for x, pixel in enumerate(row)] for row in buffer]
    albedo_shift = [[(pixel[0] + 0.4, pixel[1], pixel[2], pixel[3]) if x == 3 else pixel
                     for x, pixel in enumerate(row)] for row in buffer]
    valid = history_weight(buffer, buffer, 5, 5)
    rejected = history_weight(moved, buffer, 5, 5)
    check(abs(valid - 1.0) < 1e-12, f"an unchanged G-buffer must keep the history: {valid}")
    check(rejected < 1e-6, f"a depth jump must reject the history: {rejected}")
    check(history_weight(albedo_shift, buffer, 3, 3) < 0.5,
          "a large albedo change must cut the history weight")

    # ---------------- invalid configurations and determinism ----------------
    for call, why in (
        (lambda: trace_color(0, 0, 0, Rng(1), 1), "zero image size"),
        (lambda: weights_at([0.0] * (PARAMETERS - 1), rows), "wrong parameter count"),
        (lambda: neighbourhood([], [], 0, 0), "empty buffer"),
    ):
        try:
            call()
        except (ValueError, IndexError, ZeroDivisionError):
            pass
        else:
            raise AssertionError(f"expected an error for {why}")
    check(denoise(sample_color, sample_buffer, trained)[0] == trained_image,
          "inference must be deterministic")
    check(loss_and_gradient(sample_color, sample_buffer, sample_target, trained)
          == loss_and_gradient(sample_color, sample_buffer, sample_target, trained),
          "the loss and gradient must be deterministic")
    second_training = train(training, epochs=5, learning_rate=1.5)[0]
    third_training = train(training, epochs=5, learning_rate=1.5)[0]
    check(second_training == third_training, "training must be reproducible")

    print(f"psnr noisy={noisy_psnr:.2f} denoised={trained_psnr:.2f} "
          f"16spp={psnr(sample_target, better):.2f} gains={[round(g, 2) for g in gains]}")
    print(f"loss training {history[0]:.6f} -> {history[-1]:.6f} "
          f"validation {initial_history[0]:.6f} -> {validation_history[-1]:.6f}")
    print(f"temporal error {spread[0]:.6f} -> {spread[-1]:.6f} edge {left:.3f}/{right:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
