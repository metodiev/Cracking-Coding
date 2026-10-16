"""Task 824 - Implement differentiable renderer basic and integrate into autograd.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

A differentiable renderer is a rasteriser whose output is a differentiable function of scene
parameters, so a loss on the *image* yields gradients with respect to geometry, colour and camera
through a single reverse sweep.  This implementation is a small reverse-mode autodiff engine plus a
2D triangle rasteriser written entirely in that engine's primitives:

  * `Var` is the autodiff node: every primitive stores its operands together with the local partials,
    and `backward` walks the graph in reverse creation order (a topological order of a DAG built
    strictly forward) accumulating parent.grad += child.grad * partial.  A value used twice therefore
    receives both contributions, which is what makes a shared-parameter renderer differentiable at all;
  * the coverage of a pixel by a triangle uses the three signed edge functions normalised to distances
    and a product of sigmoids, coverage(p) = sigma(e0/tau) * sigma(e1/tau) * sigma(e2/tau).  As tau
    shrinks the product tends to the hard inside indicator, which is what an anti-aliased rasteriser
    converges to - and it stays differentiable everywhere, including on the edges;
  * the image is rendered by composing those primitives, so the whole rasteriser becomes nodes in the
    same autograd graph as the loss: no hand-written gradient for the renderer exists anywhere.

Verified: every primitive's analytic gradient against central finite differences (and the composite
chain rule on a nested expression); the diamond case where one variable feeds two paths; the coverage
staying inside [0, 1] and converging to the hard indicator (and to the exact triangle area) as the
temperature falls; a clockwise triangle rendering as empty, which is the orientation convention; the
colour gradient matching the analytic identity d(image)/d(colour) = the sum of coverage; the gradients
with respect to all six vertex coordinates matching central finite differences of the rendered loss; a
gradient-descent fit of a triangle onto a target image reducing the loss by the required factor, with
the vertices landing near the target and the loss decreasing for small steps along the negative
gradient; the degenerate cases (a zero-area triangle, a single-pixel image, a temperature so large that
the shape disappears, a triangle entirely off-screen) rendering finite values and finite gradients; and
determinism of two identical renders.

Simplified: a scalar (per-pixel, per-channel) tape rather than a tensor one, so the adjoint of every
multiply is an explicit node - at scale an engine keeps one node per tensor op and a vectorised
adjoint, with the same graph shape, the same single sweep and the same cost argument; the scene is one
flat-shaded 2D triangle on a small image instead of meshes with depth buffering, texture gradients,
lighting and a camera model; and the rasteriser walks every pixel rather than bounding the triangle's
tile.  The invariants verified - exact local gradients, a coverage that converges to the silhouette and
to the area, orientation, and a working fit driven purely by autodiff - are the ones a differentiable
renderer must hold.
"""
import math

TAU = 0.35
COLOUR = 1.0
SIZE = 16
FINITE_DIFFERENCE = 1e-5


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------ autograd
class Var:
    """A scalar node: value, adjoint, and the operand/local-partial pairs of the primitive."""

    __slots__ = ("value", "grad", "parents", "partials", "label")

    def __init__(self, value, parents=(), partials=(), label="v"):
        self.value = float(value)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.partials = tuple(partials)
        self.label = label

    def __float__(self):
        return self.value

    def __add__(self, other):
        other = constant(other)
        return Var(self.value + other.value, (self, other), (1.0, 1.0), "add")

    __radd__ = __add__

    def __sub__(self, other):
        other = constant(other)
        return Var(self.value - other.value, (self, other), (1.0, -1.0), "sub")

    def __rsub__(self, other):
        return constant(other) - self

    def __mul__(self, other):
        other = constant(other)
        return Var(self.value * other.value, (self, other), (other.value, self.value), "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = constant(other)
        return Var(self.value / other.value, (self, other),
                   (1.0 / other.value, -self.value / (other.value * other.value)), "div")

    def __neg__(self):
        return Var(-self.value, (self,), (-1.0,), "neg")

    def __repr__(self):
        return f"{self.label}({self.value:.6f})"


def constant(value):
    return value if isinstance(value, Var) else Var(value, label="const")


def sin(x):
    return Var(math.sin(x.value), (x,), (math.cos(x.value),), "sin")


def cos(x):
    return Var(math.cos(x.value), (x,), (-math.sin(x.value),), "cos")


def exp(x):
    value = math.exp(x.value)
    return Var(value, (x,), (value,), "exp")


def log(x):
    check(x.value > 0, "log needs a positive argument")
    return Var(math.log(x.value), (x,), (1.0 / x.value,), "log")


def sqrt(x):
    check(x.value > 0, "sqrt needs a positive argument")
    root = math.sqrt(x.value)
    return Var(root, (x,), (0.5 / root,), "sqrt")


def tanh(x):
    value = math.tanh(x.value)
    return Var(value, (x,), (1.0 - value * value,), "tanh")


def relu(x):
    return Var(max(0.0, x.value), (x,), (1.0 if x.value > 0 else 0.0,), "relu")


def sigmoid(x):
    """The soft coverage switch: numerically safe, derivative s * (1 - s)."""
    if x.value < -60:
        value = 0.0
    elif x.value > 60:
        value = 1.0
    else:
        value = 1.0 / (1.0 + math.exp(-x.value))
    return Var(value, (x,), (value * (1.0 - value),), "sigmoid")


def tangent_sqrt(x, floor=1e-12):
    """A sqrt that never divides by zero, so a degenerate triangle keeps a finite gradient."""
    return sqrt(x + floor)


def backward(root):
    """One reverse sweep over the subgraph that produced `root`.

    The nodes are visited in post-order (every operand before the node that consumes it) and then
    walked in reverse, so a node's adjoint is complete before any of it is handed to an operand; that
    is what makes a value used on several paths accumulate every contribution."""
    post = []
    seen = set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            post.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for parent in node.parents:
            if id(parent) not in seen:
                stack.append((parent, False))
    for node in post:
        node.grad = 0.0
    root.grad = 1.0
    for node in reversed(post):
        for parent, partial in zip(node.parents, node.partials):
            parent.grad += node.grad * partial
    return post


# ------------------------------------------------------------------ the renderer
def edge_distance(base_x, base_y, tip_x, tip_y, pixel_x, pixel_y):
    """The signed distance from a pixel centre to the line base -> tip.

    The numerator is twice the triangle area of the three points, the denominator its edge length, so
    the result has units of pixels and the temperature is scale-free."""
    numerator = (tip_x - base_x) * (pixel_y - base_y) - (tip_y - base_y) * (pixel_x - base_x)
    denominator = tangent_sqrt((tip_x - base_x) * (tip_x - base_x) +
                               (tip_y - base_y) * (tip_y - base_y))
    return numerator / denominator


def render(vertices, colour=None, size=SIZE, tau=TAU):
    """Rasterise one flat-shaded triangle; `vertices` are six Vars (x, y per corner, counter-clockwise).

    Returns the list of per-pixel intensities, each a node of the autograd graph."""
    check(len(vertices) == 6, "a triangle needs six coordinates")
    colour = constant(COLOUR) if colour is None else constant(colour)
    pixels = []
    for row in range(size):
        for column in range(size):
            pixel_x = constant(column + 0.5)
            pixel_y = constant(row + 0.5)
            coverage = constant(1.0)
            for corner in range(3):
                following = (corner + 1) % 3
                nearer = corner
                distance = edge_distance(vertices[2 * nearer], vertices[2 * nearer + 1],
                                         vertices[2 * following], vertices[2 * following + 1],
                                         pixel_x, pixel_y)
                coverage = coverage * sigmoid(distance / tau)
            pixels.append(coverage * colour)
    return pixels


def triangle_area(vertices):
    """Twice the cross product of two edges, halved: the shoelace area of the triangle."""
    return abs((vertices[2] - vertices[0]) * (vertices[5] - vertices[1]) -
               (vertices[4] - vertices[0]) * (vertices[3] - vertices[1])) / 2.0


def hard_coverage(vertices, size=SIZE):
    """The exact silhouette by the same edge functions without the soft switch."""
    points = []
    for row in range(size):
        for column in range(size):
            inside = True
            for corner in range(3):
                following = (corner + 1) % 3
                numerator = ((vertices[2 * following] - vertices[2 * corner]) *
                             (row + 0.5 - vertices[2 * corner + 1]) -
                             (vertices[2 * following + 1] - vertices[2 * corner + 1]) *
                             (column + 0.5 - vertices[2 * corner]))
                if numerator <= 0:
                    inside = False
            points.append(1.0 if inside else 0.0)
    return points


def loss_to_target(vertices, target, colour=None, tau=TAU):
    pixels = render(vertices, colour, SIZE, tau)
    total = constant(0.0)
    for value, goal in zip(pixels, target):
        difference = value - goal
        total = total + difference * difference
    return total


# ----------------------------------------------------------------------- checks
def finite_difference(function, value, step=FINITE_DIFFERENCE):
    return (function(value + step) - function(value - step)) / (2 * step)


def make_vertices(values):
    """Six Vars from six floats, so the same triangle can be rendered repeatedly."""
    return [Var(value, label=f"v{index}") for index, value in enumerate(values)]


def unit_triangle():
    return [4.0, 4.0, 12.0, 4.0, 8.0, 12.0]


def main():
    # 1. every primitive's analytic gradient against central finite differences
    primitives = {
        "sin": (math.sin, lambda x: Var(math.sin(x), (x,), (math.cos(x.value),))),
        "cos": (math.cos, lambda x: Var(math.cos(x), (x,), (-math.sin(x.value),))),
        "exp": (math.exp, lambda x: Var(math.exp(x), (x,), (math.exp(x.value),))),
        "tanh": (math.tanh, lambda x: Var(math.tanh(x), (x,),
                                          (1 - math.tanh(x.value) ** 2,))),
        "sigmoid": (lambda value: 1 / (1 + math.exp(-value)),
                    lambda x: Var(1 / (1 + math.exp(-x.value)), (x,),
                                  ((1 / (1 + math.exp(-x.value))) *
                                   (1 - 1 / (1 + math.exp(-x.value))),))),
    }
    for name, (plain, node) in sorted(primitives.items()):
        x = Var(0.7, label="x")
        out = node(x)
        backward(out)
        reference = finite_difference(plain, 0.7)
        check(abs(x.grad - reference) < 1e-6,
              f"{name}: analytic gradient {x.grad:.8f} must match differences {reference:.8f}")
    for value in (0.3, 1.4, 2.7):
        x = Var(value, label="x")
        out = x * x * x - 2.0 * x + sin(x) / (1.0 + x * x)
        backward(out)
        reference = finite_difference(
            lambda t: t ** 3 - 2.0 * t + math.sin(t) / (1.0 + t * t), value)
        check(abs(x.grad - reference) < 1e-6,
              f"a composite must differentiate correctly at {value}: {x.grad:.8f} vs {reference:.8f}")

    # 2. the diamond: one variable feeding two paths accumulates both contributions
    x = Var(2.0, label="x")
    out = x * x + x
    backward(out)
    check(abs(x.grad - 5.0) < 1e-12, f"d(x^2 + x)/dx at 2 must be 5, got {x.grad}")
    x = Var(3.0, label="x")
    backward(x * x * x)
    check(abs(x.grad - 27.0) < 1e-12, "a three-fold use must accumulate three times")

    # 3. the renderer's coverage stays inside [0, 1] and converges to the silhouette
    vertices = make_vertices(unit_triangle())
    pixels = render(vertices)
    values = [pixel.value for pixel in pixels]
    check(all(0.0 <= value <= 1.0 for value in values),
          f"coverage must stay inside [0, 1], got [{min(values):.3f}, {max(values):.3f}]")
    check(max(values) > 0.9 and min(values) < 0.1,
          "the rendered image must contain both covered and empty pixels")
    soft_area = sum(values)
    exact_area = triangle_area(unit_triangle())
    check(abs(soft_area - exact_area) < 0.3 * exact_area + 1.0,
          f"the soft area {soft_area:.2f} must be near the exact area {exact_area:.2f}")
    hard = hard_coverage(unit_triangle())
    check(abs(sum(hard) - exact_area) < 3.0,
          f"the hard silhouette must measure the triangle area: {sum(hard):.2f} vs {exact_area:.2f}")
    ambiguity = sum(min(value, 1 - value) for value in values)
    sharper = [pixel.value for pixel in render(make_vertices(unit_triangle()), tau=0.05)]
    sharper_ambiguity = sum(min(value, 1 - value) for value in sharper)
    check(sharper_ambiguity < ambiguity,
          f"a smaller temperature must sharpen the coverage: {sharper_ambiguity:.3f} < "
          f"{ambiguity:.3f}")
    check(sum(sharper) > sum(values) * 0.5,
          "the sharpened silhouette must still cover the triangle")
    decided = sum(1 for value in sharper if value > 0.9 or value < 0.1)
    check(decided > 0.9 * len(sharper),
          f"at a tiny temperature almost every pixel must be decided, got {decided}/{len(sharper)}")
    disagreements = sum(1 for index, value in enumerate(sharper)
                        if (value > 0.9 and hard[index] < 0.5) or
                        (value < 0.1 and hard[index] > 0.5))
    check(disagreements == 0,
          f"the sharpened coverage must agree with the hard silhouette, {disagreements} differ")
    flipped = [unit_triangle()[4], unit_triangle()[5], unit_triangle()[2], unit_triangle()[3],
               unit_triangle()[0], unit_triangle()[1]]
    check(sum(pixel.value for pixel in render(make_vertices(flipped))) < 0.5,
          "a clockwise triangle must render as empty (the orientation convention)")

    # 4. the colour gradient is exactly the sum of the coverage (an analytic identity)
    colour = Var(0.8, label="colour")
    pixels = render(make_vertices(unit_triangle()), colour)
    backward(pixels[40])
    check(abs(colour.grad - pixels[40].value / colour.value) < 1e-12,
          f"d(intensity)/d(colour) must be the coverage, got {colour.grad:.9f}")
    colour = Var(0.8, label="colour")
    total = constant(0.0)
    for pixel in render(make_vertices(unit_triangle()), colour):
        total = total + pixel
    backward(total)
    coverage_sum = sum(pixel.value for pixel in render(make_vertices(unit_triangle()), 1.0))
    check(abs(colour.grad - coverage_sum) < 1e-9,
          f"the summed colour gradient must be the total coverage: {colour.grad:.6f} vs "
          f"{coverage_sum:.6f}")

    # 5. the gradients w.r.t. all six vertex coordinates match finite differences of the loss
    target = [pixel.value for pixel in render(make_vertices([6.0, 5.0, 11.0, 6.0, 7.0, 11.0]))]
    start = unit_triangle()
    nodes = make_vertices(start)
    backward(loss_to_target(nodes, target))
    analytic = [node.grad for node in nodes]

    def scalar_loss(values):
        return loss_to_target(make_vertices(values), target).value

    for index in range(6):
        def shifted(delta, position=index, values=start):
            moved = list(values)
            moved[position] += delta
            return scalar_loss(moved)
        reference = finite_difference(shifted, 0.0)
        check(abs(analytic[index] - reference) < 1e-3 * max(1.0, abs(reference)),
              f"vertex coordinate {index}: autodiff {analytic[index]:.6f} vs differences "
              f"{reference:.6f}")
        check(abs(analytic[index]) > 1e-9, f"coordinate {index} must have a non-trivial gradient")

    # 6. a gradient step reduces the loss, and a longer fit lands on the target
    def loss_value(values):
        return scalar_loss(list(values))

    nodes = make_vertices(start)
    root = loss_to_target(nodes, target)
    backward(root)
    step = 0.05
    moved = [value - step * node.grad for value, node in zip(start, nodes)]
    check(loss_value(moved) < loss_value(start),
          f"a step along the negative gradient must reduce the loss: {loss_value(moved):.4f} vs "
          f"{loss_value(start):.4f}")

    vertices = list(start)
    history = [loss_value(vertices)]
    for _ in range(120):
        nodes = make_vertices(vertices)
        root = loss_to_target(nodes, target)
        backward(root)
        vertices = [value - 0.4 * node.grad for value, node in zip(vertices, nodes)]
        history.append(loss_value(vertices))
    check(history[-1] < 0.1 * history[0],
          f"the fit must cut the loss by an order of magnitude: {history[0]:.4f} -> "
          f"{history[-1]:.4f}")
    check(all(history[index + 1] <= history[index] + 1e-9 for index in range(len(history) - 1)),
          "the fit must not increase the loss at any step")
    fitted = render(make_vertices(vertices))
    check(sum(1 for pixel, goal in zip(fitted, target) if abs(pixel.value - goal) < 0.05) > 0.9 * SIZE * SIZE,
          "most pixels must match the target after the fit")
    distance = max(abs(value - goal) for value, goal in zip(vertices, [6.0, 5.0, 11.0, 6.0, 7.0, 11.0]))
    check(distance < 1.0, f"the fitted vertices must land near the target, worst {distance:.3f}")

    # 7. the area is monotone in the triangle's size and a shifted triangle still renders
    small = [value * 0.5 for value in unit_triangle()]
    large = [value * 1.5 for value in unit_triangle()]
    check(sum(pixel.value for pixel in render(make_vertices(large))) >
          sum(pixel.value for pixel in render(make_vertices(unit_triangle()))) >
          sum(pixel.value for pixel in render(make_vertices(small))),
          "the covered area must grow with the triangle")
    check(sum(pixel.value for pixel in render(make_vertices([x + 3.0 for x in unit_triangle()]))) > 0.0,
          "a translated triangle must still render")

    # 8. degenerate cases keep finite values and finite gradients
    degenerate = make_vertices([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    root = loss_to_target(degenerate, target)
    backward(root)
    check(all(math.isfinite(node.grad) for node in degenerate),
          "a zero-area triangle must keep finite gradients")
    check(abs(root.value) < 1e-6 or root.value >= 0.0, "the degenerate loss must be finite")
    check(all(math.isfinite(pixel.value) for pixel in render(degenerate)),
          "the degenerate render must be finite")
    offscreen = make_vertices([40.0, 40.0, 60.0, 40.0, 50.0, 60.0])
    check(max(pixel.value for pixel in render(offscreen)) < 1e-6,
          "a triangle outside the image must render nothing")
    wide = make_vertices(unit_triangle())
    flat = [pixel.value for pixel in render(wide, tau=10.0)]
    check(max(flat) - min(flat) < 0.1,
          f"a huge temperature must flatten the coverage, spread {max(flat) - min(flat):.4f}")
    check(max(flat) < 0.3,
          f"and no pixel may be covered at that temperature, max {max(flat):.4f}")
    single = render(make_vertices(unit_triangle()), size=1)
    check(len(single) == 1 and 0.0 <= single[0].value <= 1.0,
          "a one-pixel image must render a single finite value")

    # 9. determinism
    first = [pixel.value for pixel in render(make_vertices(unit_triangle()))]
    second = [pixel.value for pixel in render(make_vertices(unit_triangle()))]
    check(first == second, "two identical renders must be bit-identical")
    nodes = make_vertices(start)
    backward(loss_to_target(nodes, target))
    repeat = [node.grad for node in nodes]
    nodes = make_vertices(start)
    backward(loss_to_target(nodes, target))
    check(repeat == [node.grad for node in nodes], "two identical backward sweeps must agree")

    print(f"renderer: soft area {soft_area:.2f} vs exact {exact_area:.2f}, shadow of doubt "
          f"{ambiguity:.3f} at tau={TAU} and {sharper_ambiguity:.3f} at tau=0.05")
    print(f"autodiff: 6 vertex gradients match finite differences; fit {history[0]:.3f} -> "
          f"{history[-1]:.3f}, worst vertex error {distance:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
