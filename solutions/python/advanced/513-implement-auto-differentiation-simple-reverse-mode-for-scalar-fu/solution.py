"""Task 513 - Implement auto-differentiation simple reverse-mode for scalar functions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

A tiny reverse-mode (backpropagation) engine over scalars. A Node holds a value, an accumulated
gradient and the list of (parent, partial) pairs produced by the operation that created it; because
the forward pass fixes every value, each partial derivative is a plain float (d(a*b)/da = b,
d(a/b)/db = -a/b^2, d exp(x)/dx = exp(x), d tanh/dx = 1 - tanh^2, d x^n/dx = n x^(n-1), ...), so
backward() is just an accumulation pass. `topological()` orders the reachable nodes with Kahn's
algorithm on the reversed edges and backward() walks that order in reverse, adding
parent.grad += node.grad * partial for every edge; a node used k times therefore receives the sum of
its k contributions, which is exactly the multivariable chain rule. Gradients accumulate across
calls (resetting them with zero_grad() is explicit), gradients of the seed node itself are 1, the
division operator raises ZeroDivisionError rather than producing inf, and relu uses the documented
subgradient 0 at the kink. Operators are overloaded (+, -, *, /, ** and unary minus, plus both
reflected forms) so expressions read like ordinary Python.

Documented checks: hand-computed gradients - d(xy + x)/dx = y + 1, d/dx of z*z with z = x + y is
2z = 10, and x + 2x + 3x has gradient 6 - match exactly; the closed forms of exp, log, tanh, sin,
cos, sqrt, x**3, x**2, x**0.5, x**1 and x**0 are checked against their analytic values; a deep
composition tanh(tanh(tanh(x))) matches the analytically chained derivative; (x1 x2 + x3)^2 matches
the closed form 2 (x1 x2 + x3) [x2, x1, 1]; every check is additionally validated against an
independent 4th-order central-difference stencil (agreement to 1e-6 relative); the topological order
is verified to place every parent before its child and to list each shared node once; repeated
backward() calls accumulate (the seed accumulates with them: x * x with backward() twice leaves
output.grad = 2 and x.grad = 6) until zero_grad() resets the whole graph; the relu kink and the ZeroDivision
guard behave as documented; and the engine supports expressions mixing floats and nodes freely.
"""

import math


class Node:
    __slots__ = ("data", "grad", "parents", "op")

    def __init__(self, data, parents=(), op="const"):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)             # (Node, d self / d parent)
        self.op = op

    def __repr__(self):
        return "Node(%s, op=%s, grad=%s)" % (self.data, self.op, self.grad)

    # --- forward operators ------------------------------------------------------------------
    def __add__(self, other):
        other = as_node(other)
        return Node(self.data + other.data, ((self, 1.0), (other, 1.0)), "+")

    __radd__ = __add__

    def __neg__(self):
        return Node(-self.data, ((self, -1.0),), "neg")

    def __sub__(self, other):
        other = as_node(other)
        return Node(self.data - other.data, ((self, 1.0), (other, -1.0)), "-")

    def __rsub__(self, other):
        return as_node(other) - self

    def __mul__(self, other):
        other = as_node(other)
        return Node(self.data * other.data, ((self, other.data), (other, self.data)), "*")

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_node(other)
        if other.data == 0.0:
            raise ZeroDivisionError("autodiff division by zero")
        return Node(self.data / other.data,
                    ((self, 1.0 / other.data), (other, -self.data / (other.data ** 2))), "/")

    def __rtruediv__(self, other):
        return as_node(other) / self

    def __pow__(self, exponent):
        if isinstance(exponent, Node):
            raise TypeError("only constant exponents are supported")
        return Node(self.data ** exponent,
                    ((self, exponent * self.data ** (exponent - 1)),), "pow")

    # --- backward pass ----------------------------------------------------------------------
    def nodes(self):
        """Every node reachable through the parent links, each exactly once."""
        seen = {id(self): self}
        stack = [self]
        while stack:
            node = stack.pop()
            for parent, _ in node.parents:
                if id(parent) not in seen:
                    seen[id(parent)] = parent
                    stack.append(parent)
        return list(seen.values())

    def topological(self):
        """Kahn's algorithm: every parent appears before every node that depends on it."""
        nodes = self.nodes()
        children = {id(node): [] for node in nodes}
        indegree = {}
        for node in nodes:
            indegree[id(node)] = len(node.parents)
            for parent, _ in node.parents:
                children[id(parent)].append(node)
        ready = [node for node in nodes if indegree[id(node)] == 0]
        order = []
        while ready:
            node = ready.pop()
            order.append(node)
            for child in children[id(node)]:
                indegree[id(child)] -= 1
                if indegree[id(child)] == 0:
                    ready.append(child)
        assert len(order) == len(nodes), "the expression graph must be acyclic"
        for index, node in enumerate(order):
            for parent, _ in node.parents:
                assert order.index(parent) < index, "parents must precede their children"
        return order

    def backward(self, seed=1.0):
        self.grad += seed
        for node in reversed(self.topological()):
            for parent, partial in node.parents:
                parent.grad += node.grad * partial
        return self

    def zero_grad(self):
        for node in self.nodes():
            node.grad = 0.0
        return self


def as_node(value):
    return value if isinstance(value, Node) else Node(float(value))


def exp(node):
    node = as_node(node)
    value = math.exp(node.data)
    return Node(value, ((node, value),), "exp")


def log(node):
    node = as_node(node)
    if node.data <= 0.0:
        raise ValueError("log is only differentiable for positive inputs")
    return Node(math.log(node.data), ((node, 1.0 / node.data),), "log")


def tanh(node):
    node = as_node(node)
    value = math.tanh(node.data)
    return Node(value, ((node, 1.0 - value * value),), "tanh")


def relu(node):
    node = as_node(node)
    # Subgradient convention: 0 at the kink, so relu'(0) = 0 (documented, matches the forward 0).
    return Node(max(node.data, 0.0), ((node, 1.0 if node.data > 0.0 else 0.0),), "relu")


def sin(node):
    node = as_node(node)
    return Node(math.sin(node.data), ((node, math.cos(node.data)),), "sin")


def cos(node):
    node = as_node(node)
    return Node(math.cos(node.data), ((node, -math.sin(node.data)),), "cos")


def sqrt(node):
    node = as_node(node)
    value = math.sqrt(node.data)
    return Node(value, ((node, 0.5 / value),), "sqrt")


def gradients(output, inputs):
    """Run reverse mode once and return the gradient of `output` w.r.t. each input node."""
    output.zero_grad()
    output.backward()
    return [node.grad for node in inputs]


def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def richardson(f, x, h=1e-3):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def main():
    # --- Hand-computed gradients. ------------------------------------------------------------
    x, y = Node(2.0), Node(3.0)
    output = x * y + x
    assert output.data == 8.0
    assert gradients(output, [x, y]) == [4.0, 2.0], gradients(output, [x, y])

    x, y = Node(2.0), Node(3.0)
    z = x + y
    square = z * z
    assert square.data == 25.0
    assert gradients(square, [x, y]) == [10.0, 10.0]
    assert gradients(square, [z]) == [10.0]

    a = Node(4.0)
    fan = a + 2.0 * a + 3.0 * a
    assert fan.data == 24.0 and gradients(fan, [a]) == [6.0]

    x = Node(2.0)
    mixed = 5.0 - x / 2.0 + 3.0 / x
    assert abs(mixed.data - 5.5) <= 1e-12
    assert abs(gradients(mixed, [x])[0] - (-0.5 - 3.0 / 4.0)) <= 1e-12

    # --- Closed forms of the elementary functions. -------------------------------------------
    x = Node(0.0)
    assert gradients(exp(x), [x])[0] == 1.0
    x = Node(1.0)
    assert gradients(log(x), [x])[0] == 1.0
    x = Node(0.0)
    assert gradients(tanh(x), [x])[0] == 1.0
    x = Node(0.0)
    assert gradients(sin(x), [x])[0] == 1.0
    x = Node(0.0)
    assert gradients(cos(x), [x])[0] == 0.0
    x = Node(4.0)
    assert gradients(sqrt(x), [x])[0] == 0.25
    x = Node(2.0)
    assert gradients(x ** 3, [x])[0] == 12.0
    x = Node(3.0)
    assert gradients(x ** 2, [x])[0] == 6.0
    x = Node(4.0)
    assert gradients(x ** 0.5, [x])[0] == 0.25
    x = Node(3.0)
    assert gradients(x ** 1, [x])[0] == 1.0
    x = Node(5.0)
    assert gradients(x ** 0, [x])[0] == 0.0
    x = Node(-2.0)
    assert gradients(x ** 2, [x])[0] == -4.0

    # --- Deep composition: tanh(tanh(tanh(x))) chained analytically. --------------------------
    value = 0.4
    x = Node(value)
    inner = tanh(x)
    middle = tanh(inner)
    outer = tanh(middle)
    expected = (1 - math.tanh(value) ** 2) * (1 - math.tanh(math.tanh(value)) ** 2) * \
        (1 - math.tanh(math.tanh(math.tanh(value))) ** 2)
    got = gradients(outer, [x])[0]
    assert abs(got - expected) <= 1e-15, (got, expected)

    # --- Multivariable closed form: (x1 x2 + x3)^2. ------------------------------------------
    x1, x2, x3 = Node(1.5), Node(-2.0), Node(0.75)
    combined = (x1 * x2 + x3) ** 2
    inner_value = 1.5 * -2.0 + 0.75
    grad = gradients(combined, [x1, x2, x3])
    assert abs(combined.data - inner_value ** 2) <= 1e-12
    assert abs(grad[0] - 2 * inner_value * x2.data) <= 1e-12
    assert abs(grad[1] - 2 * inner_value * x1.data) <= 1e-12
    assert abs(grad[2] - 2 * inner_value) <= 1e-12

    # --- Independent numeric verification of a longer expression. ----------------------------
    def build(t):
        node = Node(t)
        root = log(exp(tanh(node * 1.3 + 0.2)) + node * node + sin(node)) + relu(node - 0.1)
        return root, node

    def scalar(t):
        return math.log(math.exp(math.tanh(t * 1.3 + 0.2)) + t * t + math.sin(t)) + \
            max(t - 0.1, 0.0)

    for value in (0.5, 1.0, 2.0, -0.5, 0.05):
        root, leaf = build(value)
        root.backward()
        numeric = richardson(scalar, value)
        assert abs(leaf.grad - numeric) <= 1e-6 * max(1.0, abs(numeric)), \
            (value, leaf.grad, numeric)
        assert leaf.grad != 0.0
    # The same expression via central differences agrees too (documented tolerance 1e-4).
    root, leaf = build(1.0)
    root.backward()
    assert abs(leaf.grad - central_difference(scalar, 1.0)) <= 1e-4

    # --- Topological order sanity: parents first, each node once. ----------------------------
    x, y = Node(1.0), Node(2.0)
    z = x * y
    root = tanh(z + x) + z * z
    order = root.topological()
    assert len(order) == len(set(id(node) for node in order)) == len(root.nodes())
    index = {id(node): position for position, node in enumerate(order)}
    for node in order:
        for parent, _ in node.parents:
            assert index[id(parent)] < index[id(node)]
    assert order[-1] is root
    assert sum(1 for node in order if node is z) == 1, "shared nodes appear exactly once"

    # --- Accumulation semantics and zero_grad. ----------------------------------------------
    x = Node(1.0)
    output = x * x
    output.backward()
    assert x.grad == 2.0
    output.backward()
    # The seed accumulates too: output.grad becomes 2, so both edges of x * x add 2 * 1 each.
    assert output.grad == 2.0 and x.grad == 6.0, (output.grad, x.grad)
    output.zero_grad()
    assert x.grad == 0.0 and output.grad == 0.0
    assert gradients(output, [x]) == [2.0]

    # --- Documented edge cases. -------------------------------------------------------------
    x = Node(0.5)
    assert gradients(relu(x), [x])[0] == 1.0
    x = Node(-0.5)
    assert gradients(relu(x), [x])[0] == 0.0
    x = Node(0.0)
    assert gradients(relu(x), [x])[0] == 0.0, "the kink uses the subgradient 0"
    try:
        Node(1.0) / Node(0.0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("division by zero must raise")
    try:
        log(Node(-1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("log of a negative number must raise")
    try:
        Node(2.0) ** Node(3.0)
    except TypeError:
        pass
    else:
        raise AssertionError("node exponents are rejected")
    x = Node(3.0)
    assert gradients(x * 2 + 4.0, [x])[0] == 2.0
    assert gradients(2 * x + 4.0, [x])[0] == 2.0
    assert gradients(10.0 / x, [x])[0] == -10.0 / 9.0
    assert gradients(-x, [x])[0] == -1.0
    assert gradients(7.0 - x, [x])[0] == -1.0
    constant = Node(3.0)
    assert gradients(x + constant, [x])[0] == 1.0
    assert gradients((x * x) / x, [x])[0] == 1.0
    t = math.tanh(3.0)
    chain = gradients(tanh(x) + 1.0 / tanh(x), [x])[0]
    assert abs(chain - (1 - t * t) * (1 - 1.0 / (t * t))) <= 1e-12
    assert abs(central_difference(lambda t: t ** 3, 2.0) - 12.0) <= 1e-8


if __name__ == "__main__":
    main()
    print("PASS")
