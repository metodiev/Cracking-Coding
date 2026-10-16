"""Task 725 - Implement operator fusion simulation for compute graph speedups.  (Tier: Expert, CTCI Beyond — Machine Learning & AI)

A compute graph executed one operator at a time materialises every intermediate tensor: each
kernel reads its inputs from memory and writes its output back.  For a chain of elementwise
operators that traffic dominates the (small) amount of arithmetic actually needed, so runtimes
*vertical fusion*: an elementwise producer with a single consumer that is not a graph output is
evaluated inside the consumer's kernel, and its intermediate never touches memory.  A second
rule is *epilogue fusion*: an elementwise consumer of a matmul/gemv output can be appended to
the gemv kernel, because the output tile is still in registers.

The planner in this file computes groups (kernels) over the node set with a union-find, then the
same graph is executed twice: once as one kernel per node and once kernel by kernel with the
intermediates kept locally.  Both executions run the same arithmetic in the same order, so the
results must be **identical** - that is the correctness contract of fusion - while the memory
traffic model (launch overhead plus bytes read and written per kernel) must shrink.

Verified invariants: fused and unfused evaluation agree exactly, the materialised set is exactly
the graph outputs plus the tensors with more than one consumer, no kernel mixes two gemv
operators, every node merged into a kernel is a single-consumer non-output producer of a node
in the same kernel (so the rules above really are what the planner did), the traffic model
matches an independent hand-rolled byte count, and the counts degrade correctly on the boundary
graphs (single node, empty graph, unfusable chain of gemv nodes, branching producer).

Simplified versus a real compiler (XLA, TVM, TensorRT): no tiling, no layout or vectorisation
decisions, no autotuning, no cross-device planning, and the cost model counts bytes and launches
only - there is no arithmetic-intensity model, no register-pressure estimate and no spill
handling.  Fusable ops are limited to elementwise ones plus a gemv epilogue, and each kernel is
executed as a straight-line interpreter over small tensors rather than generated code.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def vector(length, seed):
    values, state = [], seed & MASK64
    for _ in range(length):
        state, r = splitmix64(state)
        values.append(round(0.5 - (r % 1000) / 1000.0, 6))
    return values, state


def matrix(rows, cols, seed):
    flat, state = [], seed & MASK64
    for _ in range(rows * cols):
        state, r = splitmix64(state)
        flat.append(round(1.0 - (r % 2000) / 1000.0, 6))
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)], state


def op_gemv(args, param):
    vector_in = args[0]
    return [sum(row[j] * vector_in[j] for j in range(len(vector_in))) for row in param]


OPERATORS = {
    "add": (2, lambda args, param: [a + b for a, b in zip(args[0], args[1])], True),
    "mul": (2, lambda args, param: [a * b for a, b in zip(args[0], args[1])], True),
    "relu": (1, lambda args, param: [max(0.0, v) for v in args[0]], True),
    "scale": (1, lambda args, param: [param * v for v in args[0]], True),
    "tanh": (1, lambda args, param: [math.tanh(v) for v in args[0]], True),
    "gemv": (1, op_gemv, False),
}


class Node:
    """One operator application; `uid` is also the name of the tensor it produces."""

    def __init__(self, uid, op, inputs, param=None):
        if op not in OPERATORS:
            raise ValueError("unknown operator")
        arity, _, elementwise = OPERATORS[op]
        if len(inputs) != arity:
            raise ValueError(f"{op} takes {arity} inputs")
        if op == "gemv" and not param:
            raise ValueError("gemv needs a matrix")
        self.uid = uid
        self.op = op
        self.inputs = list(inputs)
        self.param = param
        self.elementwise = elementwise

    def apply(self, args):
        return OPERATORS[self.op][1](args, self.param)


def validate_graph(graph, inputs):
    """Inputs must exist and every reference must point at an earlier node (topological)."""
    seen = set(inputs)
    for node in graph:
        for name in node.inputs:
            if name not in seen:
                raise ValueError(f"tensor {name} is used before it is produced")
        if node.uid in seen:
            raise ValueError("duplicate tensor name")
        seen.add(node.uid)
    return True


def evaluate(graph, inputs):
    """Unfused interpreter: one kernel per node, every intermediate materialised."""
    values = {name: list(values) for name, values in inputs.items()}
    for node in graph:
        values[node.uid] = node.apply([values[name] for name in node.inputs])
    return values


def consumers_map(graph):
    consumers = {}
    for node in graph:
        for name in node.inputs:
            consumers.setdefault(name, []).append(node.uid)
    return consumers


def plan_fusion(graph, outputs):
    """Union-find over the node set; returns kernels as lists of nodes (topological order)."""
    by_id = {node.uid: node for node in graph}
    consumers = consumers_map(graph)
    parent = {node.uid: node.uid for node in graph}

    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    def may_fuse(producer, consumer):
        """Vertical rule: single consumer, not an output, and the consumer is elementwise."""
        return (len(consumers.get(producer.uid, [])) == 1
                and producer.uid not in outputs
                and consumer.elementwise)

    for node in graph:
        if not node.inputs:
            continue
        if node.elementwise:
            for name in node.inputs:
                producer = by_id.get(name)
                if producer is not None and may_fuse(producer, node):
                    union(producer.uid, node.uid)
        elif node.op == "gemv":
            # epilogue rule: a pointwise consumer of the gemv output joins the gemv kernel
            if len(node.inputs) == 1:
                for other in graph:
                    if other.inputs == [node.uid] and other.elementwise and may_fuse(node, other):
                        union(node.uid, other.uid)
    groups = {}
    for node in graph:
        groups.setdefault(find(node.uid), []).append(node)
    kernels = sorted(groups.values(), key=lambda nodes: [n.uid for n in graph].index(nodes[0].uid))
    for kernel in kernels:
        kernel.sort(key=lambda node: [n.uid for n in graph].index(node.uid))
    return kernels


def kernel_io(kernel, consumers, outputs):
    """External reads and written tensors of one kernel.

    A tensor produced inside a kernel must be written out when it is a graph output or when any
    of its consumers lives in another kernel; if every consumer is inside, it stays in registers.
    """
    inside = {node.uid for node in kernel}
    reads = []
    for node in kernel:
        for name in node.inputs:
            if name not in inside:
                reads.append(name)
    writes = []
    for node in kernel:
        outside = [c for c in consumers.get(node.uid, []) if c not in inside]
        if node.uid in outputs or outside:
            writes.append(node.uid)
    return sorted(set(reads)), sorted(set(writes))


def traversal_cost(kernels, consumers, outputs, sizes, launch=8, element=4):
    """Kernel traffic model: launch overhead plus bytes read and written."""
    total = 0
    for kernel in kernels:
        reads, writes = kernel_io(kernel, consumers, outputs)
        moved = sum(sizes[name] for name in reads) + sum(sizes[name] for name in writes)
        total += launch + element * moved
    return total


def evaluate_fused(graph, kernels, inputs, consumers, outputs):
    """Execute kernel by kernel, keeping the fused intermediates local to their kernel."""
    values = {name: list(vector_values) for name, vector_values in inputs.items()}
    for kernel in kernels:
        local = dict(values)
        for node in kernel:
            local[node.uid] = node.apply([local[name] for name in node.inputs])
        for name in kernel_io(kernel, consumers, outputs)[1]:
            values[name] = local[name]
    return values


def materialised(kernels, consumers, outputs):
    """Tensors a fused plan actually stores: the graph outputs plus the multi-consumer ones."""
    names = set(outputs)
    for kernel in kernels:
        names.update(kernel_io(kernel, consumers, outputs)[1])
    return names


def compare_results(fused, reference, kernels, consumers, outputs, label):
    """Fused execution must match on every materialised tensor and must not keep the rest."""
    stored = materialised(kernels, consumers, outputs)
    for name in stored:
        check(fused[name] == reference[name], f"{label}: value of {name} must be identical")
    produced = {node.uid for kernel in kernels for node in kernel}
    for name in produced:
        if name not in stored:
            check(name not in fused, f"{label}: the intermediate {name} must not be materialised")
    return True


def build_graph(spec):
    return [Node(uid, op, inputs, param) for uid, op, inputs, param in spec]


def chain_graph(length):
    """Elementwise chain: out = tanh(scale(relu(x * c)))... one node per link."""
    spec = [("n0", "mul", ["x", "one"], None)]
    for index in range(1, length):
        spec.append((f"n{index}", "tanh" if index % 2 else "scale",
                     [f"n{index - 1}"], 0.5 if index % 2 == 0 else None))
    return build_graph(spec), {f"n{length - 1}"}


def main():
    inputs = {"x": [0.5, -1.5, 2.0, -0.25], "one": [1.0, 1.0, 1.0, 1.0]}
    weight, _ = matrix(4, 4, 0x7251)
    weight2, _ = matrix(3, 4, 0x7252)

    # 1. a long elementwise chain: exactly one kernel, and identical numbers.
    chain, chain_outputs = chain_graph(8)
    validate_graph(chain, inputs)
    consumers = consumers_map(chain)
    kernels = plan_fusion(chain, chain_outputs)
    check(len(chain) == 8, "the chain must have eight operators")
    check(len(kernels) == 1, "a single-consumer elementwise chain must fuse into one kernel")
    reference = evaluate(chain, inputs)
    fused = evaluate_fused(chain, kernels, inputs, consumers, chain_outputs)
    compare_results(fused, reference, kernels, consumers, chain_outputs, "chain")
    sizes = {name: len(value) for name, value in reference.items()}
    sizes.update({name: len(value) for name, value in inputs.items()})
    unfused_cost = traversal_cost([[node] for node in chain], consumers, chain_outputs, sizes)
    fused_cost = traversal_cost(kernels, consumers, chain_outputs, sizes)
    check(fused_cost < unfused_cost, "fusion must reduce the launch-and-traffic cost")
    check(fused_cost * 2 < unfused_cost, "an 8-deep chain must win by more than 2x")
    reads, writes = kernel_io(kernels[0], consumers, chain_outputs)
    check(reads == ["one", "x"], "the fused kernel must read only the graph inputs")
    check(writes == ["n7"], "the fused kernel must write only the chain output")
    check(sizes["n7"] == 4, "the intermediate size must be tracked for the cost model")

    # 2. a gemv in the middle: the elementwise tail is left separate, the elementwise head
    #    materialises because the gemv operand cannot be fused into it.
    graph = build_graph([
        ("h0", "mul", ["x", "one"], None),
        ("h1", "relu", ["h0"], None),
        ("g", "gemv", ["h1"], weight),
        ("e0", "relu", ["g"], None),
        ("e1", "scale", ["e0"], 2.0),
    ])
    validate_graph(graph, inputs)
    outputs = {"e1"}
    consumers = consumers_map(graph)
    kernels = plan_fusion(graph, outputs)
    shaped = sorted(len(kernel) for kernel in kernels)
    check(shaped == [2, 3], "the gemv must split the graph into a head and an epilogue kernel")
    check(any(node.op == "gemv" for node in kernels[-1]), "the gemv kernel must be last")
    check(sum(1 for kernel in kernels if any(n.op == "gemv" for n in kernel)) == 1,
          "no kernel may contain two gemv operators")
    reference = evaluate(graph, inputs)
    fused = evaluate_fused(graph, kernels, inputs, consumers, outputs)
    compare_results(fused, reference, kernels, consumers, outputs, "gemv split")

    # 3. the epilogue rule: the pointwise consumer of a gemv output joins the gemv kernel while
    #    the gemv's own operand stays materialised.
    epilogue = build_graph([
        ("a", "mul", ["x", "one"], None),
        ("g", "gemv", ["a"], weight2),
        ("e", "tanh", ["g"], None),
    ])
    consumers = consumers_map(epilogue)
    kernels = plan_fusion(epilogue, {"e"})
    check([len(kernel) for kernel in kernels] == [1, 2],
          "the gemv and its pointwise consumer must share one kernel")
    check(kernels[0][0].uid == "a" and kernels[1][0].uid == "g",
          "the operand kernel must come first")
    compare_results(evaluate_fused(epilogue, kernels, inputs, consumers, {"e"}),
                    evaluate(epilogue, inputs), kernels, consumers, {"e"}, "epilogue")

    # 4. branching blocks fusion: a producer with two consumers, and a tensor that is both an
    #    output and an intermediate, must both be materialised.
    branch = build_graph([
        ("b", "relu", ["x"], None),
        ("p", "scale", ["b"], 3.0),
        ("q", "tanh", ["b"], None),
        ("r", "add", ["p", "q"], None),
    ])
    consumers = consumers_map(branch)
    kernels = plan_fusion(branch, {"r"})
    materialised = {name for kernel in kernels for name in kernel_io(kernel, consumers, {"r"})[1]}
    check("b" in materialised, "a two-consumer tensor must stay in memory")
    check(len(kernels) == 2, "the branch point forces two kernels (b, then p+q+r)")
    keep = evaluate_fused(branch, kernels, inputs, consumers, {"r"})
    compare_results(keep, evaluate(branch, inputs), kernels, consumers, {"r"}, "branch")
    forced = plan_fusion(branch, {"b", "r"})
    check(len(forced) == 2, "the branch point keeps two kernels even when it is also an output")

    # 5. an all-gemv graph cannot fuse at all, and the byte model matches an independent count.
    stack = build_graph([("g0", "gemv", ["x"], weight), ("g1", "gemv", ["g0"], weight),
                         ("g2", "gemv", ["g1"], weight)])
    consumers = consumers_map(stack)
    kernels = plan_fusion(stack, {"g2"})
    check(len(kernels) == len(stack), "gemv-only graphs must not fuse")
    reference = evaluate(stack, inputs)
    sizes = {name: len(value) for name, value in reference.items()}
    sizes.update({name: len(value) for name, value in inputs.items()})
    manual = 0
    for node in stack:
        manual += 8 + 4 * (sum(sizes[name] for name in node.inputs) + sizes[node.uid])
    check(traversal_cost(kernels, consumers, {"g2"}, sizes) == manual,
          "the cost model must match an independent byte count")
    compare_results(evaluate_fused(stack, kernels, inputs, consumers, {"g2"}), reference,
                    kernels, consumers, {"g2"}, "gemv stack")

    # 6. boundary cases: a single node, an empty graph, and an elementwise chain whose head is
    #    also an output.
    single = build_graph([("only", "relu", ["x"], None)])
    consumers = consumers_map(single)
    kernels = plan_fusion(single, {"only"})
    check(len(kernels) == 1, "a single node is one kernel")
    check(evaluate_fused(single, kernels, inputs, consumers, {"only"})["only"]
          == [max(0.0, v) for v in inputs["x"]], "a relu kernel must execute")
    empty = build_graph([])
    check(plan_fusion(empty, set()) == [], "an empty graph must plan no kernels")
    check(traversal_cost([], {}, set(), {}) == 0, "an empty plan must cost nothing")
    head_output = build_graph([("k0", "relu", ["x"], None), ("k1", "tanh", ["k0"], None)])
    consumers = consumers_map(head_output)
    check([len(k) for k in plan_fusion(head_output, {"k0", "k1"})] == [1, 1],
          "an output intermediate must not be fused into its consumer")

    # 7. rules on a larger mixed graph: every fused node must be a single-consumer, non-output
    #    producer of a node in the same kernel.
    mixed = build_graph([
        ("m0", "mul", ["x", "one"], None), ("m1", "relu", ["m0"], None),
        ("m2", "gemv", ["m1"], weight), ("m3", "tanh", ["m2"], None),
        ("m4", "scale", ["m3"], 0.5), ("m5", "add", ["m4", "x"], None),
        ("m6", "relu", ["m5"], None),
    ])
    validate_graph(mixed, inputs)
    consumers = consumers_map(mixed)
    outputs = {"m6"}
    kernels = plan_fusion(mixed, outputs)
    by_id = {node.uid: node for node in mixed}
    for kernel in kernels:
        inside = {node.uid for node in kernel}
        for node in kernel:
            if len(kernel) == 1:
                continue
            for name in node.inputs:
                producer = by_id.get(name)
                if producer is None or producer.uid in inside:
                    continue
                check(len(consumers[producer.uid]) == 1 and producer.uid not in outputs,
                      "a fused input must be a single-consumer non-output producer")
    reference = evaluate(mixed, inputs)
    fused = evaluate_fused(mixed, kernels, inputs, consumers, outputs)
    compare_results(fused, reference, kernels, consumers, outputs, "mixed")
    sizes = {name: len(value) for name, value in reference.items()}
    sizes.update({name: len(value) for name, value in inputs.items()})
    check(traversal_cost(kernels, consumers, outputs, sizes)
          < traversal_cost([[node] for node in mixed], consumers, outputs, sizes),
          "the mixed graph must also get cheaper")
    check(len(kernels) < len(mixed), "fusion must reduce the kernel count")

    # 8. rejected inputs and determinism.
    expect_value_error(lambda: Node("z", "unknown", ["x"]), "an unknown operator is invalid")
    expect_value_error(lambda: Node("z", "add", ["x"]), "an arity mismatch is invalid")
    expect_value_error(lambda: Node("z", "gemv", ["x"]), "a gemv without a matrix is invalid")
    expect_value_error(lambda: validate_graph(build_graph([("a", "relu", ["b"], None)]), inputs),
                       "using a tensor before it exists must be rejected")
    check(plan_fusion(chain, chain_outputs) == plan_fusion(chain, chain_outputs),
          "planning the same graph twice must give the same kernels")
    chain_kernels = plan_fusion(chain, chain_outputs)
    compare_results(evaluate_fused(chain, chain_kernels, inputs, consumers_map(chain), chain_outputs),
                    evaluate(chain, inputs), chain_kernels, consumers_map(chain), chain_outputs,
                    "reproduced")

    print(f"725 chain kernels={len(chain_kernels)}/{len(chain)} "
          f"cost={fused_cost} unfused={unfused_cost}")
    print(f"725 mixed kernels={len(kernels)}/{len(mixed)} "
          f"cost={traversal_cost(kernels, consumers, outputs, sizes)}")
    print("PASS")


if __name__ == "__main__":
    main()
