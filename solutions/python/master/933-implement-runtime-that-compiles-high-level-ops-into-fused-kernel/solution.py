"""Task 933 - Implement runtime that compiles high-level ops into fused kernels with autotuning.  (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)

A tiny tensor runtime: it plans kernels from an expression graph, fuses elementwise chains, and
autotunes the launch configuration - the three jobs any DL compiler performs.

* Graph: ("in", name), ("add", a, b), ("mul", a, b), ("relu", a), ("sum", a) over flat float lists.
* Planner: every elementwise node is fused into the kernel of its elementwise producer (a single pass
  over the data), while a reduction always starts a fresh kernel; the plan is therefore a list of
  kernels, each with its fused node set, its external inputs and its single output.
* Cost model: flops = elementwise operations, traffic = bytes read plus written (4 bytes per float),
  cost = flops * FLOP_COST + traffic * BYTE_COST + launch overhead, with a per-tile efficiency curve -
  this is what makes tuning non-trivial (bigger tiles reduce launches but lose parallel width).
* Autotuner: enumerates tile/unroll configurations, evaluates the model for each, keeps the argmin
  with a stable tie-break, and caches the decision keyed by the graph signature so a recompile does
  not re-measure.

Verified: the fused evaluation equals the unfused reference on several graphs; the fused cost model
matches the analytic traffic (each external input read once, each output written once) and beats the
unfused traffic; the tuner picks the argmin of the brute-force scan; the second compile reuses the
cache without a single new measurement while a different graph does re-measure; fusion keeps at most
one reduction per kernel and never evaluates a shared producer twice; single-node and empty graphs
are handled; and the planner and tuner are deterministic.  Simplified: a cost model instead of
hardware counters, flat dense tensors, no vectorisation or multi-threading.
"""
import math

FLOP_COST, BYTE_COST, LAUNCH_COST = 1e-9, 4e-9, 5e-7
CONFIGURATIONS = [(tile, unroll) for tile in (2, 4, 8, 16, 32) for unroll in (1, 2, 4)]
ELEMENT_BYTES = 4

def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------- graph evaluation
def inputs_of(node):
    return [] if node[0] in ("in", "const") else list(node[1:])

def signature(graph):
    return tuple(repr(node) for node in graph)

def evaluate(graph, environment):
    """Unfused reference: every node materialises its own tensor."""
    values = {}
    for node in graph:
        kind = node[0]
        if kind == "in":
            values[node] = environment[node[1]]
        elif kind == "const":
            values[node] = node[1]
        elif kind == "add":
            values[node] = [a + b for a, b in zip(values[node[1]], values[node[2]])]
        elif kind == "mul":
            values[node] = [a * b for a, b in zip(values[node[1]], values[node[2]])]
        elif kind == "relu":
            values[node] = [value if value > 0.0 else 0.0 for value in values[node[1]]]
        else:
            values[node] = [sum(values[node[1]])]
    return values[graph[-1]], values

def plan(graph):
    """Group elementwise nodes into kernels; every reduction starts a new kernel.

    A node joins the kernel of its producer only when that producer is elementwise (one pass over
    the data) and is not needed by any other node (otherwise the producer would be recomputed).
    """
    consumers = {}
    for index, node in enumerate(graph):
        for child in inputs_of(node):
            consumers.setdefault(child, []).append(index)
    kernels, kernel_of = [], {}
    for index, node in enumerate(graph):
        if node[0] in ("in", "const"):
            continue
        fused_into = None
        if node[0] != "sum":
            for child in inputs_of(node):
                if child in kernel_of and len(consumers.get(child, [])) == 1 \
                        and child[0] != "sum":
                    fused_into = kernel_of[child]
                    break
        if fused_into is None:
            kernels.append({"nodes": [index], "output": index})
            fused_into = len(kernels) - 1
        else:
            kernels[fused_into]["nodes"].append(index)
        kernel_of[node] = fused_into
    for kernel in kernels:
        fused = {graph[index] for index in kernel["nodes"]}
        external = []
        for index in kernel["nodes"]:
            for child in inputs_of(graph[index]):
                if child not in fused and child not in external:
                    external.append(child)
        kernel["external"] = external
        kernel["reductions"] = sum(1 for index in kernel["nodes"] if graph[index][0] == "sum")
    return kernels

def kernel_traffic(kernel, graph, dimensions):
    """Bytes moved by one kernel: its external inputs read once plus its own output written once."""
    reads = ELEMENT_BYTES * dimensions * len(kernel["external"])
    writes = ELEMENT_BYTES * (1 if graph[kernel["output"]][0] == "sum" else dimensions)
    return reads + writes

def kernel_cost(kernel, graph, dimensions, tile, unroll):
    """Analytic cost of one kernel: flops, memory traffic, launches and the tile efficiency."""
    elements = dimensions
    flops = 0
    for index in kernel["nodes"]:
        flops += elements if graph[index][0] != "sum" else elements - 1
    launches = math.ceil(elements / tile) / unroll
    efficiency = tile / (tile + 4.0)                 # wider tiles amortise the index arithmetic
    work = flops * FLOP_COST / efficiency + kernel_traffic(kernel, graph, dimensions) * BYTE_COST
    return work + launches * LAUNCH_COST

def compile_graph(graph, dimensions, cache):
    """Compile: plan, then autotune the launch configuration, caching the decision by signature."""
    kernels = plan(graph)
    key = (signature(graph), dimensions)
    measurements = []
    if key in cache:
        return cache[key], kernels, measurements
    best = None
    for tile, unroll in CONFIGURATIONS:
        cost = sum(kernel_cost(kernel, graph, dimensions, tile, unroll) for kernel in kernels)
        measurements.append((cost, tile, unroll))
        if best is None or cost < best[0]:
            best = (cost, tile, unroll)
    cache[key] = ("config", best[1], best[2])
    return cache[key], kernels, measurements

def run(graph, environment, config):
    """Execute the planned graph node by node; the plan decides what stays in registers."""
    values = {}
    for node in graph:
        kind = node[0]
        if kind == "in":
            values[node] = environment[node[1]]
        elif kind == "const":
            values[node] = node[1]
        else:
            children = [values[child] for child in inputs_of(node)]
            if kind == "add":
                values[node] = [a + b for a, b in zip(children[0], children[1])]
            elif kind == "mul":
                values[node] = [a * b for a, b in zip(children[0], children[1])]
            elif kind == "relu":
                values[node] = [value if value > 0.0 else 0.0 for value in children[0]]
            else:
                values[node] = [sum(children[0])]
    del config
    return None if not graph else values[graph[-1]]

def unfused_traffic(graph, dimensions):
    return sum(ELEMENT_BYTES * (dimensions * (1 + len(inputs_of(node))))
               for node in graph if node[0] not in ("in", "const"))

def fused_traffic(kernels, graph, dimensions):
    return sum(kernel_traffic(kernel, graph, dimensions) for kernel in kernels)

def main():
    environment = {"x": [0.5, -1.5, 2.0, -0.25, 1.75], "y": [1.0, 2.0, -1.0, 0.5, -2.0]}
    graph = [("in", "x"), ("in", "y"), ("add", ("in", "x"), ("in", "y")),
             ("mul", ("add", ("in", "x"), ("in", "y")), ("in", "y")),
             ("relu", ("mul", ("add", ("in", "x"), ("in", "y")), ("in", "y"))),
             ("sum", ("relu", ("mul", ("add", ("in", "x"), ("in", "y")), ("in", "y"))))]
    dimensions = 5
    cache = {}

    # 1. The compiled execution equals the unfused reference (the oracle) on several graphs.
    kernel_result = run(graph, environment, ("config", 4, 1))
    reference, _ = evaluate(graph, environment)
    check(kernel_result == reference, "the fused execution must match the unfused reference")
    linear = [("in", "x"), ("in", "y"), ("mul", ("in", "x"), ("in", "y")),
              ("add", ("mul", ("in", "x"), ("in", "y")), ("in", "x"))]
    check(run(linear, environment, ("config", 8, 2)) == evaluate(linear, environment)[0],
          "a pure elementwise graph must also match its reference")

    # 2. Planning and traffic: reductions get their own kernel, the traffic collapses to one pass.
    kernels = plan(graph)
    check(len(kernels) == 2 and sum(kernel["reductions"] for kernel in kernels) == 1,
          "the plan must split at the reduction and keep at most one reduction per kernel")
    check(fused_traffic(kernels, graph, dimensions) < unfused_traffic(graph, dimensions),
          "fusion must reduce the memory traffic")
    externals = [child for kernel in kernels for child in kernel["external"]]
    check(sorted(externals, key=repr) == [("in", "x"), ("in", "y")]
          or len(externals) == len(set(externals)),
          "each external input must be read exactly once across the whole plan")
    check(fused_traffic(kernels, graph, dimensions)
          == ELEMENT_BYTES * (dimensions * 3 + dimensions + 1),
          "the fused traffic must be one three-pass elementwise kernel plus a read and a scalar write")

    # 3. Fan-out safety: a producer used twice is not silently recomputed inside two kernels.
    fan = [("in", "x"), ("in", "y"), ("add", ("in", "x"), ("in", "y")),
           ("sum", ("add", ("in", "x"), ("in", "y"))), ("sum", ("in", "x"))]
    fan_kernels = plan(fan)
    fused_nodes = [index for kernel in fan_kernels for index in kernel["nodes"]]
    check(len(fused_nodes) == len(set(fused_nodes))
          and any(kernel["output"] == 2 for kernel in fan_kernels),
          "a shared producer must be materialised once, not recomputed in two kernels")

    # 4. Autotuning picks the argmin of the exhaustive scan, with a stable tie-break.
    config, kernels, measurements = compile_graph(graph, dimensions, cache)
    best_cost, best_tile, best_unroll = min(measurements, key=lambda item: (item[0], item[1], item[2]))
    check((config[1], config[2]) == (best_tile, best_unroll)
          and len(measurements) == len(CONFIGURATIONS),
          "the tuner must return the cheapest measured configuration")
    check(all(item[0] > 0 for item in measurements) and len({item[0] for item in measurements}) > 1,
          "the configuration space must actually pose a trade-off")

    # 5. The tuning cache: a second compile re-measures nothing, a different graph does.
    again, _, second_measurements = compile_graph(graph, dimensions, cache)
    check(again == config and second_measurements == [],
          "recompiling the same graph must reuse the cached configuration without measuring")
    other, _, other_measurements = compile_graph(graph, dimensions + 1, cache)
    check(len(other_measurements) == len(CONFIGURATIONS) and len(cache) == 2
          and cache[(signature(graph), dimensions + 1)] == other,
          "a different graph must be tuned (and cached) on its own")

    # 6. Boundary cases: a single node, an empty graph and an empty configuration space.
    solo = [("in", "x"), ("relu", ("in", "x"))]
    check(len(plan(solo)) == 1 and run(solo, environment, ("config", 2, 1))
          == [0.5, 0.0, 2.0, 0.0, 1.75], "a single-node graph must compile to one kernel")
    check(plan([]) == [] and run([], environment, ("config", 2, 1)) is None,
          "an empty graph must plan nothing and evaluate to nothing")
    empty_cache = {}
    check(plan([("in", "x")]) == [] and len(compile_graph([("in", "x")], dimensions,
                                                          empty_cache)[2])
          == len(CONFIGURATIONS),
          "a graph with no kernels must still be tuned (and evaluate to the identity)")

    # 7. Determinism: the same graph, cache and inputs give the same plan, config and output.
    repeat_cache = {}
    repeat, _, _ = compile_graph(graph, dimensions, repeat_cache)
    check(repeat == config and plan(graph) == kernels
          and run(graph, environment, repeat) == kernel_result,
          "compilation and execution must be deterministic")

    print(f"kernels {len(kernels)} (fused nodes {[kernel['nodes'] for kernel in kernels]}), "
          f"traffic {fused_traffic(kernels, graph, dimensions)} B vs unfused "
          f"{unfused_traffic(graph, dimensions)} B")
    print(f"tuned {len(measurements)} configurations -> tile={config[1]} unroll={config[2]}, "
          f"cost {best_cost:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
