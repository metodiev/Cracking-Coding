"""Task 818 - Implement model parallel training partitioner and pipeline parallel executor.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

Model parallelism splits a layered network across P devices; pipeline parallelism keeps those devices
busy by feeding microbatches.  Both halves are here, and each is checked against an independent
reference:

  * `partition_stages` assigns *contiguous* layer ranges to stages so that the heaviest stage is as
    light as possible, under a hard memory constraint (a stage's parameters plus its activations must
    fit the device).  It is a dynamic program over prefix sums, verified against a brute-force
    enumeration of every composition of the layer list into P contiguous groups, so "optimal" is
    measured rather than asserted;
  * `run_pipeline` is a tick-by-tick executor.  Every activation follows F(s, k) <- F(s - 1, k) and
    B(s, k) <- F(s, k), B(s + 1, k); a stage is a serial host (one task per tick); and the two
    schedules differ only in the *greedy rule* a free stage applies: GPipe runs all its forwards and
    then its backwards from the last microbatch backwards, while 1F1B runs a backward the moment one is
    dependency-ready and otherwise the next forward, but never keeps more than min(m, P) activations in
    flight - which is exactly the rule that makes 1F1B's peak memory the pipeline depth.

The pipeline carries real (Fraction) activations through hand-written layer forward/backward
functions, so the accumulated gradients are compared against a *closed-form* gradient of the chain
computed by a different code path (expanding the product instead of applying the chain rule), and the
loss matches exactly.

Verified: the partition covering every layer exactly once, in order, for six shapes; the DP's heaviest
stage equal to the brute-force optimum for five shapes; the memory constraint respected, and an
infeasible model (or P > L, or P = 0) rejected instead of silently over-committing a device; schedule
validity for both schedules (one task per stage per tick, every dependency finishing first, the forward
phase of GPipe taking exactly m + P - 1 ticks); the measured makespan equal to the closed form
2 * (m + P - 1) and the measured bubble fraction equal to (P - 1) / (m + P - 1) for both schedules;
GPipe holding exactly m live activations per stage while 1F1B holds at most min(m, P) and strictly
fewer whenever m > P, at no cost in makespan; the accumulated gradient equal to the closed form and
unchanged when the batch runs as one microbatch or is split in half; end-to-end runs of a 12-layer
model on 4 stages; determinism; and the degenerate shapes P = 1, m = 1 and a single-layer model.

Simplified: layers are scalar affine maps with a quadratic loss, so every gradient is an exact rational
and can be checked by hand; a stage costs one tick per layer regardless of its real FLOPs; inter-stage
communication is free (no bandwidth model - the tick count reflects the pipeline shape only); there is
no activation recomputation, no virtual/interleaved stages and no real device is touched.  The
invariants verified - contiguity, balance optimality, memory feasibility, dependency-correct
pipelining, the bubble fraction, the memory hierarchy between the two schedules and gradient
equivalence - are exactly the ones a pipeline framework must hold.
"""
from fractions import Fraction

INF = float("inf")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------ the model
class Layer:
    """A scalar affine layer with its compute and memory footprints."""

    __slots__ = ("name", "flops", "activation_mb", "param_mb", "weight", "bias")

    def __init__(self, name, flops, activation_mb, param_mb, weight=Fraction(1), bias=Fraction(0)):
        self.name = name
        self.flops = flops
        self.activation_mb = activation_mb
        self.param_mb = param_mb
        self.weight = weight
        self.bias = bias


def forward(layer, value):
    """y = w * x + b."""
    return layer.weight * value + layer.bias


def backward(layer, value, grad_out):
    """Returns (grad w, grad b, grad in) for y = w * x + b."""
    return grad_out * value, grad_out, grad_out * layer.weight


def serial_gradient(layers, batch):
    """Closed form for loss = sum over the batch of y^2, y being the last layer's output.

    The chain expands to y_j = P * x_j + sum_i T_i * b_i with P the product of all weights and T_i the
    product of the weights strictly after layer i, and the input of layer i is a_i_j = H_i * x_j +
    sum_{l < i} F_{l,i} * b_l with H_i the product of the weights before i and F_{l,i} the product of
    the weights between l and i.  Hence dL/dw_i = sum_j 2 y_j * T_i * a_i_j and dL/db_i =
    sum_j 2 y_j * T_i - with no chain rule and no backward pass anywhere."""
    weights = [layer.weight for layer in layers]
    product = Fraction(1)
    for weight in weights:
        product *= weight

    def product_range(start, end):
        out = Fraction(1)
        for weight in weights[start:end]:
            out *= weight
        return out

    tails = [product_range(index + 1, len(layers)) for index in range(len(layers))]
    heads = [product_range(0, index) for index in range(len(layers))]
    running = sum(tail * layer.bias for tail, layer in zip(tails, layers))
    outputs = [product * value + running for value in batch]
    loss = sum(y * y for y in outputs)
    inputs = []
    for index in range(len(layers)):
        offset = sum(product_range(l + 1, index) * layers[l].bias for l in range(index))
        inputs.append([heads[index] * value + offset for value in batch])
    grads_w = [sum(2 * y * tail * value for y, value in zip(outputs, inputs[index]))
               for index, tail in enumerate(tails)]
    grads_b = [sum(2 * y * tail for y in outputs) for tail in tails]
    return loss, grads_w, grads_b


def pipeline_gradient(layers, microbatches):
    """Accumulate the same gradient through the per-layer backward functions, microbatch by
    microbatch - the code path a real pipeline executor takes."""
    loss = Fraction(0)
    grads_w = [Fraction(0)] * len(layers)
    grads_b = [Fraction(0)] * len(layers)
    for microbatch in microbatches:
        activations = []
        value = microbatch
        for layer in layers:
            activations.append(value)
            value = forward(layer, value)
        loss += value * value
        grad = 2 * value
        for index in range(len(layers) - 1, -1, -1):
            grad_w, grad_b, grad = backward(layers[index], activations[index], grad)
            grads_w[index] += grad_w
            grads_b[index] += grad_b
    return loss, grads_w, grads_b


# ------------------------------------------------------------------ the partitioner
def partition_stages(layers, stages, memory_limit=None):
    """Optimal contiguous partition minimising the heaviest stage, subject to the memory limit.

    best[k][p] is the smallest achievable heaviest-stage load when the first k layers are spread over p
    stages; the range [i, k) is allowed only if its parameters plus activations fit the device."""
    count = len(layers)
    if stages < 1:
        raise ValueError("a partition needs at least one stage")
    if stages > count:
        raise ValueError(f"{stages} stages cannot hold {count} layers without an empty stage")
    flops = [0] * (count + 1)
    params = [0] * (count + 1)
    acts = [0] * (count + 1)
    for index, layer in enumerate(layers):
        flops[index + 1] = flops[index] + layer.flops
        params[index + 1] = params[index] + layer.param_mb
        acts[index + 1] = acts[index] + layer.activation_mb

    def feasible(start, end):
        if memory_limit is None:
            return True
        return (params[end] - params[start]) + (acts[end] - acts[start]) <= memory_limit

    best = [[INF] * (stages + 1) for _ in range(count + 1)]
    choice = [[-1] * (stages + 1) for _ in range(count + 1)]
    best[0][0] = 0
    for end in range(1, count + 1):
        for stage in range(1, stages + 1):
            for start in range(stage - 1, end):
                if best[start][stage - 1] == INF or not feasible(start, end):
                    continue
                candidate = max(best[start][stage - 1], flops[end] - flops[start])
                if candidate < best[end][stage]:
                    best[end][stage] = candidate
                    choice[end][stage] = start
    if best[count][stages] == INF:
        raise ValueError("no feasible partition fits the memory limit")
    ranges = []
    end = count
    for stage in range(stages, 0, -1):
        start = choice[end][stage]
        ranges.append((start, end))
        end = start
    ranges.reverse()
    return ranges


def _compositions(count, parts):
    """Every strictly increasing tuple of cut points splitting `count` items into `parts` groups."""
    if parts == 1:
        yield (0, count)
        return
    for cut in range(1, count - parts + 2):
        for rest in _compositions(count - cut, parts - 1):
            yield (0, cut) + tuple(cut + item for item in rest[1:])


def brute_force_partition(layers, stages, memory_limit=None):
    """Enumerate every contiguous composition and keep the lightest heaviest stage."""
    count = len(layers)
    if stages > count:
        raise ValueError("more stages than layers")
    flops = [layer.flops for layer in layers]
    params = [layer.param_mb for layer in layers]
    acts = [layer.activation_mb for layer in layers]
    best_peak = INF
    best_ranges = None
    for composition in _compositions(count, stages):
        ranges = [(composition[index], composition[index + 1]) for index in range(stages)]
        peak = 0
        feasible = True
        for start, end in ranges:
            peak = max(peak, sum(flops[start:end]))
            if memory_limit is not None and \
                    sum(params[start:end]) + sum(acts[start:end]) > memory_limit:
                feasible = False
                break
        if feasible and peak < best_peak:
            best_peak = peak
            best_ranges = ranges
    return best_ranges, best_peak


def stage_loads(layers, ranges):
    return [{"start": start, "end": end,
             "flops": sum(layer.flops for layer in layers[start:end]),
             "params": sum(layer.param_mb for layer in layers[start:end]),
             "acts": sum(layer.activation_mb for layer in layers[start:end])}
            for start, end in ranges]


# ------------------------------------------------------------------ the executor
def dependencies(stages, microbatches):
    """F(s,k) <- F(s-1,k); B(s,k) <- F(s,k), B(s+1,k)."""
    graph = {}
    for stage in range(stages):
        for batch in range(microbatches):
            graph[(stage, "F", batch)] = [(stage - 1, "F", batch)] if stage > 0 else []
            deps = [(stage, "F", batch)]
            if stage + 1 < stages:
                deps.append((stage + 1, "B", batch))
            graph[(stage, "B", batch)] = deps
    return graph


def run_pipeline(stages, microbatches, policy="gpipe"):
    """Tick-by-tick execution of a stage pipeline.

    gpipe: a stage runs all of its forwards (microbatches in order), then its backwards from the last
           microbatch backwards.
    1f1b:  a stage runs a backward as soon as one is dependency-ready, otherwise its next forward.

    Every stage is a serial host - one task per tick - and a task may start only once its stage is free
    and every dependency has finished, so both schedules are the same machine with a different greedy
    rule."""
    if policy not in ("gpipe", "1f1b"):
        raise ValueError(f"unknown policy {policy!r}")
    graph = dependencies(stages, microbatches)
    finish = {}
    events = []
    busy_until = {stage: 0 for stage in range(stages)}
    next_forward = {stage: 0 for stage in range(stages)}
    next_backward = {stage: microbatches - 1 for stage in range(stages)}
    started = set()
    live = set()
    peak = 0
    in_flight = min(microbatches, stages)
    tick = 0
    limit = 8 * (microbatches + stages) + 16
    while len(finish) < 2 * stages * microbatches:
        check(tick < limit, "the pipeline must drain")

        def ready(key):
            """A task may start in this tick only if every dependency finished in an earlier tick."""
            return all(dependency in finish and finish[dependency] <= tick
                       for dependency in graph[key])

        for stage in range(stages):
            if busy_until[stage] > tick:
                continue
            live_here = sum(1 for live_stage, _ in live if live_stage == stage)
            choice = None
            if policy == "gpipe":
                if next_forward[stage] < microbatches:
                    key = (stage, "F", next_forward[stage])
                    if ready(key):
                        choice = key
                if choice is None and next_backward[stage] >= 0:
                    key = (stage, "B", next_backward[stage])
                    if ready(key):
                        choice = key
            else:
                for batch in range(microbatches):
                    key = (stage, "B", batch)
                    if key in started:
                        continue
                    if ready(key):
                        choice = key
                        break
                if choice is None and next_forward[stage] < microbatches and live_here < in_flight:
                    key = (stage, "F", next_forward[stage])
                    if ready(key):
                        choice = key
            if choice is None:
                continue
            key = choice
            started.add(key)
            if key[1] == "F":
                next_forward[stage] += 1
                live.add((stage, key[2]))
            else:
                if key[2] == next_backward[stage] and policy == "gpipe":
                    next_backward[stage] -= 1
                live.discard((stage, key[2]))
            finish[key] = tick + 1
            busy_until[stage] = tick + 1
            events.append((tick, stage, key[1], key[2]))
        peak = max(peak, max((sum(1 for stage, _ in live if stage == s)
                              for s in range(stages)), default=0))
        tick += 1
    makespan = max(finish.values())
    done = len(finish)
    bubble = 1.0 - done / float(stages * makespan)
    starts = {(event_stage, kind, batch): tick
              for tick, event_stage, kind, batch in events}
    per_stage_peak = {}
    for stage in range(stages):
        top = 0
        for tick_index in range(makespan + 2):
            holder = sum(1 for batch in range(microbatches)
                         if finish[(stage, "F", batch)] <= tick_index <=
                         starts[(stage, "B", batch)])
            top = max(top, holder)
        per_stage_peak[stage] = top
    return {"events": events, "finish": finish, "makespan": makespan, "bubble": bubble,
            "peak": max(per_stage_peak.values()), "per_stage_peak": per_stage_peak}


def validate(stages, microbatches, result):
    per_tick = {}
    per_stage = {stage: [] for stage in range(stages)}
    for tick, stage, kind, batch in result["events"]:
        check(per_tick.get((stage, tick)) is None,
              f"stage {stage} ran two tasks at tick {tick}")
        per_tick[(stage, tick)] = (kind, batch)
        per_stage[stage].append((tick, kind, batch))
    graph = dependencies(stages, microbatches)
    for key, deps in graph.items():
        for dependency in deps:
            check(result["finish"][dependency] <= result["finish"][key] - 1,
                  f"{key} must start only after {dependency} finishes")
    for stage in range(stages):
        forwards = [batch for _, kind, batch in per_stage[stage] if kind == "F"]
        backwards = [batch for _, kind, batch in per_stage[stage] if kind == "B"]
        check(forwards == list(range(microbatches)),
              f"stage {stage} must run every forward exactly once, in order")
        check(sorted(backwards) == list(range(microbatches)),
              f"stage {stage} must run every backward exactly once")
        previous = -1
        for tick, _, _ in per_stage[stage]:
            check(tick > previous, "a stage must serialise its tasks")
            previous = tick


# ----------------------------------------------------------------------- checks
def sample_layers(count):
    """A deterministic heterogeneous layer list: compute and memory both vary."""
    return [Layer(f"L{index}", flops=3 + (index * 5) % 7, activation_mb=2 + index % 3,
                  param_mb=1 + (index * 3) % 5,
                  weight=Fraction(1, 2) if index % 2 else Fraction(2),
                  bias=Fraction(index - 2, 3))
            for index in range(count)]


def main():
    # 1. the partitioner covers every layer exactly once, in order, for many shapes
    for layer_count, stages in ((6, 2), (9, 3), (12, 4), (10, 5), (7, 1), (5, 5)):
        layers = sample_layers(layer_count)
        ranges = partition_stages(layers, stages)
        check(len(ranges) == stages, "the partition must have exactly `stages` ranges")
        check(ranges[0][0] == 0 and ranges[-1][1] == layer_count,
              "the partition must start at layer 0 and end at the last layer")
        for index in range(1, len(ranges)):
            check(ranges[index][0] == ranges[index - 1][1], "the ranges must be contiguous")
            check(ranges[index][0] < ranges[index][1], "a stage may not be empty")
        check(partition_stages(layers, stages) == ranges, "the partition must be deterministic")

    # 2. the dynamic program matches brute force on the heaviest stage
    for layer_count, stages in ((5, 2), (6, 3), (7, 3), (8, 4), (4, 4)):
        layers = sample_layers(layer_count)
        ranges = partition_stages(layers, stages)
        dp_peak = max(stage["flops"] for stage in stage_loads(layers, ranges))
        brute_ranges, brute_peak = brute_force_partition(layers, stages)
        check(len(brute_ranges) == stages, "the brute-force partition must use every stage")
        check(dp_peak == brute_peak,
              f"the DP must be optimal for {layer_count} layers / {stages} stages: "
              f"{dp_peak} vs {brute_peak}")
    check(len(list(_compositions(6, 3))) == 10, "6 items into 3 groups must have C(5,2) compositions")

    # 3. the memory constraint is respected, and an impossible model is rejected
    layers = sample_layers(9)
    total = sum(layer.param_mb + layer.activation_mb for layer in layers)
    tight = total // 3
    ranges = partition_stages(layers, 4, memory_limit=tight)
    for stage in stage_loads(layers, ranges):
        check(stage["params"] + stage["acts"] <= tight,
              f"stage {stage} must respect the memory limit {tight}")
    biggest = max(layer.param_mb + layer.activation_mb for layer in layers)
    try:
        partition_stages(layers, 4, memory_limit=biggest - 1)
        check(False, "a model that cannot fit must be rejected")
    except ValueError:
        pass
    try:
        partition_stages(sample_layers(3), 5)
        check(False, "more stages than layers must be rejected")
    except ValueError:
        pass
    try:
        partition_stages(sample_layers(3), 0)
        check(False, "zero stages must be rejected")
    except ValueError:
        pass

    # 4. schedule validity, makespan and bubble closed forms for both schedules
    for stages, microbatches in ((1, 4), (2, 3), (3, 5), (4, 8), (4, 3), (6, 2)):
        for policy in ("gpipe", "1f1b"):
            result = run_pipeline(stages, microbatches, policy)
            validate(stages, microbatches, result)
            makespan = result["makespan"]
            check(makespan == 2 * (microbatches + stages - 1),
                  f"{policy} P={stages} m={microbatches}: makespan {makespan} should be "
                  f"{2 * (microbatches + stages - 1)}")
            expected = (stages - 1) / float(microbatches + stages - 1)
            check(abs(result["bubble"] - expected) < 1e-9,
                  f"{policy}: bubble {result['bubble']:.4f} must equal the closed form "
                  f"{expected:.4f}")
            check(0.0 <= result["bubble"] < 1.0, "the bubble fraction must be a fraction")
            check(len(result["finish"]) == 2 * stages * microbatches,
                  "every task must run exactly once")
            if stages == 1:
                check(result["bubble"] == 0.0, "a single stage has no bubble")
    forwards_only = run_pipeline(4, 4, "gpipe")
    check(forwards_only["finish"][(3, "F", 3)] == 4 + 4 - 1,
          "GPipe's forward chain must take m + P - 1 ticks")

    # 5. memory: GPipe holds m live activations per stage, 1F1B at most min(m, P), and less when m > P
    for stages, microbatches in ((4, 8), (3, 6), (5, 5), (2, 2), (1, 3)):
        gpipe = run_pipeline(stages, microbatches, "gpipe")
        f1b = run_pipeline(stages, microbatches, "1f1b")
        check(gpipe["peak"] == microbatches,
              f"GPipe must hold m={microbatches} activations, measured {gpipe['peak']}")
        check(f1b["peak"] <= min(microbatches, stages),
              f"1F1B must hold at most min(m, P)={min(microbatches, stages)}, "
              f"measured {f1b['peak']}")
        if microbatches > stages:
            check(f1b["peak"] < gpipe["peak"],
                  f"1F1B ({f1b['peak']}) must use less memory than GPipe ({gpipe['peak']})")
        check(f1b["makespan"] == gpipe["makespan"],
              "1F1B must not cost time compared with GPipe")

    # 6. gradients: the pipeline accumulation equals the closed-form serial gradient
    layers = sample_layers(3)
    batch = [Fraction(1, 2), Fraction(-3, 4), Fraction(2), Fraction(5, 3), Fraction(-1)]
    loss, grads_w, grads_b = serial_gradient(layers, batch)
    pipe_loss, pipe_w, pipe_b = pipeline_gradient(layers, list(batch))
    check(pipe_loss == loss, f"the loss must be identical: {pipe_loss} vs {loss}")
    check(pipe_w == grads_w, f"the accumulated weights must match: {pipe_w} vs {grads_w}")
    check(pipe_b == grads_b, f"the accumulated biases must match: {pipe_b} vs {grads_b}")
    check(grads_w != [Fraction(0)] * 3, "the gradient check must be non-trivial")
    one = pipeline_gradient(layers, [batch[0]])
    check(one == serial_gradient(layers, batch[:1]),
          "a single-sample batch must match the closed form")
    total_loss = Fraction(0)
    total_w = [Fraction(0)] * 3
    for group in (batch[:1], batch[1:3], batch[3:]):
        group_loss, group_w, _ = pipeline_gradient(layers, group)
        total_loss += group_loss
        total_w = [a + b for a, b in zip(total_w, group_w)]
    check(total_loss == loss and total_w == grads_w,
          "any grouping of the batch must give the same loss and gradient")
    check(pipeline_gradient(layers, list(reversed(batch))) == (loss, grads_w, grads_b),
          "the microbatch order must not change the accumulated gradient")
    check(serial_gradient(layers, batch[:1])[1] != [Fraction(0)] * 3,
          "a single-sample batch must still produce a gradient")

    # 7. end to end: partition a 12-layer model and run both schedules on it
    layers = sample_layers(12)
    ranges = partition_stages(layers, 4)
    loads = stage_loads(layers, ranges)
    peak = max(stage["flops"] for stage in loads)
    average = sum(layer.flops for layer in layers) / 4.0
    check(peak <= average + max(layer.flops for layer in layers),
          f"the heaviest stage must stay near the average: {peak} vs {average:.1f}")
    check(sum(stage["flops"] for stage in loads) == sum(layer.flops for layer in layers),
          "the stages must account for all the compute")
    check(all(stage["end"] <= 12 for stage in loads), "no stage may address a layer past the model")
    for policy in ("gpipe", "1f1b"):
        result = run_pipeline(4, 6, policy)
        check(result["makespan"] == 2 * (6 + 3), f"{policy}: end-to-end makespan on 4 stages")
        check(len(result["finish"]) == 4 * 6 * 2, "every stage runs m forwards and m backwards")

    # 8. determinism and degenerate shapes
    check(run_pipeline(3, 4, "gpipe")["events"] == run_pipeline(3, 4, "gpipe")["events"],
          "two identical runs must produce identical events")
    solo = run_pipeline(1, 1, "gpipe")
    check(solo["makespan"] == 2 and solo["bubble"] == 0.0,
          "one stage and one microbatch: two ticks and no bubble")
    check(run_pipeline(1, 3, "1f1b")["makespan"] == 6, "one stage, three microbatches: six ticks")
    check(partition_stages(sample_layers(1), 1) == [(0, 1)], "a single-layer model has one stage")
    check(max(run_pipeline(1, 1, "1f1b")["per_stage_peak"].values()) == 1,
          "one stage never holds more than one live activation")
    try:
        run_pipeline(2, 2, "bogus")
        check(False, "an unknown policy must be rejected")
    except ValueError:
        pass

    print(f"partition: 12 layers / 4 stages -> heaviest stage {peak} flops "
          f"(average {average:.1f}); DP == brute force on 5 shapes")
    print("pipeline: makespan 2(m+P-1) and bubble (P-1)/(m+P-1) verified for "
          "(P,m) in (1,4),(2,3),(3,5),(4,8),(4,3),(6,2); 1F1B peak < GPipe peak whenever m > P")
    print("PASS")


if __name__ == "__main__":
    main()
