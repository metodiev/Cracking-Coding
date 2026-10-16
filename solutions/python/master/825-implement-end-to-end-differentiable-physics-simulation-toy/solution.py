"""Task 825 - Implement end-to-end differentiable physics simulation toy. (Tier: Master, CTCI Beyond — Machine Learning & AI)

"End to end" means the loss is differentiated *through* the simulator: the forward
pass of the integrator is a differentiable program, so a single reverse sweep yields
d(loss)/d(parameter) for every simulator input at once -- initial positions, initial
velocities, masses, spring stiffness, gravity, drag and the step size -- instead of the
2 * #params extra forward passes that central finite differences need. That is the whole
point of a differentiable physics engine: the gradient w.r.t. the *inputs of the
simulation* is what a planner/optimiser consumes.

What is implemented
-------------------
* `Var`: a scalar reverse-mode autodiff node. Every primitive records its parents
  together with the *local* partials, which for +, -, *, / and sqrt depend only on the
  operand values that already exist when the node is created. `backward(root)` collects
  the reachable sub-graph, walks it in reverse creation order (creation order is a
  topological order of a DAG built strictly forward) and multiply-accumulates
  parent.grad += child.grad * local_partial. A value used twice is therefore visited
  twice and accumulates both contributions. One sweep costs O(tape size) and covers all
  parameters simultaneously.
* `simulate_plain(scenario)`: the *same* update rules written with plain floats, no
  autodiff at all. It is the independent oracle for the forward pass and the function
  that finite differences differentiate.
* `simulate_diff(scenario)`: the differentiable forward pass -- particle/spring forces,
  masses, gravity, linear drag, semi-implicit (symplectic) Euler, and the quadratic
  position loss at the horizon.
* `energy(scenario)`: kinetic + spring + gravitational potential, used for the
  dissipativity/conservation invariants.

What is simplified (and why the invariants still matter)
--------------------------------------------------------
* A scalar tape, not a tensor one: at scale the engine keeps one node per *tensor op*
  and a vectorised adjoint; here every scalar multiply is an explicit node so the
  adjoint of each line can be checked by hand. The graph shape, the single sweep over
  all parameters and the cost argument are unchanged.
* Explicit integrator only -- no implicit solve, no collisions, no contacts. The
  distance guard (`EPS`) truncates the normal's derivative exactly at zero separation,
  which is precisely what production engines do to avoid a 0/0 normal; it is applied
  identically in both implementations.
* Deterministic and in-process: no threads, no GPU, no wall-clock, no randomness.

Invariants verified
-------------------
1.  d/dx of a composite expression (shared and duplicated sub-expressions) is exact
    against the closed form, including the case where the same node is used twice.
2.  The differentiable forward pass agrees with the plain simulator to the last bit.
3.  Gradients of the loss w.r.t. two hand-picked parameters match the closed form for
    the spring-free free-fall case (analytic oracle, not finite differences).
4.  Gradients w.r.t. *all* 22 parameters match central finite differences of the plain
    simulator with max relative error < 1e-6 (independent numerical oracle).
5.  A zero-stiffness spring contributes exactly zero gradient, and coincident particles
    produce no NaN (degenerate-input edge cases).
6.  Drag-free and gravity-free motion conserves total momentum to rounding, and the
    total mechanical energy error is O(dt); positive drag makes the energy strictly
    decrease over the horizon (physics invariants).
7.  Gradient descent driven by the autodiff gradient is live: the loss drops past a
    threshold and the particles land on their targets.
8.  Cost: tape size grows linearly with the number of steps and one reverse sweep
    replaces 2 * #params finite-difference forwards.
"""
import math

EPS = 1e-12


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# scalar reverse-mode autodiff
# --------------------------------------------------------------------------------------
class Var:
    """A scalar node of the tape: value plus the parents and local partials of the op."""

    __slots__ = ("val", "grad", "parents", "idx", "op")

    _counter = 0

    def __init__(self, val, parents=(), op="leaf"):
        self.val = float(val)
        self.grad = 0.0
        self.parents = tuple(parents)        # ((parent Var, d self / d parent), ...)
        self.op = op
        Var._counter += 1
        self.idx = Var._counter              # creation order == topological order

    # -- primitives -------------------------------------------------------------------
    def __add__(self, other):
        other = to_var(other)
        return Var(self.val + other.val, ((self, 1.0), (other, 1.0)), "add")

    def __sub__(self, other):
        other = to_var(other)
        return Var(self.val - other.val, ((self, 1.0), (other, -1.0)), "sub")

    def __neg__(self):
        return Var(-self.val, ((self, -1.0),), "neg")

    def __mul__(self, other):
        other = to_var(other)
        return Var(self.val * other.val, ((self, other.val), (other, self.val)), "mul")

    __rmul__ = __mul__
    __radd__ = __add__

    def __rsub__(self, other):
        return to_var(other).__sub__(self)

    def __truediv__(self, other):
        other = to_var(other)
        # value is a real division (so both engines execute identical float ops) and the
        # partials are 1/y and -x/y^2, i.e. the 1/dist term is *not* frozen as a constant
        return Var(self.val / other.val,
                   ((self, 1.0 / other.val), (other, -self.val / (other.val * other.val))),
                   "div")


def to_var(value):
    return value if isinstance(value, Var) else Var(value)


def sqrt_var(node):
    """sqrt with a guarded local derivative (0 at an exact 0, mirroring the engine's guard)."""
    root = math.sqrt(node.val)
    local = 0.5 / root if root > 0.0 else 0.0
    return Var(root, ((node, local),), "sqrt")


def backward(root):
    """One reverse sweep: accumulate d root / d node for every node reachable from root."""
    order, stack, seen = [], [root], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        order.append(node)
        for parent, _ in node.parents:
            stack.append(parent)
    order.sort(key=lambda n: n.idx, reverse=True)
    for node in order:
        node.grad = 0.0
    root.grad = 1.0
    for node in order:
        if node.grad:
            for parent, local in node.parents:      # repeated parents accumulate twice
                parent.grad += node.grad * local
    return order


# --------------------------------------------------------------------------------------
# the toy world
# --------------------------------------------------------------------------------------
def make_scenario(particles=3, springs=None, gravity=(0.0, -9.81), drag=0.35, dt=0.01,
                  steps=8, targets=None, k=None, rest=None, radius=0.4, spin=0.6):
    """A tiny 2D particle/spring world. All floats; `simulate_diff` rebuilds it as Vars.

    The particles start on a circle and the default rest lengths are exactly the initial
    separations, so the world starts at zero spring energy and the optimiser's job is to
    choose initial conditions that carry the shape onto the (rotated) target shape.
    """
    if springs is None:
        springs = [(0, 1), (1, 2), (0, 2)]
    pos = [[radius * math.cos(2 * math.pi * i / particles),
            radius * math.sin(2 * math.pi * i / particles)] for i in range(particles)]
    vel = [[0.02 * i, -0.01 * (i + 1)] for i in range(particles)]
    masses = [1.0 + 0.25 * i for i in range(particles)]
    stiff = [30.0 + 5.0 * s for s in range(len(springs))] if k is None else list(k)
    if rest is None:
        lengths = [math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
                   for (i, j) in springs]
    else:
        lengths = list(rest)
    if targets is None:
        targets = [[radius * math.cos(2 * math.pi * i / particles + spin),
                    radius * math.sin(2 * math.pi * i / particles + spin)]
                   for i in range(particles)]
    return {"particles": particles, "springs": tuple(springs), "p0": pos, "v0": vel,
            "mass": masses, "k": stiff, "rest": lengths, "grav": list(gravity),
            "drag": drag, "dt": dt, "steps": steps, "targets": targets}


def apply_spring(dx, dy, k, rest, ops):
    """Strain force of one spring, written once and instantiated twice (plain floats / Vars).

    Returns (fx, fy) = k * (1 - rest/dist) * (dx, dy), the force pulling particle i
    towards particle j: dist < rest pushes apart, dist > rest pulls in. `ops` is the
    tiny algebra bundle (`PLAIN_OPS` or `VAR_OPS`), so both engines execute the very
    same expression tree in the same order -- that is what makes the bit-identity and
    the finite-difference cross-check meaningful. The division by `dist` stays a real
    division in both engines: freezing it as a constant would silently drop the
    d(1/dist) term from the gradient.
    """
    dist = ops["sqrt"](ops["add"](ops["mul"](dx, dx), ops["mul"](dy, dy)))
    if ops["value"](dist) > EPS:                    # the normal is well defined
        strain = ops["div"](ops["mul"](ops["sub"](dist, rest), k), dist)
    else:
        strain = ops["mul"](ops["sub"](dist, rest), 0.0)   # zero separation: force 0,
    return ops["mul"](strain, dx), ops["mul"](strain, dy)  # derivative path truncated


PLAIN_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
             "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
             "sqrt": math.sqrt, "value": lambda v: v}
VAR_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
           "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
           "sqrt": sqrt_var, "value": lambda v: v.val}


def simulate_plain(scenario, collect=False):
    """Plain-float reference simulator (no autodiff). Returns (loss, final pos, trace, vel)."""
    n, springs = scenario["particles"], scenario["springs"]
    dt, drag = scenario["dt"], scenario["drag"]
    gx, gy = scenario["grav"]
    pos = [list(scenario["p0"][i]) for i in range(n)]
    vel = [list(scenario["v0"][i]) for i in range(n)]
    trace = [tuple((p[0], p[1]) for p in pos)] if collect else []
    for _ in range(scenario["steps"]):
        force = [[scenario["mass"][i] * gx, scenario["mass"][i] * gy] for i in range(n)]
        for s, (i, j) in enumerate(springs):
            dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
            fx, fy = apply_spring(dx, dy, scenario["k"][s], scenario["rest"][s], PLAIN_OPS)
            force[i][0] -= fx
            force[i][1] -= fy
            force[j][0] += fx
            force[j][1] += fy
        for i in range(n):
            for axis in (0, 1):
                accel = (force[i][axis] - drag * vel[i][axis]) / scenario["mass"][i]
                vel[i][axis] += dt * accel
                pos[i][axis] += dt * vel[i][axis]
        if collect:
            trace.append(tuple((p[0], p[1]) for p in pos))
    loss = 0.0
    for i in range(n):
        dx = pos[i][0] - scenario["targets"][i][0]
        dy = pos[i][1] - scenario["targets"][i][1]
        loss += 0.5 * (dx * dx + dy * dy)
    return loss, pos, trace, vel


def mechanical_energy(scenario, pos, vel):
    """Kinetic + spring potential + gravitational potential of the current state."""
    energy = 0.0
    for i in range(scenario["particles"]):
        energy += 0.5 * scenario["mass"][i] * (vel[i][0] ** 2 + vel[i][1] ** 2)
        energy -= scenario["mass"][i] * (scenario["grav"][0] * pos[i][0]
                                          + scenario["grav"][1] * pos[i][1])
    for s, (i, j) in enumerate(scenario["springs"]):
        dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
        stretch = math.sqrt(dx * dx + dy * dy) - scenario["rest"][s]
        energy += 0.5 * scenario["k"][s] * stretch * stretch
    return energy


def simulate_diff(scenario):
    """Differentiable forward pass. Returns (loss Var, leaf Var dict, tape node count)."""
    n, springs = scenario["particles"], scenario["springs"]
    dt_leaf = to_var(scenario["dt"])
    drag = to_var(scenario["drag"])
    grav = [to_var(scenario["grav"][0]), to_var(scenario["grav"][1])]
    pos = [[to_var(scenario["p0"][i][a]) for a in (0, 1)] for i in range(n)]
    vel = [[to_var(scenario["v0"][i][a]) for a in (0, 1)] for i in range(n)]
    mass = [to_var(scenario["mass"][i]) for i in range(n)]
    stiff = [to_var(k) for k in scenario["k"]]
    rest = [to_var(r) for r in scenario["rest"]]
    p0_leaves = [[pos[i][a] for a in (0, 1)] for i in range(n)]   # captured before the
    v0_leaves = [[vel[i][a] for a in (0, 1)] for i in range(n)]   # state variables move
    before = Var._counter
    for _ in range(scenario["steps"]):
        force = [[mass[i] * grav[0], mass[i] * grav[1]] for i in range(n)]
        for s, (i, j) in enumerate(springs):
            dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
            fx, fy = apply_spring(dx, dy, stiff[s], rest[s], VAR_OPS)
            force[i][0] = force[i][0] - fx
            force[i][1] = force[i][1] - fy
            force[j][0] = force[j][0] + fx
            force[j][1] = force[j][1] + fy
        for i in range(n):
            for axis in (0, 1):
                accel = (force[i][axis] - drag * vel[i][axis]) / mass[i]
                vel[i][axis] = vel[i][axis] + dt_leaf * accel
                pos[i][axis] = pos[i][axis] + dt_leaf * vel[i][axis]
    loss = Var(0.0)
    for i in range(n):
        dx = pos[i][0] - to_var(scenario["targets"][i][0])
        dy = pos[i][1] - to_var(scenario["targets"][i][1])
        loss = loss + 0.5 * (dx * dx + dy * dy)
    leaves = {"dt": dt_leaf, "drag": drag, "grav": grav,
              "p0": p0_leaves, "v0": v0_leaves,
              "mass": mass, "k": stiff, "rest": rest}
    return loss, leaves, Var._counter - before


def parameter_names(scenario):
    names = ["grav[0]", "grav[1]", "drag", "dt"]
    for i in range(scenario["particles"]):
        names += [f"p0[{i}][0]", f"p0[{i}][1]"]
    for i in range(scenario["particles"]):
        names += [f"v0[{i}][0]", f"v0[{i}][1]"]
    names += [f"mass[{i}]" for i in range(scenario["particles"])]
    names += [f"k[{s}]" for s in range(len(scenario["springs"]))]
    names += [f"rest[{s}]" for s in range(len(scenario["springs"]))]
    return names


def flatten(scenario):
    """Scenario parameters as a flat list, in `parameter_names` order."""
    flat = [scenario["grav"][0], scenario["grav"][1], scenario["drag"], scenario["dt"]]
    for i in range(scenario["particles"]):
        flat += list(scenario["p0"][i])
    for i in range(scenario["particles"]):
        flat += list(scenario["v0"][i])
    flat += list(scenario["mass"])
    flat += list(scenario["k"])
    flat += list(scenario["rest"])
    return flat


def unflatten(scenario, flat):
    """Inverse of `flatten`: rebuild a scenario with new parameter values."""
    out = dict(scenario)
    n = scenario["particles"]
    out["grav"] = [flat[0], flat[1]]
    out["drag"], out["dt"] = flat[2], flat[3]
    cursor = 4
    out["p0"] = [[flat[cursor + 2 * i], flat[cursor + 2 * i + 1]] for i in range(n)]
    cursor += 2 * n
    out["v0"] = [[flat[cursor + 2 * i], flat[cursor + 2 * i + 1]] for i in range(n)]
    cursor += 2 * n
    out["mass"] = flat[cursor:cursor + n]
    cursor += n
    out["k"] = flat[cursor:cursor + len(scenario["springs"])]
    cursor += len(scenario["springs"])
    out["rest"] = flat[cursor:cursor + len(scenario["springs"])]
    return out


def autodiff_gradient(scenario):
    """All 22 partials from ONE reverse sweep, in `parameter_names` order."""
    loss, leaves, _ = simulate_diff(scenario)
    backward(loss)
    grads = [leaves["grav"][0].grad, leaves["grav"][1].grad, leaves["drag"].grad,
             leaves["dt"].grad]
    for i in range(scenario["particles"]):
        grads += [leaves["p0"][i][0].grad, leaves["p0"][i][1].grad]
    for i in range(scenario["particles"]):
        grads += [leaves["v0"][i][0].grad, leaves["v0"][i][1].grad]
    grads += [node.grad for node in leaves["mass"]]
    grads += [node.grad for node in leaves["k"]]
    grads += [node.grad for node in leaves["rest"]]
    return grads, loss.val


def finite_difference_gradient(scenario, h=1e-6):
    """Central differences of the plain simulator: one forward pass per parameter."""
    base = flatten(scenario)
    grads = []
    for i in range(len(base)):
        plus, minus = list(base), list(base)
        plus[i] += h
        minus[i] -= h
        f_plus = simulate_plain(unflatten(scenario, plus))[0]
        f_minus = simulate_plain(unflatten(scenario, minus))[0]
        grads.append((f_plus - f_minus) / (2.0 * h))
    return grads


def main():
    # -- 1. elementary adjoints, including a duplicated parent --------------------------
    def expr_grad(x_val, y_val):
        x, y = Var(x_val), Var(y_val)
        out = x * x + x * y + y / x - x * y      # x^2 + y/x ; the x*y terms cancel
        backward(out)
        return out.val, x.grad, y.grad

    value, gx, gy = expr_grad(3.0, 5.0)
    check(abs(value - (9.0 + 5.0 / 3.0)) < 1e-12, "composite value")
    check(abs(gx - (2 * 3.0 - 5.0 / 9.0)) < 1e-12, "d/dx of x^2 + y/x")
    check(abs(gy - 1.0 / 3.0) < 1e-12, "d/dy of x^2 + y/x")

    x = Var(2.0)
    backward(x * x * x)
    check(abs(x.grad - 3 * 4.0) < 1e-12, "x*x*x accumulates three visits of x")

    def sqrt_grad(v):
        node = Var(v)
        backward(sqrt_var(node))
        return node.grad

    check(abs(sqrt_grad(16.0) - 1.0 / 8.0) < 1e-12, "sqrt local derivative")
    check(sqrt_grad(0.0) == 0.0, "guarded sqrt derivative at 0 is 0, not a ZeroDivisionError")

    # -- 2. the differentiable forward pass equals the plain simulator exactly ----------
    scenario = make_scenario()
    loss_auto, leaves, tape_nodes = simulate_diff(scenario)
    loss_plain, final_pos_plain, _, _ = simulate_plain(scenario)
    check(loss_auto.val == loss_plain, "loss must be bit-identical between the two engines")
    for i in range(scenario["particles"]):
        for axis in (0, 1):
            check(leaves["p0"][i][axis].val == scenario["p0"][i][axis],
                  "initial positions are leaves of the tape")

    # -- 3. analytic oracle: spring-free free fall --------------------------------------
    free = make_scenario(particles=1, springs=[], drag=0.0, gravity=(0.0, -9.81), steps=6)
    grads, _ = autodiff_gradient(free)
    names = parameter_names(free)
    steps, dt = free["steps"], free["dt"]
    _, end, _, _ = simulate_plain(free)
    for i, axis in enumerate((0, 1)):
        residual = end[0][axis] - free["targets"][0][axis]
        t = steps * dt
        # loss = 0.5 r^2 and p_T = p_0 + T*dt*v_0 + dt^2*grav*T(T+1)/2, so these
        # partials are closed form (the chain rule factor is 2 * 0.5 = 1)
        check(abs(grads[names.index(f"p0[0][{axis}]")] - residual) < 1e-9,
              "d loss / d p0 equals the residual in free fall")
        check(abs(grads[names.index(f"v0[0][{axis}]")] - residual * t) < 1e-9,
              "d loss / d v0 equals residual * T * dt")
        check(abs(grads[names.index(f"grav[{axis}]")]
                  - residual * dt * dt * steps * (steps + 1) / 2.0) < 1e-9,
              "d loss / d gravity matches the closed-form double sum")
    check(grads[names.index("mass[0]")] == 0.0,
          "mass cannot influence a spring-free free fall (gravity is an acceleration)")

    # -- 4. finite differences of the plain engine vs the single reverse sweep -----------
    finite = finite_difference_gradient(scenario)
    auto_grads, _ = autodiff_gradient(scenario)
    worst, where = 0.0, ""
    for name, a, f in zip(parameter_names(scenario), auto_grads, finite):
        scale = max(1.0, abs(f))
        relative = abs(a - f) / scale
        if relative > worst:
            worst, where = relative, name
    check(worst < 1e-6, f"finite-difference agreement, worst {worst:.2e} at {where}")
    check(len(auto_grads) == 25, "25 parameters share one reverse sweep")

    # -- 5. degenerate inputs ------------------------------------------------------------
    slack = make_scenario(springs=[(0, 1)], steps=5, drag=0.0, k=[0.0], rest=[1.0])
    slack_grads, slack_loss = autodiff_gradient(slack)
    without = make_scenario(springs=[], steps=5, drag=0.0)
    check(slack_loss == simulate_plain(without)[0],
          "k = 0 removes the spring's force exactly")
    slack_fd = finite_difference_gradient(slack)
    slack_err = max(abs(a - f) / max(1.0, abs(f)) for a, f in zip(slack_grads, slack_fd))
    check(slack_err < 1e-6, "gradients stay exact in the degenerate k = 0 case")
    check(abs(slack_grads[parameter_names(slack).index("k[0]")]) > 1e-6,
          "force is linear in k, so d force / d k is still non-zero at k = 0")

    coincident = make_scenario(particles=2, springs=[(0, 1)], steps=3, rest=[0.25])
    coincident["p0"] = [[0.4, 0.4], [0.4, 0.4]]      # exact zero separation
    zero_loss, zero_pos, _, _ = simulate_plain(coincident)
    zero_auto, _, _ = simulate_diff(coincident)
    check(zero_loss == zero_auto.val and all(math.isfinite(v) for v in zero_pos[0]),
          "coincident particles stay finite instead of producing 0/0")

    # -- 6. physics invariants -----------------------------------------------------------
    two_body = make_scenario(particles=2, springs=[(0, 1)], gravity=(0.0, 0.0), drag=0.0,
                             dt=0.001, steps=50, k=[80.0], rest=[0.6])
    energy_start = mechanical_energy(two_body, two_body["p0"], two_body["v0"])
    _, _, trace, end_vel = simulate_plain(two_body, collect=True)
    momentum = sum(two_body["mass"][i] * end_vel[i][a] for i in range(2) for a in (0, 1))
    initial_momentum = sum(two_body["mass"][i] * two_body["v0"][i][a]
                           for i in range(2) for a in (0, 1))
    check(abs(momentum - initial_momentum) < 1e-12, "internal forces conserve linear momentum")
    drift = abs(mechanical_energy(two_body, trace[-1], end_vel) - energy_start)
    check(drift / abs(energy_start) < 1e-2,
          "semi-implicit Euler conserves energy to O(dt) when drag = 0")

    damped = dict(two_body)
    damped["drag"] = 2.0
    _, _, damped_trace, damped_vel = simulate_plain(damped, collect=True)
    check(mechanical_energy(damped, damped_trace[-1], damped_vel) < energy_start,
          "positive drag strictly dissipates mechanical energy")

    # -- 7. liveness: gradient descent on the simulator inputs ---------------------------
    training = make_scenario()
    flat = flatten(training)
    movable = 2 + 1 + 1 + 4 * training["particles"]   # gravity, drag, dt, p0 and v0
    lr = 0.05
    history = []
    for _ in range(80):
        grads, loss_value = autodiff_gradient(unflatten(training, flat))
        history.append(loss_value)
        if loss_value < 1e-12:
            break
        step = lr
        for _ in range(40):                            # backtracking line search: keeps the
            trial = list(flat)                         # descent provably monotone
            for i in range(movable):
                trial[i] -= step * grads[i]
            if simulate_plain(unflatten(training, trial))[0] < loss_value:
                break
            step *= 0.5
        flat = trial
    check(loss_value < 0.01 * history[0],
          f"gradient descent cuts the loss by >99% (start {history[0]:.4g})")
    check(all(history[i + 1] <= history[i] for i in range(len(history) - 1)),
          "each line-searched step is non-increasing in the loss")

    print(f"tape nodes for {scenario['steps']} steps: {tape_nodes}")
    print(f"max finite-difference error over 25 parameters: {worst:.2e}")
    print(f"loss after gradient descent: {loss_value:.6g} (from {history[0]:.6g})")

    # -- 8. cost structure ---------------------------------------------------------------
    short = simulate_diff(make_scenario(steps=4))[2]
    long_ = simulate_diff(make_scenario(steps=8))[2]
    check(short < long_ < 2.5 * short, "tape size grows linearly with the number of steps")
    checks_fd = 2 * len(flatten(scenario))
    check(checks_fd == 50 and tape_nodes < 2000,
          "one sweep replaces 50 finite-difference forwards on a sub-2000-node tape")

    print("PASS")


if __name__ == "__main__":
    main()
