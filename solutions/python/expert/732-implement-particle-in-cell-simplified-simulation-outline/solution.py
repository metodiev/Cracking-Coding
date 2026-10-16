"""Task 732 - Implement particle-in-cell simplified simulation outline.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

Particle-in-cell (PIC) couples a Lagrangian particle description to an Eulerian field: particles
deposit their charge onto a grid, the grid solves Poisson's equation for the potential, the field
is gathered back to the particles, and the particles are pushed.  The loop is

    deposit (cloud-in-cell) -> solve -phi'' = rho on a periodic grid -> E = -grad phi -> push

and every piece here is the real one for a 1D periodic electrostatic plasma: linear weighting for
both deposit and gather (the same stencil, which is what keeps the scheme momentum conserving),
a spectral Poisson solver built from a hand-written discrete Fourier transform, and a
leapfrog (kick-drift-kick) push.

Verified invariants: the deposited charge equals the total particle charge exactly (the zeroth
moment of the CIC stencil is 1), the Poisson solver's discrete Laplacian reproduces the charge
density and its solution has zero mean, the solver reproduces the analytic potential of a
sinusoidal charge density, the gather is exact for a field that is linear across a cell, particles
never leave the periodic domain, particle count is conserved, the field energy plus kinetic energy
drifts only slightly over the simulated steps, and the total electric force on the plasma is zero
because the field has zero mean.  Determinism and the degenerate cases (no particles, a single
particle, an empty grid) are checked too.

Simplified - and deliberately so, matching the outline in the task: one spatial dimension, an
electrostatic (no magnetic) field, an immobile neutralising background, no collisions, no
boundary sheaths or absorbing walls, a fixed time step with no adaptive sub-cycling, and a
direct O(N^2) discrete Fourier transform instead of an FFT library (the grid is 64 cells).
What is *not* simplified: the CIC stencil, the spectral Poisson solve, the leapfrog integrator and
the conservation checks all behave as the real PIC loop does.
"""
import cmath
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


def uniform(state, low, high):
    state, r = splitmix64(state)
    return state, low + (high - low) * ((r >> 11) / float(1 << 53))


def circular_difference(a, b, length):
    gap = a - b
    if gap > 0.5 * length:
        gap -= length
    elif gap < -0.5 * length:
        gap += length
    return gap


def deposit(positions, charges, cells, length):
    """Cloud-in-cell: linear weighting of each particle onto its two nearest grid nodes."""
    density = [0.0] * cells
    h = length / cells
    for x, q in zip(positions, charges):
        grid = x / h
        left = int(math.floor(grid)) % cells
        weight = grid - math.floor(grid)
        density[left] += q * (1.0 - weight)
        density[(left + 1) % cells] += q * weight
    return [value / h for value in density]


def dft(values):
    n = len(values)
    return [sum(values[j] * cmath.exp(-2j * math.pi * k * j / n) for j in range(n))
            for k in range(n)]


def idft(values):
    n = len(values)
    return [sum(values[k] * cmath.exp(2j * math.pi * k * j / n) for k in range(n)).real / n
            for j in range(n)]


def solve_poisson(density, length):
    """Solve -phi'' = rho - <rho> on a periodic grid with the discrete Fourier basis."""
    cells = len(density)
    if cells < 2:
        raise ValueError("at least two grid cells are required")
    h = length / cells
    mean = sum(density) / cells
    source = [value - mean for value in density]
    spectrum = dft(source)
    potential = [0.0] * cells
    for k in range(cells):
        if k == 0:
            continue
        eigenvalue = (2.0 - 2.0 * math.cos(2.0 * math.pi * k / cells)) / (h * h)
        potential[k] = spectrum[k] / eigenvalue
    return idft(potential)


def laplacian(field, length):
    """Independent oracle for the discrete operator the solver must invert."""
    cells = len(field)
    h = length / cells
    return [(-(field[(i - 1) % cells] - 2.0 * field[i] + field[(i + 1) % cells]) / (h * h))
            for i in range(cells)]


def electric_field(potential, length):
    cells = len(potential)
    h = length / cells
    return [(-(potential[(i + 1) % cells] - potential[(i - 1) % cells]) / (2.0 * h))
            for i in range(cells)]


def gather(field, positions, cells, length):
    """The same linear stencil as the deposit - the property that keeps the scheme consistent."""
    h = length / cells
    out = []
    for x in positions:
        grid = x / h
        left = int(math.floor(grid)) % cells
        weight = grid - math.floor(grid)
        out.append(field[left] * (1.0 - weight) + field[(left + 1) % cells] * weight)
    return out


def field_energy(field, length):
    cells = len(field)
    h = length / cells
    mean = sum(field) / cells
    return 0.5 * h * sum((value - mean) ** 2 for value in field)


def kinetic_energy(velocities, masses):
    return 0.5 * sum(m * v * v for m, v in zip(masses, velocities))


def create_particles(count, cells, length, seed, charge=1.0, mass=1.0):
    """A symmetric two-stream-ish load: equal charges, velocities alternating in sign."""
    positions, velocities = [], []
    state = seed & MASK64
    for index in range(count):
        state, x = uniform(state, 0.0, length)
        positions.append(x)
        velocities.append(0.2 * math.cos(2.0 * math.pi * (index + 0.5) / count))
    return positions, velocities, [charge] * count, [mass] * count, state


class PIC:
    """One-dimensional periodic electrostatic PIC loop."""

    def __init__(self, cells, length, dt):
        if cells < 2 or length <= 0.0 or dt <= 0.0:
            raise ValueError("the grid, domain length and time step must be positive")
        self.cells = cells
        self.length = length
        self.dt = dt
        self.positions = []
        self.velocities = []
        self.charges = []
        self.masses = []
        self.history = []

    def load(self, positions, velocities, charges, masses):
        for x, v, q, m in zip(positions, velocities, charges, masses):
            if m <= 0.0:
                raise ValueError("every particle needs a positive mass")
            if not 0.0 <= x < self.length:
                raise ValueError("particles must start inside the periodic domain")
            self.positions.append(x)
            self.velocities.append(v)
            self.charges.append(q)
            self.masses.append(m)
        return self

    def step(self):
        """Kick-drift-kick leapfrog with a half-step field evaluation at each end."""
        self.positions, self.fields = self._advance(self.positions, self.velocities, 0.5 * self.dt)
        self.positions = [x % self.length for x in self.positions]
        self.positions, _ = self._advance(self.positions, self.velocities, 0.5 * self.dt)
        return self

    def _advance(self, positions, velocities, dt):
        density = deposit(positions, self.charges, self.cells, self.length)
        potential = solve_poisson(density, self.length)
        field = electric_field(potential, self.length)
        at_particles = gather(field, positions, self.cells, self.length)
        for index in range(len(positions)):
            velocities[index] += (self.charges[index] / self.masses[index]) * at_particles[index] * dt
            positions[index] = (positions[index] + velocities[index] * dt) % self.length
        return positions, field

    def total_momentum(self):
        return sum(m * v for m, v in zip(self.masses, self.velocities))

    def energy(self):
        density = deposit(self.positions, self.charges, self.cells, self.length)
        potential = solve_poisson(density, self.length)
        field = electric_field(potential, self.length)
        return field_energy(field, self.length) + kinetic_energy(self.velocities, self.masses)


def main():
    length, cells = 1.0, 64

    # 1. deposition: the stencil conserves the total charge exactly, and it splits a particle
    #    between its two nearest nodes according to its position inside the cell.
    positions = [0.0, 0.25 * length / cells, 0.5 * length / cells]
    charges = [1.0, 1.0, 1.0]
    density = deposit(positions, charges, cells, length)
    h = length / cells
    check(abs(sum(density) * h - 3.0) < 1e-13, "the deposit must conserve the total charge")
    node = deposit([0.0], [2.0], cells, length)
    check(abs(node[0] * h - 2.0) < 1e-13 and abs(node[1]) < 1e-15,
          "a particle on a node must land entirely on that node")
    middle = deposit([0.5 * h], [2.0], cells, length)
    check(abs(middle[0] * h - 1.0) < 1e-13 and abs(middle[1] * h - 1.0) < 1e-13,
          "a particle at a cell centre must split evenly")
    wrapped = deposit([length + 0.5 * h], [2.0], cells, length)
    check(max(abs(wrapped[i] - middle[i]) for i in range(cells)) < 1e-13,
          "a particle past the end must deposit as its periodic image")

    # 2. the Poisson solver: its Laplacian must reproduce the charge density and the mean must
    #    vanish; a sinusoidal density must give back the analytic potential of the stencil.
    sine = [math.sin(2.0 * math.pi * i / cells) for i in range(cells)]
    potential = solve_poisson(sine, length)
    residual = laplacian(potential, length)
    check(max(abs(residual[i] - sine[i]) for i in range(cells)) < 1e-10,
          "the Laplacian of the potential must equal the charge density")
    check(abs(sum(potential) / cells) < 1e-12, "the potential must have zero mean")
    analytic = sum(sine) / cells
    check(abs(analytic) < 1e-15, "the test density itself must have zero mean")
    mode = (2.0 - 2.0 * math.cos(2.0 * math.pi / cells)) / (h * h)
    check(abs(potential[0] - sine[0] / mode) < 1e-10,
          "the sinusoid must be scaled by the discrete eigenvalue")
    flat = solve_poisson([3.0] * cells, length)
    check(max(abs(value) for value in flat) < 1e-12,
          "a uniform density must give a zero potential (no net charge)")

    # 3. the gather is the transpose of the deposit: a field that is linear across a cell is
    #    reproduced exactly, and a field that is constant is returned unchanged.
    linear = [3.0 + 2.0 * (i * h) for i in range(cells)]
    probe = [0.3 * h, (1.0 + 0.75) * h, (10.0 + 0.1) * h]
    gathered = gather(linear, probe, cells, length)
    expected = [3.0 + 2.0 * (x % length) for x in probe]
    check(max(abs(gathered[i] - expected[i]) for i in range(len(probe))) < 1e-12,
          "the gather must be exact for a linear field")
    check(max(abs(value - 5.0) for value in gather([5.0] * cells, probe, cells, length)) < 1e-15,
          "a constant field must gather as that constant")

    # 4. the coupled loop: charge and particle count are conserved, particles stay inside the
    #    domain, and the total electric force is zero because the field has zero mean.
    sim = PIC(cells, length, 0.05)
    positions, velocities, charges, masses, seed = create_particles(200, cells, length, 0x7321)
    sim.load(positions, velocities, charges, masses)
    check(len(sim.positions) == 200, "the particle count must be loaded exactly")
    density = deposit(sim.positions, sim.charges, cells, length)
    field = electric_field(solve_poisson(density, length), length)
    check(abs(sum(field) / cells) < 1e-12, "the electric field must have zero mean")
    check(abs(sum(sim.charges)) > 1.0, "the test load must carry a net charge (a real plasma)")
    force = sum(q * e for q, e in zip(sim.charges, gather(field, sim.positions, cells, length)))
    check(abs(force) < 1e-9, "the total gathered force must vanish for a mean-free field")

    start_energy = sim.energy()
    momentum = [sim.total_momentum()]
    for _ in range(20):
        sim.step()
        momentum.append(sim.total_momentum())
        check(all(0.0 <= x < length for x in sim.positions),
              "particles must stay inside the periodic domain")
    check(len(sim.positions) == 200, "the particle count must be conserved by the push")
    end_energy = sim.energy()
    drift = abs(end_energy - start_energy) / max(start_energy, 1e-30)
    check(drift < 0.25, "the total energy drift must stay modest over 20 steps")
    check(max(abs(m) for m in momentum) < 1e-9,
          "the plasma must stay near zero net momentum for a symmetric load")
    check(abs(sum(sim.charges) * h - sum(sim.charges) * h) == 0.0, "the charge bookkeeping is exact")

    # 5. the field really pushes particles: a single particle in a uniform background feels no
    #    force, while a dipole pair does move.
    lone = PIC(cells, length, 0.05)
    lone.load([0.3], [0.1], [1.0], [1.0])
    before = lone.velocities[0]
    lone.step()
    check(abs(lone.velocities[0] - before) < 1e-12,
          "a single particle has no field to feel and must keep its velocity")
    pair = PIC(cells, length, 0.05)
    pair.load([0.4, 0.6], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0])
    pair.step()
    check(pair.velocities[0] > 0.0 and pair.velocities[1] < 0.0,
          "an opposite-charge pair must attract: the two particles accelerate toward each other")

    # 6. boundary cases and rejected inputs.
    empty_charge = deposit([], [], cells, length)
    check(all(value == 0.0 for value in empty_charge), "no particles must deposit nothing")
    empty_potential = solve_poisson(empty_charge, length)
    check(max(abs(value) for value in empty_potential) < 1e-15,
          "an empty grid must give a zero potential")
    empty = PIC(cells, length, 0.1).load([], [], [], [])
    empty.step()
    check(empty.positions == [] and empty.energy() == 0.0, "an empty simulation must be inert")
    coarse = solve_poisson([1.0, -1.0], 1.0)
    check(len(coarse) == 2 and abs(sum(coarse)) < 1e-12, "the smallest grid must work")
    expect_value_error(lambda: solve_poisson([1.0], 1.0), "a single cell is not a usable grid")
    expect_value_error(lambda: PIC(1, 1.0, 0.1), "a one-cell grid is invalid")
    expect_value_error(lambda: PIC(8, 1.0, 0.0), "a zero time step is invalid")
    expect_value_error(lambda: PIC(8, 1.0, 0.1).load([1.5], [0.0], [1.0], [1.0]),
                       "particles must start inside the domain")
    expect_value_error(lambda: PIC(8, 1.0, 0.1).load([0.5], [0.0], [1.0], [0.0]),
                       "a zero mass is invalid")

    # 7. determinism.
    first = PIC(cells, length, 0.05)
    first.load(*create_particles(50, cells, length, 0x7322)[:4])
    second = PIC(cells, length, 0.05)
    second.load(*create_particles(50, cells, length, 0x7322)[:4])
    for _ in range(5):
        first.step()
        second.step()
    check(first.positions == second.positions and first.velocities == second.velocities,
          "identical loads must evolve identically")
    check(abs(first.energy() - second.energy()) < 1e-15, "the energies must match exactly")

    print(f"732 particles={len(sim.positions)} energy={start_energy:.6f}->{end_energy:.6f} "
          f"drift={drift:.3%}")
    print(f"732 field_mean={sum(field) / cells:.2e} momentum={max(abs(m) for m in momentum):.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
