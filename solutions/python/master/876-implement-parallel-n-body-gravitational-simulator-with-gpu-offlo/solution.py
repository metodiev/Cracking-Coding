"""Task 876 - Implement parallel N-body gravitational simulator with GPU offload sketch.  (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)

A deterministic N-body simulator with the structure a GPU port would expose, but executed
in process with no threads:

  * ``direct_forces``   the O(n^2) reference, accumulating partner contributions in a fixed
    canonical order (increasing partner index) so the result is bit-reproducible;
  * ``tiled_forces``    the force kernel as a *tile plan*: blocks of particles of size `block`,
    each tile computing the one-sided pair forces of every unordered pair exactly once (the
    classical symmetry saving) and applying them with opposite signs.  ``canonical_tile``
    proves the kernel is the same arithmetic by reproducing the direct per-pair values bit for
    bit, and the tile order is a deterministic function of (n, block);
  * ``OffloadPlan``     the offload sketch: how many host-device transports a tile set needs,
    with and without transfer batching, and the modelled cost
    ``transports * transfer_cost + tiles * tile_cost``.

The integrator is a kick-drift-kick leapfrog (symplectic, second order), integrated with
optional Plummer softening; a zero separation without softening is rejected instead of
producing infinities.

Verified: the tiled kernel agrees with the direct kernel, reproduces single-tile arithmetic
bit for bit, halves the pair count (n(n-1)/2 instead of n(n-1)) and balances the tiles; the
net force on the system cancels to rounding (Newton's third law in floating point); a two-body
circular orbit returns to its starting point after one analytic period with the radius,
energy, angular momentum and linear momentum all conserved to their expected orders; the
centre of mass moves uniformly; a three-body system keeps its energy within the leapfrog
bound over hundreds of steps; softening keeps a near-singular pair finite while a coincident
pair without softening is rejected; batching reduces both the transport count and the modelled
offload time; and the whole simulation is bit-deterministic.

Simplified versus a production GPU N-body code: double precision on the host instead of
single-precision device arithmetic (so the checks measure algorithmic properties, not device
rounding), a simple tile plan instead of a tuned tree/PM hybrid, and modelled instead of
measured transport latency.
"""
import math

GRAVITY = 1.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def subtract(left, right):
    return [left[0] - right[0], left[1] - right[1], left[2] - right[2]]


def add(left, right):
    return [left[0] + right[0], left[1] + right[1], left[2] + right[2]]


def scale(vector, factor):
    return [value * factor for value in vector]


def norm(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def dot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross(left, right):
    return [left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


# ------------------------------------------------------------------ force kernels
def pair_force(offset, mass_i, mass_j, softening):
    """Gravitational force on body i from body j (direction = offset = r_i - r_j)."""
    distance_squared = dot(offset, offset) + softening * softening
    if distance_squared <= 0.0:
        raise ValueError("coincident particles require softening")
    inverse_cube = 1.0 / (distance_squared * math.sqrt(distance_squared))
    factor = -GRAVITY * mass_i * mass_j * inverse_cube
    return [offset[0] * factor, offset[1] * factor, offset[2] * factor]


def direct_forces(positions, masses, softening=0.0):
    """Reference kernel; body i accumulates partners in increasing index order."""
    n = len(positions)
    forces = []
    pairs = 0
    for i in range(n):
        total = [0.0, 0.0, 0.0]
        for j in range(n):
            if i == j:
                continue
            offset = subtract(positions[i], positions[j])
            contribution = pair_force(offset, masses[i], masses[j], softening)
            total = add(total, contribution)
            pairs += 1
        forces.append(total)
    return forces, pairs


def tile_pairs(n, block):
    """Upper-triangular tile list: (row block, column block) with a <= b and a minimal."""
    if block < 1:
        raise ValueError("block size must be positive")
    blocks = (n + block - 1) // block
    tiles = []
    for a in range(blocks):
        for b in range(a, blocks):
            tiles.append((a, b))
    return tiles


def tile_members(tile, n, block):
    a, b = tile
    rows = [i for i in range(a * block, min(a * block + block, n))]
    columns = [j for j in range(b * block, min(b * block + block, n))]
    return rows, columns


def tile_pair_count(n, block, tile):
    """One-sided pair count of a tile, without touching any force arithmetic."""
    rows, columns = tile_members(tile, n, block)
    if tile[0] == tile[1]:
        return len(rows) * (len(rows) - 1) // 2
    return len(rows) * len(columns)


def canonical_tile(positions, masses, n, block, tile, softening=0.0):
    """One-sided pair forces of a single tile, in a fixed order - the device kernel."""
    rows, columns = tile_members(tile, n, block)
    contributions = []
    for i in rows:
        for j in columns:
            if i == j or (tile[0] != tile[1] and j < i):
                continue
            if tile[0] == tile[1] and j <= i:
                continue
            offset = subtract(positions[i], positions[j])
            contributions.append((i, j, pair_force(offset, masses[i], masses[j], softening)))
    return contributions


def tiled_forces(positions, masses, block, softening=0.0, verify_tile=None):
    """Full force evaluation through the tile plan.

    Every unordered pair contributes exactly once: the force on i from j is added to i, and
    its exact negative is added to j.  The per-pair antisymmetry is exact (the negated
    contribution is the same floating-point triple with flipped signs), so the system-wide
    net force cancels to accumulation rounding.
    """
    n = len(positions)
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]
    pairs = 0
    transcript = None
    for tile in tile_pairs(n, block):
        contributions = canonical_tile(positions, masses, n, block, tile, softening)
        if verify_tile is not None and tile == verify_tile:
            transcript = contributions
        for i, j, value in contributions:
            forces[i] = add(forces[i], value)
            forces[j] = add(forces[j], scale(value, -1.0))
            pairs += 1
    return forces, pairs, transcript


# ------------------------------------------------------------------ offload sketch
class OffloadPlan:
    """Deterministic model of dispatching the tile set to a device."""

    def __init__(self, n, block, batch_transfers=True, transfer_cost=40.0, tile_cost=1.0):
        if n < 0:
            raise ValueError("negative particle count")
        self.n = n
        self.block = block
        self.batch_transfers = batch_transfers
        self.transfer_cost = transfer_cost
        self.tile_cost = tile_cost
        self.tiles = tile_pairs(n, block) if n > 0 else []

    def transports(self):
        """Host-device round trips: two when batched, two per tile when issued eagerly."""
        if not self.tiles:
            return 0
        return 2 if self.batch_transfers else 2 * len(self.tiles)

    def modelled_time(self):
        return self.transports() * self.transfer_cost + len(self.tiles) * self.tile_cost

    def tile_work(self):
        return [tile_pair_count(self.n, self.block, tile) for tile in self.tiles]


# ------------------------------------------------------------------ integrator
def energy(positions, masses, velocities, softening=0.0):
    kinetic = 0.0
    for index, mass in enumerate(masses):
        kinetic += 0.5 * mass * dot(velocities[index], velocities[index])
    potential = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            offset = subtract(positions[i], positions[j])
            distance = math.sqrt(dot(offset, offset) + softening * softening)
            potential -= GRAVITY * masses[i] * masses[j] / distance
    return kinetic + potential


def momentum(masses, velocities):
    total = [0.0, 0.0, 0.0]
    for mass, velocity in zip(masses, velocities):
        total = add(total, scale(velocity, mass))
    return total


def angular_momentum(positions, masses, velocities):
    total = [0.0, 0.0, 0.0]
    for position, mass, velocity in zip(positions, masses, velocities):
        total = add(total, scale(cross(position, velocity), mass))
    return total


def leapfrog(positions, velocities, masses, steps, dt, block, softening=0.0):
    """Kick-drift-kick leapfrog, forces from the tiled kernel; returns the trajectory."""
    positions = [list(position) for position in positions]
    velocities = [list(velocity) for velocity in velocities]
    forces, _pairs, _transcript = tiled_forces(positions, masses, block, softening)
    trajectory = {"energy": [], "momentum": [], "positions": []}
    for _ in range(steps):
        velocities = [add(velocities[i], scale(forces[i], 0.5 * dt / masses[i]))
                      for i in range(len(masses))]
        positions = [add(positions[i], scale(velocities[i], dt)) for i in range(len(masses))]
        forces, _pairs, _transcript = tiled_forces(positions, masses, block, softening)
        velocities = [add(velocities[i], scale(forces[i], 0.5 * dt / masses[i]))
                      for i in range(len(masses))]
        trajectory["energy"].append(energy(positions, masses, velocities, softening))
        trajectory["momentum"].append(momentum(masses, velocities))
        trajectory["positions"].append([list(position) for position in positions])
    return positions, velocities, trajectory


def two_body_circular(separation=1.0, mass=1.0):
    """Two equal masses on a circular orbit about the centre of mass."""
    speed = math.sqrt(GRAVITY * mass / (2.0 * separation))
    positions = [(separation / 2.0, 0.0, 0.0), (-separation / 2.0, 0.0, 0.0)]
    velocities = [(0.0, speed, 0.0), (0.0, -speed, 0.0)]
    masses = [mass, mass]
    period = 2.0 * math.pi * (separation ** 1.5) / math.sqrt(GRAVITY * 2.0 * mass)
    return positions, velocities, masses, period, separation / 2.0


def main():
    # ---------------- tiled kernel: agreement, pair halving, exact third law ----------------
    positions = [(0.3, 0.1, -0.2), (-0.4, 0.5, 0.1), (0.2, -0.6, 0.4), (0.9, 0.2, -0.7),
                 (-0.2, -0.3, 0.8), (0.6, 0.4, 0.3)]
    masses = [1.2, 0.8, 1.5, 0.6, 1.1, 0.9]
    direct, direct_pairs = direct_forces(positions, masses)
    tiled, tiled_pairs, _ = tiled_forces(positions, masses, block=2)
    check(tiled_pairs == len(positions) * (len(positions) - 1) // 2,
          f"the tiled kernel must compute each pair once: {tiled_pairs}")
    check(direct_pairs == len(positions) * (len(positions) - 1),
          "the reference kernel must evaluate both directions")
    check(tiled_pairs * 2 == direct_pairs, "the symmetry saving must be exactly a factor two")
    worst = 0.0
    for i in range(len(positions)):
        magnitude = max(norm(direct[i]), 1e-300)
        worst = max(worst, norm(subtract(tiled[i], direct[i])) / magnitude)
    check(worst < 1e-12, f"tiled and direct forces disagree by {worst}")
    net = [0.0, 0.0, 0.0]
    for force in tiled:
        net = add(net, force)
    strongest = max(norm(force) for force in tiled)
    check(norm(net) <= 1e-14 * strongest,
          f"the net force must cancel to rounding, got {net} against {strongest}")

    # ---------------- the tile kernel is the same arithmetic, bit for bit ----------------
    tile = (1, 2)
    transcript = tiled_forces(positions, masses, block=2, verify_tile=tile)[2]
    rows, columns = tile_members(tile, len(positions), 2)
    check(len(transcript) == len(rows) * len(columns),
          f"a full off-diagonal tile must yield {len(rows) * len(columns)} pairs")
    for i, j, value in transcript:
        reference = pair_force(subtract(positions[i], positions[j]), masses[i], masses[j], 0.0)
        check(value == reference, f"pointwise mismatch for pair ({i}, {j})")
        check(i in rows and j in columns, "tile membership must follow the block layout")

    # ---------------- tile set geometry ----------------
    for n, block in ((6, 2), (10, 4), (16, 4), (7, 3)):
        tiles = tile_pairs(n, block)
        covered = 0
        for tile in tiles:
            covered += tile_pair_count(n, block, tile)
        check(covered == n * (n - 1) // 2,
              f"tiles must cover every unordered pair exactly once for n={n}: {covered}")
        check(all(a <= b for a, b in tiles), "the plan must keep the upper triangle only")
        check(len(tiles) == len(set(tiles)), "the plan must not repeat tiles")
    plan = OffloadPlan(64, block=8)
    work = plan.tile_work()
    check(min(work) > 0 and max(work) - min(work) <= plan.block * plan.block,
          f"tile work must be balanced: min={min(work)} max={max(work)}")
    check(sum(work) == 64 * 63 // 2, f"tile work must sum to n(n-1)/2, got {sum(work)}")

    # ---------------- offload sketch: batching saves round trips ----------------
    eager = OffloadPlan(64, block=8, batch_transfers=False)
    batched = OffloadPlan(64, block=8, batch_transfers=True)
    check(batched.transports() == 2 and eager.transports() == 2 * len(eager.tiles),
          "batched transfers must be two round trips")
    check(batched.modelled_time() < eager.modelled_time(),
          f"batching must be cheaper: {batched.modelled_time()} vs {eager.modelled_time()}")
    check(OffloadPlan(0, block=8).transports() == 0, "an empty plan needs no transport")
    check(len(OffloadPlan(64, block=8).tiles) == len(OffloadPlan(64, block=8).tiles),
          "the plan must be deterministic")

    # ---------------- two-body circular orbit against the analytic period ----------------
    positions, velocities, masses, period, radius = two_body_circular()
    steps = 512
    final_positions, final_velocities, trajectory = leapfrog(
        positions, velocities, masses, steps, period / steps, block=2)
    start = positions[0]
    return_error = norm(subtract(final_positions[0], start))
    check(return_error < 0.02,
          f"the orbit must return after one period, error {return_error}")
    final_radius = norm(subtract(final_positions[0], final_positions[1])) / 2.0
    check(abs(final_radius - radius) / radius < 1e-3,
          f"the orbital radius must be preserved: {final_radius} vs {radius}")
    energies = trajectory["energy"]
    reference_energy = energy(positions, masses, velocities)
    drift = max(abs(value - reference_energy) for value in energies) / abs(reference_energy)
    check(drift < 1e-4, f"leapfrog energy drift too large: {drift}")
    check(abs(energies[-1] - energies[0]) / abs(reference_energy) < 1e-4,
          "the energy must oscillate rather than drift")
    initial_angular = angular_momentum(positions, masses, velocities)
    final_angular = angular_momentum(final_positions, masses, final_velocities)
    check(norm(subtract(final_angular, initial_angular)) / max(norm(initial_angular), 1e-300)
          < 1e-9, "a central force must conserve angular momentum")
    start_momentum = momentum(masses, velocities)
    check(max(abs(value) for value in start_momentum) < 1e-15, "the fixture starts at rest")
    for record in trajectory["momentum"]:
        check(max(abs(value) for value in record) < 1e-12,
              f"linear momentum must stay conserved: {record}")

    # ---------------- centre of mass moves uniformly ----------------
    centre_start = [sum(masses[i] * positions[i][k] for i in range(2)) / sum(masses)
                    for k in range(3)]
    centre_end = [sum(masses[i] * final_positions[i][k] for i in range(2)) / sum(masses)
                  for k in range(3)]
    check(norm(subtract(centre_end, centre_start)) < 1e-12,
          f"the centre of mass must not drift: {centre_start} -> {centre_end}")

    # ---------------- three-body energy bound over many steps ----------------
    trio_positions = [(1.0, 0.0, 0.0), (-0.5, 0.9, 0.2), (-0.5, -0.9, -0.2)]
    trio_masses = [1.0, 0.8, 0.6]
    trio_velocities = [(0.0, 0.35, 0.05), (-0.3, -0.15, 0.0), (0.25, -0.2, -0.05)]
    trio_energy = energy(trio_positions, trio_masses, trio_velocities)
    _p, _v, trio_trajectory = leapfrog(trio_positions, trio_velocities, trio_masses, 300,
                                       0.002, block=2)
    trio_drift = max(abs(value - trio_energy) for value in trio_trajectory["energy"])
    check(trio_drift / abs(trio_energy) < 1e-3,
          f"three-body energy must stay bounded: {trio_drift / abs(trio_energy)}")
    trio_initial_momentum = momentum(trio_masses, trio_velocities)
    worst_momentum = max(abs(record[k] - trio_initial_momentum[k])
                         for record in trio_trajectory["momentum"] for k in range(3))
    check(worst_momentum < 1e-9,
          f"the three-body system must conserve momentum, drift {worst_momentum}")

    # ---------------- softening and singular configurations ----------------
    coincident = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    unit_masses = [1.0, 1.0]
    try:
        direct_forces(coincident, unit_masses, softening=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("coincident particles without softening must be rejected")
    softened, _pairs = direct_forces(coincident, unit_masses, softening=0.1)
    check(all(math.isfinite(value) for force in softened for value in force),
          "softened forces must stay finite")
    check(all(force == [0.0, 0.0, 0.0] for force in softened),
          "coincident softened particles must feel no force by symmetry")
    close = [(0.0, 0.0, 0.0), (1e-9, 0.0, 0.0)]
    softened_close, _pairs = direct_forces(close, unit_masses, softening=1e-3)
    check(all(math.isfinite(value) for force in softened_close for value in force),
          "a near-singular pair must remain finite with softening")
    check(norm(softened_close[0]) < 1e4, f"softening must cap the force: {softened_close}")
    single, single_pairs = direct_forces([(0.0, 0.0, 0.0)], [1.0])
    check(single == [[0.0, 0.0, 0.0]] and single_pairs == 0,
          "a single body must feel no force")
    empty, empty_pairs = direct_forces([], [])
    check(empty == [] and empty_pairs == 0, "an empty system must be handled")

    # ---------------- invalid parameters ----------------
    for call, why in (
        (lambda: tiled_forces(positions, masses, block=0), "zero block"),
        (lambda: tile_pairs(4, 0), "zero block in the plan"),
        (lambda: OffloadPlan(-1, 2), "negative particle count"),
        (lambda: leapfrog(positions, [[0.0, 0.0, 0.0]] * len(positions), masses, 1, 0.1,
                          block=-1), "negative block"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    # ---------------- determinism ----------------
    repeat = leapfrog(positions, velocities, masses, 32, period / 512, block=2)
    check(repeat[0] == leapfrog(positions, velocities, masses, 32, period / 512, block=2)[0],
          "the integration must be bit-deterministic")
    check(tiled_forces(positions, masses, block=3)[0] == tiled_forces(positions, masses,
                                                                      block=3)[0],
          "the tiled kernel must be deterministic")

    print(f"pairs tiled={tiled_pairs} direct={direct_pairs} worst_relative_diff={worst:.3e}")
    print(f"orbit: period={period:.4f} return_error={return_error:.3e} drift={drift:.3e} "
          f"radius={final_radius:.6f}")
    print(f"three-body drift={trio_drift / abs(trio_energy):.3e} "
          f"transport: batched={batched.transports()} eager={eager.transports()}")
    print("PASS")


if __name__ == "__main__":
    main()
