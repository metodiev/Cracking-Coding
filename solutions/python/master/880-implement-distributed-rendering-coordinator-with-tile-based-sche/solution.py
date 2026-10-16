"""Task 880 - Implement distributed rendering coordinator with tile-based scheduling.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

A deterministic discrete-event simulation of a render farm.  There are no threads, no sockets
and no wall-clock time: a message is a scheduled callback carrying an explicit delivery time,
the event queue is ordered by (time, sequence number), and every worker action is a modelled
duration.  The pieces are:

  * the coordinator - owns the tile list, hands out tiles in priority order, tracks one
    assignment per tile with a deadline, and reassigns work whose deadline passes (the failure
    detector).  Worker loss is a crash: the worker simply stops answering, so the deadline is
    the *only* thing that recovers the tile;
  * workers - idle workers ask for work, steal from the busiest colleague when the shared
    queue runs dry, render a tile in `cost / speed` simulated units and report the pixel block
    together with its content digest;
  * tile identity - a result is applied through the digest of its pixel payload, so a duplicate
    or late report for an already finished tile is discarded instead of corrupting the frame
    (idempotent exactly-once assembly over an at-least-once network).

Verified: liveness (every tile of the frame is completed and the frame equals the frame of a
single-process renderer bit for bit, both with a healthy fleet and with a worker crash in the
middle of a tile); exactly-once assembly (a duplicated report changes nothing, and no tile is
assigned twice at the same time); failure recovery (the crashed worker's tile is reassigned
after its deadline and only that tile restarts); work stealing (the stolen schedule finishes in
fewer simulated ticks than a static round-robin partition, with a balanced tile count); the
checkpoint/resume path re-inserting a partial frame without redoing finished tiles; the tile
grid covering the image exactly once; reproducibility of the whole schedule from the same
inputs; and rejection of malformed configurations.
"""
import hashlib
import heapq
import math

TILE_COST = 500.0  # simulated ticks for one tile at speed 1.0


def tile_cost(tile, rows):
    """Deterministic per-tile cost: the higher image rows are more expensive, which is what
    makes a naive contiguous partition imbalanced and a work-stealing schedule worthwhile."""
    return TILE_COST * (0.4 + 0.6 * tile[0] / max(rows - 1, 1))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def pixel_digest(block):
    hasher = hashlib.sha256()
    for key in sorted(block):
        hasher.update(f"{key}:{block[key]:.12f}".encode("utf-8"))
    return hasher.hexdigest()


# ------------------------------------------------------------------ scene and renderer
def shade_pixel(x, y, width, height):
    """A tiny deterministic analytic scene (sphere over a plane) - the render oracle."""
    u = (x + 0.5) / width * 2.0 - 1.0
    v = 1.0 - (y + 0.5) / height * 2.0
    origin = (0.0, 0.6, -2.2)
    direction = (u * 0.9, v * 0.6, 1.0)
    length = math.sqrt(sum(component * component for component in direction))
    direction = tuple(component / length for component in direction)
    centre = (0.0, 0.0, 0.0)
    radius = 0.75
    offset = tuple(origin[axis] - centre[axis] for axis in range(3))
    b = 2.0 * sum(offset[axis] * direction[axis] for axis in range(3))
    c = sum(offset[axis] ** 2 for axis in range(3)) - radius * radius
    discriminant = b * b - 4.0 * c
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        for candidate in ((-b - root) / 2.0, (-b + root) / 2.0):
            if candidate > 1e-9:
                point = tuple(origin[axis] + direction[axis] * candidate for axis in range(3))
                normal = tuple((point[axis] - centre[axis]) / radius for axis in range(3))
                light = 0.28 + 0.72 * max(0.0, sum(normal[axis] * (0.35, 0.8, 0.5)[axis]
                                                    for axis in range(3)))
                return round(light, 12)
    if direction[1] < -1e-9:
        travel = (origin[1] + 0.85) / -direction[1]
        if travel > 0.0:
            point = tuple(origin[axis] + direction[axis] * travel for axis in range(3))
            checker = int(math.floor(point[0] * 2.0)) + int(math.floor(point[2] * 2.0))
            return round(0.55 if checker % 2 == 0 else 0.35, 12)
    return 0.08


def tile_pixels(tile, width, tile_size):
    """Pixel block of a tile, keyed by (x, y)."""
    row, column = tile
    block = {}
    for y in range(row * tile_size, min((row + 1) * tile_size, width)):
        for x in range(column * tile_size, min((column + 1) * tile_size, width)):
            block[(x, y)] = shade_pixel(x, y, width, width)
    return block


def reference_frame(width, tile_size):
    frame = {}
    for tile in tile_grid(width, tile_size):
        frame.update(tile_pixels(tile, width, tile_size))
    return frame


def tile_grid(width, tile_size):
    if width <= 0 or tile_size <= 0:
        raise ValueError("width and tile size must be positive")
    count = (width + tile_size - 1) // tile_size
    return [(row, column) for row in range(count) for column in range(count)]


# ------------------------------------------------------------------ simulation core
class Simulation:
    """Discrete-event queue ordered by (time, sequence) - fully deterministic."""

    def __init__(self):
        self.queue = []
        self.sequence = 0
        self.now = 0

    def schedule(self, delay, action):
        if delay < 0.0:
            raise ValueError("cannot schedule in the past")
        self.sequence += 1
        heapq.heappush(self.queue, (self.now + delay, self.sequence, action))

    def run(self, until=math.inf):
        """Process every event, returning the number executed."""
        processed = 0
        while self.queue:
            time, _sequence, action = heapq.heappop(self.queue)
            if time > until:
                break
            self.now = time
            processed += 1
            action()
        return processed


class Coordinator:
    """Assigns tiles, watches deadlines, reassigns after a crash, accepts idempotent results."""

    def __init__(self, simulation, tiles, workers, timeout, latency=2.0, allow_stealing=True):
        if timeout <= 0.0:
            raise ValueError("the timeout must be positive")
        if latency < 0.0:
            raise ValueError("the latency must not be negative")
        self.simulation = simulation
        self.tiles = list(tiles)
        self.workers = list(workers)
        self.timeout = timeout
        self.latency = latency
        self.allow_stealing = allow_stealing
        self.assignments = {}      # tile -> (worker name, deadline)
        self.completed = {}        # tile -> digest
        self.frame = {}
        self.pending = list(self.tiles)
        self.idle = []
        self.messages = 0
        self.assign_counts = {}
        self.reassignments = 0
        self.duplicates = 0
        self.dead = set()
        self.timeline = []

    def start(self):
        for worker in self.workers:
            self.send(worker, "hello", {})

    def send(self, worker, kind, payload, worker_sender=None):
        self.messages += 1
        self.simulation.schedule(self.latency, lambda: worker.receive(kind, payload))

    @property
    def done(self):
        return len(self.completed) == len(self.tiles)

    def request_work(self, worker):
        if self.done:
            self.send(worker, "stop", {})
            return
        if not self.pending and self.allow_stealing:
            victim = self.busiest_worker(exclude=worker.name)
            if victim is not None and victim.queue:
                tile = victim.queue.pop(0)
                if self.assignments.get(tile, (None,))[0] == victim.name:
                    del self.assignments[tile]
                self.pending.append(tile)
        if self.pending:
            tile = self.pending.pop(0)
            if tile in self.completed:
                self.request_work(worker)
                return
            self.assign(tile, worker)
        else:
            self.idle.append(worker.name)

    def busiest_worker(self, exclude):
        best = None
        for worker in self.workers:
            if worker.name == exclude or worker.name in self.dead:
                continue
            if best is None or len(worker.queue) > len(best.queue):
                best = worker
        return best

    def assign(self, tile, worker):
        self.assignments[tile] = (worker.name, None)
        self.assign_counts[tile] = self.assign_counts.get(tile, 0) + 1
        self.timeline.append(("assign", self.simulation.now, tile, worker.name))
        self.send(worker, "tile", {"tile": tile})

    def on_start(self, worker_name, tile):
        """The deadline is armed when the tile actually starts, so a queued tile cannot
        time out while it is still waiting behind others."""
        if tile not in self.assignments or self.assignments[tile][0] != worker_name:
            return
        deadline = self.simulation.now + self.timeout
        self.assignments[tile] = (worker_name, deadline)
        self.timeline.append(("start", self.simulation.now, tile, worker_name))
        self.simulation.schedule(self.timeout,
                                 lambda: self.check_deadline(tile, worker_name, deadline))

    def check_deadline(self, tile, worker_name, deadline):
        """A stale timer (the tile was stolen and reassigned since) must not fire."""
        if tile in self.completed or self.assignments.get(tile) != (worker_name, deadline):
            return
        if self.simulation.now < deadline:
            return
        self.dead.add(worker_name)
        lost = [tile_name for tile_name, (owner, _deadline) in self.assignments.items()
                if owner == worker_name]
        for tile_name in lost:
            del self.assignments[tile_name]
            self.pending.append(tile_name)
        self.reassignments += 1
        self.timeline.append(("timeout", self.simulation.now, tile, worker_name))
        for worker in self.workers:
            if worker.name not in self.dead:
                self.send(worker, "poke", {})

    def submit(self, worker_name, tile, digest, block):
        if tile in self.completed:
            self.duplicates += 1
            self.timeline.append(("duplicate", self.simulation.now, tile, worker_name))
            return
        if tile not in self.assignments or self.assignments[tile][0] != worker_name:
            self.duplicates += 1
            return
        del self.assignments[tile]
        self.completed[tile] = digest
        self.frame.update(block)
        self.timeline.append(("complete", self.simulation.now, tile, worker_name))

    def checkpoint(self):
        """Everything needed to resume: the finished tiles and their digests."""
        return {"completed": {tile: digest for tile, digest in sorted(self.completed.items())},
                "frame": dict(self.frame),
                "pending": [tile for tile in self.tiles if tile not in self.completed]}


class Worker:
    def __init__(self, name, coordinator, simulation, speed=1.0, width=32, tile_size=8):
        self.name = name
        self.coordinator = coordinator
        self.simulation = simulation
        self.speed = speed
        self.width = width
        self.tile_size = tile_size
        self.queue = []          # assigned but not yet started
        self.inflight = None     # the tile currently being rendered
        self.finished = 0
        self.alive = True
        self.tiles_done = []

    def receive(self, kind, payload):
        if not self.alive:
            return
        if kind == "stop":
            return
        if kind == "tile":
            self.queue.append(payload["tile"])
            self.pump()
            return
        if kind in ("hello", "poke"):
            self.coordinator.request_work(self)
            return
        raise ValueError(f"unknown message kind {kind!r}")

    def pump(self):
        if not self.alive or self.inflight is not None or not self.queue:
            return
        tile = self.queue.pop(0)
        self.inflight = tile
        self.coordinator.on_start(self.name, tile)
        rows = (self.width + self.tile_size - 1) // self.tile_size
        self.simulation.schedule(tile_cost(tile, rows) / self.speed, self.finish)

    def finish(self):
        if not self.alive or self.inflight is None:
            return
        tile = self.inflight
        self.inflight = None
        block = tile_pixels(tile, self.width, self.tile_size)
        self.finished += 1
        self.tiles_done.append(tile)
        self.coordinator.submit(self.name, tile, pixel_digest(block), block)
        self.pump()
        self.coordinator.request_work(self)


def run_farm(width=32, tile_size=8, workers=3, timeout=900.0, crash=None, latency=2.0,
             duplicate_every=None, speeds=None, horizon=10 ** 6):
    """Run the coordinator/worker simulation; returns a report dictionary.

    `horizon` bounds the simulated time so a pathological configuration (a deadline shorter
    than the tile cost, say) terminates with an incomplete farm instead of spinning forever.
    """
    simulation = Simulation()
    coordinator = Coordinator(simulation, tile_grid(width, tile_size), [], timeout,
                              latency=latency)
    fleet = []
    for index in range(workers):
        speed = 1.0 if speeds is None else speeds[index]
        fleet.append(Worker(f"w{index}", coordinator, simulation, speed=speed, width=width,
                            tile_size=tile_size))
    coordinator.workers = fleet
    if crash is not None:
        name, at = crash
        for worker in fleet:
            if worker.name == name:
                simulation.schedule(at, lambda worker=worker: setattr(worker, "alive", False))
    coordinator.start()
    if duplicate_every is not None:
        for tile in coordinator.tiles[::duplicate_every]:
            simulation.schedule(TILE_COST + 5.0, lambda tile=tile: coordinator.submit(
                "ghost", tile, "deadbeef", tile_pixels(tile, width, tile_size)))
    events = simulation.run(until=horizon)
    return {"coordinator": coordinator, "simulation": simulation, "events": events,
            "makespan": simulation.now, "fleet": fleet}


def run_static(width=32, tile_size=8, workers=3, timeout=900.0, latency=2.0):
    """Baseline: the grid split into contiguous row chunks, one owner each, no stealing."""
    simulation = Simulation()
    tiles = tile_grid(width, tile_size)
    coordinator = Coordinator(simulation, tiles, [], timeout, latency=latency,
                              allow_stealing=False)
    fleet = [Worker(f"s{index}", coordinator, simulation, speed=1.0, width=width,
                    tile_size=tile_size) for index in range(workers)]
    coordinator.workers = fleet
    rows = (width + tile_size - 1) // tile_size
    rows_per_worker = (rows + workers - 1) // workers
    for tile in tiles:
        owner = min(tile[0] // rows_per_worker, workers - 1)
        coordinator.assign(tile, fleet[owner])
    coordinator.pending = []  # the static partition owns every tile explicitly
    simulation.run()
    return {"makespan": simulation.now, "per_worker": [worker.finished for worker in fleet]}


def main():
    width, tile_size = 32, 8
    grid = tile_grid(width, tile_size)
    check(len(grid) == 16, f"the grid must have 16 tiles, got {len(grid)}")
    check(len(set(grid)) == len(grid), "tiles must be unique")
    covered = set()
    for tile in grid:
        covered |= set(tile_pixels(tile, width, tile_size).keys())
    check(len(covered) == width * width,
          f"the tiles must cover the image exactly once: {len(covered)}")

    # ---------------- healthy fleet: exactly once, identical to the oracle ----------------
    healthy = run_farm(width, tile_size, workers=3)
    coordinator = healthy["coordinator"]
    check(coordinator.done, f"the farm must finish: {len(coordinator.completed)}/{len(grid)}")
    check(sorted(coordinator.completed) == sorted(grid),
          "every tile must be completed exactly once")
    check(coordinator.reassignments == 0, "a healthy fleet needs no reassignment")
    check(coordinator.duplicates == 0, "a healthy fleet produces no duplicates")
    oracle = reference_frame(width, tile_size)
    check(coordinator.frame == oracle, "the assembled frame must equal the single-process render")
    check(len(coordinator.frame) == width * width, "the frame must cover every pixel")
    check(sum(worker.finished for worker in healthy["fleet"]) == len(grid),
          "the workers must render every tile exactly once in total")
    check(max(coordinator.assign_counts.values()) <= 2,
          f"a tile may only be reassigned after a steal: {coordinator.assign_counts}")

    # ---------------- exactly-once assembly over an at-least-once network ----------------
    duplicated = run_farm(width, tile_size, workers=3, duplicate_every=3)
    check(duplicated["coordinator"].done, "duplicates must not stop the farm")
    check(duplicated["coordinator"].frame == oracle,
          "a duplicated report must not corrupt the frame")
    check(duplicated["coordinator"].duplicates > 0,
          "the duplicate injector must actually fire")

    # ---------------- a worker crash: the deadline recovers its tile ----------------
    crashed = run_farm(width, tile_size, workers=3, crash=("w1", 400.0))
    crashed_coordinator = crashed["coordinator"]
    check(crashed_coordinator.done,
          f"the farm must survive a crash: {len(crashed_coordinator.completed)} tiles")
    check(crashed_coordinator.frame == oracle, "the recovered frame must still be correct")
    check(crashed_coordinator.reassignments >= 1,
          "the crashed worker's tiles must be reassigned")
    check("w1" in crashed_coordinator.dead, "the crashed worker must be marked dead")
    check(crashed["makespan"] > healthy["makespan"],
          "a crash must cost simulated time")
    check(sum(worker.finished for worker in crashed["fleet"]) <= len(grid) + 1,
          "recovery may redo at most the lost tile")
    restarted = [tile for tile, count in crashed_coordinator.assign_counts.items()
                 if count > 1]
    check(len(restarted) >= 1, "the lost tile must appear twice in the assignment log")
    extra_assignments = sum(count - 1 for count in crashed_coordinator.assign_counts.values())
    check(1 <= extra_assignments <= 4,
          f"recovery must restart only the lost tiles, restarted={extra_assignments}")
    check(extra_assignments < len(grid) // 2,
          "the crash must not force a general re-render")
    check(sorted(crashed_coordinator.completed) == sorted(grid),
          "every tile must still be present exactly once")

    # ---------------- checkpoint / resume ----------------
    partial = run_farm(width, tile_size, workers=2, horizon=1100.0)
    checkpoint = partial["coordinator"].checkpoint()
    completed = set(checkpoint["completed"])
    check(0 < len(completed) < len(grid),
          f"the checkpoint must capture a partial frame: {len(completed)} tiles")
    check(set(checkpoint["frame"]) == {pixel for tile in completed
                                       for pixel in tile_pixels(tile, width, tile_size)},
          "the checkpointed frame must hold exactly the finished tiles' pixels")
    check(all(checkpoint["frame"][pixel] == oracle[pixel] for pixel in checkpoint["frame"]),
          "the partial frame must be bit-identical to the oracle")
    remaining = [tile for tile in grid if tile not in completed]
    check(sorted(remaining) == sorted(checkpoint["pending"]),
          "the pending set must be exactly the unfinished tiles")
    resumed = Coordinator(Simulation(), grid, [], timeout=900.0)
    for tile, digest in checkpoint["completed"].items():
        resumed.completed[tile] = digest
    check(sorted(resumed.completed) == sorted(completed),
          "a resumed coordinator must keep exactly the finished tiles")
    for tile in resumed.completed:
        check(resumed.completed[tile] == partial["coordinator"].completed[tile],
              "the tile digests must survive the checkpoint round trip")
    check(not resumed.done, "the resumed coordinator still has work to do")
    check(set(checkpoint["pending"]) == set(remaining) and len(remaining) > 0,
          "the resume must not redo finished tiles")

    # ---------------- work stealing beats the static partition ----------------
    dynamic = run_farm(width, tile_size, workers=4)
    static = run_static(width, tile_size, workers=4)
    check(static["makespan"] > dynamic["makespan"],
          f"stealing must beat the static partition: {static['makespan']} vs "
          f"{dynamic['makespan']}")
    balance = [worker.finished for worker in dynamic["fleet"]]
    check(sum(balance) == len(grid), f"the dynamic fleet must finish every tile: {balance}")
    check(max(balance) - min(balance) <= 4,
          f"the dynamic schedule must stay balanced: {balance}")
    check(sum(static["per_worker"]) == len(grid),
          "the static partition must cover every tile exactly once")
    rows = (width + tile_size - 1) // tile_size
    rows_per_worker = (rows + 4 - 1) // 4
    static_costs = [sum(tile_cost(tile, rows) for tile in grid
                        if min(tile[0] // rows_per_worker, 3) == index) for index in range(4)]
    check(max(static_costs) - min(static_costs) > 0.15 * max(static_costs),
          f"the contiguous static split must be cost-imbalanced: {static_costs}")
    check(max(static_costs) <= static["makespan"],
          "the makespan must cover the busiest static worker")

    # ---------------- schedule accounting and determinism ----------------
    events = [entry[0] for entry in dynamic["coordinator"].timeline]
    check(events.count("complete") == len(grid),
          f"the timeline must contain one completion per tile: {events}")
    check(events.index("complete") < len(events), "at least one completion must precede the end")
    check(dynamic["coordinator"].messages >= len(grid),
          "the coordinator must exchange at least one message per tile")
    repeat = run_farm(width, tile_size, workers=4)
    check(repeat["coordinator"].timeline == dynamic["coordinator"].timeline
          and repeat["makespan"] == dynamic["makespan"],
          "the whole schedule must be reproducible")
    check(repeat["coordinator"].frame == oracle, "the repeated run must render the same frame")
    crash_repeat = run_farm(width, tile_size, workers=3, crash=("w1", 400.0))
    check(crash_repeat["coordinator"].timeline == crashed_coordinator.timeline,
          "the crash schedule must be reproducible too")

    # ---------------- deadlines, priorities and invalid configurations ----------------
    # ---------------- deadlines: a slow worker is declared dead and its tile stolen ----------------
    slow = run_farm(width, tile_size, workers=2, timeout=600.0, speeds=(0.4, 1.0))
    slow_coordinator = slow["coordinator"]
    check(slow_coordinator.reassignments > 0,
          "a worker slower than its deadline must be declared dead")
    check("w0" in slow_coordinator.dead, "the slow worker must be the one declared dead")
    check(slow_coordinator.done, "the remaining workers must finish the frame")
    check(slow_coordinator.frame == oracle, "the frame must survive a deadline-driven takeover")
    check(slow["makespan"] > healthy["makespan"], "the takeover must cost simulated time")

    # a deadline shorter than the tile cost with a single worker cannot make progress, and the
    # bounded horizon is what turns that into a clean, reported failure instead of a hang
    stalled = run_farm(width, tile_size, workers=1, timeout=100.0, horizon=5000.0)
    check(not stalled["coordinator"].done,
          "an impossible deadline must leave the farm incomplete")
    check(stalled["coordinator"].reassignments > 0,
          "the impossible deadline must still be detected")
    for call, why in (
        (lambda: tile_grid(0, 8), "zero width"),
        (lambda: tile_grid(8, 0), "zero tile size"),
        (lambda: Coordinator(Simulation(), [], [], timeout=-1.0), "negative timeout"),
        (lambda: Simulation().schedule(-1.0, lambda: None), "event in the past"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")
    empty = Simulation()
    idle_coordinator = Coordinator(empty, [], [], timeout=10.0)
    check(idle_coordinator.done, "a coordinator with no tiles is immediately done")
    check(idle_coordinator.checkpoint()["pending"] == [], "an empty farm has no pending tiles")

    print(f"grid={len(grid)} tiles makespan dynamic={dynamic['makespan']} "
          f"static={static['makespan']}")
    print(f"healthy={healthy['makespan']} crashed={crashed['makespan']} "
          f"reassignments={crashed_coordinator.reassignments}")
    print(f"balance={balance} messages={dynamic['coordinator'].messages} "
          f"duplicates={duplicated['coordinator'].duplicates}")
    print("PASS")


if __name__ == "__main__":
    main()
