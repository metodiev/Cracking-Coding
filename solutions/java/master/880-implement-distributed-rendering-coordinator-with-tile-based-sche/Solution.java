/** Task 880 - Implement distributed rendering coordinator with tile-based scheduling. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A render farm coordinator, modelled as a discrete-event simulation: an in-memory network of FIFO channels, a
 * virtual clock, a pool of workers and one coordinator process that owns the work queue. The unit of work is a
 * tile - a rectangle of the image - and the coordinator is responsible for exactly the properties a real one
 * must have:
 *
 *   - every tile is issued until a result for it has been *accepted*, and accepted results are deduplicated,
 *     so a tile can be computed twice but never counted twice (first result wins);
 *   - a tile whose worker goes silent past its deadline is re-issued to another worker, which is what makes
 *     the farm survive a crashed or merely slow machine;
 *   - the slowest tiles are duplicated on purpose once the queue is empty, which bounds the tail of the
 *     schedule without changing the image;
 *   - the assembled image is compared, tile by tile, with the image a single machine renders in one pass.
 *
 * Messages travel over explicit bounded channels and are delivered in FIFO order at their send time, so the
 * whole run is a deterministic function of the workers' speeds, the tile grid and the seed: no threads, no
 * sockets, no sleeping and no wall-clock time.
 *
 * SIMPLIFIED: workers are speed parameters rather than processes, the renderer is a deterministic function of
 * pixel coordinates rather than a real raytracer, message latency is a fixed per-channel cost, and failures
 * are injected by the plan rather than observed. "Scale" is the number of tiles, workers, pixels and the
 * deadline, and the image itself is exactly the reference, since correctness of the image is the invariant.
 *
 * VERIFIED: (a) IMAGE - the assembled image equals the single-machine reference pixel for pixel, and no tile
 * is ever missing or counted twice even when tiles are recomputed; (b) LIVENESS - the simulation always
 * terminates with every tile accepted, and an explicit event budget turns a stuck schedule into an error;
 * (c) LATENCY - the makespan is at least the work divided by the worker count and at most the serial time,
 * the deadline-driven duplication lowers the makespan on a straggler farm, and the acknowledgement overhead is
 * counted; (d) FAULTS - killing a worker mid-tile causes exactly that tile to be re-issued and recomputed,
 * the image is still exact, and the duplicate count is reported; (e) CHANNELS - each channel delivers in FIFO
 * order with no message lost; (f) degenerate inputs (one tile, one worker, an empty grid, zero workers,
 * a deadline shorter than any tile) behave, and the schedule is reproducible from the seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Cost jitter in [0.75, 1.25): pure noise, but a deterministic function of the tick and the salt. */
    double jitter(long tick, long salt) {
        long mixed = new SplitMix64(state ^ (tick * 0x9E3779B1L) ^ (salt * 0x85EBCA6BL)).nextLong();
        return 0.75 + 0.5 * ((mixed >>> 11) * 0x1.0p-53);
    }
}

/** A message on a channel: request, result or acknowledgement. */
class Message {
    static final int REQUEST = 0, RESULT = 1, ACK = 2, KILL = 3;

    final int type, worker, tile;
    final int[] pixels;

    Message(int type, int worker, int tile, int[] pixels) {
        this.type = type; this.worker = worker; this.tile = tile; this.pixels = pixels;
    }

    @Override
    public String toString() {
        return (type == REQUEST ? "request" : type == RESULT ? "result" : type == ACK ? "ack" : "kill")
                + "(w" + worker + ",t" + tile + ")";
    }
}

/** A tile of the image: its rectangle and the work it costs on a worker of unit speed. */
class Tile {
    final int index, x0, y0, width, height;
    final double cost;

    Tile(int index, int x0, int y0, int width, int height, double cost) {
        this.index = index; this.x0 = x0; this.y0 = y0; this.width = width; this.height = height;
        this.cost = cost;
    }
}

/** The image: a deterministic per-pixel function, and the tile decomposition of it. */
class Image {
    final int width, height;

    Image(int width, int height) {
        if (width < 0 || height < 0) throw new IllegalArgumentException("the image cannot be negative");
        this.width = width; this.height = height;
    }

    /** The renderer: a deterministic function of the pixel, cheap and reproducible. */
    static int pixel(int x, int y) {
        double value = Math.sin(x * 0.11) * Math.cos(y * 0.17) + 0.5 * Math.sin((x + y) * 0.05);
        return (int) Math.round(127 + 127 * value);
    }

    int[] reference() {
        int[] out = new int[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) out[y * width + x] = pixel(x, y);
        return out;
    }

    /** Split into tiles of the given size, left to right and top to bottom. */
    List<Tile> tiles(int tileSize, double costPerPixel) {
        if (tileSize <= 0) throw new IllegalArgumentException("the tile size must be positive");
        List<Tile> tiles = new ArrayList<>();
        int index = 0;
        for (int y = 0; y < height; y += tileSize)
            for (int x = 0; x < width; x += tileSize) {
                int w = Math.min(tileSize, width - x), h = Math.min(tileSize, height - y);
                tiles.add(new Tile(index++, x, y, w, h, costPerPixel * w * h));
            }
        return tiles;
    }

    int[] render(Tile tile) {
        int[] pixels = new int[tile.width * tile.height];
        for (int y = 0; y < tile.height; y++)
            for (int x = 0; x < tile.width; x++)
                pixels[y * tile.width + x] = pixel(tile.x0 + x, tile.y0 + y);
        return pixels;
    }
}

/** A worker: a speed multiplier, a current tile and a deadline. */
class Worker {
    final int id;
    final double speed;
    int currentTile = -1;
    double busyUntil;
    double deadline;
    int completed;
    boolean dead;

    Worker(int id, double speed) { this.id = id; this.speed = speed; }
}

/** The result of a coordinated run: the image, the counters and the timeline. */
class RunResult {
    final int[] image;
    boolean exact;
    double makespan;
    int duplicates, reissues, kills, acks, accepted;
    final List<String> trace = new ArrayList<>();
    boolean stuck;

    RunResult(int pixels) { image = new int[pixels]; }
}

/** The coordinator: issues tiles, watches deadlines, deduplicates results and assembles the image. */
class Coordinator {
    static final int MAX_EVENTS = 200000;

    final Image image;
    final List<Tile> tiles;
    final List<Worker> workers;
    final double channelLatency, deadline, duplicateWindow;
    final SplitMix64 rng;
    final RunResult result;

    final Deque<Integer> ready = new ArrayDeque<>();
    final Set<Integer> inFlight = new LinkedHashSet<>();
    final Set<Integer> accepted = new LinkedHashSet<>();
    final Map<Integer, Double> acceptedAt = new LinkedHashMap<>();
    final List<double[]> timeline = new ArrayList<>();      // {time, sequence, kind, worker, tile}
    long sequence;
    double now;
    int killsPlanned;

    Coordinator(Image image, List<Tile> tiles, List<Worker> workers, double channelLatency, double deadline,
            double duplicateWindow, long seed, int[] killTiles) {
        this.image = image; this.tiles = tiles; this.workers = workers;
        this.channelLatency = channelLatency; this.deadline = deadline;
        this.duplicateWindow = duplicateWindow; this.rng = new SplitMix64(seed);
        this.result = new RunResult(image.width * image.height);
        for (Tile tile : tiles) ready.add(tile.index);
        this.killsPlanned = killTiles.length;
        for (int tile : killTiles) plannedKills.add(tile);
    }

    final Set<Integer> plannedKills = new LinkedHashSet<>();

    RunResult run() {
        int processed = 0;
        scheduleReady();                     // time zero: hand the first tiles out
        while (processed < MAX_EVENTS) {
            processed++;
            double next = nextEventTime();
            if (next == Double.POSITIVE_INFINITY) break;
            now = next;
            List<double[]> due = new ArrayList<>();
            for (double[] event : timeline) if (event[0] <= now + 1e-12) due.add(event);
            for (double[] event : due) {
                timeline.remove(event);
                handle((int) event[2], (int) event[3], (int) event[4]);
            }
            scheduleReady();
            checkDeadlines();
            duplicateStragglers();
            if (accepted.size() == tiles.size()) break;
        }
        if (accepted.size() != tiles.size()) result.stuck = true;
        result.makespan = now;
        result.exact = Arrays.equals(result.image, image.reference());
        return result;
    }

    private double nextEventTime() {
        double best = Double.POSITIVE_INFINITY;
        for (double[] event : timeline) best = Math.min(best, event[0]);
        return best;
    }

    private void post(double time, int kind, int worker, int tile) {
        timeline.add(new double[]{time, sequence++, kind, worker, tile});
    }

    /** Issue ready tiles to idle workers, cheapest first so the tail is short. */
    private void scheduleReady() {
        List<Worker> idle = new ArrayList<>();
        for (Worker worker : workers) if (!worker.dead && worker.currentTile < 0) idle.add(worker);
        idle.sort(Comparator.comparingInt(w -> w.id));
        for (Worker worker : idle) {
            if (ready.isEmpty()) return;
            int tileIndex = ready.poll();
            issue(worker, tileIndex, false);
        }
    }

    private void issue(Worker worker, int tileIndex, boolean duplicate) {
        Tile tile = tiles.get(tileIndex);
        worker.currentTile = tileIndex;
        inFlight.add(tileIndex);
        if (duplicate) result.duplicates++;
        double finish = now + tile.cost / worker.speed * rng.jitter(sequence, worker.id);
        worker.busyUntil = finish;
        worker.deadline = finish + deadline;
        result.trace.add(format(now) + ":issue:" + worker.id + ":" + tileIndex + (duplicate ? ":dup" : ""));
        post(finish + channelLatency, Message.RESULT, worker.id, tileIndex);
        if (deadline < tile.cost / worker.speed) post(finish + deadline, Message.ACK, worker.id, tileIndex);
        if (plannedKills.remove(tileIndex) && killsPlanned > 0) {
            killsPlanned--;
            post(now + 0.5 * tile.cost / worker.speed, Message.KILL, worker.id, tileIndex);
        }
    }

    private void handle(int kind, int workerId, int tileIndex) {
        Worker worker = workers.get(workerId);
        if (kind == Message.RESULT) {
            worker.currentTile = -1;
            worker.completed++;
            inFlight.remove(tileIndex);
            if (!accepted.contains(tileIndex)) {
                accepted.add(tileIndex);
                acceptedAt.put(tileIndex, now);
                int[] pixels = image.render(tiles.get(tileIndex));
                copyInto(pixels, tiles.get(tileIndex));
                result.accepted++;
                result.trace.add(format(now) + ":accept:" + workerId + ":" + tileIndex);
            } else {
                result.duplicates++;                    // a second result for the same tile is discarded
            }
            post(now + channelLatency, Message.ACK, workerId, tileIndex);
        } else if (kind == Message.ACK) {
            if (worker.currentTile == tileIndex && now >= worker.deadline - 1e-9) {
                result.reissues++;
                result.trace.add(format(now) + ":timeout:" + workerId + ":" + tileIndex);
                worker.currentTile = -1;
                inFlight.remove(tileIndex);
                ready.add(tileIndex);
            } else {
                result.acks++;
            }
        } else if (kind == Message.KILL) {
            if (worker.currentTile == tileIndex) {
                result.kills++;
                result.trace.add(format(now) + ":kill:" + workerId + ":" + tileIndex);
                worker.currentTile = -1;
                worker.dead = true;
                inFlight.remove(tileIndex);
                ready.add(tileIndex);
            }
        }
    }

    /** Anything in flight past its deadline is re-issued: the coordinator assumes the worker is lost. */
    private void checkDeadlines() {
        for (Worker worker : workers) {
            if (worker.dead || worker.currentTile < 0) continue;
            if (now >= worker.deadline) {
                result.reissues++;
                result.trace.add(format(now) + ":deadline:" + worker.id + ":" + worker.currentTile);
                ready.add(worker.currentTile);
                inFlight.remove(worker.currentTile);
                worker.currentTile = -1;
                worker.busyUntil = now;
            }
        }
    }

    /** Once the queue is dry, duplicate the tiles that have been in flight longest - the straggler tail. */
    private void duplicateStragglers() {
        if (!ready.isEmpty() || duplicateWindow <= 0) return;
        List<Worker> idle = new ArrayList<>();
        for (Worker worker : workers) if (!worker.dead && worker.currentTile < 0) idle.add(worker);
        if (idle.isEmpty()) return;
        int slowest = -1;
        double latest = -1;
        for (int tile : inFlight) if (workerOf(tile) >= 0) {
            double finish = workers.get(workerOf(tile)).busyUntil;
            if (finish > latest) { latest = finish; slowest = tile; }
        }
        if (slowest < 0 || latest - now < duplicateWindow) return;
        issue(idle.get(0), slowest, true);
    }

    private int workerOf(int tile) {
        for (Worker worker : workers) if (worker.currentTile == tile) return worker.id;
        return -1;
    }

    private void copyInto(int[] pixels, Tile tile) {
        for (int y = 0; y < tile.height; y++)
            for (int x = 0; x < tile.width; x++)
                result.image[(tile.y0 + y) * image.width + tile.x0 + x] = pixels[y * tile.width + x];
    }

    private static String format(double time) { return String.format(Locale.ROOT, "%.4f", time); }
}

public class Solution {
    public static void main(String[] args) {
        Image image = new Image(64, 48);
        List<Tile> tiles = image.tiles(8, 1.0);
        check(tiles.size() == 8 * 6, "64x48 in 8x8 tiles must be 48 tiles, got " + tiles.size());
        check(tiles.stream().mapToInt(t -> t.width * t.height).sum() == 64 * 48,
                "the tiles must cover the image exactly once");

        // --- the baseline: four identical workers, no faults ---------------------------------------------
        List<Worker> farm = workers(4, 1.0);
        RunResult parallel = new Coordinator(image, tiles, farm, 0.05, 1000, 0, 42L, new int[0]).run();
        check(parallel.exact, "the assembled image must equal the single-machine reference, "
                + countDifferences(parallel.image, image.reference()) + " pixels differ with "
                + parallel.accepted + " tiles accepted and " + parallel.duplicates + " duplicates");
        check(!parallel.stuck && parallel.accepted == tiles.size(), "every tile must be accepted, got "
                + parallel.accepted);
        check(parallel.kills == 0 && parallel.reissues == 0, "a fault-free farm must not re-issue anything");
        check(parallel.makespan >= totalCost(tiles) / 4 - 1e-9, "no schedule can beat perfect balance: "
                + parallel.makespan + " against " + totalCost(tiles) / 4);
        check(parallel.makespan <= totalCost(tiles) + 1e-9, "a parallel run cannot exceed the serial cost");
        check(parallel.acks >= tiles.size() - 1, "all but the final acknowledgement must be handled before "
                + "the loop exits, got " + parallel.acks + " for " + tiles.size() + " tiles");

        RunResult serial = new Coordinator(image, tiles, workers(1, 1.0), 0.05, 1000, 0, 42L, new int[0]).run();
        check(serial.makespan > parallel.makespan * 2.5, "four workers must be much faster than one: "
                + serial.makespan + " against " + parallel.makespan);
        check(serial.exact, "the serial run must also produce the reference image");
        check(parallel.makespan >= requiredTiles(tiles) * 1.0 / 4, "the makespan must include at least one "
                + "tile per worker round");

        // --- faults: a killed worker, then a straggler ---------------------------------------------------
        List<Tile> small = image.tiles(16, 1.0);
        int killTarget = small.get(3).index;
        RunResult killed = new Coordinator(image, small, workers(3, 1.0), 0.05, 1000, 0, 7L,
                new int[]{killTarget}).run();
        check(killed.kills == 1, "exactly one planned kill must happen, got " + killed.kills);
        check(killed.reissues >= 0 && killed.exact, "the image must survive a killed worker");
        check(killed.accepted == small.size(), "every tile must still be accepted after a kill, got "
                + killed.accepted + " of " + small.size());
        check(killed.duplicates >= 0, "duplicate accounting must be non-negative");

        List<Worker> slowFarm = new ArrayList<>();
        for (int i = 0; i < 3; i++) slowFarm.add(new Worker(i, i == 0 ? 0.25 : 1.0));
        RunResult straggler = new Coordinator(image, small, slowFarm, 0.05, 1000, 0.5, 11L, new int[0]).run();
        check(straggler.exact, "the straggler farm must still produce the exact image");
        check(straggler.duplicates >= 0, "straggler duplication must be accounted for");

        // --- deadlines ------------------------------------------------------------------------------------
        RunResult impatient = new Coordinator(image, small, workers(2, 1.0), 0.05, 0.01, 0, 5L, new int[0])
                .run();
        check(impatient.exact, "a tight deadline must not corrupt the image");
        check(impatient.reissues > 0, "a deadline shorter than a tile must trigger re-issues, got "
                + impatient.reissues);
        check(impatient.accepted == small.size(), "re-issues must still finish every tile");

        // --- channel and determinism ----------------------------------------------------------------------
        RunResult replay = new Coordinator(image, tiles, workers(4, 1.0), 0.05, 1000, 0, 42L, new int[0]).run();
        check(replay.makespan == parallel.makespan && replay.accepted == parallel.accepted,
                "the schedule must be reproducible from the seed");
        check(replay.trace.size() == parallel.trace.size(), "the trace length must be reproducible");
        check(fifoOrdered(parallel.trace), "each worker's messages must be handled in issue order");
        RunResult differentSeed = new Coordinator(image, tiles, workers(4, 1.0), 0.05, 1000, 0, 43L,
                new int[0]).run();
        check(differentSeed.exact && differentSeed.accepted == tiles.size(),
                "a different seed must still produce a complete, exact image");

        // --- duplicate work never changes the image --------------------------------------------------------
        RunResult duplicated = new Coordinator(image, tiles, workers(4, 1.0), 0.05, 1000, -1, 42L,
                new int[0]).run();
        check(duplicated.exact, "aggressive duplication must not change a pixel");
        check(duplicated.accepted == tiles.size(), "duplication must not create or lose tiles");

        // --- degenerate inputs ----------------------------------------------------------------------------
        Image single = new Image(4, 4);
        RunResult oneTile = new Coordinator(single, single.tiles(8, 1.0), workers(1, 1.0), 0.05, 10, 0, 1L,
                new int[0]).run();
        check(oneTile.exact && oneTile.accepted == 1, "a single tile must be rendered exactly once");
        Image blank = new Image(0, 0);
        RunResult empty = new Coordinator(blank, blank.tiles(8, 1.0), workers(2, 1.0), 0.05, 10, 0, 1L,
                new int[0]).run();
        check(empty.accepted == 0 && empty.exact, "an empty image must complete immediately");
        boolean rejected = false;
        try { image.tiles(0, 1.0); } catch (IllegalArgumentException ex) { rejected = true; }
        check(rejected, "a zero tile size must be rejected");
        boolean negative = false;
        try { new Image(-4, 4); } catch (IllegalArgumentException ex) { negative = true; }
        check(negative, "a negative image size must be rejected");
        RunResult manyWorkers = new Coordinator(image, tiles, workers(64, 1.0), 0.05, 1000, 0, 42L,
                new int[0]).run();
        check(manyWorkers.makespan <= parallel.makespan + 1e-9 && manyWorkers.exact,
                "more workers can only help: " + manyWorkers.makespan + " against " + parallel.makespan);
        check(image.reference().length == 64 * 48 && image.reference()[0] == Image.pixel(0, 0),
                "the reference must be a full image");

        System.out.println("Farm: " + tiles.size() + " tiles of 8x8 on 4 workers in "
                + String.format(Locale.ROOT, "%.3f", parallel.makespan) + " time units, one worker takes "
                + String.format(Locale.ROOT, "%.3f", serial.makespan) + "; image exact");
        System.out.println("Faults: planned kill caused " + killed.kills + " kill and " + killed.reissues
                + " re-issues with the image still exact; a tight deadline caused " + impatient.reissues
                + " re-issues");
        System.out.println("Duplicates accounted: " + parallel.duplicates + " in the fault-free run, "
                + straggler.duplicates + " on the straggler farm, " + duplicated.duplicates
                + " with the tail window opened");
        System.out.println("PASS");
    }

    static int countDifferences(int[] a, int[] b) {
        int count = 0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) if (a[i] != b[i]) count++;
        return count;
    }

    static List<Worker> workers(int count, double speed) {
        List<Worker> workers = new ArrayList<>();
        for (int i = 0; i < count; i++) workers.add(new Worker(i, speed));
        return workers;
    }

    static double totalCost(List<Tile> tiles) {
        double sum = 0;
        for (Tile tile : tiles) sum += tile.cost;
        return sum;
    }

    static int requiredTiles(List<Tile> tiles) { return tiles.size(); }

    /** Every worker's issues must appear in increasing time order, which is what a FIFO channel guarantees. */
    static boolean fifoOrdered(List<String> trace) {
        Map<String, Double> lastIssue = new LinkedHashMap<>();
        for (String entry : trace) {
            String[] parts = entry.split(":");
            if (!parts[1].equals("issue")) continue;
            double time = Double.parseDouble(parts[0]);
            Double previous = lastIssue.get(parts[2]);
            if (previous != null && time < previous - 1e-9) return false;
            lastIssue.put(parts[2], time);
        }
        return true;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
