/** Task 600 - Implement advanced checkpoint-restore for long-running computations (save state). (Tier: Advanced, CTCI Beyond — Software Engineering & Tooling)
 *
 * Checkpoint/restore for an iterative computation, built so that resuming is *exactly* equivalent to
 * running straight through. The computation is deliberately simple but stateful in every way that matters:
 * a damped Jacobi smoothing of a 1-D field (a double[] plus the running maximum change), a running
 * accumulator and its count, the step index, and the pseudo-random forcing term's generator state. The
 * checkpoint payload is a canonical text encoding - version, step, count, accumulator, the generator state,
 * and every field value written as Double.toHexString so that no decimal round trip can lose a bit - followed
 * by a 64-bit FNV-1a checksum of the payload. Restore validates version and checksum and refuses anything
 * else, and the resumed run continues from exactly the saved bits, which is asserted by comparing the final
 * canonical encodings byte for byte rather than approximately.
 *
 * The generator is the repo's 16-bit LCG on purpose, and the reason is documented where it belongs: this task
 * needs full state capture, not statistical quality, and the LCG's entire state is one integer - which is
 * exactly why the wave-3 notes that its low bits and short cycle make it unsuitable for Monte Carlo work do
 * not apply here. The file verifies the restored generator state against a closed form: the n-th LCG value is
 * computed independently with BigInteger, using a^n and the geometric series for the affine map, and compared
 * with the restored state.
 *
 * Documented checks: a 1000-step run equals a run that checkpoints every 37 steps and restores each time
 * (identical final encodings, and identical per-step digests along the whole trajectory); restoring from a
 * checkpoint taken at every 25th step of a 600-step run (25 restarts) always reproduces the same end state; the trajectory is invariant to the chunking of
 * the work; a corrupted payload (one flipped character), a truncated payload, a wrong version field and a
 * wrong checksum each raise, and the message names the failure; encoding is a bijection on the state (decode
 * then encode reproduces the payload, for several states including 0.0, -0.0 and 1e-300, and the negative zero
 * is asserted to be distinguishable from positive zero because the hex form is used); the FNV-1a checksum is
 * sensitive (1000 distinct payloads give at least 999 distinct digests) and the checkpoint of a run is
 * independent of wall-clock time, which is asserted by construction - the payload has no time field and two
 * runs of the same length give identical bytes; the payload is compact (a reported size) and the number of
 * restored steps is counted; and a simulated crash 250 steps in (discarding the live state) resumes from the
 * checkpoint at step 200 with exactly the 50 missing steps recomputed and a state byte-identical to the one the
 * crashed run had reached. The generator state is verified against a closed form evaluated with BigInteger
 * modular exponentiation at the correct draw count (field size times steps: 40000 draws for 1000 steps over a
 * 40-cell field), which is itself part of what the capture has to preserve.
 */
import java.math.BigInteger;
import java.util.*;

public class Solution {
    static final int VERSION = 3;

    // ---------------------------------------------------------------- the computation

    static final class State {
        int step;
        int count;
        long accumulator;
        long generatorState;
        double[] field;
        double maxChange;
    }

    static State initial(int size, long seed) {
        State state = new State();
        state.step = 0;
        state.count = 0;
        state.accumulator = 0;
        state.generatorState = seed;
        state.field = new double[size];
        state.maxChange = 0.0;
        for (int i = 0; i < size; i++) state.field[i] = Math.sin(i * 0.37) * 0.5;
        return state;
    }

    /** The repo's 16-bit LCG, used here purely as a fully capturable generator state. */
    static long lcg(long state) {
        return (state * 75 + 74) % 65537;
    }

    /** One damped Jacobi smoothing step with a pseudo-random forcing term. */
    static void advance(State state, double damping) {
        int size = state.field.length;
        double[] next = new double[size];
        double localMax = 0.0;
        for (int i = 0; i < size; i++) {
            double left = state.field[i == 0 ? 0 : i - 1];
            double right = state.field[i == size - 1 ? size - 1 : i + 1];
            double average = 0.5 * (left + right);
            state.generatorState = lcg(state.generatorState);
            double forcing = (state.generatorState % 101 - 50) / 5000.0;
            next[i] = state.field[i] + damping * ((average - state.field[i]) + forcing);
            localMax = Math.max(localMax, Math.abs(next[i] - state.field[i]));
        }
        state.field = next;
        state.maxChange = localMax;
        state.step++;
        state.count++;
        long sum = 0;
        for (double value : state.field) sum += Math.round(value * 1e6);
        state.accumulator = sum;
    }

    // ---------------------------------------------------------------- serialization

    static String encode(State state) {
        StringBuilder builder = new StringBuilder();
        builder.append("v").append(VERSION);
        builder.append(";step=").append(state.step);
        builder.append(";count=").append(state.count);
        builder.append(";acc=").append(state.accumulator);
        builder.append(";rng=").append(state.generatorState);
        builder.append(";max=").append(Double.toHexString(state.maxChange));
        builder.append(";n=").append(state.field.length);
        builder.append(";f=");
        for (int i = 0; i < state.field.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(Double.toHexString(state.field[i]));
        }
        String payload = builder.toString();
        return payload + ";sum=" + Long.toHexString(fnv1a(payload));
    }

    static State decode(String checkpoint) {
        int marker = checkpoint.lastIndexOf(";sum=");
        if (marker < 0) throw new IllegalStateException("checkpoint has no checksum field");
        String payload = checkpoint.substring(0, marker);
        String checksum = checkpoint.substring(marker + 5);
        String expected = Long.toHexString(fnv1a(payload));
        if (!expected.equals(checksum)) {
            throw new IllegalStateException("checksum mismatch: expected " + expected + " but the payload carries "
                    + checksum);
        }
        Map<String, String> fields = new LinkedHashMap<>();
        String[] parts = payload.split(";");
        for (int i = 1; i < parts.length; i++) {
            int equals = parts[i].indexOf('=');
            if (equals < 0) continue;
            fields.put(parts[i].substring(0, equals), parts[i].substring(equals + 1));
        }
        int version = Integer.parseInt(parts[0].substring(1));
        if (version != VERSION) {
            throw new IllegalStateException("unsupported checkpoint version " + version + " (this build reads "
                    + VERSION + ")");
        }
        State state = new State();
        state.step = Integer.parseInt(fields.get("step"));
        state.count = Integer.parseInt(fields.get("count"));
        state.accumulator = Long.parseLong(fields.get("acc"));
        state.generatorState = Long.parseLong(fields.get("rng"));
        state.maxChange = Double.parseDouble(fields.get("max"));
        int size = Integer.parseInt(fields.get("n"));
        String[] values = fields.get("f").split(",");
        if (values.length != size) {
            throw new IllegalStateException("field length mismatch: header says " + size + " but " + values.length
                    + " values were written");
        }
        state.field = new double[size];
        for (int i = 0; i < size; i++) state.field[i] = Double.parseDouble(values[i]);
        return state;
    }

    static long fnv1a(String payload) {
        long hash = 0xCBF29CE484222325L;
        for (int i = 0; i < payload.length(); i++) {
            hash ^= payload.charAt(i);
            hash *= 0x100000001B3L;
        }
        return hash;
    }

    // ---------------------------------------------------------------- checkpointed runs

    static final class RunResult {
        State state;
        String[] checkpointsTaken;
        int restores;
        double[] stepDigests;
    }

    /** Runs `steps` iteration steps, saving a checkpoint every `every` steps and restoring from the last one. */
    static RunResult run(int steps, int every, int size, long seed, double damping) {
        RunResult result = new RunResult();
        State state = initial(size, seed);
        List<String> checkpoints = new ArrayList<>();
        result.stepDigests = new double[steps + 1];
        result.stepDigests[0] = digest(state);
        String lastCheckpoint = null;
        int sinceCheckpoint = 0;
        for (int k = 0; k < steps; k++) {
            if (every > 0 && sinceCheckpoint == every) {
                lastCheckpoint = encode(state);
                checkpoints.add(lastCheckpoint);
                result.restores++;
                sinceCheckpoint = 0;
                // Simulate a crash: throw the live state away and restore from the checkpoint just written.
                state = decode(lastCheckpoint);
            }
            advance(state, damping);
            sinceCheckpoint++;
            result.stepDigests[k + 1] = digest(state);
        }
        result.state = state;
        result.checkpointsTaken = checkpoints.toArray(new String[0]);
        return result;
    }

    static double digest(State state) {
        double acc = state.step * 1.000001 + state.count * 3.7 + state.generatorState * 0.0001;
        for (double value : state.field) acc += value;
        return acc + state.accumulator * 1e-6;
    }

    // ---------------------------------------------------------------- closed-form LCG state

    static long lcgClosedForm(long seed, long steps) {
        BigInteger modulus = BigInteger.valueOf(65537L);
        BigInteger a = BigInteger.valueOf(75L);
        BigInteger c = BigInteger.valueOf(74L);
        BigInteger power = a.modPow(BigInteger.valueOf(steps), modulus);
        BigInteger seedValue = BigInteger.valueOf(seed);
        // a^steps * seed + c * (a^steps - 1) / (a - 1), all modulo 65537 (a - 1 = 74 is invertible mod 65537).
        BigInteger geometric = power.subtract(BigInteger.ONE)
                .multiply(BigInteger.valueOf(74L).modInverse(modulus)).mod(modulus);
        BigInteger value = power.multiply(seedValue).add(c.multiply(geometric)).mod(modulus);
        return value.longValue();
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        straightThroughEqualsCheckpointed();
        restoreFromEveryCheckpoint();
        chunkingInvariance();
        payloadIntegrity();
        roundTripAndSensitivity();
        crashRecoveryAndCost();
        determinismAndNoClock();

        System.out.println("PASS");
    }

    static void straightThroughEqualsCheckpointed() {
        RunResult straight = run(1000, 0, 40, 2026L, 0.25);
        RunResult checkpointed = run(1000, 37, 40, 2026L, 0.25);
        check(straight.checkpointsTaken.length == 0, "the straight-through run takes no checkpoints");
        check(checkpointed.checkpointsTaken.length == 27, "the checkpointed run takes 27 checkpoints of 37 steps "
                + "inside the 1000 (got " + checkpointed.checkpointsTaken.length + ")");
        String expected = encode(straight.state);
        String actual = encode(checkpointed.state);
        check(expected.equals(actual), "the 1000-step run and the checkpoint-every-37-steps run end with "
                + "byte-identical canonical encodings");
        check(Arrays.equals(straight.stepDigests, checkpointed.stepDigests),
                "and their per-step digests match along the whole trajectory, not just at the end");
        // The forcing term is drawn once per cell per step, so 1000 steps over a 40-cell field consume 40000
        // LCG draws; the closed form must be evaluated at that count, and getting the count right is itself part
        // of what the state capture has to preserve.
        long draws = 1000L * 40L;
        check(checkpointed.state.step == 1000 && checkpointed.state.generatorState
                == lcgClosedForm(2026L, draws), "the restored generator state after " + draws + " draws matches the "
                + "closed form from BigInteger modular exponentiation: " + checkpointed.state.generatorState
                + " versus " + lcgClosedForm(2026L, draws));
        check(straight.state.accumulator == checkpointed.state.accumulator, "the accumulators agree exactly");
        check(straight.state.maxChange == checkpointed.state.maxChange, "and so do the running maxima");
    }

    static void restoreFromEveryCheckpoint() {
        RunResult reference = run(600, 0, 24, 777L, 0.3);
        String expected = encode(reference.state);
        for (int k = 0; k <= 600; k += 25) {
            State state = initial(24, 777L);
            for (int i = 0; i < k; i++) advance(state, 0.3);
            String checkpoint = encode(state);
            State restored = decode(checkpoint);
            check(encode(restored).equals(checkpoint), "the checkpoint taken at step " + k + " round-trips exactly");
            while (restored.step < 600) advance(restored, 0.3);
            check(encode(restored).equals(expected), "resuming from the checkpoint at step " + k + " reproduces the "
                    + "600-step end state byte for byte");
        }
    }

    static void chunkingInvariance() {
        double[] digests = new double[4];
        int[] chunks = {1, 5, 13, 100};
        for (int c = 0; c < chunks.length; c++) {
            State state = initial(16, 4242L);
            int remaining = 300;
            while (remaining > 0) {
                int block = Math.min(chunks[c], remaining);
                for (int i = 0; i < block; i++) advance(state, 0.2);
                remaining -= block;
            }
            digests[c] = digest(state);
        }
        check(digests[0] == digests[1] && digests[1] == digests[2] && digests[2] == digests[3],
                "the trajectory does not depend on how the 300 steps are chunked (" + Arrays.toString(digests) + ")");
        State direct = initial(16, 4242L);
        for (int i = 0; i < 300; i++) advance(direct, 0.2);
        check(Math.abs(digest(direct) - digests[0]) <= 1e-15, "and the chunked digest equals the direct one");
    }

    static void payloadIntegrity() {
        State state = initial(8, 99L);
        for (int i = 0; i < 5; i++) advance(state, 0.25);
        String good = encode(state);
        char[] characters = good.toCharArray();
        int position = good.length() / 2;
        characters[position] = characters[position] == 'a' ? 'b' : 'a';
        String corrupted = new String(characters);
        checkThrows(() -> decode(corrupted), "checksum", "a single flipped character is detected");
        checkThrows(() -> decode(good.substring(0, good.length() - 4)), "checksum",
                "a truncated payload is detected");
        String mutated = good.replace(";step=" + state.step + ";", ";step=" + (state.step + 1) + ";");
        check(!mutated.equals(good), "the field mutation really changed the payload");
        checkThrows(() -> decode(mutated), "checksum",
                "a mutated field value is detected unless the checksum is recomputed (it is not, here)");

        String payloadOnly = good.substring(0, good.lastIndexOf(';'));
        String bumpedPayload = payloadOnly.replaceFirst("^v" + VERSION + ";", "v9;");
        check(bumpedPayload.startsWith("v9;"), "the version bump produced a well-formed payload");
        String bumped = bumpedPayload + ";sum=" + Long.toHexString(fnv1a(bumpedPayload));
        checkThrows(() -> decode(bumped), "version",
                "with a repaired checksum the version field itself is rejected, which shows the two validations "
                        + "are independent");
    }

    interface Action {
        void apply();
    }

    static void checkThrows(Action action, String expectedSubstring, String label) {
        try {
            action.apply();
            throw new RuntimeException("FAILED: " + label + " (no exception was thrown)");
        } catch (IllegalStateException e) {
            String message = e.getMessage() == null ? "" : e.getMessage();
            if (!message.contains(expectedSubstring)) {
                throw new RuntimeException("FAILED: " + label + " (message was: " + message + ")");
            }
        }
    }

    static void roundTripAndSensitivity() {
        State state = initial(5, 11L);
        for (int i = 0; i < 3; i++) advance(state, 0.5);
        String payload = encode(state);
        State parsed = decode(payload);
        check(encode(parsed).equals(payload), "decode then encode reproduces the payload exactly");
        check(Arrays.equals(parsed.field, state.field), "the field values survive the hex round trip bit for bit");

        State zeros = initial(3, 1L);
        String zeroPayload = encode(zeros);
        check(zeroPayload.contains("0x0.0p0"), "a positive zero is written as 0x0.0p0");
        zeros.field[0] = -0.0;
        String negativeZero = encode(zeros);
        check(negativeZero.contains("-0x0.0p0"), "a negative zero is written as -0x0.0p0 and is therefore "
                + "distinguishable from positive zero, which matters because -0.0 == 0.0 in a comparison but they "
                + "are different bits");
        check(!negativeZero.equals(zeroPayload), "so the two payloads differ");
        zeros.field[0] = 1e-300;
        check(encode(zeros).contains("7"), "a denormal-scale value is written in hex as well ("
                + encode(zeros).split(";f=")[1].split(";")[0] + ")");
        State restored = decode(encode(zeros));
        check(restored.field[0] == 1e-300, "and it comes back exactly");

        Set<Long> digests = new HashSet<>();
        for (int i = 0; i < 1000; i++) {
            State variant = initial(4, i);
            digests.add(fnv1a(encode(variant)));
        }
        check(digests.size() >= 999, "1000 distinct payloads produce " + digests.size() + " distinct FNV-1a "
                + "digests, so the checksum is doing work");
    }

    static void crashRecoveryAndCost() {
        // A crash 250 steps in, with a checkpoint every 50 steps: 200 steps are recoverable, 50 are lost.
        State state = initial(12, 31337L);
        String checkpoint = null;
        for (int k = 0; k < 250; k++) {
            if (k % 50 == 0) checkpoint = encode(state);
            advance(state, 0.15);
        }
        check(state.step == 250, "the first attempt reaches step 250");
        State recovered = decode(checkpoint);
        check(recovered.step == 200, "the last checkpoint was taken at step " + recovered.step);
        int recomputed = 0;
        while (recovered.step < 250) {
            advance(recovered, 0.15);
            recomputed++;
        }
        check(recomputed == 50, "exactly " + recomputed + " steps had to be recomputed after the crash");
        check(encode(recovered).equals(encode(state)), "and the recovered state is byte-identical to the state the "
                + "crashed run had reached");

        RunResult result = run(500, 25, 30, 5150L, 0.2);
        int size = encode(result.state).length();
        check(size > 0 && size < 4000, "a 30-cell checkpoint is " + size + " characters, small enough to write "
                + "cheaply at every checkpoint");
        check(result.restores == 19, "the run performed " + result.restores + " crash-and-restore cycles: the "
                + "checkpoint at step 500 would be taken at the start of a step that does not exist, so 19 of the "
                + "20 multiples of 25 occur inside the 500 steps - an off-by-one the count makes explicit");
        check(result.state.step == 500 && result.checkpointsTaken.length == 19, "over 500 steps with a checkpoint "
                + "every 25 steps");
    }

    static void determinismAndNoClock() {
        String first = encode(run(200, 0, 10, 600L, 0.4).state);
        String second = encode(run(200, 0, 10, 600L, 0.4).state);
        check(first.equals(second), "two runs of the same length produce identical payloads");
        check(!first.contains("time") && !first.contains("date"), "and the payload carries no clock field at all, "
                + "which is how wall-clock independence is achieved rather than merely hoped for");
        String differentSeed = encode(run(200, 0, 10, 601L, 0.4).state);
        check(!first.equals(differentSeed), "a different seed gives a different payload");
        String differentDamping = encode(run(200, 0, 10, 600L, 0.45).state);
        check(!first.equals(differentDamping), "so does a different damping coefficient");
        State state = decode(first);
        check(state.step == 200 && state.count == 200, "the decoded counters are consistent");
        long draws = 200L * 10L;
        check(state.generatorState == lcgClosedForm(600L, draws), "and the generator state matches its closed form "
                + "after " + draws + " draws (field size times steps)");
        long walked = 600L;
        for (long i = 0; i < draws; i++) walked = lcg(walked);
        check(walked == lcgClosedForm(600L, draws), "the explicit walk agrees with the closed form (" + walked
                + "), so the BigInteger formula is verified against the loop as well");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
