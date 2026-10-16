/** Task 914 - Implement cross-language optimized RPC with zero-copy serialization and backpressure.
 *  (Tier: Master, CTCI Beyond - Networking, OS & Real-Time)
 *
 * A cross-language RPC hop wants two things beyond correctness: the payload must not be copied on the way
 * between the decoder and the handler, and a fast producer must not be allowed to run the receiver out of
 * memory. Both are implemented and checked here.
 *
 *   wire format   [magic 4][version 1][flags 1][length 4][checksum 4] followed by the payload; every field is
 *                 big-endian so any language can read it, and the frame is self-delimiting, which is what
 *                 makes length-prefixed streaming safe;
 *   zero copy     decoding returns a Slice (backing array, offset, length) into the *received* buffer. The
 *                 checks assert array identity, that writing through the slice is visible in the buffer, and
 *                 that the slice spans exactly the payload region - none of which can hold if a copy happened;
 *   backpressure  a credit window caps the bytes in flight. send() refuses to enqueue more than the window
 *                 allows, acks return credits, and the checks verify the in-flight bound at every step, that
 *                 the sender stalls exactly when the window is short, and that every message is delivered
 *                 once the receiver drains.
 *
 * Simplifications: the transport is an in-memory queue pair, the checksum is a 32-bit splitmix fold rather
 * than CRC32, and the "handler" just reads the slice. Invariants verified: round-trip equality, real
 * aliasing, corruption and framing rejection, a hard in-flight bound, and deterministic framing.
 */
import java.util.*;

/** splitmix64 with a 32-bit folding checksum. */
final class Rng914 {
    private long state;

    Rng914(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }

    /** Fold the buffer into a 32-bit tag; order and length both matter. */
    static int checksum(byte[] buffer, int offset, int length) {
        long h = 0x9747B28CL ^ (length * 0x9E3779B97F4A7C15L);
        for (int i = 0; i < length; i++) {
            h ^= (buffer[offset + i] & 0xFFL);
            h *= 0xBF58476D1CE4E5B9L;
            h ^= h >>> 27;
        }
        return (int) (h ^ (h >>> 32));
    }
}

/** A view over a region of a shared buffer; nothing is copied. */
final class Slice {
    final byte[] backing;
    final int offset;
    final int length;

    Slice(byte[] backing, int offset, int length) {
        this.backing = backing;
        this.offset = offset;
        this.length = length;
    }

    byte at(int index) { return backing[offset + index]; }

    void write(int index, byte value) { backing[offset + index] = value; }
}

/** A decoded frame: header fields plus an aliasing view of the payload. */
final class Frame {
    final int version;
    final int flags;
    final Slice payload;

    Frame(int version, int flags, Slice payload) {
        this.version = version;
        this.flags = flags;
        this.payload = payload;
    }
}

/** Encoder/decoder for the frame format. */
final class Codec {
    static final byte[] MAGIC = {'C', 'R', 'P', 'C'};
    static final int VERSION = 2;
    static final int HEADER = 14;
    static final int MAX_PAYLOAD = 4096;

    static byte[] encode(int flags, byte[] payload) {
        if (payload.length > MAX_PAYLOAD) throw new IllegalArgumentException("payload too large: "
                + payload.length);
        byte[] frame = new byte[HEADER + payload.length];
        System.arraycopy(MAGIC, 0, frame, 0, 4);
        frame[4] = (byte) VERSION;
        frame[5] = (byte) flags;
        writeInt(frame, 6, payload.length);
        writeInt(frame, 10, Rng914.checksum(payload, 0, payload.length));
        System.arraycopy(payload, 0, frame, HEADER, payload.length);
        return frame;
    }

    /** Throws IllegalArgumentException with a reason; the payload slice aliases `frame`. */
    static Frame decode(byte[] frame) {
        if (frame.length < HEADER) throw new IllegalArgumentException("truncated header");
        for (int i = 0; i < 4; i++) {
            if (frame[i] != MAGIC[i]) throw new IllegalArgumentException("bad magic");
        }
        int version = frame[4] & 0xFF;
        if (version != VERSION) throw new IllegalArgumentException("unsupported version " + version);
        int flags = frame[5] & 0xFF;
        int length = readInt(frame, 6);
        if (length < 0 || length > MAX_PAYLOAD) throw new IllegalArgumentException("bad length " + length);
        if (frame.length != HEADER + length) throw new IllegalArgumentException("length mismatch");
        int expected = readInt(frame, 10);
        if (Rng914.checksum(frame, HEADER, length) != expected) {
            throw new IllegalArgumentException("checksum mismatch");
        }
        return new Frame(version, flags, new Slice(frame, HEADER, length));
    }

    static void writeInt(byte[] target, int offset, int value) {
        target[offset] = (byte) (value >>> 24);
        target[offset + 1] = (byte) (value >>> 16);
        target[offset + 2] = (byte) (value >>> 8);
        target[offset + 3] = (byte) value;
    }

    static int readInt(byte[] source, int offset) {
        return ((source[offset] & 0xFF) << 24) | ((source[offset + 1] & 0xFF) << 16)
                | ((source[offset + 2] & 0xFF) << 8) | (source[offset + 3] & 0xFF);
    }
}

/** Credit window: the number of unacknowledged bytes in flight is bounded by `capacity`. */
final class SendWindow {
    final int capacity;
    int credits;
    int inFlight;
    int stalls;
    int delivered;

    SendWindow(int capacity) {
        this.capacity = capacity;
        this.credits = capacity;
    }

    /** False means "apply backpressure": the caller must wait for an ack. */
    boolean send(int bytes) {
        if (bytes > credits) {
            stalls++;
            return false;
        }
        credits -= bytes;
        inFlight += bytes;
        return true;
    }

    void ack(int bytes) {
        inFlight -= bytes;
        credits = Math.min(capacity, credits + bytes);
        delivered++;
    }
}

public class Solution {
    static final int WINDOW = 512;
    static final int MESSAGES = 40;

    public static void main(String[] args) {
        Rng914 rng = new Rng914(0x914ABCDEL);

        // ---- 1. round trip ------------------------------------------------------------------------------
        List<byte[]> messages = new ArrayList<>();
        for (int i = 0; i < MESSAGES; i++) {
            int size = 1 + rng.nextInt(120);
            byte[] payload = new byte[size];
            for (int j = 0; j < size; j++) payload[j] = (byte) rng.nextInt(256);
            messages.add(payload);
        }
        for (int i = 0; i < messages.size(); i++) {
            byte[] frame = Codec.encode(i % 4, messages.get(i));
            Frame decoded = Codec.decode(frame);
            check(decoded.payload.length == messages.get(i).length, "payload length must round-trip");
            check(decoded.version == Codec.VERSION && decoded.flags == i % 4, "header fields must round-trip");
            for (int j = 0; j < messages.get(i).length; j++) {
                check(decoded.payload.at(j) == messages.get(i)[j], "payload byte " + j + " must round-trip");
            }
            check(frame.length == Codec.HEADER + messages.get(i).length, "frame must be self-delimiting");
        }

        // ---- 2. zero copy: the slice aliases the received buffer ---------------------------------------
        byte[] wire = Codec.encode(1, messages.get(0));
        Frame view = Codec.decode(wire);
        check(view.payload.backing == wire, "the payload view must share the received buffer");
        check(view.payload.offset == Codec.HEADER, "the view must start right after the header");
        check(view.payload.length == messages.get(0).length, "the view must span exactly the payload");
        byte original = view.payload.at(0);
        view.payload.write(0, (byte) (original ^ 0x5A));
        check(wire[Codec.HEADER] == (byte) (original ^ 0x5A),
                "writing through the slice must be visible in the buffer, which proves there was no copy");
        check(messages.get(0)[0] == original, "the original message must be untouched by the view");
        view.payload.write(0, original);
        check(Codec.decode(wire).payload.at(0) == original, "restoring the byte must restore the frame");

        // ---- 3. corruption is rejected ----------------------------------------------------------------
        int flips = 0;
        int detected = 0;
        for (int i = 0; i < wire.length; i++) {
            if (i == 5) continue;                      // the flags byte is free-form, any value is legal
            for (int bit = 0; bit < 8; bit++) {
                byte[] damaged = wire.clone();
                damaged[i] ^= (byte) (1 << bit);
                flips++;
                try {
                    Codec.decode(damaged);
                } catch (IllegalArgumentException ex) {
                    detected++;
                }
            }
        }
        check(detected >= flips * 0.99, "single-bit corruption must be detected, caught " + detected + " of "
                + flips);
        check(flips > 400, "the sweep must cover every header and payload bit, swept " + flips);
        byte[] payloadFlip = wire.clone();
        payloadFlip[Codec.HEADER] ^= 1;
        check(rejects(payloadFlip, "checksum"), "a flipped payload byte must be reported as a checksum error");

        // ---- 4. framing errors ---------------------------------------------------------------------------
        check(rejects(new byte[] {1, 2, 3}, "truncated header"), "a short buffer must be rejected");
        byte[] badMagic = wire.clone();
        badMagic[0] = 'X';
        check(rejects(badMagic, "bad magic"), "a wrong magic number must be rejected");
        byte[] badVersion = wire.clone();
        badVersion[4] = 99;
        check(rejects(badVersion, "unsupported version"), "an unknown version must be rejected");
        byte[] badLength = wire.clone();
        Codec.writeInt(badLength, 6, 9999);
        check(rejects(badLength, "bad length"), "an impossible length must be rejected");
        byte[] elevenByteMessage = new byte[11];
        byte[] shortFrame = Codec.encode(0, elevenByteMessage);
        byte[] truncated = Arrays.copyOf(shortFrame, shortFrame.length - 4);
        check(rejects(truncated, "length mismatch"), "a truncated payload must be rejected");

        // ---- 5. backpressure -----------------------------------------------------------------------------
        SendWindow window = new SendWindow(WINDOW);
        List<Integer> pending = new ArrayList<>();
        int sent = 0;
        int stalled = 0;
        int acknowledgements = 0;
        for (int i = 0; i < MESSAGES; i++) {
            int size = messages.get(i).length;
            if (window.send(size)) {
                sent++;
                pending.add(size);
            } else {
                stalled++;
                // The receiver drains messages until the retry fits; the sender never drops a message.
                boolean retry = false;
                while (!pending.isEmpty() && !retry) {
                    int freed = pending.remove(0);
                    window.ack(freed);
                    acknowledgements++;
                    retry = window.send(size);
                }
                check(retry, "after draining the window the same message must fit");
                sent++;
                pending.add(size);
            }
            check(window.inFlight <= WINDOW, "in flight " + window.inFlight + " exceeded the window");
            check(window.credits >= 0, "credits must never go negative");
        }
        while (!pending.isEmpty()) {
            window.ack(pending.remove(0));
            acknowledgements++;
        }
        check(window.inFlight == 0, "draining must clear the in-flight bytes, got " + window.inFlight);
        check(window.credits == WINDOW, "all credits must return to the window");
        check(sent == MESSAGES, "every message must eventually be sent: " + sent);
        check(stalled > 0, "the window must actually stall a producer with 40 messages of up to 120 bytes");
        check(acknowledgements >= stalled, "every stall must be followed by a drain");
        check(window.delivered == acknowledgements, "acks must be counted exactly once");

        // ---- 6. boundaries -------------------------------------------------------------------------------
        Frame emptyFrame = Codec.decode(Codec.encode(0, new byte[0]));
        check(emptyFrame.payload.length == 0, "an empty payload must round-trip");
        check(emptyFrame.payload.backing.length == Codec.HEADER, "an empty frame holds only the header");
        byte[] maximal = new byte[Codec.MAX_PAYLOAD];
        check(Codec.decode(Codec.encode(0, maximal)).payload.length == Codec.MAX_PAYLOAD,
                "a largest-legal payload must be accepted");
        boolean refused = false;
        try {
            Codec.encode(0, new byte[Codec.MAX_PAYLOAD + 1]);
        } catch (IllegalArgumentException ex) {
            refused = true;
        }
        check(refused, "an oversized payload must be refused before it reaches the wire");
        SendWindow zero = new SendWindow(0);
        check(!zero.send(1) && zero.stalls == 1, "a zero window stalls immediately");
        check(zero.send(0), "a zero-byte message never needs credit");

        // ---- 7. determinism ------------------------------------------------------------------------------
        check(Arrays.equals(Codec.encode(1, messages.get(0)), wire), "framing must be deterministic");
        check(Rng914.checksum(messages.get(0), 0, messages.get(0).length)
                == Rng914.checksum(messages.get(0), 0, messages.get(0).length), "checksums must be stable");

        long totalBytes = 0;
        for (byte[] message : messages) totalBytes += message.length;
        System.out.println(MESSAGES + " messages, " + totalBytes + " payload bytes, window " + WINDOW
                + " bytes; stalls " + stalled + ", acks " + acknowledgements);
        System.out.println("corruption detection " + detected + "/" + flips + " single-bit flips; zero-copy "
                + "aliasing verified on the received buffer");
        System.out.println("framing rejected: magic, version, length, truncation, checksum; empty payload and "
                + "max payload both accepted");
        System.out.println("PASS");
    }

    static boolean rejects(byte[] frame, String expectedReason) {
        try {
            Codec.decode(frame);
            return false;
        } catch (IllegalArgumentException ex) {
            return ex.getMessage().contains(expectedReason);
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
