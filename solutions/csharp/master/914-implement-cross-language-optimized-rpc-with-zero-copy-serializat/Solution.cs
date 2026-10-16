// Task 914 - Implement cross-language optimized RPC with zero-copy serialization and backpressure.
// (Tier: Master, Beyond - Distributed Systems)
//
// A compact RPC wire format plus the transport discipline around it:
//
//  * frame = varint(payload length) | opcode | payload | crc32, so a reader finds every message boundary in
//    one pass and never needs a second buffer to interpret the stream;
//  * decoding is zero copy - a decoded message is an ArraySegment that still points into the transport
//    buffer, which the checks prove by comparing the segment's backing array with the transport array
//    instead of taking the comment's word for it;
//  * the sender is credit limited: a frame is written only while the peer has granted enough window,
//    otherwise it waits in a local FIFO queue.  In-flight bytes never exceed the window, nothing is dropped
//    or reordered, a frame larger than the window is reported as a stall rather than retried forever, and
//    the credit ledger is checked as an identity (granted = consumed + available).
//
// The cross-language aspect is honoured by keeping the encoding to plain varints and a shift-only crc32 that
// any language reproduces bit for bit; a second, independently written varint encoder is compared byte by
// byte at every boundary value.
//
// Simplifications: one in-process link, a fixed integer window, no sockets or threads and no congestion
// control beyond that window.  Verified: round-trips including the empty payload, varint boundaries against
// the independent encoder, the zero-copy backing-array identity, corruption detection, the window and ledger
// invariants, ordering with no loss, the oversized-frame stall, and determinism.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal static int WriteVarint(List<byte> sink, long value)
    {
        ulong remaining = (ulong)value;
        int written = 0;
        while (remaining >= 0x80)
        {
            sink.Add((byte)(remaining | 0x80));
            remaining >>= 7;
            written++;
        }
        sink.Add((byte)remaining);
        return written + 1;
    }

    internal static long ReadVarint(byte[] buffer, ref int offset)
    {
        long value = 0;
        int shift = 0;
        while (true)
        {
            byte current = buffer[offset++];
            value |= (long)(current & 0x7F) << shift;
            if ((current & 0x80) == 0) return value;
            shift += 7;
        }
    }

    /** An independently written varint encoder: different loop shape, same wire format. */
    private static List<byte> ReferenceVarint(long value)
    {
        List<byte> bytes = new List<byte>();
        long rest = value;
        do
        {
            byte chunk = (byte)(rest & 0x7F);
            rest = (long)((ulong)rest >> 7);
            if (rest != 0) chunk |= 0x80;
            bytes.Add(chunk);
        }
        while (rest != 0);
        return bytes;
    }

    internal static uint Crc32(byte[] data, int offset, int length)
    {
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < length; i++)
        {
            crc ^= data[offset + i];
            for (int bit = 0; bit < 8; bit++) crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
        }
        return ~crc;
    }

    internal static byte[] Encode(byte opcode, byte[] payload)
    {
        List<byte> frame = new List<byte>();
        WriteVarint(frame, payload.Length);
        frame.Add(opcode);
        frame.AddRange(payload);
        uint crc = Crc32(payload, 0, payload.Length);
        for (int shift = 0; shift < 32; shift += 8) frame.Add((byte)(crc >> shift));
        return frame.ToArray();
    }

    /** Zero-copy decode: the payload segment still points into the transport buffer it was read from. */
    internal static bool TryDecode(byte[] transport, ref int offset, out byte opcode, out ArraySegment<byte> payload, out string error)
    {
        opcode = 0;
        payload = default;
        error = string.Empty;
        if (offset >= transport.Length)
        {
            error = "no frame at offset " + offset;
            return false;
        }
        int start = offset;
        long length = ReadVarint(transport, ref offset);
        if (length < 0 || offset + 1 + length + 4 > transport.Length)
        {
            error = "truncated frame at " + start;
            return false;
        }
        opcode = transport[offset++];
        payload = new ArraySegment<byte>(transport, offset, (int)length);
        offset += (int)length;
        uint expected = 0;
        for (int shift = 0; shift < 32; shift += 8) expected |= (uint)transport[offset++] << shift;
        if (Crc32(transport, payload.Offset, payload.Count) != expected)
        {
            error = "checksum mismatch at frame starting at " + start;
            return false;
        }
        return true;
    }

    /** A credit-limited link: frames leave the queue only while the peer has granted enough window. */
    internal sealed class Link
    {
        private readonly int _window;
        private readonly List<(byte[] Frame, int Id)> _waiting = new List<(byte[], int)>();
        private readonly List<(int Offset, int Length, int Id)> _inFlight = new List<(int, int, int)>();
        private readonly byte[] _transport = new byte[64 * 1024];
        private int _used;
        private int _inFlightBytes;

        internal int Available;
        internal int Granted;
        internal int Consumed;
        internal int MaxInFlight;
        internal int Stalled;
        internal int Sent;
        internal int Waiting => _waiting.Count;
        internal int InFlightBytes => _inFlightBytes;
        internal readonly List<int> DeliveredIds = new List<int>();
        internal readonly List<string> DeliveredText = new List<string>();

        internal Link(int window)
        {
            _window = window;
        }

        internal bool TrySend(byte[] frame, int id)
        {
            if (frame.Length > _window)
            {
                Stalled++; // the window can never hold this frame, so it is reported instead of retried forever
                return false;
            }
            if (frame.Length > Available)
            {
                _waiting.Add((frame, id));
                return false;
            }
            Emit(frame, id);
            return true;
        }

        private void Emit(byte[] frame, int id)
        {
            Array.Copy(frame, 0, _transport, _used, frame.Length);
            _inFlight.Add((_used, frame.Length, id));
            _used += frame.Length;
            _inFlightBytes += frame.Length;
            Available -= frame.Length;
            MaxInFlight = Math.Max(MaxInFlight, _inFlightBytes);
            Sent++;
        }

        internal void Grant(int credits)
        {
            Available += credits;
            Granted += credits;
        }

        /** The receiver drains at most `budget` bytes worth of frames and grants the credits straight back. */
        internal int Drain(int budget)
        {
            int drained = 0;
            while (_inFlight.Count > 0)
            {
                (int offset, int length, int id) = _inFlight[0];
                if (drained + length > budget) break;
                _inFlight.RemoveAt(0);
                int cursor = offset;
                if (!TryDecode(_transport, ref cursor, out byte opcode, out ArraySegment<byte> payload, out string error))
                {
                    throw new Exception("FAIL: receiver could not decode a drained frame: " + error);
                }
                DeliveredIds.Add(id);
                DeliveredText.Add(Text(payload));
                if (opcode != (byte)(id % 256)) throw new Exception("FAIL: opcode does not match the message id");
                drained += length;
                _inFlightBytes -= length;
                Consumed += length;
                Available += length;
                Granted += length;
            }
            while (_waiting.Count > 0 && _waiting[0].Frame.Length <= Available)
            {
                (byte[] frame, int id) = _waiting[0];
                _waiting.RemoveAt(0);
                Emit(frame, id);
            }
            return drained;
        }
    }

    private static byte[] Payload(string text) => Encoding.UTF8.GetBytes(text);

    private static string Text(ArraySegment<byte> segment) =>
        Encoding.UTF8.GetString(segment.Array!, segment.Offset, segment.Count);

    private static byte[] Copy(byte[] source)
    {
        byte[] clone = new byte[source.Length];
        Array.Copy(source, clone, source.Length);
        return clone;
    }

    public static void Main()
    {
        // ---- 1. round-trips and varint boundaries ------------------------------------------------------
        string[] messages = { string.Empty, "ping", "a slightly longer rpc payload with fields", new string('x', 300) };
        int roundTrips = 0;
        foreach (string message in messages)
        {
            byte[] frame = Encode(7, Payload(message));
            byte[] single = Copy(frame);
            int offset = 0;
            bool ok = TryDecode(single, ref offset, out byte opcode, out ArraySegment<byte> payload, out string error);
            Check(ok, "frame decodes (" + error + ")");
            Check(opcode == 7 && Text(payload) == message, "frame round-trips a payload of length " + message.Length);
            Check(offset == frame.Length, "the reader lands exactly on the frame boundary");
            roundTrips++;
        }
        Check(roundTrips == 4, "four payload shapes round-tripped, including the empty one");
        long[] boundaries = { 0, 1, 127, 128, 16383, 16384, 2097151, int.MaxValue };
        int[] expectedLengths = { 1, 1, 1, 2, 2, 3, 3, 5 };
        int varintChecks = 0;
        for (int i = 0; i < boundaries.Length; i++)
        {
            List<byte> encoded = new List<byte>();
            int written = WriteVarint(encoded, boundaries[i]);
            Check(written == expectedLengths[i], "varint " + boundaries[i] + " takes " + expectedLengths[i] + " byte(s)");
            byte[] packed = encoded.ToArray();
            int offset = 0;
            Check(ReadVarint(packed, ref offset) == boundaries[i] && offset == packed.Length,
                "varint " + boundaries[i] + " decodes back to itself");
            List<byte> reference = ReferenceVarint(boundaries[i]);
            bool equal = reference.Count == encoded.Count;
            for (int j = 0; j < reference.Count && equal; j++) equal = reference[j] == encoded[j];
            Check(equal, "the independent varint encoder agrees byte for byte at " + boundaries[i]);
            varintChecks += 3;
        }
        Check(Encode(1, Payload("x")).Length == 1 + 1 + 1 + 4, "a one-byte payload frame is header + payload + crc");

        // ---- 2. zero-copy decoding ------------------------------------------------------------------
        byte[] transport = new byte[512];
        int cursor = 0;
        List<int> offsets = new List<int>();
        for (int i = 0; i < 5; i++)
        {
            byte[] frame = Encode((byte)(10 + i), Payload("message-" + i));
            Array.Copy(frame, 0, transport, cursor, frame.Length);
            cursor += frame.Length;
        }
        int reader = 0;
        bool sameBacking = true;
        for (int i = 0; i < 5; i++)
        {
            Check(TryDecode(transport, ref reader, out byte opcode, out ArraySegment<byte> payload, out _), "zero-copy frame decodes");
            sameBacking &= ReferenceEquals(payload.Array, transport);
            offsets.Add(payload.Offset);
            Check(opcode == (byte)(10 + i) && Text(payload) == "message-" + i, "zero-copy payload matches the sent text");
        }
        Check(sameBacking, "every decoded payload still points at the transport buffer (no copy)");
        bool monotone = true;
        for (int i = 1; i < offsets.Count; i++) monotone &= offsets[i] > offsets[i - 1];
        Check(monotone, "the payload offsets advance monotonically inside the single buffer");
        Check(reader == cursor, "the zero-copy pass ended exactly at the end of the batch");

        // ---- 3. corruption detection ------------------------------------------------------------------
        byte[] corrupt = Copy(transport);
        corrupt[offsets[1] + 1] ^= 0x40; // flip a bit inside the second payload
        int corruptReader = 0;
        Check(TryDecode(corrupt, ref corruptReader, out _, out _, out _), "the first frame still decodes in the corrupted buffer");
        Check(!TryDecode(corrupt, ref corruptReader, out _, out _, out string corruptError), "the corrupted frame is rejected");
        Check(corruptError.Contains("checksum", StringComparison.Ordinal), "and the rejection names the checksum (" + corruptError + ")");
        Check(Crc32(Payload("abc"), 0, 3) != Crc32(Payload("abd"), 0, 3), "crc32 separates different payloads");

        // ---- 4. backpressure over the link ---------------------------------------------------------------
        Link link = new Link(64);
        link.Grant(64);
        int queued = 0;
        for (int i = 0; i < 12; i++)
        {
            byte[] frame = Encode((byte)i, Payload("payload-" + i));
            if (!link.TrySend(frame, i)) queued++;
            Check(link.InFlightBytes <= 64, "in-flight bytes stay inside the window (" + link.InFlightBytes + ")");
            Check(link.Available >= 0, "credits are never overdrawn");
        }
        Check(queued > 0 && link.Waiting == queued, "the window forced " + queued + " frames to wait locally");
        Check(link.Sent + link.Waiting == 12, "every frame is sent or waiting: " + link.Sent + " + " + link.Waiting + " = 12");
        Check(link.Stalled == 0, "no frame exceeded the window");
        int delivered = 0;
        int rounds = 0;
        while (delivered < 12 && rounds < 40)
        {
            link.Drain(24);
            delivered = link.DeliveredIds.Count;
            rounds++;
            Check(link.InFlightBytes <= 64, "post-drain in-flight bytes stay inside the window");
            Check(link.Granted == link.Consumed + link.Available + link.InFlightBytes,
                "the credit ledger balances after every round");
        }
        Check(link.DeliveredIds.Count == 12, "all twelve frames arrived (" + link.DeliveredIds.Count + ")");
        bool inOrder = true;
        for (int i = 0; i < link.DeliveredIds.Count; i++) inOrder &= link.DeliveredIds[i] == i;
        Check(inOrder, "frames arrived in FIFO order with no loss and no duplication");
        Check(link.DeliveredText[3] == "payload-3" && link.DeliveredText[11] == "payload-11", "payloads survived the queueing");
        Check(link.MaxInFlight <= 64, "the observed maximum in flight was " + link.MaxInFlight + " bytes");

        // ---- 5. boundaries ---------------------------------------------------------------------------------
        Link tiny = new Link(8);
        tiny.Grant(8);
        Check(!tiny.TrySend(Encode(1, Payload(new string('y', 40))), 0) && tiny.Stalled == 1 && tiny.Waiting == 0,
            "a frame larger than the whole window is reported as a stall, and never queued");
        Link fitting = new Link(64);
        fitting.Grant(64);
        byte[] fit = Encode(1, new byte[58]); // 1 + 1 + 58 + 4 = 64 bytes exactly
        Check(fit.Length == 64, "the exactly-fitting frame is 64 bytes (" + fit.Length + ")");
        Check(fitting.TrySend(fit, 7) && fitting.Available == 0, "a frame that exactly fills the window is accepted");
        Check(!fitting.TrySend(Encode(1, new byte[1]), 8) && fitting.Waiting == 1, "the next frame has to wait");
        byte[] emptyFrame = Encode(0, new byte[0]);
        Check(emptyFrame.Length == 6, "an empty payload frame is six bytes");
        int emptyCursor = 0;
        Check(TryDecode(emptyFrame, ref emptyCursor, out _, out ArraySegment<byte> emptyPayload, out _) && emptyPayload.Count == 0,
            "an empty payload decodes to a zero-length segment");

        // ---- 6. determinism -----------------------------------------------------------------------------------
        byte[] first = Encode(3, Payload("same"));
        byte[] second = Encode(3, Payload("same"));
        bool identical = first.Length == second.Length;
        for (int i = 0; i < first.Length && identical; i++) identical = first[i] == second[i];
        Check(identical, "encoding the same message twice produces identical bytes");

        Console.WriteLine("frames: " + roundTrips + " payload shapes round-tripped, " + varintChecks +
            " varint boundary checks against an independently written encoder");
        Console.WriteLine("zero copy: " + offsets.Count + " payloads decoded in place inside one " + transport.Length +
            "-byte transport buffer, corruption in the second frame detected by crc32");
        Console.WriteLine("backpressure: window 64 bytes, " + link.Sent + " frames sent, " + link.Waiting +
            " queued, max in flight " + link.MaxInFlight + " bytes, credit ledger balanced");
        Console.WriteLine("PASS");
    }
}
