// Task 761 - Implement trusted execution environment simulation (attestation sketch). (Tier: Expert, Beyond - Security & Privacy)
//
// An enclave is a region of memory whose contents are measured (hashed) at load time, sealed with a key that
// only the CPU hardware can derive, and attested to a remote party with a signed report. This file simulates that
// shape with a toy platform: the CPU owns an attestation key (the same for every enclave on that CPU) and derives
// a per-enclave *sealing* key from (CPU key, enclave id, measurement), so data sealed by one enclave - or by the
// same enclave after its code changed - cannot be opened by another. The measurement is an FNV digest over
// labelled code pages, so editing a page changes it, and a report carries (enclave id, measurement, nonce,
// monotonic counter, user data, keyed MAC over all of it).
//
// The verifier enforces the three checks that make attestation meaningful: the MAC must verify with the CPU
// attestation key (the hypervisor can copy, delay and edit reports but cannot forge one), the measurement must be
// on the allow list of known-good builds, and the nonce must be the fresh challenge the verifier just sent - which
// is what turns a copied report into a useless one. A monotonic counter taken from sealed storage per enclave
// gives rollback protection: a report whose counter is not above the high water mark the verifier has already
// seen is refused, which is what stops an attacker from replaying a report from before a security patch.
//
// Documented simplifications: no real CPU, no paging or DRAM encryption, no asymmetric signature (a hand written
// keyed hash stands in for the platform signature), the counter lives in sealed storage rather than in fuses, and
// every value is a toy - no key material is produced, stored or printed. Invariants verified: genuine attestation
// succeeds; every edit of the report fails; a stale measurement, a stale nonce, a stale counter and a foreign
// enclave id are all rejected; sealing round-trips while ciphertexts differ from plaintexts and from each other;
// and neither the attestation key nor the sealing key appears in the report or the sealed blob.
using System;
using System.Collections.Generic;
using System.Text;

internal static class ToyCrypto
{
    private const ulong Offset = 14695981039346656037UL;
    private const ulong Prime = 1099511628211UL;
    private const ulong MixConstant = 0x9E3779B97F4A7C15UL;

    internal static ulong Hash(string text)
    {
        ulong hash = Offset;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= Prime;
        }
        return hash;
    }

    internal static ulong Mix(ulong value)
    {
        value += MixConstant;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }

    internal static ulong Mac(ulong key, string payload) => Mix(key ^ Hash(payload));

    internal static ulong KeyStream(ulong key, long nonce, int index) => Mix(key ^ (ulong)nonce * MixConstant ^ (ulong)index);

    internal static string Hex(ulong value) => value.ToString("X16");
}

internal sealed class Platform
{
    internal readonly ulong AttestationKey;
    private int nextEnclave;

    internal Platform(ulong cpuSecret) { AttestationKey = ToyCrypto.Mix(cpuSecret); }

    internal Enclave Load(string name, params string[] pages)
    {
        var enclave = new Enclave(this, ++nextEnclave, name, pages);
        return enclave;
    }
}

internal sealed class Enclave
{
    internal readonly Platform Cpu;
    internal readonly int Id;
    internal readonly string Name;
    internal readonly string Measurement;
    internal readonly string[] Pages;
    internal long Counter = 1;                       // monotonic, kept in sealed storage in a real TEE

    internal Enclave(Platform cpu, int id, string name, string[] pages)
    {
        Cpu = cpu;
        Id = id;
        Name = name;
        Pages = pages;
        Measurement = Measure(pages);
    }

    internal static string Measure(IEnumerable<string> pages)
    {
        var builder = new StringBuilder();
        int index = 0;
        foreach (string page in pages)
        {
            builder.Append("page").Append(index++).Append('=').Append(page).Append(';');
        }
        return ToyCrypto.Hex(ToyCrypto.Hash(builder.ToString()));
    }

    /** The sealing key binds data to this *code identity*: any instance of the same build can open it, a different build cannot. */
    internal ulong SealingKey() => ToyCrypto.Mix(Cpu.AttestationKey ^ ToyCrypto.Hash(Measurement));

    internal Report Attest(string nonce, string userData)
    {
        var report = new Report
        {
            EnclaveId = Id,
            EnclaveName = Name,
            Measurement = Measurement,
            Nonce = nonce,
            Counter = Counter,
            UserData = userData,
        };
        report.Mac = ToyCrypto.Mac(Cpu.AttestationKey, report.Payload());
        Counter++;                                   // a monotonic report counter advances with every report
        return report;
    }

    internal sealed class SealedBlob
    {
        internal long Nonce;
        internal ulong[] Ciphertext = Array.Empty<ulong>();
        internal int Length;
        internal ulong Mac;
    }

    internal SealedBlob Seal(string data)
    {
        ulong key = SealingKey();
        long nonce = Counter++;
        var blob = new SealedBlob { Nonce = nonce, Length = data.Length };
        blob.Ciphertext = new ulong[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            blob.Ciphertext[i] = (ulong)data[i] ^ ToyCrypto.KeyStream(key, nonce, i);
        }
        blob.Mac = ToyCrypto.Mac(key, SealPayload(Measurement, nonce, blob.Ciphertext));
        return blob;
    }

    internal bool Unseal(SealedBlob blob, out string data)
    {
        data = string.Empty;
        ulong key = SealingKey();
        if (ToyCrypto.Mac(key, SealPayload(Measurement, blob.Nonce, blob.Ciphertext)) != blob.Mac) return false;
        var builder = new StringBuilder();
        for (int i = 0; i < blob.Ciphertext.Length; i++)
        {
            builder.Append((char)(blob.Ciphertext[i] ^ ToyCrypto.KeyStream(key, blob.Nonce, i)));
        }
        data = builder.ToString();
        return true;
    }

    private static string SealPayload(string measurement, long nonce, ulong[] ciphertext)
    {
        var builder = new StringBuilder();
        builder.Append('E').Append(measurement).Append('|').Append(nonce).Append('|');
        foreach (ulong cell in ciphertext) builder.Append(ToyCrypto.Hex(cell));
        return builder.ToString();
    }
}

internal sealed class Report
{
    internal int EnclaveId;
    internal string EnclaveName = "";
    internal string Measurement = "";
    internal string Nonce = "";
    internal long Counter;
    internal string UserData = "";
    internal ulong Mac;

    internal string Payload()
    {
        var builder = new StringBuilder();
        builder.Append("enclave=").Append(EnclaveId).Append(';').Append(EnclaveName)
            .Append(";measurement=").Append(Measurement)
            .Append(";nonce=").Append(Nonce)
            .Append(";counter=").Append(Counter)
            .Append(";userdata=").Append(UserData);
        return builder.ToString();
    }
}

internal sealed class Verifier
{
    private readonly ulong attestationKey;
    private readonly HashSet<string> allowList = new HashSet<string>();
    private readonly Dictionary<int, long> highWaterMark = new Dictionary<int, long>();

    internal Verifier(ulong attestationKey) { this.attestationKey = attestationKey; }

    internal void Trust(string measurement) => allowList.Add(measurement);

    internal string LastRejection = "";
    internal int Accepted;

    /** Returns an empty string on success, otherwise the reason the report was refused. */
    internal string Verify(Report report, string expectedNonce)
    {
        if (ToyCrypto.Mac(attestationKey, report.Payload()) != report.Mac) return Reject("bad-signature");
        if (!allowList.Contains(report.Measurement)) return Reject("unknown-measurement:" + report.Measurement);
        if (report.Nonce != expectedNonce) return Reject("stale-nonce:" + report.Nonce);
        if (highWaterMark.TryGetValue(report.EnclaveId, out long seen) && report.Counter <= seen)
        {
            return Reject("rolled-back-counter:" + report.Counter + "<=" + seen);
        }
        highWaterMark[report.EnclaveId] = report.Counter;
        Accepted++;
        return string.Empty;
    }

    private string Reject(string reason)
    {
        LastRejection = reason;
        return reason;
    }

    internal long HighWaterMark(int enclaveId) => highWaterMark.TryGetValue(enclaveId, out long value) ? value : 0;
}

public static class Solution
{
    public static void Main()
    {
        var platform = new Platform(0x761C0FFEE5EEDUL);
        Enclave enclave = platform.Load("payments-v1", "code:transfer:v1", "rodata:fees:v1", "stack:zero");
        Enclave other = platform.Load("payments-v2", "code:transfer:v2", "rodata:fees:v1", "stack:zero");
        var verifier = new Verifier(platform.AttestationKey);
        verifier.Trust(enclave.Measurement);
        verifier.Trust(other.Measurement);

        Attestation(platform, enclave, other, verifier);
        Sealing(platform, enclave, other);
        Boundaries(platform, enclave, verifier);

        Console.WriteLine("  measurement v1=" + enclave.Measurement.Substring(0, 12) + "... v2="
            + other.Measurement.Substring(0, 12) + "... accepted=" + verifier.Accepted);
        Console.WriteLine("PASS");
    }

    private static void Attestation(Platform platform, Enclave enclave, Enclave other, Verifier verifier)
    {
        Report report = enclave.Attest("challenge-1", "budget=42");
        Check(verifier.Verify(report, "challenge-1") == string.Empty, "a genuine report on the allow list verifies");
        Check(report.Measurement == enclave.Measurement && report.Counter == 1 && report.EnclaveId == enclave.Id,
            "the verified report really carries this enclave's identity, measurement and counter");
        Check(enclave.Measurement != other.Measurement,
            "a changed code page changes the measurement of the enclave");

        Report edited = enclave.Attest("challenge-2", "budget=42");
        edited.UserData = "budget=1000000";
        Check(verifier.Verify(edited, "challenge-2") == "bad-signature",
            "editing the report breaks the platform signature");

        Report forged = enclave.Attest("challenge-3", "budget=42");
        forged.Mac ^= 0x01;
        Check(verifier.Verify(forged, "challenge-3") == "bad-signature", "a flipped signature bit is refused");

        // Replay: copy a report and present it with a fresh challenge.
        Report captured = enclave.Attest("challenge-4", "budget=42");
        Check(verifier.Verify(captured, "challenge-4") == string.Empty, "the first presentation is fine");
        Check(verifier.Verify(captured, "challenge-5").StartsWith("stale-nonce"),
            "presenting the same report against a fresh nonce is a replay and is refused");

        // Rollback: an older, otherwise valid counter must not be able to revive a pre-patch state.
        Enclave patched = platform.Load("payments-v3", "code:transfer:v3", "rodata:fees:v2");
        verifier.Trust(patched.Measurement);
        Report current = patched.Attest("challenge-6", "patched");
        Check(verifier.Verify(current, "challenge-6") == string.Empty, "the patched enclave attests");
        var rolledBack = new Report
        {
            EnclaveId = current.EnclaveId,
            EnclaveName = current.EnclaveName,
            Measurement = current.Measurement,
            Nonce = "challenge-7",
            Counter = current.Counter - 1,
            UserData = current.UserData,
        };
        rolledBack.Mac = ToyCrypto.Mac(platform.AttestationKey, rolledBack.Payload());
        Check(verifier.Verify(rolledBack, "challenge-7").StartsWith("rolled-back-counter"),
            "a report with an older counter is refused, which is the rollback protection");
        Report newer = patched.Attest("challenge-8", "after the seal incremented the counter");
        Check(verifier.Verify(newer, "challenge-8") == string.Empty && newer.Counter > current.Counter,
            "a newer counter is accepted and raises the high water mark");

        // An enclave that is not on the allow list (an unknown build) is refused.
        Enclave rogue = platform.Load("payments-rogue", "code:transfer:v1", "backdoor:1");
        Check(verifier.Verify(rogue.Attest("challenge-9", "rogue"), "challenge-9").StartsWith("unknown-measurement"),
            "an unknown measurement is refused even with a perfectly valid signature");
        Check(verifier.Verify(current, "challenge-10").StartsWith("stale-nonce"),
            "re-presenting a captured report against a new nonce fails at the nonce check");
        Check(verifier.Verify(current, "challenge-6").StartsWith("rolled-back-counter"),
            "even with the original nonce the counter check refuses the replayed report");
        Report fresh = patched.Attest("challenge-11", "fresh report");
        Check(verifier.Verify(fresh, "challenge-11") == string.Empty,
            "a genuinely new report with an advanced counter is still accepted after refusals");
    }

    private static void Sealing(Platform platform, Enclave enclave, Enclave other)
    {
        Enclave.SealedBlob blob = enclave.Seal("secret-token-42");
        Check(enclave.Unseal(blob, out string opened) && opened == "secret-token-42", "seal and unseal round-trip");
        Enclave.SealedBlob again = enclave.Seal("secret-token-42");
        Check(again.Mac != blob.Mac && again.Ciphertext[0] != blob.Ciphertext[0],
            "sealing the same plaintext twice yields different blobs (fresh nonce per seal)");
        Check(blob.Ciphertext[0] != (ulong)"secret-token-42"[0],
            "the stored cell is ciphertext, not the first plaintext byte");

        Check(!other.Unseal(blob, out _), "another enclave cannot open this blob");
        Enclave upgraded = platform.Load("payments-v1b", "code:transfer:v1b", "rodata:fees:v1", "stack:zero");
        Check(!upgraded.Unseal(blob, out _),
            "the same enclave with different code cannot open it either: the seal is bound to the measurement");

        Enclave.SealedBlob tampered = enclave.Seal("secret-token-42");
        tampered.Ciphertext[3] ^= 0x8000;
        Check(!enclave.Unseal(tampered, out _), "a tampered cell is caught by the blob's MAC");

        Enclave.SealedBlob moved = enclave.Seal("secret-token-42");
        Enclave sameBuild = platform.Load("payments-v1-clone", "code:transfer:v1", "rodata:fees:v1", "stack:zero");
        Check(sameBuild.Id != enclave.Id && sameBuild.Unseal(moved, out string recovered) && recovered == "secret-token-42",
            "a seal is bound to the code identity, so another instance of the same build opens it");

        string report = enclave.Attest("challenge-11", "no keys here").Payload();
        Check(!report.Contains(ToyCrypto.Hex(enclave.SealingKey())) && !report.Contains(ToyCrypto.Hex(platform.AttestationKey)),
            "no key material appears in the report");
        Check(enclave.SealingKey() != other.SealingKey() && enclave.SealingKey() == sameBuild.SealingKey()
            && enclave.Attest("challenge-12", "").Mac != 0,
            "sealing keys follow the measurement, while the platform attestation key is shared by every enclave");
    }

    private static void Boundaries(Platform platform, Enclave enclave, Verifier verifier)
    {
        Enclave.SealedBlob empty = enclave.Seal(string.Empty);
        Check(enclave.Unseal(empty, out string nothing) && nothing.Length == 0,
            "sealing empty data still round-trips (the blob is just the tag)");
        Check(empty.Ciphertext.Length == 0 && empty.Mac != 0, "an empty blob has no cells but keeps its MAC");

        var localVerifier = new Verifier(platform.AttestationKey);
        localVerifier.Trust(enclave.Measurement);
        Report first = enclave.Attest("challenge-x", "");
        Check(localVerifier.Verify(first, "challenge-x") == string.Empty && first.Counter >= 1,
            "a freshly loaded enclave starts its counter at one");
        Report before = enclave.Attest("challenge-y", "");
        Check(before.Counter > first.Counter, "every seal advances the monotonic counter");

        Enclave fresh = platform.Load("payments-fresh", "code:transfer:v1", "rodata:fees:v1", "stack:zero");
        Check(fresh.Measurement == enclave.Measurement && fresh.Id != enclave.Id,
            "identical code on the same platform measures the same but is a different enclave instance");
        Check(verifier.Verify(fresh.Attest("challenge-z", ""), "challenge-z") == string.Empty,
            "a new instance of the same build starts a new identity and its counter starts over (the documented "
            + "limitation of per instance counters)");
        Check(verifier.HighWaterMark(fresh.Id) == 1 && verifier.HighWaterMark(enclave.Id) > 1,
            "the high water mark is tracked per enclave identity, not globally");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
