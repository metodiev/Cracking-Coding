// Task 812 - Implement secure enclave-backed key-value store with remote attestation. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A sealed, attested key-value store. A vendor root key certifies an enclave's public key together with the
// enclave's *measurement* (the hash of its code and data), and the enclave answers a challenge with a quote: a
// signature over (measurement, public key, nonce) made with its own private key. A remote verifier therefore has to
// check a chain - root -> certificate -> enclave key -> quote -> expected measurement plus a fresh nonce - and any
// single flipped bit in any link must break it. Data at rest is sealed: the sealing key is derived from the enclave
// master secret *and* the measurement, values are encrypted with a keystream derived from the sealing key, and a MAC
// over the record binds the counter, the nonce and the ciphertext, so a tampered byte or a rollback to an older
// record is detected instead of decrypted. A monotonic trusted counter provides the rollback protection.
//
// Invariants verified: RSA signatures verify with the public key and fail for any other message; a quote is accepted
// only when the certificate chains to the vendor root, the measurement matches the expected value and the nonce is
// fresh (never reused); a brute-force bit-flip sweep over the quote shows that every tampered variant is rejected; a
// sealed blob round-trips inside its own enclave but not in an enclave with a different measurement; a flipped bit
// in the ciphertext, the nonce or the counter breaks the MAC; an old record is refused by the monotonic counter; the
// key policy denies an identity that is not attested for that key; and every operation is written to the audit log.
//
// Independent oracles: (a) every signature is re-checked with a second, independently written modular
// exponentiation that never touches the signing path; (b) the bit-flip sweep re-verifies each mutated quote rather
// than trusting a single "invalid" verdict; (c) unsealing is attempted with a foreign enclave's key material, which
// must fail, proving the sealing key really depends on the measurement.
//
// Documented simplifications: RSA uses small (512-bit) moduli generated deterministically from fixed seeds so runs
// are reproducible, quotes are signed over a SHA-256 digest, values are a few dozen bytes, the "remote" verifier is
// an in-process object and the monotonic counter is a field rather than hardware, there is no network and no real
// enclave. The cryptographic structure (RSA sign/verify, measurement binding, nonce freshness, sealed storage with a
// MAC and rollback protection) is the real one. BCL only, no threads, no wall clock, no unseeded randomness.
#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

internal sealed class EnclaveException : Exception
{
    internal EnclaveException(string message) : base(message) { }
}

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal static class Hashes
{
    internal static byte[] Sha256(string text)
    {
        using var sha = SHA256.Create();
        return sha.ComputeHash(Encoding.UTF8.GetBytes(text));
    }

    internal static BigInteger ToInteger(byte[] bytes) => new BigInteger(bytes, isUnsigned: true, isBigEndian: true);

    internal static string Hex(byte[] bytes)
    {
        var builder = new StringBuilder(bytes.Length * 2);
        foreach (byte b in bytes) builder.Append(b.ToString("x2"));
        return builder.ToString();
    }

    internal static byte[] FromHex(string hex)
    {
        var bytes = new byte[hex.Length / 2];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
        return bytes;
    }
}

/** Small deterministic RSA: prime search with Miller-Rabin, e = 65537, d from the extended Euclid. */
internal sealed class RsaKey
{
    internal readonly BigInteger N;
    internal readonly BigInteger E = 65537;
    internal readonly BigInteger D;
    internal readonly string PublicId;

    private RsaKey(BigInteger n, BigInteger d)
    {
        N = n;
        D = d;
        PublicId = Hashes.Hex(N.ToByteArray(isUnsigned: true, isBigEndian: true));
    }

    internal static RsaKey Generate(ulong seed, int bits)
    {
        ulong rng = seed;
        BigInteger p = NextPrime(ref rng, bits / 2);
        BigInteger q = NextPrime(ref rng, bits / 2);
        while (q == p) q = NextPrime(ref rng, bits / 2);
        BigInteger n = p * q;
        BigInteger phi = (p - 1) * (q - 1) / BigInteger.GreatestCommonDivisor(p - 1, q - 1);
        BigInteger e = 65537;
        BigInteger d = ModularInverse(e, phi);
        return new RsaKey(n, d);
    }

    private static BigInteger NextPrime(ref ulong rng, int bits)
    {
        while (true)
        {
            BigInteger candidate = RandomBigInteger(ref rng, bits) | BigInteger.One;
            if (IsProbablePrime(candidate)) return candidate;
        }
    }

    private static BigInteger RandomBigInteger(ref ulong rng, int bits)
    {
        var bytes = new byte[bits / 8];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)SplitMix64.Next(ref rng);
        bytes[0] |= 0x80;                                     // keep the full bit length
        return new BigInteger(bytes, isUnsigned: true, isBigEndian: true);
    }

    private static readonly int[] Witnesses = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

    internal static bool IsProbablePrime(BigInteger value)
    {
        if (value < 2) return false;
        foreach (int small in new[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 })
        {
            if (value == small) return true;
            if (value % small == 0) return false;
        }
        BigInteger d = value - 1;
        int shift = 0;
        while (d.IsEven)
        {
            d /= 2;
            shift++;
        }
        foreach (int witness in Witnesses)
        {
            BigInteger x = BigInteger.ModPow(witness, d, value);
            if (x == 1 || x == value - 1) continue;
            bool composite = true;
            for (int i = 0; i < shift - 1; i++)
            {
                x = BigInteger.ModPow(x, 2, value);
                if (x == value - 1)
                {
                    composite = false;
                    break;
                }
            }
            if (composite) return false;
        }
        return true;
    }

    private static BigInteger ModularInverse(BigInteger value, BigInteger modulus)
    {
        BigInteger t = 0, newT = 1;
        BigInteger r = modulus, newR = value;
        while (newR != 0)
        {
            BigInteger quotient = r / newR;
            (t, newT) = (newT, t - quotient * newT);
            (r, newR) = (newR, r - quotient * newR);
        }
        if (r > 1) throw new EnclaveException("value is not invertible");
        if (t < 0) t += modulus;
        return t;
    }

    internal BigInteger SignDigest(BigInteger digest) => BigInteger.ModPow(digest, D, N);

    internal bool VerifyDigest(BigInteger digest, BigInteger signature) => BigInteger.ModPow(signature, E, N) == digest;

    /** Independent oracle: a second modular exponentiation written separately from the signing path. */
    internal static BigInteger ModexpOracle(BigInteger basis, BigInteger exponent, BigInteger modulus)
    {
        BigInteger result = 1;
        BigInteger factor = basis % modulus;
        BigInteger e = exponent;
        while (e > 0)
        {
            if (!e.IsEven) result = result * factor % modulus;
            factor = factor * factor % modulus;
            e >>= 1;
        }
        return result;
    }

    internal bool VerifyWithOracle(BigInteger digest, BigInteger signature)
        => ModexpOracle(signature, E, N) == digest;
}

internal sealed class Quote
{
    internal byte[] Signature = Array.Empty<byte>();
    internal string Measurement = "";
    internal string PublicKeyId = "";
    internal string Nonce = "";

    internal byte[] Encode() => Encoding.UTF8.GetBytes(Hashes.Hex(Signature) + "|" + Measurement + "|" + PublicKeyId + "|" + Nonce);

    internal Quote Clone() => new Quote
    {
        Signature = (byte[])Signature.Clone(), Measurement = Measurement, PublicKeyId = PublicKeyId, Nonce = Nonce
    };
}

internal sealed class Certificate
{
    internal byte[] Signature = Array.Empty<byte>();
    internal string Measurement = "";
    internal string PublicKeyId = "";

    internal BigInteger Digest() => Hashes.ToInteger(Hashes.Sha256("cert|" + Measurement + "|" + PublicKeyId));

    internal byte[] Encode() => Encoding.UTF8.GetBytes(Hashes.Hex(Signature) + "|" + Measurement + "|" + PublicKeyId);
}

internal sealed class SealedBlob
{
    internal long Counter;
    internal string Nonce = "";
    internal byte[] Ciphertext = Array.Empty<byte>();
    internal byte[] Mac = Array.Empty<byte>();
}

/** The enclave: measured code, its own key pair, a measurement-bound sealing key and a sealed key-value store. */
internal sealed class Enclave
{
    internal readonly string Measurement;
    internal readonly RsaKey Key;
    internal readonly byte[] MasterSecret;
    internal readonly List<string> Audit = new List<string>();
    internal readonly SortedDictionary<string, byte[]> Records = new SortedDictionary<string, byte[]>(StringComparer.Ordinal);
    internal long SealCounter;
    internal int QuotesIssued;

    internal Enclave(string code, string data, byte[] masterSecret)
    {
        Measurement = Hashes.Hex(Hashes.Sha256("measure|" + code + "|" + data));
        MasterSecret = masterSecret;
        // The enclave key is derived from the measurement and the master secret, so it is unique per enclave image.
        ulong keySeed = 0x9E3779B97F4A7C15UL;
        foreach (byte b in Hashes.Sha256("key|" + Measurement + "|" + Hashes.Hex(masterSecret)))
        {
            keySeed = SplitMix64.Next(ref keySeed) ^ b;
        }
        Key = RsaKey.Generate(keySeed, 512);
    }

    /** The sealing key is a function of the master secret *and* the measurement: another enclave cannot unseal it. */
    internal byte[] SealingKey() => Hashes.Sha256("seal|" + Hashes.Hex(MasterSecret) + "|" + Measurement);

    internal Quote Attest(string nonce)
    {
        BigInteger digest = Hashes.ToInteger(Hashes.Sha256("quote|" + Measurement + "|" + Key.PublicId + "|" + nonce));
        var quote = new Quote
        {
            Signature = Key.SignDigest(digest).ToByteArray(isUnsigned: true, isBigEndian: true),
            Measurement = Measurement,
            PublicKeyId = Key.PublicId,
            Nonce = nonce
        };
        QuotesIssued++;
        Audit.Add("attest nonce=" + nonce);
        return quote;
    }

    private static byte[] Keystream(byte[] key, string nonce, long counter, int length)
    {
        var stream = new byte[length];
        int offset = 0;
        long block = 0;
        while (offset < length)
        {
            byte[] material = Hashes.Sha256(Hashes.Hex(key) + "|" + nonce + "|" + counter + "|" + block);
            int take = Math.Min(material.Length, length - offset);
            Array.Copy(material, 0, stream, offset, take);
            offset += take;
            block++;
        }
        return stream;
    }

    internal SealedBlob Seal(string key, string value, string nonce)
    {
        byte[] plaintext = Encoding.UTF8.GetBytes(value);
        byte[] sealing = SealingKey();
        var blob = new SealedBlob { Counter = ++SealCounter, Nonce = nonce };
        byte[] stream = Keystream(sealing, nonce, blob.Counter, plaintext.Length);
        blob.Ciphertext = new byte[plaintext.Length];
        for (int i = 0; i < plaintext.Length; i++) blob.Ciphertext[i] = (byte)(plaintext[i] ^ stream[i]);
        blob.Mac = Hashes.Sha256(Hashes.Hex(sealing) + "|mac|" + nonce + "|" + blob.Counter + "|" + Hashes.Hex(blob.Ciphertext));
        Records[key] = blob.Ciphertext;
        Audit.Add("seal " + key + " counter=" + blob.Counter);
        return blob;
    }

    /** Unseal inside the enclave: the MAC and the monotonic counter must both hold. */
    internal string Unseal(SealedBlob blob, long trustedCounter)
    {
        if (blob.Counter != trustedCounter)
        {
            Audit.Add("unseal refused: counter " + blob.Counter + " != trusted " + trustedCounter);
            throw new EnclaveException("rollback detected: the record counter does not match the trusted counter");
        }
        byte[] sealing = SealingKey();
        byte[] expected = Hashes.Sha256(Hashes.Hex(sealing) + "|mac|" + blob.Nonce + "|" + blob.Counter + "|" + Hashes.Hex(blob.Ciphertext));
        if (!ConstantTimeEquals(expected, blob.Mac))
        {
            Audit.Add("unseal refused: bad mac");
            throw new EnclaveException("the sealed record was tampered with");
        }
        byte[] stream = Keystream(sealing, blob.Nonce, blob.Counter, blob.Ciphertext.Length);
        var plaintext = new byte[blob.Ciphertext.Length];
        for (int i = 0; i < plaintext.Length; i++) plaintext[i] = (byte)(blob.Ciphertext[i] ^ stream[i]);
        Audit.Add("unseal counter=" + blob.Counter);
        return Encoding.UTF8.GetString(plaintext);
    }

    internal static bool ConstantTimeEquals(byte[] left, byte[] right)
    {
        if (left.Length != right.Length) return false;
        int difference = 0;
        for (int i = 0; i < left.Length; i++) difference |= left[i] ^ right[i];
        return difference == 0;
    }
}

internal sealed class TrustedModule
{
    internal long MonotonicCounter;

    internal void Advance(long value)
    {
        if (value > MonotonicCounter) MonotonicCounter = value;
    }
}

internal sealed class KeyPolicy
{
    private readonly SortedDictionary<string, SortedSet<string>> _allowed = new SortedDictionary<string, SortedSet<string>>(StringComparer.Ordinal);

    internal void Allow(string key, string measurement)
    {
        if (!_allowed.TryGetValue(key, out SortedSet<string>? measurements))
        {
            measurements = new SortedSet<string>(StringComparer.Ordinal);
            _allowed[key] = measurements;
        }
        measurements.Add(measurement);
    }

    internal bool IsAllowed(string key, string measurement) => _allowed.TryGetValue(key, out SortedSet<string>? measurements) && measurements.Contains(measurement);
}

internal sealed class Verdict
{
    internal bool Accepted;
    internal string Reason = "";
    internal bool ChainOk;
    internal bool MeasurementOk;
    internal bool NonceFresh;
    internal bool SignatureOk;

    internal static Verdict Reject(string reason) => new Verdict { Reason = reason };
}

internal sealed class RemoteVerifier
{
    internal readonly RsaKey VendorRoot;
    private readonly SortedSet<string> _usedNonces = new SortedSet<string>(StringComparer.Ordinal);
    internal readonly List<string> Log = new List<string>();

    internal RemoteVerifier(RsaKey vendorRoot) { VendorRoot = vendorRoot; }

    /** Issues the certificate that binds an enclave's public key to its measurement. */
    internal Certificate Certify(string measurement, string publicKeyId)
    {
        var certificate = new Certificate { Measurement = measurement, PublicKeyId = publicKeyId };
        certificate.Signature = VendorRoot.SignDigest(certificate.Digest()).ToByteArray(isUnsigned: true, isBigEndian: true);
        return certificate;
    }

    internal Verdict Verify(Quote quote, Certificate certificate, string expectedMeasurement, string nonce, bool acceptNonceReuse)
    {
        var verdict = new Verdict();
        verdict.ChainOk = VendorRoot.VerifyWithOracle(certificate.Digest(), Hashes.ToInteger(certificate.Signature));
        if (!verdict.ChainOk) return Verdict.Reject("the certificate does not chain to the vendor root");
        if (certificate.Measurement != quote.Measurement || certificate.PublicKeyId != quote.PublicKeyId)
            return Verdict.Reject("the certificate does not bind this measurement and key");
        verdict.MeasurementOk = quote.Measurement == expectedMeasurement;
        if (!verdict.MeasurementOk) return Verdict.Reject("the enclave measurement is not the expected one");
        verdict.NonceFresh = quote.Nonce == nonce && (acceptNonceReuse || !_usedNonces.Contains(nonce));
        if (!verdict.NonceFresh) return Verdict.Reject("the quote nonce is stale or does not match the challenge");
        BigInteger digest = Hashes.ToInteger(Hashes.Sha256("quote|" + quote.Measurement + "|" + quote.PublicKeyId + "|" + quote.Nonce));
        // The quote is verified with the enclave key that the certificate vouches for, using the independent modexp.
        verdict.SignatureOk = VerifyWithEnclaveKey(digest, quote, certificate);
        if (!verdict.SignatureOk) return Verdict.Reject("the quote signature does not verify");
        if (!acceptNonceReuse) _usedNonces.Add(nonce);
        verdict.Accepted = true;
        verdict.Reason = "accepted";
        Log.Add("verify measurement=" + quote.Measurement.Substring(0, 8) + " nonce=" + nonce + " accepted");
        return verdict;
    }

    /** The verifier only has the public modulus, which it learns from the certificate/public identity. */
    private readonly SortedDictionary<string, RsaKey> _registered = new SortedDictionary<string, RsaKey>(StringComparer.Ordinal);

    internal void Register(RsaKey key) => _registered[key.PublicId] = key;

    private bool VerifyWithEnclaveKey(BigInteger digest, Quote quote, Certificate certificate)
    {
        if (!_registered.TryGetValue(quote.PublicKeyId, out RsaKey? key)) return false;
        if (key.PublicId != certificate.PublicKeyId) return false;
        return key.VerifyWithOracle(digest, Hashes.ToInteger(quote.Signature));
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. RSA: deterministic keys, real sign/verify, an independent oracle, and tamper detection.
        RsaKey vendorRoot = RsaKey.Generate(0x5EEDUL, 512);
        BigInteger digest = Hashes.ToInteger(Hashes.Sha256("payload"));
        BigInteger signature = vendorRoot.SignDigest(digest);
        Check(vendorRoot.VerifyDigest(digest, signature), "an RSA signature verifies with the public key");
        Check(vendorRoot.VerifyWithOracle(digest, signature), "an independently written modexp agrees with the verification");
        Check(RsaKey.IsProbablePrime(BigInteger.Parse("170141183460469231731687303715884105727")), "Miller-Rabin accepts a known Mersenne prime");
        Check(!RsaKey.IsProbablePrime(BigInteger.Parse("170141183460469231731687303715884105725")), "Miller-Rabin rejects a composite neighbour");
        BigInteger other = Hashes.ToInteger(Hashes.Sha256("payload2"));
        Check(!vendorRoot.VerifyDigest(other, signature), "a signature does not verify for a different message");
        Check(vendorRoot.N.GetBitLength() == 512, "the modulus has the expected size in bits");

        // 4-5. Attestation: a measurement-bound certificate and a nonce-bound quote.
        byte[] master = Hashes.Sha256("master-seed");
        var enclave = new Enclave("store-v1", "libkv", master);
        var verifier = new RemoteVerifier(vendorRoot);
        verifier.Register(enclave.Key);
        Certificate certificate = verifier.Certify(enclave.Measurement, enclave.Key.PublicId);
        Quote quote = enclave.Attest("nonce-1");
        Verdict verdict = verifier.Verify(quote, certificate, enclave.Measurement, "nonce-1", acceptNonceReuse: false);
        Check(verdict.Accepted, "a genuine enclave's quote is accepted (" + verdict.Reason + ")");
        Check(verdict.ChainOk && verdict.MeasurementOk && verdict.NonceFresh && verdict.SignatureOk,
            "every link of the attestation chain was checked");

        // 6-8. The verifier rejects a wrong measurement, a replayed nonce and a mismatched challenge.
        var impostor = new Enclave("store-v1-tampered", "libkv", Hashes.Sha256("other-master"));
        verifier.Register(impostor.Key);
        Verdict measurement = verifier.Verify(impostor.Attest("nonce-2"), verifier.Certify(impostor.Measurement, impostor.Key.PublicId),
            enclave.Measurement, "nonce-2", acceptNonceReuse: false);
        Check(!measurement.Accepted, "a quote with the wrong measurement is rejected");
        Check(measurement.Reason.Contains("measurement"), "the rejection names the measurement mismatch (" + measurement.Reason + ")");
        Verdict replay = verifier.Verify(quote, certificate, enclave.Measurement, "nonce-1", acceptNonceReuse: false);
        Check(!replay.Accepted, "a replayed nonce is rejected");
        Verdict mismatched = verifier.Verify(enclave.Attest("nonce-3"), certificate, enclave.Measurement, "nonce-4", acceptNonceReuse: false);
        Check(!mismatched.Accepted, "a quote answering a different challenge is rejected");
        Verdict selfSigned = verifier.Verify(enclave.Attest("nonce-5"),
            verifier.Certify(enclave.Measurement, impostor.Key.PublicId), enclave.Measurement, "nonce-5", acceptNonceReuse: false);
        Check(!selfSigned.Accepted, "a certificate that does not bind the quoting key is rejected");

        // 9-10. Brute-force bit-flip sweep: every tampered quote must fail verification.
        int mutated = 0;
        int accepted = 0;
        for (int index = 0; index < quote.Signature.Length; index += Math.Max(1, quote.Signature.Length / 24))
        {
            for (int bit = 0; bit < 8; bit++)
            {
                Quote tamperedQuote = quote.Clone();
                tamperedQuote.Signature[index] ^= (byte)(1 << bit);
                Verdict tamperedVerdict = verifier.Verify(tamperedQuote, certificate, enclave.Measurement, "nonce-1", acceptNonceReuse: true);
                mutated++;
                if (tamperedVerdict.Accepted) accepted++;
            }
        }
        Check(mutated >= 64, "the sweep flipped a bit in every sampled signature byte (" + mutated + " variants)");
        Check(accepted == 0, "every bit-flipped quote was rejected");
        Quote mutatedQuote = quote.Clone();
        mutatedQuote.Signature[0] ^= 0x01;
        Check(Hashes.Hex(mutatedQuote.Encode()) != Hashes.Hex(quote.Encode()),
            "a mutated quote also changes on the wire (the encoding covers the signature)");
        Quote badMeasurement = quote.Clone();
        badMeasurement.Measurement = enclave.Measurement.Substring(0, 63) + "0";
        Check(!verifier.Verify(badMeasurement, certificate, enclave.Measurement, "nonce-1", acceptNonceReuse: true).Accepted,
            "a flipped measurement hash is rejected");

        // 11-13. Sealed storage: round-trip, tamper detection, and a foreign enclave that cannot unseal.
        var module = new TrustedModule();
        SealedBlob blob = enclave.Seal("account", "balance=100", "seal-nonce-1");
        module.Advance(blob.Counter);
        Check(enclave.Unseal(blob, module.MonotonicCounter) == "balance=100", "the sealed record round-trips inside its enclave");
        Check(!Encoding.UTF8.GetString(blob.Ciphertext).Contains("balance"), "the ciphertext does not contain the plaintext");
        Check(enclave.Records.ContainsKey("account") && enclave.Records["account"].Length == blob.Ciphertext.Length,
            "the enclave keeps only the sealed bytes for the key");
        SealedBlob tamperedBlob = new SealedBlob
        {
            Counter = blob.Counter, Nonce = blob.Nonce, Ciphertext = (byte[])blob.Ciphertext.Clone(), Mac = blob.Mac
        };
        tamperedBlob.Ciphertext[0] ^= 0x01;
        Check(Throws(() => enclave.Unseal(tamperedBlob, module.MonotonicCounter)), "a flipped ciphertext bit breaks the MAC");
        SealedBlob tamperedMac = new SealedBlob
        {
            Counter = blob.Counter, Nonce = blob.Nonce, Ciphertext = blob.Ciphertext, Mac = (byte[])blob.Mac.Clone()
        };
        tamperedMac.Mac[0] ^= 0x01;
        Check(Throws(() => enclave.Unseal(tamperedMac, module.MonotonicCounter)), "a flipped MAC bit is rejected");
        var foreignEnclave = new Enclave("store-v2", "libkv", master);
        Check(foreignEnclave.Measurement != enclave.Measurement, "a different code and data produce a different measurement");
        Check(Throws(() => foreignEnclave.Unseal(blob, blob.Counter)), "a foreign enclave cannot unseal the record (measurement-bound key)");

        // 14-15. Rollback protection: an older record is refused by the monotonic counter.
        SealedBlob second = enclave.Seal("account", "balance=250", "seal-nonce-2");
        module.Advance(second.Counter);
        Check(second.Counter > blob.Counter, "each seal advances the record counter");
        Check(enclave.Unseal(second, module.MonotonicCounter) == "balance=250", "the newest record unseals");
        Check(Throws(() => enclave.Unseal(blob, module.MonotonicCounter)), "an older record is refused as a rollback");
        SealedBlob relabelled = new SealedBlob
        {
            Counter = module.MonotonicCounter, Nonce = blob.Nonce, Ciphertext = blob.Ciphertext, Mac = blob.Mac
        };
        Check(Throws(() => enclave.Unseal(relabelled, module.MonotonicCounter)), "re-labelling an old record with the current counter breaks the MAC");

        // 16-17. Access control: the policy must attest the enclave for the key, and the audit trail records it.
        var policy = new KeyPolicy();
        policy.Allow("account", enclave.Measurement);
        Check(policy.IsAllowed("account", enclave.Measurement), "the attested enclave is allowed to open the account key");
        Check(!policy.IsAllowed("account", foreignEnclave.Measurement), "an enclave with a different measurement is denied");
        Check(!policy.IsAllowed("other-key", enclave.Measurement), "a key that was never granted is denied");
        int auditBefore = enclave.Audit.Count;
        enclave.Seal("account", "balance=300", "seal-nonce-3");
        Check(enclave.Audit.Count > auditBefore && enclave.Audit[enclave.Audit.Count - 1].StartsWith("seal", StringComparison.Ordinal),
            "the enclave records every sealing operation in its audit log");

        // 18-19. Boundary cases: empty and long values, and a nonce that changes the keystream.
        SealedBlob empty = enclave.Seal("empty", "", "seal-nonce-4");
        module.Advance(empty.Counter);
        Check(enclave.Unseal(empty, module.MonotonicCounter) == "" && empty.Ciphertext.Length == 0,
            "an empty value seals and unseals as empty");
        var longValue = new StringBuilder();
        for (int i = 0; i < 200; i++) longValue.Append((char)('a' + i % 26));
        SealedBlob large = enclave.Seal("large", longValue.ToString(), "seal-nonce-5");
        module.Advance(large.Counter);
        Check(enclave.Unseal(large, module.MonotonicCounter) == longValue.ToString(), "a 200-byte value round-trips");
        Check(large.Ciphertext.Length == 200, "the ciphertext length matches the plaintext length");
        SealedBlob renamed = new SealedBlob { Counter = large.Counter, Nonce = "seal-nonce-6", Ciphertext = large.Ciphertext, Mac = large.Mac };
        Check(Throws(() => enclave.Unseal(renamed, module.MonotonicCounter)), "changing the nonce invalidates the MAC");

        // 20-21. Determinism: identical code, data and master secret give identical keys and quotes.
        var repeat = new Enclave("store-v1", "libkv", master);
        Check(repeat.Measurement == enclave.Measurement && repeat.Key.PublicId == enclave.Key.PublicId,
            "identical enclaves derive identical measurements and keys");
        Check(Hashes.Hex(repeat.Attest("nonce-1").Signature) == Hashes.Hex(quote.Signature), "identical attestations produce identical quotes");
        var differentSeed = RsaKey.Generate(0x1234UL, 512);
        Check(differentSeed.PublicId != vendorRoot.PublicId && differentSeed.PublicId.Length == vendorRoot.PublicId.Length,
            "a different seed produces a different (independent) root key of the same size");
        Check(RsaKey.Generate(0x5EEDUL, 512).PublicId == vendorRoot.PublicId, "the same seed reproduces the same key deterministically");
        Check(vendorRoot.PublicId != enclave.Key.PublicId, "the enclave key is independent of the vendor root");

        Console.WriteLine("  enclave: measurement=" + enclave.Measurement.Substring(0, 16) + " quotes=" + enclave.QuotesIssued
            + " seals=" + enclave.SealCounter + " swept=" + mutated + " audit entries=" + enclave.Audit.Count);
        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
