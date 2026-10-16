// Task 841 - Implement secure enclaves chain-of-trust verification demo. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// The remote-attestation chain a relying party walks before it will hand a secret to an enclave, modelled end to
// end: a root authority signs a platform certificate that binds a platform public key to an enclave measurement
// policy (the measurement whitelist with the minimum security version each measurement must report); the platform
// key signs a quote over the measurement, a verifier-chosen nonce, the security version, the attributes and the
// epoch; the relying party verifies the chain, checks freshness against the nonce it just issued, checks the
// policy and checks its own revocation lists. The epoch is an explicit parameter everywhere — nothing here reads a
// clock — and the verification order is fixed so that every failure maps to one precise reason code.
//
// Verified, each by recomputation:
//   * the signature scheme: key generation is reproducible, signing and verification round-trip, a tampered quote
//     field or a signature made with another key is rejected, and the *unpadded* variant is shown to be forged by
//     multiplying two signatures (s1*s2 verifies the product of the two hashes) before the hashed variant is
//     adopted — measured, not asserted;
//   * the chain: a well-formed chain is accepted and yields the right identity and measurement, and every attack
//     in the battery is rejected with its own reason code — re-signed but unknown measurement, re-signed but
//     rolled-back security version, replayed nonce, revoked platform key, revoked measurement, expired or
//     not-yet-valid certificate, and a certificate minted by an untrusted key;
//   * reason coverage: the set of verdicts observed by the battery equals the full set the verifier can return, so
//     no branch is dead code;
//   * boundaries: sec version exactly at the policy minimum is accepted and one below is not; epoch exactly at the
//     certificate's notAfter is accepted and one past it is not;
//   * cost and liveness at scale: 2 modular exponentiations per quote, reduced to N+1 for N quotes from one
//     platform once the certificate is cached, and a batch of 200 quotes is fully classified with no retries.
//
// Documented simplifications (toy scale, no real key material): the RSA moduli are 64 bits, factorable in
// milliseconds, so the security measured here is that of the *protocol logic*, not of the ciphers — the file even
// factorises its own moduli to show it; a real deployment uses 2048-bit moduli, PSS-style padding and a hardware
// root of trust instead of a hard-coded root key. The hash is a 64-bit splitmix mixing of the canonical quote
// encoding; timestamps are counter-style epochs supplied by the caller. Randomness is hand-written splitmix64 with
// explicit seeds, so every verdict in the output is reproducible.
using System;
using System.Collections.Generic;
using System.Numerics;

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

internal static class Hashing
{
    /// <summary>64-bit digest of a canonical string encoding: the full-domain hash of the toy signature scheme.</summary>
    internal static BigInteger Digest(string canonical)
    {
        ulong state = 0xCBF29CE484222325UL;
        foreach (char c in canonical)
        {
            state ^= (ulong)(c + 1);
            state = SplitMix64.Next(ref state);
        }
        state = SplitMix64.Next(ref state);
        return new BigInteger(state);
    }
}

/// <summary>A toy RSA key pair. 64-bit moduli only: this models the protocol, never a real cipher.</summary>
internal sealed class RsaKey
{
    internal BigInteger N { get; }
    internal BigInteger E { get; }
    internal BigInteger D { get; }
    internal string Id { get; }

    internal RsaKey(BigInteger p, BigInteger q, BigInteger e, string id)
    {
        N = p * q;
        E = e;
        D = Rsa.ModInverse(e, (p - 1) * (q - 1));
        Id = id;
    }

    internal int ModulusBits => (int)N.GetBitLength();
}

internal static class Rsa
{
    internal const long PublicExponent = 65537;

    /// <summary>Extended Euclid; both moduli used here are products of two odd primes, so the inverse exists.</summary>
    internal static BigInteger ModInverse(BigInteger a, BigInteger m)
    {
        BigInteger oldR = a % m, r = m, oldS = BigInteger.One, s = BigInteger.Zero;
        while (r != 0)
        {
            BigInteger quotient = oldR / r;
            (oldR, r) = (r, oldR - quotient * r);
            (oldS, s) = (s, oldS - quotient * s);
        }
        if (oldR != 1) throw new ArgumentException("the exponent is not invertible modulo phi");
        return oldS < 0 ? oldS + m : oldS;
    }

    internal static BigInteger Sign(RsaKey key, BigInteger digest)
    {
        Verifications.Exponentiations++;
        return BigInteger.ModPow(digest, key.D, key.N);
    }

    internal static bool Verify(RsaKey key, BigInteger digest, BigInteger signature)
    {
        Verifications.Exponentiations++;
        if (signature < 0 || signature >= key.N) return false;
        return BigInteger.ModPow(signature, key.E, key.N) == digest % key.N;
    }

    /// <summary>Naive factorisation, only to report how little the toy modulus protects.</summary>
    internal static (BigInteger P, BigInteger Q) Factor(BigInteger n)
    {
        for (BigInteger f = 3; f * f <= n; f += 2)
        {
            if (n % f == 0) return (f, n / f);
        }
        return (BigInteger.Zero, BigInteger.Zero);
    }
}

/// <summary>Counter of modular exponentiations, so the verification cost is measured rather than claimed.</summary>
internal static class Verifications
{
    internal static long Exponentiations;
}

internal sealed class MeasurementPolicy
{
    internal string Measurement { get; }
    internal int MinimumSvn { get; }

    internal MeasurementPolicy(string measurement, int minimumSvn)
    {
        Measurement = measurement;
        MinimumSvn = minimumSvn;
    }
}

/// <summary>The platform certificate: a platform public key plus the policy the root authority vouches for.</summary>
internal sealed class PlatformCertificate
{
    internal string Serial { get; }
    internal string PlatformId { get; }
    internal RsaKey PlatformKey { get; }
    internal List<MeasurementPolicy> Policies { get; }
    internal long NotBefore { get; }
    internal long NotAfter { get; }
    internal BigInteger Signature { get; private set; }

    internal PlatformCertificate(string serial, string platformId, RsaKey platformKey,
        List<MeasurementPolicy> policies, long notBefore, long notAfter)
    {
        Serial = serial;
        PlatformId = platformId;
        PlatformKey = platformKey;
        Policies = policies;
        NotBefore = notBefore;
        NotAfter = notAfter;
        Signature = BigInteger.Zero;
    }

    /// <summary>Canonical encoding: the exact byte string the root authority signs.</summary>
    internal string Canonical()
    {
        var parts = new List<string> { "cert", Serial, PlatformId, PlatformKey.Id, PlatformKey.N.ToString(),
            NotBefore.ToString(), NotAfter.ToString() };
        foreach (MeasurementPolicy policy in Policies) parts.Add(policy.Measurement + ":" + policy.MinimumSvn);
        return string.Join("|", parts);
    }

    internal BigInteger Digest() => Hashing.Digest(Canonical());

    internal void SignWith(RsaKey authority) => Signature = Rsa.Sign(authority, Digest());

    internal MeasurementPolicy? PolicyFor(string measurement)
    {
        foreach (MeasurementPolicy policy in Policies) if (policy.Measurement == measurement) return policy;
        return null;
    }
}

/// <summary>The enclave quote: what the platform asserts about the enclave, bound to a verifier nonce.</summary>
internal sealed class Quote
{
    internal string Measurement { get; }
    internal string Nonce { get; }
    internal int Svn { get; }
    internal string Attributes { get; }
    internal long Epoch { get; }
    internal string CertSerial { get; }
    internal BigInteger Signature { get; private set; }

    internal Quote(string measurement, string nonce, int svn, string attributes, long epoch, string certSerial)
    {
        Measurement = measurement;
        Nonce = nonce;
        Svn = svn;
        Attributes = attributes;
        Epoch = epoch;
        CertSerial = certSerial;
        Signature = BigInteger.Zero;
    }

    internal string Canonical()
    {
        return string.Join("|", "quote", Measurement, Nonce, Svn.ToString(), Attributes, Epoch.ToString(), CertSerial);
    }

    internal BigInteger Digest() => Hashing.Digest(Canonical());

    internal void SignWith(RsaKey platformKey) => Signature = Rsa.Sign(platformKey, Digest());

    /// <summary>An attacker's edit: the fields change, the signature does not — which is what the chain must catch.</summary>
    internal Quote Mutate(string measurement, string nonce, int svn, string attributes, long epoch, string certSerial)
    {
        var mutated = new Quote(measurement, nonce, svn, attributes, epoch, certSerial);
        mutated.Signature = Signature;
        return mutated;
    }
}

internal enum Verdict
{
    Accepted,
    CertNotYetValid,
    CertExpired,
    CertSignatureInvalid,
    CertRevoked,
    QuoteSignatureInvalid,
    StaleNonce,
    UnknownMeasurement,
    RevokedMeasurement,
    SvlBelowMinimum,
}

internal sealed class TrustStore
{
    internal RsaKey Root { get; }
    internal HashSet<string> RevokedSerials { get; } = new HashSet<string>();
    internal HashSet<string> RevokedMeasurements { get; } = new HashSet<string>();
    internal long Epoch { get; }

    internal TrustStore(RsaKey root, long epoch)
    {
        Root = root;
        Epoch = epoch;
    }
}

internal sealed class VerificationResult
{
    internal Verdict Verdict { get; }
    internal string Detail { get; }

    internal VerificationResult(Verdict verdict, string detail)
    {
        Verdict = verdict;
        Detail = detail;
    }
}

/// <summary>
/// The relying party. Verification order is fixed: validity window, chain signature, revocation, quote signature,
/// freshness, measurement policy, security version. The certificate cache is what makes verification cheap when
/// many quotes come from the same platform, exactly as a real attestation service would.
/// </summary>
internal sealed class Verifier
{
    private readonly Dictionary<string, PlatformCertificate> _certificateCache = new Dictionary<string, PlatformCertificate>();
    internal long CertificatesFetched { get; private set; }

    internal VerificationResult Verify(Quote quote, PlatformCertificate certificate, TrustStore store, string expectedNonce)
    {
        // The chain is fetched over the wire only when the certificate has not been seen and validated yet; a cached
        // certificate skips the root signature check but never the revocation and policy checks.
        PlatformCertificate cert;
        bool cachedHit = _certificateCache.TryGetValue(certificate.Serial, out PlatformCertificate? cached)
            && cached.Canonical() == certificate.Canonical();
        if (cachedHit && cached is not null)
        {
            cert = cached;
        }
        else
        {
            CertificatesFetched++;
            cert = certificate;
        }

        if (store.Epoch < cert.NotBefore) return new VerificationResult(Verdict.CertNotYetValid, "epoch before notBefore");
        if (store.Epoch > cert.NotAfter) return new VerificationResult(Verdict.CertExpired, "epoch past notAfter");
        if (!cachedHit && !Rsa.Verify(store.Root, cert.Digest(), cert.Signature))
            return new VerificationResult(Verdict.CertSignatureInvalid, "certificate not signed by the trusted root");
        if (store.RevokedSerials.Contains(cert.Serial))
            return new VerificationResult(Verdict.CertRevoked, "platform certificate serial is revoked");
        if (!Rsa.Verify(cert.PlatformKey, quote.Digest(), quote.Signature))
            return new VerificationResult(Verdict.QuoteSignatureInvalid, "quote not signed by the certified platform key");
        if (quote.Nonce != expectedNonce)
            return new VerificationResult(Verdict.StaleNonce, "quote nonce does not match the challenge");
        MeasurementPolicy? policy = cert.PolicyFor(quote.Measurement);
        if (policy is null) return new VerificationResult(Verdict.UnknownMeasurement, "measurement not in the policy");
        if (store.RevokedMeasurements.Contains(quote.Measurement))
            return new VerificationResult(Verdict.RevokedMeasurement, "measurement is revoked");
        if (quote.Svn < policy.MinimumSvn)
            return new VerificationResult(Verdict.SvlBelowMinimum, "security version below policy minimum");

        _certificateCache[cert.Serial] = cert;
        return new VerificationResult(Verdict.Accepted, cert.PlatformId + ":" + quote.Measurement + ":svn" + quote.Svn);
    }

    /// <summary>Batch verification, used to measure the cost of a large enrolment wave.</summary>
    internal Dictionary<Verdict, int> VerifyBatch(IReadOnlyList<(Quote Quote, PlatformCertificate Cert)> batch,
        TrustStore store, Func<int, string> nonceFor)
    {
        var histogram = new Dictionary<Verdict, int>();
        for (int i = 0; i < batch.Count; i++)
        {
            VerificationResult result = Verify(batch[i].Quote, batch[i].Cert, store, nonceFor(i));
            histogram.TryGetValue(result.Verdict, out int seen);
            histogram[result.Verdict] = seen + 1;
        }
        return histogram;
    }
}

internal static class Enclave
{
    // Fixed 32-bit prime pairs: the moduli are 64 bits and are factorable in this process, which the file prints.
    internal static readonly BigInteger[] Primes =
    {
        4294967291, 4294967279, 4294967231, 4294967197, 4294967189, 4294967161,
        4294967143, 4294967111, 4294967087, 4294967029, 4294966997, 4294966981,
    };

    internal static RsaKey KeyAt(int index, string id) => new RsaKey(Primes[index], Primes[index + 1], Rsa.PublicExponent, id);

    internal static string Measurement(int seed) => "mrenclave-" + (0x1000 + seed * 7919).ToString("X8").ToLowerInvariant();
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static PlatformCertificate MakeCertificate(RsaKey root, RsaKey platform, string serial, long notBefore, long notAfter)
    {
        var policies = new List<MeasurementPolicy>
        {
            new MeasurementPolicy(Enclave.Measurement(0), 3),
            new MeasurementPolicy(Enclave.Measurement(1), 1),
            new MeasurementPolicy(Enclave.Measurement(2), 5),
        };
        var cert = new PlatformCertificate(serial, "platform-" + serial, platform, policies, notBefore, notAfter);
        cert.SignWith(root);
        return cert;
    }

    private static Quote MakeQuote(string measurement, string nonce, int svn, long epoch, string serial, string attributes = "debug=0")
    {
        return new Quote(measurement, nonce, svn, attributes, epoch, serial);
    }

    public static void Main()
    {
        RsaKey root = Enclave.KeyAt(0, "root-ca");
        RsaKey platform = Enclave.KeyAt(2, "platform-key");
        RsaKey rogue = Enclave.KeyAt(4, "rogue-key");
        RsaKey enclave = Enclave.KeyAt(6, "enclave-key");
        Check(root.ModulusBits == 64 && platform.ModulusBits == 64,
            "the toy moduli are 64 bits (" + root.N + "), which is the scale this demo commits to");

        // ---- the signature scheme --------------------------------------------------------------------
        {
            RsaKey again = new RsaKey(Enclave.Primes[0], Enclave.Primes[1], Rsa.PublicExponent, "root-ca");
            Check(again.N == root.N && again.D == root.D, "key generation is reproducible from the same primes");
            BigInteger digest = Hashing.Digest("hello-attestation");
            BigInteger signature = Rsa.Sign(root, digest);
            Check(Rsa.Verify(root, digest, signature), "a signature verifies under its own key");
            Check(!Rsa.Verify(root, Hashing.Digest("hello-attestatio"), signature), "a changed digest fails verification");
            Check(!Rsa.Verify(platform, digest, signature), "a signature does not verify under another key");
            Check(!Rsa.Verify(root, digest, signature + 1), "a tampered signature fails verification");

            // The unpadded variant is multiplicatively forgeable: s1*s2 is a valid signature on h1*h2.
            BigInteger h1 = 3, h2 = 5;
            BigInteger forged = Rsa.Sign(root, h1) * Rsa.Sign(root, h2) % root.N;
            Check(Rsa.Verify(root, h1 * h2, forged), "unpadded signing is forged by multiplying two signatures: the "
                + "product of two signatures verifies the product of the two digests");
            BigInteger h3 = Hashing.Digest("victim-message");
            Check(!Rsa.Verify(root, h1 * h2, Rsa.Sign(root, h3)), "an unrelated signature does not pass the same check");
            (BigInteger p, BigInteger q) = Rsa.Factor(root.N);
            Check(p * q == root.N && p > 1, "the 64-bit modulus factorises in-process (" + p + " x " + q
                + "), which is why the ciphers here are toy scale");
            Console.WriteLine("signatures: round trip ok, unpadded product forgery demonstrated, modulus factored in "
                + "process");
        }

        // ---- the happy path --------------------------------------------------------------------------
        {
            var store = new TrustStore(root, 100);
            PlatformCertificate cert = MakeCertificate(root, platform, "A1", 90, 200);
            Quote quote = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "A1");
            quote.SignWith(platform);
            var verifier = new Verifier();
            VerificationResult result = verifier.Verify(quote, cert, store, "nonce-1");
            Check(result.Verdict == Verdict.Accepted, "a well-formed chain is accepted (" + result.Verdict + ": "
                + result.Detail + ")");
            Check(result.Detail == "platform-A1:" + Enclave.Measurement(0) + ":svn3", "the verifier reports the "
                + "identity, measurement and security version it accepted");
            Console.WriteLine("happy path: " + result.Detail);
        }

        // ---- every attack, with its own reason code ---------------------------------------------------
        {
            var store = new TrustStore(root, 100);
            PlatformCertificate cert = MakeCertificate(root, platform, "A1", 90, 200);
            var verifier = new Verifier();
            var observed = new Dictionary<Verdict, string>();

            void Expect(string name, Quote quote, PlatformCertificate certificate, TrustStore against, string nonce, Verdict expected)
            {
                VerificationResult result = verifier.Verify(quote, certificate, against, nonce);
                observed[result.Verdict] = name;
                Check(result.Verdict == expected, name + " is rejected as " + expected + " (got " + result.Verdict
                    + ": " + result.Detail + ")");
            }

            Quote good = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "A1");
            good.SignWith(platform);

            // Tampering after signing: the fields change, the signature does not, so the chain must catch it.
            Quote editedMeasurement = good.Mutate(Enclave.Measurement(1), "nonce-1", 3, "debug=0", 100, "A1");
            Quote movedEpoch = good.Mutate(Enclave.Measurement(0), "nonce-1", 3, "debug=0", 101, "A1");
            Quote changedAttributes = good.Mutate(Enclave.Measurement(0), "nonce-1", 3, "debug=1", 100, "A1");
            Quote wrongSerial = good.Mutate(Enclave.Measurement(0), "nonce-1", 3, "debug=0", 100, "B9");
            Expect("a quote whose measurement was edited after signing", editedMeasurement, cert, store, "nonce-1",
                Verdict.QuoteSignatureInvalid);
            Expect("a quote whose epoch was edited", movedEpoch, cert, store, "nonce-1", Verdict.QuoteSignatureInvalid);
            Expect("a quote whose attributes were edited", changedAttributes, cert, store, "nonce-1",
                Verdict.QuoteSignatureInvalid);
            Expect("a quote naming another certificate", wrongSerial, cert, store, "nonce-1", Verdict.QuoteSignatureInvalid);

            // A malicious but enrolled platform: it can sign whatever it likes, the policy is what stops it.
            Quote unknown = MakeQuote("mrenclave-deadbeef", "nonce-1", 9, 100, "A1");
            unknown.SignWith(platform);
            Expect("a re-signed quote with an unknown measurement", unknown, cert, store, "nonce-1",
                Verdict.UnknownMeasurement);

            Quote oldSvn = MakeQuote(Enclave.Measurement(0), "nonce-1", 2, 100, "A1");
            oldSvn.SignWith(platform);
            Expect("a re-signed quote with a rolled-back security version", oldSvn, cert, store, "nonce-1",
                Verdict.SvlBelowMinimum);

            Expect("a replayed quote from an earlier challenge", good, cert, store, "nonce-2", Verdict.StaleNonce);

            Quote revokedMeasurement = MakeQuote(Enclave.Measurement(1), "nonce-1", 1, 100, "A1");
            revokedMeasurement.SignWith(platform);
            var revokedStore = new TrustStore(root, 100);
            revokedStore.RevokedMeasurements.Add(Enclave.Measurement(1));
            Expect("a quote for a revoked measurement", revokedMeasurement, cert, revokedStore, "nonce-1",
                Verdict.RevokedMeasurement);

            var keyRevokedStore = new TrustStore(root, 100);
            keyRevokedStore.RevokedSerials.Add("A1");
            Expect("a quote from a revoked platform certificate", good, cert, keyRevokedStore, "nonce-1", Verdict.CertRevoked);

            PlatformCertificate expired = MakeCertificate(root, platform, "A2", 90, 99);
            Quote live = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "A2");
            live.SignWith(platform);
            Expect("a quote on an expired certificate", live, expired, store, "nonce-1", Verdict.CertExpired);

            PlatformCertificate future = MakeCertificate(root, platform, "A3", 101, 200);
            Quote early = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "A3");
            early.SignWith(platform);
            Expect("a quote on a not-yet-valid certificate", early, future, store, "nonce-1", Verdict.CertNotYetValid);

            PlatformCertificate forged = MakeCertificate(rogue, platform, "A4", 90, 200);
            Quote onForged = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "A4");
            onForged.SignWith(platform);
            Expect("a certificate minted by an untrusted key", onForged, forged, store, "nonce-1",
                Verdict.CertSignatureInvalid);

            Quote fromRogue = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "A1");
            fromRogue.SignWith(enclave);
            Expect("a quote signed by an uncertified key", fromRogue, cert, store, "nonce-1", Verdict.QuoteSignatureInvalid);

            var expectedReasons = new List<Verdict>
            {
                Verdict.CertNotYetValid, Verdict.CertExpired, Verdict.CertSignatureInvalid, Verdict.CertRevoked,
                Verdict.QuoteSignatureInvalid, Verdict.StaleNonce, Verdict.UnknownMeasurement, Verdict.RevokedMeasurement,
                Verdict.SvlBelowMinimum,
            };
            bool covered = observed.Count == expectedReasons.Count;
            foreach (Verdict verdict in expectedReasons) if (!observed.ContainsKey(verdict)) covered = false;
            Check(covered, "the battery exercised every rejection reason the verifier can return (" + observed.Count
                + " distinct codes), so no branch is dead code");
            Console.WriteLine("attack battery: " + observed.Count + " rejection codes exercised, all with the expected "
                + "reason");
        }

        // ---- policy and validity boundaries ----------------------------------------------------------
        {
            var store = new TrustStore(root, 100);
            PlatformCertificate cert = MakeCertificate(root, platform, "B1", 100, 100);   // inclusive window
            var verifier = new Verifier();
            Quote atMinimum = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "B1");
            atMinimum.SignWith(platform);
            Check(verifier.Verify(atMinimum, cert, store, "nonce-1").Verdict == Verdict.Accepted,
                "a security version exactly at the policy minimum is accepted, and an epoch exactly at notAfter is "
                + "inside the validity window");
            Quote belowMinimum = MakeQuote(Enclave.Measurement(0), "nonce-1", 2, 100, "B1");
            belowMinimum.SignWith(platform);
            Check(verifier.Verify(belowMinimum, cert, store, "nonce-1").Verdict == Verdict.SvlBelowMinimum,
                "one below the minimum is rejected");
            Quote aboveMinimum = MakeQuote(Enclave.Measurement(0), "nonce-1", 4, 100, "B1");
            aboveMinimum.SignWith(platform);
            Check(verifier.Verify(aboveMinimum, cert, store, "nonce-1").Verdict == Verdict.Accepted,
                "a higher security version is accepted");

            var emptyPolicyCert = new PlatformCertificate("C1", "platform-C1", platform, new List<MeasurementPolicy>(), 90, 200);
            emptyPolicyCert.SignWith(root);
            Quote anyQuote = MakeQuote(Enclave.Measurement(0), "nonce-1", 9, 100, "C1");
            anyQuote.SignWith(platform);
            Check(verifier.Verify(anyQuote, emptyPolicyCert, store, "nonce-1").Verdict == Verdict.UnknownMeasurement,
                "an empty policy whitelist accepts no measurement at all");

            var lateStore = new TrustStore(root, 101);
            Check(verifier.Verify(atMinimum, cert, lateStore, "nonce-1").Verdict == Verdict.CertExpired,
                "one epoch past notAfter is rejected");
            Check(verifier.Verify(atMinimum, cert, lateStore, "nonce-2").Verdict == Verdict.CertExpired,
                "the validity window is checked before freshness");
            Console.WriteLine("boundaries: svn min/above accepted, below rejected; epoch at notAfter accepted, +1 "
                + "rejected; empty policy rejects all measurements");
        }

        // ---- cost and liveness at scale --------------------------------------------------------------
        {
            var store = new TrustStore(root, 100);
            PlatformCertificate cert = MakeCertificate(root, platform, "D1", 90, 200);
            var verifier = new Verifier();
            const int Quotes = 200;
            var batch = new List<(Quote, PlatformCertificate)>();
            int expectedAccepted = 0;
            for (int i = 0; i < Quotes; i++)
            {
                // Every fourth quote is a malicious-platform quote with a rolled-back security version.
                bool hostile = i % 4 == 3;
                // The hostile quotes re-sign a rolled-back security version for the measurement whose policy floor
                // is 3, so the policy check (not the signature) is what rejects them.
                Quote quote = MakeQuote(Enclave.Measurement(hostile ? 0 : i % 2), "nonce-" + i, hostile ? 1 : 3, 100, "D1");
                quote.SignWith(platform);
                batch.Add((quote, cert));
                if (!hostile) expectedAccepted++;
            }
            long before = Verifications.Exponentiations;
            Dictionary<Verdict, int> histogram = verifier.VerifyBatch(batch, store, i => "nonce-" + i);
            long spent = Verifications.Exponentiations - before;
            Check(histogram.TryGetValue(Verdict.Accepted, out int accepted) && accepted == expectedAccepted,
                "the batch accepts exactly the " + expectedAccepted + " well-formed quotes (" + accepted + ")");
            Check(histogram.TryGetValue(Verdict.SvlBelowMinimum, out int rejected) && rejected == Quotes - expectedAccepted,
                "and rejects the " + (Quotes - expectedAccepted) + " rolled-back quotes with the policy reason");
            Check(spent == Quotes + 1, "verification costs exactly one exponentiation per quote plus one for the "
                + "certificate after the cache is used (" + spent + " for " + Quotes + " quotes)");
            Check(verifier.CertificatesFetched == 1, "the certificate is fetched from the wire once and then reused ("
                + verifier.CertificatesFetched + " fetch)" );

            // No cache at all: every quote re-fetches and re-verifies the certificate chain.
            long beforeUncached = Verifications.Exponentiations;
            foreach ((Quote quote, PlatformCertificate certificate) in batch)
                new Verifier().Verify(quote, certificate, store, "nonce-" + 0);
            long spentUncached = Verifications.Exponentiations - beforeUncached;
            Check(spentUncached == 2L * Quotes, "a fresh verifier with no cache costs two exponentiations per quote ("
                + spentUncached + "), which is the saving the cache buys");
            Console.WriteLine("batch of " + Quotes + ": accepted=" + expectedAccepted + " policyRejected="
                + (Quotes - expectedAccepted) + " expCached=" + spent + " expUncached=" + spentUncached
                + " certFetches=" + verifier.CertificatesFetched);
        }

        // ---- determinism -----------------------------------------------------------------------------
        {
            var store = new TrustStore(root, 100);
            PlatformCertificate cert = MakeCertificate(root, platform, "E1", 90, 200);
            Quote first = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "E1");
            first.SignWith(platform);
            Quote second = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "E1");
            second.SignWith(platform);
            Check(first.Signature == second.Signature, "signing the same quote twice gives the same signature");
            Check(cert.Digest() == MakeCertificate(root, platform, "E1", 90, 200).Digest(),
                "the canonical certificate encoding is stable across constructions");
            var verifierA = new Verifier();
            var verifierB = new Verifier();
            Check(verifierA.Verify(first, cert, store, "nonce-1").Verdict
                == verifierB.Verify(second, cert, store, "nonce-1").Verdict, "verification is reproducible");
            Quote changed = MakeQuote(Enclave.Measurement(0), "nonce-1", 3, 100, "E1");
            changed.SignWith(rogue);
            Check(changed.Signature != first.Signature, "a different signing key gives a different signature");
            Console.WriteLine("determinism: signatures and verdicts reproducible across verifier instances");
        }

        Console.WriteLine("PASS");
    }
}
