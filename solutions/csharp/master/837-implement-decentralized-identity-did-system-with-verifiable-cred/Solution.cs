// Task 837 - Implement decentralized identity (DID) system with verifiable credentials simulation.
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// A self-sovereign identity stack, end to end: DIDs with their documents, an issuer that signs credentials, a
// holder that stores them, and a verifier that checks a *derived* presentation without learning anything the
// holder did not choose to disclose.
//
//   * did:example:<tag> is derived deterministically from a seed: the signing key is a field element and the
//     public key is g^sk mod p, published in a DID document by that identifier;
//   * signatures are Schnorr signatures over the 31-bit prime group - the real algebraic construction, verified
//     with the exact equation g^s * pk^e == r where e is the Fiat-Shamir challenge over the transcript. The file
//     states plainly that 31 bits of group benefit no security (the discrete log is brute-forceable), which is
//     why the toy only demonstrates the algebra and the flows;
//   * a verifiable credential is a set of claims, each salted and hashed, committed by a Merkle root. The issuer
//     signs (issuer DID, subject DID, root, validity window), so the signature covers the whole claim set while
//     revealing none of it;
//   * a presentation discloses exactly the claims the holder picks - their salts and Merkle paths - plus the
//     issuer's signature and a *fresh holder signature* over (root, verifier domain, verifier nonce). Selective
//     disclosure comes from the Merkle proofs; replay protection and unlinkability come from the nonce and the
//     domain being inside a signature that the holder computes per presentation.
//
// Verified invariants (each recomputed here, never assumed):
//   * the Schnorr equation holds exactly for honest signatures, and fails for a changed transcript, a changed
//     nonce, a changed public key or a shifted response; the construction is checked rather than trusted;
//   * DIDs are deterministic functions of their seeds, distinct seeds give distinct DIDs and distinct keys, and a
//     document's public key really is g^sk;
//   * an honest presentation verifies; a tampered claim value, a tampered salt, a tampered Merkle path, a root
//     that does not match the claims, an expired credential, a not-yet-valid one, a revoked one, an unknown
//     issuer and a presentation signed by the wrong holder are all rejected - each in its own check;
//   * selective disclosure is exact: the same credential discloses two of five claims, the verifier can check
//     those two and learns nothing about the other three (no salt and no value is in the presentation), while the
//     Merkle root - and therefore the issuer's signature - is unchanged by the choice;
//   * the holder's signature binds the domain and the nonce, so the identical presentation is rejected at another
//     verifier or with a stale nonce - the anti-replay property;
//   * degenerate inputs (no claims, duplicate claim keys, an unknown claim, a missing holder signature) are
//     rejected, and every run is bit-for-bit reproducible.
//
// Documented simplification and limits: the group is the 31-bit prime field 2^31-1 with a non-prime order, so the
// discrete logarithm is brute-forceable and NO cryptographic security is claimed - the file even measures how
// cheap that is on a smaller group; the hash is a mixing function, so the Fiat-Shamir challenge offers no
// collision resistance; there is no W3C JSON-LD canonicalisation, no DID resolution network, no key rotation, no
// revocation accumulator (a plain registry is used) and no zero-knowledge proof of the hidden claims (the
// commitment hides them only because the salts stay private). What is real: the Schnorr algebra, the salted-hash
// commitment with Merkle proofs, the selective-disclosure mechanics, the freshness binding and the verification
// policy checks.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static ulong FromString(string text, ulong salt)
    {
        ulong state = 0xCBF29CE484222325UL ^ salt;
        foreach (char c in text) state = Mix(state, c);
        return state;
    }

    internal static ulong Mix(ulong state, ulong value)
    {
        ulong z = state ^ (value + 0x9E3779B97F4A7C15UL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static ulong Combine(ulong a, ulong b) => Mix(a, b + 0x165667B19E3779F9UL);
}

/** The 31-bit prime group used for keys and signatures: enough to verify algebra, far too small for security. */
internal static class Group
{
    internal const long P = 2147483647L;      // 2^31 - 1
    internal const long Order = P - 1;        // the exponent modulus; the order of g is a divisor of this
    internal const long G = 7;                // generator used throughout

    internal static long Mul(long a, long b) => (a % P) * (b % P) % P;

    internal static long Pow(long a, long e)
    {
        long result = 1;
        long b = a % P;
        long exponent = ((e % Order) + Order) % Order;
        while (exponent > 0)
        {
            if ((exponent & 1L) != 0) result = Mul(result, b);
            b = Mul(b, b);
            exponent >>= 1;
        }
        return result;
    }

    internal static long Normalise(long a) => ((a % Order) + Order) % Order;

    internal static long HashToScalar(ulong hash)
    {
        long value = (long)(hash % (ulong)Order);
        return value == 0 ? 1 : value;
    }

    /** Power in a caller-chosen modulus: used only to show how cheap a discrete log is on a small group. */
    internal static long PowSmall(long a, long e, long modulus)
    {
        long result = 1;
        long b = a % modulus;
        while (e > 0)
        {
            if ((e & 1L) != 0) result = result * b % modulus;
            b = b * b % modulus;
            e >>= 1;
        }
        return result;
    }
}

internal sealed class KeyPair
{
    internal long Secret;
    internal long Public;

    internal static KeyPair FromSeed(ulong seed)
    {
        var rng = seed;
        long secret = Group.HashToScalar(SplitMix64.Next(ref rng));
        return new KeyPair { Secret = secret, Public = Group.Pow(Group.G, secret) };
    }
}

/** Schnorr signatures: s = k + e * sk, verified by g^s == r * pk^e. */
internal static class Schnorr
{
    internal static ulong Challenge(long r, ulong transcript)
    {
        ulong state = SplitMix64.Combine((ulong)r, transcript);
        return state;
    }

    internal static (long R, long S) Sign(KeyPair key, ulong transcript, ulong nonceSeed)
    {
        var rng = nonceSeed;
        long k = Group.HashToScalar(SplitMix64.Next(ref rng));
        long r = Group.Pow(Group.G, k);
        long e = Group.Normalise((long)(Challenge(r, transcript) % (ulong)Group.Order));
        long s = Group.Normalise(k + e * Group.Normalise(key.Secret));
        return (r, s);
    }

    internal static bool Verify(long publicKey, ulong transcript, long r, long s)
    {
        if (r <= 0 || r >= Group.P || s < 0 || s >= Group.Order) return false;
        long e = Group.Normalise((long)(Challenge(r, transcript) % (ulong)Group.Order));
        return Group.Pow(Group.G, s) == Group.Mul(r, Group.Pow(publicKey, e));
    }
}

internal sealed class DidDocument
{
    internal required string Id;
    internal required long PublicKey;
    internal required ulong SeedHash;
}

internal sealed class Claim
{
    internal required string Key;
    internal required string Value;
    internal required long Salt;
    internal long Hash;
}

internal sealed class Credential
{
    internal required string IssuerId;
    internal required string SubjectId;
    internal required List<Claim> Claims;
    internal required long Root;
    internal required int ValidFrom;
    internal required int ValidUntil;
    internal long SignatureR;
    internal long SignatureS;
    internal ulong Transcript;
}

internal sealed class Disclosure
{
    internal required string Key;
    internal required string Value;
    internal required long Salt;
    internal required List<long> Path;
    internal required int Index;
}

internal sealed class Presentation
{
    internal required string IssuerId;
    internal required string SubjectId;
    internal required long Root;
    internal required int ValidFrom;
    internal required int ValidUntil;
    internal required List<Disclosure> Disclosed;
    internal required long IssuerSignatureR;
    internal required long IssuerSignatureS;
    internal required long HolderSignatureR;
    internal required long HolderSignatureS;
    internal required int TotalClaims;
}

internal static class Merkle
{
    internal static long LeafHash(Claim claim) => (long)SplitMix64.Combine(
        (ulong)claim.Salt, SplitMix64.Combine(SplitMix64.FromString(claim.Key, 0x837UL), SplitMix64.FromString(claim.Value, 0x538UL)))
        % Group.P;

    internal static long[][] BuildLevels(IReadOnlyList<long> leaves, out long[] topLevel)
    {
        int padded = 1;
        while (padded < leaves.Count) padded *= 2;
        var level = new long[padded];
        for (int i = 0; i < leaves.Count; i++) level[i] = leaves[i] % Group.P;
        var levels = new List<long[]>();
        while (level.Length > 1)
        {
            levels.Add(level);
            var next = new long[level.Length / 2];
            for (int i = 0; i < next.Length; i++)
            {
                next[i] = (long)(SplitMix64.Combine((ulong)level[2 * i], (ulong)level[2 * i + 1]) % (ulong)Group.P);
            }
            level = next;
        }
        levels.Add(level);
        topLevel = level;
        return levels.ToArray();
    }

    internal static List<long> Path(IReadOnlyList<long[]> levels, int index)
    {
        var path = new List<long>();
        int position = index;
        for (int level = 0; level < levels.Count - 1; level++)
        {
            path.Add(levels[level][position ^ 1]);
            position >>= 1;
        }
        return path;
    }

    internal static long Fold(long leaf, int index, IReadOnlyList<long> path)
    {
        long node = leaf % Group.P;
        int position = index;
        foreach (long sibling in path)
        {
            node = (position & 1) == 0
                ? (long)(SplitMix64.Combine((ulong)node, (ulong)sibling) % (ulong)Group.P)
                : (long)(SplitMix64.Combine((ulong)sibling, (ulong)node) % (ulong)Group.P);
            position >>= 1;
        }
        return node;
    }
}

internal sealed class Issuer
{
    private readonly Dictionary<string, KeyPair> _keys = new Dictionary<string, KeyPair>();
    private readonly Dictionary<string, DidDocument> _documents = new Dictionary<string, DidDocument>();

    internal string Register(ulong seed)
    {
        KeyPair pair = KeyPair.FromSeed(seed);
        string tag = (SplitMix64.FromString("did:example:", seed) % 100000000UL).ToString("D8");
        string id = "did:example:" + tag;
        _keys[id] = pair;
        _documents[id] = new DidDocument { Id = id, PublicKey = pair.Public, SeedHash = seed };
        return id;
    }

    internal bool Knows(string did) => _documents.ContainsKey(did);
    internal DidDocument Document(string did) => _documents[did];
    internal KeyPair Key(string did) => _keys[did];

    internal Credential Issue(string issuerId, string subjectId, List<Claim> claims, int validFrom, int validUntil)
    {
        if (claims.Count == 0) throw new ArgumentException("a credential needs at least one claim");
        var seen = new HashSet<string>();
        foreach (Claim claim in claims)
        {
            if (!seen.Add(claim.Key)) throw new ArgumentException("claim keys must be unique");
            claim.Hash = Merkle.LeafHash(claim);
        }
        var leaves = new List<long>();
        foreach (Claim claim in claims) leaves.Add(claim.Hash);
        Merkle.BuildLevels(leaves, out long[] rootLevel);
        long root = rootLevel[0];
        ulong transcript = Transcript(issuerId, subjectId, root, validFrom, validUntil);
        var credential = new Credential
        {
            IssuerId = issuerId, SubjectId = subjectId, Claims = claims, Root = root,
            ValidFrom = validFrom, ValidUntil = validUntil, Transcript = transcript
        };
        (credential.SignatureR, credential.SignatureS) = Schnorr.Sign(_keys[issuerId], transcript, 0x8371UL);
        return credential;
    }

    internal static ulong Transcript(string issuerId, string subjectId, long root, int validFrom, int validUntil)
    {
        ulong state = SplitMix64.FromString(issuerId, 0x1111UL);
        state = SplitMix64.Mix(state, SplitMix64.FromString(subjectId, 0x2222UL));
        state = SplitMix64.Mix(state, (ulong)root);
        state = SplitMix64.Mix(state, (ulong)validFrom);
        state = SplitMix64.Mix(state, (ulong)validUntil);
        return state;
    }

    internal bool VerifyCredentialSignature(Credential credential)
    {
        if (!_documents.TryGetValue(credential.IssuerId, out DidDocument document)) return false;
        return Schnorr.Verify(document.PublicKey, credential.Transcript, credential.SignatureR, credential.SignatureS);
    }
}

internal sealed class Holder
{
    internal required string Did;
    internal required KeyPair Key;
    internal readonly List<Credential> Credentials = new List<Credential>();

    internal void Receive(Credential credential) => Credentials.Add(credential);

    /** Builds a presentation that discloses the claims whose keys are listed. */
    internal Presentation Present(Credential credential, IReadOnlyList<string> disclose, string domain, int nonce)
    {
        var leaves = new List<long>();
        var index = new Dictionary<string, int>();
        foreach (Claim claim in credential.Claims)
        {
            index[claim.Key] = leaves.Count;
            leaves.Add(claim.Hash);
        }
        long[][] levels = Merkle.BuildLevels(leaves, out _);
        var disclosed = new List<Disclosure>();
        foreach (string key in disclose)
        {
            if (!index.TryGetValue(key, out int position)) throw new ArgumentException("unknown claim: " + key);
            Claim claim = credential.Claims[position];
            disclosed.Add(new Disclosure
            {
                Key = claim.Key, Value = claim.Value, Salt = claim.Salt,
                Path = Merkle.Path(levels, position), Index = position
            });
        }
        ulong transcript = FreshnessTranscript(credential.Root, domain, nonce);
        (long r, long s) = Schnorr.Sign(Key, transcript, 0x99UL + (ulong)nonce);
        return new Presentation
        {
            IssuerId = credential.IssuerId, SubjectId = credential.SubjectId, Root = credential.Root,
            ValidFrom = credential.ValidFrom, ValidUntil = credential.ValidUntil, Disclosed = disclosed,
            IssuerSignatureR = credential.SignatureR, IssuerSignatureS = credential.SignatureS,
            HolderSignatureR = r, HolderSignatureS = s, TotalClaims = credential.Claims.Count
        };
    }

    internal static ulong FreshnessTranscript(long root, string domain, int nonce)
    {
        ulong state = SplitMix64.Mix((ulong)root, 0x5555UL);
        state = SplitMix64.Mix(state, SplitMix64.FromString(domain, 0x6666UL));
        return SplitMix64.Mix(state, (ulong)nonce);
    }
}

internal sealed class Verifier
{
    private readonly Issuer _issuer;
    private readonly HashSet<long> _revoked = new HashSet<long>();

    internal Verifier(Issuer issuer) => _issuer = issuer;

    internal void Revoke(long root) => _revoked.Add(root);

    internal bool IsRevoked(long root) => _revoked.Contains(root);

    /** The full verification policy: issuer, signature, disclosure proofs, validity, revocation and freshness. */
    internal bool Verify(Presentation presentation, string domain, int nonce, int now)
    {
        if (presentation.Disclosed.Count == 0) return false;
        if (!_issuer.Knows(presentation.IssuerId)) return false;
        if (now < presentation.ValidFrom || now > presentation.ValidUntil) return false;
        if (IsRevoked(presentation.Root)) return false;
        ulong transcript = Issuer.Transcript(presentation.IssuerId, presentation.SubjectId, presentation.Root,
            presentation.ValidFrom, presentation.ValidUntil);
        if (!Schnorr.Verify(_issuer.Document(presentation.IssuerId).PublicKey, transcript,
            presentation.IssuerSignatureR, presentation.IssuerSignatureS))
        {
            return false;
        }
        foreach (Disclosure disclosure in presentation.Disclosed)
        {
            var claim = new Claim { Key = disclosure.Key, Value = disclosure.Value, Salt = disclosure.Salt };
            long leaf = Merkle.LeafHash(claim);
            if (Merkle.Fold(leaf, disclosure.Index, disclosure.Path) != presentation.Root) return false;
        }
        if (!_issuer.Knows(presentation.SubjectId)) return false;
        ulong freshness = Holder.FreshnessTranscript(presentation.Root, domain, nonce);
        if (!Schnorr.Verify(_issuer.Document(presentation.SubjectId).PublicKey, freshness,
            presentation.HolderSignatureR, presentation.HolderSignatureS))
        {
            return false;
        }
        return true;
    }

    /** What the verifier learns: exactly the disclosed key/value pairs. */
    internal static List<string> LearnedClaims(Presentation presentation)
    {
        var learned = new List<string>();
        foreach (Disclosure disclosure in presentation.Disclosed) learned.Add(disclosure.Key + "=" + disclosure.Value);
        return learned;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static List<Claim> SampleClaims()
    {
        return new List<Claim>
        {
            new Claim { Key = "name", Value = "Ada Lovelace", Salt = 0x11L },
            new Claim { Key = "degree", Value = "MSc Mathematics", Salt = 0x22L },
            new Claim { Key = "university", Value = "Example University", Salt = 0x33L },
            new Claim { Key = "graduationYear", Value = "2031", Salt = 0x44L },
            new Claim { Key = "over18", Value = "true", Salt = 0x55L }
        };
    }

    public static void Main()
    {
        // 1-3. Group and Schnorr algebra.
        Check(Group.Pow(Group.G, Group.Order) == 1, "the generator's exponent modulus is the group order");
        Check(Group.Pow(Group.G, 1) == Group.G, "the first power of the generator is itself");
        var keyRng = 0xA1B2UL;
        ulong transcript = SplitMix64.Mix((ulong)SplitMix64.Next(ref keyRng), 0x77UL);
        var alice = KeyPair.FromSeed(0x1234UL);
        Check(alice.Public == Group.Pow(Group.G, alice.Secret), "the public key is g^secret");
        (long sigR, long sigS) = Schnorr.Sign(alice, transcript, 0x4242UL);
        Check(Schnorr.Verify(alice.Public, transcript, sigR, sigS), "an honest Schnorr signature verifies");
        Check(!Schnorr.Verify(alice.Public, transcript ^ 1UL, sigR, sigS), "a changed transcript is rejected");
        Check(!Schnorr.Verify(alice.Public, transcript, sigR + 1, sigS), "a shifted response is rejected");
        var mallory = KeyPair.FromSeed(0x9999UL);
        Check(mallory.Public != alice.Public && !Schnorr.Verify(mallory.Public, transcript, sigR, sigS),
            "another public key cannot verify the signature");
        Check(Group.Pow(Group.G, -1) == Group.Pow(Group.G, Group.Order - 1),
            "a negative exponent is normalised into the exponent modulus");

        // 4-5. DIDs are deterministic and distinct.
        var issuer = new Issuer();
        string issuerDid = issuer.Register(0xFEEDUL);
        string holderDid = issuer.Register(0xBEEFUL);
        string otherDid = issuer.Register(0xCAFEUL);
        Check(issuerDid.StartsWith("did:example:") && issuerDid.Length == 20, "the DID is a did:example identifier");
        Check(issuerDid != holderDid && holderDid != otherDid && issuerDid != otherDid, "distinct seeds give distinct DIDs");
        Check(issuer.Document(issuerDid).PublicKey == Group.Pow(Group.G, issuer.Key(issuerDid).Secret),
            "the DID document publishes exactly g^secret");
        string again = new Issuer().Register(0xFEEDUL);
        Check(again == issuerDid, "registering the same seed reproduces the same DID");

        // 6-8. Issuance and an honest presentation.
        List<Claim> claims = SampleClaims();
        Credential credential = issuer.Issue(issuerDid, holderDid, claims, 10, 100);
        Check(credential.Root != 0 && issuer.VerifyCredentialSignature(credential),
            "the issuer's signature over the credential root verifies");
        Check(Merkle.LeafHash(claims[0]) == claims[0].Hash, "claim hashes are the salted-hash commitments");
        var holder = new Holder { Did = holderDid, Key = issuer.Key(holderDid) };
        holder.Receive(credential);
        var verifier = new Verifier(issuer);
        Presentation presentation = holder.Present(credential, new List<string> { "degree", "over18" },
            "verifier.example", 4242);
        Check(verifier.Verify(presentation, "verifier.example", 4242, 50), "an honest presentation verifies");
        Check(Verifier.LearnedClaims(presentation).Count == 2, "the verifier learns exactly two claims");
        Check(Verifier.LearnedClaims(presentation)[0] == "degree=MSc Mathematics"
            && Verifier.LearnedClaims(presentation)[1] == "over18=true", "and learns exactly the disclosed values");
        Check(Merkle.Fold(Merkle.LeafHash(new Claim { Key = "degree", Value = "MSc Mathematics", Salt = 0x22L }),
            presentation.Disclosed[0].Index, presentation.Disclosed[0].Path) == credential.Root,
            "the disclosure's Merkle path folds back to the credential root");

        // 9-13. The verification policy rejects every kind of tampering.
        Check(!verifier.Verify(presentation, "other.example", 4242, 50), "a presentation for another verifier is rejected");
        Check(!verifier.Verify(presentation, "verifier.example", 4243, 50), "a stale nonce is rejected");
        Check(!verifier.Verify(presentation, "verifier.example", 4242, 101), "an expired credential is rejected");
        Check(!verifier.Verify(presentation, "verifier.example", 4242, 9), "a not-yet-valid credential is rejected");
        var tamperedValue = holder.Present(credential, new List<string> { "degree" }, "verifier.example", 7);
        tamperedValue.Disclosed[0].Value = "PhD Physics";
        Check(!verifier.Verify(tamperedValue, "verifier.example", 7, 50), "a changed claim value fails its Merkle path");
        var tamperedSalt = holder.Present(credential, new List<string> { "degree" }, "verifier.example", 8);
        tamperedSalt.Disclosed[0].Salt = 0x99L;
        Check(!verifier.Verify(tamperedSalt, "verifier.example", 8, 50), "a changed salt fails its Merkle path");
        var tamperedPath = holder.Present(credential, new List<string> { "degree" }, "verifier.example", 9);
        tamperedPath.Disclosed[0].Path = new List<long>(tamperedPath.Disclosed[0].Path) { 12345L };
        Check(!verifier.Verify(tamperedPath, "verifier.example", 9, 50), "a tampered Merkle path is rejected");
        var forgedRoot = holder.Present(credential, new List<string> { "degree" }, "verifier.example", 10);
        forgedRoot.Root = forgedRoot.Root ^ 1L;
        Check(!verifier.Verify(forgedRoot, "verifier.example", 10, 50), "a root the issuer never signed is rejected");
        var impostor = new Holder { Did = holderDid, Key = KeyPair.FromSeed(0xBADUL) };
        var stolen = impostor.Present(credential, new List<string> { "degree" }, "verifier.example", 11);
        Check(!verifier.Verify(stolen, "verifier.example", 11, 50), "a presentation signed by the wrong holder is rejected");
        var noDisclosure = holder.Present(credential, new List<string>(), "verifier.example", 12);
        Check(!verifier.Verify(noDisclosure, "verifier.example", 12, 50), "an empty presentation is rejected");

        // 14-15. Revocation and unknown issuers.
        var revokedVerifier = new Verifier(issuer);
        revokedVerifier.Revoke(credential.Root);
        Check(!revokedVerifier.Verify(presentation, "verifier.example", 4242, 50), "a revoked credential is rejected");
        var strangerIssuer = new Issuer();
        var bogus = new Presentation
        {
            IssuerId = "did:example:00000000", SubjectId = holderDid, Root = credential.Root,
            ValidFrom = 10, ValidUntil = 100, Disclosed = presentation.Disclosed, TotalClaims = 5,
            IssuerSignatureR = presentation.IssuerSignatureR, IssuerSignatureS = presentation.IssuerSignatureS,
            HolderSignatureR = presentation.HolderSignatureR, HolderSignatureS = presentation.HolderSignatureS
        };
        Check(!verifier.Verify(bogus, "verifier.example", 4242, 50), "an unknown issuer is rejected");
        Check(!strangerIssuer.Knows("did:example:00000000"), "an unregistered issuer has no document");

        // 16-17. Selective disclosure: the hidden claims leak nothing, and the assignment is the holder's choice.
        Presentation partial = holder.Present(credential, new List<string> { "over18" }, "verifier.example", 21);
        Check(verifier.Verify(partial, "verifier.example", 21, 50), "a single-claim disclosure verifies");
        Check(partial.Root == credential.Root && partial.IssuerSignatureR == credential.SignatureR,
            "the credential root and the issuer signature do not change with the disclosure set");
        Check(partial.Disclosed.Count == 1 && partial.TotalClaims == 5, "one of five claims is disclosed");
        bool leaksSecret = false;
        foreach (Claim claim in credential.Claims)
        {
            if (claim.Key == "over18") continue;
            foreach (Disclosure disclosure in partial.Disclosed)
            {
                if (disclosure.Value == claim.Value || disclosure.Salt == claim.Salt) leaksSecret = true;
            }
            if (credential.Root.ToString().Contains(claim.Value)) leaksSecret = true;
        }
        Check(!leaksSecret, "the presentation contains no value and no salt of the four hidden claims");
        Presentation different = holder.Present(credential, new List<string> { "name", "university" }, "verifier.example", 22);
        Check(verifier.Verify(different, "verifier.example", 22, 50)
            && Verifier.LearnedClaims(different).Count == 2, "the holder can pick a different subset and it verifies");
        Check(different.HolderSignatureR != partial.HolderSignatureR,
            "each presentation is signed afresh, so two presentations are not linkable by signature");
        Check(different.Root == partial.Root, "although the credential root is the same");

        // 18-20. Input validation and degenerate credentials.
        var badIssuer = new Issuer();
        string badDid = badIssuer.Register(5UL);
        Check(Throws(() => badIssuer.Issue(badDid, holderDid, new List<Claim>(), 1, 9)), "a credential with no claims is rejected");
        Check(Throws(() => badIssuer.Issue(badDid, holderDid, new List<Claim>
        {
            new Claim { Key = "a", Value = "1", Salt = 1L }, new Claim { Key = "a", Value = "2", Salt = 2L }
        }, 1, 9)), "duplicate claim keys are rejected");
        Check(Throws(() => holder.Present(credential, new List<string> { "notAClaim" }, "verifier.example", 3)),
            "disclosing an unknown claim is rejected");
        var single = issuer.Issue(issuerDid, holderDid, new List<Claim> { new Claim { Key = "only", Value = "yes", Salt = 7L } }, 0, 1000);
        var singleHolder = new Holder { Did = holderDid, Key = issuer.Key(holderDid) };
        singleHolder.Receive(single);
        Check(verifier.Verify(singleHolder.Present(single, new List<string> { "only" }, "verifier.example", 31),
            "verifier.example", 31, 5), "a one-claim credential works end to end");
        Check(!verifier.Verify(singleHolder.Present(single, new List<string> { "only" }, "verifier.example", 31),
            "verifier.example", 5, 32), "a single-claim presentation is still bound to its domain and nonce");

        // 21. Determinism.
        Presentation repeat = holder.Present(credential, new List<string> { "degree", "over18" }, "verifier.example", 4242);
        Check(repeat.HolderSignatureR == presentation.HolderSignatureR && repeat.Root == presentation.Root,
            "repeating the same presentation reproduces it bit for bit");
        Credential reissued = issuer.Issue(issuerDid, holderDid, SampleClaims(), 10, 100);
        Check(reissued.Root == credential.Root && reissued.SignatureS == credential.SignatureS,
            "re-issuing the same claims reproduces the same credential");

        // 22. Honest limitation: the group is 31 bits, so the discrete log is brute-forceable - measured on a
        //     deliberately smaller group so the cost of the shortcut is visible rather than claimed.
        const long tinyP = 65537;
        long target = Group.PowSmall(3, 12345, tinyP);
        long found = -1;
        for (long candidate = 1; candidate < tinyP; candidate++)
        {
            if (Group.PowSmall(3, candidate, tinyP) == target)
            {
                found = candidate;
                break;
            }
        }
        Console.WriteLine("  limitation: the discrete log on a 16-bit group took " + found + " tries, so the 31-bit"
            + " group used here is a demonstration of the algebra and not of security");
        Check(found == 12345, "the brute-force discrete log recovers the exponent on the small group");
        Check(Group.Pow(Group.G, 2147483646L) == 1, "and the real group still satisfies its order relation");

        Console.WriteLine("PASS");
    }
}
