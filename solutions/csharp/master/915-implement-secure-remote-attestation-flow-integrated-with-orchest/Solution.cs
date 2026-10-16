// Task 915 - Implement secure remote attestation flow integrated with orchestration for trust chain.
// (Tier: Master, Beyond - Security & Privacy)
//
// A node proves *what* it is running before an orchestrator will place work on it:
//
//  * the firmware/appraisal log is folded into a PCR with a hash chain (pcr_i = H(pcr_{i-1} || digest_i)),
//    so any change to any component moves the final value;
//  * the attestation key is trusted only through an endorsement chain root -> intermediate -> AK, where each
//    certificate carries a digest over the parent key, its own key and its serial, which is re-derived here
//    independently of the node's own claim;
//  * a quote binds the PCR and the verifier's challenge nonce with the AK, and it is accepted only if the
//    chain verifies, the AK is not revoked, the endorsement is still inside its validity window, the nonce
//    has never been used before and the PCR is on the expected-value policy.
//
// The orchestration half is the payoff: the scheduler admits a shard only onto a node that attested, so
// placement completeness (every shard placed) and placement safety (nothing on an untrusted node) are both
// checked after a round that mixes healthy, tampered and revoked nodes.
//
// Simplifications: SHA-256 digests instead of real asymmetric signatures (integrity only, no secrecy), an
// in-memory revocation list and a logical clock instead of wall-clock validity.  Verified: the PCR against a
// brute-force fold, five distinct rejection paths, the unrevoked/fresh/valid happy path, placement
// completeness and safety, the empty/single-measurement and unknown-key boundaries, and determinism.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal static string Digest(string material) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(material))).ToLowerInvariant().Substring(0, 32);

    internal static string HashPair(string left, string right) => Digest(left + "|" + right);

    internal sealed class Endorsement
    {
        internal string Subject = string.Empty;
        internal string ParentKey = string.Empty;
        internal string PublicKey = string.Empty;
        internal long Serial;
        internal long ValidUntil;
        internal string Signature = string.Empty;

        internal string ComputeSignature() => HashPair(ParentKey, PublicKey + "#" + Serial.ToString(CultureInfo.InvariantCulture));
    }

    internal sealed class Quote
    {
        internal string AkPublicKey = string.Empty;
        internal string Pcr = string.Empty;
        internal string Nonce = string.Empty;
        internal string Signature = string.Empty;

        internal string ComputeSignature() => Digest(AkPublicKey + "|" + Pcr + "|" + Nonce);
    }

    internal sealed class Node
    {
        internal string Name = string.Empty;
        internal string PublicKey = string.Empty;
        internal List<string> Measurements = new List<string>();
        internal string Pcr = string.Empty;

        /** Fold the appraisal log into the PCR; the initial value is the platform's root of trust. */
        internal string ComputePcr()
        {
            string pcr = Digest("root-of-trust");
            foreach (string measurement in Measurements) pcr = HashPair(pcr, measurement);
            return pcr;
        }

        internal Quote Attest(string nonce)
        {
            Pcr = ComputePcr();
            Quote quote = new Quote { AkPublicKey = PublicKey, Pcr = Pcr, Nonce = nonce };
            quote.Signature = quote.ComputeSignature();
            return quote;
        }
    }

    internal sealed class Verifier
    {
        private readonly Dictionary<string, Endorsement> _endorsements = new Dictionary<string, Endorsement>();
        private readonly HashSet<string> _revoked = new HashSet<string>();
        private readonly HashSet<string> _usedNonces = new HashSet<string>();
        private readonly HashSet<string> _policy = new HashSet<string>();
        private readonly string _rootKey;
        internal long Now = 1000;

        internal Verifier(string rootKey)
        {
            _rootKey = rootKey;
        }

        internal void Endorse(string parentKey, Endorsement endorsement)
        {
            endorsement.ParentKey = parentKey;
            endorsement.Signature = endorsement.ComputeSignature();
            _endorsements[endorsement.PublicKey] = endorsement;
        }

        internal void Revoke(string publicKey) => _revoked.Add(publicKey);

        internal void Expect(string pcr) => _policy.Add(pcr);

        internal Endorsement? Lookup(string publicKey) => _endorsements.TryGetValue(publicKey, out Endorsement value) ? value : null;

        internal int ChainLength(string akPublicKey)
        {
            int depth = 0;
            string cursor = akPublicKey;
            HashSet<string> seen = new HashSet<string>();
            while (_endorsements.TryGetValue(cursor, out Endorsement endorsement) && seen.Add(cursor))
            {
                depth++;
                cursor = endorsement.ParentKey;
            }
            return depth;
        }

        /** Independent chain check: every link's digest must be re-derivable, ending at the trusted root. */
        internal bool ChainVerifies(string akPublicKey)
        {
            string cursor = akPublicKey;
            HashSet<string> seen = new HashSet<string>();
            while (true)
            {
                if (!_endorsements.TryGetValue(cursor, out Endorsement endorsement)) return cursor == _rootKey;
                if (!seen.Add(cursor)) return false;
                if (endorsement.ComputeSignature() != endorsement.Signature) return false;
                cursor = endorsement.ParentKey;
            }
        }

        internal (bool Accepted, string Reason) Verify(Quote quote, string expectedNonce)
        {
            if (_revoked.Contains(quote.AkPublicKey)) return (false, "revoked AK");
            if (!_endorsements.TryGetValue(quote.AkPublicKey, out Endorsement endorsement)) return (false, "unknown AK");
            if (!ChainVerifies(quote.AkPublicKey)) return (false, "endorsement chain does not verify");
            if (endorsement.ValidUntil < Now) return (false, "endorsement expired");
            if (quote.Nonce != expectedNonce) return (false, "nonce does not match the challenge");
            if (!_usedNonces.Add(quote.Nonce)) return (false, "replayed nonce");
            if (quote.ComputeSignature() != quote.Signature) return (false, "quote signature does not verify");
            if (!_policy.Contains(quote.Pcr)) return (false, "PCR is not on the expected-value policy");
            return (true, "attested");
        }
    }

    internal sealed class Fabric
    {
        internal readonly List<string> TrustedNodes = new List<string>();
        internal readonly Dictionary<int, string> Placement = new Dictionary<int, string>();

        /** Only attested nodes may host shards: placement completeness and safety both have to hold. */
        internal void Place(int shards, IEnumerable<string> attested)
        {
            List<string> candidates = new List<string>(attested);
            if (candidates.Count == 0) return;
            for (int shard = 0; shard < shards; shard++) Placement[shard] = candidates[shard % candidates.Count];
        }
    }

    internal static Node MakeNode(string name, params string[] measurements)
    {
        Node node = new Node { Name = name, PublicKey = "ak:" + name };
        node.Measurements.AddRange(measurements);
        node.Pcr = node.ComputePcr();
        return node;
    }

    public static void Main()
    {
        const string rootKey = "mfr-root-key";
        const string intermediateKey = "mfr-intermediate-key";
        Verifier verifier = new Verifier(rootKey);

        Endorsement intermediate = new Endorsement
        {
            Subject = "mfr-intermediate",
            PublicKey = intermediateKey,
            Serial = 1,
            ValidUntil = 5000,
        };
        verifier.Endorse(rootKey, intermediate);

        Node healthy = MakeNode("node-h", "boot-1.4", "kernel-6.6", "app-2.1");
        Node healthy2 = MakeNode("node-h2", "boot-1.4", "kernel-6.6", "app-2.1");
        Node tampered = MakeNode("node-t", "boot-1.4", "kernel-6.6", "app-2.0");
        Node revokedNode = MakeNode("node-r", "boot-1.4", "kernel-6.6", "app-2.1");
        Node expiring = MakeNode("node-e", "boot-1.4", "kernel-6.6", "app-2.1");
        Node unknown = MakeNode("node-u", "boot-1.4", "kernel-6.6", "app-2.1");

        foreach (Node node in new[] { healthy, healthy2, tampered, revokedNode, expiring })
        {
            Endorsement endorsement = new Endorsement
            {
                Subject = node.Name,
                PublicKey = node.PublicKey,
                Serial = 10,
                ValidUntil = node == expiring ? verifier.Now - 1 : 5000,
            };
            verifier.Endorse(intermediateKey, endorsement);
        }
        verifier.Revoke(revokedNode.PublicKey);
        verifier.Expect(healthy.ComputePcr());

        // ---- 1. the PCR against an independent brute-force fold ---------------------------------------
        string pcr = Digest("root-of-trust");
        for (int i = 0; i < healthy.Measurements.Count; i++)
        {
            pcr = HashPair(pcr, healthy.Measurements[i]);
        }
        Check(pcr == healthy.ComputePcr(), "the PCR matches an independently folded hash chain");
        Check(pcr != Digest("root-of-trust"), "the PCR moved away from the platform root of trust");
        Check(Digest("root-of-trust").Length == 32, "digests are 32 hex characters");
        Node noMeasurements = MakeNode("node-empty");
        Check(noMeasurements.ComputePcr() == Digest("root-of-trust"), "an empty appraisal log leaves the PCR at the root of trust");
        Node single = MakeNode("node-s", "boot-1.4");
        Check(single.ComputePcr() == HashPair(Digest("root-of-trust"), "boot-1.4"), "a single-component log folds exactly once");

        // ---- 2. the endorsement chain ------------------------------------------------------------------
        Check(verifier.ChainLength(healthy.PublicKey) == 2, "the chain root -> intermediate -> AK has depth 2");
        Check(verifier.ChainVerifies(healthy.PublicKey), "the healthy node's endorsement chain verifies");
        Check(verifier.ChainLength("ak:not-endorsed") == 0 && !verifier.ChainVerifies("ak:not-endorsed"),
            "an unendorsed key has no chain");
        Endorsement forged = new Endorsement
        {
            Subject = "rogue",
            PublicKey = "ak:rogue",
            Serial = 11,
            ValidUntil = 5000,
        };
        verifier.Endorse("mfr-rogue-key", forged);
        Check(!verifier.ChainVerifies("ak:rogue"), "a certificate issued by an untrusted parent is rejected");

        // ---- 3. quotes: the happy path and every rejection path ----------------------------------------
        const string challenge = "challenge-17";
        Quote good = healthy.Attest(challenge);
        (bool accepted, string reason) = verifier.Verify(good, challenge);
        Check(accepted, "the healthy node's fresh quote is accepted (" + reason + ")");

        Quote replay = healthy.Attest(challenge);
        Check(!verifier.Verify(replay, challenge).Accepted, "the same nonce cannot be used twice");
        Quote fresh = healthy.Attest("challenge-18");
        Check(verifier.Verify(fresh, "challenge-18").Accepted, "a new challenge is accepted again");

        Quote tamperedQuote = tampered.Attest("challenge-19");
        (bool tamperedAccepted, string tamperedReason) = verifier.Verify(tamperedQuote, "challenge-19");
        Check(!tamperedAccepted && tamperedReason.Contains("policy", StringComparison.Ordinal),
            "a node running a different appraisal is rejected (" + tamperedReason + ")");
        Quote badSignature = new Quote { AkPublicKey = healthy.PublicKey, Pcr = healthy.ComputePcr(), Nonce = "challenge-20" };
        badSignature.Signature = Digest("forged");
        Check(!verifier.Verify(badSignature, "challenge-20").Accepted, "a quote with a broken signature is rejected");

        // tampering with a certificate after it was issued breaks the chain
        Endorsement? hacked = verifier.Lookup(healthy2.PublicKey);
        if (hacked != null) hacked.ValidUntil = 5000;
        Check(verifier.ChainVerifies(healthy2.PublicKey), "an untouched certificate still verifies");
        if (hacked != null) hacked.Serial = 99;
        Check(!verifier.ChainVerifies(healthy2.PublicKey), "a certificate whose serial was changed fails the chain check");
        if (hacked != null) hacked.Serial = 10;
        Check(verifier.ChainVerifies(healthy2.PublicKey), "restoring the serial restores the chain");

        Quote revokedQuote = revokedNode.Attest("challenge-21");
        Check(!verifier.Verify(revokedQuote, "challenge-21").Accepted, "a revoked attestation key is rejected");
        Quote expiredQuote = expiring.Attest("challenge-22");
        Check(!verifier.Verify(expiredQuote, "challenge-22").Accepted, "an expired endorsement is rejected");
        Quote unknownQuote = unknown.Attest("challenge-23");
        Check(!verifier.Verify(unknownQuote, "challenge-23").Accepted, "an unknown key is rejected");
        Check(!verifier.Verify(good, "challenge-24").Accepted, "a quote answering a different challenge is rejected");

        // ---- 4. orchestration uses only attested nodes ---------------------------------------------------
        Fabric fabric = new Fabric();
        List<string> attested = new List<string>();
        int denied = 0;
        foreach (Node node in new[] { healthy, healthy2, tampered, revokedNode, expiring, unknown })
        {
            Quote quote = node.Attest("challenge-" + node.Name);
            if (verifier.Verify(quote, "challenge-" + node.Name).Accepted) attested.Add(node.Name);
            else denied++;
        }
        Check(attested.Count == 2 && denied == 4,
            "two nodes attested and four were denied (" + attested.Count + " accepted, " + denied + " denied)");
        fabric.Place(8, attested);
        Check(fabric.Placement.Count == 8, "every shard was placed on the attested set");
        bool safe = true;
        foreach (KeyValuePair<int, string> entry in fabric.Placement) safe &= attested.Contains(entry.Value);
        Check(safe, "no shard landed on an unattested node");
        Fabric empty = new Fabric();
        empty.Place(4, new List<string>());
        Check(empty.Placement.Count == 0, "with no attested node nothing is placed at all");

        // ---- 5. determinism ---------------------------------------------------------------------------------
        Node replayNode = MakeNode("node-h", "boot-1.4", "kernel-6.6", "app-2.1");
        Check(replayNode.ComputePcr() == healthy.ComputePcr(), "the same appraisal log folds to the same PCR");
        Check(intermediate.ComputeSignature() == HashPair(rootKey, intermediateKey + "#1"),
            "certificate digests are pure functions of parent, key and serial");
        Node reordered = MakeNode("node-h", "app-2.1", "kernel-6.6", "boot-1.4");
        Check(reordered.ComputePcr() != healthy.ComputePcr(), "measurement order matters to the PCR");

        Console.WriteLine("chain: depth " + verifier.ChainLength(healthy.PublicKey) + " verified for the healthy node, rogue parent rejected");
        Console.WriteLine("quotes: fresh accepted, then replay, revocation, expiry, unknown key, bad signature and policy mismatch all rejected");
        Console.WriteLine("orchestration: " + attested.Count + " attested nodes host all 8 shards, " + denied + " nodes excluded");
        Console.WriteLine("PASS");
    }
}
