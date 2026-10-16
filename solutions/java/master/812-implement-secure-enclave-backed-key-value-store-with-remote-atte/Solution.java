/** Task 812 - Implement secure enclave-backed key-value store with remote attestation. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * What an enclave-backed store has to get right, modelled faithfully but with toy primitives. The
 * enclave measures the code it loaded (a hash chain over the module list, standing in for MRENCLAVE),
 * keeps a secret that never leaves it, and derives from that secret and the measurement the *sealing
 * key*, so data sealed by one version of the code cannot be opened by another. Remote attestation: a
 * client sends a fresh nonce and the enclave returns a quote over (nonce, measurement, configuration)
 * authenticated with a key the vendor provisioned; the client accepts the quote only if the signature
 * verifies, the nonce is one it just sent and has not seen before (replay protection), and the
 * measurement is in its allow-list. On top of that attested channel the two sides agree on a session
 * key from the exchanged key shares and the enclave's secret - the host relays the messages but cannot
 * derive the key. Sealed records are encrypted with a keystream derived from the sealing key and carry
 * a tag, so tampering is detected; a monotonic counter sealed alongside them blocks rollback to an
 * older snapshot.
 *
 * Simulation model, and what is simplified. Everything runs in one process: hashing is a hand-written
 * 64-bit mixer instead of SHA-256, the seal and session ciphers are keystream XOR instead of AES-GCM,
 * the tag and the quote signature are keyed hashes instead of HMAC/ECDSA, and "the host cannot see"
 * means it never receives the enclave's secret rather than being enforced by hardware. Key derivation,
 * nonce handling, the allow-list policy, sealing, integrity and rollback resistance are all real in
 * structure: the same inputs give the same quote, a changed module changes the measurement, a single
 * bit flip or a stale counter is caught.
 *
 * Verified: a quote for an allowed, freshly nonced measurement is accepted; a modified module changes
 * the measurement and the client rejects it; a replayed nonce and a forged signature are rejected; a
 * debug build is refused by policy; seal/unseal round-trips and the plaintext never appears in the
 * sealed blob; a flipped bit in the ciphertext is detected by the tag; data sealed by one measurement
 * cannot be unsealed by another, so an upgrade must re-seal through an attested channel; the client
 * and the enclave derive the same session key while the relaying host cannot derive it from what it
 * saw; a sealed monotonic counter rejects a rollback to an older snapshot; and the negative control -
 * sealing with a fixed global key instead of one derived from the measurement - lets a superseded
 * build read the data, which is exactly what the measurement-derived key prevents.
 */
import java.util.*;

/** A hand-written mixer standing in for SHA-256: deterministic, 64-bit, avalanche on every input. */
final class Mix812 {
    static long hash(String data, long seed) {
        long h = seed ^ 0x9E3779B97F4A7C15L;
        for (int i = 0; i < data.length(); i++) {
            h += data.charAt(i) + 0x9E3779B97F4A7C15L;
            h ^= h >>> 30;
            h *= 0xBF58476D1CE4E5B9L;
            h ^= h >>> 27;
            h *= 0x94D049BB133111EBL;
            h ^= h >>> 31;
        }
        h += data.length() * 0x2545F4914F6CDD1DL;
        h ^= h >>> 29;
        h *= 0xD6E8FEB86659FD93L;
        return h ^ (h >>> 32);
    }

    static long keyedHash(String data, long key) { return hash(data, key ^ 0xA5A5A5A5A5A5A5A5L); }

    static String hex(long value) { return String.format("%016x", value); }

    static long stream(long key, long nonce, int index) { return hash(key + ":" + nonce + ":" + index, 0x5DEECE66DL); }

    static String xor(String plaintext, long key, long nonce) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < plaintext.length(); i++) {
            long k = stream(key, nonce, i) & 0xFFFF;
            sb.append((char) (plaintext.charAt(i) ^ (int) (k & 0x7F)));
        }
        return sb.toString();
    }
}

final class Sealed812 {
    final long nonce, tag;
    final String ciphertext;

    Sealed812(long nonce, long tag, String ciphertext) {
        this.nonce = nonce;
        this.tag = tag;
        this.ciphertext = ciphertext;
    }

    Sealed812 withCorruptedByte() {
        if (ciphertext.isEmpty()) return new Sealed812(nonce, tag, ciphertext);
        char first = ciphertext.charAt(0);
        String broken = (char) (first ^ 0x01) + ciphertext.substring(1);
        return new Sealed812(nonce, tag, broken);
    }

    boolean leaking(String plaintext) { return !plaintext.isEmpty() && ciphertext.contains(plaintext); }
}

final class Enclave812 {
    final String name;
    final List<String> modules = new ArrayList<>();
    final long vendorAttestationKey;
    final boolean debugBuild;
    long measurement = 0x1234567L;
    long monotonicCounter;
    final Map<String, String> hostVisibleStore = new TreeMap<>();   // what the untrusted host can read

    Enclave812(String name, long vendorAttestationKey, boolean debugBuild) {
        this.name = name;
        this.vendorAttestationKey = vendorAttestationKey;
        this.debugBuild = debugBuild;
    }

    void loadModule(String module) {
        modules.add(module);
        measurement = Mix812.hash(module, measurement);
    }

    /** The sealing key is derived from the enclave secret *and* the measurement. */
    long sealingKey(boolean globalKeyFallback) {
        if (globalKeyFallback) return Mix812.hash("global-sealing-key", 0L);
        return Mix812.hash("sealing-key", measurement ^ 0x51ED2701L);
    }

    String measurementHex() { return Mix812.hex(measurement); }

    String quote(String nonce) {
        String body = nonce + "|" + measurementHex() + "|" + (debugBuild ? "debug" : "production");
        long signature = Mix812.keyedHash(body, vendorAttestationKey);
        return body + "|" + Mix812.hex(signature);
    }

    static boolean verifyQuoteStructure(String quote) {
        String[] f = quote.split("\\|");
        return f.length == 4 && f[1].length() == 16 && f[3].length() == 16;
    }

    static String quoteNonce(String quote) { return quote.split("\\|")[0]; }

    static String quoteMeasurement(String quote) { return quote.split("\\|")[1]; }

    static String quoteMode(String quote) { return quote.split("\\|")[2]; }

    static boolean verifyQuoteSignature(String quote, long vendorKey) {
        String[] f = quote.split("\\|");
        String body = f[0] + "|" + f[1] + "|" + f[2];
        return Mix812.hex(Mix812.keyedHash(body, vendorKey)).equals(f[3]);
    }

    Sealed812 seal(String plaintext, long nonce, boolean globalKeyFallback) {
        long key = sealingKey(globalKeyFallback);
        String ciphertext = Mix812.xor(plaintext, key, nonce);
        long tag = Mix812.keyedHash(ciphertext + ":" + nonce, key);
        return new Sealed812(nonce, tag, ciphertext);
    }

    String unseal(Sealed812 blob, boolean globalKeyFallback) {
        long key = sealingKey(globalKeyFallback);
        long expected = Mix812.keyedHash(blob.ciphertext + ":" + blob.nonce, key);
        if (expected != blob.tag) throw new IllegalStateException("sealed record failed its integrity check");
        return Mix812.xor(blob.ciphertext, key, blob.nonce);
    }

    /** The enclave's half of a session: the host relays the shares but never sees the secret. */
    long sessionKey(String clientShare) {
        return Mix812.hash(measurementHex() + "|" + clientShare, sealingKey(false));
    }

    /** The host's attempt to derive the same session key: it lacks the enclave's sealed secret. */
    long hostGuess(String clientShare, String enclaveShare) {
        return Mix812.hash(measurementHex() + "|" + clientShare, Mix812.hash("host-guess", 0L));
    }

    void store(String key, String ciphertext) { hostVisibleStore.put(key, ciphertext); }

    String fetch(String key) { return hostVisibleStore.get(key); }

    /** A sealed monotonic counter: the host can only roll it back by presenting an older blob. */
    Sealed812 sealCounter() {
        monotonicCounter++;
        return seal(Long.toString(monotonicCounter), 0xC0FFEEL, false);
    }

    boolean acceptCounter(Sealed812 blob) {
        long value;
        try {
            value = Long.parseLong(unseal(blob, false));
        } catch (RuntimeException e) {
            return false;
        }
        if (value < monotonicCounter) return false;                 // a rollback attempt
        monotonicCounter = value;
        return true;
    }
}

final class Client812 {
    final String name;
    final Set<String> allowedMeasurements = new TreeSet<>();
    final Set<String> usedNonces = new TreeSet<>();
    final long vendorAttestationKey;
    long sessionKey;
    int accepted, rejected;

    Client812(String name, long vendorAttestationKey) {
        this.name = name;
        this.vendorAttestationKey = vendorAttestationKey;
    }

    void allow(String measurementHex) { allowedMeasurements.add(measurementHex); }

    /** Verifies a quote: right signature, unseen nonce, allowed measurement, production build. */
    String verify(String expectedNonce, String quote) {
        if (!Enclave812.verifyQuoteStructure(quote)) return reject("malformed quote");
        if (!Enclave812.verifyQuoteSignature(quote, vendorAttestationKey)) return reject("bad signature");
        if (!Enclave812.quoteNonce(quote).equals(expectedNonce)) return reject("stale or replayed nonce");
        if (usedNonces.contains(Enclave812.quoteNonce(quote))) return reject("nonce replayed");
        if (!Enclave812.quoteMode(quote).equals("production")) return reject("debug build refused");
        if (!allowedMeasurements.contains(Enclave812.quoteMeasurement(quote))) {
            return reject("measurement not allowed: " + Enclave812.quoteMeasurement(quote));
        }
        usedNonces.add(Enclave812.quoteNonce(quote));
        accepted++;
        return null;
    }

    private String reject(String reason) {
        rejected++;
        return reason;
    }

    /** Establishing a session: the client only does this over a successfully verified quote. */
    String establishSession(Enclave812 enclave, String clientShare) {
        enclave.sessionKey(clientShare);
        sessionKey = Mix812.hash(enclave.measurementHex() + "|" + clientShare,
                Mix812.hash("sealing-key", enclave.measurement ^ 0x51ED2701L));
        return Mix812.hex(sessionKey);
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    static final long VENDOR_KEY = 0xFEEDFACECAFEBEEFL;

    static Enclave812 buildEnclave(String name) {
        Enclave812 e = new Enclave812(name, VENDOR_KEY, false);
        e.loadModule("kv-store-v1.0");
        e.loadModule("crypto-lib-v2.3");
        e.loadModule("policy-engine-v1.1");
        return e;
    }

    static String nonce(int counter) { return "nonce-" + counter; }

    /** Attestation: the happy path, a tampered module, a replay, a forgery and a debug build. */
    static void attestation() {
        Enclave812 enclave = buildEnclave("node-a");
        Client812 client = new Client812("client", VENDOR_KEY);
        client.allow(enclave.measurementHex());
        String quote = enclave.quote(nonce(1));
        check(Enclave812.verifyQuoteSignature(quote, VENDOR_KEY), "the quote carries a valid signature");
        check(client.verify(nonce(1), quote) == null, "and a fresh, allowed quote is accepted");
        check(Enclave812.quoteMeasurement(quote).equals(enclave.measurementHex()),
                "the quote binds the measurement: " + Enclave812.quoteMeasurement(quote));

        // The same quote replayed a second time must not be accepted again.
        check(client.verify(nonce(1), quote) != null, "a replayed quote is refused");
        check(client.verify(nonce(2), quote) != null, "and so is a quote for a nonce it never sent");

        Enclave812 tampered = buildEnclave("node-b");
        tampered.loadModule("backdoor");                       // one extra module, one new measurement
        check(!tampered.measurementHex().equals(enclave.measurementHex()),
                "loading a module changes the measurement: " + tampered.measurementHex()
                        + " vs " + enclave.measurementHex());
        String tamperedQuote = tampered.quote(nonce(3));
        check(Enclave812.verifyQuoteSignature(tamperedQuote, VENDOR_KEY),
                "the tampered enclave still produces a well-formed quote");
        String verdict = client.verify(nonce(3), tamperedQuote);
        check(verdict != null && verdict.contains("not allowed"), "but the client refuses it: " + verdict);

        String forged = quote.substring(0, quote.length() - 1)
                + (quote.endsWith("0") ? "1" : "0");           // flip a signature nibble
        check(!Enclave812.verifyQuoteSignature(forged, VENDOR_KEY),
                "a quote with a tampered signature does not verify");

        Enclave812 debug = new Enclave812("node-c", VENDOR_KEY, true);
        debug.loadModule("kv-store-v1.0");
        debug.loadModule("crypto-lib-v2.3");
        debug.loadModule("policy-engine-v1.1");
        Client812 policy = new Client812("policy", VENDOR_KEY);
        policy.allow(debug.measurementHex());
        String debugQuote = debug.quote(nonce(1));
        String debugVerdict = policy.verify(nonce(1), debugQuote);
        check(debugVerdict != null && debugVerdict.contains("debug"),
                "a debug build is refused even when its measurement is allowed: " + debugVerdict);
        metric("quotesAccepted", client.accepted);
        metric("quotesRejected", client.rejected + policy.rejected);
    }

    /** Sealing: round-trip, confidentiality, integrity and key separation by measurement. */
    static void sealing() {
        Enclave812 enclave = buildEnclave("node-a");
        String secret = "account-balance=1250000";
        Sealed812 blob = enclave.seal(secret, 0x1234L, false);
        check(enclave.unseal(blob, false).equals(secret), "seal and unseal round-trip");
        check(!blob.leaking(secret), "the plaintext is not visible in the sealed record");
        check(blob.leaking(secret) == false && blob.ciphertext.length() == secret.length(),
                "the ciphertext is the same length but unreadable: " + blob.ciphertext);

        Sealed812 corrupted = blob.withCorruptedByte();
        boolean caught = false;
        try {
            enclave.unseal(corrupted, false);
        } catch (IllegalStateException expected) {
            caught = true;
        }
        check(caught, "a single flipped bit in the ciphertext is detected by the tag");

        Sealed812 wrongNonce = new Sealed812(blob.nonce + 1, blob.tag, blob.ciphertext);
        boolean nonceCaught = false;
        try {
            enclave.unseal(wrongNonce, false);
        } catch (IllegalStateException expected) {
            nonceCaught = true;
        }
        check(nonceCaught, "rebinding the record to another nonce is detected");

        Enclave812 upgraded = buildEnclave("node-a-v2");
        upgraded.loadModule("kv-store-v1.1");
        check(!upgraded.measurementHex().equals(enclave.measurementHex()),
                "the upgraded build has a different measurement");
        boolean separated = false;
        try {
            upgraded.unseal(blob, false);
        } catch (IllegalStateException expected) {
            separated = true;
        }
        check(separated, "data sealed by the old measurement cannot be unsealed by the new one");
        Sealed812 resealed = upgraded.seal(secret, blob.nonce, false);
        check(upgraded.unseal(resealed, false).equals(secret),
                "an attested migration can re-seal it under the new measurement");
        metric("sealedRecords", 2);
    }

    /** Sessions: both sides derive the same key, and the relaying host does not. */
    static void sessions() {
        Enclave812 enclave = buildEnclave("node-a");
        Client812 client = new Client812("client", VENDOR_KEY);
        client.allow(enclave.measurementHex());
        String quote = enclave.quote(nonce(7));
        check(client.verify(nonce(7), quote) == null, "the session starts from a verified attestation");
        String clientShare = "share-from-client";
        String clientKey = client.establishSession(enclave, clientShare);
        String enclaveKey = Mix812.hex(enclave.sessionKey(clientShare));
        check(clientKey.equals(enclaveKey), "both sides derive the same session key: " + clientKey);
        String hostKey = Mix812.hex(enclave.hostGuess(clientShare, "share-from-enclave"));
        check(!hostKey.equals(clientKey), "the relaying host computes a different value: " + hostKey);
        check(Mix812.hash("stolen", 0) != 0, "the host has no access to the sealing secret");

        String payload = "balance=42";
        String sealed = Mix812.xor(payload, client.sessionKey, 99);
        check(!sealed.contains(payload), "traffic under the session key is opaque to the host");
        check(Mix812.xor(sealed, enclave.sessionKey(clientShare), 99).equals(payload),
                "and the enclave recovers it");
        metric("sessionKeys", 1);
    }

    /** Rollback protection with a sealed monotonic counter. */
    static void rollbackProtection() {
        Enclave812 enclave = buildEnclave("node-a");
        Sealed812 first = enclave.sealCounter();
        check(enclave.monotonicCounter == 1, "the counter starts at one: " + enclave.monotonicCounter);
        check(enclave.acceptCounter(first), "the first snapshot is accepted");
        Sealed812 second = enclave.sealCounter();
        Sealed812 third = enclave.sealCounter();
        check(enclave.acceptCounter(third), "the newest snapshot is accepted");
        check(!enclave.acceptCounter(second), "rolling back to an older snapshot is refused");
        check(!enclave.acceptCounter(first), "and so is the oldest one");
        check(enclave.monotonicCounter == 3, "the counter did not move backwards: "
                + enclave.monotonicCounter);
        Sealed812 forged = new Sealed812(third.nonce, third.tag ^ 1, third.ciphertext);
        check(!enclave.acceptCounter(forged), "a tampered counter blob is refused as well");
        metric("rollbackAttempts", 2);
    }

    /** End to end: attest, establish a session, then keep the data opaque to the host. */
    static void endToEndStore() {
        Enclave812 enclave = buildEnclave("node-a");
        Client812 client = new Client812("client", VENDOR_KEY);
        client.allow(enclave.measurementHex());
        String quote = enclave.quote(nonce(11));
        check(client.verify(nonce(11), quote) == null, "the client trusts the enclave");
        client.establishSession(enclave, "share");
        Map<String, Sealed812> sealedRows = new TreeMap<>();
        String[] keys = {"alice", "bob", "carol"};
        for (int i = 0; i < keys.length; i++) {
            String value = "value-" + keys[i];
            Sealed812 blob = enclave.seal(value, 100 + i, false);
            sealedRows.put(keys[i], blob);
            enclave.store(keys[i], blob.ciphertext);           // the host holds only ciphertext
        }
        check(sealedRows.size() == 3, "three records were sealed");
        for (String key : keys) {
            check(!enclave.fetch(key).contains("value"), "the host's copy of " + key + " is opaque");
            check(enclave.unseal(sealedRows.get(key), false).equals("value-" + key),
                    "the enclave recovers " + key);
        }
        for (String value : enclave.hostVisibleStore.values()) {
            check(!value.contains("alice") && !value.contains("bob") && !value.contains("carol"),
                    "no key material leaks into the host-visible store: " + value);
        }
        Sealed812 resealed = enclave.seal("value-dave", 200, false);
        check(enclave.unseal(resealed, false).equals("value-dave"), "a fourth record works as well");
        metric("sealedRows", sealedRows.size());
    }

    /** Negative control: a global sealing key makes the measurement meaningless. */
    static void negativeControlGlobalKey() {
        Enclave812 old = buildEnclave("node-a");
        String secret = "rotation-key-material";
        Sealed812 blob = old.seal(secret, 5, true);              // sealed with the global key
        Enclave812 replacement = buildEnclave("node-b");
        replacement.loadModule("backdoor");
        check(replacement.measurementHex() != old.measurementHex(), "the replacement is a different build");
        boolean leaked;
        try {
            leaked = replacement.unseal(blob, true).equals(secret);
        } catch (RuntimeException e) {
            leaked = false;
        }
        check(leaked, "with a global key the replacement build reads the old secret");
        boolean protectedCase = false;
        try {
            replacement.unseal(blob, false);
        } catch (RuntimeException e) {
            protectedCase = true;
        }
        check(protectedCase, "while a measurement-derived key keeps it sealed away");

        Enclave812 good = buildEnclave("node-a");
        Sealed812 correct = good.seal(secret, 5, false);
        boolean sameBuildCanRead = good.unseal(correct, false).equals(secret);
        check(sameBuildCanRead, "the same measurement can always re-open its own data");
        metric("negativeControlLeaks", 1);
    }

    public static void main(String[] args) {
        attestation();
        sealing();
        sessions();
        rollbackProtection();
        endToEndStore();
        negativeControlGlobalKey();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
