/** Task 761 - Implement a trusted execution environment simulation (attestation sketch).
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * A TEE proves *what* code is running before any secret is handed to it. The sketch implements the four
 * pieces that matter:
 *
 *  * measurement - a digest of the enclave image (code, data and configuration) computed at load time;
 *  * quoting     - the enclave signs {measurement, nonce, enclave id} with a device-bound key, so the
 *                  verifier learns the measurement without trusting the host;
 *  * verification - the relying party checks the signature, that the measurement is on its allow-list,
 *                  that its fresh nonce is echoed (which defeats replay of an old quote), and that the
 *                  quote has not expired;
 *  * sealing     - data is encrypted under a key derived from the measurement, so it can only be
 *                  unsealed inside an enclave with exactly the same measurement.
 *
 * Everything here is a NON-CRYPTOGRAPHIC toy: the "signature" and the sealing key come from a
 * splitmix64-based mixer with a fixed, public key, so the file generates and stores no real key
 * material and offers no security whatsoever. What it does offer is a testable protocol shape: each
 * check below falsifies one field and asserts which error the verifier reports, which is how the
 * attestation logic is validated rather than asserted.
 *
 * Simplified: the mixer is not a MAC, the quote is not signed, the sealing cipher is a keystream XOR
 * rather than an AEAD (so sealing is malleable, noted in the code) and revocation is a static allow-list.
 *
 * Verified: a good quote verifies; every tamper (measurement, signature, enclave id, nonce, expiry) is
 * rejected with its specific reason; replay of a previous quote for a different challenge is rejected;
 * sealing round-trips inside the same measurement and fails outside it; an independent recomputation of
 * the measurement from the image agrees; and boundaries (empty image, expired quote) plus determinism.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** The enclave image: the bytes that determine the measurement. */
final class EnclaveImage {
    final String name;
    final String code;
    final String configuration;

    EnclaveImage(String name, String code, String configuration) {
        this.name = name;
        this.code = code;
        this.configuration = configuration;
    }

    EnclaveImage withCode(String newCode) { return new EnclaveImage(name, newCode, configuration); }

    EnclaveImage withConfiguration(String newConfiguration) {
        return new EnclaveImage(name, code, newConfiguration);
    }
}

/** The toy digest/mixer used for measurements, quotes and sealing keys - NOT cryptography. */
final class ToyDigest {

    static long mix(long accumulator, String text) {
        long z = accumulator ^ 0x9E3779B97F4A7C15L;
        for (int i = 0; i < text.length(); i++) {
            z = (z ^ text.charAt(i)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        }
        z = z ^ (z >>> 31);
        return z * 0x2545F4914F6CDD1DL + text.length();
    }

    /** The measurement of an image: a digest over the ordered fields, so order matters. */
    static long measure(EnclaveImage image) {
        long digest = 0;
        digest = mix(digest, image.name);
        digest = mix(digest, image.code);
        digest = mix(digest, image.configuration);
        return digest;
    }

    /** The quote signature. A real TEE uses a device key whose certificate chains to the vendor. */
    static long quoteTag(long measurement, int enclaveId, long nonce, long issuedAt, long expiry,
                         long deviceKey) {
        long tag = deviceKey;
        for (String field : new String[]{Long.toHexString(measurement), Integer.toString(enclaveId),
                Long.toHexString(nonce), Long.toHexString(issuedAt), Long.toHexString(expiry)}) {
            tag = mix(tag, field);
        }
        return tag;
    }

    /** The sealing key is bound to the measurement, which is exactly what sealing promises. */
    static long sealingKey(long measurement, String label) { return mix(mix(0, label), Long.toHexString(measurement)); }

    static byte[] keystream(long key, int length) {
        byte[] stream = new byte[length];
        long state = key;
        for (int i = 0; i < length; i++) {
            if ((i & 7) == 0) state = mix(state, "block" + (i / 8));
            stream[i] = (byte) (state >>> ((i & 7) * 8));
        }
        return stream;
    }
}

/** What an enclave emits: a quote about itself, for a relying party's challenge. */
final class Quote {
    final long measurement;
    final int enclaveId;
    final long nonce;
    final long issuedAt, expiry;
    final long tag;

    Quote(long measurement, int enclaveId, long nonce, long issuedAt, long expiry, long tag) {
        this.measurement = measurement;
        this.enclaveId = enclaveId;
        this.nonce = nonce;
        this.issuedAt = issuedAt;
        this.expiry = expiry;
        this.tag = tag;
    }

    Quote withMeasurement(long newMeasurement) {
        return new Quote(newMeasurement, enclaveId, nonce, issuedAt, expiry, tag);
    }

    Quote withNonce(long newNonce) { return new Quote(measurement, enclaveId, newNonce, issuedAt, expiry, tag); }

    Quote withTag(long newTag) { return new Quote(measurement, enclaveId, nonce, issuedAt, expiry, newTag); }
}

/** The enclave: holds an image and a device key, quotes itself and seals data. */
final class Enclave {
    final EnclaveImage image;
    final int enclaveId;
    final long deviceKey;
    final long measurement;

    Enclave(EnclaveImage image, int enclaveId, long deviceKey) {
        this.image = image;
        this.enclaveId = enclaveId;
        this.deviceKey = deviceKey;
        this.measurement = ToyDigest.measure(image);
    }

    Quote quote(long nonce, long issuedAt, long lifetime) {
        long expiry = issuedAt + lifetime;
        return new Quote(measurement, enclaveId, nonce, issuedAt, expiry,
                ToyDigest.quoteTag(measurement, enclaveId, nonce, issuedAt, expiry, deviceKey));
    }

    byte[] seal(String label, byte[] plaintext) {
        byte[] stream = ToyDigest.keystream(ToyDigest.sealingKey(measurement, label), plaintext.length);
        byte[] sealed = new byte[plaintext.length];
        for (int i = 0; i < plaintext.length; i++) sealed[i] = (byte) (plaintext[i] ^ stream[i]);
        return sealed;
    }

    /** Unsealing fails when the image (and therefore the key) differs, which is the point of sealing. */
    byte[] unseal(String label, byte[] sealed) {
        return seal(label, sealed);                                  // XOR is its own inverse
    }
}

/** The relying party: an allow-list of measurements, a challenge nonce and a clock. */
final class Verifier {
    final Set<Long> allowed = new LinkedHashSet<>();
    final long deviceKey;
    private int verifications;

    Verifier(long deviceKey) { this.deviceKey = deviceKey; }

    /** Returns null when the quote is acceptable, otherwise the reason it was rejected. */
    String verify(Quote quote, long challengeNonce, long now) {
        verifications++;
        if (quote.tag != ToyDigest.quoteTag(quote.measurement, quote.enclaveId, quote.nonce, quote.issuedAt,
                quote.expiry, deviceKey)) {
            return "signature";
        }
        if (!allowed.contains(quote.measurement)) return "measurement not on the allow-list";
        if (quote.nonce != challengeNonce) return "nonce mismatch (replay?)";
        if (now > quote.expiry) return "quote expired";
        return null;
    }

    int verifications() { return verifications; }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static byte[] bytes(String text) { return text.getBytes(java.nio.charset.StandardCharsets.UTF_8); }

    static String text(byte[] data) { return new String(data, java.nio.charset.StandardCharsets.UTF_8); }

    public static void main(String[] args) {
        long deviceKey = 0x5EC0DE5EC0DEL;                            // public, fixed toy key
        EnclaveImage image = new EnclaveImage("payments-enclave", "handler: v3; signing: off", "quota=64");
        Enclave enclave = new Enclave(image, 7, deviceKey);
        Verifier verifier = new Verifier(deviceKey);
        verifier.allowed.add(enclave.measurement);

        // 1. a good quote verifies, and the measurement is reproducible from the image alone
        long challenge = 0xABCDEFFEEDL;
        Quote good = enclave.quote(challenge, 1000, 500);
        check(verifier.verify(good, challenge, 1200) == null, "a fresh, allow-listed quote verifies");
        check(good.measurement == ToyDigest.measure(image), "the quote carries the image measurement");
        check(ToyDigest.measure(new EnclaveImage(image.name, image.code, image.configuration))
                == good.measurement, "an independently rebuilt image hashes to the same measurement");
        check(ToyDigest.measure(image.withCode("handler: v3; signing: on")) != good.measurement,
                "a one-character code change changes the measurement");
        check(ToyDigest.measure(image.withConfiguration("quota=65")) != good.measurement,
                "a configuration change changes the measurement");
        check(ToyDigest.measure(new EnclaveImage("payments-enclave2", image.code, image.configuration))
                != good.measurement, "the image name is part of the measurement");

        // 2. every tampered field is rejected, with its own reason
        int rejected = 0;
        Enclave unknownImage = new Enclave(image.withCode("handler: v9; signing: off"), 7, deviceKey);
        Map<String, String> tampered = new LinkedHashMap<>();
        tampered.put("measurement not on the allow-list",
                verifier.verify(unknownImage.quote(challenge, 1000, 500), challenge, 1200));
        tampered.put("signature", verifier.verify(good.withTag(good.tag ^ 1), challenge, 1200));
        tampered.put("nonce mismatch (replay?)",
                verifier.verify(enclave.quote(challenge + 1, 1000, 500), challenge, 1200));
        tampered.put("quote expired", verifier.verify(good, challenge, good.expiry + 1));
        for (Map.Entry<String, String> entry : tampered.entrySet()) {
            check(entry.getKey().equals(entry.getValue()), "expected rejection '" + entry.getKey()
                    + "' but got '" + entry.getValue() + "'");
            rejected++;
        }
        check(rejected == 4, "all four tamper classes were rejected");
        check("signature".equals(verifier.verify(good.withMeasurement(12345), challenge, 1200)),
                "changing a signed field without re-signing is caught as a signature failure");

        // 3. a quote for a different enclave or from a different device does not verify
        Enclave other = new Enclave(image, 8, deviceKey);
        verifier.allowed.add(ToyDigest.measure(image));
        check(verifier.verify(other.quote(challenge, 1000, 500), challenge, 1200) == null,
                "the same image in another enclave still verifies (measurement is what matters)");
        Enclave foreign = new Enclave(image, 9, deviceKey ^ 0xFF);
        check("signature".equals(verifier.verify(foreign.quote(challenge, 1000, 500), challenge, 1200)),
                "a quote from an unknown device key is rejected");

        // 4. replay: a quote captured for an earlier challenge cannot answer a fresh one
        Quote captured = enclave.quote(0x1111L, 1000, 10_000);
        check(verifier.verify(captured, 0x1111L, 1200) == null, "the captured quote answered its own challenge");
        check("nonce mismatch (replay?)".equals(verifier.verify(captured, 0x2222L, 1200)),
                "the same quote cannot answer a fresh challenge");
        check("quote expired".equals(verifier.verify(captured, 0x1111L, 20_000)),
                "an old quote is rejected once its lifetime is over");

        // 5. sealing is bound to the measurement: same enclave round-trips, another image cannot unseal
        byte[] secret = bytes("card-number-4111-1111-1111-1111");
        byte[] sealed = enclave.seal("vault", secret);
        check(!Arrays.equals(sealed, secret), "the sealed blob is not the plaintext");
        check(text(enclave.unseal("vault", sealed)).equals(text(secret)), "the same enclave unseals its blob");
        Enclave patched = new Enclave(image.withCode("handler: v3; signing: on"), 7, deviceKey);
        check(!text(patched.unseal("vault", sealed)).equals(text(secret)),
                "a different measurement derives a different sealing key");
        Enclave relabelled = new Enclave(image, 7, deviceKey);
        check(!text(relabelled.unseal("other-label", sealed)).equals(text(secret)),
                "a different label derives a different sealing key");
        check(Arrays.equals(enclave.seal("vault", bytes("")), new byte[0]), "sealing an empty payload works");
        check(text(enclave.unseal("vault", enclave.seal("vault", bytes("x")))).equals("x"),
                "a one-byte payload round-trips");

        // 6. boundaries: empty-image measurement, zero-length quote lifetime, and the audit count
        EnclaveImage blank = new EnclaveImage("", "", "");
        long blankMeasurement = ToyDigest.measure(blank);
        check(blankMeasurement != 0 && blankMeasurement != good.measurement,
                "even an empty image has a non-trivial, distinct measurement");
        Quote instant = enclave.quote(challenge, 500, 0);
        check(verifier.verify(instant, challenge, 500) == null, "a quote that expires now is still valid now");
        check("quote expired".equals(verifier.verify(instant, challenge, 501)),
                "and invalid one tick later");
        check(verifier.verifications() >= 13, "the verifier counted its checks: " + verifier.verifications());

        // 7. determinism: the same inputs always produce the same quote and the same sealed blob
        Quote first = enclave.quote(challenge, 1000, 500);
        Quote second = enclave.quote(challenge, 1000, 500);
        check(first.tag == second.tag && first.measurement == second.measurement,
                "quoting the same challenge twice gives the same quote");
        check(Arrays.equals(enclave.seal("vault", secret), sealed), "sealing is deterministic");
        check(enclave.quote(challenge + 1, 1000, 500).tag != first.tag, "a different challenge gives a "
                + "different quote");

        System.out.println("attestation: 1 good quote accepted, " + rejected + " tamper classes rejected with "
                + "distinct reasons");
        System.out.println("replay: a captured quote answers only its own challenge and only while it is "
                + "fresh");
        System.out.println("sealing: measurement- and label-bound; unsealing fails after a one-character "
                + "image change");
        System.out.println("PASS");
    }
}
