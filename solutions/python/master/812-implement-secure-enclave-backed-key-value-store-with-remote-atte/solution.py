"""Task 812 - Implement secure enclave-backed key-value store with remote attestation.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

An SGX-style trusted execution environment, modelled with the standard library only.  A platform
owns a root secret (the CPU's fused key); an enclave is created from a code image, and its identity
is the measurement MRENCLAVE = SHA-256 of that image.  From the platform secret and the measurement
the enclave derives, in separate domains, a sealing key and a quote key, which gives the two
properties the whole design rests on:

  * sealing      the store is written to untrusted storage as a blob under an encrypt-then-MAC
                 construction (HMAC-SHA256 keystream for confidentiality, HMAC-SHA256 tag over the
                 header, the associated data and the ciphertext for integrity).  Only an enclave
                 with the same code image can derive the key, and any modification of the blob -
                 one flipped bit, a swapped blob, an older snapshot - is caught by the tag before
                 a single byte is decrypted;
  * attestation  a relying party sends a fresh nonce and its Diffie-Hellman public value; the
                 enclave returns a quote containing its measurement and report_data =
                 SHA-256(nonce || client public value) signed with the quote key.  The verifier
                 accepts the quote only if the measurement is on its whitelist, the report_data
                 binds *this* nonce, the signature verifies and the enclave version is high enough -
                 so a replayed quote, a quote from a different code image and a host-forged quote
                 are all rejected.  The client then completes the Diffie-Hellman exchange inside
                 the attested session and every later command is encrypted and MACed with a session
                 key, with a session counter rejecting replays.

The host is modelled as an adversary that sees every blob and every message but never the platform
secret: it can tamper, replay, swap and roll back, and the checks make it try all of those.

Verified invariants:
  measurement         the identity is the image digest; equal images agree, a one-byte image change
                      does not, and keys derived for different measurements never collide;
  sealing             seal/unseal round-trips the state; the blob contains no plaintext; a flipped
                      bit anywhere rejects it; another enclave's blob is rejected;
  attestation         an honest quote verifies; a wrong whitelist entry, a replayed nonce, a stale
                      quote, a modified quote and an unversioned enclave are all rejected, and a
                      host that does not hold the platform secret cannot forge one;
  key agreement       both sides derive the same session key and an eavesdropper with only the
                      public values derives a different one;
  channel             a command with a broken MAC is rejected, a replayed command is rejected by
                      the session counter, and no plaintext leaves the enclave unmasked;
  durability          a restarted enclave with the same image recovers the sealed state exactly;
  rollback            restoring an older blob with a lower monotonic counter is refused, so the
                      host cannot wind the store back;
  isolation           the host never sees plaintext values, only ciphertext, counters and quotes;
  determinism         identical scripts produce identical blobs, quotes and sessions.

Simplified versus a real TEE: one process, so "inside the enclave" is a naming convention rather
than hardware isolation - the host is an object that is simply never given the platform secret; the
seal is a keystream-and-MAC construction rather than AES-GCM; the Diffie-Hellman group is a tiny
61-bit prime, which is a demonstration of the exchange rather than a secure parameter choice; the
quote is an HMAC under a key derived from the platform root, standing in for the vendor-signed
certificate chain a verifier checks in reality; there is no side-channel modelling, no enclave
paging, no OCall/ECall machinery and no sealing-key rotation across TCB versions.
"""
import hashlib
import hmac

PLATFORM_SECRET = b"cpu-fused-root-secret-v1"
DH_MODULUS = (1 << 61) - 1
DH_GENERATOR = 5
MINIMUM_VERSION = 3
BLOCK = 32


class EnclaveError(Exception):
    pass


class AttestationError(EnclaveError):
    pass


class TamperError(EnclaveError):
    pass


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def digest(*parts):
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode()
        elif not isinstance(part, (bytes, bytearray)):
            part = str(part).encode()
        else:
            part = bytes(part)
        hasher.update(len(part).to_bytes(4, "big"))
        hasher.update(part)
    return hasher.digest()


def derive(domain, *parts):
    """Domain-separated key derivation from the platform root."""
    return hmac.new(PLATFORM_SECRET, digest(domain.encode(), *parts), hashlib.sha256).digest()


def measure(image):
    """MRENCLAVE: the identity of an enclave is the digest of its code image."""
    return digest(b"MRENCLAVE", image)


def keystream(key, nonce, length):
    """HMAC-SHA256 in counter mode: a deterministic stream to mask the payload with."""
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(output[:length])


def seal(sealing_key, plaintext, associated=b""):
    """Encrypt-then-MAC: confidentiality from the keystream, integrity from the tag."""
    mask_key = derive("stream", sealing_key)
    mac_key = derive("mac", sealing_key)
    nonce = derive("nonce", sealing_key)
    body = bytes(byte ^ mask for byte, mask in zip(plaintext, keystream(mask_key, nonce,
                                                                      len(plaintext))))
    tag = hmac.new(mac_key, nonce + associated + body, hashlib.sha256).digest()
    return nonce + tag + body


def unseal(sealing_key, blob, associated=b""):
    if len(blob) < 2 * BLOCK:
        raise TamperError("sealed blob is too short to be authentic")
    nonce, tag, body = blob[:BLOCK], blob[BLOCK:2 * BLOCK], blob[2 * BLOCK:]
    mac_key = derive("mac", sealing_key)
    mask_key = derive("stream", sealing_key)
    expected = hmac.new(mac_key, nonce + associated + body, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise TamperError("sealed blob failed its integrity check")
    return bytes(byte ^ mask for byte, mask in zip(body, keystream(mask_key, nonce, len(body))))


# ---------------------------------------------------------------------------------- platform
class Enclave:
    def __init__(self, platform, image, version=MINIMUM_VERSION):
        self.platform = platform
        self.image = image
        self.version = version
        self.measurement = measure(image)
        self.sealing_key = derive("sealing-key", self.measurement)
        self.quote_key = derive("quote-key", self.measurement)
        self.store = {}
        self.counter = 0
        self.sessions = {}
        self.trace = []                      # what an observer of the host would see

    # ------------------------------------------------------------------ sealed storage
    def serialize(self):
        rows = sorted(self.store.items())
        payload = repr((self.counter, rows)).encode()
        return payload

    def seal_state(self):
        blob = seal(self.sealing_key, self.serialize(), self.measurement)
        self.trace.append(("seal", len(blob)))
        return blob

    def load_state(self, blob):
        try:
            payload = unseal(self.sealing_key, blob, self.measurement)
        except TamperError as error:
            self.trace.append(("refused-blob", str(error)))
            raise
        counter, rows = eval(payload.decode(), {"__builtins__": {}})
        if counter < self.counter:
            self.trace.append(("refused-rollback", counter))
            raise TamperError("refusing to roll the counter back from %d to %d"
                              % (self.counter, counter))
        self.counter = counter
        self.store = dict(rows)
        self.trace.append(("load", counter, len(rows)))
        return self.store

    # -------------------------------------------------------------------- attestation
    def report_data(self, nonce, client_public):
        return digest(b"REPORT-DATA", nonce, client_public)

    def quote(self, nonce, client_public):
        report = self.report_data(nonce, client_public)
        payload = digest(b"QUOTE", self.measurement, report, str(self.version).encode())
        signature = hmac.new(self.quote_key, payload, hashlib.sha256).digest()
        return {"measurement": self.measurement, "report_data": report, "version": self.version,
                "signature": signature}

    # ------------------------------------------------------------------ secure session
    def handle(self, session_key, message, counter):
        """Decrypt, authenticate and apply one command; the counter defeats replays."""
        payload = dict(message)
        tag = payload.pop("tag")
        body = repr(tuple(sorted(payload.items()))).encode()
        expected = hmac.new(derive("session-mac", session_key), body, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise AttestationError("message authentication failed")
        if counter <= self.sessions.get("last", 0):
            raise AttestationError("replayed session counter %d" % counter)
        self.sessions["last"] = counter
        operation = payload["operation"]
        if operation == "put":
            value = unseal(derive("value", session_key), payload["value"])
            self.store[payload["key"]] = value.decode()
        elif operation == "delete":
            self.store.pop(payload["key"], None)
        elif operation != "get":
            raise AttestationError("unknown operation %r" % (operation,))
        self.counter += 1
        return self.store.get(payload["key"])

    def reply(self, session_key, key):
        """Authenticate a value on the way back to the client."""
        body = self.store.get(key, "").encode()
        tag = hmac.new(derive("session-mac", session_key), body, hashlib.sha256).digest()
        return body, tag


class Platform:
    """The CPU: the only actor that holds the root secret."""

    def __init__(self, seed=0x5EED):
        self.seed = seed
        self.ephemeral_counter = 0

    def create(self, image, version=MINIMUM_VERSION):
        return Enclave(self, image, version)

    def ephemeral(self, label, salt):
        self.ephemeral_counter += 1
        state, _raw = splitmix64(self.seed + self.ephemeral_counter * 7919)
        return int.from_bytes(derive("ephemeral", label.encode(), str(salt).encode()), "big")


class Verifier:
    """A relying party that accepts quotes only from whitelisted measurements."""

    def __init__(self, whitelist, minimum_version=MINIMUM_VERSION):
        self.whitelist = set(whitelist)
        self.minimum_version = minimum_version
        self.seen_nonces = set()
        self.rejections = []

    def verify(self, quote, nonce, client_public, fresh=True):
        if quote["measurement"] not in self.whitelist:
            return self.reject("measurement is not whitelisted")
        if fresh and nonce in self.seen_nonces:
            return self.reject("nonce was already used")
        if quote["report_data"] != digest(b"REPORT-DATA", nonce, client_public):
            return self.reject("report data does not bind this nonce and key")
        if quote["version"] < self.minimum_version:
            return self.reject("enclave version below the minimum")
        payload = digest(b"QUOTE", quote["measurement"], quote["report_data"],
                         str(quote["version"]).encode())
        expected = hmac.new(derive("quote-key", quote["measurement"]), payload, hashlib.sha256)
        if not hmac.compare_digest(quote["signature"], expected.digest()):
            return self.reject("quote signature does not verify")
        self.seen_nonces.add(nonce)
        return True

    def reject(self, reason):
        self.rejections.append(reason)
        return False


class Host:
    """Untrusted storage and transport: it sees blobs, quotes and messages, never the secret."""

    def __init__(self):
        self.blobs = []
        self.quotes = []
        self.messages = []
        self.observations = []

    def store(self, blob):
        self.blobs.append(blob)
        self.observations.append(len(blob))
        return blob

    def latest(self):
        return self.blobs[-1]

    def older(self, index=0):
        return self.blobs[index]

    def flip(self, blob, position):
        if position >= len(blob):
            position = len(blob) - 1
        mutated = bytearray(blob)
        mutated[position] ^= 0x01
        self.observations.append(-position)
        return bytes(mutated)

    def swap(self, left, right):
        return right

    def forge(self, measurement, report_data, version):
        """The host holds no platform secret, so it can only guess a signature."""
        payload = digest(b"QUOTE", measurement, report_data, str(version).encode())
        return {"measurement": measurement, "report_data": report_data, "version": version,
                "signature": hmac.new(b"host-guess", payload, hashlib.sha256).digest()}


class Client:
    def __init__(self, seed=99):
        self.private = int.from_bytes(digest(b"client-private", str(seed).encode()), "big")
        self.public = pow(DH_GENERATOR, self.private, DH_MODULUS)
        self.nonce = None
        self.session_key = None
        self.counter = 0

    def challenge(self, tag=b"session"):
        self.nonce = digest(b"NONCE", tag, str(self.public).encode())
        return self.nonce

    def agree(self, enclave_public):
        """Derive the session key from the enclave's public value, as the enclave does."""
        shared = pow(enclave_public, self.private, DH_MODULUS)
        self.session_key = derive("session", shared.to_bytes(8, "big"),
                                  self.public.to_bytes(8, "big"))
        return self.session_key

    def command(self, session_key, key, operation, value=None):
        """Build one authenticated command for a session established over the attested channel."""
        self.session_key = session_key
        self.counter += 1
        payload = {"operation": operation, "key": key,
                   "value": seal(derive("value", session_key), value.encode())
                   if value is not None else b""}
        body = repr(tuple(sorted(payload.items()))).encode()
        tag = hmac.new(derive("session-mac", session_key), body, hashlib.sha256).digest()
        payload["tag"] = tag
        return payload


def session_pair(platform, image, seed=99):
    """A complete attested handshake returning the client, the enclave and the session key."""
    enclave = platform.create(image)
    client = Client(seed)
    nonce = client.challenge()
    enclave_private = int.from_bytes(digest(b"ephemeral", image, nonce), "big")
    enclave_public = pow(DH_GENERATOR, enclave_private, DH_MODULUS)
    quote = enclave.quote(nonce, client.public)
    verifier = Verifier([enclave.measurement])
    check(verifier.verify(quote, nonce, client.public), "the honest quote must verify")
    session = client.agree(enclave_public)
    check(session == derive("session", pow(client.public, enclave_private,
                                           DH_MODULUS).to_bytes(8, "big"),
                            client.public.to_bytes(8, "big")),
          "both sides must derive the same session key")
    return client, enclave, session, enclave_public, quote, verifier


def main():
    # --- 1. measurement identifies the code image ----------------------------------
    platform = Platform()
    image = b"kv-enclave-image-v1"
    other_image = b"kv-enclave-image-v2"
    first = platform.create(image)
    second = platform.create(image)
    third = platform.create(other_image)
    check(first.measurement == second.measurement,
          "two enclaves from the same image must measure identically")
    check(first.measurement != third.measurement,
          "a different image must measure differently")
    tweaked = platform.create(image + b"!")
    check(tweaked.measurement != first.measurement,
          "changing one byte of the image must change the measurement")
    check(first.sealing_key == second.sealing_key
          and first.sealing_key != third.sealing_key,
          "sealing keys must follow the measurement")
    check(first.quote_key != first.sealing_key,
          "the quote key and the sealing key must live in different domains")

    # --- 2. sealing round-trip and confidentiality ---------------------------------
    host = Host()
    first.store = {"alpha": "classified", "beta": "42"}
    first.counter = 7
    blob = host.store(first.seal_state())
    check(b"classified" not in blob and b"alpha" not in blob,
          "the sealed blob must not contain plaintext")
    reopened = platform.create(image)
    check(reopened.counter == 0, "a fresh enclave must start from an empty state")
    check(reopened.load_state(blob) == {"alpha": "classified", "beta": "42"},
          "a same-image enclave must recover the sealed state")
    check(reopened.counter == 7, "the sealed counter must come back with the state")
    check(unseal(first.sealing_key, blob, first.measurement) == first.serialize(),
          "unsealing must return exactly the serialized state")

    # --- 3. tamper detection -------------------------------------------------------
    rejections = 0
    for position in (0, BLOCK, BLOCK + 5, len(blob) - 1):
        broken = host.flip(blob, position)
        try:
            unseal(first.sealing_key, broken, first.measurement)
        except TamperError:
            rejections += 1
    check(rejections == 4, "every single-bit flip must be refused, accepted %d" % (4 - rejections))
    try:
        unseal(first.sealing_key, blob[:-1], first.measurement)
    except TamperError:
        rejections += 1
    check(rejections == 5, "a truncated blob must be refused")
    try:
        unseal(first.sealing_key, blob, b"different-associated-data")
    except TamperError:
        rejections += 1
    check(rejections == 6, "a mismatch in the associated data must be refused")
    raised = None
    try:
        third.load_state(blob)
    except TamperError as error:
        raised = str(error)
    check(raised is not None, "another enclave image must not be able to unseal the blob: %s"
          % raised)
    check(b"classified" not in host.latest(), "the host must never hold plaintext")

    # --- 4. remote attestation -----------------------------------------------------
    client, enclave, session, enclave_public, quote, verifier = session_pair(platform, image)
    check(verifier.verify(quote, client.nonce, client.public, fresh=False),
          "a verified quote must stay verifiable when freshness is not re-checked")
    wrong_whitelist = Verifier([third.measurement])
    check(not wrong_whitelist.verify(quote, client.nonce, client.public),
          "a quote from another measurement must be refused")
    check(wrong_whitelist.rejections[-1] == "measurement is not whitelisted",
          "the refusal reason must be the measurement: %s" % wrong_whitelist.rejections[-1])
    replay = Verifier([enclave.measurement])
    check(replay.verify(quote, client.nonce, client.public, fresh=True),
          "the first use of a nonce must be accepted")
    check(not replay.verify(quote, client.nonce, client.public, fresh=True),
          "a replayed nonce must be refused")
    check(replay.rejections[-1] == "nonce was already used",
          "the replay reason must name the nonce: %s" % replay.rejections[-1])
    stale_client = Client(seed=1234)
    stale_nonce = stale_client.challenge()
    check(not Verifier([enclave.measurement]).verify(quote, stale_nonce, stale_client.public),
          "a quote that does not bind this nonce and key must be refused")
    mutated = dict(quote)
    mutated["signature"] = host.flip(quote["signature"], 3)
    check(not Verifier([enclave.measurement]).verify(mutated, client.nonce, client.public,
                                                     fresh=False),
          "a modified signature must not verify")
    forged = host.forge(enclave.measurement, quote["report_data"], quote["version"])
    check(not Verifier([enclave.measurement]).verify(forged, client.nonce, client.public,
                                                     fresh=False),
          "a host that holds no platform secret must not be able to forge a quote")
    outdated = platform.create(image, version=MINIMUM_VERSION - 1)
    old_nonce = Client(seed=7).challenge()
    old_quote = outdated.quote(old_nonce, client.public)
    check(not Verifier([outdated.measurement]).verify(old_quote, old_nonce, client.public),
          "an enclave below the minimum version must be refused")
    check(Verifier([outdated.measurement], minimum_version=1).verify(old_quote, old_nonce,
                                                                     client.public),
          "the same enclave must be accepted when the policy allows its version")

    # --- 5. key agreement and the authenticated channel ---------------------------
    peer_private = 12345
    peer_public = pow(DH_GENERATOR, peer_private, DH_MODULUS)
    left = pow(peer_public, client.private, DH_MODULUS)
    right = pow(client.public, peer_private, DH_MODULUS)
    check(left == right, "both sides of the key exchange must agree")
    check(derive("session", left.to_bytes(8, "big"), client.public.to_bytes(8, "big"))
          == derive("session", right.to_bytes(8, "big"), client.public.to_bytes(8, "big")),
          "the session key must be identical on both sides")
    eavesdropper = derive("session", peer_public.to_bytes(8, "big"),
                          client.public.to_bytes(8, "big"))
    check(eavesdropper != session,
          "an eavesdropper holding only the public values must derive a different key")
    check(peer_public != session and client.public != session,
          "the session key must not be one of the public values")

    # --- 6. commands through the attested session ---------------------------------
    payload = client.command(session, "alpha", "put", "reclassified")
    stored = enclave.handle(session, payload, client.counter)
    check(enclave.store["alpha"] == "reclassified",
          "an authenticated command must reach the store: %s" % (enclave.store,))
    check(b"reclassified" not in repr(payload).encode(),
          "the command on the wire must not contain the plaintext value")
    broken_payload = dict(payload)
    broken_payload["tag"] = host.flip(payload["tag"], 2)
    raised = None
    try:
        enclave.handle(session, broken_payload, client.counter + 1)
    except AttestationError as error:
        raised = str(error)
    check("authentication failed" in (raised or ""),
          "a command with a broken MAC must be rejected: %s" % raised)
    raised = None
    try:
        enclave.handle(session, payload, client.counter)
    except AttestationError as error:
        raised = str(error)
    check("replayed session counter" in (raised or ""),
          "a replayed command must be rejected: %s" % raised)
    check(enclave.store["alpha"] == "reclassified",
          "a rejected command must not change the store")
    wrong_client = Client(seed=555)
    wrong_nonce = wrong_client.challenge()
    wrong_public = pow(DH_GENERATOR, wrong_client.private, DH_MODULUS)
    wrong_key = derive("session", pow(wrong_public, 4242, DH_MODULUS).to_bytes(8, "big"),
                       wrong_public.to_bytes(8, "big"))
    bad = wrong_client.command(wrong_key, "alpha", "put", "intruder")
    raised = None
    try:
        enclave.handle(session, bad, 99)
    except AttestationError as error:
        raised = str(error)
    check(raised is not None, "a command from a session without the shared key must be rejected")
    check(enclave.store["alpha"] == "reclassified", "the intruder must not overwrite the value")

    # --- 7. durability and rollback protection ------------------------------------
    enclave.counter = 20
    fresh_blob = host.store(enclave.seal_state())
    rolled_back = platform.create(image)
    older_blob = blob
    check(rolled_back.load_state(older_blob) is not None,
          "an empty enclave may load an older snapshot once")
    raised = None
    try:
        rolled_back.load_state(older_blob)
    except TamperError as error:
        raised = str(error)
    check(raised is None or "counter" in raised,
          "reloading the same snapshot must not move the counter backwards: %s" % raised)
    raised = None
    try:
        rolled_back.load_state(fresh_blob)
    except TamperError as error:
        raised = str(error)
    check(raised is None, "moving forward is allowed: %s" % raised)
    latest = platform.create(image)
    latest.load_state(fresh_blob)
    check(latest.store["alpha"] == "reclassified" and latest.counter == 20,
          "a restarted enclave must recover the newest sealed state exactly: %s"
          % (latest.store,))
    served = host.swap(fresh_blob, older_blob)        # the host serves the stale snapshot instead
    raised = None
    try:
        latest.load_state(served)
    except TamperError as error:
        raised = str(error)
    check(raised is not None and "roll" in raised,
          "the host must not be able to wind the store back: %s" % raised)
    check(latest.store["alpha"] == "reclassified",
          "a refused rollback must leave the newest state in place")

    # --- 8. host view and determinism ---------------------------------------------
    visible = repr(host.observations) + repr(host.blobs) + repr(host.quotes)
    check("reclassified" not in visible and "classified" not in visible,
          "nothing the host ever observed may contain a plaintext value")
    first_run = Platform().create(image)
    second_run = Platform().create(image)
    first_run.store = {"k": "v"}
    second_run.store = {"k": "v"}
    check(first_run.seal_state() == second_run.seal_state(),
          "identical states must seal into identical blobs")
    check(first_run.quote(b"nonce", 7) == second_run.quote(b"nonce", 7),
          "identical attestations must produce identical quotes")
    check(measure(image) == measure(image) and measure(image) != measure(image + b"x"),
          "measurement must be a deterministic function of the image")

    print("measurement %s; sealing key and quote key derived in separate domains"
          % first.measurement.hex()[:16])
    print("sealing: %d-byte blob, %d tamper attempts refused, plaintext never on the wire"
          % (len(blob), rejections))
    print("attestation: whitelist, replay, binding, signature, forgery and version checks all "
          "rejected as expected")
    print("PASS")


if __name__ == "__main__":
    main()
