/** Task 666 - Implement conflict-free replicated JSON structure (complex CRDT). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A replicated JSON document built from two CRDTs glued together by paths:
 *
 *   maps   - every (path, key) slot is a last-writer-wins register keyed by the stamp
 *            (sequence, replica); a delete is just a tombstone written through the same register, so
 *            set/delete and set/set conflicts resolve identically on every replica;
 *   arrays - an RGA list: each element carries the id of the element it was inserted after, so
 *            concurrent inserts at one index are ordered by descending (sequence, replica) and every
 *            replica agrees; removal is an LWW tombstone per element, and an element inserted after a
 *            removed element still has a well defined place.
 *
 * Objects are implicit in the paths ("user.name"), so a child operation is causally dependent on the
 * register that creates its parent: an operation whose parent is unknown is buffered and retried, which
 * makes delivery order irrelevant.
 *
 * Verified: three replicas converge for several delivery permutations and match a hand-written JSON
 * rendering; redelivery is idempotent; concurrent sets of one key resolve by stamp everywhere; a delete
 * beats/looses against a concurrent set according to its stamp; nested objects and arrays materialise
 * as expected; concurrent array insertions at one index agree; removing an element hides it while the
 * elements anchored to it survive; an operation for an unknown parent buffers and then applies; and
 * invalid paths, empty keys and out-of-range indices are rejected.
 */
import java.util.*;

final class JsonEntry {
    final int nodeId;
    final long stamp;
    final boolean deleted;

    JsonEntry(int nodeId, long stamp, boolean deleted) {
        this.nodeId = nodeId;
        this.stamp = stamp;
        this.deleted = deleted;
    }
}

final class ArrayElement {
    final int id;
    final int replica;
    final int sequence;
    final String value;
    final int origin;
    final int depth;

    ArrayElement(int id, int replica, int sequence, String value, int origin, int depth) {
        this.id = id;
        this.replica = replica;
        this.sequence = sequence;
        this.value = value;
        this.origin = origin;
        this.depth = depth;
    }
}

final class JsonOp {
    static final int SET = 0, DELETE = 1, ARRAY_INSERT = 2, ARRAY_REMOVE = 3;
    final int type;
    final int replica;
    final int sequence;
    final String path;
    final String key;
    final String value;
    final int anchor;
    final int[] targets;

    JsonOp(int type, int replica, int sequence, String path, String key, String value, int anchor, int[] targets) {
        this.type = type;
        this.replica = replica;
        this.sequence = sequence;
        this.path = path;
        this.key = key;
        this.value = value;
        this.anchor = anchor;
        this.targets = targets;
    }

    static JsonOp set(int replica, int sequence, String path, String key, String value) {
        if (key.isEmpty()) throw new IllegalArgumentException("empty key");
        if (path.startsWith(".") || path.endsWith(".") || path.contains("..")) {
            throw new IllegalArgumentException("malformed path: " + path);
        }
        return new JsonOp(SET, replica, sequence, path, key, value, -1, new int[0]);
    }

    static JsonOp delete(int replica, int sequence, String path, String key) {
        return new JsonOp(DELETE, replica, sequence, path, key, "", -1, new int[0]);
    }

    static JsonOp arrayInsert(int replica, int sequence, String path, int anchor, String value) {
        if (anchor < -1) throw new IllegalArgumentException("bad anchor");
        return new JsonOp(ARRAY_INSERT, replica, sequence, path, "", value, anchor, new int[0]);
    }

    static JsonOp arrayRemove(int replica, int sequence, String path, int[] targets) {
        if (targets.length == 0) throw new IllegalArgumentException("nothing to remove");
        return new JsonOp(ARRAY_REMOVE, replica, sequence, path, "", "", -1, targets);
    }

    long stamp() { return sequence * 1000L + replica + 1; }

    int nodeId() { return replica * 1_000_000 + sequence; }

    String describe() {
        switch (type) {
            case SET: return "set " + path + "." + key + "=" + value;
            case DELETE: return "del " + path + "." + key;
            case ARRAY_INSERT: return "push " + path + " after " + anchor + "=" + value;
            default: return "remove " + path + " " + Arrays.toString(targets);
        }
    }
}

final class JsonCrdt {
    final int replicaId;
    final Map<String, Map<String, JsonEntry>> registers = new LinkedHashMap<>();
    final Map<Integer, String> scalars = new LinkedHashMap<>();
    final Map<String, List<ArrayElement>> arrays = new LinkedHashMap<>();
    final Map<Integer, Long> removed = new LinkedHashMap<>();       // element id -> tombstone stamp
    final Map<String, JsonOp> buffered = new LinkedHashMap<>();
    private int sequenceCounter;

    JsonCrdt(int replicaId) { this.replicaId = replicaId; }

    private static String childPath(String path, String key) {
        return path.isEmpty() ? key : path + "." + key;
    }

    boolean hasPath(String path) {
        if (path.isEmpty()) return true;
        int separator = path.lastIndexOf('.');
        String parent = separator < 0 ? "" : path.substring(0, separator);
        String key = separator < 0 ? path : path.substring(separator + 1);
        return isLive(parent, key);
    }

    private boolean isLive(String parent, String key) {
        Map<String, JsonEntry> entries = registers.get(parent);
        if (entries == null) return false;
        JsonEntry entry = entries.get(key);
        return entry != null && !entry.deleted;
    }

    // ------------------------------------------------------------------ local edits
    JsonOp set(String path, String key, String value) {
        if (!hasPath(path)) throw new IllegalArgumentException("unknown parent path: " + path);
        JsonOp op = JsonOp.set(replicaId, sequenceCounter++, path, key, value);
        deliver(op);
        return op;
    }

    JsonOp delete(String path, String key) {
        if (!hasPath(path)) throw new IllegalArgumentException("unknown parent path: " + path);
        JsonOp op = JsonOp.delete(replicaId, sequenceCounter++, path, key);
        deliver(op);
        return op;
    }

    JsonOp arrayInsert(String path, int index, String value) {
        if (!hasPath(path)) throw new IllegalArgumentException("unknown parent path: " + path);
        List<ArrayElement> elements = visible(path);
        if (index < 0 || index > elements.size()) throw new IllegalArgumentException("index out of range: " + index);
        int anchor = index == 0 ? -1 : elements.get(index - 1).id;
        JsonOp op = JsonOp.arrayInsert(replicaId, sequenceCounter++, path, anchor, value);
        deliver(op);
        return op;
    }

    JsonOp arrayRemove(String path, int index) {
        List<ArrayElement> elements = visible(path);
        if (index < 0 || index >= elements.size()) throw new IllegalArgumentException("index out of range: " + index);
        JsonOp op = JsonOp.arrayRemove(replicaId, sequenceCounter++, path, new int[]{elements.get(index).id});
        deliver(op);
        return op;
    }

    List<ArrayElement> visible(String path) {
        List<ArrayElement> out = new ArrayList<>();
        for (ArrayElement element : arrays.getOrDefault(path, List.of())) {
            if (!removed.containsKey(element.id)) out.add(element);
        }
        return out;
    }

    // ------------------------------------------------------------------ replication
    boolean deliver(JsonOp op) {
        if (buffered.containsKey(bufferKey(op))) return true;        // already waiting
        if (!ready(op)) {
            buffered.put(bufferKey(op), op);
            return false;
        }
        apply(op);
        boolean progress = true;
        while (progress) {
            progress = false;
            for (String key : new ArrayList<>(buffered.keySet())) {
                JsonOp waiting = buffered.get(key);
                if (ready(waiting)) {
                    buffered.remove(key);
                    apply(waiting);
                    progress = true;
                }
            }
        }
        return true;
    }

    int bufferedCount() { return buffered.size(); }

    private static String bufferKey(JsonOp op) {
        return op.type + "/" + op.replica + "/" + op.sequence;
    }

    private boolean ready(JsonOp op) {
        if (!hasPath(op.path)) return false;
        if (op.type == JsonOp.ARRAY_INSERT) {
            return op.anchor < 0 || indexOfElement(op.path, op.anchor) >= 0;
        }
        if (op.type == JsonOp.ARRAY_REMOVE) {
            for (int target : op.targets) if (indexOfElement(op.path, target) < 0) return false;
        }
        return true;
    }

    private int indexOfElement(String path, int elementId) {
        List<ArrayElement> elements = arrays.get(path);
        if (elements == null) return -1;
        for (int i = 0; i < elements.size(); i++) if (elements.get(i).id == elementId) return i;
        return -1;
    }

    private static boolean greaterSibling(ArrayElement element, int sequence, int replica) {
        if (element.sequence != sequence) return element.sequence > sequence;
        return element.replica > replica;
    }

    private int insertionPoint(String path, int anchor, int sequence, int replica) {
        List<ArrayElement> elements = arrays.get(path);
        int anchorIndex = anchor < 0 ? -1 : indexOfElement(path, anchor);
        int depth = anchor < 0 ? 0 : elements.get(anchorIndex).depth + 1;
        int at = anchor < 0 ? 0 : anchorIndex + 1;
        while (at < elements.size()) {
            ArrayElement candidate = elements.get(at);
            if (candidate.depth < depth) break;
            if (candidate.depth == depth) {
                if (greaterSibling(candidate, sequence, replica)) {
                    at++;
                    continue;
                }
                break;
            }
            at++;
        }
        return at;
    }

    private void apply(JsonOp op) {
        switch (op.type) {
            case JsonOp.SET: {
                Map<String, JsonEntry> entries = registers.computeIfAbsent(op.path, key -> new LinkedHashMap<>());
                JsonEntry existing = entries.get(op.key);
                if (existing == null || op.stamp() > existing.stamp) {
                    entries.put(op.key, new JsonEntry(op.nodeId(), op.stamp(), false));
                }
                scalars.putIfAbsent(op.nodeId(), op.value);
                break;
            }
            case JsonOp.DELETE: {
                Map<String, JsonEntry> entries = registers.computeIfAbsent(op.path, key -> new LinkedHashMap<>());
                JsonEntry existing = entries.get(op.key);
                if (existing == null || op.stamp() > existing.stamp) {
                    entries.put(op.key, new JsonEntry(op.nodeId(), op.stamp(), true));
                }
                break;
            }
            case JsonOp.ARRAY_INSERT: {
                List<ArrayElement> elements = arrays.computeIfAbsent(op.path, key -> new ArrayList<>());
                int id = op.nodeId();
                if (indexOfElement(op.path, id) >= 0) return;        // idempotent
                int depth = op.anchor < 0 ? 0 : elements.get(indexOfElement(op.path, op.anchor)).depth + 1;
                int at = insertionPoint(op.path, op.anchor, op.sequence, op.replica);
                elements.add(at, new ArrayElement(id, op.replica, op.sequence, op.value, op.anchor, depth));
                break;
            }
            default: {
                for (int target : op.targets) {
                    Long existing = removed.get(target);
                    if (existing == null || op.stamp() > existing) removed.put(target, op.stamp());
                }
                break;
            }
        }
    }

    // ------------------------------------------------------------------ materialisation
    String render() { return renderObject(""); }

    private String renderObject(String path) {
        Map<String, JsonEntry> entries = registers.get(path);
        TreeMap<String, String> rendered = new TreeMap<>();
        if (entries != null) {
            for (Map.Entry<String, JsonEntry> entry : entries.entrySet()) {
                if (entry.getValue().deleted) continue;
                String child = childPath(path, entry.getKey());
                rendered.put(entry.getKey(), renderValue(child, entry.getValue().nodeId));
            }
        }
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, String> entry : rendered.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(entry.getKey()).append("\":").append(entry.getValue());
        }
        return sb.append('}').toString();
    }

    private String renderValue(String path, int nodeId) {
        if (arrays.containsKey(path)) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (ArrayElement element : visible(path)) {
                if (!first) sb.append(',');
                first = false;
                sb.append('"').append(element.value).append('"');
            }
            return sb.append(']').toString();
        }
        if (registers.containsKey(path)) return renderObject(path);
        String scalar = scalars.get(nodeId);
        return scalar == null ? "\"\"" : "\"" + scalar + "\"";
    }

    boolean structureIsConsistent() {
        Set<Integer> elementIds = new HashSet<>();
        for (List<ArrayElement> elements : arrays.values()) {
            for (ArrayElement element : elements) if (!elementIds.add(element.id)) return false;
        }
        for (Map<String, JsonEntry> entries : registers.values()) {
            for (JsonEntry entry : entries.values()) {
                if (!entry.deleted && !scalars.containsKey(entry.nodeId)) return false;
            }
        }
        for (int target : removed.keySet()) if (!elementIds.contains(target)) return false;
        return true;
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static List<JsonOp> sessionOps() {
        List<JsonOp> ops = new ArrayList<>();
        ops.add(JsonOp.set(0, 0, "", "user", "object"));
        ops.add(JsonOp.set(0, 1, "user", "name", "ada"));
        ops.add(JsonOp.set(1, 0, "", "user", "object"));
        ops.add(JsonOp.set(1, 1, "user", "role", "admin"));
        ops.add(JsonOp.set(0, 2, "user", "name", "grace"));
        ops.add(JsonOp.set(1, 2, "", "tags", "array"));
        JsonOp yElement = JsonOp.arrayInsert(2, 0, "tags", -1, "y");
        ops.add(yElement);
        ops.add(JsonOp.arrayInsert(1, 3, "tags", yElement.nodeId(), "z"));
        ops.add(JsonOp.arrayInsert(1, 4, "tags", -1, "x"));
        ops.add(JsonOp.set(2, 0, "user", "role", "guest"));
        ops.add(JsonOp.delete(2, 1, "user", "role"));
        return ops;
    }

    public static void main(String[] args) {
        // --- building a nested document locally --------------------------------
        JsonCrdt local = new JsonCrdt(0);
        local.set("", "user", "object");
        local.set("user", "name", "ada");
        local.set("user", "age", "36");
        local.set("", "tags", "array");
        local.arrayInsert("tags", 0, "red");
        local.arrayInsert("tags", 1, "blue");
        check(local.render().equals("{\"tags\":[\"red\",\"blue\"],\"user\":{\"age\":\"36\",\"name\":\"ada\"}}"),
                "nested rendering, got " + local.render());
        local.delete("user", "age");
        check(local.render().equals("{\"tags\":[\"red\",\"blue\"],\"user\":{\"name\":\"ada\"}}"),
                "a deleted key disappears, got " + local.render());
        local.arrayRemove("tags", 0);
        check(local.render().equals("{\"tags\":[\"blue\"],\"user\":{\"name\":\"ada\"}}"),
                "a removed element disappears, got " + local.render());
        check(local.structureIsConsistent(), "the structure invariant holds");

        // --- arrays: insert in the middle and anchored to a removed element ----
        JsonCrdt anchored = new JsonCrdt(0);
        anchored.set("", "list", "array");
        anchored.arrayInsert("list", 0, "a");
        JsonOp second = anchored.arrayInsert("list", 1, "b");
        JsonOp third = anchored.arrayInsert("list", 2, "c");
        check(third.nodeId() != second.nodeId(), "each element gets its own id");
        check(anchored.render().equals("{\"list\":[\"a\",\"b\",\"c\"]}"), "three appends, got " + anchored.render());
        anchored.arrayRemove("list", 1);
        check(anchored.render().equals("{\"list\":[\"a\",\"c\"]}"), "the middle element is gone");
        JsonOp resurrect = JsonOp.arrayInsert(5, 0, "list", second.nodeId(), "x");
        anchored.deliver(resurrect);
        check(anchored.render().equals("{\"list\":[\"a\",\"c\",\"x\"]}"),
                "an element anchored to a removed element still lands in order, got " + anchored.render());
        check(anchored.visible("list").size() == 3, "three elements are visible");

        // --- concurrent sets resolve by stamp ---------------------------------
        JsonCrdt one = new JsonCrdt(0);
        JsonCrdt two = new JsonCrdt(1);
        JsonOp create = JsonOp.set(0, 0, "", "user", "object");
        one.deliver(create);
        two.deliver(create);
        JsonOp setNameA = JsonOp.set(0, 1, "user", "name", "ada");
        JsonOp setNameB = JsonOp.set(1, 1, "user", "name", "grace");
        one.deliver(setNameA);
        one.deliver(setNameB);
        two.deliver(setNameB);
        two.deliver(setNameA);
        check(one.render().equals(two.render()), "concurrent sets converge");
        check(one.render().equals("{\"user\":{\"name\":\"grace\"}}"),
                "the higher (sequence, replica) stamp wins, got " + one.render());

        // --- delete against a concurrent set -----------------------------------
        JsonCrdt deleting = new JsonCrdt(0);
        JsonCrdt setting = new JsonCrdt(1);
        JsonOp root = JsonOp.set(0, 0, "", "cfg", "object");
        JsonOp setKey = JsonOp.set(0, 1, "cfg", "mode", "fast");
        JsonOp deleteKey = JsonOp.delete(0, 2, "cfg", "mode");
        deleting.deliver(root);
        deleting.deliver(setKey);
        deleting.deliver(deleteKey);
        setting.deliver(root);
        setting.deliver(deleteKey);
        setting.deliver(setKey);
        check(deleting.render().equals(setting.render()), "delete and set converge regardless of order");
        check(deleting.render().equals("{\"cfg\":{}}"), "the later delete wins, got " + deleting.render());
        check(deleting.scalars.size() >= 1, "the losing scalar is still recorded internally");

        // --- buffering a causally late operation ------------------------------
        JsonCrdt ordered = new JsonCrdt(0);
        JsonCrdt shuffled = new JsonCrdt(0);
        JsonOp makeUser = JsonOp.set(0, 0, "", "user", "object");
        JsonOp nameUser = JsonOp.set(0, 1, "user", "name", "ada");
        ordered.deliver(makeUser);
        ordered.deliver(nameUser);
        shuffled.deliver(nameUser);
        check(shuffled.bufferedCount() == 1, "a child of an unknown parent is buffered");
        check(shuffled.render().equals("{}"), "and nothing is materialised yet");
        shuffled.deliver(makeUser);
        check(shuffled.bufferedCount() == 0, "the buffer drains when the parent arrives");
        check(shuffled.render().equals(ordered.render()), "out of order delivery matches in order delivery");

        // --- idempotent redelivery ---------------------------------------------
        String beforeRedelivery = ordered.render();
        for (int repeat = 0; repeat < 3; repeat++) {
            ordered.deliver(makeUser);
            ordered.deliver(nameUser);
        }
        check(ordered.render().equals(beforeRedelivery), "redelivery changes nothing");

        // --- three replicas converge over permutations -------------------------
        List<JsonOp> ops = sessionOps();
        JsonCrdt canonical = new JsonCrdt(0);
        for (JsonOp op : ops) canonical.deliver(op);
        String canonicalJson = canonical.render();
        check(canonical.bufferedCount() == 0, "the canonical run drained its buffer");
        check(canonical.structureIsConsistent(), "the canonical document is consistent");
        check(canonicalJson.contains("\"name\":\"grace\""), "the newest name wins, got " + canonicalJson);
        check(!canonicalJson.contains("\"role\""), "the deleted key is gone, got " + canonicalJson);
        check(canonicalJson.contains("\"tags\":[\"x\",\"y\",\"z\"]"),
                "the array materialises as [x, y, z], got " + canonicalJson);

        int[][] permutations = {
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
                {3, 7, 1, 9, 0, 5, 2, 8, 4, 6, 10},
                {5, 0, 8, 2, 6, 1, 9, 3, 7, 4, 10},
                {2, 4, 9, 1, 7, 3, 5, 0, 8, 6, 10}};
        for (int p = 0; p < permutations.length; p++) {
            for (int replica = 0; replica < 3; replica++) {
                JsonCrdt editor = new JsonCrdt(replica);
                for (int step = 0; step < ops.size(); step++) editor.deliver(ops.get(permutations[p][step]));
                check(editor.render().equals(canonicalJson),
                        "permutation " + p + " replica " + replica + " converges to " + canonicalJson
                                + " but got " + editor.render());
                check(editor.bufferedCount() == 0, "permutation " + p + " left nothing buffered");
                check(editor.structureIsConsistent(), "permutation " + p + " stays consistent");
            }
        }

        // --- two replicas editing the same array concurrently ------------------
        JsonCrdt left = new JsonCrdt(0);
        JsonCrdt right = new JsonCrdt(1);
        JsonOp tagArray = JsonOp.set(0, 0, "", "tags", "array");
        left.deliver(tagArray);
        right.deliver(tagArray);
        JsonOp leftPush = JsonOp.arrayInsert(0, 1, "tags", -1, "L");
        JsonOp rightPush = JsonOp.arrayInsert(1, 0, "tags", -1, "R");
        left.deliver(leftPush);
        left.deliver(rightPush);
        right.deliver(rightPush);
        right.deliver(leftPush);
        check(left.render().equals(right.render()), "concurrent array inserts at the head converge");
        check(left.render().equals("{\"tags\":[\"L\",\"R\"]}"),
                "the higher (sequence, replica) element goes first, got " + left.render());
        JsonOp leftRemove = JsonOp.arrayRemove(0, 2, "tags", new int[]{leftPush.nodeId()});
        right.deliver(leftRemove);
        left.deliver(leftRemove);
        check(left.render().equals(right.render()), "removal converges");
        check(left.render().equals("{\"tags\":[\"R\"]}"), "the removed element disappears everywhere, got "
                + left.render());

        // --- deterministic rebuild --------------------------------------------
        JsonCrdt rebuilt = new JsonCrdt(0);
        for (JsonOp op : ops) rebuilt.deliver(op);
        check(rebuilt.render().equals(canonicalJson), "rebuilding from the same operations is deterministic");

        // --- rejected input ----------------------------------------------------
        boolean rejected = false;
        try {
            JsonOp.set(0, 0, "", "", "value");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty key must be rejected");
        rejected = false;
        try {
            JsonOp.set(0, 0, "user.", "name", "ada");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a malformed path must be rejected");
        rejected = false;
        try {
            new JsonCrdt(0).set("user", "name", "ada");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "setting under an unknown parent must be rejected");
        rejected = false;
        try {
            JsonCrdt small = new JsonCrdt(0);
            small.set("", "tags", "array");
            small.arrayInsert("tags", 4, "x");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an out-of-range array index must be rejected");
        rejected = false;
        try {
            JsonCrdt small = new JsonCrdt(0);
            small.set("", "tags", "array");
            small.arrayRemove("tags", 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "removing from an empty array must be rejected");

        System.out.println("PASS");
    }
}
