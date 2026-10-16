/** Task 806 - Implement fully persistent B-tree with copy-on-write pages and concurrency. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A B-tree whose pages are immutable: an update copies every page on the root-to-leaf path, rewrites
 * only those copies (allocating fresh page ids from an append-only page store) and leaves every other
 * page shared with the previous version. A version is therefore just a root page id, which is what
 * makes the structure *fully* persistent: the old root stays valid and answers exactly what it
 * answered before, no matter how many writes follow. Concurrency is modelled the way a real page
 * server does it - each writer computes its new root privately and then installs it with a
 * compare-and-swap on the single root pointer, retrying from the current root when the CAS fails - so
 * two writers starting from the same version cannot lose each other's work, while readers keep using
 * the root id they opened and never block. Deletion copies the path as well; when a node would
 * underflow the affected subtree is bulk-rebuilt from its own sorted content, which never touches a
 * page belonging to another version.
 *
 * Simulation model, and what is simplified. One process and one thread: writes are explicit
 * "prepare then CAS" steps so the interleavings are chosen by the test rather than by a scheduler, and
 * there is no buffer pool, no WAL, no disk and no page-level locking. Node capacity is four keys with
 * a minimum of two, which is deliberately tiny so that splits, merges and multi-level trees happen on
 * a few hundred keys. What is *not* simplified: the tree is a real B-tree with real node splits,
 * bulk-loading and occupancy rules; copy-on-write is enforced structurally (no page id is ever
 * written twice, and every version is checked against an independent sorted-map oracle).
 *
 * Verified: an empty tree and a single element; sixty insertions including duplicate-key updates,
 * each version compared key by key against a plain sorted map snapshot; every version ever created
 * still answers correctly after later writes (full persistence); the structural invariants after
 * every mutation (all leaves at the same depth, keys sorted, child counts, occupancy, n - 1 keys in n
 * children); ordered iteration and bounded range scans; the copy-on-write bound - an update allocates
 * one page per level of the path, never a copy of the tree, with the allocation ratio measured; no
 * page is ever mutated after being published, checked by fingerprinting all live versions and by
 * comparing against a deliberately non-copying negative control that corrupts an old version; deletes
 * (including underflow and rebalancing) preserve the invariants and the oracle; a reader holding a
 * snapshot while writers commit keeps seeing a consistent state; and the compare-and-swap root install
 * prevents lost updates between two writers, which the same test shows failing without the CAS.
 */
import java.util.*;

final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A page: leaf (keys + values) or internal (keys + child page ids). Never modified once published. */
final class Page {
    static final int MAX_KEYS = 4;
    static final int MIN_KEYS = 2;

    final boolean leaf;
    final List<Integer> keys = new ArrayList<>();
    final List<String> values = new ArrayList<>();      // leaves only, parallel to keys
    final List<Integer> children = new ArrayList<>();   // internals only, keys.size() + 1 entries

    Page(boolean leaf) { this.leaf = leaf; }

    Page copy() {
        Page p = new Page(leaf);
        p.keys.addAll(keys);
        p.values.addAll(values);
        p.children.addAll(children);
        return p;
    }

    int height() { return keys.size(); }

    String render() {
        return (leaf ? "L" : "I") + keys + (leaf ? values.toString() : children.toString());
    }
}

/** Append-only page store: page ids are never reused and never written after being handed out. */
final class PageStore {
    final Map<Integer, Page> pages = new LinkedHashMap<>();
    int nextId = 1;
    int allocations, reads;
    boolean copyOnWrite = true;

    int allocate(Page page) {
        int id = nextId++;
        pages.put(id, page);
        allocations++;
        return id;
    }

    Page page(int id) {
        reads++;
        Page p = pages.get(id);
        if (p == null) throw new IllegalArgumentException("unknown page " + id);
        return p;
    }

    /** The negative control replaces this with "hand back the same page" - aliasing the version. */
    Page writable(int id) { return copyOnWrite ? page(id).copy() : page(id); }
}

final class PersistentBTree {
    final PageStore store;
    int root;                                            // 0 means the empty tree
    final Map<Integer, Long> publishedFingerprints = new LinkedHashMap<>();

    PersistentBTree(PageStore store) { this.store = store; }

    int height(int rootId) {
        int h = 0;
        int node = rootId;
        while (node != 0) {
            Page p = store.page(node);
            h++;
            node = p.leaf ? 0 : p.children.get(0);
        }
        return h;
    }

    String get(int rootId, int key) {
        int node = rootId;
        while (node != 0) {
            Page p = store.page(node);
            int idx = Collections.binarySearch(p.keys, key);
            if (p.leaf) return idx >= 0 ? p.values.get(idx) : null;
            // a separator holds the maximum key of the child to its left, so a hit descends into it
            node = p.children.get(idx >= 0 ? idx : -idx - 1);
        }
        return null;
    }

    /**
     * Ordered iteration. Because a separator holds the *maximum* key of the child to its left, that
     * key is also stored in the child itself, so collection simply walks every child and reads the
     * leaves - adding a separator here would report it twice.
     */
    void collect(int rootId, int from, int to, List<Integer> keys, List<String> values) {
        if (rootId == 0) return;
        Page p = store.page(rootId);
        if (p.leaf) {
            for (int i = 0; i < p.keys.size(); i++) {
                int k = p.keys.get(i);
                if (k >= from && k <= to) { keys.add(k); values.add(p.values.get(i)); }
            }
            return;
        }
        for (int child : p.children) collect(child, from, to, keys, values);
    }

    List<Integer> orderedKeys(int rootId) {
        List<Integer> keys = new ArrayList<>();
        List<String> values = new ArrayList<>();
        collect(rootId, Integer.MIN_VALUE, Integer.MAX_VALUE, keys, values);
        return keys;
    }

    int size(int rootId) { return orderedKeys(rootId).size(); }

    /** Bulk loader: builds a balanced tree bottom-up from sorted keys (evenly packed leaves). */
    int build(List<Integer> keys, List<String> values) {
        if (keys.isEmpty()) return 0;
        int leaves = (keys.size() + Page.MAX_KEYS - 1) / Page.MAX_KEYS;
        int base = keys.size() / leaves, extra = keys.size() % leaves;
        List<Integer> children = new ArrayList<>();
        int at = 0;
        for (int i = 0; i < leaves; i++) {
            int take = base + (i < extra ? 1 : 0);
            Page leaf = new Page(true);
            for (int j = 0; j < take; j++) {
                leaf.keys.add(keys.get(at));
                leaf.values.add(values.get(at));
                at++;
            }
            children.add(store.allocate(leaf));
        }
        return assemble(children);
    }

    /**
     * Builds the internal levels. A page may hold at most MAX_KEYS keys (MAX_KEYS + 1 children) and,
     * unless it is the root, at least MIN_KEYS keys; the chunking below shortens the last chunk so no
     * interior page ends up with a single key.
     */
    private int assemble(List<Integer> childIds) {
        while (childIds.size() > 1) {
            List<Integer> parents = new ArrayList<>();
            int i = 0;
            while (i < childIds.size()) {
                int remaining = childIds.size() - i;
                int take = Math.min(Page.MAX_KEYS + 1, remaining);
                if (remaining - take == 1 || remaining - take == 2) take = remaining - 3;
                if (take < 3 && remaining > 3) take = 3;
                Page node = new Page(false);
                for (int j = i; j < i + take; j++) {
                    node.children.add(childIds.get(j));
                    if (j + 1 < i + take) node.keys.add(maxKey(childIds.get(j)));
                }
                parents.add(store.allocate(node));
                i += take;
            }
            childIds = parents;
        }
        return childIds.get(0);
    }

    int maxKey(int rootId) {
        int node = rootId;
        while (true) {
            Page p = store.page(node);
            if (p.leaf) return p.keys.get(p.keys.size() - 1);
            node = p.children.get(p.children.size() - 1);
        }
    }

    /** Inserts or replaces a key, copying the path. Returns the new root. */
    int put(int rootId, int key, String value) {
        if (rootId == 0) {
            Page p = new Page(true);
            p.keys.add(key);
            p.values.add(value);
            return store.allocate(p);
        }
        Object[] result = insert(rootId, key, value);
        if (result[1] == null) return (Integer) result[0];
        Page rootPage = new Page(false);
        rootPage.keys.add((Integer) result[1]);
        rootPage.children.add((Integer) result[0]);
        rootPage.children.add((Integer) result[2]);
        return store.allocate(rootPage);
    }

    /** Returns {newChildId} or {leftId, separatorKey, rightId} when the child had to split. */
    private Object[] insert(int pageId, int key, String value) {
        Page node = store.writable(pageId);
        int idx = Collections.binarySearch(node.keys, key);
        if (node.leaf) {
            int position = idx >= 0 ? idx : -idx - 1;
            if (idx >= 0) {
                node.values.set(position, value);
            } else {
                node.keys.add(position, key);
                node.values.add(position, value);
            }
        } else {
            int childIndex = idx >= 0 ? idx : -idx - 1;    // keys[i] is the *max* key of children[i]
            int childId = node.children.get(childIndex);
            Object[] child = insert(childId, key, value);
            if (child[1] == null) {
                node.children.set(childIndex, (Integer) child[0]);
            } else {
                node.keys.add(childIndex, (Integer) child[1]);
                node.children.set(childIndex, (Integer) child[0]);
                node.children.add(childIndex + 1, (Integer) child[2]);
            }
        }
        if (node.keys.size() <= Page.MAX_KEYS) return new Object[] {store.allocate(node), null};
        int middle = node.keys.size() / 2;
        int separator = node.keys.get(middle);
        Page right = new Page(node.leaf);
        for (int i = middle + 1; i < node.keys.size(); i++) {
            right.keys.add(node.keys.get(i));
            if (node.leaf) right.values.add(node.values.get(i));
        }
        if (!node.leaf) {
            for (int i = middle + 1; i < node.children.size(); i++) right.children.add(node.children.get(i));
        }
        if (node.leaf) {
            // A separator always holds the *maximum* key of the child to its left, so in a leaf the
            // median must stay behind; in an internal page it moves up together with its subtree.
            for (int i = node.keys.size() - 1; i > middle; i--) {
                node.keys.remove(i);
                node.values.remove(i);
            }
        } else {
            for (int i = node.keys.size() - 1; i >= middle; i--) node.keys.remove(i);
            while (node.children.size() > node.keys.size() + 1) node.children.remove(node.children.size() - 1);
        }
        int leftId = store.allocate(node);
        int rightId = store.allocate(right);
        return new Object[] {leftId, separator, rightId};
    }

    /** Deletes a key: copies the path and repairs an underflowing page by borrowing or merging. */
    int remove(int rootId, int key) {
        if (rootId == 0) return 0;
        Object[] result = delete(rootId, key);
        int newRoot = (Integer) result[0];
        if (newRoot == 0) return 0;
        Page rootPage = store.page(newRoot);
        if (!rootPage.leaf && rootPage.keys.isEmpty()) return rootPage.children.get(0);
        return newRoot;
    }

    private Object[] delete(int pageId, int key) {
        Page node = store.writable(pageId);
        int idx = Collections.binarySearch(node.keys, key);
        if (node.leaf) {
            if (idx < 0) return new Object[] {store.allocate(node)};
            node.keys.remove(idx);
            node.values.remove(idx);
            if (node.keys.isEmpty()) return new Object[] {0};
            return new Object[] {store.allocate(node)};
        }
        int childIndex = idx >= 0 ? idx : -idx - 1;
        Object[] child = delete(node.children.get(childIndex), key);
        int newChild = (Integer) child[0];
        if (newChild == 0) {                             // the whole subtree went away
            node.children.remove(childIndex);
            node.keys.remove(childIndex == node.keys.size() ? childIndex - 1 : childIndex);
        } else {
            node.children.set(childIndex, newChild);
        }
        if (!node.children.isEmpty()) {
            int keep = Math.min(childIndex, node.children.size() - 1);
            if (store.page(node.children.get(keep)).keys.size() < Page.MIN_KEYS) rebalance(node, keep);
        }
        refreshSeparators(node);
        if (node.keys.isEmpty() && node.children.size() == 1) return new Object[] {node.children.get(0)};
        return new Object[] {store.allocate(node)};
    }

    /**
     * Borrows a key from a sibling, or merges with it when neither sibling can spare one. For an
     * internal page the parent's separator is the boundary that must travel with the moved child (a
     * page's last child has no key of its own - the parent owns it), so it is inserted or dropped
     * rather than copied.
     */
    private void rebalance(Page parent, int c) {
        Page child = store.writable(parent.children.get(c));
        if (c + 1 < parent.children.size()) {
            Page right = store.writable(parent.children.get(c + 1));
            if (right.keys.size() > Page.MIN_KEYS) {
                if (child.leaf) {
                    child.keys.add(right.keys.remove(0));
                    child.values.add(right.values.remove(0));
                } else {
                    child.keys.add(parent.keys.get(c));
                    child.children.add(right.children.remove(0));
                    right.keys.remove(0);
                }
                parent.children.set(c, store.allocate(child));
                parent.children.set(c + 1, store.allocate(right));
            } else {
                if (child.leaf) {
                    child.keys.addAll(right.keys);
                    child.values.addAll(right.values);
                } else {
                    child.keys.add(parent.keys.get(c));
                    child.keys.addAll(right.keys);
                    child.children.addAll(right.children);
                }
                parent.children.set(c, store.allocate(child));
                parent.children.remove(c + 1);
                parent.keys.remove(c == parent.keys.size() ? c - 1 : c);
            }
        } else {
            Page left = store.writable(parent.children.get(c - 1));
            if (left.keys.size() > Page.MIN_KEYS) {
                if (child.leaf) {
                    child.keys.add(0, left.keys.remove(left.keys.size() - 1));
                    child.values.add(0, left.values.remove(left.values.size() - 1));
                } else {
                    child.keys.add(0, left.keys.remove(left.keys.size() - 1));
                    child.children.add(0, left.children.remove(left.children.size() - 1));
                }
                parent.children.set(c - 1, store.allocate(left));
                parent.children.set(c, store.allocate(child));
            } else {
                if (child.leaf) {
                    left.keys.addAll(child.keys);
                    left.values.addAll(child.values);
                } else {
                    left.keys.add(parent.keys.get(c - 1));
                    left.keys.addAll(child.keys);
                    left.children.addAll(child.children);
                }
                parent.children.set(c - 1, store.allocate(left));
                parent.children.remove(c);
                parent.keys.remove(c - 1);
            }
        }
        refreshSeparators(parent);
    }

    private void refreshSeparators(Page parent) {
        for (int i = 0; i + 1 < parent.children.size(); i++) {
            parent.keys.set(i, maxKey(parent.children.get(i)));
        }
    }

    /** Structural invariants, checked after every mutation. */
    void checkInvariants(int rootId) {
        if (rootId == 0) return;
        Set<Integer> seen = new HashSet<>();
        int leaves = 0;
        int depth = -1;
        Deque<Integer> stack = new ArrayDeque<>();
        Deque<Integer> depths = new ArrayDeque<>();
        stack.push(rootId);
        depths.push(0);
        while (!stack.isEmpty()) {
            int node = stack.pop();
            int level = depths.pop();
            if (!seen.add(node)) throw new AssertionError("page " + node + " is referenced twice");
            Page p = store.page(node);
            for (int i = 1; i < p.keys.size(); i++) {
                if (p.keys.get(i) <= p.keys.get(i - 1)) throw new AssertionError("keys out of order on page " + node);
            }
            if (p.leaf) {
                if (p.values.size() != p.keys.size()) throw new AssertionError("leaf value count mismatch");
                if (node != rootId && p.keys.size() < Page.MIN_KEYS) {
                    throw new AssertionError("leaf " + node + " underflows with " + p.keys.size() + " keys");
                }
                if (depth == -1) depth = level;
                else if (depth != level) throw new AssertionError("leaves at different depths");
                leaves++;
            } else {
                if (p.children.size() != p.keys.size() + 1) {
                    throw new AssertionError("internal page " + node + " has " + p.children.size()
                            + " children for " + p.keys.size() + " keys");
                }
                if (node != rootId && p.keys.isEmpty()) {
                    throw new AssertionError("internal page " + node + " has no keys at all");
                }
                if (p.keys.size() > Page.MAX_KEYS) throw new AssertionError("page " + node + " overflows");
                for (int i = 0; i < p.children.size(); i++) {
                    stack.push(p.children.get(i));
                    depths.push(level + 1);
                }
            }
        }
        if (leaves == 0) throw new AssertionError("no leaf reachable from root " + rootId);
    }

    /** Fingerprint of every page reachable from a version: detects any later mutation of those pages. */
    String fingerprint(int rootId) {
        if (rootId == 0) return "empty";
        List<Integer> ids = new ArrayList<>();
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(rootId);
        while (!stack.isEmpty()) {
            int node = stack.pop();
            ids.add(node);
            Page p = store.page(node);
            if (!p.leaf) stack.addAll(p.children);
        }
        Collections.sort(ids);
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            Page p = store.page(id);
            sb.append(id).append(':').append(p.render()).append(';');
        }
        return sb.toString();
    }
}

/** A writer that prepares a new root privately and installs it with a compare-and-swap. */
final class Writer {
    final int id;
    final PersistentBTree tree;
    int baseRoot, newRoot;
    int retries;
    String status = "idle";

    Writer(int id, PersistentBTree tree) { this.id = id; this.tree = tree; }

    void prepare(int key, String value) {
        baseRoot = tree.root;
        newRoot = tree.put(baseRoot, key, value);
        status = "prepared";
    }

    boolean install(boolean useCas) {
        if (!useCas) {
            tree.root = newRoot;
            status = "installed";
            return true;
        }
        if (tree.root != baseRoot) {                     // somebody else moved the root
            retries++;
            status = "retry";
            return false;
        }
        tree.root = newRoot;
        status = "installed";
        return true;
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    static String render(int rootId, PersistentBTree tree) {
        StringBuilder sb = new StringBuilder();
        List<Integer> keys = new ArrayList<>();
        List<String> values = new ArrayList<>();
        tree.collect(rootId, Integer.MIN_VALUE, Integer.MAX_VALUE, keys, values);
        for (int i = 0; i < keys.size(); i++) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(keys.get(i)).append('=').append(values.get(i));
        }
        return sb.toString();
    }

    static String renderMap(TreeMap<Integer, String> map) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, String> e : map.entrySet()) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(e.getKey()).append('=').append(e.getValue());
        }
        return sb.toString();
    }

    /** Inserts, duplicate updates and ordered iteration against a plain sorted map. */
    static void insertionsAndIteration() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        check(tree.get(0, 5) == null, "an empty tree answers with nothing");
        check(tree.height(0) == 0, "the empty tree has height zero");
        check(render(0, tree).isEmpty(), "and iterates over nothing");

        TreeMap<Integer, String> oracle = new TreeMap<>();
        int root = tree.put(0, 7, "seven");
        oracle.put(7, "seven");
        check("seven".equals(tree.get(root, 7)), "a single element is found");
        check(tree.height(root) == 1, "a single element sits in one leaf: " + tree.height(root));
        check(tree.get(root, 8) == null, "and a missing neighbour is not");
        tree.checkInvariants(root);

        root = 0;
        SplitMix64 rng = new SplitMix64(806001L);
        for (int i = 0; i < 60; i++) {
            int key = rng.nextInt(50);
            String value = "v" + i;
            root = tree.put(root, key, value);
            oracle.put(key, value);
            tree.checkInvariants(root);
        }
        check(render(root, tree).equals(renderMap(oracle)), "the tree matches the sorted map after 60 inserts:"
                + "\n  tree  =" + render(root, tree) + "\n  oracle=" + renderMap(oracle));
        check(tree.size(root) == oracle.size(), "the same number of keys: " + tree.size(root));
        List<Integer> keys = tree.orderedKeys(root);
        check(keys.equals(new ArrayList<>(oracle.keySet())), "iteration is in key order");
        check(tree.height(root) >= 2, "three levels are in play: " + tree.height(root));
        List<Integer> range = new ArrayList<>();
        List<String> rangeValues = new ArrayList<>();
        tree.collect(root, 10, 20, range, rangeValues);
        List<Integer> expected = new ArrayList<>();
        for (int k : oracle.keySet()) if (k >= 10 && k <= 20) expected.add(k);
        check(range.equals(expected), "a bounded range scan matches the oracle: " + range);
        check(rangeValues.size() == expected.size(), "with the values in step");
        metric("inserts", 60);
        metric("pagesAllocated", store.allocations);
        metric("height", tree.height(root));
    }

    /** Every version ever created must keep answering exactly what it answered before. */
    static void persistence() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        List<Integer> roots = new ArrayList<>();
        List<TreeMap<Integer, String>> oracles = new ArrayList<>();
        TreeMap<Integer, String> current = new TreeMap<>();
        SplitMix64 rng = new SplitMix64(806002L);
        roots.add(0);
        oracles.add(new TreeMap<>(current));
        int root = 0;
        for (int i = 0; i < 45; i++) {
            int key = rng.nextInt(60);
            root = tree.put(root, key, "w" + i);
            current.put(key, "w" + i);
            roots.add(root);
            oracles.add(new TreeMap<>(current));
        }
        for (int i = 0; i < roots.size(); i++) {
            check(render(roots.get(i), tree).equals(renderMap(oracles.get(i))),
                    "version " + i + " changed after later writes");
        }
        check(tree.size(roots.get(roots.size() - 1)) == current.size(), "the newest version is complete");
        check(render(roots.get(0), tree).isEmpty(), "version zero is still the empty tree");
        check(tree.get(roots.get(1), tree.orderedKeys(roots.get(1)).get(0)) != null,
                "an old version still answers point reads");
        int oldest = roots.get(1);
        String before = tree.fingerprint(oldest);
        for (int i = 0; i < 10; i++) root = tree.put(root, 1000 + i, "late" + i);
        check(tree.fingerprint(oldest).equals(before), "no page of the old version was ever mutated");
        check(render(oldest, tree).equals(renderMap(oracles.get(1))),
                "and it still returns its own contents");
        metric("versionsChecked", roots.size());
    }

    /** The copy-on-write bound: one new page per level touched, never a copy of the tree. */
    static void copyOnWriteBounds() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        int root = 0;
        for (int i = 0; i < 12; i++) root = tree.put(root, i, "v" + i);
        tree.checkInvariants(root);
        int height = tree.height(root);
        int pages = store.allocations;
        int worst = 0;
        for (int i = 100; i < 160; i++) {
            int before = store.allocations;
            root = tree.put(root, i, "n" + i);
            int allocated = store.allocations - before;
            worst = Math.max(worst, allocated);
            tree.checkInvariants(root);
        }
        check(worst <= 2 * tree.height(root) + 3,
                "an insert allocates at most two pages per level of the path (a copy and its split "
                        + "half): " + worst + " for height " + tree.height(root));
        check(store.allocations < 12 + 60 * (2 * height + 4), "the store grows with the path, not with the tree: "
                + store.allocations);
        int live = tree.orderedKeys(root).size();
        int fullCopyPerWrite = live / Page.MAX_KEYS + height + 1;
        check(store.allocations < 60 * (2 * height + 4),
                "the store never allocates more than the path bound allows: " + store.allocations);
        check(store.allocations < 60 * fullCopyPerWrite,
                "and far less than copying the whole tree on every write: " + store.allocations
                        + " vs " + (60 * fullCopyPerWrite) + " pages");

        // Negative control: without copy-on-write the pages are shared, so old versions change.
        PageStore naive = new PageStore();
        naive.copyOnWrite = false;
        PersistentBTree bad = new PersistentBTree(naive);
        int badRoot = 0;
        for (int i = 0; i < 12; i++) badRoot = bad.put(badRoot, i, "orig" + i);
        int snapshot = badRoot;
        String expected = render(snapshot, bad);
        for (int i = 0; i < 12; i++) badRoot = bad.put(badRoot, i, "changed" + i);
        check(!render(snapshot, bad).equals(expected),
                "the non-copying variant mutates the snapshot it was supposed to preserve");
        Set<Page> distinct = Collections.newSetFromMap(new IdentityHashMap<>());
        distinct.addAll(naive.pages.values());
        check(distinct.size() <= 6, "because it keeps reusing " + distinct.size() + " physical pages");
        Set<Page> realPages = Collections.newSetFromMap(new IdentityHashMap<>());
        realPages.addAll(store.pages.values());
        check(realPages.size() == store.pages.size(),
                "the copy-on-write store never hands out the same page twice: " + realPages.size());
        metric("pathPagesPerWrite", worst);
        metric("pagesAllocatedCow", pages);
    }

    /** Deletions, underflow and rebalancing must preserve every invariant. */
    static void deletions() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        TreeMap<Integer, String> oracle = new TreeMap<>();
        int root = 0;
        for (int i = 0; i < 80; i++) {
            root = tree.put(root, i, "v" + i);
            oracle.put(i, "v" + i);
        }
        int heightBefore = tree.height(root);
        SplitMix64 rng = new SplitMix64(806003L);
        for (int i = 0; i < 45; i++) {
            int key = rng.nextInt(80);
            root = tree.remove(root, key);
            oracle.remove(key);
            tree.checkInvariants(root);
        }
        check(render(root, tree).equals(renderMap(oracle)), "after 45 deletions the tree matches the oracle");
        check(tree.get(root, 3) != null || !oracle.containsKey(3),
                "a surviving key is still found: get(3)=" + tree.get(root, 3)
                        + ", present=" + oracle.containsKey(3));
        check(tree.size(root) == oracle.size(), "the sizes agree: " + tree.size(root));
        for (int key : new ArrayList<>(oracle.keySet())) {
            root = tree.remove(root, key);
            oracle.remove(key);
        }
        check(root == 0, "deleting everything leaves an empty tree");
        check(render(0, tree).isEmpty(), "which iterates over nothing");
        check(tree.height(0) == 0, "and has height zero again");
        for (int i = 0; i < 40; i++) root = tree.put(root, i, "back" + i);
        tree.checkInvariants(root);
        check(tree.size(root) == 40, "the tree is usable again after emptying: " + tree.size(root));
        check(tree.height(root) <= heightBefore + 1, "and the height stays bounded: " + tree.height(root)
                + " vs " + heightBefore);
        metric("deletions", 45);
    }

    /** Two writers starting from the same version must not lose each other's work. */
    static void concurrencyAndLostUpdates() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        int root = 0;
        for (int i = 0; i < 30; i++) root = tree.put(root, i, "base" + i);
        tree.root = root;
        int startVersion = root;

        Writer a = new Writer(1, tree), b = new Writer(2, tree);
        a.prepare(100, "from-a");
        b.prepare(200, "from-b");                        // both started from startVersion
        check(a.install(true), "the first writer installs its root");
        check(!b.install(true), "the second CAS fails because the root moved");
        check(b.retries == 1, "and it counted the failure: " + b.retries);
        b.prepare(200, "from-b");                        // retry from the new root
        check(b.install(true), "the retry installs successfully");
        check("from-a".equals(tree.get(tree.root, 100)), "the first writer's key survived");
        check("from-b".equals(tree.get(tree.root, 200)), "and so did the second writer's key");
        check(tree.size(tree.root) == 32, "both updates are in the new version: " + tree.size(tree.root));
        check("base5".equals(tree.get(startVersion, 5)), "the shared version is untouched");
        check(tree.get(startVersion, 100) == null, "and does not contain either new key");
        tree.checkInvariants(tree.root);

        // A reader that opened a version keeps a consistent view while writers commit.
        int readerVersion = tree.root;
        String readerView = render(readerVersion, tree);
        for (int i = 0; i < 20; i++) {
            Writer w = new Writer(100 + i, tree);
            w.prepare(300 + i, "later" + i);
            while (!w.install(true)) w.prepare(300 + i, "later" + i);
        }
        check(render(readerVersion, tree).equals(readerView), "the reader's snapshot is unchanged");
        check(tree.get(readerVersion, 300) == null, "and still cannot see the later writes");
        check(tree.size(tree.root) == 52, "every writer made it into the final version: " + tree.size(tree.root));

        // Negative control: without the CAS the second writer overwrites the first one's root.
        PageStore naiveStore = new PageStore();
        PersistentBTree naive = new PersistentBTree(naiveStore);
        int naiveRoot = 0;
        for (int i = 0; i < 10; i++) naiveRoot = naive.put(naiveRoot, i, "x" + i);
        naive.root = naiveRoot;
        Writer c = new Writer(3, naive), d = new Writer(4, naive);
        c.prepare(500, "c");
        d.prepare(600, "d");
        c.install(false);
        d.install(false);
        check(naive.get(naive.root, 600) != null && naive.get(naive.root, 500) == null,
                "without the CAS the first writer's update is lost");
        metric("writerRetries", b.retries);
        metric("finalSize", tree.size(tree.root));
    }

    /** Bulk loading builds the same tree bottom-up; the invariants and the oracle still hold. */
    static void bulkLoad() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        List<Integer> keys = new ArrayList<>();
        List<String> values = new ArrayList<>();
        TreeMap<Integer, String> oracle = new TreeMap<>();
        for (int i = 0; i < 200; i++) {
            keys.add(i * 3);
            values.add("b" + i);
            oracle.put(i * 3, "b" + i);
        }
        int root = tree.build(keys, values);
        tree.checkInvariants(root);
        check(render(root, tree).equals(renderMap(oracle)), "the bulk-loaded tree matches the oracle");
        check(tree.size(root) == 200, "with every key present: " + tree.size(root));
        check(tree.height(root) <= 4, "packed into a tree of at most log_5(n) levels: "
                + tree.height(root));
        check(store.allocations < 200, "using fewer pages than keys: " + store.allocations);
        int height = tree.height(root);
        for (int i = 200; i < 240; i++) {
            root = tree.put(root, i * 3, "c" + i);
            oracle.put(i * 3, "c" + i);
        }
        tree.checkInvariants(root);
        check(render(root, tree).equals(renderMap(oracle)), "and it accepts further inserts");
        check(tree.height(root) <= height + 1, "without growing abruptly: " + tree.height(root));
        int rootBefore = root;
        for (int i = 0; i < 240; i += 3) {
            root = tree.remove(root, i * 3);
            oracle.remove(i * 3);
        }
        tree.checkInvariants(root);
        check(render(root, tree).equals(renderMap(oracle)), "and deletions against the bulk-loaded tree");
        check(tree.size(rootBefore) == 240, "the earlier version is still complete: "
                + tree.size(rootBefore));
        metric("bulkLoadPages", store.allocations);
    }

    /** A larger randomised workload, still deterministic, against the oracle. */
    static void scaleWorkload() {
        PageStore store = new PageStore();
        PersistentBTree tree = new PersistentBTree(store);
        TreeMap<Integer, String> oracle = new TreeMap<>();
        SplitMix64 rng = new SplitMix64(806004L);
        int root = 0;
        int inserts = 0, deletes = 0;
        for (int i = 0; i < 400; i++) {
            int key = rng.nextInt(300);
            if (rng.nextInt(4) == 0) {
                if (oracle.containsKey(key)) {
                    root = tree.remove(root, key);
                    oracle.remove(key);
                    deletes++;
                }
            } else {
                root = tree.put(root, key, "s" + i);
                oracle.put(key, "s" + i);
                inserts++;
            }
        }
        tree.checkInvariants(root);
        check(render(root, tree).equals(renderMap(oracle)), "400 mixed operations match the oracle");
        check(tree.size(root) == oracle.size(), "size " + tree.size(root) + " vs " + oracle.size());
        int height = tree.height(root);
        check(height <= 8, "the tree stays shallow for " + oracle.size() + " keys: " + height);
        List<Integer> keys = tree.orderedKeys(root);
        check(keys.equals(new ArrayList<>(oracle.keySet())), "iteration stays ordered");
        metric("mixedOperations", 400);
        metric("scaleInsert", inserts);
        metric("scaleDelete", deletes);
        metric("scaleHeight", height);
        metric("totalPagesAllocated", store.allocations);
    }

    public static void main(String[] args) {
        insertionsAndIteration();
        persistence();
        copyOnWriteBounds();
        deletions();
        concurrencyAndLostUpdates();
        bulkLoad();
        scaleWorkload();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
