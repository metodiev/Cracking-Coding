/** Task 067 - Implement max-heap push/pop. (Tier: Beginner, CTCI Ch 4) */
import java.util.NoSuchElementException;

class MaxHeap {
    private int[] data = new int[4];
    private int size = 0;

    void push(int value) {
        if (size == data.length) grow();
        data[size] = value;
        siftUp(size++);
    }

    int pop() {
        if (size == 0) throw new NoSuchElementException("heap empty");
        int top = data[0];
        data[0] = data[--size];
        siftDown(0);
        return top;
    }

    int peek() {
        if (size == 0) throw new NoSuchElementException("heap empty");
        return data[0];
    }

    int size() { return size; }

    private void grow() {
        int[] bigger = new int[data.length * 2];
        System.arraycopy(data, 0, bigger, 0, size);
        data = bigger;
    }

    private void siftUp(int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (data[parent] >= data[i]) break;
            swap(parent, i);
            i = parent;
        }
    }

    private void siftDown(int i) {
        while (true) {
            int largest = i, left = 2 * i + 1, right = 2 * i + 2;
            if (left < size && data[left] > data[largest]) largest = left;
            if (right < size && data[right] > data[largest]) largest = right;
            if (largest == i) break;
            swap(i, largest);
            i = largest;
        }
    }

    private void swap(int a, int b) {
        int t = data[a]; data[a] = data[b]; data[b] = t;
    }
}

public class Solution {
    public static void main(String[] args) {
        MaxHeap h = new MaxHeap();
        for (int v : new int[]{5, 3, 8, 1, 9, 2}) h.push(v);
        check(h.peek() == 9, "max on top");
        StringBuilder drained = new StringBuilder();
        for (int i = 0; i < 6; i++) drained.append(h.pop());
        check(drained.toString().equals("985321"), "descending drain, got " + drained);
        check(h.size() == 0, "emptied");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
