# Check if a string is a palindrome 

# Check if a String Is a Palindrome in Java


A **palindrome** is a string that reads the same forward and backward.


## 1. Reverse the String (Simplest)

```java
public static boolean isPalindrome(String s) {
    String reversed = new StringBuilder(s).reverse().toString();
    return s.equals(reversed);
}
```

### Explanation

* Reverse the string
* Compare it with the original

### Pros

* Very readable
* Easy to remember

### Cons

* Uses extra memory `O(n)`
* Case-sensitive


## 2. Two-Pointer Technique (Best Practice)

```java
public static boolean isPalindrome(String s) {
    int left = 0, right = s.length() - 1;

    while (left < right) {
        if (s.charAt(left) != s.charAt(right)) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}
```

### Explanation

* Compare characters from both ends
* Move inward until the center

### Pros

* `O(n)` time
* `O(1)` memory
* Fastest and most efficient

### Recommended for production code 

## 3. Recursive Approach

```java
public static boolean isPalindrome(String s) {
    return isPalindrome(s, 0, s.length() - 1);
}

private static boolean isPalindrome(String s, int left, int right) {
    if (left >= right) return true;
    if (s.charAt(left) != s.charAt(right)) return false;
    return isPalindrome(s, left + 1, right - 1);
}
```

### Explanation

* Compare first and last characters recursively

### Cons

* Risk of stack overflow
* Slower than iteration

### Mostly educational


## 4. Using a Character Array

```java
public static boolean isPalindrome(String s) {
    char[] chars = s.toCharArray();
    int i = 0, j = chars.length - 1;

    while (i < j) {
        if (chars[i++] != chars[j--]) {
            return false;
        }
    }
    return true;
}
```

### Explanation

* Convert string to array
* Use two pointers

### Notes

* Slightly faster than `charAt` in tight loops


## 5. Java Streams (Functional Style)

```java
public static boolean isPalindrome(String s) {
    return IntStream.range(0, s.length() / 2)
            .allMatch(i -> s.charAt(i) == s.charAt(s.length() - i - 1));
}
```

### Explanation

* Compare mirrored indices using streams

### Cons

* Less readable
* Slower


## 6.. Using a Deque (Educational)

```java
public static boolean isPalindrome(String s) {
    Deque<Character> deque = new ArrayDeque<>();
    for (char c : s.toCharArray()) {
        deque.add(c);
    }

    while (deque.size() > 1) {
        if (!deque.pollFirst().equals(deque.pollLast())) {
            return false;
        }
    }
    return true;
}
```

### Cons

* Overkill
* Worse performance



---

## Performance Comparison

| Approach       | Time | Memory | Use Case         |
| -------------- | ---- | ------ | ---------------- |
| Two-pointer    | O(n) | O(1)   | Production ⭐     |
| Reverse string | O(n) | O(n)   | Simple tasks     |
| Streams        | O(n) | O(1)   | Functional style |
| Unicode-safe   | O(n) | O(n)   | Emojis / i18n    |
| Regex          | O(n) | O(n)   | Short strings    |

---

## Final Recommendation

```java
public static boolean isPalindrome(String s) {
    int left = 0, right = s.length() - 1;

    while (left < right) {
        if (s.charAt(left++) != s.charAt(right--)) {
            return false;
        }
    }
    return true;
}
```

✅ Best balance of **performance, clarity, and correctness**

---

## Possible Extensions

* JUnit test suite
* Benchmark with JMH
* Streaming input (very large strings)
* Multilingual normalization

---

Happy co



## Performance Summary 

| Approach       | Time | Memory | Production? |
| -------------- | ---- | ------ |---------|
| Two-pointer    | O(n) | O(1)   | YES     |
| Reverse string | O(n) | O(n)   | OK      |
| Streams        | O(n) | O(1)   | Rare    | 
