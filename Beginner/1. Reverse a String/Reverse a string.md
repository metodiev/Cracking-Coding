# Reverse a String in C++

This project demonstrates **multiple ways to reverse a string** in **C++**, including iterative, recursive, and STL-based approaches.


## Task Description

**Problem:**  
Given a string `s`, reverse it using different C++ methods.

**Example:**

```cpp
Input: "hello"
Output: "olleh"
```

## 1. Using std::reverse (STL)

```cpp
#include <algorithm>
#include <string>
#include <iostream>

std::string reverseString1(const std::string& str) {
    std::string reversed = str;
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

int main() {
    std::cout << reverseString1("hello") << std::endl; // olleh
}

```

## 2. Using Iterative Swap

```cpp
std::string reverseString2(std::string str) {
    int n = str.length();
    for (int i = 0; i < n / 2; i++) {
        std::swap(str[i], str[n - i - 1]);
    }
    return str;
}
```

## 3. Using Recursion

```cpp
std::string reverseString3(const std::string& str) {
    if (str.empty()) return "";
    return reverseString3(str.substr(1)) + str[0];
}

```

## 4. Using Stack

```cpp
#include <stack>

std::string reverseString4(const std::string& str) {
    std::stack<char> st;
    for (char c : str) st.push(c);
    std::string reversed;
    while (!st.empty()) {
        reversed += st.top();
        st.pop();
    }
    return reversed;
}

```