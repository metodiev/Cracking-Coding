#include <iostream>

std::string reverseString(std::string str) {
    int n = str.length();
    for (int i = 0; i < n/2; i++) {
        std::swap(str[i], str[n-i-1]);
    }
    return str;
}


int main() {
    std::cout << reverseString("hello 312 3425 ") << std::endl; // olleh
}