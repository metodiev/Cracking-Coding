#include <algorithm>
#include <string>
#include <iostream>

std::string reverseString(const std:: string& str) {
    std::string reversed = str;
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

int main() {
    std::cout << reverseString("hello string 312 123 ") << std::endl; // olleh
}