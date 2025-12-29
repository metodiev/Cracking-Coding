#include <iostream>

std::string reverseString(std::string str) {
   if (str.empty()) return "";
      return reverseString(str.substr(1)) + str[0];
}
