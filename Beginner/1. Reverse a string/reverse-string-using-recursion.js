function reverseString(string) {
    if (str == "") return "";
        return reverseString(str.substr(1) + str[0]);
}

//Example
console.log(reverseString("JavaScript Hello World 123 test 312"));