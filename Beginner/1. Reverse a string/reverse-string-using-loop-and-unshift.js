function reverseString(str) {
    const reversedArray = [];
    for (const char of str) {
        reversedArray.unshift(char);
    }
    return reversedArray.join('');
}

//Example
console.log(reverseString("JavaScript Hello World 31 123 test 312"))