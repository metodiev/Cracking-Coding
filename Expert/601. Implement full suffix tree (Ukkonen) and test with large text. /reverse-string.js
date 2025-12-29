/**
 * Reverse String Methods
 * Demonstrates multiple ways to reverse a string in JavaScript.
 */

// 1. Using split, reverse, join
function reverseString1(str) {
    return str.split('').reverse().join('');
}

// 2. Using for loop
function reverseString2(str) {
    let reversed = '';
    for (let i = str.length - 1; i >= 0; i--) {
        reversed += str[i];
    }
    return reversed;
}

// 3. Using recursion
function reverseString3(str) {
    if (str === '') return '';
    return reverseString3(str.substr(1)) + str[0];
}

// 4. Using reduce
function reverseString4(str) {
    return str.split('').reduce((rev, char) => char + rev, '');
}

// 5. Using ES6 spread operator
const reverseString5 = str => [...str].reverse().join('');

// Example Usage
if (require.main === module) {
    const testStr = "hello";
    console.log(reverseString1(testStr));
    console.log(reverseString2(testStr));
    console.log(reverseString3(testStr));
    console.log(reverseString4(testStr));
    console.log(reverseString5(testStr));
}

// Export functions
module.exports = {
    reverseString1,
    reverseString2,
    reverseString3,
    reverseString4,
    reverseString5
};
