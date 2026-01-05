public class PalindromeUsingStringBuilder {

    public static boolean isPalindrome(String str){
        String reversed = new StringBuilder(str).reverse().toString().toLowerCase();

        return str.toLowerCase().equals(reversed);

    }

    public static void main(String[] args) {
        //call the function and check if the word is palindrome
       // String str = "Detartrated";
        String str = "Bob";
        boolean isWordPalindrome = isPalindrome(str);
        System.out.println("Is palindrome:" + isWordPalindrome);
    }
}
