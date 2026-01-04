public class ReverseAStringUsingForLoop {
    public static String reverseAstring(String str){
        String reversedString = "";
        for (int i = str.length() -1 ; i >= 0 ; i--) {
            reversedString += str.charAt(i);
        }
        return reversedString;
    }

    public static void main(String[] args) {
        //call reverse String function
        String str = "This string has to be reversed 100 312 Hello";
        String reversedString = reverseAstring(str);
        System.out.println(reversedString);
    }
}
