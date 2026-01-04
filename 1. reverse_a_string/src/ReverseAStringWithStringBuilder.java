public class ReverseAStringWithStringBuilder {

    public static String reverseTheString(String str){
        String reversedString = new StringBuilder(str).reverse().toString();
        return reversedString;
    }

    public static void main(String[] args) {
        //Call reversed String function
        String str = "We need to reverse this string, hello 3123 123 ";
        String reversedString = reverseTheString(str);
        System.out.println(reversedString);
    }
}
