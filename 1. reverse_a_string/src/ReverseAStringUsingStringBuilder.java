public class ReverseAStringUsingStringBuilder {
    public static String reverseAstring(String str){
        StringBuilder sb = new StringBuilder();
        for (int i = str.length() -1; i >= 0  ; i--) {
            sb.append(str.charAt(i));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        //call the function
        String str = "This string will be reversed Hello, 123";
        String reversedString = reverseAstring(str);

        System.out.println(reversedString);
    }
}
