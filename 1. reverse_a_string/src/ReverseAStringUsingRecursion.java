public class ReverseAStringUsingRecursion {
    public static int counter = 0;
    public static String reverseAString(String str){
        //bottom of the recursion
        if (str.isEmpty()){
            return str;
        }
        System.out.println("Counter: " + counter);
        System.out.println("str.substring(1): " + str.substring(1));
        System.out.println("str.chartAt(0):" + str.charAt(0));
        System.out.println("str.substring(1)) + str.charAt(0):" +
                (str.substring(1)) + str.charAt(0));
        counter ++;
        return reverseAString(str.substring(1)) + str.charAt(0);
    }

    public static void main(String[] args) {
        //call reverse string recursive version

        String str = "This string will be reversed, Hello 3123 123 ";
        String reversedString = reverseAString(str);
        System.out.println(reversedString);
    }
}
