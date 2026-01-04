public class ReverseAStringUsingCharacterArray {

    public static String reverseString(String str){
        char [] chars = str.toCharArray();
        int left = 0;
        int right = chars.length -1;
        while ( left < right){
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }

        return new String(chars);
    }

    public static void main(String[] args) {
        //call function to reverse the string using char arrays
        String str = "This array will be reversed";
        String reversedString = reverseString(str);
        System.out.println(reversedString);
    }
}
