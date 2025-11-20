public class ReverseCharactersOnly{

    public static String reverseIgnoreSpace(String s){
        char [] arr = s.toCharArray();
        int left = 0, right = arr.length -1;

        while (left < right) {
            if (arr[left] == ' ') {
                left++;
            } else if (arr[right] == ' ') {
                right--;
            } else {
                char temp = arr[left];
                arr[left]  = arr[right];
                arr[right] = temp;
                left++;
                right--;
            }
        }
        return new String(arr);
    }

    public static void main(String[] args){
        System.out.println(reverseIgnoreSpace("a b c"));
    }
}