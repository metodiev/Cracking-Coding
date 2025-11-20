public class ReverseCharArrayInPlace{
    public static void reverseCharArray(char[] s){
        int left =0, right = s.length -1;

        while(left < right){
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;

            left++;
            right--;

        }
    }

    public static void main(String[] args){
        char[] arr = {'h','e','l','l','o'};
        reverseCharArray(arr);
        System.out.println(arr);

    }
}