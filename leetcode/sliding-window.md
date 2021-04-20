[3. 无重复字符的最长子串](#3-无重复字符的最长子串)

[209. 长度最小的子数组](#209-长度最小的子数组)

**[76. 最小覆盖子串](#76-最小覆盖子串)**

**[567. 字符串的排列](#567-字符串的排列)**、[438. 找到字符串中所有字母异位词](#438-找到字符串中所有字母异位词)



[239. 滑动窗口最大值](#239-滑动窗口最大值)



### 滑动窗口做题思路

**滑动窗口可以理解为双指针的一种特殊用法，因此同样适用于子串数组链表题。**一前一后两个指针，中间是窗口。

- 前指针不断向前移动，直到窗口中的字符满足条件。

- 移动后面的指针，直到不满足条件。**每次移动，都要用可行解去更新最优解。**

再重复上述操作，直至遍历完整个数组。第 1 步相当于在寻找一个**「可行解」**，然后第 2 步在**优化这个「可行解」**，最终找到**最优解**

**模板：**以[76. 最小覆盖子串](#76-最小覆盖子串)，为例，反复理解。

```java
public String minWindow(String s, String t) {
    int left = 0, right = 0;
    String res = "";
    Map<Character, Integer> need = new HashMap<>();
    Map<Character, Integer> window = new HashMap<>();
    int valid = 0;
    //先统计t中每个字符的个数
    for (char c : t.toCharArray()) {
        need.put(c, need.getOrDefault(c, 0) + 1);
    }
    while (right < s.length()) {
        // 增大窗口
        char c = s.charAt(right);    
        right++;
        //更新窗口数据统计 先put再加
        if (need.containsKey(c)) {
            window.put(c, window.getOrDefault(c, 0) + 1);
            if (window.get(c).equals(need.get(c))) {
                valid++;
            }
        }
        //判断窗口是不是要收缩
        while (valid == need.size()) {
            //更新可行解
            if (right - left + 1 < res.length() || res.length() == 0) {
                res = s.substring(left, right); 
            }
            //缩小窗口
            char d = s.charAt(left);
            left++;
            //更新窗口数据统计 与上面的完全对称 先减再put
            if (need.containsKey(d)) {
                if (window.get(d).equals(need.get(d))) {
                    valid--;
                }
                window.put(d, window.get(d) - 1);
            }
        }     
    }
    return res;
}
```

[3. 无重复字符的最长子串](#3-无重复字符的最长子串)、[209. 长度最小的子数组](#209-长度最小的子数组)这两道题比较基础，思路一样。



### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

Difficulty: **中等**


给定一个字符串，请你找出其中不含有重复字符的 **最长子串 **的长度。

**示例 1:**

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

**示例 4:**

```
输入: s = ""
输出: 0
```

**提示：**

*   0 <= s.length <= 5 * 10<sup>4</sup>
*   `s` 由英文字母、数字、符号和空格组成

**思路：滑动窗口。**left和right两个指针，[left, right]区间内没有重复的时候，right不断向右移动，在这个过程中不断更新result。如果遇到重复，left不断向前移动直到没有重复。

由于s由字母、数字、符号、空格组成，组成元素比较多，不太适合用数组，用Set或者Map比较好。

```java
public int resgthOfLongestSubstring(String s) {
    Set<Character> set = new HashSet<>();
    int right = 0, left = 0, res = 0;
    while (right < s.length()) {
        if (!set.contains(s.charAt(right))) {
            set.add(s.charAt(right));
            res = Math.max(res, set.size());
            right++;
        } else {
            set.remove(s.charAt(left++));
        }
    }
    return res;
}
```

了解一下另一种思路：用map记录当前char和当前位置，遇到重复的时候，直接读map中存的重复char的位置，相当于left直接跳到位。

```java
public int lengthOfLongestSubstring(String s) {
    Map<Character, Integer> map = new HashMap<>();
    int res = 0;
    int left = 0, right = 0;
    while (right < s.length()) {
        if (map.containsKey(s.charAt(right))) {
            left = Math.max(left, map.get(s.charAt(right)) + 1);
        }
        map.put(s.charAt(right), right);
        res = Math.max(res, right - left + 1);
        right++;
    }
    return res;
}
```

**相关高频题：**

[209. 长度最小的子数组](#209-长度最小的子数组)



### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

Difficulty: **中等**


给定一个含有 `n`个正整数的数组和一个正整数 `target` **。**

找出该数组中满足其和`≥ target`的长度最小的 **连续子数组** [nums<sub style="display: inline;">l</sub>, nums<sub style="display: inline;">l+1</sub>, ..., nums<sub style="display: inline;">r-1</sub>, nums<sub style="display: inline;">r</sub>] ，并返回其长度**。**如果不存在符合条件的子数组，返回 `0` 。

**示例 1：**

```
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
```

**示例 2：**

```
输入：target = 4, nums = [1,4,4]
输出：1
```

**示例 3：**

```
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
```

**提示：**

*   1 <= target <= 10<sup>9</sup>
*   1 <= nums.length <= 10<sup>5</sup>
*   1 <= nums[i] <= 10<sup>5</sup>

**进阶：**

*   如果你已经实现`O(n)` 时间复杂度的解法, 请尝试设计一个 `O(n log(n))` 时间复杂度的解法。

**思路：滑动窗口。**比较简单，窗口的条件是子数组的和大于等于target，更新最优解即可。

```java
public int minSubArrayLen(int s, int[] nums) {
    if (nums.length == 0) {
        return 0;
    }
    int res = Integer.MAX_VALUE;
    int left = 0, right = 0;
    int sum = 0;
    while (right < nums.length) {
        sum += nums[right];
        while (sum >= s) {
            res = Math.min(res, right - left + 1);
            sum -= nums[left];
            left++;
        }
        right++;
    }
    return res == Integer.MAX_VALUE ? 0 : res;
}
```



### **[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)**

Difficulty: **困难**


给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

**注意：**如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

**示例 1：**

```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```

**示例 2：**

```
输入：s = "a", t = "a"
输出："a"
```

**提示：**

*   1 <= s.length, t.length <= 10<sup>5</sup>
*   `s` 和 `t` 由英文字母组成

**进阶：**你能设计一个在 `o(n)` 时间内解决此问题的算法吗？

**思路：滑动窗口。**显然是一个滑动窗口题，在模板基础上，需要收缩的条件是什么，这里用的valid变量。

```java
public String minWindow(String s, String t) {
    int left = 0, right = 0;
    String res = "";
    Map<Character, Integer> need = new HashMap<>();
    Map<Character, Integer> window = new HashMap<>();
    //right右移-扩大窗口时，每当window中当前字符的个数等于t中当前字符的个数，valid+1，
    //left左移-缩小窗口，寻找最优解时，每当window中当前字符的个数等于t中当前字符的个数，valid-1。
    int valid = 0;
    //先统计t中每个字符的个数
    for (char c : t.toCharArray()) {
        need.put(c, need.getOrDefault(c, 0) + 1);
    }
    while (right < s.length()) {
        // 增大窗口
        char c = s.charAt(right);    
        right++;
        //更新窗口数据统计
        if (need.containsKey(c)) {
            window.put(c, window.getOrDefault(c, 0) + 1);
            if (window.get(c).equals(need.get(c))) {
                valid++;
            }
        }
        //判断窗口是不是要收缩
        while (valid == need.size()) {
            //更新可行解
            if (right - left + 1 < res.length() || res.length() == 0) {
                res = s.substring(left, right); //注意前避后开
            }
            
            char d = s.charAt(left);
            left++;
            //更新窗口数据统计 与上面的完全对称
            if (need.containsKey(d)) {
                if (window.get(d).equals(need.get(d))) {
                    valid--;
                }
                window.put(d, window.get(d) - 1);
            }
        }     
    }
    return res;
}
```



### [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

**字节四面笔试的时候遇到了567这道题。**

Difficulty: **中等**


给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1`的排列。

换句话说，第一个字符串的排列之一是第二个字符串的 **子串** 。

**示例 1：**

```
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
```

**示例 2：**

```
输入: s1= "ab" s2 = "eidboaoo"
输出: False
```

**提示：**

*   输入的字符串只包含小写字母
*   两个字符串的长度都在 `[1, 10,000]` 之间

**思路：与[76. 最小覆盖子串](#76-最小覆盖子串)类似，维持一个长度为s1.length()的窗口，统计窗口内的频率，当窗口中所有字母的频率和s1所有字母的频率相同时，说明找到了符合的排列。**

[438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)与这道题方法完全一致，把满足条件的都添加到list就可以了，就不写这道题了。

```java
public boolean checkInclusion(String s1, String s2) {
    int left = 0, right = 0, valid = 0;
    Map<Character, Integer> need = new HashMap<>();
    Map<Character, Integer> window = new HashMap<>();
    for (char c : s1.toCharArray()) {
        need.put(c, need.getOrDefault(c, 0) + 1);
    }
    while (right < s2.length()) {
        char c = s2.charAt(right);
        right++;
        if (need.containsKey(c)) {
            window.put(c, window.getOrDefault(c, 0) + 1);
            //添加之后相等，添加了以后valid的数量会变多一个。
            if (window.get(c).equals(need.get(c))) {
                valid++;
            }
        }
        //right大于等于s1.length()的时候，窗口大小和s1相等了，right每移动一次，left移动一次
        if (right >= s1.length()) {
            char d = s2.charAt(left);
            left++;
            if (valid == need.size()) {
                return true;
            }
            if (need.containsKey(d)) {
                //删之前相等，删了以后valid的数量会变少一个。
                if (window.get(d).equals(need.get(d))) {
                    valid--;
                }
                window.put(d, window.get(d) - 1);
            }
        }
    }
    return false;
}
```

也可以用数组代替map，此时valid变量的含义变了成了有效字符的数量，如果valid=s1.length()，说明找到了。

```java
public boolean checkInclusion(String s1, String s2) {
    int[] fres1 = new int[256];
    int[] fres2 = new int[256];
    for (char c : s1.toCharArray()) {
        fres1[c]++;
    }
    int left = 0, right = 0, valid = 0;
    while (right < s2.length()) {
        char c = s2.charAt(right);
        fres2[c]++;
        //如果添加以后没有超过s1需要的，是有效添加，valid+1。
        if (fres1[c] >= fres2[c]) {
            valid++;
        }
        if (right >= s1.length()) {
            char d = s2.charAt(left);
            fres2[d]--;
            //如果删除以后比s1需要的少，是有效删除，valid-1。
            if (fres1[d] > fres2[d]) {
                valid--;
            }
            left++;
        }
        if (valid == s1.length()) {
            return true;
        }
        right++;
    }
    return false;
}
```



### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

**美团实习笔试遇到了这道题**

Difficulty: **困难**


给你一个整数数组 `nums`，有一个大小为 `k`的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

**示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**示例 2：**

```
输入：nums = [1], k = 1
输出：[1]
```

**示例 3：**

```
输入：nums = [1,-1], k = 1
输出：[1,-1]
```

**示例 4：**

```
输入：nums = [9,11], k = 2
输出：[11]
```

**示例 5：**

```
输入：nums = [4,-2], k = 2
输出：[4]
```

**提示：**

*   1 <= nums.length <= 10<sup>5</sup>
*   -10<sup>4</sup> <= nums[i] <= 10<sup>4</sup>
*   `1 <= k <= nums.length`

**思路：单调队列。**

对于窗口内的一个数：

- 如果窗口内左边的数都比它小，那么左边这些数都不是窗口内最大的数，容器内没必要保存前面这些数，**从右到左删除前面比它小的数**，只保存当前数就可以了。
- 如果前一个数比它大，那不能删除前面的，**还要把当前数放到容器当中**，因为当窗口最左边到当前数的时候，前面那个数就不起作用了，当前数有可能是这个窗口内的最大数，因此要把当前数放到容器中。

因此实际上容器内的值是从大到小排列的，而且既要因为窗口右移而从左边删除容器内的值，也要从右边容器尾部比当前值小的，因此是一个**双向单调递减队列。**

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    Deque<Integer> deque = new LinkedList<>();
    int[] res = new int[nums.length - k + 1];
    for(int i = 0; i < nums.length; i++){
        //当前数大于队尾的数，删除队尾。
        while(deque.size() > 0 && nums[i] > nums[deque.peekLast()]){
            deque.pollLast();
        }
        deque.offerLast(i);
        //如果队首元素在窗口之外，需要将队首元素删除
        if(deque.size() > 0 && deque.peekFirst() <= i - k){
            deque.pollFirst();
        }
        //更新窗口内的最大值
        if(i - k + 1 >= 0){
            res[i - k + 1] = nums[deque.peekFirst()];
        }
    }
    return res;
}
```

