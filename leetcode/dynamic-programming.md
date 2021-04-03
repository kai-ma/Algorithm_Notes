### 非常典型

​	**[91. 解码方法](#91-解码方法)**



### 基础动态规划

​	**[70. 爬楼梯](#70-爬楼梯)**

​			[198. 打家劫舍](#198-打家劫舍)

​	**[121. 买卖股票的最佳时机](#121-买卖股票的最佳时机)**

​	[122. 买卖股票的最佳时机(二)](#122-买卖股票的最佳时机(二))

⚡**[42. 接雨水](#42-接雨水)**

​			[238. 除自身以外数组的乘积](#238-除自身以外数组的乘积)

#### 二维数组	

​	[62. 不同路径](#62-不同路径)

​	[63. 不同路径II](#63-不同路径II)

​	[64. 最小路径和](#64-最小路径和)



### 子序列\子数组问题

​	**[53. 最大子序和](#53-最大子序和)**

​	[152. 乘积最大子数组](#152-乘积最大子数组)

​	**[300. 最长递增子序列](#300-最长递增子序列)**

​	[32. 最长有效括号](#32-最长有效括号)



### 树的动态规划

​	**[96. 不同的二叉搜索树](#96-不同的二叉搜索树)**



### 回文串问题

​	**[5. 最长回文子串](#5-最长回文子串)**

​		[647. 回文子串](#647-回文子串)

​		[516. 最长回文子序列](#516-最长回文子序列)

Todo：

[516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

[336. 回文对](https://leetcode-cn.com/problems/palindrome-pairs/)、

[214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)



### 双字符串问题

​	**[1143. 最长公共子序列](#1143-最长公共子序列)**

​		[1035. 不相交的线](#1035-不相交的线)

​	[718. 最长重复子数组](#718-最长重复子数组)

​	[583. 两个字符串的删除操作](#583-两个字符串的删除操作)

​		[712. 两个字符串的最小ASCII删除和](#712-两个字符串的最小ASCII删除和)

​	**[72. 编辑距离](#72-编辑距离)**

​		[161. 相隔为1的编辑距离](#161-相隔为1的编辑距离)

​	[44. 通配符匹配](#44-通配符匹配)

⚡[10. 正则表达式匹配](#10-正则表达式匹配)



### 背包问题

​	[322. 零钱兑换](#322-零钱兑换)

​		[279. 完全平方数](#279-完全平方数)

​		[983. 最低票价](#983-最低票价)

todo[377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/) 



### 其他，很值得思考

​	[221. 最大正方形](#221-最大正方形)



步骤：

- 确定状态与选择，定义子问题
- 写出子问题的递推关系——状态转移方程
- 确定 DP 数组的计算顺序
- 空间优化（可选）

以[322. 零钱兑换](#322-零钱兑换)为例，状态就是递归函数的传参，也就是目标金额amount。选择就是导致状态变化的行为，选择一枚零钱，就相当于减小了目标金额。子问题就是输入一个目标金额 `n`，返回凑出目标金额 `n` 的最少硬币数量。

递推关系、状态转移方程：

```python
for coin in coins:
    dp(n) = min(dp(n), 1 + dp(n - coin))
```

**动态规划的本质是不重复求解子问题，保存子问题的解，通过状态转移方程直接计算出当前问题，大大压缩时间复杂度。**

如果直接用递归，很多情况下由于重复计算，会超时。带记忆的递归相当于动态规划。



## 非常典型

### [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

Difficulty: **中等**


一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

要 **解码** 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"111"` 可以将 `"1"` 中的每个 `"1"` 映射为 `"A"` ，从而得到 `"AAA"` ，或者可以将 `"11"` 和 `"1"`（分别为 `"K"` 和 `"A"` ）映射为 `"KA"` 。注意，`"06"` 不能映射为 `"F"` ，因为 `"6"` 和 `"06"` 不同。

给你一个只含数字的 **非空** 字符串 `num` ，请计算并返回 **解码** 方法的 **总数** 。

题目数据保证答案肯定是一个 **32 位** 的整数。

**示例 1：**

```
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
```

**示例 2：**

```
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```

**示例 3：**

```
输入：s = "0"
输出：0
解释：没有字符映射到以 0 开头的数字。含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
```

**示例 4：**

```
输入：s = "06"
输出：0
解释："06" 不能映射到 "F" ，因为字符串开头的 0 无法指向一个有效的字符。 
```

**提示：**

*   `1 <= s.length <= 100`
*   `s` 只包含数字，并且可能包含前导零。

非常典型的动态规划题。看到这个题的第一想法是回溯/递归。但是！！如果用递归的话，复杂度太高，非常多重复的计算。可以用带记忆化的递归，如果已经算过了，直接拿来用。

```java
public int numDecodings(String s) {
    if (s.length() == 0) {
        return 0;
    }
    return numDecodings(s, 0, new HashMap<Integer, Integer>());
}

public int numDecodings(String s, int idx, HashMap<Integer, Integer> map) {
    if (idx == s.length()) {
        return 1;
    }
    int sum = map.getOrDefault(idx, 0);
    if (sum != 0) {
        return sum;
    }
    for (int i = idx; i < idx + 2 && i < s.length(); i++) {
        if (i == idx && s.charAt(i) == '0') {
            return 0;
        }
        int num = Integer.valueOf(s.substring(idx, i + 1));
        if (num >= 1 && num <= 26) {
            sum += numDecodings(s, i + 1, map);
        } else {
            map.put(i, sum);
            return sum;
        }
    }
    map.put(idx, sum);
    return sum;
}
```

**带记忆化的递归其实就是动态规划！**

```java
public int numDecodings(String s) {
    if (s == null || s.length() == 0) {
        return 0;
    }
    int n = s.length();
    int[] dp = new int[n + 1];
    dp[0] = 1;  //设置成1方便dp[2]加
    dp[1] = s.charAt(0) != '0' ? 1 : 0;
    for (int i = 2; i <= n; i++) {
        int first = Integer.parseInt(s.substring(i - 1, i));
        int second = Integer.parseInt(s.substring(i - 2, i));
        if (first >= 1 && first <= 9) {
            dp[i] += dp[i - 1];
        }
        if (second >= 10 && second <= 26) {
            dp[i] += dp[i - 2];
        }
    }
    return dp[n];
}
```



## 基础动态规划

### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

Difficulty: **简单**


假设你正在爬楼梯。需要 _n_ 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**注意：给定 _n_ 是一个正整数。**

**示例 1：**

```
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
```

**示例 2：**

```
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```

**思路：动态规划。实际上是斐波那契数列**

本问题其实常规解法可以分成多个子问题，爬第n阶楼梯的方法数量，等于两部分部分之和：

- 爬上 n-1 阶楼梯的方法数量。因为再爬1阶就能到第n阶。

- 爬上 n-2 阶楼梯的方法数量，因为再爬2阶就能到第n阶。

所以我们得到公式 `dp[n] = dp[n-1] + dp[n-2]`

base case：`dp[1] = 1; dp[2] = 2;`

```java
    public int climbStairs(int n) {
        //n 为正整数，n = 1, 2的特殊情况也被处理了
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i < n + 1; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

	//dp[i]只与dp[i - 1]和dp[i - 2]有关，可以进一步压缩空间复杂度
	public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int count1 = 1;
        int count2 = 2;
        for (int i = 3; i <= n; i++) {
            int temp = count2;
            count2 = temp + count1;
            count1 = temp;
        }
        return count2;
    }
```



### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

Difficulty: **中等**


你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**示例 1：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2：**

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

**提示：**

*   `0 <= nums.length <= 100`
*   `0 <= nums[i] <= 400`

思路：类似于跳台阶问题。

dp[i]表示前 i 间房屋能偷窃到的最高总金额，状态转移方程：`Math.max(dp[i-1], dp[i-2] + nums[i-1])`。

```java
    public int rob(int[] nums) {
        int len = nums.length;
        if(len == 0)
            return 0;
        int[] dp = new int[len + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        for(int i = 2; i <= len; i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i-1]);
        }
        return dp[len];
    }
```



### [213\. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

Difficulty: **中等**


你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下** ，能够偷窃到的最高金额。

**示例 1：**

```
输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
```

**示例 2：**

```
输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 3：**

```
输入：nums = [0]
输出：0
```

**提示：**

*   `1 <= nums.length <= 100`
*   `0 <= nums[i] <= 1000`

**思路：**

数组是个环，也就是说偷第一家，最后一家就不能偷；偷最后一家，第一家就不能偷。

所以，我们问题分成求 nums[0:n - 1]或者 nums[1:n]，就变成了[198. 打家劫舍](#198-打家劫舍)。





### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

**相同题：[剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)**

Difficulty: **简单**


给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

**示例 1：**

```
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**示例 2：**

```
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```

**提示：**

*   1 <= prices.length <= 10<sup>5</sup>
*   0 <= prices[i] <= 10<sup>4</sup>

**思路1：动态规划。**dp[i] 表示前i天的最大利润，则：`dp[i] = max(dp[i-1], prices[i]-minprice)`，minprice是从0到i-1的最小价格。

**思路2：==要是炒股能知道哪一天是最低价就好了。==**

**贪心法。对于任意第i天，如果在这一天卖股票，在这一天的最大收益 `maxprofit =  price[i] - 从起始到i-1处的最小价格`。**每一天都计算这一天的maxprofit，并去更新最大收益，最终即可得到最大收益。

方法2其实是方法1的空间优化方法。

```java
//方法1，动态规划。
public int maxProfit(int[] prices) {
    int minprice = prices[0];
    int[] dp = new int[prices.length];
    for (int i = 1; i < prices.length; i++){
        minprice = Math.min(minprice, prices[i]);
        dp[i] = Math.max(dp[i - 1], prices[i] - minprice);
    }
    return dp[prices.length - 1];
}

//方法2：贪心法
public int maxProfit(int prices[]) {
    int minprice = Integer.MAX_VALUE;
    int maxprofit = 0;
    for (int i = 0; i < prices.length; i++) {
        if (prices[i] < minprice) {
            minprice = prices[i];
        } else if (prices[i] - minprice > maxprofit) {
            maxprofit = prices[i] - minprice;
        }
    }
    return maxprofit;
}
```



### [122. 买卖股票的最佳时机(二)](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

Difficulty: **简单**


给定一个数组，它的第 _i_ 个元素是一支给定股票第 _i_ 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
```

**示例 2:**

```
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```

**示例 3:**

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

**提示：**

*   `1 <= prices.length <= 3 * 10 ^ 4`
*   `0 <= prices[i] <= 10 ^ 4`

**思路1：动态规划。**用两个数组，cash[i]表示当前持有现金时的最大收益，hold[i]表示当前持有股票时的最大收益。遍历数组不断更新即可。

![image.png](https://pic.leetcode-cn.com/041a4d01398359409ecc69dacc13a44d179dd1a2a9f43b1def80e9a6acceee55-image.png)

```java
public int maxProfit(int[] prices) {
    int len = prices.length;
    if (len <= 1) {
        return 0;
    }

    // cash：持有现金
    // stock：持有股票
    // 状态数组
    // 状态转移：cash → stock → cash → stock → cash → stock → cash
    int[] cash = new int[len];
    int[] stock = new int[len];

    cash[0] = 0;
    stock[0] = -prices[0];

    for (int i = 1; i < len; i++) {
        // 这两行调换顺序也是可以的
        cash[i] = Math.max(cash[i - 1], stock[i - 1] + prices[i]);
        stock[i] = Math.max(stock[i - 1], cash[i - 1] - prices[i]);
    }
    return cash[len - 1];
}
```

**实际上，cash[i]相当于不持有股票，stock[i]相当于持有股票，两者可以合并成一个宽度为2的二维数组，有这么一个想法就好了，后面的股票题会用到。**

而且，观察stock和cash，只与上一天有关，可以压缩成两个变量。

**思路2：贪心法——在每一步总是做出在当前看来最好的选择，只要比前一天高，就卖**。由于可以交易无限次，因此所有上涨交易日都买卖（赚到所有利润），所有下降交易日都不买卖（永不亏钱）。

```java
public int maxProfit(int[] prices) {
    if(prices.length<=1) return 0;
    int max=0;
    for(int i=1;i<prices.length;i++){
        if(prices[i]>prices[i-1]){
            max+=prices[i]-prices[i-1];
        }
    }
    return max;
}
```



### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

Difficulty: **困难**

**相关题：**[238. 除自身以外数组的乘积](#238-除自身以外数组的乘积)


给定 _n_ 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例 1：**

![](images/动态规划/rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

**示例 2：**

```
输入：height = [4,2,0,3,2,5]
输出：9
```

**提示：**

*   `n == height.length`
*   0 <= n <= 3 * 10<sup>4</sup>
*   0 <= height[i] <= 10<sup>5</sup>

进阶：[407. 接雨水 II](https://leetcode-cn.com/problems/trapping-rain-water-ii/) 面试一般不会这么难

**方法1：动态规划。left[i]表示i左侧的最大值，即区间[0,i)的最大值，right[i]表示i右侧的最大值，即区间(i,len-1]的最大值。如果i位置比左右两侧的最大值要小，说明i位置可以存水。水量取决于短板**

```java
	public int trap(int[] height) {
        int res = 0;
        int[] left = new int[height.length];
        int[] right = new int[height.length];

        for (int i = 1; i < height.length - 1; i++) {
            left[i] = Math.max(left[i - 1], height[i - 1]);
        }
        for (int i = height.length - 2; i >= 0; i--) {
            right[i] = Math.max(right[i + 1], height[i + 1]);
        }
        for (int i = 1; i < height.length - 1; i++) {
            int min = Math.min(left[i], right[i]);
            if (height[i] < min) {
                res = res + (min - height[i]);
            }
        }
        return res;
    }
```

**方法2：单调递减栈。**如果是单调递减数组，不能蓄水，因此需要把单调递减的先存起来，遇到后面大的数再看能不能在栈顶这个小的数的位置上蓄水。

```java
	public int trap(int[] height) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < height.length; i++) {
            //如果栈不空并且当前指向的高度大于栈顶高度就一直循环
            while (!stack.empty() && height[i] > height[stack.peek()]) {
                //取出要出栈的元素
                int h = height[stack.pop()];
                // 栈空，原栈顶元素和当前元素之间没有空隙了，不会再有雨水。
                if (!stack.empty()) {
                    //两堵墙之前的距离。
                    int distance = i - stack.peek() - 1;
                    int min = Math.min(height[stack.peek()], height[i]);
                    res = res + distance * (min - h);
                }
            }
            //当前指向的墙入栈
            stack.push(i);
        }
        return res;
    }
```

**方法3：双指针。**假设一开始`left-1`大于`right+1`，则之后`right`一直向左移动，直到`right+1`大于`left-1`。在这段时间内`right`所遍历的所有点都是左侧最高点`maxleft`大于右侧最高点`maxright`的，所以只需要判断`maxright`与当前高度的关系就能知道i处能不能蓄水。反之`left`右移，所经过的点只要判断`maxleft`与当前高度的关系就行。

```java
	public int trap(int[] height) {
        int leftMax = 0, rightMax = 0;
        int left = 0, right = height.length - 1;
        int res = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] < leftMax) {
                    //当前值比leftMax小，并且一定比rightMax小，此处可以蓄水。
                    res += leftMax - height[left];
                } else {
                    //当前值比leftMax大，此处不能蓄水，更新leftMax。
                    leftMax = height[left];
                }
                left++;
            } else {
                if (height[right] < rightMax) {
                    res += rightMax - height[right];
                } else {
                    rightMax = height[right];
                }
                right--;
            }
        }
        return res;
    }
```

**直接用leftMax和rightMax比较会更容易理解：**

对于位置`left`而言，它左边最大值一定是left_max，右边最大值“大于等于”right_max，这时候，如果`left_max<right_max`成立，那么它就知道自己能存多少水了。无论右边将来会不会出现更大的right_max，都不影响这个结果。 所以当`left_max<right_max`时，我们就希望去处理left下标，反之，我们希望去处理right下标。

注意while循环结束条件是left<=right。

```java
	public int trap(int[] height) {
        int leftMax = 0, rightMax = 0;
        int left = 0, right = height.length - 1;
        int res = 0;
        while (left <= right) {
            if(leftMax < rightMax){
                res += Math.max(0, leftMax - height[left]);
                leftMax = Math.max(leftMax, height[left]);
                left++;
            }else{
                res += Math.max(0, rightMax - height[right]);
                rightMax = Math.max(rightMax, height[right]);
                right--;
            }
        }
        return res;
    }
```

方法4：韦恩图。很难想出来。[题解](https://leetcode-cn.com/problems/trapping-rain-water/solution/wei-en-tu-jie-fa-zui-jian-dan-yi-dong-10xing-jie-j/)

<img src="images/栈/53ab7a66023039ed4dce42b709b4997d2ba0089077912d39a0b31d3572a55d0b-trapping_rain_water.png" alt="trapping_rain_water.png" style="zoom:50%;" />

图1从左往右`S1+=max1且max1逐步增大`。图2从右往左`S2+=max2且max2逐步增大`。S1 + S2会覆盖整个矩形，并且：重复面积 = 柱子面积 + 积水面积。最终， 积水面积 = S1 + S2 - 矩形面积 - 柱子面积



### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

Difficulty: **中等**


给你一个长度为 _n_ 的整数数组 `nums`，其中 _n_ > 1，返回输出数组 `output` ，其中 `output[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

**示例:**

```
输入: [1,2,3,4]
输出: [24,12,8,6]
```

**提示：**题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。

**说明:** 请**不要使用除法，**且在 O(_n_) 时间复杂度内完成此题。

**进阶：**  
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组**不被视为**额外空间。）

**思路：如果能完全理解[42. 接雨水](#42-接雨水)，那么这道题就应该能耐迎刃而解。**

**初步方法：**left[i] 和 right[i] 分别表示i左右两侧，不包括nums[i]的乘积列表。**状态转移方程：res[i] = left[i] * right[i];**

```java
public int[] productExceptSelf(int[] nums) {
    int length = nums.length;
    // left[i] 和 right[i] 分别表示i左右两侧，不包括nums[i]的乘积列表。res[i] = left[i] * right[i];
    int[] left = new int[length], right = new int[length], res = new int[length];
    left[0] = 1;
    for (int i = 1; i < length; i++) {
        left[i] = nums[i - 1] * left[i - 1];
    }
    right[length - 1] = 1;
    for (int i = length - 2; i >= 0; i--) {
        right[i] = nums[i + 1] * right[i + 1];
    }
    for (int i = 0; i < length; i++) {
        res[i] = left[i] * right[i];
    }
    return res;
}
```

**优化方法：压缩空间到常数空间。**题目已经提示了： 出于对空间复杂度分析的目的，输出数组**不被视为**额外空间。即可以把输出数组作为left，然后再想办法省去right数组即可。对于当前right[i]，只有right[i+1]影响结果，因此right数组可以用变量替代。

```java
public int[] productExceptSelf(int[] nums) {
    int length = nums.length;
    int[] res = new int[length];

    res[0] = 1;
    //把res数组当成left数组
    for (int i = 1; i < length; i++) {
        res[i] = nums[i - 1] * res[i - 1];
    }

    //用right变量替代right数组
    int right = 1;
    for (int i = length - 1; i >= 0; i--) {
        res[i] = res[i] * right;
        right *= nums[i];
    }
    return res;
}
```



### 二维数组

### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

Difficulty: **中等**


一个机器人位于一个 `m x n`网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

**示例 1：**

![](images/动态规划/robot_maze.png)

```
输入：m = 3, n = 7
输出：28
```

**示例 2：**

```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1\. 向右 -> 向下 -> 向下
2\. 向下 -> 向下 -> 向右
3\. 向下 -> 向右 -> 向下
```

**示例 3：**

```
输入：m = 7, n = 3
输出：28
```

**示例 4：**

```
输入：m = 3, n = 3
输出：6
```

**提示：**

*   `1 <= m, n <= 100`
*   题目数据保证答案小于等于 2 * 10<sup>9</sup>

**思路：**等价于杨辉三角形，每个位置的路径 = 该位置左边的路径 + 该位置上边的路径。

```java
public int uniquePaths(int m, int n) {
    if (m == 0 || n == 0) {
        return 0;
    }
    int[][] dp = new int[n][m];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (j == 0 || i == 0) {
                dp[i][j] = 1;
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }
    return dp[n - 1][m - 1];
}
```

### [63. 不同路径II](https://leetcode-cn.com/problems/unique-paths-ii/)

Difficulty: **中等**


一个机器人位于一个 _m x n_ 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

![](images/动态规划/robot_maze.png)

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

**示例 1：**

![](images/动态规划/robot1.jpg)

```
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1\. 向右 -> 向右 -> 向下 -> 向下
2\. 向下 -> 向下 -> 向右 -> 向右
```

**示例 2：**

![](images/动态规划/robot2.jpg)

```
输入：obstacleGrid = [[0,1],[0,0]]
输出：1
```

**提示：**

*   `m == obstacleGrid.length`
*   `n == obstacleGrid[i].length`
*   `1 <= m, n <= 100`
*   `obstacleGrid[i][j]` 为 `0` 或 `1`

思路：如果网格 `(i, j)` 上有障碍物，则 `dp[i][j] `值为 0，表示走到该格子的方法数为 0。如果没有障碍物，`dp[i][j] = dp[i - 1][j] + dp[i][j - 1];`，与[62. 不同路径](#62-不同路径)相同。

注意：base case里面第一行和第一列，如果遇到一个障碍，后面的路一定就走不成了。

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    if (obstacleGrid == null || obstacleGrid.length == 0) {
        return 0;
    }

    // 定义 dp 数组并初始化第 1 行和第 1 列。
    int m = obstacleGrid.length, n = obstacleGrid[0].length;
    int[][] dp = new int[m][n];
	dp[0][0] = 1;
    // 根据状态转移方程 dp[i][j] = dp[i - 1][j] + dp[i][j - 1] 进行递推。
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 0) {
                if(j == 0){
                    dp[i][j] = dp[i - 1][0];
                }else if(i == 0){
                    dp[i][j] = dp[0][j - 1];
                }else{
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
    }
    return dp[m - 1][n - 1];
}
```





### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

Difficulty: **中等**


给定一个包含非负整数的 `_m_ x _n_` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

**示例 1：**

![](images/动态规划/minpath.jpg)

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**示例 2：**

```
输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

**提示：**

*   `m == grid.length`
*   `n == grid[i].length`
*   `1 <= m, n <= 200`
*   `0 <= grid[i][j] <= 100`

**思路：非常明显的动态规划。画出二维数组填表即可**

```java
	public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        //第一列 只能从上到下
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        //第一行 只能从左到右
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        //选择左边或者上边更小的路径
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }
```





## 子序列、子数组问题

一般子问题是以当前位置为结尾的....。

### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

相同题：

Difficulty: **简单**


给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例 1：**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**示例 2：**

```
输入：nums = [1]
输出：1
```

**示例 3：**

```
输入：nums = [0]
输出：0
```

**示例 4：**

```
输入：nums = [-1]
输出：-1
```

**示例 5：**

```
输入：nums = [-100000]
输出：-100000
```

**提示：**

*   1 <= nums.length <= 3 * 10<sup>4</sup>
*   -10<sup>5</sup> <= nums[i] <= 10<sup>5</sup>

**进阶：**如果你已经实现复杂度为 `O(n)` 的解法，尝试使用更为精妙的 **分治法** 求解。

**思路1：动态规划。**子问题是**以i位置为结尾的连续最大和**，把所有的dp[i]都计算出来，最大的就是该数组的连续子序列的最大和。状态转移方程： `dp[i] = Math.max(nums[i], nums[i] + dp[i - 1])`

**思路2：贪心法。**对于第i位置，考虑前面的以i-1为结尾的连续最大和，如果大于0，**对当前位置的有贡献**。如果小于零，从当前位置重新算。实际上相当于动态规划解法的空间优化到O(1)的解法。

```java
//思路1：动态规划
public int maxSubArray(int[] nums) {
    if (nums.length == 0) {
        return 0;
    }
    int[] dp = new int[nums.length];
    // base case
    // 第一个元素前面没有子数组
    dp[0] = nums[0];
    int res = nums[0];
    for (int i = 1; i < nums.length; i++) {
        // 状态转移方程
        dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
        res = Math.max(res, dp[i]);
    }
    return res;
}

//思路2：贪心法
public static int maxSubArray(int[] A) {
    int maxSoFar = A[0], maxEndingHere = A[0];
    for (int i = 1; i < A.length; ++i) {
        maxEndingHere = Math.max(maxEndingHere + A[i], A[i]);
        maxSoFar = Math.max(maxSoFar, maxEndingHere);
    }
    return maxSoFar;
}
```





### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

Difficulty: **中等**


给你一个整数数组 `nums` ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

**示例 1:**

```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

**示例 2:**

```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

**方法1：动态规划**，imax记录包括当前数为结尾的连续子数组的最大值，`imax = max(imax * nums[i], nums[i])`， 由于存在负数，那么会导致最大的变最小的，最小的变最大的。因此还需要维护当前最小值imin，`imin = min(imin * nums[i], nums[i])`，**当出现负数时交换imax与imin**，然后再进行计算。

```java
	public int maxProduct(int[] nums) {
        int res = nums[0];
        int imax = nums[0];   //记录包括当前数为结尾的连续子数组的最大值
        int imin = nums[0];	//记录包括当前数为结尾的连续子数组的最小值
        for(int i = 1; i < nums.length; i++){
            if(nums[i] < 0){  //乘以一个负数，会让大数变小，小数变大。因此进行交换
                int tmp = imax;
                imax = imin;
                imin = tmp;
            }
            imax = Math.max(nums[i], nums[i] * imax);
            imin = Math.min(nums[i], nums[i] * imin);
            res = Math.max(res, imax);
        }
        return res;
    }
```



### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

Difficulty: **中等**


给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

**示例 2：**

```
输入：nums = [0,1,0,3,2,3]
输出：4
```

**示例 3：**

```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

**提示：**

*   `1 <= nums.length <= 2500`
*   -10<sup>4</sup> <= nums[i] <= 10<sup>4</sup>

**进阶：**

*   你可以设计时间复杂度为 O(n<sup>2</sup>)的解决方案吗？
*   你能将算法的时间复杂度降低到 `O(n log(n))` 吗?

**方法1：动态规划。dp[i]表示以i结尾的最长递增子序列。初始情况是dp数组全位1，状态转移方程是`dp[i] = max(1 + dp[j] if j < i and nums[j] < nums[i])`**。最大的dp[i]为最长递增子序列。

```java
	public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return len;
        }

        int[] dp = new int[len];
        Arrays.fill(dp, 1);
		int res = 0;
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
```

**方法2：贪心法+二分法。较难想到，很经典。**

思路：如果已经得到的上升子序列的**结尾的数越小**，那么遍历的时候后面接上一个数，**才会有更大的可能构成一个长度更长的上升子序列。**因此，我们可以记录在长度固定的情况下，结尾最小的那个元素的数值。

`tail[i]` 表示：长度为 `i + 1` 的 **所有** 上升子序列的结尾的最小值。

遍历过程中不断用当前值去更新tail数组。

- 如果当前数大于tail数组目前的最后一个元素，那么说明目前的最长上升子序列该增长了，直接把这个数放到最后面。
- 否则，用当前数去替换tail数组中第一个比它大的数。这样做的逻辑支撑是：假设tail数组中第一个比它大的数是tail[j]，tail[0]-tail[j-1]都比它小，说明找到了结尾更小的长度为j+1的上升子序列。tail[j+1]到tail[i]都比它大，插入当前元素并不影响长度大于j+1的上升子序列。
- 至于寻找tail数组(排序数组)中第一个比它大的位置，当然是用二分法。

```java
public int lengthOfLIS(int[] nums) {
    int len = nums.length;
    if (len <= 1) {
        return len;
    }

    int[] tail = new int[len];
    // 遍历第 1 个数，直接放在有序数组 tail 的开头
    tail[0] = nums[0];
    // end 表示有序数组 tail 的最后一个已经赋值元素的索引
    int end = 0;

    for (int i = 1; i < len; i++) {
        // 比 tail 数组实际有效的末尾的那个元素还大，直接添加在那个元素的后面，所以 end 先加 1
        if (nums[i] > tail[end]) {
            end++;
            tail[end] = nums[i];
        } else {
            // 使用二分查找法，在有序数组 tail 中
            // 找到第 1 个大于等于 nums[i] 的元素，尝试让那个元素更小
            int left = 0;
            int right = end;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (tail[mid] < nums[i]) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            tail[left] = nums[i];
        }
    }
    return end + 1;
}
```



### [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

Difficulty: **困难**


给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**示例 1：**

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

**示例 2：**

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

**示例 3：**

```
输入：s = ""
输出：0
```

**提示：**

*   0 <= s.length <= 3 * 10<sup>4</sup>
*   `s[i]` 为 `'('` 或 `')'`

**方法1：栈。括号题第一时间就应该想到栈**

```java
    public int longestValidParentheses(String s) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        //栈内存的是「最后一个没有被匹配的右括号的下标」。预先把-1压入栈，如果s="()"，1-(-1)=2。
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            //遇到'('，'('不会是有效对的结尾，直接入栈当前下标
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                //遇到')'，先把栈顶弹出来，
                //如果栈内空了，说明没有匹配。更新「最后一个没有被匹配的右括号的下标」
                //如果栈内没有空，说明当前和被弹出的栈顶配对，更新最长有效括号。
                stack.pop();
                if (stack.empty()) {
                    stack.push(i);
                } else {
                    res = Math.max(res, i - stack.peek());
                }
            }
        }
        return res;
    }
```

**方法2：动态规划。**

```java
public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            //当前是')'才会有匹配
            if (i > 0 && s.charAt(i) == ')') {
                //如果前一个是'('，匹配成功，加上'('前面的匹配对，也就是dp[i - 2]
                if (s.charAt(i - 1) == '(') {
                    if (i - 2 >= 0) {
                        dp[i] = dp[i - 2] + 2;
                    } else {
                        dp[i] = 2;
                    }
                } //如果前一个是')'，应该根据前一个的最长有小括号，再往前看1个是不是(。比如((()))
                else if (i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    //如果往前能匹配上，并且前面还不为空，加上更前面的匹配对
                    if (i - dp[i - 1] - 2 >= 0) {
                        dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2];
                    } else {
                        dp[i] = dp[i - 1] + 2;
                    }
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
```





## 树的动态规划

### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

Difficulty: **中等**


给定一个整数 _n_，求以 1 ... _n_ 为节点组成的二叉搜索树有多少种？

**示例:**

```
输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

**思路：**这种1到n个的题目，往往有子问题的规律，要用动态规划来做。这篇题解的画图很清晰：https://leetcode-cn.com/problems/unique-binary-search-trees/solution/shou-hua-tu-jie-san-chong-xie-fa-dp-di-gui-ji-yi-h/

```java
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            for (int j = 0; j <= i - 1; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        
        return dp[n];
    }
```



## 回文串问题

### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

Difficulty: **中等**


给你一个字符串 `s`，找到 `s` 中最长的回文子串。

**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**示例 2：**

```
输入：s = "cbbd"
输出："bb"
```

**示例 3：**

```
输入：s = "a"
输出："a"
```

**示例 4：**

```
输入：s = "ac"
输出："a"
```

**提示：**

*   `1 <= s.length <= 1000`
*   `s` 仅由数字和英文字母（大写和/或小写）组成

**方法1：动态规划。**如果一个字符串已经是/不是回文串，在它的左右各加一个字符，直接就能判断出来这个新字符串是不是回文串。也就是说先判断出中间，再判断两边的话，可以利用到中间的判断，这是一种子问题，可以用动态规划去求解。

`dp[i][j]` 表示子串 `s[i..j]` 是否为回文子串

- 当`i = j`时，`dp[i][j]`是回文子串
- 当`j - i <= 2`时，`dp[i][j] = s[i] == s[j] `，即中间没有字母或中间有一个字母。
- 当`j - i > 2`时，`dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1] == true`

特别注意：`dp[i][j]`的值依赖于`dp[i + 1][j - 1]`的值，因此遍历顺序应该是先遍历i + 1，再遍历i；先遍历j，再遍历j + 1。用i--，j++即可。

**方法2：从中间开始向两边扩散。**对于每一个位置，都把它当作中心，然后向两边扩散。注意：如果只取这个位置然后扩散的话，只能找到长度为单数的回文串。对于长度为双数的也应该考虑到，选取相邻的两个数为中心，向两边扩散。时间复杂度同样也为O(N<sup>2</sup>)，空间复杂度O(1)，比动态规划方法还要好。

- 一种巧妙的方法是：想象给每个字符中都插入一个符号。例如babad，变成b#a#b#a#d，以字母为中心向两边拓展是单数长度字符串，以#为中心向两边拓展是双数长度字符串。

建议先掌握动态规划方法，还有一些相似的题，没办法用扩展法，只能用动态规划。

方法1：动态规划

```java
	public String longestPalindrome(String s) {
        int len = s.length();
        if (len <= 1) {
            return s;
        }
        boolean[][] dp = new boolean[len][len];
        int start = 0, end = 0;
        for (int i = len - 1; i >= 0; i--) {
            dp[i][i] = true;
            for (int j = i + 1; j < len; j++) {
                dp[i][j] = (j - i <= 2 || dp[i + 1][j - 1]) && s.charAt(i) == s.charAt(j);
                if (dp[i][j] && j - i + 1 > end - start) {
                    start = i;
                    end = j;
                }
            }
        }
        return s.substring(start, end + 1);
    }
```

方法2：从中心向两边扩展

```java
 	public String longestPalindrome(String s) {
        if (s.length() <= 1) {
            return s;
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expand(s, i, i);
            int len2 = expand(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expand(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
```

假象有#：

```java
	public String longestPalindrome(String s) {
        int len = s.length();
        String res = "";
        for (int center = 0; center < len * 2 - 1; center++) {
            int left = center / 2;
            int right = left + center % 2;
            while (left >= 0 && right < len && s.charAt(left) == s.charAt(right)) {
                String tmp = s.substring(left, right + 1);
                if (tmp.length() > res.length()) {
                    res = tmp;
                }
                left--;
                right++;
            }
        }
        return res;
    }
```

这个问题还有一个巧妙的解法：Manacher's Algorithm（马拉车算法），时间复杂度只需要 O(N)，不过该解法比较复杂，我个人认为很少有人能第一次遇到这道题就能想出来这种方法，面试时候不会问到这么变态的，没必要掌握。有兴趣的同学可以自行搜索一下。

**相关高频题：**

[516. 最长回文子序列](#516-最长回文子序列)



### [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

Difficulty: **中等**


给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

**示例 1：**

```
输入："abc"
输出：3
解释：三个回文子串: "a", "b", "c"
```

**示例 2：**

```
输入："aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

**提示：**

*   输入的字符串长度不会超过 1000 。

**思路：与[5. 最长回文子串](#5-最长回文子串)完全相同。**把更新大小换成增加数量即可。



### [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

Difficulty: **中等**


给定一个字符串 `s` ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 `s` 的最大长度为 `1000` 。

**示例 1:**  
输入:

```
"bbbab"
```

输出:

```
4
```

一个可能的最长回文子序列为 "bbbb"。

**示例 2:**  
输入:

```
"cbbd"
```

输出:

```
2
```

一个可能的最长回文子序列为 "bb"。

**提示：**

*   `1 <= s.length <= 1000`
*   `s` 只包含小写英文字母

**思路：**与[5. 最长回文子串](#5-最长回文子串)相似，`dp[i][j]`表示从i到j前闭后闭子字符串内的最长回文子序列。

- 当`i = j`时，`dp[i][j] = 1`; 当i = j - 1时，如果`s.charAt(i) = s.charAt(j)`，`dp[i][j] = 2`，否则`dp[i][j] = 0`。
- 如果`s.charAt(i) = s.charAt(j)`， `dp[i][j] = dp[i + 1][j - 1] + 2`
- 否则，`dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1])`

```java
	public int longestPalindromeSubseq(String s) {
        int len = s.length();
        int[][] dp = new int[len][len];
        for (int i = len - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < len; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][len - 1];
    }
```



## 两个字符串问题

**一般都是用两个指针 `i,j` 分别指向两个字符串的最后，然后一步步往前走，缩小问题的规模。**

一般为了避免一个字符串长度为0，会把dp数组的长度增加1，`int[][] dp = new int[s1.length + 1][s2.length + 1];`

**然后画出二维数组的表格，填表格即可。**



### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

Difficulty: **中等**


给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长公共子序列的长度。

一个字符串的子序列是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。  
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回 0。

**示例 1:**

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

**示例 2:**

```
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。
```

**示例 3:**

```
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。
```

**提示:**

*   `1 <= text1.length <= 1000`
*   `1 <= text2.length <= 1000`
*   输入的字符串只含有小写英文字符。

**思路：LCS问题是非常经典的动态规划问题。**解决双字符串这类题一般都是用两个指针 `i,j` 分别指向两个字符串的最后，然后一步步往前走，缩小问题的规模。

**`dp[i][j]` 表示：将 `word1[0..i)` 与 `word2[0..j)` 的LCS。**由于要考虑空字符串，前闭后开。

- base case：如果i=0或者j=0，最长公共子序列是0。
- 如果`word1[i - 1] == word2[j - 1]`，`dp[i][j] = dp[i - 1][j - 1] + 1`。
- 如果不相等，需要往回找，往回退，在已经遍历过的里面找最大的。退的越多LCS越小，因此两个字符串中某一个退一格，取较大的那个。`dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);`

```java
	public int longestCommonSubsequence(String text1, String text2) {
        char[] s1 = text1.toCharArray();
        char[] s2 = text2.toCharArray();
        int[][] dp = new int[s1.length + 1][s2.length + 1];

        for (int i = 1; i < s1.length + 1; i++) {
            for (int j = 1; j < s2.length + 1; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[s1.length][s2.length];
    }
```

相关题目：

[718. 最长重复子数组](#718-最长重复子数组)



### [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

Difficulty: **中等**


我们在两条独立的水平线上按给定的顺序写下 `A` 和 `B` 中的整数。

现在，我们可以绘制一些连接两个数字 `A[i]` 和 `B[j]` 的直线，只要 `A[i] == B[j]`，且我们绘制的直线不与任何其他连线（非水平线）相交。

以这种方法绘制线条，并返回我们可以绘制的最大连线数。

**示例 1：**

<img src="images/动态规划/142.png" style="zoom: 33%;" />

```
输入：A = [1,4,2], B = [1,2,4]
输出：2
解释：
我们可以画出两条不交叉的线，如上图所示。
我们无法画出第三条不相交的直线，因为从 A[1]=4 到 B[2]=4 的直线将与从 A[2]=2 到 B[1]=2 的直线相交。
```

**示例 2：**

```
输入：A = [2,5,1,2,5], B = [10,5,2,1,5,2]
输出：3
```

**示例 3：**

```
输入：A = [1,3,7,1,7,5], B = [1,9,2,5,1]
输出：2
```

**提示：**

1.  `1 <= A.length <= 500`
2.  `1 <= B.length <= 500`
3.  `1 <= A[i], B[i] <= 2000`

**思路：不相交意味着保持先对顺序不变，即寻找最长公共子数组，和[1143. 最长公共子序列](#1143-最长公共子序列)本质上是一道题。**



### [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

Difficulty: **中等**


给两个整数数组 `A` 和 `B` ，返回两个数组中公共的、长度最长的子数组的长度。

**示例：**

```
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。
```

**提示：**

*   `1 <= len(A), len(B) <= 1000`
*   `0 <= A[i], B[i] < 100`

**建议和[1143. 最长公共子序列](#1143-最长公共子序列)一起做。**

**方法1：与[1143. 最长公共子序列](#1143-最长公共子序列)类似，思考两道题的不同点。**子数组要求连续，子序列可以间断。因此在1143LCS的基础上增加对结尾的要求，**定义`dp[i][j]`表示<font color=red>以`A[i]`和`B[j]`为结尾</font>的公共子数组的最长长度，最长重复子数组就是dp数组中的最大值。**

由于要考虑空数组，前闭后开。

- base case：如果i=0或者j=0，最长重复子数组是0。
- 如果`word1[i - 1] == word2[j - 1]`，`dp[i][j] = dp[i - 1][j - 1] + 1`，并且更新最长重复子数组。
- 如果不相等，不存在以当前为结尾的最长重复子数组，`dp[i][j] = 0`

```java
	public int findLength(int[] A, int[] B) {
        int res = 0;
        int[][] dp = new int[A.length + 1][B.length + 1];
        for (int i = 1; i <= A.length; i++) {
            for (int j = 1; j <= B.length; j++) {
                if (A[i - 1] == B[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    res = Math.max(res, dp[i][j]);
                }
            }
        }
        return res;
    }
```

**方法2：滑动窗口** 具体参考[题解](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/solution/wu-li-jie-fa-by-stg-2/)

![错开比较.gif](images/动态规划/9ed48b9b51214a8bafffcad17356d438b4c969b4999623247278d23f1e43977f-错开比较.gif)



### [583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

Difficulty: **中等**


给定两个单词 word1 和 word2，找到使得 word1和word2相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

**示例：**

```
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
```

**提示：**

1.  给定单词的长度不超过500。
2.  给定单词中的字符只含有小写字母。

**思路：还是那句话，解决双字符串这类题一般都是用两个指针 `i,j` 分别指向两个字符串的最后，然后一步步往前走，缩小问题的规模。**

**`dp[i][j]` 表示：将 `word1[0..i)` 与 `word2[0..j)` 的删除操作。**由于要考虑空字符串，前闭后开。

- base case：如果i=0或者j=0，一个是空字符串，另一个需要全删除来变为空。删除操作是另一个的长度。
- 如果`word1[i - 1] == word2[j - 1]`，不需要删除，`dp[i][j] = dp[i - 1][j - 1]`。
- 如果不相等，删除i或者j，哪边更小删除哪个。`dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1]);`

```java
	public int minDistance(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        for (int i = 1; i <= s1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= s2.length(); j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }

	//也可以把base case融到双层遍历中。
	public int minDistance(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = i + j;
                } else if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }
```



### [712. 两个字符串的最小ASCII删除和](https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings/)

Difficulty: **中等**


给定两个字符串`s1, s2`，找到使两个字符串相等所需删除字符的ASCII值的最小和。

**示例 1:**

```
输入: s1 = "sea", s2 = "eat"
输出: 231
解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
在 "eat" 中删除 "t" 并将 116 加入总和。
结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。
```

**示例 2:**

```
输入: s1 = "delete", s2 = "leet"
输出: 403
解释: 在 "delete" 中删除 "dee" 字符串变成 "let"，
将 100[d]+101[e]+101[e] 加入总和。在 "leet" 中删除 "e" 将 101[e] 加入总和。
结束时，两个字符串都等于 "let"，结果即为 100+101+101+101 = 403 。
如果改为将两个字符串转换为 "lee" 或 "eet"，我们会得到 433 或 417 的结果，比答案更大。
```

**注意:**

*   `0 < s1.length, s2.length <= 1000`。
*   所有字符串中的字符ASCII值在`[97, 122]`之间。

**思路：与[583. 两个字符串的删除操作](#583-两个字符串的删除操作)相同，理解了583一定会做这道题。**

**还是那句话，解决双字符串这类题一般都是用两个指针 `i,j` 分别指向两个字符串的最后，然后一步步往前走，缩小问题的规模。**

**`dp[i][j]` 表示：将 `word1[0..i)` 与 `word2[0..j)` 的最小ASCII删除和。**由于要考虑空字符串，前闭后开。

- base case：如果i=0或者j=0，一个是空字符串，另一个需要全删除来变为空。删除操作是另一个的ASCII和。
- 如果`word1[i - 1] == word2[j - 1]`，不需要删除，`dp[i][j] = dp[i - 1][j - 1]`。
- 如果不相等，删除i或者j，哪边ASCII删除和更小删除哪个。`dp[i][j] = Math.min(dp[i - 1][j] + s1.charAt(i - 1), dp[i][j - 1] + s2.charAt(j - 1));`

```java
	public int minimumDeleteSum(String s1, String s2) {
        int len1 = s1.length(), len2 = s2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        for (int i = 1; i < len1 + 1; i++) {
            dp[i][0] = dp[i - 1][0] + s1.charAt(i - 1);
        }
        for (int j = 1; j < len2 + 1; j++) {
            dp[0][j] = dp[0][j - 1] + s2.charAt(j - 1);
        }
        for (int i = 1; i < len1 + 1; i++) {
            for (int j = 1; j < len2 + 1; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + s1.charAt(i - 1),
                            			dp[i][j - 1] + s2.charAt(j - 1));
                }
            }
        }
        return dp[len1][len2];
    }
```



### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

Difficulty: **困难**


给你两个单词 `word1` 和 `word2`，请你计算出将 `word1` 转换成 `word2`所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

*   插入一个字符
*   删除一个字符
*   替换一个字符

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

**提示：**

*   `0 <= word1.length, word2.length <= 500`
*   `word1` 和 `word2` 由小写英文字母组成

**最好先做了[1143. 最长公共子序列](#1143-最长公共子序列)，[583. 两个字符串的删除操作](#583-两个字符串的删除操作)，再做这道题。**

**思路：比较难。是一道经典的两个字符串动态规划问题。解决这类题一般都是用两个指针 `i,j` 分别指向两个字符串的最后，然后一步步往前走，缩小问题的规模**。

**`dp[i][j]` 表示：将 `word1[0..i)` 转换成为 `word2[0..j)` 的最小编辑距离。**由于要考虑空字符串，前闭后开。

- base case：如果i=0或者j=0，把一个字符串变为另一个的编辑距离是另一个字符串的长度。
- 如果`word1[i - 1] == word2[j - 1]`，则最小编辑距离较`dp[i-1][j-1]`没有增加，`dp[i][j] = dp[i - 1][j - 1]`。
- 如果不相等，`dp[i-1][j-1] `表示替换操作，`dp[i-1][j] `表示删除操作，`dp[i][j-1] `表示插入操作。`dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1)`

```java
public int minDistance(String word1, String word2) {
    int n1 = word1.length();
    int n2 = word2.length();
    int[][] dp = new int[n1 + 1][n2 + 1];
    // 第一行 word1为""
    for (int j = 1; j <= n2; j++) {
        dp[0][j] = dp[0][j - 1] + 1;
    }
    // 第一列 word2为""
    for (int i = 1; i <= n1; i++) {
        dp[i][0] = dp[i - 1][0] + 1;
    }

    for (int i = 1; i <= n1; i++) {
        for (int j = 1; j <= n2; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
            }
        }
    }
    return dp[n1][n2];
}
```

**进阶：如何还原出最小编辑步骤呢？**

只需要在动态规划的过程中记录信息，把int类型的dp数组换成自定义类Node的数组，Node除了有最小距离的记录，还记录着操作以及指向上一步的Node，最终倒着回退，递归后续遍历即可还原路径和操作。

```java
// int[][] dp;
Node[][] dp;

class Node {
    int val;
    // 0 代表啥都不做
    // 1 代表插入
    // 2 代表删除
    // 3 代表替换
    int choice;
    Node prev; //上一个操作
}
```

相关题如下：

### [161. 相隔为1的编辑距离](https://leetcode-cn.com/problems/one-edit-distance/)

Difficulty: **中等**


给定两个字符串s和t，判断他们的编辑距离是否为 1。

**注意：**

满足编辑距离等于 1 有三种可能的情形：

1.  往 _**s**_ 中插入一个字符得到 _**t**_
2.  从**s**中删除一个字符得到 _**t**_
3.  在 _**s**_ 中替换一个字符得到 _**t**_

**示例 1：**

```
输入: s = "ab", t = "acb"
输出: true
解释: 可以将 'c' 插入字符串 s 来得到 t。
```

**示例 2:**

```
输入: s = "cab", t = "ad"
输出: false
解释: 无法通过 1 步操作使 s 变为 t。
```

**示例 3:**

```
输入: s = "1203", t = "1213"
输出: true
解释: 可以将字符串 s 中的 '0' 替换为 '1' 来得到 t。
```

**思路：直接比较即可。理解了[72. 编辑距离](#72-编辑距离)这道题就可以迎刃而解。**

```java
	public boolean isOneEditDistance(String s, String t) {
        int len1 = s.length(), len2 = t.length();
        if (Math.abs(len1 - len2) > 1) {
            return false;
        }
        for (int i = 0; i < Math.min(len1, len2); i++) {
            if (s.charAt(i) != t.charAt(i)) {
                //给s插入一个字符
                return s.substring(i).equals(t.substring(i + 1)) || s.substring(i + 1).equals(t.substring(i + 1)) || s.substring(i + 1).equals(t.substring(i));

            }
        }
        return len1 != len2;
    }
```



### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

Difficulty: **困难**


给定一个字符串 (`s`) 和一个字符模式 (`p`) ，实现一个支持 `'?'` 和 `'*'` 的通配符匹配。

```
'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
```

两个字符串**完全匹配**才算匹配成功。

**说明:**

*   `s` 可能为空，且只包含从 `a-z` 的小写字母。
*   `p` 可能为空，且只包含从 `a-z` 的小写字母，以及字符 `?` 和 `*`。

**示例 1:**

```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入:
s = "aa"
p = "*"
输出: true
解释: '*' 可以匹配任意字符串。
```

**示例 3:**

```
输入:
s = "cb"
p = "?a"
输出: false
解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。
```

**示例 4:**

```
输入:
s = "adceb"
p = "*a*b"
输出: true
解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".
```

**示例 5:**

```
输入:
s = "acdcb"
p = "a*c?b"
输出: false
```

**思路：**

**boolean `dp[i][j]`表示从第0个元素开始，s长度为i，p长度为j，能否匹配。**

basecase：如果s长度为0，p必须全都是*才行。

最简单的情况：

`s.charAt(i - 1) == p.charAt(j - 1)` 或者`p.charAt(j - 1) == '?'`，则`dp[i][j] = f[i - 1][j - 1];`

其余情况，除非j处是*，才有可能匹配 可能匹配零个或多个

**匹配零个：`f[i][j] = f[i][j - 1]`** 

**匹配多个：s往回退一格即可，`f[i - 1][j]`，退回1格后p不动，又可以匹配0个或多个。**

```java
	public boolean isMatch(String s, String p) {
        int m = p.length(), n = s.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++) {
            if (p.charAt(i - 1) != '*') {
                break;
            }
            dp[i][0] = true;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(i - 1) == s.charAt(j - 1) || p.charAt(i - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(i - 1) == '*') {
                    //匹配多个 dp[i - 1][j]
                    //匹配0个 dp[i][j - 1]
                    dp[i][j] = dp[i - 1][j] | dp[i][j - 1];
                }
            }
        }
        return dp[m][n];
    }
```



### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

Difficulty: **困难**


给你一个字符串 `s` 和一个字符规律 `p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。

*   `'.'` 匹配任意单个字符
*   `'*'` 匹配零个或多个前面的那一个元素

所谓匹配，是要涵盖 **整个 **字符串 `s`的，而不是部分字符串。

**示例 1：**

```
输入：s = "aa" p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入：s = "aa" p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**示例 3：**

```
输入：s = "ab" p = ".*"
输出：true
解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

**示例 4：**

```
输入：s = "aab" p = "c*a*b"
输出：true
解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
```

**示例 5：**

```
输入：s = "mississippi" p = "mis*is*p*."
输出：false
```

**提示：**

*   `0 <= s.length <= 20`
*   `0 <= p.length <= 30`
*   `s` 可能为空，且只包含从 `a-z` 的小写字母。
*   `p` 可能为空，且只包含从 `a-z` 的小写字母，以及字符 `.` 和 `*`。
*   保证每次出现字符 `*` 时，前面都匹配到有效的字符

**思路：与[44. 通配符匹配](#44-通配符匹配)类似，如果能完全理解44题，这道题应该不难**。

**boolean `dp[i][j]`表示从第0个元素开始，s长度为i，p长度为j，能否匹配。**

最简单的情况：

`s.charAt(i - 1) == p.charAt(j - 1)` 或者`p.charAt(j - 1) == '.'`，则`dp[i][j] = f[i - 1][j - 1];`

其余情况，除非j处是*，才有可能匹配 可能匹配零个或多个

**匹配零个：`f[i][j] = f[i][j - 2]`** 

**匹配多个：s往回退一格即可，`f[i - 1][j]`，退回1格后p不动，又可以匹配0个或多个。注意有个前提条件是：`s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.'`**

```java
	public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] f = new boolean[m + 1][n + 1];

        f[0][0] = true;
        //basecase: s是空字符串，p从charAt(1)开始全是*才可以
        for (int i = 2; i <= n; i++) {
            dp[0][i] = dp[0][i - 2] && p.charAt(i - 1) == '*';
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    f[i][j] = f[i - 1][j - 1];
                }
                //当前是*，可以匹配0个或多个。
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2] ||
                            f[i - 1][j] && (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.');
                }
            }
        }
        return f[m][n];
    }
```



## 背包问题

**核心：dp[i]表示以当前位置为结尾的最...。尝试用前面的可行解填充当前的。**

其实这类题用回溯法也同样能解，但是如果剪枝不够完全，就会超时。回溯是递归的一种，又回到了那个问题，动态规划是带记忆化的回溯。

### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

Difficulty: **中等**


给定不同面额的硬币 `coins` 和一个总金额 `amount`。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

你可以认为每种硬币的数量是无限的。

**示例 1：**

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

**示例 2：**

```
输入：coins = [2], amount = 3
输出：-1
```

**示例 3：**

```
输入：coins = [1], amount = 0
输出：0
```

**示例 4：**

```
输入：coins = [1], amount = 1
输出：1
```

**示例 5：**

```
输入：coins = [1], amount = 2
输出：2
```

**提示：**

*   `1 <= coins.length <= 12`
*   1 <= coins[i] <= 2<sup>31</sup> - 1
*   0 <= amount <= 10<sup>4</sup>

**思路：类似于完全背包问题，cost数组填充dp数组**

**dp[i]表示组成i的最少硬币个数，dp[i] = 1 + Math.min(dp[i-amout[0]], dp[i-amout[1]]....dp[i-amount[amount.length-1]])，前提是i<=amount并且dp[i-amout]有效。**

```java
public int coinChange(int [] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;

    for(int i = 0; i < dp.length; i++){
        for(int coin : coins){
            if(i >= coin){
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] == amount + 1 ? -1 : dp[amount];
}
```

**评论区遍历剪枝的方法：** 

```java
class Solution {
   int res = Integer.MAX_VALUE;

    public int coinChange(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }
        Arrays.sort(coins);
        mincoin(coins, amount, coins.length - 1, 0);
        return res == Integer.MAX_VALUE ? -1 : res;
    }

    private void mincoin(int[] coins, int amount, int index, int count) {
        if (amount == 0) {
            res = Math.min(res, count);
            return;
        }
        if (index < 0) {
            return;
        }
        for(int i = amount/coins[index];i>=0 && i+count<res; i--){//4ms
            mincoin(coins, amount - (i * coins[index]), index - 1, count + i);
        }
    }
}
```



**相关题：**

[279. 完全平方数](#279-完全平方数)

[983. 最低票价](#983-最低票价)



### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

Difficulty: **中等**


给定正整数 _n_，找到若干个完全平方数（比如 `1, 4, 9, 16, ...`）使得它们的和等于 _n_。你需要让组成和的完全平方数的个数最少。

给你一个整数 `n` ，返回和为 `n` 的完全平方数的 **最少数量** 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

**示例 1：**

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**示例 2：**

```
输入：n = 13
输出：2
解释：13 = 4 + 9
```

**提示：**

*   1 <= n <= 10<sup>4</sup

**思路：类似[322. 零钱兑换](#322-零钱兑换)，用背包填充，尝试用前面的可行解填充当前的。**

```java
    public int numSquares(int n) {
        int[] dp = new int[n + 1]; // 默认初始化值都为0
        for (int i = 1; i <= n; i++) {
            dp[i] = i; // 最坏的情况就是每次+1
            for (int j = 1; i - j * j >= 0; j++) {
                // 尝试用前面的可行解填充当前的
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1); 
            }
        }
        return dp[n];
    }
```

其他方法：BFS，按层级遍历，找到的第一个就是最短的。注意用visited数组避免重复计算。

```java
    public int numSquares(int n) {
        Queue<Integer> queue = new LinkedList<>();
        HashSet<Integer> visited = new HashSet<>();
        int level = 0;
        queue.add(n);
        while (!queue.isEmpty()) {
            int size = queue.size();
            level++; // 开始生成下一层
            for (int i = 0; i < size; i++) {
                int cur = queue.poll();
                //依次减 1, 4, 9... 生成下一层的节点
                for (int j = 1; j * j <= cur; j++) {
                    int next = cur - j * j;
                    if (next == 0) {
                        return level;
                    }
                    if (!visited.contains(next)) {
                        queue.offer(next);
                        visited.add(next);
                    }
                }
            }
        }
        return -1;
    }
```

### [983. 最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/)

Difficulty: **中等**


在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 `days` 的数组给出。每一项是一个从 `1` 到 `365` 的整数。

火车票有三种不同的销售方式：

*   一张为期一天的通行证售价为 `costs[0]` 美元；
*   一张为期七天的通行证售价为 `costs[1]` 美元；
*   一张为期三十天的通行证售价为 `costs[2]` 美元。

通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张为期 7 天的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。

返回你想要完成在给定的列表 `days` 中列出的每一天的旅行所需要的最低消费。

**示例 1：**

```
输入：days = [1,4,6,7,8,20], costs = [2,7,15]
输出：11
解释： 
例如，这里有一种购买通行证的方法，可以让你完成你的旅行计划：
在第 1 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 1 天生效。
在第 3 天，你花了 costs[1] = $7 买了一张为期 7 天的通行证，它将在第 3, 4, ..., 9 天生效。
在第 20 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 20 天生效。
你总共花了 $11，并完成了你计划的每一天旅行。
```

**示例 2：**

```
输入：days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
输出：17
解释：
例如，这里有一种购买通行证的方法，可以让你完成你的旅行计划： 
在第 1 天，你花了 costs[2] = $15 买了一张为期 30 天的通行证，它将在第 1, 2, ..., 30 天生效。
在第 31 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 31 天生效。 
你总共花了 $17，并完成了你计划的每一天旅行。
```

**提示：**

1.  `1 <= days.length <= 365`
2.  `1 <= days[i] <= 365`
3.  `days` 按顺序严格递增
4.  `costs.length == 3`
5.  `1 <= costs[i] <= 1000`

**思路：类似[322. 零钱兑换](#322-零钱兑换)，用背包填充，cost数组填充dp数组，尝试用前面的可行解填充当前的。**

- 前面买7天或者30天的可能覆盖后面的，后面的结果依赖于前面的结果，因此dp当前值要回看之前值。

- dp[i]表示到了第i天的最低票价，dp[i] = Math.min(cost[0]+dp[i-1], cost[1]+dp[i-7], cost[2]+dp[i-30])，前提是dp[i-1]、dp[i-7]、dp[i-30]有效，如果无效，直接买一张1、7、30天票。

- 对于days中不存在的天数，不需要买票，最低票价等于前一天的最低票价。dp[i] = dp[i-1]

```java
	public int mincostTickets(int[] days, int[] costs) {
        //dp[i]表示到了第i天的最低票价
        int[] dp = new int[days[days.length - 1] + 1];

        //base case: 第0天一定不用买票 则花费0元
        dp[0] = 0;
        //标记一下需要买票的日子
        for (int day : days) {
            dp[day] = Integer.MAX_VALUE;
        }

        for (int i = 1; i < dp.length; i++) {
            //不需要买票
            if (dp[i] == 0) {
                //不需要出行的时候就是前一天花的钱
                dp[i] = dp[i - 1];
                continue;
            }
            //当天需要买票
            int n1 = dp[i - 1] + costs[0];
            //7天前能到达，就用7天前+今天新买。到达不了，就今天直接买一张7天票。
            int n2 = i > 7 ? dp[i - 7] + costs[1] : costs[1];
            //30天与7天 同理
            int n3 = i > 30 ? dp[i - 30] + costs[2] : costs[2];

            dp[i] = Math.min(n1, Math.min(n2, n3));
        }
        //最后一天花费多少钱
        return dp[days[days.length - 1]];
    }
```



## 其他

### [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

Difficulty: **中等**


在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

**示例 1：**

![](images/动态规划/max1grid.jpg)

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4
```

**示例 2：**

![](images/动态规划/max2grid.jpg)

```
输入：matrix = [["0","1"],["1","0"]]
输出：1
```

**示例 3：**

```
输入：matrix = [["0"]]
输出：0
```

**提示：**

*   `m == matrix.length`
*   `n == matrix[i].length`
*   `1 <= m, n <= 300`
*   `matrix[i][j]` 为 `'0'` 或 `'1'`

**思路：一定是要用动态固化的，不然不利用子问题的话，这道题复杂度太高。关键是怎么定义状态以及寻找状态方程呢？参考[题解](https://leetcode-cn.com/problems/maximal-square/solution/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/)**

![image.png](images/动态规划/221-最大正方形.png)

`dp(i, j)` 是以 `matrix(i - 1, j - 1)` 为 **右下角** 的正方形的最大边长。等同于：`dp(i + 1, j + 1)` 是以 `matrix(i, j)` 为右下角的正方形的最大边长。

就像木桶的短板理论那样——附近的最小边长，才与 ? 的最长边长有关。

递推公式：

如果`grid[i - 1][j - 1] = '1'`，`dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;`三个正方形中最小的一个+1。

```java
	public int maximalSquare(char[][] matrix) {
        // base condition
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) return 0;

        int height = matrix.length;
        int width = matrix[0].length;
        int maxSide = 0;

        // 相当于已经预处理新增第一行、第一列均为0
        int[][] dp = new int[height + 1][width + 1];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                if (matrix[row][col] == '1') {
                    dp[row + 1][col + 1] = Math.min(Math.min(dp[row + 1][col], dp[row][col + 1]), dp[row][col]) + 1;
                    maxSide = Math.max(maxSide, dp[row + 1][col + 1]);
                }
            }
        }
        return maxSide * maxSide;
    }
```





## 扔鸡蛋问题

超时方法：

```java
public int superEggDrop(int K, int N) {
    int[][] middleResults = new int[K + 1][N + 1];
    for (int i = 1; i <= N; i++) {
        middleResults[1][i] = i; // only one egg
        middleResults[0][i] = 0; // no egg
    }
    for (int i = 1; i <= K; i++) {
        middleResults[i][0] = 0; // zero floor
    }

    for (int k = 2; k <= K; k++) { // start from two egg
        for (int n = 1; n <= N; n++) {
            int tMinDrop = N * N;
            for (int x = 1; x <= n; x++) {
                tMinDrop = Math.min(tMinDrop, 1 + Math.max(middleResults[k - 1][x - 1], middleResults[k][n - x]));
            }
            middleResults[k][n] = tMinDrop;
        }
    }

    return middleResults[K][N];
}
```

```java
import java.util.Arrays;

public class Solution {

    public int superEggDrop(int K, int N) {
        // dp[i][j]：一共有 i 层楼梯的情况下，使用 j 个鸡蛋的最少仍的次数
        int[][] dp = new int[N + 1][K + 1];
        
        // 初始化
        for (int i = 0; i <= N; i++) {
            Arrays.fill(dp[i], i);
        }
        for (int j = 0; j <= K; j++) {
            dp[0][j] = 0;
        }

        dp[1][0] = 0;
        for (int j = 1; j <= K; j++) {
            dp[1][j] = 1;
        }
        for (int i = 0; i <= N; i++) {
            dp[i][0] = 0;
            dp[i][1] = i;
        }

        // 开始递推
        for (int i = 2; i <= N; i++) {
            for (int j = 2; j <= K; j++) {
                // 在区间 [1, i] 里确定一个最优值
                int left = 1;
                int right = i;
                while (left < right) {
                    // 找 dp[k - 1][j - 1] <= dp[i - mid][j] 的最大值 k
                    int mid = left + (right - left + 1) / 2;
                    
                    int breakCount = dp[mid - 1][j - 1];
                    int notBreakCount = dp[i - mid][j];
                    if (breakCount > notBreakCount) {
                        // 排除法（减治思想）写对二分见第 35 题，先想什么时候不是解
                        // 严格大于的时候一定不是解，此时 mid 一定不是解
                        // 下一轮搜索区间是 [left, mid - 1]
                        right = mid - 1;
                    } else {
                        // 这个区间一定是上一个区间的反面，即 [mid, right]
                        // 注意这个时候取中间数要上取整，int mid = left + (right - left + 1) / 2;
                        left = mid;
                    }
                }
                // left 这个下标就是最优的 k 值，把它代入转移方程 Math.max(dp[k - 1][j - 1], dp[i - k][j]) + 1) 即可
                dp[i][j] = Math.max(dp[left - 1][j - 1], dp[i - left][j]) + 1;
            }
        }
        return dp[N][K];
    }
}

作者：liweiwei1419
链接：https://leetcode-cn.com/problems/super-egg-drop/solution/dong-tai-gui-hua-zhi-jie-shi-guan-fang-ti-jie-fang/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



```java
class Solution {
    public int superEggDrop(int K, int N) {
        // only one egg situation
        int[] dp = new int[N + 1];
        for (int i = 0; i <= N; ++i)
            dp[i] = i;

        // two and more eggs
        for (int k = 2; k <= K; ++k) {
            int[] dp2 = new int[N + 1];
            int x = 1; // start from floor 1
            for (int n = 1; n <= N; ++n) {
                // start to calculate from bottom
                // Notice max(dp[x-1], dp2[n-x]) > max(dp[x], dp2[n-x-1])
                // is simply max(T1(x-1), T2(x-1)) > max(T1(x), T2(x)).
                while (x < n && Math.max(dp[x - 1], dp2[n - x]) > Math.max(dp[x], dp2[n - x - 1]))
                    x++;

                // The final answer happens at this x.
                dp2[n] = 1 + Math.max(dp[x - 1], dp2[n - x]);
            }

            dp = dp2;
        }

        return dp[N];
    }
}

作者：shellbye
链接：https://leetcode-cn.com/problems/super-egg-drop/solution/ji-dan-diao-luo-xiang-jie-by-shellbye/
```

