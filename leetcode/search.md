### DFS 深度优先搜索

[200. 岛屿数量](#200-岛屿数量)

​	[695. 岛屿的最大面积](#695-岛屿的最大面积)

​	[463. 岛屿的周长](#463-岛屿的周长)



### 记忆化DFS

[329. 矩阵中的最长递增路径](#329-矩阵中的最长递增路径)



### BFS

[752. 打开转盘锁](#752-打开转盘锁)



**一般可以抽象为图的问题，用BFS、DFS、Union Find这三种解法。**BFS适合求最短问题。



## DFS 深度优先搜索

    private void dfs(char[][] grid, boolean[][] visited, int i, int j){
    	//基础条件 越界或者遍历过，返回
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || visited[i][j]){
            return;
        }
        //避免重复
        visited[i][j] = true;
        if(grid[i][j] == '1'){
            //DFS深度优先搜索
            dfs(grid, i + 1, j);
            dfs(grid, i - 1, j);
            dfs(grid, i, j + 1);
            dfs(grid, i, j - 1);
        }
    }


### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

Difficulty: **中等**


给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**示例 1：**

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

**提示：**

*   `m == grid.length`
*   `n == grid[i].length`
*   `1 <= m, n <= 300`
*   `grid[i][j]` 的值为 `'0'` 或 `'1'`

**方法1：**用DFS深度优先搜索，如果遇到1，就用DFS搜索所有相邻的1，搜索过程中把1变成其他值，防止重复。如果不想改变原数组，也可以用visited数组标记哪些已经遍历过了。

也可以用BFS，不过这道题用BFS没有DFS简洁，就不写了。

```java
public int numIslands(char[][] grid) {
    int res = 0;
    for(int i = 0; i < grid.length; i++){
        for(int j = 0; j < grid[0].length; j++){
            if(grid[i][j] == '1'){
                dfs(grid, i, j);
                res++;
            }
        }
    }
    return res;
}

private void dfs(char[][] grid, int i, int j){
    if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length){
        return;
    }
    if(grid[i][j] == '1'){
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
}
```

**方法2：**可以用Union并查集，把所有的1都连接在一起，最后看有几个连接。连接是只需要连接右边的和左边的，如果四个方向都连接会出现重复。

```java
class Solution {
    int[] parent;

    public int numIslands(char[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return 0;
        int row = grid.length;
        int col = grid[0].length;
        parent = new int[row * col];
        Arrays.fill(parent, -1);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    parent[i * col + j] = i * col + j;
                    // union current+top
                    if (i > 0 && grid[i - 1][j] == '1') {
                        union(i * col + j, (i - 1) * col + j);
                    }
                    // union current+left
                    if (j > 0 && grid[i][j - 1] == '1') {
                        union(i * col + j, i * col + (j - 1));
                    }
                }
            }
        }
        Set<Integer> set = new HashSet<>();
        //原来位置是0的parent[k]一定是-1，其余的只需要找有几个不同的parent即可。
        for (int k = 0; k < parent.length; k++) {
            if (parent[k] != -1) {
                set.add(find(k));
            }
        }
        return set.size();
    }

    private void union(int x, int y) {
        int px = find(x);
        int py = find(y);
        parent[px] = parent[py];
    }
	//沿路压缩
    private int find(int x) {
        if (parent[x] == x) {
            return x;
        }
        parent[x] = find(parent[x]);
        return parent[x];
    }
}
```

**相关高频题：**

[695. 岛屿的最大面积](#695-岛屿的最大面积)

[463. 岛屿的周长](#463-岛屿的周长)



### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

Difficulty: **中等**


给定一个包含了一些 `0` 和 `1` 的非空二维数组 `grid` 。

一个 **岛屿** 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在水平或者竖直方向上相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 `0` 。)

**示例 1:**

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
```

对于上面这个给定矩阵应返回 `6`。注意答案不应该是 `11` ，因为岛屿只能包含水平或垂直的四个方向的 `1` 。

**示例 2:**

```
[[0,0,0,0,0,0,0,0]]
```

对于上面这个给定的矩阵, 返回 `0`。

**注意: **给定的矩阵`grid` 的长度和宽度都不超过 50。

**方法1：**DFS，与[200. 岛屿数量](#200-岛屿数量)相同，把void返回类型的DFS换成int类型即可。

```java
	public int maxAreaOfIsland(int[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    res = Math.max(res, dfs(grid, i, j));
                }
            }
        }
        return res;
    }

    private int dfs(int[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length) {
            return 0;
        }
        if (grid[i][j] == 1) {
            grid[i][j] = 0;
            return 1 + dfs(grid, i + 1, j) + dfs(grid, i - 1, j) +
                    dfs(grid, i, j + 1) + dfs(grid, i, j - 1);
        }
        return 0;
    }
```

**方法2：**Union，这里用parent数组的相反数代表数量。

```java
	public int maxAreaOfIsland(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return 0;
        int row = grid.length;
        int col = grid[0].length;
        parent = new int[row * col];
        Arrays.fill(parent, Integer.MAX_VALUE);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    parent[i * col + j] = -1;
                    // union current+top
                    if (i > 0 && grid[i - 1][j] == 1) {
                        union(i * col + j, (i - 1) * col + j);
                    }
                    // union current+left
                    if (j > 0 && grid[i][j - 1] == 1) {
                        union(i * col + j, i * col + (j - 1));
                    }

                }
            }
        }
        int max = 0;
        for (int num : parent) {
            max = Math.max(max, -num);
        }
        return max;
    }

    int[] parent;

    private void union(int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px != py) {
            parent[px] = parent[py] + parent[px];
            parent[py] = px;
        }
    }

    private int find(int x) {
        if (parent[x] < 0) return x;
        parent[x] = find(parent[x]);
        return parent[x];
    }
```



### [463. 岛屿的周长](https://leetcode-cn.com/problems/island-perimeter/)

Difficulty: **简单**


给定一个 `row x col` 的二维网格地图 `grid` ，其中：`grid[i][j] = 1` 表示陆地， `grid[i][j] = 0` 表示水域。

网格中的格子 **水平和垂直** 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。

岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

**示例 1：**

![](images/搜索/island.png)

```
输入：grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
输出：16
解释：它的周长是上面图片中的 16 个黄色的边
```

**示例 2：**

```
输入：grid = [[1]]
输出：4
```

**示例 3：**

```
输入：grid = [[1,0]]
输出：4
```

**提示：**

*   `row == grid.length`
*   `col == grid[i].length`
*   `1 <= row, col <= 100`
*   `grid[i][j]` 为 `0` 或 `1`

**方法1：DFS。**从岛屿遇到边缘或者湖面，边的数量就+1。注意不能把岛屿换成0，不然会混。可以换成别的值，也可以用boolean visited数组。

```java
	public int islandPerimeter(int[][] grid) {
        int res = 0;
        if (grid == null || grid.length == 0) return res;
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    return dfs(grid, i, j, visited);
                }
            }
        }
        return res;
    }

    private final static int[] move = new int[]{0, 1, 0, -1, 0};

    private int dfs(int[][] grid, int i, int j, boolean[][] visited) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0) return 1;
        if (visited[i][j]) return 0;
        visited[i][j] = true;
        int res = 0;
        for (int k = 0; k < 4; k++) {
            int r = i + move[k];
            int c = j + move[k + 1];
            res += dfs(grid, r, c, visited);
        }
        return res;
    }
```

方法2：由于只有一个岛屿，可以遍历统计。周长 = 土地数 * 4 - 2 * 土地接壤数 

```java
   public int islandPerimeter(int[][] grid) {
        int land = 0; // 土地个数
        int border = 0; // 接壤边界的条数

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    land++;
                    if (i < grid.length - 1 && grid[i + 1][j] == 1) {
                        border++;
                    }
                    if (j < grid[0].length - 1 && grid[i][j + 1] == 1) {
                        border++;
                    }
                }
            }
        }
        return 4 * land - 2 * border;
    }
```



## 记忆化DFS

### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

Difficulty: **困难**


给定一个 `m x n` 整数矩阵 `matrix` ，找出其中 **最长递增路径** 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 **不能** 在 **对角线** 方向上移动或移动到 **边界外**（即不允许环绕）。

**示例 1：**

![](images/搜索/grid1.jpg)

```
输入：matrix = [[9,9,4],[6,6,8],[2,1,1]]
输出：4 
解释：最长递增路径为 [1, 2, 6, 9]。
```

**示例 2：**

![](images/搜索/tmp-grid.jpg)

```
输入：matrix = [[3,4,5],[3,2,6],[2,2,1]]
输出：4 
解释：最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
```

**示例 3：**

```
输入：matrix = [[1]]
输出：1
```

**提示：**

*   `m == matrix.length`
*   `n == matrix[i].length`
*   `1 <= m, n <= 200`
*   0 <= `matrix[i][j]` <= 2<sup>31</sup> - 1

思路：记忆化DFS。

如果一个位置已经计算出来了它的最长严格递增，相邻的比它小的数走它这条路可以直接用它的结果。

由于严格递增，不需要用visited数组来避免遍历重复的路（不严格递增的话，如果有相邻相等元素，不用visited数组的话会一条路来回走死循环）。

```java
class Solution {
    public static final int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix.length == 0) return 0;
        int m = matrix.length, n = matrix[0].length;
        int[][] cache = new int[m][n];
        int max = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int len = dfs(matrix, i, j, m, n, cache);
                max = Math.max(max, len);
            }
        }
        return max;
    }

    public int dfs(int[][] matrix, int i, int j, int m, int n, int[][] cache) {
        if (cache[i][j] != 0) return cache[i][j];
        int max = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] <= matrix[i][j]) continue;
            int len = 1 + dfs(matrix, x, y, m, n, cache);
            max = Math.max(max, len);
        }
        cache[i][j] = max;
        return max;
    }
}
```



## BFS

### [752. 打开转盘锁](https://leetcode-cn.com/problems/open-the-lock/)

Difficulty: **中等**


你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'` 。每个拨轮可以自由旋转：例如把 `'9'` 变为  `'0'`<font face="Helvetica Neue, Helvetica, Arial, sans-serif" color="#333333" style="display: inline;"><span style="background-color: rgb(255, 255, 255); font-size: 14px; display: inline;">，</span></font>`'0'` 变为 `'9'` 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 `'0000'` ，一个代表四个拨轮的数字的字符串。

列表 `deadends` 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 `target` 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

**示例 1:**

```
输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
```

**示例 2:**

```
输入: deadends = ["8888"], target = "0009"
输出：1
解释：
把最后一位反向旋转一次即可 "0000" -> "0009"。
```

**示例 3:**

```
输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
输出：-1
解释：
无法旋转到目标数字且不被锁定。
```

**示例 4:**

```
输入: deadends = ["0000"], target = "8888"
输出：-1
```

**提示：**

1.  死亡列表 `deadends` 的长度范围为 `[1, 500]`。
2.  目标数字 `target` 不会在 `deadends` 之中。
3.  每个 `deadends` 和 `target` 中的字符串的数字会在 10,000 个可能的情况 `'0000'` 到 `'9999'` 中产生。

**思路：BFS**。[这个题解不错](https://leetcode-cn.com/problems/open-the-lock/solution/wo-xie-liao-yi-tao-bfs-suan-fa-kuang-jia-jian-dao-/)，了解一下里面的双向BFS。

```java
public int openLock(String[] deadends, String target) {
    if(target==null || target.length()==0){
        return -1;
    }
    Set<String> deads=new HashSet<>(Arrays.asList(deadends));
    String start="0000";
    if(deads.contains(target) || deads.contains(start)){
        return -1;
    }

    Queue<String> queue = new LinkedList<>();
    Set<String> visited=new HashSet<>();
    queue.offer(start);
    visited.add(start);
    int step=0;
    while(!queue.isEmpty()){
        for(int i=queue.size(); i>0; i--){
            String cur=queue.poll();
            if(target.equals(cur)){ // 找到了目标返回步骤数
                return step;
            }
            List<String> nexts=getNexts(cur);
            for(String str:nexts){
                if(!deads.contains(str) && visited.add(str)){
                    queue.offer(str);
                }
            }
        }
        step++;
    }
    return -1;
}

// 获得当前值转动一位可以转动到的所有情况
private List<String> getNexts(String cur){
    List<String> list=new ArrayList<>();
    for(int i=0; i<4; i++){
        StringBuilder curBuilder=new StringBuilder(cur);
        char modChar=cur.charAt(i);
        curBuilder.setCharAt(i,modChar=='0'?'9':(char)(modChar-1));
        list.add(curBuilder.toString());
        curBuilder.setCharAt(i,modChar=='9'?'0':(char)(modChar+1));
        list.add(curBuilder.toString());
    }
    return list;
}
```

