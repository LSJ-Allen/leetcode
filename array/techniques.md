# Array Problem-Solving Techniques Guide

## 1. Two Pointers Technique

### Core Concept
A method that uses two references to traverse an array, processing elements by comparing or manipulating them based on their relationship.

### Variations & Use Cases

1. **Fast & Slow Pointers (Floyd's Technique)**
   - **When to use:** 
     - Finding middle element
     - Cycle detection
     - Finding duplicates
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(fast && fast->next) {
         slow = slow->next;
         fast = fast->next->next;
     }
     ```

2. **Left & Right Pointers (Converging Pointers)**
   - **When to use:**
     - Searching pairs in sorted array
     - Partitioning arrays
     - Two-sum variations
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(left < right) {
         if(condition) left++;
         else right--;
     }
     ```

3. **Same Direction Pointers**
   - **When to use:**
     - In-place array modifications
     - Removing duplicates
     - Maintaining window
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     for(fast = 0; fast < n; fast++) {
         if(condition) {
             swap(nums[slow++], nums[fast]);
         }
     }
     ```

### LeetCode Problems

#### Easy
1. **[167] Two Sum II - Input Array Is Sorted**
   - **Approach:** Left & Right pointers
   - **Pattern:** Compare sum with target
   ```cpp
   while(left < right) {
       sum = nums[left] + nums[right];
       if(sum == target) return {left+1, right+1};
       sum < target ? left++ : right--;
   }
   ```

2. **[283] Move Zeroes**
   - **Approach:** Same Direction pointers
   - **Pattern:** Partition non-zeros and zeros
   ```cpp
   for(int i = 0; i < n; i++) {
       if(nums[i] != 0) {
           swap(nums[nonZeroPos++], nums[i]);
       }
   }
   ```

#### Medium
1. **[11] Container With Most Water**
   - **Approach:** Left & Right pointers
   - **Pattern:** Area maximization
   ```cpp
   while(left < right) {
       maxArea = max(maxArea, min(height[left], height[right]) * (right-left));
       height[left] < height[right] ? left++ : right--;
   }
   ```

2. **[75] Sort Colors**
   - **Approach:** Three pointers (Dutch Flag)
   - **Pattern:** Three-way partitioning
   ```cpp
   while(mid <= high) {
       if(nums[mid] == 0) swap(nums[low++], nums[mid++]);
       else if(nums[mid] == 2) swap(nums[mid], nums[high--]);
       else mid++;
   }
   ```

#### Hard
1. **[42] Trapping Rain Water**
   - **Approach:** Left & Right pointers with max tracking
   - **Pattern:** Two-way comparison
   ```cpp
   while(left < right) {
       if(height[left] < height[right]) {
           height[left] >= leftMax ? leftMax = height[left] : water += leftMax - height[left];
           left++;
       } else {
           // Similar for right side
       }
   }
   ```

## 2. Sliding Window Technique

### Core Concept
A technique to process array elements in a window that slides through the array, maintaining a set of elements that satisfy certain conditions.

### Variations & Use Cases

1. **Fixed Size Window**
   - **When to use:**
     - Subarray of exact size k
     - Average/sum calculations
     - Pattern matching of fixed length
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     for(int i = 0; i < k; i++) windowSum += nums[i];
     for(int i = k; i < n; i++) {
         windowSum += nums[i] - nums[i-k];
         // Process window
     }
     ```

2. **Variable Size Window**
   - **When to use:**
     - Longest/shortest subarray meeting condition
     - Sum/product constraints
     - String pattern matching
   - **Time:** O(n), **Space:** O(1) or O(k)
   - **Example Pattern:**
     ```cpp
     while(right < n) {
         // Add element to window
         while(!valid()) {
             // Remove element from window
             left++;
         }
         // Update result
         right++;
     }
     ```

## 3. Prefix Sum Technique

### Core Concept
A preprocessing technique that builds an auxiliary array containing cumulative sums, enabling efficient range queries and subarray calculations.

### Variations & Use Cases

1. **1D Prefix Sum**
   - **When to use:**
     - Range sum queries
     - Subarray sum calculations
     - Cumulative statistics
   - **Time:** O(1) per query after O(n) preprocessing
   - **Example Pattern:**
     ```cpp
     for(int i = 1; i < n; i++) {
         prefix[i] = prefix[i-1] + nums[i];
     }
     // Query: sum(i,j) = prefix[j] - prefix[i-1]
     ```

2. **2D Prefix Sum**
   - **When to use:**
     - Rectangle sum queries
     - 2D range calculations
     - Matrix region processing
   - **Time:** O(1) per query after O(m*n) preprocessing
   - **Example Pattern:**
     ```cpp
     for(int i = 1; i <= m; i++) {
         for(int j = 1; j <= n; j++) {
             dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
         }
     }
     ```

3. **Difference Array**
   - **When to use:**
     - Range update operations
     - Interval modifications
     - Frequency counting in ranges
   - **Time:** O(1) per update, O(n) for final array
   - **Example Pattern:**
     ```cpp
     void addToRange(int l, int r, int val) {
         diff[l] += val;
         diff[r + 1] -= val;
     }
     ```

### LeetCode Problems

#### Easy
1. **[303] Range Sum Query - Immutable**
   - **Approach:** 1D Prefix Sum
   - **Pattern:** Range query optimization
   ```cpp
   vector<int> prefixSum;
   NumArray(vector<int>& nums) {
       prefixSum = vector<int>(nums.size() + 1);
       for(int i = 0; i < nums.size(); i++)
           prefixSum[i + 1] = prefixSum[i] + nums[i];
   }
   int sumRange(int i, int j) {
       return prefixSum[j + 1] - prefixSum[i];
   }
   ```

#### Medium
1. **[304] Range Sum Query 2D - Immutable**
   - **Approach:** 2D Prefix Sum
   - **Pattern:** 2D range query
   ```cpp
   NumMatrix(vector<vector<int>>& matrix) {
       if(matrix.empty()) return;
       dp.resize(matrix.size() + 1, vector<int>(matrix[0].size() + 1));
       for(int i = 1; i <= matrix.size(); i++)
           for(int j = 1; j <= matrix[0].size(); j++)
               dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
   }
   ```

2. **[560] Subarray Sum Equals K**
   - **Approach:** Prefix Sum with Hash Map
   - **Pattern:** Count subarrays with sum
   ```cpp
   for(int i = 0; i < nums.size(); i++) {
       sum += nums[i];
       if(map.find(sum - k) != map.end())
           count += map[sum - k];
       map[sum]++;
   }
   ```