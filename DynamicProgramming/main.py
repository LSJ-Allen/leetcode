from typing import *
from collections import defaultdict
import functools

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        Approach:

        1D dp. dp[i] = the longest increasing subsequence ending at position i
        We can only add ith element to a subsequence if it's beigger than the last element in the 
        subsequence
        """
        n = len(nums)

        # since dp[i] is the longest increasing subsequence ending at position i
        # the minimum length is 1, thus init dp to be all 1s
        dp = [1] * n

        dp[0] = 1
        
        # first loop build the dp table
        for i in range(1, n):
            # second loop look for any past subsequences which can be extended by adding i
            for j in range(0, i + 1):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)

    # 120 Triangle
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        """
        Approach:

        The triangle is bottom filled n*n 2d array
        Use a 2d dp table of n*n where dp[i][j] = the min path sum starting rom triangle[i][j]
        dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1]) the min of two neighbors
        For example:
            triangle:
            2
            3 4
            6 5 7
            4 1 8 3

            dp:
            11
            9 10
            7 6 10
            4 1 8 3
        
        The last row of dp equals to last row of triangle
        """

        # init dp to be the same size as triangle
        n = len(triangle)
        dp = [[0] * i for i in range(1, n+1)]

        # deep copy
        dp[n-1] = triangle[n-1][:]
        for i in range(n-2, -1, -1):
            for j in range(0, i + 1):
                dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])
        return dp[0][0]

    # 63 Unique Path II
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """
        Approach:

        2D dp with the same size as the grid with dp[i][j] = # of unique paths to arrive at
        i, j. dp[i][j] = dp[i-1][j] + dp[j-1][i]. If a grid contains an obstical dp[i][j] = 0
        """

        # edge case: if start block is obstacle return 0
        if obstacleGrid[0][0] == 1:
            return 0
        
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0]* n for i in range(m)]

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    dp[i][j] = 1
                    continue
                
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                    continue

                topVal = 0 if i == 0 else dp[i-1][j]
                leftVal = 0 if j == 0 else dp[i][j-1]
                dp[i][j] = topVal + leftVal
        
        return dp[m-1][n-1]
                
    # 97. Interleaving String
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        """
        Approach:

        2D dp with dimetion (len(s1) + 1) * (len(s2) + 1)
        dp[i][j] is a boolean value that represents whether the string slice
        s1[0:i] and s2[0:j] can interleave and make up the first (i+j) characters
        in s3.

        Example: aa and dbbc can make up the first 6 characters so dp[2][4] is True

        dp init
        dp[0][1] = s2[0] == s3[0], dp[1][0] = s1[0] == s3[0]

        dp update:
        When we update dp[i][j], if dp[i-1][j] == True, it means that s1[0:i-1] and s2[0:j]
        is valid and we are tring to extend the string by adding s1[i].
        if dp[i][j-1] == True, we are tring to extend by adding s2[j].
        if neither the left nor the top cell are true, dp[i][j] is also false.

        At the end return dp[i][j]
        """
        m, n = len(s1), len(s2)

        # edge case: if any of the string is empty
        if m + n != len(s3):
            return False
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(n + 1):
                if i > 0:
                    dp[i][j] |= dp[i-1][j] and s1[i-1] == s3[i+j-1]
                if j > 0:
                    dp[i][j] |= dp[i][j-1] and s2[j-1] == s3[i+j-1]
        
        return dp[m][n]
        
    # 72 Edit Distance
    def minDistance(self, word1: str, word2: str) -> int:
        """
        Approach:

        this question is quite similar to 97. We use a 2d dp approach with a
        m+1 by n+1 dp. dp[i][j] represents the minimum operations needed to turn
        word1[0:i] to word2[0:j].

        The first row represents converting empty string to word2[j]
        and first col represents converting word1[i] to empty string

        Suppose we are at i,j. To insert we will look at the left cell dp[i-1][j]
        because it captures the subproblem where we convert word[0:i-1] to word2[0:j]
        To delete we look at dp[i][j-1], to replace we look at dp[i-1][j-1]
        """

        m, n = len(word1), len(word2)
        dp = [[0] * n+1 for _ in range(m+1)]

        # init first row
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(m + 1):
            dp[i][0] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                topVal = dp[i-1][j]
                LeftVal = dp[i][j-1]
                topLeftVal = dp[i-1][j-1]

                # if the current character is the same, we don't need to do aything
                # we will get the value from top left
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(topLeftVal, topVal, LeftVal) + 1
        return dp[m][n]
    
    # 123. Best Time to Buy and Sell Stock III
    def maxProfit(self, prices: List[int]) -> int:
        """
        Dp with state machine similar to other buying stock problems
        have 1 dp array for each state, dp[i] = max profit ending in that state at day i

        States:
        buy1, sell1, buy2, sell2

        Transitions:
        buy1 -> buy1: keep holding
        start -> buy1: purchase first stock

        sell1 -> sell1: do not buy
        buy1 -> sell1: sell

        sell1 -> buy2: purchase 2nd stock
        buy2 -> buy2: keep holding 2nd

        buy2 -> sell2: sell 2nd stock
        sell2 -> sell2: keep the previous sell state

        buy1[0] = -prices[0]
        buy2[0] = - prices[0]
        to represent impossible state
        
        example:
        prices = [3,3,5,0,0,3,1,4]
        b1 = [-3, -3, -3]
        s1 = [0, 0, 2]
        b2 = [-3, -3, -3,]
        s2 = [0, 0, 2]

        # only need 4 variables because the current state is only dependent on previous one,
        # don't need to track earlier ones
        """

        b1 = -prices[0]
        s1 = 0
        b2 = -prices[0]
        s2 = 0
        for i in range(1, len(prices)):
            b1 = max(b1, -prices[i])
            s1 = max(s1, b1 + prices[i])
            b2 = max(b2, s1 - prices[i])
            s2 = max(s2, b2 + prices[i])
        
        return s2
    
    # 188. Best Time to Buy and Sell Stock IV
    def maxProfit2(self, k: int, prices: List[int]) -> int:
        """
        Approach:

        Extension from #123. Here we track 2k states (k buy states and k sell states)

        States:
        buy_k
        sell_k

        Transition:
        sell_(k-1) -> buy_k
        buy_k -> buy_k

        buy_k -> sell_k
        sell_k -> sell_k
        """

        # init state at time 0
        # all buy state at 0 would be -prices[0] cuz the only possible way to enter buy state at 0
        # is to buy the stock
        # all sell states at 0 are 0 because the only possible way to enter sell state is to sell
        # at the same price
        buyStates = [-prices[0]] * k
        sellStates = [0] * k

        for i in range(1, len(prices)):
            # update buy1 and sell1
            buyStates[0] = max(buyStates[0], -prices[i])
            sellStates[0] = max(sellStates[0], buyStates[0] + prices[i])

            # update the subsequent states
            for j in range(1, k):
                buyStates[j] = max(buyStates[j], sellStates[j-1] - prices[i])
                sellStates[j] = max(sellStates[j], buyStates[j] + prices[i])

        return sellStates[k-1]
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        Approach:

        This problem is the string counter part of the problem coin change. Imagine the string is the
        total amount and wordDict contains "coins"

        1D dp. dp[i] = whether the string s[0:i+1] can be segmented. For each i we iterate through word Dict
        to find a case such that the string slice s[0:i+1] ends with the word and dp[i-wordlen] is true
        """

        dp = [False] * len(s)

        for i in range(0, len(s)):
            for word in wordDict:
                if i < len(word) - 1:
                    continue
                
                
                if i == len(word) - 1:
                    dp[i] = s[0:i+1] == word or dp[i]
                else:
                    dp[i] = s[0:i+1].endswith(word) and dp[i-len(word)] or dp[i]
                
        return dp[-1]

    # 1230. Toss Strange Coins
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        """
        Approach:

        1. state variables: i, the first i coins. j: number of heads
            dp[i][j] = the probability of obtaining j heads using the first i coins
            dp[-1][-1] solves the problem
        2. dp[i][j] = dp[i - 1][j - 1] * prob[i - 1] + dp[i - 1][j] * (1 - prob[i - 1])
        3. first row contains 0, can't get probability with 0 coin
        """
        n = len(prob)
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] * (1 - prob[i - 1])
            for j in range(1, target + 1):
                if j > i:
                    break
                dp[i][j] = dp[i - 1][j - 1] * prob[i - 1] + dp[i - 1][j] * (1 - prob[i - 1])

        return dp[n][target]

    def deleteAndEarn(self, nums: List[int]) -> int:
        """
        Approach:

        1d dp.
        state variable: num, a unique number in nums
        dp function: maxPoints(num) return the maximum points that we can gain if we 
        only consider  all elemnts in nums between 0 and num

        recurrence relation:
        maxPoints(x) = max(maxPoints(x-1), maxPoints(x-2) + gain)
        gain = # of occurrences of x * x

        base case:
        maxPoints(0) = 0
        maxPoints(1) = # of 1s

        The result would be maxPoints(max Num)
        """
        points = defaultdict(int)
        maxNumber = 0
        for num in nums:
            points[num] += num
            maxNumber = max(maxNumber, num)

        @functools.cache
        def maxPoints(num):
            # base case
            if num == 0:
                return 0
            
            if num == 1:
                return points[1]
            
            return max(maxPoints(num - 1), maxPoints(num - 2) + points[num])
        
        return maxPoints(maxNumber)
        
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        """
        Approach:

        dp cuz asking for maximum value and future state depend on present decisions,'
        ie. if we choose left now, it can't be used in the future anymore

        1. Find hte function that will compute the anser for a given state.
            states: need to know the following to desribe a state
            1. the multiplier we are using, state var: i
            2. the left index of the left most element remaining in nums
            3. the right index of the right most eledment remaining in nums
            we can calculate right = n - (i-left) - 1 so we only have 2 independent state 
            variables: i, left

            dp(i, left) = max possible score if we have used i multipliers and have picked left #
            of elements from the left side.

            to answer the problem return dp(0,0), the max possible score if we haven't done
            anything yet and have all the multiplers and elements from nums to use.

            Note that we use left to track the index of the left most element instead of 
            actually removing the element

        2. Recurrence Relation:
            leftGain = multi[i] * nums[left] + dp(i+1, left+1)
            rightGain = multi[i] * nums[right] + dp(i+1, left)
            dp(i, left) = max(leftGain, rightGain)
        
        3. Base case, dp(m, left) = 0. If i == m, we have no operations left and can not gain
        any score.

        Bottom up approach:
        i: range [0, m]
        left: range[0, m]

        dp size = (m+1)*(m+1) with the last row being 0 to reflect dp(m, left) = 0 for any left
        value

        fill dp table from last row to top
        """
        n, m = len(nums), len(multipliers)

        # create dp and initialize the last row
        dp = [[0] * (m+1) for _ in range(m + 1)]

        # i start from the second last row
        for i in range(m - 1, -1, -1):
            # left index cannot be bigger than i, the dp array is a 
            # triangle
            for left in range(i, -1, -1):
                mult = multipliers[i]
                right = n - 1 - (i - left)
                dp[i][left] = max(mult * nums[left] + dp[i+1][left+1], 
                                  mult * nums[right] + dp[i+1][left])
        return dp[0][0]

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        # use a hash table to store function return
        n = len(jobDifficulty)
        @functools.cache
        def dp(i : int, day: int) -> int:
            # base case if day > i: this is a impossible state retrun -1
            if day > i:
                return -1
            # base dp(0,0)
            if i == 0:
                if day == 0:
                    return jobDifficulty[0]
            
            curDay = dp(i - 1, day)

            if day == 0:
                return max(curDay, jobDifficulty[i])
            else:
                prevDay = dp(i - 1, day - 1)
                if curDay == -1:
                    return prevDay + jobDifficulty[i]
                else:
                    return min(prevDay + jobDifficulty[i], max(curDay, jobDifficulty[i]))
            
        
        return dp(n - 1, d - 1)
    
    # 1048. Lonagest String Chain
    def longestStrChain(self, words: List[str]) -> int:
        """
        Approach

        Longest subsequence of string chain, dp approach
        the original order does not matter, sort the words by length in descending order

        1. state variable i: ith word
            dp[i] = longest sequence ends at word i in the sorted list
            max dp gives the answer
        2. At each state we need to decide whether we can add current word to a sequence.
            dp[i] = 1 + max(dp[j]) for j in 0 to i-1 if j is ith predecessor
        3. base case
            dp[0] = 1
            
        """
        words.sort(key=lambda x: len(x))
        # use dict instead of array since we want to access ith word's predecessor in constant time
        dp = {}
        for word in words:
            dp[word] = 1

        longest = 1
        for i in range(1, len(words)):
            word = words[i]
            # go through each possible predecessor obtained by removing 1 character
            for j in range(len(word)):
                predecessor = word[0:j] + word[j+1:]
                if predecessor in dp:
                    dp[words[i]] = max(dp[words[i]], 1 + dp[predecessor])
                    longest = max(longest, dp[words[i]])
        return longest
    
    # 276. Paint Fence
    def numWays(self, n: int, k: int) -> int:
        """
        Approach:

        dp
        1. start var: i to represent ith post
            dp(i) = # of ways to paint ith post
        2. To paint i, there a in total k* dp(i-1) ways, but not all of them are legal
        If i-1 and i is not same color, we can proceed without worrying breaking the law.
        That gives us (k-1) * dp(i-1) ways. If i-1 and i is same color, we need to make sure i-2
        and i-1 is not the same color. How many ways to paint i-2 with different color than
        i-1? (k-1)*dp(i-2).
            dp(i) = (k-1)*(dp(i-1) + dp(i-2))
        3. dp[0] = k dp(1) = k^2
        """
        # edge case
        if n == 1:
            return k
        
        dp = [0] * n
        dp[0] = k
        dp[1] = k**2

        for i in range(2, n):
            dp[i] = (k-1)*(dp[i-1] + dp[i-2])

        return dp[-1]
    
    # 91. Decode Ways
    def numDecodings(self, s: str) -> int:
        """
        Approach:

        dp 
        1. let i be the first i chars, dp(i) = # of ways to decode first i chars
        2. for a char, it can be intepreted as a single encoding or double encoding with prev char
            for single: # of ways to decode is dp(i-1)
            for double: # of ways to decode is dp(i-2)
            
            dp(i) = dp(i-1) if single encoding is valid + dp(i-2) if double encoding is valid

        3. base case dp(0) = 1, decoding empty string returns empty decode, it's still 1
        way of decoding
            dp(1) = 1
        """
        # edge case
        if s[0] == "0":
            return 0
        if "00" in s:
            return 0
        dp = [0] * (len(s) + 1)
        
        # base case
        dp[0] = 1
        dp[1] = 1

        def isValid(s: str):
            return int(s) <= 26 and int(s) >= 10
        
        for i in range(2, len(s) + 1):
            doubleEncoding = s[i-2] + s[i-1]
            dp[i] = (dp[i-1] if s[i-1] != "0" else 0) + (dp[i-2] if isValid(doubleEncoding) else 0)
            
        return dp[-1]

    # 1626 Best Team With No Conflicts
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        # sort based on ages
        arr = list(zip(scores, ages))
        arr.sort(key=lambda x: (x[1], x[0]))

        n = len(arr)
        dp = [0] * n

        # base case
        dp[0] = arr[0][0]

        for i in range(1, n):
            # the min value of dp[i] is score[i]
            dp[i] = arr[i][0]

            # check all possible sub sequences
            for j in range(0, i):
                # skip if current score is smaller
                if arr[i][0] < arr[j][0]:
                    continue

                dp[i] = max(dp[i], arr[i][0] + dp[j])
        
        return max(dp)
    
    # 1463. Cherry Pickup II
    def cherryPickup(self, grid: List[List[int]]) -> int:
        """
        Approach:

        multi-dimensinal dp
        1. What info do we need to describe a given state?
            need to know robot 1's position and robot 2's position. These robots will always be on the same row
            so we need 3 state variables
            i: ith row
            j: r1's col
            k: r2's col

            dp[i][j][k] = max score of the state i j k
            max dp[-1] gives the answer
        2. Given state i j k, how prev states can we start and arriv e at i j k?
            the row must be i-1 since we can not traverse on the same row.
            for a given j, valid prev js are j - 1, j, j + 1
            for k, valid prev ks are k-1, k, k+1
            in total 9 valid prev states. We pick the best and start from there to arrive at i j k

            dp[i][j][k] = max(prevStates) + grid[i][j] + grid[i][k]

        3. Base case
            dp[0][0][n-1] = grid[0][0] + grid[0][n-1]
            for robot 1 i, j must be <= i, if j > i, it is an impossible state, return 0
            for robot 2, i+k must >= n-1
            
        """
        m, n = len(grid), len(grid[0])


        # create a m*n*n matrix for dp
        dp = [[[0] * n for j in range(n)] for i in range(m)]

        # base case
        dp[0][0][n-1] = grid[0][0] + grid[0][n-1]
        for i in range(m):
            for j in range(n):
                for k in range(n):
                    # j == k is a impossible state, skip
                    if j == k:
                        continue

                    # robot 1 cannot across the i-j = 0 diagonal
                    # and robot 2 cannot across the i+K = n-1 diagonal
                    # if out of bounds continue
                    if i - j < 0 or i + k  < n - 1:
                        continue
                    
                    # recurrence relation, to arrive at current state i, j, k
                    # there are 9 possible previous states with i-1
                    # (j - 1, k + 1),(j, k + 1),(j + 1, k + 1)
                    # (j - 1, k),(j, k),(j + 1, k)
                    # (j - 1, k - 1),(j, k - 1),(j + 1, k - 1)
                    prevStates = []

                    for s in [j - 1, j, j + 1]:
                        for t in [k - 1, k, k + 1]:
                            if (s >= 0 and s < n) and (t >= 0 and t < n):
                                prevStates.append(dp[i-1][s][t])


                    dp[i][j][k] = max(prevStates) + grid[i][j] + grid[i][k]
        
        # find the max value on the last row
        maxVal = 0
        for j in range(n):
            for k in range(n):
                maxVal = max(dp[m-1][j][k], maxVal)
        
        return maxVal

def main():
    s = Solution()
    print(s.cherryPickup([[1,0,0,0,0,0,1],[2,0,0,0,0,3,0],[2,0,9,0,0,0,0],[0,3,0,5,4,0,0],[1,0,2,3,0,0,6]]))

if __name__ == "__main__":
    main()