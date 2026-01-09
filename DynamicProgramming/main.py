from typing import *
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
        
                    


def main():
    s = Solution()
    print(s.isInterleave("", "", ""))

if __name__ == "__main__":
    main()