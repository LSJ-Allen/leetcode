from typing import *

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        Approach:

        Every maximal rectangle is uniquely determined by its shortest bar. For each bar, we find the largest 
        rectangle where that bar is the minimum height. 
        
        Use a monotonic increasing stack to track boundaries. Iterate through heights, if we encounter a bar taller than stack's top bar,
        we push to the stack. If we encounter a shorter bar, we will pop all taller bars one by one from the stack and calculate the
        rectangle where the poped bar is the minimum height. The area of such rectangle can be calculated by 
        poped's height * width (defined by right boundary - left boundary)

        The right boundary, the position of the last bar that can join the current rectangle, is i

        The left boundary, is the top bar sitting in the stack now.

        Therefore the width is i - stack's top - 1 (both boundaries are exclusive)

        """

        # stores the index of heights
        stack = []
        max_area = 0
        for i, h in enumerate(heights):
            # if stack is not empty and h is smaller than stack's top, pop from stack until it's empty or
            # the top element is smaller
            while stack and h < heights[stack[-1]]:
                bar_index = stack.pop()
                height = heights[bar_index]

                # if the stack is empty, left_boundary defaults to -1
                left_boundary = stack[-1] if len(stack) > 0 else -1
                right_boundary = i
                width = right_boundary - left_boundary - 1
                max_area = max(max_area, width * height)
            
            stack.append(i)

        # process the remaining bars in the stack
        while stack:
            bar_index = stack.pop()
            height = heights[bar_index]
            left_boundary = stack[-1] if len(stack) > 0 else -1
            right_boundary = len(heights)
            width = right_boundary - left_boundary - 1
            max_area = max(max_area, width * height)
        
        return max_area
    
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        """
        Approach:

        The idea is to treat each row a histogram same size as the row. The height of each cell is equal to
        the number of 1's in the current and previous rows. Now, the problem is turned into finding the largest
        rectangle in a histogram.
        """
        def largestRectangleArea(heights: List[int]) -> int:
            # stores the index of heights
            stack = []
            max_area = 0
            for i, h in enumerate(heights):
                # if stack is not empty and h is smaller than stack's top, pop from stack until it's empty or
                # the top element is smaller
                while stack and h < heights[stack[-1]]:
                    bar_index = stack.pop()
                    height = heights[bar_index]

                    # if the stack is empty, left_boundary defaults to -1
                    left_boundary = stack[-1] if len(stack) > 0 else -1
                    right_boundary = i
                    width = right_boundary - left_boundary - 1
                    max_area = max(max_area, width * height)
                
                stack.append(i)

            # process the remaining bars in the stack
            while stack:
                bar_index = stack.pop()
                height = heights[bar_index]
                left_boundary = stack[-1] if len(stack) > 0 else -1
                right_boundary = len(heights)
                width = right_boundary - left_boundary - 1
                max_area = max(max_area, width * height)
            
            return max_area
        
        m , n = len(matrix), len(matrix[0])
        histograms = [[0] * n for _ in range(m)]

        histograms[0] = [int(i) for i in matrix[0]]

        max_area = largestRectangleArea(histograms[0])
        for i in range(1, m):
            for j in range(n):
                histograms[i][j] = 1 + histograms[i-1][j] if matrix[i][j] == "1" else 0
            max_area = max(largestRectangleArea(histograms[i]), max_area)
        
        return max_area
    
    # 221. Maximal Square
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        """
        Approach:

        2d dp with same dimension as input matrix, dp[i][j] stores the side length of the rectangle
        with it's bottom right corner at i,j. This info tells us how many "1"s are on top and 
        how many "1"s are left

        Subproblem, calculate current square size based on the square on top, on left, and on top left corner
        dp[i][j] = 1 + min(top, left, top left)
        """
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]

        # track max width
        max_width = 0

        # init first row and first column
        for j in range(n):
            dp[0][j] = 0 if matrix[0][j] == "0" else 1
            max_width = max(dp[0][j], max_width)
        
        for i in range(m):
            dp[i][0] = 0 if matrix[i][0] == "0" else 1
            max_width = max(dp[i][0], max_width)

        # populate dp and keep track of max square
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == "0":
                    dp[i][j] = 0
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                    max_width = max(dp[i][j], max_width)
        
        return max_width**2
                    
def main():
    s = Solution()
    print(s.maximalSquare([["1","1","1","1","0"],
                           ["1","1","1","1","0"],
                           ["1","1","1","1","1"],
                           ["1","1","1","1","1"],
                           ["0","0","1","1","1"]]))

if __name__ == "__main__":
    main()