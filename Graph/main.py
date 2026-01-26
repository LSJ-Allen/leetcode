from typing import *
from collections import deque

class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        """
        Approach:

        Standard graph bfs
        """

        m, n = len(maze), len(maze[0])
        visited = [[False] * n for _ in range(m)]

        # mark entrance as visited
        visited[entrance[0]][entrance[1]] = True

        # use queue to store nodes as [i, j]
        q = deque()
        q.append(entrance)

        # track number of steps
        numSteps = 0
        while(len(q) > 0):
            
            # process all nodes at this level
            for _ in range(len(q)):
                i, j = q.popleft()

                if [i, j] != entrance:
                    # check if at an exit
                    if (i == 0 or i == m - 1) or (j == 0 or j == n - 1) and (maze[i][j] == '.'):
                        return numSteps
                # iterate 4 neighbors
                # Top
                if i > 0 and maze[i-1][j] == "." and not visited[i-1][j]:
                    q.append([i-1, j])
                    visited[i-1][j] = True

                
                # down
                if i < m - 1 and maze[i+1][j] == "." and not visited[i+1][j]:
                    q.append([i+1, j])
                    visited[i+1][j] = True

                # left
                if j > 0 and maze[i][j-1] == "." and not visited[i][j-1]:
                    q.append([i, j-1])
                    visited[i][j-1] = True
                
                # right
                if j < n - 1 and maze[i][j+1] == "." and not visited[i][j+1]:
                    q.append([i, j+1])
                    visited[i][j+1] = True

            numSteps += 1

        return -1
        
    

def main():
    s = Solution()
    print(s.nearestExit([["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], [1,0]))
    pass

if __name__ == "__main__":
    main()