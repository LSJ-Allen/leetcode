from typing import *
from collections import deque, defaultdict
import heapq

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
        
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """
        Appraoch:

        traverse through all emails and use union find to build groups.
        each account contains email that are originally in the same set.

        first turn accounts into email name hash map.
        After union find finishes, go each element, find its root, find the name attached to 
        that root.
        """
        class UnionFind():
            def __init__(self):
                # hash map to store email:parent email
                self.parents = {}

                # email: rank
                self.rank = defaultdict(int)
            
            def find(self, x: str) -> str:
                # path compression
                if self.parents[x] != x:
                    self.parents[x] = self.find(self.parents[x])
                    
                return self.parents[x]
            
            def union(self, x: str, y: str) -> None:
                xRoot = self.find(x)
                yRoot = self.find(y)

                if xRoot == yRoot:
                    return
                
                if self.rank[xRoot] < self.rank[yRoot]:
                    self.parents[xRoot] = yRoot
                elif self.rank[xRoot] > self.rank[yRoot]:
                    self.parents[yRoot] = xRoot
                else:
                    # merge y to x by defaul
                    self.parents[yRoot] = xRoot
                    self.rank[xRoot] += 1

            def isConnected(self, x: str, y: str) -> bool:
                return self.find(x) == self.find(y)

            # insert an element
            def insert(self, x: str, parent: str) -> None:
                if parent not in self.parents:
                    self.parents[parent] = parent
                
                # if it's the first time seeing this email, just add it
                if x not in self.parents:
                    self.parents[x] = self.find(parent)
                else:
                    # if we have seen x before, we merge the current set and the set x
                    # belongs
                    self.union(x, parent)

        uf = UnionFind()

        # email to name dict
        emailToName = {}
        
        for s in accounts:
            name = s[0]
            for email in s[1:]:
                emailToName[email] = name

                # by default the first email in an account is the parent
                uf.insert(email, s[1])
        
        # construct final answer
        parentToEmails = defaultdict(list)
        for email, parent in uf.parents.items():
            # find parent one more time to ensure its the root
            realParent = uf.find(parent)
            parentToEmails[realParent].append(email)
        
        return [[emailToName[parent]] + sorted(emails) for parent, emails in parentToEmails.items()]

    # 787. Cheapest Flights Within K Stops
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        Approach:

        Dijkstra while tracking the number to steps
        """
        adj = [[] for _ in range(n)]
        for e in flights:
            adj[e[0]].append((e[1], e[2]))
        
        stops = [float('inf')] * n
        pq = []
        # (dist_from_src_node, node, number_of_stops_from_src_node)
        heapq.heappush(pq, (0, src, 0))
        
        while pq:
            dist, node, steps = heapq.heappop(pq)
            # We have already encountered a path with a lower cost and fewer stops,
            # or the number of stops exceeds the limit.
            if steps >= stops[node] or steps > k + 1:
                continue
            stops[node] = steps
            if node == dst:
                return dist
            for neighbor, price in adj[node]:
                heapq.heappush(pq, (dist + price, neighbor, steps + 1))
        
        return -1

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """
        Approach:

        Equations make up a directed graph. Use graph traversal to find value of queries
        """
        adjList = defaultdict(list)
        for i, equation in enumerate(equations):
            src = equation[0]
            dst = equation[1]
            adjList[src].append((dst, values[i]))
            adjList[dst].append((src, 1/values[i]))
        
        result = []

        # dfs to find the path from src to dst
        def dfs(src: str, dst: str, visited: dict, value) -> float:
            # edge case
            if dst not in adjList or src not in adjList:
                return -1.0
            
            visited[src] = True
            # recursion base case
            if src == dst:
                return value
            
            # go through every neighbor
            for neighbor, val in adjList[src]:
                # check if neighbor is visited
                if visited[neighbor]:
                    continue
                result = dfs(neighbor, dst, visited, value*val)
                if result != -1:
                    return result
            
            return -1.0
        
        visited = defaultdict(bool)
        # for each query, perform graph traversal to find the value of that query
        for query in queries:
            queryResult = dfs(query[0], query[1], visited, 1.0)
            result.append(queryResult)
            visited.clear()
        return result

def main():
    s = Solution()
    print(s.calcEquation(
        [["a","b"],["b","c"]],
        [2.0,3.0],
        [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
    ))

if __name__ == "__main__":
    main()