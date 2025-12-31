#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>

using namespace std;

namespace Soution {
    // 743. Network Delay Time
    int networkDelayTime(vector<vector<int> > &times, int n, int k) {
        /**
         * Approach:
         *
         * Use shortest path algorithm to obtain shortest distances from source node to all
         * nodes and return the largest one.
         *
         * Dijkstra's Algorithm:
         *
         * A greedy algorithm using a min-heap priority queue to find the shortest path
         * from a source node to all other nodes in a weighted graph (non-negative weights).
         *
         * Process:
         * 1. Initialize a dist[] array with dist[source] = 0 and all others = ∞
         * 2. Push {0, source} to the priority queue (min-heap by distance)
         * 3. While the queue is not empty:
         *    a. Pop the node with the smallest distance
         *    b. Skip if this distance is outdated (> dist[node])
         *    c. For each neighbor, calculate new_dist = dist[node] + edge_weight
         *    d. If new_dist < dist[neighbor], update dist[neighbor] and push to queue
         *
         * Key Insights:
         * - Always processes the closest unvisited node (greedy choice)
         * - Nodes can be added to the queue multiple times with different distances
         * - The distance check (dist > dist[node]) filters out outdated queue entries
         * - No visited array needed - dist[] serves as both distance tracker and visited indicator
         * - Once a node is processed with its optimal distance, any future encounters are skipped
         *
         * Time: O((V + E) log V)  |  Space: O(V + E)
         */
        vector<vector<pair<int, int> > > adj_lists(n);

        // build the adjacency list, stores {source_node, weight}
        for (const auto &time: times) {
            pair<int, int> edge = {time[1] - 1, time[2]};
            adj_lists[time[0] - 1].push_back(edge);
        }

        // start Dijkstra's
        vector<int> distances(n, INT_MAX);
        distances[k - 1] = 0;

        auto cmp = [](pair<int, int> p1, pair<int, int> p2) {
            return p1.second > p2.second;
        };
        priority_queue<pair<int, int>, vector<pair<int, int> >, decltype(cmp)> pq(cmp);
        pq.push({k - 1, 0});

        while (!pq.empty()) {
            const pair<int, int> node = pq.top();
            pq.pop();

            // if the distance of node is bigger than what is stored, a better distance
            // has been found, skip processing this node again.
            if (node.second > distances[node.first]) continue;
            const auto &neighbors = adj_lists[node.first];

            for (const auto &neighbor: neighbors) {
                int neighbor_dist = distances[neighbor.first];
                int cur_dist = distances[node.first] + neighbor.second;

                // if not visited neighbor, push
                if (cur_dist < neighbor_dist) {
                    distances[neighbor.first] = cur_dist;
                    pq.push({neighbor.first, cur_dist});
                }
            }
        }

        int result = -1;
        for (int dist: distances) {
            result = max(result, dist);
        }

        return result == INT_MAX ? -1 : result;
    }

    // 1970. Last Day Where You can Still Cross
    namespace LastDay {
        // find operation in union find
        int findRoot(vector<int> &parent, int x) {
            /**
             * example:
             * parent = [0, 0, 1] x = 2
             * parent[2] != 2 -> recursion call find(parent[2])
             * find(1)
             * parent[1] != 1 -> recursion call find(parent[1])
             * find(0)
             * parent[0] = 0; return 0
             * parent[1] = 0
             * return 0;
             * parent[2] = 0;
             * return 0;
             *
             * now all elements have 1 common roots, path is compressed.
             *
             */
            if (parent[x] != x) {
                parent[x] = findRoot(parent, parent[x]);
            }

            return parent[x];
        }

        void unite(vector<int> &parent, vector<int> &rank, int x, int y) {
            int rootX = findRoot(parent, x);
            int rootY = findRoot(parent, y);

            // if they have the same root, x and y are already in the same set, return
            if (rootX == rootY) return;

            // if tree x and tree y has diff size, merge the smaller one into the bigger one
            // the rank doesn't change as tree depth is not increased
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                // if tied, merge y into x by default, rank increase by one.
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }

        int latestDayToCross(int row, int col, vector<vector<int> > &cells) {
            /**
             * Approach:
             *
             * Use the union find algorithm in reverse order from the last day to the first day.
             * Imagine the grid is intially flooded and wer are removing flood day by day.
             * Each time a flooded cell is removed, a new cell joins and it unions all of its 0
             * neighbors. Repeat this process untill a cell at the top and a cell at the bottom
             * is connected, in union find algorithm, they would share the same parent.
             */

            // intialize a parent array to the size of row * col, the number of cells
            // the cell at i,j is represented by the index = row*i + j;
            vector<int> parent(row * col);
            vector<int> rank(row * col);

            // at the start each cell is its own parent
            for (int index = 0; index < row * col; index++) {
                parent[index] = index;
            }

            // construct the initial graph from the last day, traverse days in reverse order
            // and remove flooded cells.
            vector<vector<int> > grid(row, vector<int>(col, 0));
            int days = cells.size();
            for (auto &coord: cells) {
                // turn into 0 based
                coord[0] = coord[0] - 1;
                coord[1] = coord[1] - 1;
                grid[coord[0]][coord[1]] = 1;
            }

            for (int day = cells.size() - 1; day >= 0; day--) {
                // check if top row is connected to bottom row
                for (int jTop = 0; jTop < col; jTop++) {
                    int rootTop = findRoot(parent, jTop);
                    if (grid[0][jTop] == 1) continue;
                    for (int jBot = 0; jBot < col; jBot++) {
                        if (grid[row - 1][jBot] == 1) continue;
                        int rootBot = findRoot(parent, (row - 1) * col + jBot);
                        if (rootBot == rootTop) {
                            // a connection is found, return days
                            return day + 1;
                        }
                    }
                }

                // remove flooded cell
                const vector<int> &flooded = cells[day];
                int i = flooded[0], j = flooded[1];
                grid[i][j] = 0;

                // union the removed flooded cell with its neighbors
                // top neighbor
                if (i > 0 && grid[i - 1][j] == 0) {
                    unite(parent, rank, (i - 1) * col + j, i * col + j);
                }

                // bot neighbor
                if (i < row - 1 && grid[i + 1][j] == 0) {
                    unite(parent, rank, (i + 1) * col + j, i * col + j);
                }

                // left
                if (j > 0 && grid[i][j - 1] == 0) {
                    unite(parent, rank, i * col + j - 1, i * col + j);
                }

                // right
                if (j < col - 1 && grid[i][j + 1] == 0) {
                    unite(parent, rank, i * col + j + 1, i * col + j);
                }
            }
            return 0;
        }
    }
};

int main() {
    vector<vector<int> > cells = {{1, 2}, {2, 1}, {3, 3}, {2, 2}, {1, 1}, {1, 3}, {2, 3}, {3, 2}, {3, 1}};
    // cout << Soution::LastDay::latestDayToCross(3, 3, cells) << endl;

    vector<vector<int> > cells2 = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
    // cout << Soution::LastDay::latestDayToCross(2, 2, cells2);

    vector<vector<int> > cells3 = {
        {4, 2}, {6, 2}, {2, 1}, {4, 1}, {6, 1}, {3, 1}, {2, 2}, {3, 2}, {1, 1}, {5, 1}, {5, 2}, {1, 2}
    };
    cout << Soution::LastDay::latestDayToCross(6, 2, cells3);
    return 0;
}