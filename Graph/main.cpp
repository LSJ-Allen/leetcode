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
};

int main() {
    vector<vector<int> > times = {
        {3, 5, 78}, {2, 1, 1}, {1, 3, 0}, {4, 3, 59}, {5, 3, 85}, {5, 2, 22}, {2, 4, 23}, {1, 4, 43}, {4, 5, 75},
        {5, 1, 15}, {1, 5, 91}, {4, 1, 16}, {3, 2, 98}, {3, 4, 22}, {5, 4, 31}, {1, 2, 0}, {2, 5, 4}, {4, 2, 51},
        {3, 1, 36}, {2, 3, 59}
    };
    cout << Soution::networkDelayTime(times, 5, 5);
    return 0;
}
