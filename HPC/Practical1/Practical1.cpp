#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int MAX = 100; // Maximum size for the graph

vector<int> graph[MAX]; // Graph representation
bool visited[MAX];      // Array to mark visited nodes

// Breadth-First Search function
void bfs(int start_node) {
    queue<int> q;
    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty()) {
        int current_node = q.front();
        q.pop();
        cout << current_node << " ";

        #pragma omp parallel for
        for (int i = 0; i < graph[current_node].size(); i++) {
            int adj_node = graph[current_node][i];
            if (!visited[adj_node]) {
                #pragma omp critical
                {
                    visited[adj_node] = true;
                    q.push(adj_node);
                }
            }
        }
    }
}

// Depth-First Search function
void dfs(int start_node) {
    stack<int> s;
    s.push(start_node);

    while (!s.empty()) {
        int current_node = s.top();
        s.pop();
        if (!visited[current_node]) {
            visited[current_node] = true;
            cout << current_node << " ";

            #pragma omp parallel for
            for (int i = 0; i < graph[current_node].size(); i++) {
                int adj_node = graph[current_node][i];
                if (!visited[adj_node]) {
                    #pragma omp critical
                    {
                        s.push(adj_node);
                    }
                }
            }
        }
    }
}

int main() {
    int n, m; // n: number of nodes, m: number of edges
    cout << "Enter the number of nodes and edges: ";
    cin >> n >> m;

    // Input edges
    cout << "Enter the edges (node pairs):\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // For undirected graph
    }

    // Initialize visited array
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    // Get number of threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    cout << "Number of threads: " << num_threads << endl;

    // Measure BFS execution time
    auto bfs_start = high_resolution_clock::now();
    cout << "Breadth-First Search (BFS): ";
    bfs(0); // Start BFS from node 0
    cout << endl;
    auto bfs_stop = high_resolution_clock::now();
    auto bfs_duration = duration_cast<microseconds>(bfs_stop - bfs_start);
    cout << "BFS Execution Time: " << bfs_duration.count() << " microseconds" << endl;

    // Reset visited array
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    // Measure DFS execution time
    auto dfs_start = high_resolution_clock::now();
    cout << "Depth-First Search (DFS): ";
    dfs(0); // Start DFS from node 0
    cout << endl;
    auto dfs_stop = high_resolution_clock::now();
    auto dfs_duration = duration_cast<microseconds>(dfs_stop - dfs_start);
    cout << "DFS Execution Time: " << dfs_duration.count() << " microseconds" << endl;

    return 0;
}

// g++ -fopenmp prac1.cpp -o Myexe
// ./Myexe


// Enter the number of nodes and edges: 6 7
// Enter the edges (node pairs):
// 0 1
// 0 2
// 1 3
// 1 4
// 2 5
// 3 5
// 4 5
//
// Number of threads: 2
// Breadth-First Search (BFS): 0 1 2 3 4 5 
// BFS Execution Time: 39764 microseconds
// Depth-First Search (DFS): 0 2 5 3 1 4 
// DFS Execution Time: 12 microseconds