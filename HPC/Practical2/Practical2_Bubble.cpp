#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Sequential bubble sort implementation
void sequential_bubble_sort(vector<int>& arr) {
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;
        for (int i = 0; i < arr.size() - 1; ++i) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

// Parallel bubble sort implementation using odd-even transposition
void parallel_bubble_sort(vector<int>& arr) {
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;
        #pragma omp parallel for
        for (int i = 0; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
        #pragma omp parallel for
        for (int i = 1; i < arr.size() - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

int main() {
    int n;
    cout << "Enter the number of elements in the array: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i) {
        cout << "Element " << i + 1 << ": ";
        cin >> arr[i];
    }

    // Copy the original array for sequential sorting
    vector<int> seq_arr = arr;

    double start, end;

    // Sort sequentially and measure time
    start = omp_get_wtime();
    sequential_bubble_sort(seq_arr);
    cout << "Sorted array using sequential bubble sort: ";
    for (int i = 0; i < n; ++i) {
        cout << seq_arr[i] << " ";
    }
    cout << endl;
    end = omp_get_wtime();
    cout << "Sequential bubble sort time: " << end - start << " seconds" << endl;

    // Sort in parallel and measure time
    start = omp_get_wtime();
    parallel_bubble_sort(arr);
    cout << "Sorted array using parallel bubble sort: ";
    for (int i = 0; i < n; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
    end = omp_get_wtime();
    cout << "Parallel bubble sort time: " << end - start << " seconds" << endl;

    return 0;
}


// g++ -fopenmp prac2.cpp -o prac2
// ./prac2

// Enter the number of elements in the array: 5
// Enter 5 elements:
// Element 1: 1
// Element 2: 5
// Element 3: 8
// Element 4: 9
// Element 5: 7
// Sorted array using sequential bubble sort: 1 5 7 8 9 
// Sequential bubble sort time: 2.4993e-05 seconds
// Sorted array using parallel bubble sort: 1 5 7 8 9 
// Parallel bubble sort time: 0.000153543 seconds