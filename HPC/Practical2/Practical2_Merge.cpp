#include <iostream>
#include <omp.h>

using namespace std;

void merge(int arr[], int low, int mid, int high) {
    // Create arrays of left and right partitions
    int n1 = mid - low + 1;
    int n2 = high - mid;

    int left[n1];
    int right[n2];

    // Copy all left elements
    for (int i = 0; i < n1; i++) left[i] = arr[low + i];

    // Copy all right elements
    for (int j = 0; j < n2; j++) right[j] = arr[mid + 1 + j];

    // Compare and place elements
    int i = 0, j = 0, k = low;

    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) {
            arr[k] = left[i];
            i++;
        } else {
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    // If any elements are left out
    while (i < n1) {
        arr[k] = left[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = right[j];
        j++;
        k++;
    }
}

void parallelMergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            {
                parallelMergeSort(arr, low, mid);
            }

#pragma omp section
            {
                parallelMergeSort(arr, mid + 1, high);
            }
        }
        merge(arr, low, mid, high);
    }
}

void mergeSort(int arr[], int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

int main() {
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    if (n <= 0) {
        cout << "Invalid array size.\n";
        return 1;
    }

    int original_arr[n];
    int arr[n];
    double start_time, end_time;

    // Input array elements from the user
    cout << "Enter " << n << " elements of the array:\n";
    for (int i = 0; i < n; i++) {
        cin >> original_arr[i];
        arr[i] = original_arr[i];
    }

    // Measure Sequential Time
    start_time = omp_get_wtime();
    mergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds\n";
    
    // Display sorted array
    cout << "Sorted array (sequential):\n";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Reset the array
    for (int i = 0; i < n; i++) {
        arr[i] = original_arr[i];
    }

    // Measure Parallel time
    start_time = omp_get_wtime();
    parallelMergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Time taken by parallel algorithm: " << end_time - start_time << " seconds\n";
    
    // Display sorted array
    cout << "Sorted array (parallel):\n";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}

// g++ -fopenmp prac2a.cpp -o prac2a
// ./prac2a


// Enter the size of the array: 5
// Enter 5 elements of the array:
// 1
// 9
// 8
// 3
// 2
// Time taken by sequential algorithm: 1.93e-06 seconds
// Sorted array (sequential):
// 1 2 3 8 9 
// Time taken by parallel algorithm: 0.000187842 seconds
// Sorted array (parallel):
// 1 2 3 8 9 