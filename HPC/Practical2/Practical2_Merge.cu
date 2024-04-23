#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel for Bubble Sort
__global__ void bubble_sort(int* d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    for (int i = 0; i < size - 1; i++) {
        int j = idx + i; // Offset to perform the bubble sort step
        if (j < size - 1 && d_arr[j] > d_arr[j + 1]) { // Swap if out of order
            int temp = d_arr[j];
            d_arr[j] = d_arr[j + 1];
            d_arr[j + 1] = temp;
        }
        __syncthreads(); // Synchronize threads within block
    }
}

// Function for Bubble Sort on CPU
void bubble_sort_cpu(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) { // Swap if out of order
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    // Test data
    int h_arr[] = {64, 34, 25, 12, 22, 11, 90};
    int size = sizeof(h_arr) / sizeof(h_arr[0]);

    // Bubble Sort on CPU
    auto start = std::chrono::high_resolution_clock::now();
    bubble_sort_cpu(h_arr, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sequential Merge Sort took " << duration.count() << " seconds\n";

    // Copying data to the device for parallel Bubble Sort
    int* d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Bubble Sort on GPU
    start = std::chrono::high_resolution_clock::now();
    int blockSize = 256; // Threads per block
    int gridSize = (size + blockSize - 1) / blockSize; // Blocks
    bubble_sort<<<gridSize, blockSize>>>(d_arr, size);
    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost); // Copy back to host
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Parallel Merge Sort took " << duration.count() << " seconds\n";

    // Display sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < size; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr);

    return 0;
}