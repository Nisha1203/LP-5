#include <iostream>
#include <ctime>   // Include <ctime> for time()
using namespace std;

__global__
void add(int* A, int* B, int* C, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    int N;
    cout << "Enter the size of vectors: ";
    cin >> N;

    int* A, * B, * C;
    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);

    // Allocate host memory
    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    // Initialize host arrays
    cout << "Enter elements of vector A: ";
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }

    cout << "Enter elements of vector B: ";
    for (int i = 0; i < N; ++i) {
        cin >> B[i];
    }

    cout << "Vector A: ";
    for (int i = 0; i < N; ++i) {
        cout << A[i] << " ";
    }
    cout << endl;

    cout << "Vector B: ";
    for (int i = 0; i < N; ++i) {
        cout << B[i] << " ";
    }
    cout << endl;

    int* X, * Y, * Z;

    // Allocate device memory
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    // Check for CUDA memory allocation errors
    if (X == nullptr || Y == nullptr || Z == nullptr) {
        cerr << "CUDA memory allocation failed" << endl;
        return 1;
    }

    // Copy data from host to device
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    clock_t start_time = clock();
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
    cudaDeviceSynchronize(); // Wait for all threads to finish
    clock_t end_time = clock();

    // Check for kernel launch errors
    cudaError_t kernelLaunchError = cudaGetLastError();
    if (kernelLaunchError != cudaSuccess) {
        cerr << "CUDA kernel launch failed: " << cudaGetErrorString(kernelLaunchError);
        return 1;
    }

    // Copy result from device to host
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    // Check for CUDA memcpy errors
    cudaError_t memcpyError = cudaGetLastError();
    if (memcpyError != cudaSuccess)
    {
        cerr << "CUDA memcpy failed: " << cudaGetErrorString(memcpyError) << endl;
        return 1;
    }

    cout << "Addition: ";
    for (int i = 0; i < N; ++i) {
        cout << C[i] << " ";
    }
    cout << endl;

    // Free device memory
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    double time_taken = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Time taken: " << time_taken << " seconds" << endl;
    cout << "Number of threads used: " << blocksPerGrid * threadsPerBlock << endl;

    return 0;
}

// nvcc mul.cu -o mul
// ./mul