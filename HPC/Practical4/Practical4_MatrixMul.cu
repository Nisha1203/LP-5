#include <iostream>
#include <limits>
#include <cuda.h>
using namespace std;

#define BLOCK_SIZE 2

__global__ void gpuMM(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;
    for (int n = 0; n < N; ++n)
        sum += A[row * N + n] * B[n * N + col];
    C[row * N + col] = sum;
}

int main(int argc, char *argv[])
{
    int N;
    cout << "Enter the size of the matrix: ";
    cin >> N;
    N *= 2; // Multiply by 2 to make the matrix size equal to user input * BLOCK_SIZE
    cout << "\nExecuting Matrix Multiplication" << endl;
    cout << "Matrix size: " << N << "x" << N << endl;

    // Allocate memory on the host
    float *hA, *hB, *hC;
    hA = new float[N * N];
    hB = new float[N * N];
    hC = new float[N * N];

    // Initialize matrices on the host with user input
    cout << "Enter elements of matrix A: ";
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            cin >> hA[j * N + i];
        }
    }
    cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear input buffer

    cout << "Enter elements of matrix B: ";
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            cin >> hB[j * N + i];
        }
    }
    cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear input buffer

    // Allocate memory on the device
    int size = N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // Copy matrices from the host to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    // Execute the matrix multiplication kernel
    gpuMM<<<grid, threadBlock>>>(dA, dB, dC, N);

    // Copy the GPU result back to CPU
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    // Display the result
    cout << "\nResultant matrix:\n";
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            cout << hC[row * N + col] << " ";
        }
        cout << endl;
    }

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Free host memory
    delete[] hA;
    delete[] hB;
    delete[] hC;

    cout << "Finished." << endl;
    return 0;
}

// nvcc add.cu -o add
// ./add