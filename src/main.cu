#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>

// Error checking macro
#define cudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

// Kernel declaration
__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    // Print device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
    {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Test vector addition
    const int N = 1000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (h_a == NULL || h_b == NULL || h_c == NULL)
    {
        printf("Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for d_a: %s\n", cudaGetErrorString(err));
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for d_b: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    err = cudaMalloc(&d_c, size);
    if (err != cudaSuccess)
    {
        printf("Failed to allocate device memory for d_c: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    // Copy inputs to device
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaCheckError(); // Check for kernel launch errors

    // Synchronize to wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("Failed to copy result from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return EXIT_FAILURE;
    }

    // Verify
    bool success = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_c[i] - 3.0f) > 1e-5f)
        {
            printf("Verification failed at index %d: %f != 3.0\n", i, h_c[i]);
            success = false;
            break;
        }
    }
    if (success)
        printf("CUDA test passed successfully!\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}