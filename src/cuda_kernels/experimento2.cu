#include <cuda_runtime.h>
#include <iostream>

__global__ void calcularPromedioParcialKernel(const float* d_imagenes,
                                              float* d_promedio_parcial,
                                              int m,
                                              int n,
                                              int chunk) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n) return;

    float suma = 0.0;
    for (int k = j; k < chunk; k += n) {
        suma += d_imagenes[k];
    }
    
    d_promedio_parcial[j] = (float)(suma / m);
}


__global__ void reducirPromedioParcialKernel(const float* promedios_parciales,
                                             float* promedio,
                                             int NSTREAMS,
                                             int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n) return;

    float sum = 0.0f;

    for (int i = 0; i < NSTREAMS; i++) {
        sum += promedios_parciales[i * n + j];
    }

    promedio[j] = sum;
}


__global__ void calcularImagenesCentradasStreams(float* d_imagenes,
                                                 const float* d_promedio,
                                                 int n,
                                                 int chunk_elements) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n) return;

    for (int idx = j; idx < chunk_elements; idx += n) {
        d_imagenes[idx] -= d_promedio[j];
    }
}


#define TILE 16

__global__ void acumularCovarianzaBatchKernel(
    const float* d_imagenes,       // batch de imágenes: batch_m * n   // promedio global: n
    float* d_covarianza,        // acumulador global: n * n
    int batch_m,
    int n,
    int m
) {
    __shared__ float sharedA[TILE * TILE];
    __shared__ float sharedB[TILE * TILE];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int j  = blockIdx.y * TILE + ty;
    int jp = blockIdx.x * TILE + tx;

    float suma = 0.0;

    int C = (batch_m + TILE - 1) / TILE;

    for (int c = 0; c < C; c++) {
        int kA = c * TILE + tx;
        int kB = c * TILE + ty;

        if (kA < batch_m && j < n) {
            sharedA[ty * TILE + tx] =d_imagenes[kA * n + j];
        } else {
            sharedA[ty * TILE + tx] = 0.0;
        }

        if (kB < batch_m && jp < n) {
            sharedB[tx * TILE + ty] =d_imagenes[kB * n + jp];
        } else {
            sharedB[tx * TILE + ty] = 0.0;
        }

        __syncthreads();

        for (int kk = 0; kk < TILE; kk++) {
            suma += sharedA[ty * TILE + kk] *
                    sharedB[tx * TILE + kk];
        }

        __syncthreads();
    }

     if (j < n && jp < n) {
        atomicAdd(&d_covarianza[j * n + jp], suma/(float)m);
    }
}