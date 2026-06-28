#include <cuda_runtime.h>


__global__ void calcularPromedioParcialKernel(const double* d_imagenes,
                                              double* d_promedio_parcial,
                                              int m,
                                              int n,
                                              int chunk) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n) return;

    double suma = 0.0;
    for (int k = j; k < chunk; k += n) {
        suma += d_imagenes[k];
    }
    
    d_promedio_parcial[j] = (double)(suma / m);
}


__global__ void reducirPromedioParcialKernel(const double* promedios_parciales,
                                             double* promedio,
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


__global__ void calcularImagenesCentradasStreams(double* d_imagenes,
                                                 const double* d_promedio,
                                                 int m,
                                                 int n,
                                                 int chunk,
                                                 int chunk_relative_offset) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    for (int k = j; k < chunk; k += n) {
        int promedio_j = j+chunk_relative_offset;
        if (promedio_j >= n) promedio_j -= n;
        d_imagenes[k * n + j] = d_imagenes[k * n + j] - d_promedio[promedio_j];
    }
}
