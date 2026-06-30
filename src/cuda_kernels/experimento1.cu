#include <cuda_runtime.h>


__global__ void calcularPromedioKernel(const float* d_imagenes, float* d_promedio, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float suma = 0.0;
        for (int k = 0; k < m; ++k) {
            suma += d_imagenes[k * n + j];
        }
        
        d_promedio[j] = (float)(suma / m);
    }
}


__global__ void calcularImagenesCentradas(float* d_imagenes, float* d_promedio, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        for (int k = 0; k < m; ++k) {
            d_imagenes[k * n + j] = d_imagenes[k * n + j] - d_promedio[j];
        }
        
    }
}


__global__ void calcularTileCovarianza(float* d_imagenes, float* d_covarianza, int m, int n) {
    __shared__ float sharedA[256];
    __shared__ float sharedB[256];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int j  = blockIdx.y * 16 + ty; // Pixel de la matriz de la izquierda
    int jp = blockIdx.x * 16 + tx; // Pixel de la matriz de la derecha

    int C = (m + 15) >> 4; //Cuantos grupos de 16 imagenes hay

    float suma = 0.0;

    for (int c=0; c<C; c++) {
        int kA = c * 16 + threadIdx.x; // Que imagen estamos cargando 
        int kB = c * 16 + threadIdx.y;


        if(kA < m && j < n){
            sharedA[ty * 16 + tx] = d_imagenes[kA*n + j];
        } else {
            sharedA[ty * 16 + tx] = 0.0;
        }
        
        if(kB < m && jp < n){
            sharedB[tx * 16 + ty] = d_imagenes[kB*n + jp];
        }else{
            sharedB[tx * 16 + ty] = 0.0;
        }
        
        __syncthreads(); //Espera a que todas las hebras del bloque terminen de cargar a memoria compartida antes de usarla

        for (int kk = 0; kk < 16; kk++) {
            suma += sharedA[ty * 16 + kk] * sharedB[tx * 16 + kk];
        }

        __syncthreads(); //Espera a que todas las hebras del bloque terminen de sumar antes de sobreescribir la memoria compartida
    }

    if (j < n && jp < n) {
        d_covarianza[j * n + jp] = 1.0 / m * suma;
    }
}
