# define cimg_display 1
# define cimg_use_png
# include "CImg.h"
# include <iostream>
# include <vector>
#include <cuda_runtime.h>
#include <fstream>

using namespace cimg_library;


__global__ void calcularPromedioKernel(const double* d_imagenes, double* d_promedio, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double suma = 0.0;
        for (int k = 0; k < m; ++k) {
            suma += d_imagenes[k * n + j];
        }
        
        d_promedio[j] = (double)(suma / m);
        
    }
}
__global__ void calcularImagenesCentradas(double* d_imagenes, double* d_promedio, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        for (int k = 0; k < m; ++k) {
            d_imagenes[k * n + j] = d_imagenes[k * n + j] - d_promedio[j];
        }
        
    }
}

__global__ void calcularTileCovarianza(double* d_imagenes, double* d_covarianza, int m, int n) {
    __shared__ double sharedA[256];
    __shared__ double sharedB[256];


    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int j  = blockIdx.y * 16 + ty; // Pixel de la matriz de la izquierda
    int jp = blockIdx.x * 16 + tx; // Pixel de la matriz de la derecha


    int C = (m + 15) / 16; //Cuantos grupos de 16 imagenes hay

    double suma = 0.0f;

    for (int c=0; c<C; c++) {

        int kA = c * 16 + threadIdx.x; // Que imagen estamos cargando 
        int kB = c * 16 + threadIdx.y;


        if(kA < m && j < n){
            sharedA[ty * 16 + tx] = d_imagenes[kA*n + j];
        } else {
            sharedA[ty * 16 + tx] = 0.0f;
        }
        
        if(kB < m && jp < n){
            sharedB[tx * 16 + ty] = d_imagenes[kB*n + jp];
        }else{
            sharedB[tx * 16 + ty] = 0.0f;
        }
        
        __syncthreads(); //Espera a que todas las hebras del bloque terminen de cargar a memoria compartida antes de usarla

        for (int kk = 0; kk < 16; kk++) {
            suma += sharedA[ty * 16 + kk] * sharedB[tx * 16 + kk];
        }

        __syncthreads(); //Espera a que todas las hebras del bloque terminen de sumar antes de sobreescribir la memoria compartida
    }

    if (j < n && jp < n) {
        d_covarianza[j * n + jp] = 1.0f / m * suma;
    }

}


int main() {

    int crop_size = 128;
    const int m = 100;          // Imágenes
    const int N = crop_size;          // Lado
    const int n = N * N;        // Píxeles por imagen
    const size_t tamaño_total = m * n * sizeof(double);
    const size_t tamaño_imagen = n * sizeof(double);
    const size_t tamaño_cov = n * n * sizeof(double);

    // 1. Asignar Memoria Pinned en el Host
    double* h_imagenes = nullptr;
    double* h_promedio = nullptr;
    double* h_covarianza = nullptr;
    
    cudaMallocHost((void**)&h_imagenes, tamaño_total);
    cudaMallocHost((void**)&h_promedio, tamaño_imagen);
    cudaMallocHost((void**)&h_covarianza, tamaño_cov);

    
    // Cargar imagenes
    for(int k = 0; k < 100; k++){

        std::string filename = "DIV2K_valid_LR_bicubic/X4/0" + std::to_string(801+k) + "x4.png";
        CImg<unsigned char> imagen(filename.c_str()); 

        CImg<unsigned char> gray = imagen.get_RGBtoYCbCr().get_channel(0); //Transformar a escala de grises
        
        int x0 = (gray.width()  - crop_size) / 2; 
        int y0 = (gray.height() - crop_size) / 2;

        int x1 = x0 + crop_size - 1;
        int y1 = y0 + crop_size - 1;

        CImg<unsigned char> center = gray.get_crop(x0, y0, x1, y1); //Truncar la imagen al tamaño deseado

        for (int i = 0; i < n; ++i) {
            h_imagenes[k * n + i] = (double)center[i]; // Poblar el vector con las imagenes cargadas
        }
    
    }


    // Reservar memoria en el Device
    double *d_imagenes, *d_promedio, *d_covarianza; 
    cudaMalloc(&d_imagenes, tamaño_total);
    cudaMalloc(&d_promedio, tamaño_imagen);
    cudaMalloc(&d_covarianza, tamaño_cov);
 


    cudaMemcpy(d_imagenes, h_imagenes, tamaño_total, cudaMemcpyHostToDevice); //Cargar las imagenes al device


    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles
    calcularPromedioKernel<<<blocksPerGrid, threadsPerBlock>>>(d_imagenes, d_promedio, m, n); // Llamar al kernel, cada thread procesa un pixel de las imagenes

    //cudaDeviceSynchronize(); Creo que esto no es necesario porque los kernels están en el mismo stream   

    calcularImagenesCentradas<<<blocksPerGrid, threadsPerBlock>>>(d_imagenes, d_promedio, m, n);
  
    //cudaDeviceSynchronize(); Creo que esto no es necesario porque los kernels están en el mismo stream   

    dim3 threadsPerBlock2(16, 16);
    dim3 blocksPerGrid2(1024, 1024);

    calcularTileCovarianza<<<blocksPerGrid2, threadsPerBlock2>>>(d_imagenes, d_covarianza, m, n);

    errMemcpy = cudaMemcpy(h_covarianza, d_covarianza, tamaño_cov, cudaMemcpyDeviceToHost);
    if (errMemcpy != cudaSuccess) {
        std::cerr << "Error en cudaMemcpy: " << cudaGetErrorString(errMemcpy) << std::endl;
    }


    CImg<double> visual3(h_covarianza, n, n);
    visual3.display("Covarianza (Pinned Memory)");
    
    
    //Liberación de memoria

    cudaFree(d_imagenes);
    cudaFree(d_promedio);
    cudaFree(d_covarianza);


    cudaFreeHost(h_imagenes);
    cudaFreeHost(h_promedio);
    cudaFreeHost(h_covarianza);

    // int device; Esto era para ver la info de la GPU
    // cudaGetDevice(&device);

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);

    // std::cout << "GPU: " << prop.name << std::endl;
    // std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    // std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    // std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "Warp size: " << prop.warpSize << std::endl;

    // int max_threads_total = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    // std::cout << "Max resident threads total: " << max_threads_total << std::endl;



    return 0;
}
