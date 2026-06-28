# define cimg_use_png

#include "../src/CImg.h"
#include "../src/cuda_kernels/experimento1.cu"
#include <cuda_runtime.h>
#include <iostream>

using namespace cimg_library;


int main() {
    int crop_size = 128;
    const int m = 100;          // Imágenes
    const int N = crop_size;    // Lado
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
        std::string filename = "data/DIV2K_valid_LR_bicubic/X4/0" + std::to_string(801+k) + "x4.png";
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

    cudaError_t errMemcpy = cudaMemcpy(h_covarianza, d_covarianza, tamaño_cov, cudaMemcpyDeviceToHost);
    if (errMemcpy != cudaSuccess) {
        std::cerr << "Error en cudaMemcpy: " << cudaGetErrorString(errMemcpy) << std::endl;
    }


    //Liberación de memoria
    cudaFree(d_imagenes);
    cudaFree(d_promedio);
    cudaFree(d_covarianza);


    cudaFreeHost(h_imagenes);
    cudaFreeHost(h_promedio);
    cudaFreeHost(h_covarianza);

    return 0;
}
