# define cimg_use_png

#include "../src/CImg.h"
#include "../src/cuda_kernels/experimento2.cu"
#include <cuda_runtime.h>
#include <iostream>

using namespace cimg_library;


int main() {
    const int NSTREAMS = 8;

    int crop_size = 128;
    const int m = 100;          // Imágenes
    const int N = crop_size;    // Lado
    const int n = N * N;        // Píxeles por imagen
    const size_t tamaño_total = m * n * sizeof(double);
    const size_t tamaño_imagen = n * sizeof(double);
    const size_t tamaño_promedio_parcial = tamaño_imagen * NSTREAMS;
    // const size_t tamaño_cov = n * n * sizeof(double);
    
    // 1. Asignar Memoria Pinned en el Host
    double* h_imagenes = nullptr;
    double* h_promedio = nullptr;
    // double* h_covarianza = nullptr;
    double* h_promedio_parcial = nullptr;
    
    cudaMallocHost((void**)&h_imagenes, tamaño_total);
    cudaMallocHost((void**)&h_promedio, tamaño_imagen);
    // cudaMallocHost((void**)&h_covarianza, tamaño_cov);
    cudaMallocHost((void**)&h_promedio_parcial, tamaño_promedio_parcial);

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
    double *d_imagenes, *d_promedio,/* *d_covarianza,*/ *d_promedio_parcial;
    cudaMalloc(&d_imagenes, tamaño_total);
    cudaMalloc(&d_promedio, tamaño_imagen);
    // cudaMalloc(&d_covarianza, tamaño_cov);
    cudaMalloc(&d_promedio_parcial, tamaño_promedio_parcial);

    const int chunk = tamaño_total / NSTREAMS; // Tamano de cada lote
    
    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamCreate(&streams[s]);

    for (int s = 0; s < NSTREAMS; ++s) {
        int offset = s * chunk;
        int offset_promedio = s * tamaño_imagen;
        int threadsPerBlock = 256;
        int blocksPerGrid = (chunk + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles

        // Transferencia H−>D asincrona
        cudaMemcpyAsync(d_imagenes + offset, h_imagenes + offset, chunk, cudaMemcpyHostToDevice, streams[s]); //Cargar las imagenes al device
        calcularPromedioParcialKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(d_imagenes + offset,
                                                                                         d_promedio_parcial + offset_promedio,
                                                                                         m,
                                                                                         n,
                                                                                         chunk/sizeof(double));//,
                                                                                         // chunk%n); // Llamar al kernel, cada thread procesa un pixel de las imagenes
    }

    // Esperar a que todos los streams terminen
    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamSynchronize(streams[s]);
    for (int s = 0; s < NSTREAMS; ++s) {
        int offset = s * chunk;
        int threadsPerBlock = 256;
        int blocksPerGrid = (chunk + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles

        // Transferencia H−>D asincrona
        cudaMemcpyAsync(d_imagenes + offset, h_imagenes + offset, chunk, cudaMemcpyHostToDevice, streams[s]); //Cargar las imagenes al device
        calcularImagenesCentradasStreams<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(d_imagenes + offset,
                                                                                            d_promedio,
                                                                                            m,
                                                                                            n,
                                                                                            chunk/sizeof(double),
                                                                                            chunk%n); // Llamar al kernel, cada thread procesa un pixel de las imagenes
    }
    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamDestroy(streams[s]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles
    reducirPromedioParcialKernel<<<blocksPerGrid, threadsPerBlock>>>(d_promedio_parcial, d_promedio, NSTREAMS, n);

    // //Liberación de memoria
    cudaFree(d_imagenes);
    cudaFree(d_promedio);
    // cudaFree(d_covarianza);
    cudaFree(d_promedio_parcial);


    cudaFreeHost(h_imagenes);
    cudaFreeHost(h_promedio);
    // cudaFreeHost(h_covarianza);
    cudaFreeHost(h_promedio_parcial);

    return 0;
}
