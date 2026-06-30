# define cimg_use_png

#include "../src/CImg.h"
#include "../src/cuda_kernels/experimento2.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace cimg_library;


int main(int argc, char* argv[]) {
    int NSTREAMS = 8;

    if (argc == 3) {
        if (std::string(argv[1]) == "-NSTREAMS") {
            NSTREAMS = std::stoi(argv[2]);
        } else {
            std::cerr << "Error: -NSTREAMS debe ir acompañado por un entero.\n";
            return 1;
        }
    } else if (argc != 1) {
        std::cerr << "Error: Incompatible el numero de arguments.\n";
        return 1;
    }

    int crop_size = 128;
    const int m = 100;          // Imágenes
    const int N = crop_size;    // Lado
    const int n = N * N;        // Píxeles por imagen
    const size_t tamaño_total = m * n * sizeof(float);
    const size_t tamaño_imagen = n * sizeof(float);
    const size_t tamaño_promedio_parcial = tamaño_imagen * NSTREAMS;
    const size_t tamaño_cov = n * n * sizeof(float);
    
    // 1. Asignar Memoria Pinned en el Host
    float* h_imagenes = nullptr;
    float* h_promedio = nullptr;
    float* h_covarianza = nullptr;
    float* h_promedio_parcial = nullptr;
    
    cudaMallocHost((void**)&h_imagenes, tamaño_total);
    cudaMallocHost((void**)&h_promedio, tamaño_imagen);
    cudaMallocHost((void**)&h_covarianza, tamaño_cov);
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
            h_imagenes[k * n + i] = (float)center[i]; // Poblar el vector con las imagenes cargadas
        }
    }

    // Reservar memoria en el Device
    float *d_imagenes, *d_promedio,*d_covarianza,*d_promedio_parcial;
    cudaMalloc(&d_imagenes, tamaño_total);
    cudaMalloc(&d_promedio, tamaño_imagen);
    cudaMalloc(&d_covarianza, tamaño_cov);
    cudaMalloc(&d_promedio_parcial, tamaño_promedio_parcial);

    
    
    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamCreate(&streams[s]);

    int imagenes_por_stream = (m + NSTREAMS - 1) / NSTREAMS;

    for (int s = 0; s < NSTREAMS; ++s) {

        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles
        int imagen_inicio = s * imagenes_por_stream;
        int batch_m = std::min(imagenes_por_stream, m - imagen_inicio);

        size_t offset_elementos = imagen_inicio * n;
        size_t batch_bytes = batch_m * n * sizeof(float);

        // Transferencia H−>D asincrona
        cudaMemcpyAsync(d_imagenes + offset_elementos, h_imagenes + offset_elementos, batch_bytes, cudaMemcpyHostToDevice, streams[s]); //Cargar las imagenes al device
        calcularPromedioParcialKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(d_imagenes + offset_elementos,
                                                                                         d_promedio_parcial + s*n,
                                                                                         m,
                                                                                         n,
                                                                                         batch_m * n);//,
                                                                                         // chunk%n); // Llamar al kernel, cada thread procesa un pixel de las imagenes
    }

    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamSynchronize(streams[s]);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles
    reducirPromedioParcialKernel<<<blocksPerGrid, threadsPerBlock>>>(d_promedio_parcial, d_promedio, NSTREAMS, n);


    cudaDeviceSynchronize();

    

    // Esperar a que todos los streams terminen
    
    for (int s = 0; s < NSTREAMS; ++s) {

        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; //Se calculan cuantos bloques se necesitan en funcion de la cantidad de pixeles
        int imagen_inicio = s * imagenes_por_stream;
        int batch_m = std::min(imagenes_por_stream, m - imagen_inicio);

        size_t offset_elementos = imagen_inicio * n;


        // Transferencia H−>D asincrona
        //Cargar las imagenes al device
        calcularImagenesCentradasStreams<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(d_imagenes + offset_elementos,
                                                                                            d_promedio,
                                                                                            n,
                                                                                            batch_m * n); // Llamar al kernel, cada thread procesa un pixel de las imagenes
    }
    
    for (int s = 0; s < NSTREAMS; s++)
    cudaStreamSynchronize(streams[s]);

    

   
    cudaMemset(d_covarianza, 0, n * n * sizeof(float));
    dim3 threadsPerBlockCov(16, 16);
    dim3 blocksPerGridCov((n + 15) / 16, (n + 15) / 16);

    for (int s = 0; s < NSTREAMS; ++s) {
        int imagen_inicio = s * imagenes_por_stream;
        int batch_m = std::min(imagenes_por_stream, m - imagen_inicio);

        if (batch_m <= 0) continue;

        size_t offset_elementos = imagen_inicio * n;


    // Cada stream calcula la contribución de su batch a la covarianza
    acumularCovarianzaBatchKernel<<<blocksPerGridCov,
                                    threadsPerBlockCov,
                                    0,
                                    streams[s]>>>(
                                    d_imagenes + offset_elementos,
                                    d_covarianza,
                                    batch_m,
                                    n,
                                    m
                                );
    }

    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamSynchronize(streams[s]);


    // Transferencia D->H asincrona
    cudaMemcpy(h_covarianza, d_covarianza, n * n * sizeof(float), cudaMemcpyDeviceToHost);

     cudaError_t errMemcpy = cudaMemcpy(h_covarianza, d_covarianza, tamaño_cov, cudaMemcpyDeviceToHost);
    if (errMemcpy != cudaSuccess) {
        std::cerr << "Error en cudaMemcpy: " << cudaGetErrorString(errMemcpy) << std::endl;
    }


    for (int s = 0; s < NSTREAMS; s++)
        cudaStreamDestroy(streams[s]);


    // //Liberación de memoria
    cudaFree(d_imagenes);
    cudaFree(d_promedio);
    cudaFree(d_covarianza);
    cudaFree(d_promedio_parcial);


    cudaFreeHost(h_imagenes);
    cudaFreeHost(h_promedio);
    cudaFreeHost(h_covarianza);
    cudaFreeHost(h_promedio_parcial);

    return 0;
}
