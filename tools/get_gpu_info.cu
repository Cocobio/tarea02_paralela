#include <cuda_runtime.h>
#include <iostream>


int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;

    int max_threads_total = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    std::cout << "Max resident threads total: " << max_threads_total << std::endl;

    return 0;
}
