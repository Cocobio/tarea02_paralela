# Tarea 02 - Procesamiento de imagenes en CUDA

Este repositorio contiene implementaciones CUDA para procesar un conjunto de
imagenes DIV2K y calcular informacion estadistica sobre ellas. El flujo actual
carga 100 imagenes, las convierte a escala de grises, toma un recorte central
de `128x128`, calcula el vector promedio, centra las imagenes y construye la
matriz de covarianza.

Hay dos ejecutables principales:

- `cuda_naive`: implementacion tradicional que copia todo el dataset a la GPU y
  usa el stream por defecto.
- `cuda_streams`: implementacion por batches que usa multiples CUDA Streams. La
  cantidad de streams se configura en tiempo de ejecucion.

## Estructura del repositorio

- `run.py`: script principal para compilar y ejecutar las pruebas CUDA.
- `test_suite/`: programas ejecutables de prueba (`cuda_naive.cu` y
  `cuda_streams.cu`).
- `src/cuda_kernels/`: kernels CUDA usados por cada implementacion.
- `src/CImg.h`: biblioteca usada para cargar y preprocesar imagenes.
- `tools/get_gpu_info.cu`: utilidad opcional para consultar informacion basica
  de la GPU disponible.

Los directorios `data/` y `bin/` se generan localmente

## Requisitos

- Python 3.
- CUDA Toolkit con `nvcc`.
- Una GPU NVIDIA compatible con CUDA.
- Bibliotecas de sistema usadas durante la compilacion:
  - `libX11`
  - `libpng`
  - `zlib`
- Herramientas para preparar el dataset:
  - `wget`
  - `unzip`

## Preparar el dataset

Los programas esperan encontrar las 100 imagenes de validacion de DIV2K en:

```text
data/DIV2K_valid_LR_bicubic/X4/
```

Para descargar y descomprimir el dataset:

```bash
mkdir -p data
cd data
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
unzip DIV2K_valid_LR_bicubic_X4.zip
rm -f DIV2K_valid_LR_bicubic_X4.zip
cd -
```

## Compilar

Desde la raiz del repositorio, ejecuta:

```bash
python3 run.py compile
```

Esto crea el directorio `bin/` si no existe y compila los programas de
`test_suite/`:

- `bin/cuda_naive.o`
- `bin/cuda_streams.o`

## Ejecutar

Ejecutar la version tradicional:

```bash
python3 run.py cuda_naive
```

Ejecutar la version con CUDA Streams:

```bash
python3 run.py cuda_streams --NSTREAMS 8
```

se puede cambiar `8` por otra cantidad de streams, por ejemplo:

```bash
python3 run.py cuda_streams --NSTREAMS 1
python3 run.py cuda_streams --NSTREAMS 2
python3 run.py cuda_streams --NSTREAMS 4
python3 run.py cuda_streams --NSTREAMS 16
```

El script ejecuta cada prueba varias veces y muestra el tiempo total medido para
cada ejecucion.

## Utilidad opcional: informacion de GPU

Para compilar la utilidad:

```bash
mkdir -p bin
nvcc tools/get_gpu_info.cu -o bin/get_gpu_info.o \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64
```

Para ejecutarla:

```bash
./bin/get_gpu_info.o
```

La utilidad imprime datos como nombre de la GPU, cantidad de SMs, numero maximo
de threads por bloque, numero maximo de threads por SM y tamano de warp.

