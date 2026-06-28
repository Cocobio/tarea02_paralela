# Lista de tareas e ideas para la ejecucion de la tarea

Tasks / ToDo
---
- [ ] Descargar imagenes
- [ ] Testear **Nacho**
    - [ ] CUDA sin streams
    - [ ] uso de CUDA streams

Work in progress
---
- Implementar
    - [ ] Solucion tradicional
        - [ ] Calculo de matriz de covarianzas **Caro**
- Miscelaneos
    - [ ] Agregar el README.md explicando como se compila y ejecuta **Gian**
- [ ] Implementar
    - [ ] Solucion con streams **Nacho, Gian y Caro**
        - [ ] Calculo de matriz de covarianzas
- Informe
- [ ] Hacer informe **Gian :)**
    - [ ] Analizar limitaciones de PCIe
    - [ ] Analizar limitaciones de vRAM
    - [ ] Tablas:
        - [ ] Tiempo de ejecucion totales
        - [ ] Grafico de SpeedUp
        - [ ] Profiling mostrando tiempos Host - GPU

Done / Completed
---
- [x] Primera reunion del *Dream Team*
- Implementar
    -  Solucion tradicional
        - [x] Calculo de vector promedio **Caro**
        - [x] Calculo de vector centrado **Caro**
    - [ ] Solucion con streams **Nacho, Gian y Caro**
        * Usar carga por batches y CUDA streams
        - Permitir que la cantidad de streams $S$ sea configurable "en T de ejecucion"
        - [x] Calculo de vector promedio
        - [x] Calculo de vector centrado
- Miscelaneos
    - [x] Crear un script de compilacion/testing **Nacho**
    - [x] Agregar al README.md como descargar el set de datos y dejar bajo folder marcado en gitignore **Nacho**
- Informe
    - [-] Pedirle a Mabel un overleaf, por favor cito <3
