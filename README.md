# cuda-nccl-overlap
Debugging for CUDA compute and NCCL P2P overlapping

## Requirements
1. Your system must has openmpi, cublas, and nccl
2. I've compiled this code on CUDA-12.1 and NCCL-2.17.1

## How to Run
```bash
cmake -B build && cmake --build build
# then use 2 cards
bash run.sh
```
