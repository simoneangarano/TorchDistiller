#!/bin/bash

# Configuration
CUDA_GENCODE="\
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_86,code=compute_86"

cd src
nvcc -I/usr/local/cuda/include --expt-extended-lambda -O3 -c -o bn.o bn.cu -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
