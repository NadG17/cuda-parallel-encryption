#!/bin/bash
# Colab install and build
apt-get update -qq
apt-get install -y build-essential git cuda-toolkit-11-8
cd cuda-parallel-encryption
nvcc src/*.cu -I include -o cuda_encrypt -O3
echo "Built CUDA encryption binary"
