#include "aes.h"
#include "common.h"
#include <cuda_runtime.h>

__device__ const uint8_t d_sbox[256] = {/* AES S-box table */};

// AES key expansion omitted for brevity (include full implementation)

__global__ void aes_encrypt_block_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    const uint8_t* __restrict__ roundKeys,
    int blockCount, AESMode mode, const uint8_t* iv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= blockCount) return;

    uint8_t state[16];
    memcpy(state, in + idx*16, 16);

    if (mode == CBC && idx == 0) {
        for (int i = 0; i < 16; i++) state[i] ^= iv[i];
    } else if (mode == CBC) {
        // previous cipher text: assume in contains chaining
        for (int i = 0; i < 16; i++)
          state[i] ^= in[(idx-1)*16 + i];
    }

    aes_rounds(state, roundKeys);

    memcpy(out + idx*16, state, 16);
}

void aes_encrypt_gpu(
    const std::vector<uint8_t>& plaintext,
    std::vector<uint8_t>& ciphertext,
    const KeyBundle& keyBundle, const Config& cfg
) {
    size_t nbytes = plaintext.size();
    int blockCount = (nbytes + 15) / 16;
    size_t bufSize = blockCount * 16;

    uint8_t *d_in, *d_out, *d_roundKeys, *d_iv;
    cudaMalloc(&d_in, bufSize);
    cudaMalloc(&d_out, bufSize);
    cudaMalloc(&d_roundKeys, keyBundle.roundKeyBytes());
    if (cfg.mode == CBC) cudaMalloc(&d_iv, 16);

    cudaMemcpy(d_in, plaintext.data(), bufSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roundKeys, keyBundle.roundKeys.data(), keyBundle.roundKeyBytes(), cudaMemcpyHostToDevice);
    if (cfg.mode == CBC) cudaMemcpy(d_iv, cfg.iv.data(), 16, cudaMemcpyHostToDevice);

    int threads = cfg.gpu_threads;
    int blocks = (blockCount + threads - 1) / threads;
    aes_encrypt_block_kernel<<<blocks, threads>>>(d_in, d_out, d_roundKeys, blockCount, cfg.mode, d_iv);
    cudaDeviceSynchronize();

    cudaMemcpy(ciphertext.data(), d_out, bufSize, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_roundKeys);
    if (cfg.mode == CBC) cudaFree(d_iv);
}
