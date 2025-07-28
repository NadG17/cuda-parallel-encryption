#include "rc4.h"
#include "common.h"
#include <cuda_runtime.h>

__global__ void rc4_encrypt_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    const uint8_t* __restrict__ key,
    int keylen, size_t nbytes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbytes) return;

    // Simple parallel RC4: each thread generates keystream for its idx
    uint8_t i=idx % keylen, j=0, S[256];
    for (int u=0; u<256; u++) S[u] = u;
    for (int u=0; u<256; u++) {
        j = j + S[u] + key[u % keylen];
        uint8_t t = S[u]; S[u] = S[j]; S[j] = t;
    }
    // PRGA step to generate one byte
    i = (i + 1);
    j = (j + S[i]);
    uint8_t t = S[i]; S[i] = S[j]; S[j] = t;
    uint8_t keystream = S[(S[i] + S[j]) & 0xFF];
    out[idx] = in[idx] ^ keystream;
}

void rc4_encrypt_gpu(
    const std::vector<uint8_t>& plaintext,
    std::vector<uint8_t>& ciphertext,
    const KeyBundle& keyBundle
) {
    size_t nbytes = plaintext.size();
    uint8_t *d_in, *d_out, *d_key;
    cudaMalloc(&d_in, nbytes);
    cudaMalloc(&d_out, nbytes);
    cudaMalloc(&d_key, keyBundle.key.size());
    cudaMemcpy(d_in, plaintext.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, keyBundle.key.data(), keyBundle.key.size(), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (nbytes + threads - 1) / threads;
    rc4_encrypt_kernel<<<blocks, threads>>>(d_in, d_out, d_key, keyBundle.key.size(), nbytes);
    cudaDeviceSynchronize();

    cudaMemcpy(ciphertext.data(), d_out, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_key);
}
