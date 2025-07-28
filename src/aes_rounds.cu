#include "aes.h"
#include "common.h"
#include <cuda_runtime.h>

__device__ void SubBytes(uint8_t state[16]) {
    for (int i = 0; i < 16; i++)
        state[i] = d_sbox[state[i]];
}

__device__ void ShiftRows(uint8_t state[16]) {
    uint8_t tmp;

    // Row 1
    tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    // Row 2
    tmp = state[2];
    state[2] = state[10];
    state[10] = tmp;
    tmp = state[6];
    state[6] = state[14];
    state[14] = tmp;

    // Row 3
    tmp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = tmp;
}

__device__ void MixColumns(uint8_t state[16]) {
    uint8_t Tmp, Tm, t;
    for (int i = 0; i < 4; ++i) {
        t = state[i*4];
        Tmp = state[i*4] ^ state[i*4 + 1] ^ state[i*4 + 2] ^ state[i*4 + 3];
        Tm = state[i*4] ^ state[i*4 + 1]; Tm = xtime(Tm); state[i*4] ^= Tm ^ Tmp;
        Tm = state[i*4 + 1] ^ state[i*4 + 2]; Tm = xtime(Tm); state[i*4 + 1] ^= Tm ^ Tmp;
        Tm = state[i*4 + 2] ^ state[i*4 + 3]; Tm = xtime(Tm); state[i*4 + 2] ^= Tm ^ Tmp;
        Tm = state[i*4 + 3] ^ t; Tm = xtime(Tm); state[i*4 + 3] ^= Tm ^ Tmp;
    }
}

__device__ void AddRoundKey(uint8_t state[16], const uint8_t* roundKey) {
    for (int i = 0; i < 16; i++)
        state[i] ^= roundKey[i];
}

// xtime helper
__device__ uint8_t xtime(uint8_t x) {
    return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
}

// Main AES Rounds function
__device__ void aes_rounds(uint8_t state[16], const uint8_t* roundKeys) {
    AddRoundKey(state, roundKeys);  // Initial AddRoundKey

    for (int round = 1; round < 10; round++) {
        SubBytes(state);
        ShiftRows(state);
        MixColumns(state);
        AddRoundKey(state, roundKeys + round * 16);
    }

    SubBytes(state);
    ShiftRows(state);
    AddRoundKey(state, roundKeys + 10 * 16); // Final round
}
