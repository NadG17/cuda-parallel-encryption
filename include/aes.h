#pragma once
#include <vector>
#include "common.h"
void aes_encrypt_gpu(const std::vector<uint8_t>&, std::vector<uint8_t>&, const KeyBundle&, const Config&);
__device__ void aes_rounds(uint8_t state[16], const uint8_t* roundKeys);

