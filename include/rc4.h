#pragma once
#include <vector>
#include "common.h"
void rc4_encrypt_gpu(const std::vector<uint8_t>&, std::vector<uint8_t>&, const KeyBundle&);
